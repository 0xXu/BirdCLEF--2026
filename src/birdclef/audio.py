import hashlib
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from birdclef.config import CFG
from birdclef.deps import T, cv2, h5py, librosa, require_dependencies, sf


class LogMelExtractor:
    """Reusable waveform-to-LogMel converter.

    Constructing torchaudio transforms per sample is expensive. Keeping one
    extractor per Dataset/worker matches the high-scoring waveform-first
    pattern while preserving the existing LogMel front-end parameters.
    """

    def __init__(self, cfg: CFG):
        require_dependencies(("torchaudio", T))
        self.cfg = cfg
        self.mel_transform = T.MelSpectrogram(
            sample_rate=cfg.fs,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            power=cfg.mel_power,
            norm=cfg.mel_norm,
            mel_scale=cfg.mel_scale,
        )
        self.db_transform = T.AmplitudeToDB(stype="power", top_db=cfg.mel_top_db)

    def __call__(self, audio_data: np.ndarray) -> np.ndarray:
        if np.isnan(audio_data).any():
            fill_value = float(np.nanmean(audio_data)) if not np.isnan(audio_data).all() else 0.0
            audio_data = np.nan_to_num(audio_data, nan=fill_value)

        waveform = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0)
        logmel = self.db_transform(self.mel_transform(waveform)).squeeze(0).numpy().astype(np.float32)
        mn, mx = float(logmel.min()), float(logmel.max())
        return ((logmel - mn) / (mx - mn + 1e-7)).astype(np.float32)


def audio2logmel(audio_data: np.ndarray, cfg: CFG) -> np.ndarray:
    return LogMelExtractor(cfg)(audio_data)


def logmel_to_shape(logmel: np.ndarray, cfg: CFG) -> np.ndarray:
    require_dependencies(("cv2", cv2))
    height, width = cfg.target_shape
    if logmel.shape != cfg.target_shape:
        logmel = cv2.resize(logmel, (width, height), interpolation=cv2.INTER_LINEAR)
    return logmel.astype(np.float32)


def waveform_cache_path(source_path: str | Path, cfg: CFG) -> Path:
    source = str(Path(source_path))
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    stem = Path(source).stem.replace("/", "_")
    return cfg.waveform_cache_dir / f"{stem}_{digest}.hdf5"


def load_audio_mono(path: str | Path, cfg: CFG) -> np.ndarray:
    require_dependencies(("librosa", librosa), ("soundfile", sf))
    try:
        audio, orig_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if orig_sr != cfg.fs:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=cfg.fs)
    except Exception:
        audio, _ = librosa.load(str(path), sr=cfg.fs)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    if len(audio) == 0:
        audio = np.zeros(cfg.target_samples, dtype=np.float32)
    return audio


def write_waveform_hdf5(source_path: str | Path, cfg: CFG) -> tuple[str, str, int, float, bool]:
    require_dependencies(("h5py", h5py))
    source_path = Path(source_path)
    target_path = waveform_cache_path(source_path, cfg)
    if target_path.exists():
        try:
            with h5py.File(target_path, "r", swmr=True) as f:
                n_samples = int(f["au"].shape[0])
            return str(source_path), str(target_path), n_samples, n_samples / cfg.fs, False
        except OSError:
            target_path.unlink(missing_ok=True)

    audio = load_audio_mono(source_path, cfg)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp.hdf5")
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("au", data=audio.astype(np.float32), compression="lzf")
        f.attrs["sr"] = cfg.fs
        f.attrs["source_path"] = str(source_path)
        f.attrs["duration_s"] = len(audio) / cfg.fs
    tmp_path.replace(target_path)
    return str(source_path), str(target_path), len(audio), len(audio) / cfg.fs, True


def precompute_waveform_cache(df: pd.DataFrame, cfg: CFG, n_workers: int = 4) -> pd.DataFrame:
    require_dependencies(("h5py", h5py))
    unique_paths = sorted({str(Path(path)) for path in df["filepath"].dropna().astype(str)})
    print(f"Computing waveform HDF5 cache: {len(unique_paths)} unique audio files with {n_workers} worker(s)")
    start_time = time.time()
    rows = []
    failed = 0

    def _process(path: str):
        return write_waveform_hdf5(path, cfg)

    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as executor:
        futures = {executor.submit(_process, path): path for path in unique_paths}
        for done, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                source, cache, n_samples, duration_s, written = future.result()
                rows.append(
                    {
                        "filepath": source,
                        "waveform_cache_path": cache,
                        "n_samples": n_samples,
                        "duration_s_cache": duration_s,
                        "written": written,
                    }
                )
            except Exception as exc:
                failed += 1
                print(f"Failed to cache {path}: {exc}")

            if done % 3000 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / done * (len(unique_paths) - done)
                print(f"  {done}/{len(unique_paths)} done | {elapsed/60:.1f} min | ETA {eta/60:.1f} min")

    manifest = pd.DataFrame(rows).sort_values("filepath").reset_index(drop=True)
    cfg.waveform_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cfg.waveform_manifest_path, index=False)
    print(
        f"Waveform cache ready in {(time.time() - start_time)/60:.1f} min | "
        f"failed={failed} | manifest={cfg.waveform_manifest_path}"
    )
    if failed:
        raise RuntimeError(f"Failed to cache {failed} audio file(s)")
    return manifest


def load_or_build_cache(df: pd.DataFrame, cfg: CFG, n_workers: int = 4) -> pd.DataFrame:
    if cfg.waveform_manifest_path.exists():
        manifest = pd.read_csv(cfg.waveform_manifest_path)
        cached = set(manifest["filepath"].astype(str))
        required = {str(Path(path)) for path in df["filepath"].dropna().astype(str)}
        missing = sorted(required - cached)
        path_to_cache = dict(zip(manifest["filepath"].astype(str), manifest["waveform_cache_path"].astype(str)))
        missing_existing = [path for path in required & cached if not Path(path_to_cache[path]).exists()]
        if not missing and not missing_existing:
            print(f"Loaded waveform cache manifest {cfg.waveform_manifest_path} ({len(manifest)} files)")
            return manifest
        print(
            f"Waveform cache incomplete: missing_manifest={len(missing)} "
            f"missing_files={len(missing_existing)}. Rebuilding manifest."
        )

    return precompute_waveform_cache(df, cfg, n_workers=n_workers)


class WaveformStore:
    def __init__(self, cfg: CFG):
        require_dependencies(("h5py", h5py))
        self.cfg = cfg
        self.max_items = max(0, int(cfg.waveform_lru_size))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, source_path: str | Path) -> np.ndarray:
        source = str(Path(source_path))
        cached = self._cache.get(source)
        if cached is not None:
            self._cache.move_to_end(source)
            return cached.copy()

        h5_path = waveform_cache_path(source, self.cfg)
        if h5_path.exists():
            with h5py.File(h5_path, "r", swmr=True) as f:
                audio = np.asarray(f["au"], dtype=np.float32)
        else:
            audio = load_audio_mono(source, self.cfg)

        if self.max_items > 0:
            self._cache[source] = audio
            self._cache.move_to_end(source)
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)
        return audio.copy()
