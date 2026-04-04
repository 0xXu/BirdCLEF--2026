import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from birdclef.config import CFG
from birdclef.deps import T, cv2, librosa, require_dependencies, sf


def audio2pcen(audio_data: np.ndarray, cfg: CFG) -> np.ndarray:
    require_dependencies(("librosa", librosa), ("torchaudio", T))
    if np.isnan(audio_data).any():
        fill_value = float(np.nanmean(audio_data)) if not np.isnan(audio_data).all() else 0.0
        audio_data = np.nan_to_num(audio_data, nan=fill_value)

    waveform = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0)
    mel_transform = T.MelSpectrogram(
        sample_rate=cfg.fs,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
        power=1.0,
    )
    spec = mel_transform(waveform).squeeze(0).numpy()
    pcen = librosa.pcen(
        spec * (2**31),
        sr=cfg.fs,
        hop_length=cfg.hop_length,
        gain=cfg.pcen_gain,
        bias=cfg.pcen_bias,
        power=cfg.pcen_power,
        time_constant=cfg.pcen_time_constant,
        eps=cfg.pcen_eps,
    ).astype(np.float32)
    mn, mx = pcen.min(), pcen.max()
    return ((pcen - mn) / (mx - mn + 1e-8)).astype(np.float32)


def process_audio_file(
    path: str | Path,
    cfg: CFG,
    offset: float | None = None,
) -> np.ndarray | None:
    require_dependencies(("cv2", cv2), ("librosa", librosa), ("soundfile", sf), ("torchaudio", T))
    try:
        try:
            audio, orig_sr = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if orig_sr != cfg.fs:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=cfg.fs)
        except Exception:
            audio, _ = librosa.load(str(path), sr=cfg.fs)

        if len(audio) < cfg.target_samples:
            audio = np.tile(audio, math.ceil(cfg.target_samples / max(1, len(audio))))

        start = int(float(offset) * cfg.fs) if offset is not None else max(
            0, int(len(audio) / 2 - cfg.target_samples / 2)
        )
        seg = audio[start : start + cfg.target_samples]
        if len(seg) < cfg.target_samples:
            seg = np.pad(seg, (0, cfg.target_samples - len(seg)))

        pcen = audio2pcen(seg, cfg)
        if pcen.shape != cfg.target_shape:
            pcen = cv2.resize(pcen, cfg.target_shape, interpolation=cv2.INTER_LINEAR)
        return pcen.astype(np.float32)
    except Exception:
        return None


def precompute_all_pcen(df: pd.DataFrame, cfg: CFG, n_workers: int = 4) -> np.ndarray:
    rows = len(df)
    height, width = cfg.target_shape
    cache = np.zeros((rows, height, width), dtype=np.float16)

    def _process(i: int) -> tuple[int, np.ndarray | None]:
        row = df.iloc[i]
        offset = row.get("offset", None)
        offset = None if pd.isna(offset) else float(offset)
        spec = process_audio_file(row["filepath"], cfg, offset=offset)
        return i, spec.astype(np.float16) if spec is not None else None

    print(f"Computing PCEN cache: {rows} files with {n_workers} worker(s)")
    start_time = time.time()
    failed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process, i): i for i in range(rows)}
        for done, future in enumerate(as_completed(futures), start=1):
            idx, spec = future.result()
            if spec is not None:
                cache[idx] = spec
            else:
                failed += 1
            if done % 3000 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / done * (rows - done)
                print(f"  {done}/{rows} done | {elapsed/60:.1f} min | ETA {eta/60:.1f} min")

    print(
        f"Cache ready in {(time.time() - start_time)/60:.1f} min | "
        f"failed={failed} | ram={cache.nbytes/1e9:.2f} GB"
    )
    return cache


def load_or_build_cache(df: pd.DataFrame, cfg: CFG, n_workers: int = 4) -> np.ndarray:
    cache_path = cfg.cache_path
    if cache_path.exists():
        print(f"Loading cache from {cache_path}")
        cache = np.load(str(cache_path))
        print(f"Loaded cache {cache.shape} ({cache.nbytes/1e9:.2f} GB)")
        return cache

    cache = precompute_all_pcen(df, cfg, n_workers=n_workers)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), cache)
    print(f"Saved cache to {cache_path}")
    return cache
