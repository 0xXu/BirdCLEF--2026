from pathlib import Path

import numpy as np
import pandas as pd
import torch

from birdclef.audio import audio2logmel, load_audio_mono, logmel_to_shape
from birdclef.config import CFG
from birdclef.infer import discover_checkpoints
from birdclef.model import BirdCLEFModel
from birdclef.postprocess import load_calibration_table, postprocess_soundscape_predictions
from birdclef.utils import load_species_ids


def _parse_site_metadata(filename: str) -> dict[str, object]:
    stem = Path(filename).stem
    tokens = stem.split("_")
    site = next((token for token in tokens if token.startswith("S") and token[1:].isdigit()), None)
    date = next((token for token in tokens if len(token) == 8 and token.isdigit()), None)
    time_token = next((token for token in tokens if len(token) == 6 and token.isdigit()), None)
    hour = int(time_token[:2]) if time_token else None
    return {"soundscape_site": site, "soundscape_date": date, "soundscape_hour": hour}


def _load_teacher_models(cfg: CFG) -> tuple[list[BirdCLEFModel], np.ndarray]:
    ckpt_files = discover_checkpoints(cfg)
    if not ckpt_files:
        raise FileNotFoundError(f"No model_fold*.pth checkpoints found in {cfg.output_dir}")

    models = []
    weights = []
    for ckpt_path in ckpt_files:
        model = BirdCLEFModel(cfg, load_backbone_weights=False).to(cfg.device)
        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        score = float(ckpt.get("val_auc", 0.0))
        weights.append(max(score, 1e-6))
        print(f"Loaded teacher {ckpt_path.name} val_auc={score:.4f}")

    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = weights_arr / weights_arr.sum()
    return models, weights_arr


def _segment_offsets(audio_len: int, frame_start_sec: float, cfg: CFG, n_crops: int) -> list[int]:
    half = cfg.target_samples // 2
    target_center = int((frame_start_sec + 2.5) * cfg.fs)
    base_offset = target_center - half
    max_offset = max(0, audio_len - cfg.target_samples)
    if n_crops <= 1:
        return [max(0, min(base_offset, max_offset))]
    shift = cfg.target_samples // 4
    offsets = [
        max(0, min(base_offset - shift, max_offset)),
        max(0, min(base_offset, max_offset)),
        max(0, min(base_offset + shift, max_offset)),
    ]
    if n_crops > 3:
        extra = np.linspace(base_offset - shift, base_offset + shift, n_crops)
        offsets = [max(0, min(int(x), max_offset)) for x in extra]
    return offsets[:n_crops]


def _segments_to_tensor(segments: list[np.ndarray], cfg: CFG) -> torch.Tensor:
    tensors = []
    for seg in segments:
        if len(seg) < cfg.target_samples:
            seg = np.pad(seg, (0, cfg.target_samples - len(seg)))
        else:
            seg = seg[: cfg.target_samples]
        spec = logmel_to_shape(audio2logmel(seg, cfg), cfg)
        tensors.append(torch.from_numpy(spec.astype(np.float32)).unsqueeze(0))
    return torch.stack(tensors, dim=0)


def _predict_audio_frames(
    audio: np.ndarray,
    models: list[BirdCLEFModel],
    model_weights: np.ndarray,
    cfg: CFG,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    n_frames = int(np.ceil(len(audio) / (5 * cfg.fs)))
    frames = [(float(i * 5), float((i + 1) * 5)) for i in range(n_frames)]
    crop_specs: list[tuple[int, int]] = []
    crop_segments = []

    for frame_idx, (start_sec, _) in enumerate(frames):
        for offset in _segment_offsets(len(audio), start_sec, cfg, cfg.pseudo_teacher_tta_crops):
            crop_specs.append((frame_idx, offset))
            crop_segments.append(audio[offset : offset + cfg.target_samples])

    all_crop_preds = np.zeros((len(crop_segments), len(load_species_ids(cfg))), dtype=np.float32)
    with torch.no_grad():
        for batch_start in range(0, len(crop_segments), cfg.pseudo_batch_size):
            batch_segments = crop_segments[batch_start : batch_start + cfg.pseudo_batch_size]
            x = _segments_to_tensor(batch_segments, cfg).to(cfg.device)
            batch_preds = []
            for model, weight in zip(models, model_weights):
                probs = torch.sigmoid(model(x)).cpu().numpy()
                batch_preds.append(probs * float(weight))
            all_crop_preds[batch_start : batch_start + len(batch_segments)] = np.sum(batch_preds, axis=0)

    frame_preds = np.zeros((len(frames), all_crop_preds.shape[1]), dtype=np.float32)
    frame_counts = np.zeros(len(frames), dtype=np.float32)
    for crop_idx, (frame_idx, _) in enumerate(crop_specs):
        frame_preds[frame_idx] += all_crop_preds[crop_idx]
        frame_counts[frame_idx] += 1
    frame_preds = frame_preds / np.maximum(frame_counts[:, None], 1.0)
    return frame_preds, frames


def _build_pseudo_rows(
    ogg_path: Path,
    preds: np.ndarray,
    frames: list[tuple[float, float]],
    species_ids: list[str],
    cfg: CFG,
) -> list[dict[str, object]]:
    rows = []
    pred_cols = [f"pseudo_{label}" for label in species_ids]
    meta = _parse_site_metadata(ogg_path.name)
    for frame_idx, ((start_sec, end_sec), probs) in enumerate(zip(frames, preds)):
        top_idx = int(np.argmax(probs))
        primary_prob = float(probs[top_idx])
        if primary_prob < cfg.pseudo_min_primary_prob:
            continue

        selected = np.where(probs >= cfg.pseudo_label_prob)[0].tolist()
        if top_idx not in selected:
            selected.insert(0, top_idx)
        selected = sorted(selected, key=lambda idx: float(probs[idx]), reverse=True)[: cfg.pseudo_max_labels]
        labels = [species_ids[idx] for idx in selected]
        row_id = f"{ogg_path.stem}_{int(end_sec)}"
        row = {
            "filename": ogg_path.name,
            "audio_id": ogg_path.stem,
            "filepath": str(ogg_path),
            "row_id": row_id,
            "source": "pseudo_soundscape",
            "primary_label": labels[0],
            "secondary_labels": str(labels[1:]),
            "all_labels": str(labels),
            "offset": max(0.0, (start_sec + end_sec) / 2.0 - cfg.target_duration / 2.0),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "rating": 5.0,
            "pseudo_primary_prob": primary_prob,
            "pseudo_n_labels": len(labels),
            "pseudo_weight": cfg.pseudo_sampling_weight,
            **meta,
        }
        row.update({col: float(prob) for col, prob in zip(pred_cols, probs)})
        rows.append(row)
    return rows


def generate_pseudo_labels(cfg: CFG) -> pd.DataFrame:
    species_ids = load_species_ids(cfg)
    models, weights = _load_teacher_models(cfg)
    calibration = load_calibration_table(cfg)
    if calibration is not None:
        print(f"Loaded calibration table {cfg.resolved_calibration_path}")
    sc_dir = cfg.train_datadir.parent / "train_soundscapes"
    soundscapes = sorted(sc_dir.glob("*.ogg"))
    if cfg.pseudo_max_files is not None:
        soundscapes = soundscapes[: cfg.pseudo_max_files]
    if not soundscapes:
        raise FileNotFoundError(f"No train soundscapes found in {sc_dir}")

    rows = []
    print(
        f"Generating pseudo labels for {len(soundscapes)} soundscape(s) | "
        f"tta_crops={cfg.pseudo_teacher_tta_crops} batch={cfg.pseudo_batch_size}"
    )
    for idx, ogg_path in enumerate(soundscapes, start=1):
        audio = load_audio_mono(ogg_path, cfg)
        preds, frames = _predict_audio_frames(audio, models, weights, cfg)
        preds = postprocess_soundscape_predictions(preds, species_ids, cfg, calibration=calibration)
        file_rows = _build_pseudo_rows(ogg_path, preds, frames, species_ids, cfg)
        rows.extend(file_rows)
        if idx % 25 == 0 or idx == len(soundscapes):
            print(f"  {idx}/{len(soundscapes)} soundscapes | pseudo_rows={len(rows)}")

    pseudo_df = pd.DataFrame(rows)
    out_path = cfg.resolved_pseudo_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pseudo_df.to_csv(out_path, index=False)
    print(f"Saved pseudo labels to {out_path} | rows={len(pseudo_df)}")
    if rows:
        print(
            "Pseudo confidence:",
            f"mean={pseudo_df['pseudo_primary_prob'].mean():.4f}",
            f"min={pseudo_df['pseudo_primary_prob'].min():.4f}",
            f"max={pseudo_df['pseudo_primary_prob'].max():.4f}",
        )
    return pseudo_df


def load_pseudo_training_frame(cfg: CFG) -> pd.DataFrame | None:
    path = cfg.resolved_pseudo_path
    if not path.exists():
        print(f"Pseudo labels not found at {path}; no pseudo rows loaded.")
        return None

    pseudo_df = pd.read_csv(path)
    if pseudo_df.empty:
        print(f"Pseudo label file is empty: {path}")
        return None
    pseudo_df = pseudo_df[pseudo_df["pseudo_primary_prob"] >= cfg.pseudo_min_primary_prob].copy()
    if pseudo_df.empty:
        print("No pseudo rows left after confidence filtering.")
        return None

    pseudo_df["source"] = "pseudo_soundscape"
    pseudo_df["rating"] = pseudo_df.get("rating", 5.0)
    pseudo_df["pseudo_weight"] = cfg.pseudo_sampling_weight
    print(
        f"Loaded pseudo rows from {path}: rows={len(pseudo_df)} "
        f"mean_conf={pseudo_df['pseudo_primary_prob'].mean():.4f}"
    )
    return pseudo_df.reset_index(drop=True)


def fold_safe_pseudo_subset(
    pseudo_df: pd.DataFrame | None,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame | None:
    if pseudo_df is None or pseudo_df.empty:
        return None

    # Pseudo rows are transductive train-soundscape samples; the hard rule is
    # to never train on the same soundscape file that is currently validating.
    _ = train_df
    val_audio_ids = set(val_df["audio_id"].astype(str))
    subset = pseudo_df[~pseudo_df["audio_id"].astype(str).isin(val_audio_ids)].copy()
    if subset.empty:
        return None
    return subset.reset_index(drop=True)
