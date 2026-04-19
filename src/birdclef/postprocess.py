import json

import numpy as np
import pandas as pd

from birdclef.config import CFG
from birdclef.deps import roc_auc_score


EPS = 1e-5


def _logit(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, EPS, 1.0 - EPS)
    return np.log(probs / (1.0 - probs))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _macro_auc(targets: np.ndarray, probs: np.ndarray) -> float:
    if roc_auc_score is None:
        return float("nan")
    scores = []
    for class_idx in range(targets.shape[1]):
        y_true = targets[:, class_idx]
        positives = int(y_true.sum())
        negatives = int(len(y_true) - positives)
        if positives == 0 or negatives == 0:
            continue
        scores.append(float(roc_auc_score(y_true, probs[:, class_idx])))
    return float(np.mean(scores)) if scores else float("nan")


def load_postprocess_params(cfg: CFG) -> dict[str, object]:
    params: dict[str, object] = {
        "file_level_top_k": cfg.postprocess_file_level_top_k,
        "rank_power": cfg.postprocess_rank_power,
        "delta_alpha": cfg.postprocess_delta_alpha,
        "adaptive_delta": cfg.postprocess_adaptive_delta,
        "threshold_sharpening": cfg.postprocess_threshold_sharpening,
        "taxon_temperature": cfg.postprocess_taxon_temperature,
        "taxon_temperatures": cfg.postprocess_taxon_temperatures,
    }
    path = cfg.resolved_postprocess_params_path
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        params.update(saved)
    return params


def load_per_class_thresholds(cfg: CFG, species_ids: list[str]) -> np.ndarray | None:
    path = cfg.resolved_per_class_thresholds_path
    if not path.exists():
        return None
    table = pd.read_csv(path)
    if "primary_label" not in table.columns or "threshold" not in table.columns:
        raise KeyError(f"Threshold table {path} must contain primary_label and threshold columns")
    table["primary_label"] = table["primary_label"].astype(str)
    by_label = table.set_index("primary_label")["threshold"]
    return np.array([float(by_label.get(label, 0.5)) for label in species_ids], dtype=np.float32)


def load_taxon_temperature_vector(
    cfg: CFG,
    species_ids: list[str],
    params: dict[str, object] | None = None,
) -> np.ndarray | None:
    params = params or load_postprocess_params(cfg)
    if not bool(params.get("taxon_temperature", True)):
        return None

    taxon_temperatures = params.get("taxon_temperatures", cfg.postprocess_taxon_temperatures)
    if not isinstance(taxon_temperatures, dict) or not taxon_temperatures:
        return None
    if not cfg.taxonomy_csv.exists():
        return None

    taxonomy = pd.read_csv(cfg.taxonomy_csv)
    if "primary_label" not in taxonomy.columns or "class_name" not in taxonomy.columns:
        return None

    taxonomy = taxonomy[["primary_label", "class_name"]].copy()
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    class_by_label = taxonomy.set_index("primary_label")["class_name"]
    temps = np.ones(len(species_ids), dtype=np.float32)
    for idx, label in enumerate(species_ids):
        class_name = class_by_label.get(str(label))
        if pd.isna(class_name):
            continue
        temps[idx] = float(taxon_temperatures.get(str(class_name), 1.0))
    return temps


def apply_taxon_temperature(
    preds: np.ndarray,
    taxon_temperature_vector: np.ndarray | None,
) -> np.ndarray:
    if taxon_temperature_vector is None:
        return preds.astype(np.float32, copy=True)
    temps = np.maximum(taxon_temperature_vector.astype(np.float32), EPS)
    out = _sigmoid(_logit(preds.astype(np.float32)) / temps[None, :])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def file_level_confidence_scale(preds: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or preds.shape[0] < 2:
        return preds.astype(np.float32, copy=True)
    preds = preds.astype(np.float32, copy=True)
    top_k = min(int(top_k), preds.shape[0])
    top_k_mean = np.sort(preds, axis=0)[-top_k:, :].mean(axis=0, keepdims=True)
    return np.clip(preds * top_k_mean, 0.0, 1.0).astype(np.float32)


def rank_aware_scaling(preds: np.ndarray, power: float) -> np.ndarray:
    if power <= 0:
        return preds.astype(np.float32, copy=True)
    preds = preds.astype(np.float32, copy=True)
    file_max = preds.max(axis=0, keepdims=True)
    scale = np.power(np.clip(file_max, 0.0, 1.0), float(power))
    return np.clip(preds * scale, 0.0, 1.0).astype(np.float32)


def delta_shift_smooth(preds: np.ndarray, alpha: float, adaptive: bool = False) -> np.ndarray:
    if alpha <= 0 or preds.shape[0] < 2:
        return preds.astype(np.float32, copy=True)
    preds = preds.astype(np.float32, copy=True)
    prev_view = np.concatenate([preds[:1], preds[:-1]], axis=0)
    next_view = np.concatenate([preds[1:], preds[-1:]], axis=0)
    neighbor = 0.5 * (prev_view + next_view)
    if adaptive:
        confidence = preds.max(axis=1, keepdims=True)
        alpha_arr = float(alpha) * (1.0 - np.clip(confidence, 0.0, 1.0))
    else:
        alpha_arr = float(alpha)
    out = (1.0 - alpha_arr) * preds + alpha_arr * neighbor
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def optimize_per_class_thresholds(
    scores: np.ndarray,
    targets: np.ndarray,
    thresholds: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    best_thresholds = np.full(scores.shape[1], 0.5, dtype=np.float32)
    best_scores = np.zeros(scores.shape[1], dtype=np.float32)
    for class_idx in range(scores.shape[1]):
        y_true = targets[:, class_idx].astype(np.int32)
        if int(y_true.sum()) == 0:
            continue
        best_f1 = 0.0
        best_t = 0.5
        for threshold in thresholds:
            y_pred = (scores[:, class_idx] > threshold).astype(np.int32)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            if tp + fp == 0 or tp + fn == 0:
                continue
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2.0 * precision * recall / (precision + recall + EPS)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(threshold)
        best_thresholds[class_idx] = best_t
        best_scores[class_idx] = best_f1
    return best_thresholds, best_scores


def apply_per_class_thresholds(scores: np.ndarray, thresholds: np.ndarray | None) -> np.ndarray:
    if thresholds is None:
        return scores.astype(np.float32, copy=True)
    scores = scores.astype(np.float32, copy=True)
    thresholds = np.maximum(thresholds.astype(np.float32), EPS)
    scaled = np.empty_like(scores, dtype=np.float32)
    for class_idx, threshold in enumerate(thresholds):
        class_scores = scores[:, class_idx]
        above = class_scores > threshold
        scaled[above, class_idx] = (
            0.5 + 0.5 * (class_scores[above] - threshold) / (1.0 - threshold + EPS)
        )
        scaled[~above, class_idx] = 0.5 * class_scores[~above] / threshold
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def apply_pantanal_sequence_postprocess(
    preds: np.ndarray,
    file_level_top_k: int,
    rank_power: float,
    delta_alpha: float,
    adaptive_delta: bool,
) -> np.ndarray:
    out = file_level_confidence_scale(preds, file_level_top_k)
    out = rank_aware_scaling(out, rank_power)
    out = delta_shift_smooth(out, delta_alpha, adaptive=adaptive_delta)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def postprocess_soundscape_predictions(
    preds: np.ndarray,
    species_ids: list[str],
    cfg: CFG,
    postprocess_params: dict[str, object] | None = None,
    thresholds: np.ndarray | None = None,
    taxon_temperature_vector: np.ndarray | None = None,
) -> np.ndarray:
    params = postprocess_params or load_postprocess_params(cfg)
    out = apply_taxon_temperature(preds, taxon_temperature_vector)
    out = apply_pantanal_sequence_postprocess(
        out,
        file_level_top_k=int(params.get("file_level_top_k", cfg.postprocess_file_level_top_k)),
        rank_power=float(params.get("rank_power", cfg.postprocess_rank_power)),
        delta_alpha=float(params.get("delta_alpha", cfg.postprocess_delta_alpha)),
        adaptive_delta=bool(params.get("adaptive_delta", cfg.postprocess_adaptive_delta)),
    )
    if bool(params.get("threshold_sharpening", cfg.postprocess_threshold_sharpening)):
        if thresholds is None:
            thresholds = np.full(len(species_ids), 0.5, dtype=np.float32)
        out = apply_per_class_thresholds(out, thresholds)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _oof_targets(oof_df: pd.DataFrame, species_ids: list[str]) -> np.ndarray:
    target_cols = [f"target_{label}" for label in species_ids]
    if all(col in oof_df.columns for col in target_cols):
        return oof_df[target_cols].to_numpy(dtype=np.float32)

    targets = np.zeros((len(oof_df), len(species_ids)), dtype=np.float32)
    label_to_idx = {label: idx for idx, label in enumerate(species_ids)}
    for row_idx, row in oof_df.iterrows():
        label = str(row.get("primary_label", ""))
        if label in label_to_idx:
            targets[row_idx, label_to_idx[label]] = 1.0
    return targets


def _ordered_oof_subset(oof_df: pd.DataFrame, species_ids: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    pred_cols = [f"pred_{label}" for label in species_ids]
    missing = [col for col in pred_cols if col not in oof_df.columns]
    if missing:
        raise KeyError(f"OOF predictions missing {len(missing)} prediction columns")

    df = oof_df.copy()
    if "source" in df.columns:
        soundscape = df[df["source"].astype(str).str.contains("soundscape", na=False)].copy()
        if len(soundscape) >= 50:
            df = soundscape

    group_col = "filepath" if "filepath" in df.columns else ("audio_id" if "audio_id" in df.columns else None)
    sort_cols = []
    if group_col is not None:
        sort_cols.append(group_col)
    for col in ("start_sec", "end_sec", "offset"):
        if col in df.columns:
            sort_cols.append(col)
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df, df[pred_cols].to_numpy(dtype=np.float32), _oof_targets(df, species_ids)


def _apply_groupwise_postprocess(
    df: pd.DataFrame,
    probs: np.ndarray,
    *,
    file_level_top_k: int,
    rank_power: float,
    delta_alpha: float,
    adaptive_delta: bool,
) -> np.ndarray:
    group_col = "filepath" if "filepath" in df.columns else ("audio_id" if "audio_id" in df.columns else None)
    if group_col is None:
        return apply_pantanal_sequence_postprocess(
            probs,
            file_level_top_k=file_level_top_k,
            rank_power=rank_power,
            delta_alpha=delta_alpha,
            adaptive_delta=adaptive_delta,
        )

    out = np.zeros_like(probs, dtype=np.float32)
    for _, idx in df.groupby(group_col, sort=False).groups.items():
        idx_arr = np.array(list(idx), dtype=int)
        out[idx_arr] = apply_pantanal_sequence_postprocess(
            probs[idx_arr],
            file_level_top_k=file_level_top_k,
            rank_power=rank_power,
            delta_alpha=delta_alpha,
            adaptive_delta=adaptive_delta,
        )
    return out


def fit_pantanal_postprocess_from_oof(
    oof_df: pd.DataFrame,
    species_ids: list[str],
    cfg: CFG,
) -> tuple[dict[str, object], pd.DataFrame]:
    df, probs, targets = _ordered_oof_subset(oof_df, species_ids)
    base_auc = _macro_auc(targets, probs)

    def clean_json_number(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    best_params = {
        "file_level_top_k": cfg.postprocess_file_level_top_k,
        "rank_power": cfg.postprocess_rank_power,
        "delta_alpha": cfg.postprocess_delta_alpha,
        "adaptive_delta": cfg.postprocess_adaptive_delta,
        "threshold_sharpening": cfg.postprocess_threshold_sharpening,
        "taxon_temperature": cfg.postprocess_taxon_temperature,
        "taxon_temperatures": cfg.postprocess_taxon_temperatures,
        "source": "oof_grid_search",
        "oof_rows": int(len(df)),
        "base_auc": clean_json_number(base_auc),
    }
    best_auc = float("-inf") if np.isnan(base_auc) else base_auc
    best_probs = probs

    for top_k in cfg.postprocess_file_top_k_grid:
        for rank_power in cfg.postprocess_rank_power_grid:
            for delta_alpha in cfg.postprocess_delta_alpha_grid:
                candidate = _apply_groupwise_postprocess(
                    df,
                    probs,
                    file_level_top_k=int(top_k),
                    rank_power=float(rank_power),
                    delta_alpha=float(delta_alpha),
                    adaptive_delta=cfg.postprocess_adaptive_delta,
                )
                auc = _macro_auc(targets, candidate)
                if np.isnan(auc):
                    continue
                if auc > best_auc:
                    best_auc = auc
                    best_probs = candidate
                    best_params.update(
                        {
                            "file_level_top_k": int(top_k),
                            "rank_power": float(rank_power),
                            "delta_alpha": float(delta_alpha),
                            "oof_auc": float(auc),
                        }
                    )

    best_params.setdefault("oof_auc", clean_json_number(best_auc))
    thresholds, f1_scores = optimize_per_class_thresholds(
        best_probs,
        targets,
        thresholds=cfg.postprocess_threshold_grid,
    )

    threshold_table = pd.DataFrame(
        {
            "primary_label": species_ids,
            "threshold": thresholds,
            "f1": f1_scores,
            "positives": targets.sum(axis=0).astype(int),
        }
    )

    params_path = cfg.resolved_postprocess_params_path
    params_path.parent.mkdir(parents=True, exist_ok=True)
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, sort_keys=True, allow_nan=False)

    threshold_path = cfg.resolved_per_class_thresholds_path
    threshold_table.to_csv(threshold_path, index=False)
    print(
        f"Saved Pantanal-style postprocess params to {params_path} | "
        "base_auc="
        f"{base_auc:.4f} post_auc="
        f"{best_params['oof_auc'] if best_params['oof_auc'] is not None else float('nan'):.4f}"
    )
    print(f"Saved per-class thresholds to {threshold_path} | rows={len(threshold_table)}")
    return best_params, threshold_table
