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


def smooth_soundscape_predictions(preds: np.ndarray, kernel: tuple[float, ...]) -> np.ndarray:
    if preds.shape[0] < 2 or not kernel:
        return preds.astype(np.float32, copy=True)
    weights = np.array(kernel, dtype=np.float32)
    weights = weights / weights.sum()
    radius = len(weights) // 2
    padded = np.pad(preds.astype(np.float32), ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(preds, dtype=np.float32)
    for frame_idx in range(preds.shape[0]):
        out[frame_idx] = (padded[frame_idx : frame_idx + len(weights)] * weights[:, None]).sum(axis=0)
    return out


def apply_soundscape_max_boost(
    preds: np.ndarray,
    weight: float,
    threshold: float,
    power: float,
) -> np.ndarray:
    if weight <= 0:
        return preds.astype(np.float32, copy=True)
    preds = preds.astype(np.float32, copy=True)
    soundscape_max = preds.max(axis=0, keepdims=True)
    gate = (soundscape_max >= threshold).astype(np.float32)
    context = np.power(np.clip(soundscape_max, 0.0, 1.0), max(power, EPS))
    boosted = preds + weight * gate * context * np.maximum(context - preds, 0.0)
    return np.clip(boosted, 0.0, 1.0).astype(np.float32)


def load_calibration_table(cfg: CFG) -> pd.DataFrame | None:
    path = cfg.resolved_calibration_path
    if not path.exists():
        return None
    table = pd.read_csv(path)
    required = {"primary_label", "temperature", "bias", "power"}
    missing = required - set(table.columns)
    if missing:
        raise KeyError(f"Calibration table {path} missing columns: {sorted(missing)}")
    return table


def apply_per_class_calibration(
    preds: np.ndarray,
    species_ids: list[str],
    calibration: pd.DataFrame | None,
    blend: float,
) -> np.ndarray:
    if calibration is None or calibration.empty or blend <= 0:
        return preds.astype(np.float32, copy=True)

    calibrated = preds.astype(np.float32, copy=True)
    by_label = calibration.set_index("primary_label")
    for class_idx, label in enumerate(species_ids):
        if label not in by_label.index:
            continue
        row = by_label.loc[label]
        temperature = max(float(row["temperature"]), EPS)
        bias = float(row["bias"])
        power = max(float(row.get("power", 1.0)), EPS)
        class_probs = _sigmoid(_logit(calibrated[:, class_idx]) / temperature + bias)
        calibrated[:, class_idx] = np.power(np.clip(class_probs, 0.0, 1.0), power)

    blend = float(np.clip(blend, 0.0, 1.0))
    return ((1.0 - blend) * preds + blend * calibrated).astype(np.float32)


def postprocess_soundscape_predictions(
    preds: np.ndarray,
    species_ids: list[str],
    cfg: CFG,
    calibration: pd.DataFrame | None = None,
) -> np.ndarray:
    out = smooth_soundscape_predictions(preds, cfg.soundscape_smooth_kernel)
    out = apply_soundscape_max_boost(
        out,
        weight=cfg.soundscape_max_boost,
        threshold=cfg.soundscape_boost_threshold,
        power=cfg.soundscape_boost_power,
    )
    out = apply_per_class_calibration(out, species_ids, calibration, cfg.calibration_blend)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, EPS, 1.0 - EPS)
    return float(-(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)).mean())


def _fit_temperature_bias(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    logits = _logit(y_prob)
    best_temp = 1.0
    best_bias = 0.0
    best_loss = _binary_cross_entropy(y_true, y_prob)

    for bias in np.linspace(-4.0, 4.0, 33):
        probs = _sigmoid(logits + bias)
        loss = _binary_cross_entropy(y_true, probs)
        if loss < best_loss:
            best_loss = loss
            best_bias = float(bias)

    for temperature in np.array([0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0], dtype=np.float32):
        probs = _sigmoid(logits / float(temperature) + best_bias)
        loss = _binary_cross_entropy(y_true, probs)
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temperature)

    for bias in np.linspace(best_bias - 0.5, best_bias + 0.5, 21):
        probs = _sigmoid(logits / best_temp + bias)
        loss = _binary_cross_entropy(y_true, probs)
        if loss < best_loss:
            best_loss = loss
            best_bias = float(bias)

    return best_temp, best_bias, best_loss


def fit_calibration_table_from_oof(
    oof_df: pd.DataFrame,
    species_ids: list[str],
    cfg: CFG,
) -> pd.DataFrame:
    rows = []
    for label in species_ids:
        pred_col = f"pred_{label}"
        target_col = f"target_{label}"
        if pred_col not in oof_df.columns:
            continue

        y_prob = oof_df[pred_col].to_numpy(dtype=np.float32)
        if target_col in oof_df.columns:
            y_true = oof_df[target_col].to_numpy(dtype=np.float32)
        else:
            y_true = (oof_df["primary_label"].astype(str).values == label).astype(np.float32)

        y_true = (y_true > 0.5).astype(np.float32)
        positives = int(y_true.sum())
        negatives = int(len(y_true) - positives)
        temperature = 1.0
        bias = 0.0
        bce = _binary_cross_entropy(y_true, y_prob) if len(y_true) else np.nan
        if positives >= cfg.calibration_min_positives and negatives > 0:
            temperature, bias, bce = _fit_temperature_bias(y_true, y_prob)

        auc = np.nan
        if roc_auc_score is not None and positives > 0 and negatives > 0:
            auc = float(roc_auc_score(y_true, y_prob))

        rows.append(
            {
                "primary_label": label,
                "positives": positives,
                "negatives": negatives,
                "auc": auc,
                "mean_target": float(y_true.mean()) if len(y_true) else np.nan,
                "mean_pred": float(y_prob.mean()) if len(y_prob) else np.nan,
                "temperature": temperature,
                "bias": bias,
                "power": 1.0,
                "bce": bce,
            }
        )

    table = pd.DataFrame(rows)
    out_path = cfg.resolved_calibration_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"Saved per-class calibration table to {out_path} | rows={len(table)}")
    return table
