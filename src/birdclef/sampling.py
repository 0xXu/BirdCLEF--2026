import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from birdclef.config import CFG
from birdclef.validation import parse_label_list, row_labels


def dataframe_row_key(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ("filepath", "source", "primary_label", "offset", "start_sec", "end_sec"):
        if col in df.columns:
            values = df[col].fillna("__nan__").astype(str)
        else:
            values = pd.Series("__nan__", index=df.index)
        parts.append(values)
    key = parts[0]
    for values in parts[1:]:
        key = key + "|" + values
    return key


def _taxonomy_group_map(cfg: CFG) -> dict[str, str]:
    taxonomy = pd.read_csv(cfg.taxonomy_csv)
    group_col = "class_name" if "class_name" in taxonomy.columns else None
    if group_col is None:
        for candidate in ("family", "category", "class"):
            if candidate in taxonomy.columns:
                group_col = candidate
                break
    if group_col is None:
        return {}
    return dict(zip(taxonomy["primary_label"].astype(str), taxonomy[group_col].fillna("unknown").astype(str)))


def _safe_labels(value: object) -> list[str]:
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return parse_label_list(value)


def _row_all_labels(row: pd.Series) -> list[str]:
    if "all_labels" in row and pd.notna(row["all_labels"]):
        labels = _safe_labels(row["all_labels"])
        if labels:
            return labels
    return row_labels(row)


def label_counts(df: pd.DataFrame, species_ids: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in species_ids}
    valid = set(species_ids)
    for _, row in df.iterrows():
        for label in set(_row_all_labels(row)):
            if label in valid:
                counts[label] += 1
    return counts


def attach_hard_negative_metadata(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    path = cfg.resolved_hard_negative_path
    out = df.copy()
    if not path.exists():
        out["hard_negative_score"] = 0.0
        out["hard_negative_count"] = 0
        out["hard_negative_labels"] = "[]"
        return out

    hard_df = pd.read_csv(path)
    if hard_df.empty or "row_key" not in hard_df.columns:
        out["hard_negative_score"] = 0.0
        out["hard_negative_count"] = 0
        out["hard_negative_labels"] = "[]"
        return out

    out["_row_key"] = dataframe_row_key(out)
    keep_cols = ["row_key", "hard_negative_score", "hard_negative_count", "hard_negative_labels"]
    out = out.merge(hard_df[keep_cols], left_on="_row_key", right_on="row_key", how="left")
    out = out.drop(columns=["_row_key", "row_key"])
    out["hard_negative_score"] = out["hard_negative_score"].fillna(0.0).astype(float)
    out["hard_negative_count"] = out["hard_negative_count"].fillna(0).astype(int)
    out["hard_negative_labels"] = out["hard_negative_labels"].fillna("[]")
    matched = int((out["hard_negative_score"] > 0).sum())
    if matched:
        print(f"Attached hard-negative replay rows: {matched}/{len(out)} from {path}")
    return out


def make_weighted_sampler(df: pd.DataFrame, cfg: CFG, species_ids: list[str]) -> WeightedRandomSampler:
    rating_map = {5.0: 1.00, 4.0: 0.80, 3.0: 0.60, 2.0: 0.40, 1.0: 0.25, 0.0: 0.50}

    def score_rating(rating: object) -> float:
        try:
            return rating_map.get(round(float(rating), 1), 0.50)
        except (TypeError, ValueError):
            return 0.50

    counts = label_counts(df, species_ids)
    nonzero_counts = [count for count in counts.values() if count > 0]
    median_count = float(np.median(nonzero_counts)) if nonzero_counts else 1.0
    max_count = float(max(nonzero_counts)) if nonzero_counts else 1.0
    taxonomy_map = _taxonomy_group_map(cfg)
    primary_groups = df["primary_label"].astype(str).map(taxonomy_map).fillna("unknown")
    group_counts = primary_groups.value_counts().to_dict()
    max_group_count = float(max(group_counts.values())) if group_counts else 1.0

    weights = df["rating"].apply(score_rating).to_numpy(dtype=np.float32)
    sources = df["source"].astype(str) if "source" in df.columns else pd.Series("train_audio", index=df.index)
    soundscape_like = sources.str.contains("soundscape", na=False)
    weights *= np.where(soundscape_like.values, cfg.soundscape_sampling_weight, 1.0).astype(np.float32)

    if "row_weight" in df.columns:
        row_weight = pd.to_numeric(df["row_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        weights *= row_weight

    if "pseudo_weight" in df.columns:
        pseudo_weight = pd.to_numeric(df["pseudo_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        weights *= pseudo_weight

    for i, (_, row) in enumerate(df.iterrows()):
        labels = [label for label in _row_all_labels(row) if label in counts and counts[label] > 0]
        if not labels:
            continue
        rarity = max((max_count / max(1.0, counts[label])) ** cfg.rare_class_sampling_alpha for label in labels)
        if any(counts[label] <= cfg.rare_class_threshold for label in labels):
            rarity *= cfg.rare_class_extra_boost
        weights[i] *= min(rarity, cfg.rare_class_sampling_max)

        group = taxonomy_map.get(str(row["primary_label"]), "unknown")
        group_count = max(1.0, float(group_counts.get(group, 1.0)))
        group_weight = (max_group_count / group_count) ** cfg.taxonomy_group_sampling_alpha
        weights[i] *= min(group_weight, cfg.taxonomy_group_sampling_max)

    if "hard_negative_score" in df.columns:
        hard_scores = pd.to_numeric(df["hard_negative_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        weights *= 1.0 + cfg.hard_negative_sample_boost * np.clip(hard_scores, 0.0, 1.0)

    weights = np.clip(weights, cfg.sampler_min_weight, cfg.sampler_max_weight).astype(np.float32)
    print(
        "Balanced sampler ready | "
        f"n={len(weights)} min={weights.min():.3f} median={np.median(weights):.3f} "
        f"max={weights.max():.3f} median_class_count={median_count:.1f}"
    )
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def compute_positive_class_weights(df: pd.DataFrame, species_ids: list[str], cfg: CFG) -> torch.Tensor:
    counts = label_counts(df, species_ids)
    nonzero_counts = np.array([count for count in counts.values() if count > 0], dtype=np.float32)
    median_count = float(np.median(nonzero_counts)) if len(nonzero_counts) else 1.0
    weights = []
    for label in species_ids:
        count = max(1.0, float(counts.get(label, 0)))
        weight = (median_count / count) ** cfg.class_loss_weight_alpha
        if count <= cfg.rare_class_threshold:
            weight *= cfg.rare_class_loss_boost
        weights.append(float(np.clip(weight, 1.0, cfg.class_loss_weight_max)))
    tensor = torch.tensor(weights, dtype=torch.float32)
    print(
        "Positive class loss weights | "
        f"min={tensor.min().item():.3f} median={tensor.median().item():.3f} max={tensor.max().item():.3f}"
    )
    return tensor


def build_hard_negative_table_from_oof(
    oof_df: pd.DataFrame,
    species_ids: list[str],
    cfg: CFG,
) -> pd.DataFrame:
    pred_cols = [f"pred_{label}" for label in species_ids]
    target_cols = [f"target_{label}" for label in species_ids]
    if not all(col in oof_df.columns for col in pred_cols + target_cols):
        return pd.DataFrame()

    probs = oof_df[pred_cols].to_numpy(dtype=np.float32)
    targets = (oof_df[target_cols].to_numpy(dtype=np.float32) > 0.5).astype(np.float32)
    candidate_scores = np.where(targets < 0.5, probs, 0.0)
    rows = []
    keys = dataframe_row_key(oof_df)
    for row_idx in range(len(oof_df)):
        scores = candidate_scores[row_idx]
        hard_idx = np.flatnonzero(scores >= cfg.hard_negative_threshold)
        if len(hard_idx) == 0:
            continue
        hard_idx = sorted(hard_idx.tolist(), key=lambda idx: float(scores[idx]), reverse=True)[
            : cfg.hard_negative_topk
        ]
        labels = [species_ids[idx] for idx in hard_idx]
        rows.append(
            {
                "row_key": keys.iloc[row_idx],
                "filepath": oof_df.iloc[row_idx].get("filepath", ""),
                "source": oof_df.iloc[row_idx].get("source", ""),
                "primary_label": oof_df.iloc[row_idx].get("primary_label", ""),
                "hard_negative_labels": str(labels),
                "hard_negative_count": len(labels),
                "hard_negative_score": float(max(scores[idx] for idx in hard_idx)),
            }
        )

    table = pd.DataFrame(rows)
    out_path = cfg.resolved_hard_negative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"Saved hard-negative replay table to {out_path} | rows={len(table)}")
    return table
