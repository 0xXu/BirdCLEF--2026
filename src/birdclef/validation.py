import ast
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from birdclef.config import CFG
from birdclef.deps import require_dependencies, roc_auc_score
from birdclef.utils import load_species_ids


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    strategy: str
    group_col: str


def parse_label_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    raw = str(value).strip()
    if raw in ("", "[]", "nan", "None"):
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass

    return [x.strip() for x in raw.replace(",", ";").split(";") if x.strip()]


def row_labels(row: pd.Series) -> list[str]:
    labels = [str(row["primary_label"]).strip()]
    labels.extend(parse_label_list(row.get("secondary_labels", "[]")))
    return [label for label in labels if label and label != "nan"]


def make_multilabel_targets(df: pd.DataFrame, species_ids: list[str]) -> np.ndarray:
    label2idx = {label: idx for idx, label in enumerate(species_ids)}
    targets = np.zeros((len(df), len(species_ids)), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        for label in row_labels(row):
            idx = label2idx.get(label)
            if idx is not None:
                targets[i, idx] = 1.0
    return targets


class MultiLabelStratifiedGroupKFold:
    """Greedy multi-label stratified group split.

    The assignment balances class counts across folds while keeping all rows
    from the same group in a single validation fold.
    """

    def __init__(self, n_splits: int, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, labels: np.ndarray, groups: np.ndarray):
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if len(labels) != len(groups):
            raise ValueError("labels and groups must have the same length")

        rng = random.Random(self.random_state)
        group_to_rows: dict[object, list[int]] = {}
        for idx, group in enumerate(groups):
            group_to_rows.setdefault(group, []).append(idx)

        if len(group_to_rows) < self.n_splits:
            raise ValueError(
                f"Need at least {self.n_splits} groups, found {len(group_to_rows)}"
            )

        class_counts = labels.sum(axis=0)
        active = class_counts > 0
        safe_counts = np.where(active, class_counts, 1.0)

        group_counts = []
        for group, rows in group_to_rows.items():
            counts = labels[rows].sum(axis=0)
            group_counts.append((group, rows, counts))

        rng.shuffle(group_counts)
        group_counts.sort(key=lambda x: (-np.std(x[2][active]), -x[2].sum()))

        fold_counts = np.zeros((self.n_splits, labels.shape[1]), dtype=np.float64)
        fold_sizes = np.zeros(self.n_splits, dtype=np.int64)
        fold_groups: list[list[object]] = [[] for _ in range(self.n_splits)]

        for group, rows, counts in group_counts:
            best_fold = None
            best_score = None
            for fold in range(self.n_splits):
                fold_counts[fold] += counts
                fold_sizes[fold] += len(rows)
                balance = (fold_counts[:, active] / safe_counts[active]).std(axis=0).mean()
                size_penalty = fold_sizes.std() / max(1.0, fold_sizes.mean())
                score = balance + 0.01 * size_penalty
                fold_counts[fold] -= counts
                fold_sizes[fold] -= len(rows)
                if best_score is None or score < best_score:
                    best_score = score
                    best_fold = fold

            assert best_fold is not None
            fold_counts[best_fold] += counts
            fold_sizes[best_fold] += len(rows)
            fold_groups[best_fold].append(group)

        all_indices = np.arange(len(labels))
        for fold, groups_in_fold in enumerate(fold_groups):
            val_mask = np.isin(groups, groups_in_fold)
            yield all_indices[~val_mask], all_indices[val_mask]


def _make_group_folds(
    df: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    cfg: CFG,
    strategy: str,
    group_col: str,
    base_indices: np.ndarray | None = None,
) -> list[FoldSplit]:
    n_splits = min(cfg.n_fold, len(set(groups.tolist())))
    if n_splits < 2:
        raise ValueError(f"{strategy} needs at least 2 unique groups in {group_col}")

    splitter = MultiLabelStratifiedGroupKFold(n_splits=n_splits, random_state=cfg.seed)
    folds = []
    base = np.arange(len(df)) if base_indices is None else base_indices
    for fold, (train_local, val_local) in enumerate(splitter.split(labels, groups)):
        folds.append(
            FoldSplit(
                fold=fold,
                train_idx=base[train_local],
                val_idx=base[val_local],
                strategy=strategy,
                group_col=group_col,
            )
        )
    return folds


def make_folds(df: pd.DataFrame, cfg: CFG) -> list[FoldSplit]:
    species_ids = load_species_ids(cfg)
    labels = make_multilabel_targets(df, species_ids)
    strategy = cfg.cv_strategy

    if strategy == "mlsgkf_audio_id":
        group_col = cfg.group_col
        if group_col not in df.columns:
            raise KeyError(f"Configured group_col={group_col!r} is not in the dataframe")
        return _make_group_folds(
            df=df,
            labels=labels,
            groups=df[group_col].astype(str).values,
            cfg=cfg,
            strategy=strategy,
            group_col=group_col,
        )

    if strategy in {"soundscape_site", "soundscape_date", "soundscape_hour"}:
        suffix = strategy.removeprefix("soundscape_")
        group_col = f"soundscape_{suffix}"
        if group_col not in df.columns:
            raise KeyError(f"{strategy} requires dataframe column {group_col!r}")
        sc_mask = (df["source"].astype(str) == "soundscape") & df[group_col].notna()
        sc_indices = np.flatnonzero(sc_mask.values)
        if len(sc_indices) == 0:
            raise ValueError(f"{strategy} found no soundscape rows with {group_col}")

        sc_df = df.iloc[sc_indices]
        sc_labels = labels[sc_indices]
        folds = _make_group_folds(
            df=sc_df,
            labels=sc_labels,
            groups=sc_df[group_col].astype(str).values,
            cfg=cfg,
            strategy=strategy,
            group_col=group_col,
            base_indices=sc_indices,
        )
        all_indices = np.arange(len(df))
        return [
            FoldSplit(
                fold=split.fold,
                train_idx=np.setdiff1d(all_indices, split.val_idx, assume_unique=False),
                val_idx=split.val_idx,
                strategy=split.strategy,
                group_col=split.group_col,
            )
            for split in folds
        ]

    if strategy == "source_holdout":
        val_mask = df["source"].astype(str) == "soundscape"
        if val_mask.sum() == 0 or (~val_mask).sum() == 0:
            raise ValueError("source_holdout requires both train_audio and soundscape rows")
        return [
            FoldSplit(
                fold=0,
                train_idx=np.flatnonzero((~val_mask).values),
                val_idx=np.flatnonzero(val_mask.values),
                strategy=strategy,
                group_col="source",
            )
        ]

    raise ValueError(
        f"Unknown cv_strategy={strategy!r}. "
        "Use mlsgkf_audio_id, soundscape_site, soundscape_date, "
        "soundscape_hour, or source_holdout."
    )


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def rank_normalize(preds: np.ndarray) -> np.ndarray:
    ranks = np.zeros_like(preds, dtype=np.float32)
    denom = max(1, preds.shape[0] - 1)
    for col in range(preds.shape[1]):
        order = np.argsort(preds[:, col], kind="mergesort")
        ranks[order, col] = np.arange(preds.shape[0], dtype=np.float32) / denom
    return ranks


def calculate_macro_auc(targets: np.ndarray, outputs: np.ndarray, from_logits: bool = True) -> float:
    metrics = calculate_per_class_auc(targets, outputs, from_logits=from_logits)
    valid = metrics["auc"].dropna()
    return float(valid.mean()) if len(valid) else 0.0


def calculate_per_class_auc(
    targets: np.ndarray,
    outputs: np.ndarray,
    species_ids: list[str] | None = None,
    from_logits: bool = True,
) -> pd.DataFrame:
    require_dependencies(("scikit-learn", roc_auc_score))
    targets = (targets > 0.5).astype(np.float32)
    probs = sigmoid(outputs) if from_logits else outputs
    rows = []
    for idx in range(targets.shape[1]):
        positives = int(targets[:, idx].sum())
        negatives = int(targets.shape[0] - positives)
        auc = np.nan
        if positives > 0 and negatives > 0:
            auc = float(roc_auc_score(targets[:, idx], probs[:, idx]))
        rows.append(
            {
                "class_idx": idx,
                "primary_label": species_ids[idx] if species_ids else str(idx),
                "auc": auc,
                "positives": positives,
                "negatives": negatives,
            }
        )
    return pd.DataFrame(rows)


def calculate_group_auc(
    df: pd.DataFrame,
    targets: np.ndarray,
    outputs: np.ndarray,
    group_col: str,
    from_logits: bool = True,
) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "auc", "rows"])
    rows = []
    for group, idx in df.groupby(group_col, dropna=False).groups.items():
        idx_arr = np.array(list(idx), dtype=int)
        if len(idx_arr) < 2:
            continue
        auc = calculate_macro_auc(targets[idx_arr], outputs[idx_arr], from_logits=from_logits)
        rows.append({group_col: group, "auc": auc, "rows": len(idx_arr)})
    return pd.DataFrame(rows)


def attach_taxonomy_groups(per_class: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    taxonomy = pd.read_csv(cfg.taxonomy_csv)
    group_col = None
    for candidate in ("class_name", "class", "category", "family"):
        if candidate in taxonomy.columns:
            group_col = candidate
            break
    if group_col is None:
        per_class["taxonomy_group"] = "unknown"
        return per_class
    groups = taxonomy[["primary_label", group_col]].copy()
    groups = groups.rename(columns={group_col: "taxonomy_group"})
    return per_class.merge(groups, on="primary_label", how="left")


def summarize_fold_distribution(
    df: pd.DataFrame,
    folds: list[FoldSplit],
    cfg: CFG,
) -> pd.DataFrame:
    species_ids = load_species_ids(cfg)
    labels = make_multilabel_targets(df, species_ids)
    rows = []
    for split in folds:
        val_labels = labels[split.val_idx]
        rows.append(
            {
                "fold": split.fold,
                "strategy": split.strategy,
                "group_col": split.group_col,
                "train_rows": len(split.train_idx),
                "val_rows": len(split.val_idx),
                "train_groups": df.iloc[split.train_idx][split.group_col].nunique()
                if split.group_col in df.columns
                else np.nan,
                "val_groups": df.iloc[split.val_idx][split.group_col].nunique()
                if split.group_col in df.columns
                else np.nan,
                "val_positive_classes": int((val_labels.sum(axis=0) > 0).sum()),
                "val_missing_classes": int((val_labels.sum(axis=0) == 0).sum()),
                "val_source_counts": df.iloc[split.val_idx]["source"].value_counts().to_dict()
                if "source" in df.columns
                else {},
            }
        )
    return pd.DataFrame(rows)


def save_oof_report(
    df: pd.DataFrame,
    oof_logits: np.ndarray,
    oof_targets: np.ndarray,
    fold_ids: np.ndarray,
    cfg: CFG,
) -> pd.DataFrame | None:
    species_ids = load_species_ids(cfg)
    valid_mask = fold_ids >= 0
    if not valid_mask.any():
        return None

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = sigmoid(oof_logits[valid_mask])
    report_df = df.iloc[np.flatnonzero(valid_mask)].reset_index(drop=True).copy()
    report_df.insert(0, "fold", fold_ids[valid_mask])
    for idx, label in enumerate(species_ids):
        report_df[f"pred_{label}"] = probs[:, idx]

    for fold in sorted(report_df["fold"].unique()):
        fold_df = report_df[report_df["fold"] == fold].copy()
        fold_df.to_csv(out_dir / f"oof_predictions_fold{int(fold)}.csv", index=False)

    merged_df = merge_oof_reports(cfg)
    if merged_df is None:
        merged_df = report_df
        merged_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    return save_oof_metrics(merged_df, cfg)


def merge_oof_reports(cfg: CFG) -> pd.DataFrame | None:
    out_dir = Path(cfg.output_dir)
    fold_paths = sorted(out_dir.glob("oof_predictions_fold*.csv"))
    if not fold_paths:
        return None

    by_fold = {}
    for path in fold_paths:
        fold_df = pd.read_csv(path)
        if "fold" not in fold_df.columns or fold_df.empty:
            continue
        fold = int(fold_df["fold"].iloc[0])
        by_fold[fold] = fold_df
    if not by_fold:
        return None

    merged = pd.concat([by_fold[fold] for fold in sorted(by_fold)], ignore_index=True)
    merged = merged.drop_duplicates(subset=["fold", "filepath", "offset", "primary_label"])
    merged.to_csv(out_dir / "oof_predictions.csv", index=False)
    return merged


def save_oof_metrics(oof_df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    out_dir = Path(cfg.output_dir)
    species_ids = load_species_ids(cfg)
    pred_cols = [f"pred_{label}" for label in species_ids]
    missing = [col for col in pred_cols if col not in oof_df.columns]
    if missing:
        raise KeyError(f"OOF predictions missing {len(missing)} prediction columns")

    targets = make_multilabel_targets(oof_df, species_ids)
    probs = oof_df[pred_cols].to_numpy(dtype=np.float32)

    per_class = calculate_per_class_auc(
        targets,
        probs,
        species_ids=species_ids,
        from_logits=False,
    )
    per_class = attach_taxonomy_groups(per_class, cfg)
    per_class.to_csv(out_dir / "per_class_auc.csv", index=False)

    if cfg.rank_normalize_oof:
        rank_probs = rank_normalize(probs)
        per_class_rank = calculate_per_class_auc(
            targets,
            rank_probs,
            species_ids=species_ids,
            from_logits=False,
        )
        per_class_rank = attach_taxonomy_groups(per_class_rank, cfg)
        per_class_rank.to_csv(out_dir / "per_class_auc_rank.csv", index=False)

    source_auc = calculate_group_auc(
        oof_df.reset_index(drop=True),
        targets,
        probs,
        group_col="source",
        from_logits=False,
    )
    source_auc.to_csv(out_dir / "source_auc.csv", index=False)

    if "taxonomy_group" in per_class.columns:
        taxonomy_auc = (
            per_class.dropna(subset=["auc"])
            .groupby("taxonomy_group", dropna=False)
            .agg(auc=("auc", "mean"), classes=("auc", "count"), positives=("positives", "sum"))
            .reset_index()
        )
        taxonomy_auc.to_csv(out_dir / "taxonomy_group_auc.csv", index=False)
    return per_class
