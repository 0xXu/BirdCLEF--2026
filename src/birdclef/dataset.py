import ast
from pathlib import Path

import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate

from birdclef.config import CFG
from birdclef.utils import load_species_ids


def build_train_df(cfg: CFG) -> pd.DataFrame:
    taxonomy = pd.read_csv(cfg.taxonomy_csv)
    valid_labels = set(taxonomy["primary_label"].astype(str))

    train_df = pd.read_csv(cfg.train_csv)
    train_df["primary_label"] = train_df["primary_label"].astype(str)
    train_df = train_df[train_df["primary_label"].isin(valid_labels)].copy()
    train_df["filepath"] = train_df["filename"].apply(lambda x: str(cfg.train_datadir / str(x)))
    train_df["source"] = "train_audio"
    train_df["offset"] = np.nan

    sc_labels_df = pd.read_csv(cfg.sc_labels_csv)

    def hms_to_sec(ts: str) -> int:
        h, m, s = str(ts).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    sc_labels_df["start_sec"] = sc_labels_df["start"].apply(hms_to_sec)
    sc_dir = cfg.train_datadir.parent / "train_soundscapes"
    sc_rows = []

    for _, row in sc_labels_df.iterrows():
        labels = [
            label.strip()
            for label in str(row["primary_label"]).split(";")
            if label.strip() in valid_labels
        ]
        if not labels:
            continue
        audio_path = sc_dir / row["filename"]
        if not audio_path.exists():
            continue
        sc_rows.append(
            {
                "filepath": str(audio_path),
                "primary_label": labels[0],
                "secondary_labels": str(labels[1:]) if len(labels) > 1 else "[]",
                "source": "soundscape",
                "offset": float(row["start_sec"]),
                "rating": 0.0,
            }
        )

    sc_df = pd.DataFrame(sc_rows)
    combined_df = pd.concat([train_df, sc_df], ignore_index=True, sort=False)
    combined_df["primary_label"] = combined_df["primary_label"].astype(str)
    print(
        "Built training frame:",
        f"train_audio={len(train_df)} soundscape={len(sc_df)} total={len(combined_df)}",
    )
    return combined_df


class BirdCLEFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache: np.ndarray, cfg: CFG, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.cache = cache
        self.cfg = cfg
        self.mode = mode
        self.species_ids = load_species_ids(cfg)
        self.num_classes = len(self.species_ids)
        self.label2idx = {label: idx for idx, label in enumerate(self.species_ids)}
        self.targets = self._precompute_targets()

    def _precompute_targets(self) -> torch.Tensor:
        out = np.zeros((len(self.df), self.num_classes), dtype=np.float32)
        secondary_weight = self.cfg.secondary_weight
        for i, row in self.df.iterrows():
            primary = str(row["primary_label"])
            if primary in self.label2idx:
                out[i, self.label2idx[primary]] = 1.0

            secondary = row.get("secondary_labels", "[]")
            if pd.notna(secondary) and str(secondary) not in ("[]", "", "nan"):
                try:
                    labels = ast.literal_eval(str(secondary))
                except (ValueError, SyntaxError):
                    labels = []
                for label in labels:
                    label = str(label).strip()
                    if label in self.label2idx:
                        out[i, self.label2idx[label]] = secondary_weight
        return torch.from_numpy(out)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        spec = self.cache[idx].astype(np.float32)
        flat = spec.ravel()
        p99_index = int(0.99 * len(flat))
        p99 = float(np.partition(flat, p99_index)[p99_index])
        if p99 > 1e-8:
            spec = spec / p99
        np.clip(spec, 0, 1, out=spec)

        spec_tensor = torch.from_numpy(spec).unsqueeze(0)
        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec_tensor = self._spec_augment(spec_tensor)

        return {"melspec": spec_tensor, "target": self.targets[idx]}

    def _spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                width = random.randint(5, 30)
                start = random.randint(0, max(0, spec.shape[2] - width))
                spec[0, :, start : start + width] = 0
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                height = random.randint(5, 20)
                start = random.randint(0, max(0, spec.shape[1] - height))
                spec[0, start : start + height, :] = 0
        if random.random() < 0.5:
            spec = spec * random.uniform(0.7, 1.3) + random.uniform(-0.1, 0.1)
            spec = torch.clamp(spec, 0, 1)
        return spec


def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    rating_map = {5.0: 1.00, 4.0: 0.80, 3.0: 0.60, 2.0: 0.40, 1.0: 0.25, 0.0: 0.50}

    def score_rating(rating: object) -> float:
        try:
            return rating_map.get(round(float(rating), 1), 0.50)
        except (TypeError, ValueError):
            return 0.50

    weights = df["rating"].apply(score_rating).values.astype("float32")
    species_counts = df["primary_label"].value_counts().to_dict()
    max_count = max(species_counts.values())
    for i, label in enumerate(df["primary_label"]):
        weights[i] *= (max_count / species_counts[label]) ** 0.3

    print(
        f"WeightedRandomSampler ready | n={len(weights)} "
        f"min={weights.min():.3f} max={weights.max():.3f}"
    )
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


collate_fn = default_collate
