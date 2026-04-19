import ast
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from birdclef.audio import LogMelExtractor, WaveformStore, logmel_to_shape
from birdclef.config import CFG
from birdclef.deps import sf
from birdclef.utils import load_species_ids
from birdclef.validation import parse_label_list


def _audio_id_from_filename(filename: object) -> str:
    raw = str(filename)
    return str(Path(raw).with_suffix(""))


def _parse_soundscape_filename(filename: object) -> dict[str, object]:
    stem = Path(str(filename)).stem
    tokens = [token for token in re.split(r"[_\-\s]+", stem) if token]
    site = tokens[0] if tokens else stem
    date = next(
        (
            token
            for token in tokens
            if re.fullmatch(r"\d{8}", token) or re.fullmatch(r"\d{4}\d{2}\d{2}", token)
        ),
        None,
    )
    time_token = next(
        (
            token
            for token in tokens
            if re.fullmatch(r"\d{6}", token) or re.fullmatch(r"\d{2}:\d{2}:\d{2}", token)
        ),
        None,
    )
    hour = None
    if time_token is not None:
        hour = int(time_token[:2])
    return {
        "soundscape_site": site,
        "soundscape_date": date,
        "soundscape_hour": hour,
    }


def build_train_df(cfg: CFG) -> pd.DataFrame:
    taxonomy = pd.read_csv(cfg.taxonomy_csv)
    valid_labels = set(taxonomy["primary_label"].astype(str))

    train_df = pd.read_csv(cfg.train_csv)
    train_df["primary_label"] = train_df["primary_label"].astype(str)
    train_df = train_df[train_df["primary_label"].isin(valid_labels)].copy()
    train_df["audio_id"] = train_df["filename"].apply(_audio_id_from_filename)
    train_df["filepath"] = train_df["filename"].apply(lambda x: str(cfg.train_datadir / str(x)))
    train_df["source"] = "train_audio"
    train_df["offset"] = np.nan
    train_df["start_sec"] = np.nan
    train_df["end_sec"] = np.nan
    train_df["soundscape_site"] = np.nan
    train_df["soundscape_date"] = np.nan
    train_df["soundscape_hour"] = np.nan
    train_df["row_weight"] = 1.0
    train_df["is_background"] = 0

    sc_labels_df = pd.read_csv(cfg.sc_labels_csv)

    def hms_to_sec(ts: str) -> int:
        if isinstance(ts, (int, float)) and not pd.isna(ts):
            return int(ts)
        h, m, s = str(ts).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    if "start_sec" not in sc_labels_df.columns:
        sc_labels_df["start_sec"] = sc_labels_df["start"].apply(hms_to_sec)
    if "end_sec" not in sc_labels_df.columns and "end" in sc_labels_df.columns:
        sc_labels_df["end_sec"] = sc_labels_df["end"].apply(hms_to_sec)
    dedup_cols = ["filename", "start_sec", "primary_label"]
    if "end_sec" in sc_labels_df.columns:
        dedup_cols.insert(2, "end_sec")
    before_dedup = len(sc_labels_df)
    sc_labels_df = sc_labels_df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    if before_dedup != len(sc_labels_df):
        print(f"Dropped duplicated soundscape labels: {before_dedup - len(sc_labels_df)}")

    sc_dir = cfg.train_datadir.parent / "train_soundscapes"
    sc_rows = []

    labels_by_window: dict[tuple[str, float, float], list[str]] = {}
    labels_by_start: dict[tuple[str, float], list[str]] = {}
    file_to_label_rows: dict[str, list[pd.Series]] = {}
    for _, row in sc_labels_df.iterrows():
        labels = [
            label.strip()
            for label in str(row["primary_label"]).split(";")
            if label.strip() in valid_labels
        ]
        if not labels:
            continue
        filename = str(row["filename"])
        file_to_label_rows.setdefault(filename, []).append(row)
        start_sec = float(row["start_sec"])
        end_sec = float(row.get("end_sec", start_sec + cfg.soundscape_window_sec))
        window_key = (filename, start_sec, end_sec)
        start_key = (filename, start_sec)
        for target in (labels_by_window.setdefault(window_key, []), labels_by_start.setdefault(start_key, [])):
            for label in labels:
                if label not in target:
                    target.append(label)

    def soundscape_duration(audio_path: Path, label_rows: list[pd.Series]) -> float:
        if sf is not None:
            try:
                info = sf.info(str(audio_path))
                if info.frames > 0 and info.samplerate > 0:
                    return float(info.frames) / float(info.samplerate)
            except Exception:
                pass
        max_label_end = 0.0
        for label_row in label_rows:
            start_sec = float(label_row["start_sec"])
            max_label_end = max(
                max_label_end,
                float(label_row.get("end_sec", start_sec + cfg.soundscape_window_sec)),
            )
        rounded_label_end = math.ceil(max_label_end / cfg.soundscape_window_sec) * cfg.soundscape_window_sec
        return max(cfg.soundscape_window_sec, rounded_label_end)

    def append_soundscape_row(
        *,
        audio_path: Path,
        filename: str,
        start_sec: float,
        end_sec: float,
        labels: list[str],
    ) -> None:
        soundscape_meta = _parse_soundscape_filename(filename)
        has_labels = len(labels) > 0
        label_center = (start_sec + end_sec) / 2.0
        context_offset = max(0.0, label_center - cfg.target_duration / 2.0)
        sc_rows.append(
            {
                "filename": filename,
                "audio_id": _audio_id_from_filename(filename),
                "filepath": str(audio_path),
                "row_id": f"{Path(filename).stem}_{int(round(end_sec))}",
                "primary_label": labels[0] if has_labels else cfg.soundscape_background_label,
                "secondary_labels": str(labels[1:]) if has_labels else "[]",
                "all_labels": str(labels),
                "source": "soundscape",
                "offset": context_offset,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "rating": 5.0 if has_labels else 0.0,
                "row_weight": (
                    cfg.soundscape_positive_weight if has_labels else cfg.soundscape_background_weight
                ),
                "is_background": 0 if has_labels else 1,
                **soundscape_meta,
            }
        )

    if cfg.soundscape_full_windows:
        if cfg.soundscape_include_unlabeled_files:
            soundscape_paths = sorted(sc_dir.glob("*.ogg"))
        else:
            soundscape_paths = [sc_dir / filename for filename in sorted(file_to_label_rows)]
        for audio_path in soundscape_paths:
            if not audio_path.exists():
                continue
            filename = audio_path.name
            label_rows = file_to_label_rows.get(filename, [])
            duration = soundscape_duration(audio_path, label_rows)
            n_frames = max(1, int(math.ceil(duration / cfg.soundscape_window_sec)))
            for frame_idx in range(n_frames):
                start_sec = float(frame_idx * cfg.soundscape_window_sec)
                end_sec = float(start_sec + cfg.soundscape_window_sec)
                labels = labels_by_window.get(
                    (filename, start_sec, end_sec),
                    labels_by_start.get((filename, start_sec), []),
                )
                append_soundscape_row(
                    audio_path=audio_path,
                    filename=filename,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    labels=list(labels),
                )
    else:
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
            start_sec = float(row["start_sec"])
            end_sec = float(row.get("end_sec", start_sec + cfg.soundscape_window_sec))
            append_soundscape_row(
                audio_path=audio_path,
                filename=str(row["filename"]),
                start_sec=start_sec,
                end_sec=end_sec,
                labels=labels,
            )

    sc_df = pd.DataFrame(sc_rows)
    train_df["all_labels"] = train_df.apply(
        lambda row: str([str(row["primary_label"])] + parse_label_list(row.get("secondary_labels", "[]"))),
        axis=1,
    )
    combined_df = pd.concat([train_df, sc_df], ignore_index=True, sort=False)
    combined_df["primary_label"] = combined_df["primary_label"].astype(str)
    soundscape_background = int(sc_df["is_background"].sum()) if not sc_df.empty else 0
    soundscape_positive = int(len(sc_df) - soundscape_background)
    print(
        "Built training frame:",
        f"train_audio={len(train_df)} soundscape={len(sc_df)} "
        f"soundscape_positive={soundscape_positive} soundscape_background={soundscape_background} "
        f"total={len(combined_df)}",
    )
    return combined_df


class BirdCLEFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: CFG, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.mode = mode
        self.species_ids = load_species_ids(cfg)
        self.num_classes = len(self.species_ids)
        self.label2idx = {label: idx for idx, label in enumerate(self.species_ids)}
        self.targets, self.target_masks = self._precompute_targets_and_masks()
        self.sample_weights, self.loss_weights = self._precompute_training_weights()
        self.waveforms = WaveformStore(cfg)
        self.logmel = LogMelExtractor(cfg)

    def _precompute_targets_and_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        targets = np.zeros((len(self.df), self.num_classes), dtype=np.float32)
        masks = np.ones((len(self.df), self.num_classes), dtype=np.float32)

        for i, row in self.df.iterrows():
            primary = str(row["primary_label"]).strip()
            source = str(row.get("source", "train_audio"))

            if source == "pseudo_soundscape":
                pseudo_type = str(row.get("pseudo_type", "positive"))
                if pseudo_type == "background":
                    targets[i, :] = 0.0
                    masks[i, :] = 1.0
                    continue

                if pseudo_type == "hard_negative":
                    targets[i, :] = 0.0
                    masks[i, :] = 0.0
                    hard_labels = parse_label_list(row.get("hard_negative_labels", "[]"))
                    if not hard_labels:
                        probs = [
                            (label, float(row.get(f"pseudo_{label}", 0.0)))
                            for label in self.species_ids
                        ]
                        ranked = sorted(probs, key=lambda item: item[1], reverse=True)
                        hard_labels = [
                            label for label, _ in ranked[: self.cfg.pseudo_hard_negative_max_labels]
                        ]
                    for label in hard_labels:
                        idx = self.label2idx.get(label)
                        if idx is not None:
                            masks[i, idx] = 1.0
                    continue

                for label in self.species_ids:
                    idx = self.label2idx[label]
                    prob = row.get(f"pseudo_{label}", np.nan)
                    if pd.isna(prob):
                        masks[i, idx] = 0.0
                        continue
                    prob_f = float(prob)
                    targets[i, idx] = prob_f
                    masks[i, idx] = 1.0 if prob_f >= self.cfg.pseudo_mask_prob else 0.0
                primary_idx = self.label2idx.get(primary)
                if primary_idx is not None:
                    masks[i, primary_idx] = 1.0
                continue

            if source == "soundscape":
                labels = parse_label_list(row.get("all_labels", row.get("secondary_labels", "[]")))
                if primary in self.label2idx:
                    labels = [primary] + [label for label in labels if label != primary]
                for label in labels:
                    idx = self.label2idx.get(str(label).strip())
                    if idx is not None:
                        targets[i, idx] = self.cfg.soundscape_label_weight
                        masks[i, idx] = 1.0
                continue

            primary_idx = self.label2idx.get(primary)
            if primary_idx is not None:
                targets[i, primary_idx] = 1.0

            secondary = row.get("secondary_labels", "[]")
            if pd.notna(secondary) and str(secondary) not in ("[]", "", "nan"):
                try:
                    labels = ast.literal_eval(str(secondary))
                except (ValueError, SyntaxError):
                    labels = parse_label_list(secondary)
                for label in labels:
                    idx = self.label2idx.get(str(label).strip())
                    if idx is None:
                        continue
                    masks[i, idx] = 0.0

        return torch.from_numpy(targets), torch.from_numpy(masks)

    def _precompute_training_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        sample_weights = np.ones(len(self.df), dtype=np.float32)
        loss_weights = np.ones((len(self.df), self.num_classes), dtype=np.float32)

        if "pseudo_weight" in self.df.columns:
            sample_weights *= pd.to_numeric(
                self.df["pseudo_weight"], errors="coerce"
            ).fillna(1.0).to_numpy(dtype=np.float32)

        if "row_weight" in self.df.columns:
            sample_weights *= pd.to_numeric(
                self.df["row_weight"], errors="coerce"
            ).fillna(1.0).to_numpy(dtype=np.float32)

        if "hard_negative_score" in self.df.columns:
            scores = pd.to_numeric(
                self.df["hard_negative_score"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=np.float32)
            sample_weights *= 1.0 + self.cfg.hard_negative_sample_boost * np.clip(scores, 0.0, 1.0)

        if "hard_negative_labels" in self.df.columns:
            for row_idx, value in enumerate(self.df["hard_negative_labels"].fillna("[]")):
                for label in parse_label_list(value):
                    class_idx = self.label2idx.get(label)
                    if class_idx is not None:
                        loss_weights[row_idx, class_idx] *= self.cfg.hard_negative_loss_boost

        return torch.from_numpy(sample_weights), torch.from_numpy(loss_weights)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        audio = self.waveforms.load(row["filepath"])
        seg = self._crop_audio(audio, row)
        spec = logmel_to_shape(self.logmel(seg), self.cfg)

        flat = spec.ravel()
        p99_index = int(0.99 * len(flat))
        p99 = float(np.partition(flat, p99_index)[p99_index])
        if p99 > 1e-8:
            spec = spec / p99
        np.clip(spec, 0, 1, out=spec)

        spec_tensor = torch.from_numpy(spec.astype(np.float32)).unsqueeze(0)
        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec_tensor = self._spec_augment(spec_tensor)

        return {
            "melspec": spec_tensor,
            "target": self.targets[idx],
            "target_mask": self.target_masks[idx],
            "sample_weight": self.sample_weights[idx],
            "loss_weight": self.loss_weights[idx],
        }

    def _crop_audio(self, audio: np.ndarray, row: pd.Series) -> np.ndarray:
        target = self.cfg.target_samples
        if len(audio) < target:
            pad_total = target - len(audio)
            if self.mode == "train" and self.cfg.random_pad_train:
                left = random.randint(0, pad_total)
            else:
                left = pad_total // 2
            right = pad_total - left
            return np.pad(audio, (left, right)).astype(np.float32)

        max_start = len(audio) - target
        source = str(row.get("source", "train_audio"))
        start_sec = row.get("start_sec", np.nan)
        end_sec = row.get("end_sec", np.nan)

        if source in {"soundscape", "pseudo_soundscape"} and pd.notna(start_sec) and pd.notna(end_sec):
            event_start = float(start_sec)
            event_end = float(end_sec)
            if self.mode == "train":
                low = max(0.0, event_end - self.cfg.target_duration)
                high = min(event_start, max_start / self.cfg.fs)
                center_start = max(0.0, (event_start + event_end) / 2.0 - self.cfg.target_duration / 2.0)
                jitter_low = center_start - self.cfg.event_crop_jitter_sec
                jitter_high = center_start + self.cfg.event_crop_jitter_sec
                low = max(low, jitter_low)
                high = min(high, jitter_high)
                if high <= low:
                    start = int(max(0, min(center_start * self.cfg.fs, max_start)))
                else:
                    start = int(random.uniform(low, high) * self.cfg.fs)
            else:
                center_start = (event_start + event_end) / 2.0 - self.cfg.target_duration / 2.0
                start = int(center_start * self.cfg.fs)
        elif self.mode == "train":
            start = random.randint(0, max_start)
        else:
            offset = row.get("offset", np.nan)
            if pd.notna(offset):
                start = int(float(offset) * self.cfg.fs)
            else:
                start = max_start // 2

        start = max(0, min(int(start), max_start))
        return audio[start : start + target].astype(np.float32)

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

collate_fn = default_collate
