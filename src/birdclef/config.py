import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class CFG:
    seed: int = 42
    debug: bool = False
    num_workers: int = 0
    output_dir: Path = Path("./kaggle/working")

    train_datadir: Path = Path("./kaggle/input/birdclef-2026/train_audio")
    train_csv: Path = Path("./kaggle/input/birdclef-2026/train.csv")
    test_sc_dir: Path = Path("./kaggle/input/birdclef-2026/test_soundscapes")
    submission_csv: Path = Path("./kaggle/input/birdclef-2026/sample_submission.csv")
    taxonomy_csv: Path = Path("./kaggle/input/birdclef-2026/taxonomy.csv")
    sc_labels_csv: Path = Path("./kaggle/input/birdclef-2026/train_soundscapes_labels.csv")

    model_name: str = "tf_efficientnet_es"
    pretrained_path: Path = Path(
        "./kaggle/input/models/timm/tf-efficientnet/pytorch/tf-efficientnet-es/1"
    )
    in_channels: int = 1

    fs: int = 32_000
    target_duration: float = 10.0
    target_shape: tuple[int, int] = (256, 256)
    n_fft: int = 2048
    win_length: int = 626
    hop_length: int = 313
    n_mels: int = 256
    fmin: int = 20
    fmax: int = 16_000
    mel_power: float = 2.0
    mel_top_db: float = 80.0
    mel_norm: str = "slaney"
    mel_scale: str = "htk"
    waveform_cache_name: str = "waveform_cache_32k"
    waveform_lru_size: int = 256
    event_crop_jitter_sec: float = 2.5
    random_pad_train: bool = True
    soundscape_window_sec: float = 5.0
    soundscape_full_windows: bool = True
    soundscape_include_unlabeled_files: bool = False
    soundscape_background_label: str = "__background__"
    soundscape_positive_weight: float = 1.00
    soundscape_background_weight: float = 0.35
    pseudo_path: Path | None = None
    pseudo_min_primary_prob: float = 0.50
    pseudo_label_prob: float = 0.35
    pseudo_mask_prob: float = 0.10
    pseudo_max_labels: int = 5
    pseudo_sampling_weight: float = 0.40
    pseudo_include_background: bool = True
    pseudo_background_max_prob: float = 0.08
    pseudo_background_weight: float = 0.12
    pseudo_include_hard_negatives: bool = True
    pseudo_hard_negative_min_prob: float = 0.20
    pseudo_hard_negative_weight: float = 0.25
    pseudo_hard_negative_max_labels: int = 5
    pseudo_teacher_tta_crops: int = 3
    pseudo_batch_size: int = 16
    pseudo_max_files: int | None = None
    postprocess_params_path: Path | None = None
    per_class_thresholds_path: Path | None = None
    postprocess_file_level_top_k: int = 2
    postprocess_rank_power: float = 0.4
    postprocess_delta_alpha: float = 0.15
    postprocess_adaptive_delta: bool = True
    postprocess_threshold_sharpening: bool = True
    postprocess_taxon_temperature: bool = True
    postprocess_taxon_temperatures: dict[str, float] = field(
        default_factory=lambda: {
            "Aves": 1.10,
            "Insecta": 0.95,
            "Amphibia": 0.95,
            "Reptilia": 0.95,
            "Mammalia": 0.95,
        }
    )
    postprocess_file_top_k_grid: tuple[int, ...] = (0, 1, 2, 3)
    postprocess_rank_power_grid: tuple[float, ...] = (0.0, 0.4, 0.5)
    postprocess_delta_alpha_grid: tuple[float, ...] = (0.0, 0.15, 0.20)
    postprocess_threshold_grid: tuple[float, ...] = (
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    )
    hard_negative_path: Path | None = None
    hard_negative_threshold: float = 0.35
    hard_negative_topk: int = 5
    hard_negative_sample_boost: float = 2.5
    hard_negative_loss_boost: float = 2.0
    rare_class_threshold: int = 30
    rare_class_sampling_alpha: float = 0.55
    rare_class_sampling_max: float = 8.0
    rare_class_extra_boost: float = 1.75
    rare_class_loss_boost: float = 1.50
    taxonomy_group_sampling_alpha: float = 0.35
    taxonomy_group_sampling_max: float = 3.0
    class_loss_weight_alpha: float = 0.35
    class_loss_weight_max: float = 4.0
    soundscape_sampling_weight: float = 1.25
    sampler_min_weight: float = 0.05
    sampler_max_weight: float = 12.0

    epochs: int = 6
    batch_size: int = 16
    n_fold: int = 5
    selected_folds: list[int] = field(default_factory=lambda: [0])
    cv_strategy: str = "mlsgkf_audio_id"
    group_col: str = "audio_id"
    save_oof: bool = True
    rank_normalize_oof: bool = True
    use_compile: bool = False

    lr: float = 7e-4
    weight_decay: float = 1e-2
    min_lr: float = 1e-6
    t_max: int = 6

    focal_gamma: float = 2.0
    focal_weight: float = 0.7
    bce_weight: float = 0.3
    label_smoothing: float = 0.08

    aug_prob: float = 0.5
    mixup_alpha: float = 0.5
    mixup_prob: float = 0.5
    cutmix_prob: float = 0.3

    soundscape_label_weight: float = 1.0
    use_rating_weight: bool = True
    tta_crops: int = 3
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    cache_override: Path | None = None

    def apply_debug_settings(self) -> None:
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]

    @property
    def target_samples(self) -> int:
        return int(self.target_duration * self.fs)

    @property
    def waveform_cache_dir(self) -> Path:
        if self.cache_override is not None:
            return self.cache_override
        return self.output_dir / self.waveform_cache_name

    @property
    def waveform_manifest_path(self) -> Path:
        return self.waveform_cache_dir / "manifest.csv"

    @property
    def resolved_pseudo_path(self) -> Path:
        if self.pseudo_path is not None:
            return self.pseudo_path
        return self.output_dir / "pseudo_labels.csv"

    @property
    def resolved_postprocess_params_path(self) -> Path:
        if self.postprocess_params_path is not None:
            return self.postprocess_params_path
        return self.output_dir / "postprocess_params.json"

    @property
    def resolved_per_class_thresholds_path(self) -> Path:
        if self.per_class_thresholds_path is not None:
            return self.per_class_thresholds_path
        return self.output_dir / "per_class_thresholds.csv"

    @property
    def resolved_hard_negative_path(self) -> Path:
        if self.hard_negative_path is not None:
            return self.hard_negative_path
        return self.output_dir / "hard_negatives.csv"


def parse_folds(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(x) for x in raw.split(",") if x.strip())


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x.strip())


def parse_taxon_temperatures(raw: str) -> dict[str, float]:
    out = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        out[key.strip()] = float(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local-first BirdCLEF 2026 training and inference pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_args(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--data-root", type=Path, default=None)
        cmd.add_argument("--output-dir", type=Path, default=Path("./kaggle/working"))
        cmd.add_argument("--epochs", type=int, default=6)
        cmd.add_argument("--batch-size", type=int, default=16)
        cmd.add_argument("--folds", type=str, default="0")
        cmd.add_argument("--cv-strategy", type=str, default="mlsgkf_audio_id")
        cmd.add_argument("--group-col", type=str, default="audio_id")
        cmd.add_argument("--disable-oof", action="store_true")
        cmd.add_argument("--disable-rank-oof", action="store_true")
        cmd.add_argument("--pseudo-path", type=Path, default=None)
        cmd.add_argument("--pseudo-min-primary-prob", type=float, default=0.50)
        cmd.add_argument("--pseudo-label-prob", type=float, default=0.35)
        cmd.add_argument("--pseudo-mask-prob", type=float, default=0.10)
        cmd.add_argument("--pseudo-max-labels", type=int, default=5)
        cmd.add_argument("--pseudo-sampling-weight", type=float, default=0.40)
        cmd.add_argument("--disable-soundscape-full-windows", action="store_true")
        cmd.add_argument("--include-unlabeled-soundscape-files", action="store_true")
        cmd.add_argument("--soundscape-window-sec", type=float, default=5.0)
        cmd.add_argument("--soundscape-positive-weight", type=float, default=1.00)
        cmd.add_argument("--soundscape-background-weight", type=float, default=0.35)
        cmd.add_argument("--disable-pseudo-background", action="store_true")
        cmd.add_argument("--pseudo-background-max-prob", type=float, default=0.08)
        cmd.add_argument("--pseudo-background-weight", type=float, default=0.12)
        cmd.add_argument("--disable-pseudo-hard-negatives", action="store_true")
        cmd.add_argument("--pseudo-hard-negative-min-prob", type=float, default=0.20)
        cmd.add_argument("--pseudo-hard-negative-weight", type=float, default=0.25)
        cmd.add_argument("--pseudo-hard-negative-max-labels", type=int, default=5)
        cmd.add_argument("--postprocess-params-path", type=Path, default=None)
        cmd.add_argument("--per-class-thresholds-path", type=Path, default=None)
        cmd.add_argument("--postprocess-file-level-top-k", type=int, default=2)
        cmd.add_argument("--postprocess-rank-power", type=float, default=0.4)
        cmd.add_argument("--postprocess-delta-alpha", type=float, default=0.15)
        cmd.add_argument("--disable-adaptive-delta", action="store_true")
        cmd.add_argument("--disable-threshold-sharpening", action="store_true")
        cmd.add_argument("--disable-taxon-temperature", action="store_true")
        cmd.add_argument(
            "--postprocess-taxon-temperatures",
            type=str,
            default="Aves:1.10,Insecta:0.95,Amphibia:0.95,Reptilia:0.95,Mammalia:0.95",
        )
        cmd.add_argument("--postprocess-file-top-k-grid", type=str, default="0,1,2,3")
        cmd.add_argument("--postprocess-rank-power-grid", type=str, default="0.0,0.4,0.5")
        cmd.add_argument("--postprocess-delta-alpha-grid", type=str, default="0.0,0.15,0.20")
        cmd.add_argument(
            "--postprocess-threshold-grid",
            type=str,
            default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
        )
        cmd.add_argument("--hard-negative-path", type=Path, default=None)
        cmd.add_argument("--hard-negative-threshold", type=float, default=0.35)
        cmd.add_argument("--hard-negative-topk", type=int, default=5)
        cmd.add_argument("--rare-class-threshold", type=int, default=30)
        cmd.add_argument("--waveform-lru-size", type=int, default=256)
        cmd.add_argument("--event-crop-jitter-sec", type=float, default=2.5)
        cmd.add_argument("--debug", action="store_true")
        cmd.add_argument("--num-workers", type=int, default=0)
        cmd.add_argument("--cache-path", type=Path, default=None)

    eda = subparsers.add_parser("eda", help="Run dataset summaries and save plots.")
    add_shared_args(eda)

    cache = subparsers.add_parser("cache", help="Precompute and save waveform HDF5 cache.")
    add_shared_args(cache)
    cache.add_argument("--cache-workers", type=int, default=4)

    train = subparsers.add_parser("train", help="Train selected folds.")
    add_shared_args(train)
    train.add_argument("--cache-workers", type=int, default=4)
    train.add_argument("--lr", type=float, default=7e-4)
    train.add_argument("--disable-rating-sampler", action="store_true")

    infer = subparsers.add_parser("infer", help="Run TTA inference and build submission.")
    add_shared_args(infer)
    infer.add_argument("--tta-crops", type=int, default=3)

    pseudo = subparsers.add_parser("pseudo", help="Generate teacher pseudo labels for train soundscapes.")
    add_shared_args(pseudo)
    pseudo.add_argument("--tta-crops", type=int, default=3)
    pseudo.add_argument("--pseudo-batch-size", type=int, default=16)
    pseudo.add_argument("--pseudo-max-files", type=int, default=None)

    merge_oof = subparsers.add_parser("merge-oof", help="Merge per-fold OOF files and rebuild metrics.")
    add_shared_args(merge_oof)

    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> CFG:
    cfg = CFG()
    cfg.output_dir = args.output_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_root is not None:
        root = args.data_root
        cfg.train_datadir = root / "train_audio"
        cfg.train_csv = root / "train.csv"
        cfg.test_sc_dir = root / "test_soundscapes"
        cfg.submission_csv = root / "sample_submission.csv"
        cfg.taxonomy_csv = root / "taxonomy.csv"
        cfg.sc_labels_csv = root / "train_soundscapes_labels.csv"

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.t_max = args.epochs
    cfg.debug = args.debug
    cfg.num_workers = args.num_workers
    cfg.selected_folds = parse_folds(args.folds)
    cfg.cv_strategy = args.cv_strategy
    cfg.group_col = args.group_col
    cfg.save_oof = not args.disable_oof
    cfg.rank_normalize_oof = not args.disable_rank_oof
    cfg.pseudo_path = args.pseudo_path
    cfg.pseudo_min_primary_prob = args.pseudo_min_primary_prob
    cfg.pseudo_label_prob = args.pseudo_label_prob
    cfg.pseudo_mask_prob = args.pseudo_mask_prob
    cfg.pseudo_max_labels = args.pseudo_max_labels
    cfg.pseudo_sampling_weight = args.pseudo_sampling_weight
    cfg.soundscape_full_windows = not args.disable_soundscape_full_windows
    cfg.soundscape_include_unlabeled_files = args.include_unlabeled_soundscape_files
    cfg.soundscape_window_sec = args.soundscape_window_sec
    cfg.soundscape_positive_weight = args.soundscape_positive_weight
    cfg.soundscape_background_weight = args.soundscape_background_weight
    cfg.pseudo_include_background = not args.disable_pseudo_background
    cfg.pseudo_background_max_prob = args.pseudo_background_max_prob
    cfg.pseudo_background_weight = args.pseudo_background_weight
    cfg.pseudo_include_hard_negatives = not args.disable_pseudo_hard_negatives
    cfg.pseudo_hard_negative_min_prob = args.pseudo_hard_negative_min_prob
    cfg.pseudo_hard_negative_weight = args.pseudo_hard_negative_weight
    cfg.pseudo_hard_negative_max_labels = args.pseudo_hard_negative_max_labels
    cfg.postprocess_params_path = args.postprocess_params_path
    cfg.per_class_thresholds_path = args.per_class_thresholds_path
    cfg.postprocess_file_level_top_k = args.postprocess_file_level_top_k
    cfg.postprocess_rank_power = args.postprocess_rank_power
    cfg.postprocess_delta_alpha = args.postprocess_delta_alpha
    cfg.postprocess_adaptive_delta = not args.disable_adaptive_delta
    cfg.postprocess_threshold_sharpening = not args.disable_threshold_sharpening
    cfg.postprocess_taxon_temperature = not args.disable_taxon_temperature
    cfg.postprocess_taxon_temperatures = parse_taxon_temperatures(args.postprocess_taxon_temperatures)
    cfg.postprocess_file_top_k_grid = parse_int_tuple(args.postprocess_file_top_k_grid)
    cfg.postprocess_rank_power_grid = parse_float_tuple(args.postprocess_rank_power_grid)
    cfg.postprocess_delta_alpha_grid = parse_float_tuple(args.postprocess_delta_alpha_grid)
    cfg.postprocess_threshold_grid = parse_float_tuple(args.postprocess_threshold_grid)
    cfg.hard_negative_path = args.hard_negative_path
    cfg.hard_negative_threshold = args.hard_negative_threshold
    cfg.hard_negative_topk = args.hard_negative_topk
    cfg.rare_class_threshold = args.rare_class_threshold
    cfg.waveform_lru_size = args.waveform_lru_size
    cfg.event_crop_jitter_sec = args.event_crop_jitter_sec
    cfg.cache_override = args.cache_path

    if hasattr(args, "lr"):
        cfg.lr = args.lr
    if hasattr(args, "disable_rating_sampler"):
        cfg.use_rating_weight = not args.disable_rating_sampler
    if hasattr(args, "tta_crops"):
        cfg.tta_crops = args.tta_crops
        cfg.pseudo_teacher_tta_crops = args.tta_crops
    if hasattr(args, "pseudo_batch_size"):
        cfg.pseudo_batch_size = args.pseudo_batch_size
    if hasattr(args, "pseudo_max_files"):
        cfg.pseudo_max_files = args.pseudo_max_files
    cfg.apply_debug_settings()
    return cfg
