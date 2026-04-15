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

    secondary_weight: float = 0.5
    use_rating_weight: bool = True
    tta_enabled: bool = True
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
    def cache_path(self) -> Path:
        if self.cache_override is not None:
            return self.cache_override
        height, _ = self.target_shape
        duration = f"{self.target_duration:g}s".replace(".", "p")
        return self.output_dir / f"logmel_cache_{duration}_{height}.npy"


def parse_folds(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


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
        cmd.add_argument("--debug", action="store_true")
        cmd.add_argument("--num-workers", type=int, default=0)
        cmd.add_argument("--cache-path", type=Path, default=None)

    eda = subparsers.add_parser("eda", help="Run dataset summaries and save plots.")
    add_shared_args(eda)

    cache = subparsers.add_parser("cache", help="Precompute and save LogMel cache.")
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
    infer.add_argument("--disable-tta", action="store_true")

    merge_oof = subparsers.add_parser("merge-oof", help="Merge per-fold OOF files and rebuild metrics.")
    add_shared_args(merge_oof)

    all_cmd = subparsers.add_parser(
        "all", help="Run EDA, cache build, training, and inference sequentially."
    )
    add_shared_args(all_cmd)
    all_cmd.add_argument("--cache-workers", type=int, default=4)
    all_cmd.add_argument("--lr", type=float, default=7e-4)
    all_cmd.add_argument("--tta-crops", type=int, default=3)
    all_cmd.add_argument("--disable-rating-sampler", action="store_true")
    all_cmd.add_argument("--disable-tta", action="store_true")

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
    cfg.cache_override = args.cache_path

    if hasattr(args, "lr"):
        cfg.lr = args.lr
    if hasattr(args, "disable_rating_sampler"):
        cfg.use_rating_weight = not args.disable_rating_sampler
    if hasattr(args, "tta_crops"):
        cfg.tta_crops = args.tta_crops
    if hasattr(args, "disable_tta"):
        cfg.tta_enabled = not args.disable_tta

    cfg.apply_debug_settings()
    return cfg
