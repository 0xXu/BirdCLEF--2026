from birdclef.config import build_cfg, parse_args
from birdclef.utils import set_plot_theme, set_seed
from birdclef.deps import plt


def run_cache_step(cfg, cache_workers: int):
    from birdclef.audio import load_or_build_cache
    from birdclef.dataset import build_train_df

    df = build_train_df(cfg)
    cache = load_or_build_cache(df, cfg, n_workers=cache_workers)
    return df, cache


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    set_seed(cfg.seed)
    if plt is not None:
        set_plot_theme()

    print(
        f"device={cfg.device} model={cfg.model_name} batch={cfg.batch_size} "
        f"epochs={cfg.epochs} folds={cfg.selected_folds}"
    )

    if args.command == "eda":
        from birdclef.eda import run_eda

        run_eda(cfg)
        return
    if args.command == "cache":
        run_cache_step(cfg, args.cache_workers)
        return
    if args.command == "train":
        from birdclef.train import run_training

        df, cache = run_cache_step(cfg, args.cache_workers)
        run_training(df, cache, cfg)
        return
    if args.command == "infer":
        from birdclef.infer import predict_soundscapes_tta

        predict_soundscapes_tta(cfg)
        return
    if args.command == "all":
        from birdclef.eda import run_eda
        from birdclef.infer import predict_soundscapes_tta
        from birdclef.train import run_training

        run_eda(cfg)
        df, cache = run_cache_step(cfg, args.cache_workers)
        run_training(df, cache, cfg)
        predict_soundscapes_tta(cfg)
        return

    raise ValueError(f"Unknown command: {args.command}")
