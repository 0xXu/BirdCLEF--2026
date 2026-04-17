from birdclef.config import build_cfg, parse_args
from birdclef.utils import set_plot_theme, set_seed
from birdclef.deps import plt


def run_cache_step(cfg, cache_workers: int):
    from birdclef.audio import load_or_build_cache
    from birdclef.dataset import build_train_df
    from birdclef.pseudo import load_pseudo_training_frame

    df = build_train_df(cfg)
    cache_df = df
    pseudo_df = load_pseudo_training_frame(cfg)
    if pseudo_df is not None:
        import pandas as pd

        cache_df = pd.concat([df, pseudo_df], ignore_index=True, sort=False)
    load_or_build_cache(cache_df, cfg, n_workers=cache_workers)
    return df


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)
    set_seed(cfg.seed)
    if plt is not None:
        set_plot_theme()

    print(
        f"device={cfg.device} model={cfg.model_name} batch={cfg.batch_size} "
        f"epochs={cfg.epochs} folds={cfg.selected_folds} cv={cfg.cv_strategy}"
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

        df = run_cache_step(cfg, args.cache_workers)
        run_training(df, cfg)
        return
    if args.command == "infer":
        from birdclef.infer import predict_soundscapes_tta

        predict_soundscapes_tta(cfg)
        return
    if args.command == "pseudo":
        from birdclef.pseudo import generate_pseudo_labels

        generate_pseudo_labels(cfg)
        return
    if args.command == "merge-oof":
        from birdclef.validation import merge_oof_reports, save_oof_metrics

        merged = merge_oof_reports(cfg)
        if merged is None:
            raise FileNotFoundError(f"No oof_predictions_fold*.csv files found in {cfg.output_dir}")
        per_class = save_oof_metrics(merged, cfg)
        valid_auc = per_class["auc"].dropna()
        if len(valid_auc):
            print(f"Merged OOF rows={len(merged)} macro_auc={valid_auc.mean():.4f}")
        else:
            print(f"Merged OOF rows={len(merged)} macro_auc=nan")
        return
    raise ValueError(f"Unknown command: {args.command}")
