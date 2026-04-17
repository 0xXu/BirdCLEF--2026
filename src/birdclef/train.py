import gc
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from birdclef.config import CFG
from birdclef.dataset import BirdCLEFDataset, collate_fn
from birdclef.deps import require_dependencies, roc_auc_score, tqdm
from birdclef.model import BirdCLEFModel
from birdclef.deps import plt
from birdclef.pseudo import fold_safe_pseudo_subset, load_pseudo_training_frame
from birdclef.sampling import (
    attach_hard_negative_metadata,
    compute_positive_class_weights,
    make_weighted_sampler,
)
from birdclef.utils import load_species_ids
from birdclef.validation import (
    calculate_group_auc,
    make_folds,
    save_oof_report,
    summarize_fold_distribution,
)


def mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
    loss_weight: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y + (1 - lam) * y[idx]
    mixed_mask = mask * mask[idx]
    mixed_sample_weight = lam * sample_weight + (1 - lam) * sample_weight[idx]
    mixed_loss_weight = lam * loss_weight + (1 - lam) * loss_weight[idx]
    return mixed_x, mixed_y, mixed_mask, mixed_sample_weight, mixed_loss_weight


def cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
    loss_weight: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, height, width = x.shape
    cut_h = int(height * (1 - lam) ** 0.5)
    cut_w = int(width * (1 - lam) ** 0.5)
    cx, cy = random.randint(0, width), random.randint(0, height)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(width, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(height, cy + cut_h // 2)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_act = 1 - (x2 - x1) * (y2 - y1) / (width * height)
    mixed_y = lam_act * y + (1 - lam_act) * y[idx]
    mixed_mask = mask * mask[idx]
    mixed_sample_weight = lam_act * sample_weight + (1 - lam_act) * sample_weight[idx]
    mixed_loss_weight = lam_act * loss_weight + (1 - lam_act) * loss_weight[idx]
    return mixed, mixed_y, mixed_mask, mixed_sample_weight, mixed_loss_weight


class FocalBCELoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        focal_w: float = 0.7,
        bce_w: float = 0.3,
        smoothing: float = 0.05,
        positive_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.focal_w = focal_w
        self.bce_w = bce_w
        self.smoothing = smoothing
        if positive_class_weights is not None:
            self.register_buffer("positive_class_weights", positive_class_weights.float())
        else:
            self.positive_class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        loss_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        targets_s = targets * (1 - self.smoothing) + self.smoothing / logits.shape[-1]
        bce_raw = F.binary_cross_entropy_with_logits(logits, targets_s, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets_s + (1 - probs) * (1 - targets_s)
        focal_raw = (1 - p_t) ** self.gamma * bce_raw

        element_weight = torch.ones_like(bce_raw)
        if mask is not None:
            element_weight = element_weight * mask
        if self.positive_class_weights is not None:
            pos_weights = self.positive_class_weights.to(logits.device).view(1, -1)
            element_weight = element_weight * (1.0 + (pos_weights - 1.0) * targets_s)
        if loss_weight is not None:
            element_weight = element_weight * loss_weight
        if sample_weight is not None:
            element_weight = element_weight * sample_weight.view(-1, 1)

        denom = element_weight.sum().clamp_min(1.0)
        focal = (focal_raw * element_weight).sum() / denom
        bce = (bce_raw * element_weight).sum() / denom
        return self.focal_w * focal + self.bce_w * bce


def get_optimizer(model: nn.Module, cfg: CFG) -> optim.Optimizer:
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    return optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr * 0.1},
            {"params": head_params, "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )


def get_scheduler(optimizer: optim.Optimizer, cfg: CFG) -> lr_scheduler.CosineAnnealingLR:
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.t_max, eta_min=cfg.min_lr)


def calculate_auc(targets: np.ndarray, outputs: np.ndarray) -> float:
    require_dependencies(("scikit-learn", roc_auc_score))
    targets = (targets > 0.5).astype(np.float32)
    probs = 1 / (1 + np.exp(-outputs))
    aucs = []
    for idx in range(targets.shape[1]):
        positives = targets[:, idx].sum()
        if 0 < positives < targets.shape[0]:
            aucs.append(roc_auc_score(targets[:, idx], probs[:, idx]))
    return float(np.mean(aucs)) if aucs else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    cfg: CFG,
) -> tuple[float, float]:
    model.train()
    losses = []
    auc_outputs = []
    auc_targets = []
    collect_from = max(0, len(loader) - 20)

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    for step, batch in pbar:
        inputs = batch["melspec"].to(cfg.device)
        targets = batch["target"].to(cfg.device)
        masks = batch["target_mask"].to(cfg.device)
        sample_weights = batch["sample_weight"].to(cfg.device)
        loss_weights = batch["loss_weight"].to(cfg.device)

        aug_roll = random.random()
        if aug_roll < cfg.mixup_prob:
            inputs, targets, masks, sample_weights, loss_weights = mixup(
                inputs, targets, masks, sample_weights, loss_weights, cfg.mixup_alpha
            )
        elif aug_roll < cfg.mixup_prob + cfg.cutmix_prob:
            inputs, targets, masks, sample_weights, loss_weights = cutmix(
                inputs, targets, masks, sample_weights, loss_weights
            )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets, masks, sample_weights, loss_weights)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step >= collect_from:
            auc_outputs.append(outputs.detach().cpu().numpy())
            auc_targets.append(targets.detach().cpu().numpy())

        pbar.set_postfix(
            loss=f"{np.mean(losses[-10:]):.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )

    train_auc = (
        calculate_auc(np.concatenate(auc_targets), np.concatenate(auc_outputs))
        if auc_outputs
        else 0.0
    )
    return float(np.mean(losses)), train_auc


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: CFG,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    auc_outputs = []
    auc_targets = []
    losses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            inputs = batch["melspec"].to(cfg.device)
            targets = batch["target"].to(cfg.device)
            masks = batch["target_mask"].to(cfg.device)
            sample_weights = batch["sample_weight"].to(cfg.device)
            loss_weights = batch["loss_weight"].to(cfg.device)
            outputs = model(inputs)
            losses.append(criterion(outputs, targets, masks, sample_weights, loss_weights).item())
            auc_outputs.append(outputs.cpu().numpy())
            auc_targets.append(targets.cpu().numpy())

    outputs_arr = np.concatenate(auc_outputs) if auc_outputs else np.zeros((0, 0), dtype=np.float32)
    targets_arr = np.concatenate(auc_targets) if auc_targets else np.zeros((0, 0), dtype=np.float32)
    val_auc = calculate_auc(targets_arr, outputs_arr) if len(outputs_arr) else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, val_auc, outputs_arr, targets_arr


def plot_training_history(history: dict[int, dict[str, list[float]]], cfg: CFG) -> None:
    require_dependencies(("matplotlib", plt))
    if not history:
        return

    fig, axes = plt.subplots(1, len(history), figsize=(6 * len(history), 4))
    fig.patch.set_facecolor("#0d1117")
    if len(history) == 1:
        axes = [axes]

    for ax, (fold_id, values) in zip(axes, history.items()):
        epochs = range(1, len(values["val_auc"]) + 1)
        ax.plot(epochs, values["train_auc"], "o-", color="#58a6ff", linewidth=2, label="Train AUC")
        ax.plot(epochs, values["val_auc"], "s-", color="#3fb950", linewidth=2, label="Val AUC")
        best_epoch = int(np.argmax(values["val_auc"])) + 1
        ax.axvline(best_epoch, color="#f78166", linestyle="--", alpha=0.6)
        ax.set_title(f"Fold {fold_id}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(facecolor="#21262d")

    out_path = cfg.output_dir / "training_history.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved {out_path}")


def run_training(df, cfg: CFG) -> dict[int, dict[str, list[float]]]:
    require_dependencies(("scikit-learn", roc_auc_score), ("matplotlib", plt))
    species_ids = load_species_ids(cfg)
    folds = make_folds(df, cfg)
    pseudo_df = load_pseudo_training_frame(cfg)
    fold_summary = summarize_fold_distribution(df, folds, cfg)
    fold_summary_path = cfg.output_dir / "fold_distribution.csv"
    fold_summary.to_csv(fold_summary_path, index=False)
    print(f"Saved fold distribution to {fold_summary_path}")
    print(fold_summary.to_string(index=False))

    oof_logits = np.zeros((len(df), len(species_ids)), dtype=np.float32)
    oof_targets = np.zeros((len(df), len(species_ids)), dtype=np.float32)
    oof_fold_ids = np.full(len(df), -1, dtype=np.int32)

    history = {}
    best_scores: list[tuple[int, float]] = []

    for split in folds:
        fold = split.fold
        if fold not in cfg.selected_folds:
            continue

        print(f"\n{'=' * 40} Fold {fold} {'=' * 40}")
        train_idx = split.train_idx
        val_idx = split.val_idx
        train_df_f = df.iloc[train_idx].reset_index(drop=True)
        val_df_f = df.iloc[val_idx].reset_index(drop=True)
        fold_pseudo_df = fold_safe_pseudo_subset(pseudo_df, train_df_f, val_df_f)
        if fold_pseudo_df is not None:
            train_df_f = pd.concat([train_df_f, fold_pseudo_df], ignore_index=True, sort=False)
            print(
                f"Added pseudo rows to fold {fold}: pseudo={len(fold_pseudo_df)} "
                f"train_total={len(train_df_f)}"
            )
        train_df_f = attach_hard_negative_metadata(train_df_f, cfg)
        print(
            f"Train={len(train_df_f)} Val={len(val_df_f)} "
            f"strategy={split.strategy} group={split.group_col}"
        )

        positive_class_weights = compute_positive_class_weights(train_df_f, species_ids, cfg).to(cfg.device)
        criterion = FocalBCELoss(
            gamma=cfg.focal_gamma,
            focal_w=cfg.focal_weight,
            bce_w=cfg.bce_weight,
            smoothing=cfg.label_smoothing,
            positive_class_weights=positive_class_weights,
        )

        train_ds = BirdCLEFDataset(train_df_f, cfg, mode="train")
        val_ds = BirdCLEFDataset(val_df_f, cfg, mode="valid")

        sampler = None
        shuffle_train = True
        if cfg.use_rating_weight:
            sampler = make_weighted_sampler(train_df_f.fillna({"rating": 0.0}), cfg, species_ids)
            shuffle_train = False

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle_train,
            sampler=sampler,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
        )

        model = BirdCLEFModel(cfg).to(cfg.device)
        if cfg.use_compile:
            try:
                model = torch.compile(model, backend="inductor", mode="reduce-overhead")
                print("torch.compile enabled")
            except Exception as exc:
                print(f"torch.compile failed: {exc}")

        optimizer = get_optimizer(model, cfg)
        scheduler = get_scheduler(optimizer, cfg)
        best_auc = 0.0
        fold_history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
        best_val_logits = None
        best_val_targets = None

        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch + 1}/{cfg.epochs}")
            train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, cfg)
            val_loss, val_auc, val_logits, val_targets = validate(model, val_loader, criterion, cfg)
            scheduler.step()

            fold_history["train_loss"].append(train_loss)
            fold_history["val_loss"].append(val_loss)
            fold_history["train_auc"].append(train_auc)
            fold_history["val_auc"].append(val_auc)

            print(f"Train loss={train_loss:.4f} auc={train_auc:.4f}")
            print(f"Val   loss={val_loss:.4f} auc={val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_val_logits = val_logits
                best_val_targets = val_targets
                ckpt_path = cfg.output_dir / f"model_fold{fold}.pth"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_auc": val_auc,
                        "cfg": cfg.__dict__,
                    },
                    ckpt_path,
                )
                print(f"New best checkpoint saved to {ckpt_path}")

        history[fold] = fold_history
        best_scores.append((fold, best_auc))
        print(f"Fold {fold} best AUC: {best_auc:.4f}")
        if best_val_logits is not None and best_val_targets is not None:
            oof_logits[val_idx] = best_val_logits
            oof_targets[val_idx] = best_val_targets
            oof_fold_ids[val_idx] = fold
            source_auc = calculate_group_auc(
                val_df_f,
                best_val_targets,
                best_val_logits,
                group_col="source",
                from_logits=True,
            )
            if not source_auc.empty:
                print("Best validation AUC by source:")
                print(source_auc.to_string(index=False))

        del model, optimizer, scheduler, train_loader, val_loader
        gc.collect()

    if best_scores:
        pd.DataFrame(
            [{"fold": fold, "best_auc": score} for fold, score in best_scores]
        ).to_csv(cfg.output_dir / "fold_metrics.csv", index=False)
        for fold, score in best_scores:
            print(f"Fold {fold}: {score:.4f}")
        print(f"Mean AUC: {np.mean([score for _, score in best_scores]):.4f}")

    if cfg.save_oof:
        per_class = save_oof_report(df, oof_logits, oof_targets, oof_fold_ids, cfg)
        if per_class is not None:
            valid_auc = per_class["auc"].dropna()
            if len(valid_auc):
                print("OOF macro AUC:", f"{valid_auc.mean():.4f}")

    plot_training_history(history, cfg)
    return history
