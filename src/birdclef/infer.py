import numpy as np
import pandas as pd
import torch

from birdclef.audio import audio2logmel
from birdclef.config import CFG
from birdclef.deps import cv2, librosa, require_dependencies, tqdm
from birdclef.model import BirdCLEFModel
from birdclef.utils import load_species_ids


def audio_to_tensor(seg: np.ndarray, cfg: CFG) -> torch.Tensor:
    require_dependencies(("cv2", cv2))
    if len(seg) < cfg.target_samples:
        seg = np.pad(seg, (0, cfg.target_samples - len(seg)))
    else:
        seg = seg[: cfg.target_samples]
    logmel = audio2logmel(seg, cfg)
    if logmel.shape != cfg.target_shape:
        logmel = cv2.resize(logmel, cfg.target_shape, interpolation=cv2.INTER_LINEAR)
    return torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def discover_checkpoints(cfg: CFG) -> list:
    return sorted(cfg.output_dir.glob("model_fold*.pth"))


def predict_soundscapes_tta(cfg: CFG) -> pd.DataFrame:
    require_dependencies(("librosa", librosa))
    species_ids = load_species_ids(cfg)
    ckpt_files = discover_checkpoints(cfg)
    if not ckpt_files:
        raise FileNotFoundError("No model_fold*.pth checkpoints found. Run training first.")

    models = []
    for ckpt_path in ckpt_files:
        model = BirdCLEFModel(cfg).to(cfg.device)
        ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        print(f"Loaded {ckpt_path.name} val_auc={ckpt.get('val_auc', 0.0):.4f}")

    test_oggs = sorted(cfg.test_sc_dir.glob("*.ogg"))
    if not test_oggs:
        fallback_dir = cfg.train_datadir.parent / "train_soundscapes"
        test_oggs = sorted(fallback_dir.glob("*.ogg"))[:8]

    all_row_ids = []
    all_preds = []
    n_crops = cfg.tta_crops if cfg.tta_enabled else 1
    print(f"Running inference on {len(test_oggs)} soundscape(s) with {n_crops} crop(s)")

    for ogg_path in tqdm(test_oggs, desc="TTA Inference"):
        ss_id = ogg_path.stem
        row_ids = [f"{ss_id}_{t}" for t in range(5, 65, 5)]
        try:
            audio, _ = librosa.load(str(ogg_path), sr=cfg.fs)
        except Exception as exc:
            print(f"Skipped {ogg_path.name}: {exc}")
            all_row_ids.extend(row_ids)
            all_preds.append(np.zeros((12, len(species_ids)), dtype=np.float32))
            continue

        seg_preds = []
        for t in range(0, 60, 5):
            audio_len = len(audio)
            half = cfg.target_samples // 2
            target_center = int((t + 2.5) * cfg.fs)
            base_offset = target_center - half
            max_offset = max(0, audio_len - cfg.target_samples)

            if n_crops == 1:
                offsets = [max(0, min(base_offset, max_offset))]
            else:
                shift = cfg.target_samples // 4
                offsets = [
                    max(0, min(base_offset - shift, max_offset)),
                    max(0, min(base_offset, max_offset)),
                    max(0, min(base_offset + shift, max_offset)),
                ][:n_crops]

            crop_probs = []
            for offset in offsets:
                seg = audio[int(offset) : int(offset) + cfg.target_samples]
                x = audio_to_tensor(seg, cfg).to(cfg.device)
                fold_probs = []
                with torch.no_grad():
                    for model in models:
                        fold_probs.append(torch.sigmoid(model(x)).cpu().numpy()[0])
                crop_probs.append(np.mean(fold_probs, axis=0))

            seg_preds.append(np.mean(crop_probs, axis=0))

        all_row_ids.extend(row_ids)
        all_preds.append(np.array(seg_preds))

    preds_arr = np.concatenate(all_preds)
    sample_sub = pd.read_csv(cfg.submission_csv)
    sub_df = pd.DataFrame(preds_arr, columns=species_ids)
    sub_df.insert(0, "row_id", all_row_ids)
    for col in set(sample_sub.columns) - set(sub_df.columns):
        sub_df[col] = 0.0
    sub_df = sub_df[sample_sub.columns].fillna(0.0)

    out_path = cfg.output_dir / "submission.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} | rows={len(sub_df)} cols={len(sub_df.columns)}")
    return sub_df
