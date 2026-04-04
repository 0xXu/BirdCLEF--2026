import logging
import os
import random
import warnings

import pandas as pd
import numpy as np
import torch

from birdclef.config import CFG
from birdclef.deps import plt, require_dependencies


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

PALETTE = [
    "#58a6ff",
    "#3fb950",
    "#f78166",
    "#d2a8ff",
    "#ffa657",
    "#79c0ff",
    "#56d364",
    "#ff7b72",
    "#bc8cff",
    "#ffb74d",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_plot_theme() -> None:
    require_dependencies(("matplotlib", plt))
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#c9d1d9",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "text.color": "#c9d1d9",
            "grid.color": "#21262d",
            "grid.linewidth": 0.6,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
        }
    )


def load_base_frames(cfg: CFG) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    sc_labels_df = pd.read_csv(cfg.sc_labels_csv)
    sample_sub_df = pd.read_csv(cfg.submission_csv)
    return train_df, taxonomy_df, sc_labels_df, sample_sub_df


def load_species_ids(cfg: CFG) -> list[str]:
    return pd.read_csv(cfg.taxonomy_csv)["primary_label"].astype(str).tolist()
