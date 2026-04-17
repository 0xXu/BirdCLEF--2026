import math
import random
import importlib

import numpy as np
import pandas as pd

from birdclef.config import CFG
from birdclef.deps import gridspec, librosa, plt, require_dependencies
from birdclef.utils import PALETTE, load_base_frames


def run_eda(cfg: CFG) -> None:
    require_dependencies(
        ("matplotlib", plt),
        ("matplotlib.gridspec", gridspec),
        ("librosa", librosa),
    )
    train_df, taxonomy_df, sc_labels_df, sample_sub_df = load_base_frames(cfg)
    species_counts = train_df["primary_label"].value_counts()

    print("=== Basic Statistics ===")
    print(f"train.csv records      : {len(train_df):,}")
    print(f"taxonomy.csv species   : {len(taxonomy_df):,}")
    print(f"soundscape label rows  : {len(sc_labels_df):,}")
    print(f"submission species cols: {len(sample_sub_df.columns) - 1:,}")
    print(f"unique train species   : {species_counts.shape[0]:,}")
    print(f"mean recordings/species: {species_counts.mean():.1f}")
    print(f"median recordings/spec.: {species_counts.median():.0f}")
    if "rating" in train_df.columns:
        print("\n=== Rating Distribution ===")
        print(train_df["rating"].value_counts().sort_index().to_string())

    save_species_distribution_plot(train_df, species_counts, cfg)
    save_species_world_map(train_df, species_counts, cfg)
    save_rating_taxonomy_plot(train_df, taxonomy_df, sc_labels_df, species_counts, cfg)
    save_audio_examples(train_df, species_counts, cfg)


def save_species_world_map(train_df, species_counts, cfg: CFG, top_n: int = 20, max_points: int = 8000) -> None:
    lat_col = next((col for col in ["latitude", "lat"] if col in train_df.columns), None)
    lon_col = next((col for col in ["longitude", "lon", "lng"] if col in train_df.columns), None)

    if lat_col is None or lon_col is None:
        print("Skipping world map: no latitude/longitude columns found.")
        return

    geo_df = train_df[[lat_col, lon_col, "primary_label"]].dropna().copy()
    geo_df = geo_df.rename(columns={lat_col: "lat", lon_col: "lon"})
    geo_df = geo_df[geo_df["lat"].between(-90, 90) & geo_df["lon"].between(-180, 180)]

    if len(geo_df) > max_points:
        geo_df = geo_df.sample(max_points, random_state=cfg.seed)

    top_species = species_counts.head(top_n).index.tolist()
    color_map = {species: PALETTE[i % len(PALETTE)] for i, species in enumerate(top_species)}
    point_colors = [color_map.get(species, "#30363d") for species in geo_df["primary_label"]]

    fig = plt.figure(figsize=(22, 11))
    fig.patch.set_facecolor("#0d1117")
    map_drawn = False

    if importlib.util.find_spec("geopandas") is not None:
        import geopandas as gpd

        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_facecolor("#0d2137")
            world.plot(ax=ax, color="#1c2b3a", edgecolor="#3d5a73", linewidth=0.4)
            ax.scatter(
                geo_df["lon"].values,
                geo_df["lat"].values,
                c=point_colors,
                s=8,
                alpha=0.6,
                linewidths=0,
                zorder=5,
            )
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(color="#21262d", linewidth=0.4, linestyle="--", alpha=0.5)
            map_drawn = True
        except Exception as exc:
            print(f"Geopandas world plot failed ({exc}); using scatter plot.")

    if not map_drawn:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("#0d2137")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.axhline(0, color="#21262d", linewidth=0.6, linestyle="--", alpha=0.6)
        ax.axhline(23.5, color="#21262d", linewidth=0.4, linestyle=":", alpha=0.4)
        ax.axhline(-23.5, color="#21262d", linewidth=0.4, linestyle=":", alpha=0.4)
        ax.axhline(66.5, color="#21262d", linewidth=0.4, linestyle=":", alpha=0.4)
        ax.axhline(-66.5, color="#21262d", linewidth=0.4, linestyle=":", alpha=0.4)
        ax.grid(color="#21262d", linewidth=0.4, linestyle="--", alpha=0.5)
        ax.scatter(
            geo_df["lon"].values,
            geo_df["lat"].values,
            c=point_colors,
            s=8,
            alpha=0.6,
            linewidths=0,
            zorder=5,
        )

    ax.set_title(
        f"BirdCLEF 2026 Recording Locations ({len(geo_df):,} recordings, top-{top_n} highlighted)",
        fontsize=14,
        pad=12,
    )
    out_path = cfg.output_dir / "eda_world_map.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_species_distribution_plot(train_df, species_counts, cfg: CFG) -> None:
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    top40 = species_counts.head(40)
    colors_top = [PALETTE[i % len(PALETTE)] for i in range(len(top40))]
    bars = ax1.bar(range(len(top40)), top40.values, color=colors_top, width=0.75, alpha=0.9)
    ax1.set_xticks(range(len(top40)))
    ax1.set_xticklabels(top40.index, rotation=55, ha="right", fontsize=8)
    ax1.set_title("Top 40 Species by Recording Count")
    ax1.set_ylabel("Recording Count")
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#8b949e",
        )

    ax2 = fig.add_subplot(gs[1, 0])
    sorted_counts = species_counts.values
    ax2.fill_between(range(len(sorted_counts)), sorted_counts, alpha=0.6, color="#58a6ff")
    ax2.plot(sorted_counts, color="#79c0ff", linewidth=1.2)
    ax2.axhline(sorted_counts.mean(), color="#f78166", linestyle="--", linewidth=1.2)
    ax2.axhline(np.median(sorted_counts), color="#3fb950", linestyle="--", linewidth=1.2)
    ax2.set_title("Long-Tail Distribution")
    ax2.set_xlabel("Species Rank (descending)")
    ax2.set_ylabel("Recording Count")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(species_counts.values, bins=40, color="#d2a8ff", alpha=0.85, edgecolor="#30363d")
    ax3.set_title("Recording Count Histogram")
    ax3.set_xlabel("Recordings per Species")
    ax3.set_ylabel("Number of Species")
    ax3.set_yscale("log")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 0])
    bins = [0, 5, 10, 20, 50, 100, 200, 500, 10_000]
    labels = ["1-5", "6-10", "11-20", "21-50", "51-100", "101-200", "201-500", "500+"]
    sp_bin = pd.cut(species_counts.values, bins=bins, labels=labels)
    bin_cnt = pd.Series(sp_bin).value_counts().sort_index()
    ax4.pie(
        bin_cnt.values,
        labels=bin_cnt.index,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        colors=PALETTE[: len(bin_cnt)],
        startangle=140,
        pctdistance=0.75,
        textprops={"fontsize": 8},
    )
    ax4.set_title("Species Distribution by Count Bin")

    ax5 = fig.add_subplot(gs[2, 1])
    if "filename" in train_df.columns:
        source_type = train_df["filename"].apply(
            lambda x: "XenoCanto"
            if str(x).split("/")[-1].startswith("XC")
            else ("iNaturalist" if str(x).split("/")[-1].startswith("iNat") else "Other")
        )
        src_counts = source_type.value_counts()
        ax5.bar(
            src_counts.index,
            src_counts.values,
            color=[PALETTE[0], PALETTE[2], PALETTE[4]][: len(src_counts)],
            alpha=0.9,
            edgecolor="#30363d",
            width=0.5,
        )
        ax5.set_title("Recording Source Distribution")
        ax5.set_ylabel("Recording Count")
        ax5.grid(axis="y", alpha=0.3)

    plt.suptitle("BirdCLEF 2026 EDA: Species Distribution", fontsize=16, y=1.01)
    out_path = cfg.output_dir / "eda_species_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_rating_taxonomy_plot(train_df, taxonomy_df, sc_labels_df, species_counts, cfg: CFG) -> None:
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    if "rating" in train_df.columns:
        rating_counts = train_df["rating"].value_counts().sort_index()
        ax1.bar(
            rating_counts.index.astype(str),
            rating_counts.values,
            color=PALETTE[: len(rating_counts)],
            alpha=0.9,
            edgecolor="#30363d",
            width=0.6,
        )
        ax1.set_title("Rating Distribution")
        ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    if "rating" in train_df.columns:
        temp_df = train_df.copy()
        temp_df["sp_count"] = temp_df["primary_label"].map(species_counts)
        groups = [
            temp_df.loc[temp_df["rating"] == r, "sp_count"].values
            for r in sorted(temp_df["rating"].dropna().unique())
        ]
        if groups:
            bp = ax2.boxplot(
                groups,
                patch_artist=True,
                labels=[str(r) for r in sorted(temp_df["rating"].dropna().unique())],
                medianprops={"color": "#f78166", "linewidth": 2},
            )
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax2.set_title("Rating vs Species Count")
            ax2.set_yscale("log")
            ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    group_col = "order" if "order" in taxonomy_df.columns else "family"
    if group_col in taxonomy_df.columns:
        top_counts = taxonomy_df[group_col].value_counts().head(15)
        ax3.barh(
            top_counts.index[::-1],
            top_counts.values[::-1],
            color=[PALETTE[i % len(PALETTE)] for i in range(len(top_counts))],
            alpha=0.9,
            edgecolor="#30363d",
        )
        ax3.set_title(f"Top 15 {group_col.title()} by Species Count")
        ax3.grid(axis="x", alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0:2])
    label_col = "birds" if "birds" in sc_labels_df.columns else "primary_label"
    sc_species = (
        sc_labels_df[label_col]
        .astype(str)
        .str.split(r"[;, ]+", expand=False)
        .explode()
        .str.strip()
    )
    sc_species = sc_species[sc_species.ne("")]
    sc_top = sc_species.value_counts().head(30)
    ax4.bar(
        range(len(sc_top)),
        sc_top.values,
        color=[PALETTE[i % len(PALETTE)] for i in range(len(sc_top))],
        alpha=0.9,
        edgecolor="#30363d",
        width=0.75,
    )
    ax4.set_xticks(range(len(sc_top)))
    ax4.set_xticklabels(sc_top.index, rotation=50, ha="right", fontsize=8)
    ax4.set_title("Top 30 Soundscape Labels")
    ax4.grid(axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    train_species = set(train_df["primary_label"].astype(str).unique())
    sc_species_set = set(sc_species.astype(str).unique()) - {""}
    sizes = [
        len(train_species & sc_species_set),
        len(train_species - sc_species_set),
        len(sc_species_set - train_species),
    ]
    ax5.pie(
        sizes,
        labels=["Both", "Train only", "Soundscape only"],
        colors=[PALETTE[1], PALETTE[0], PALETTE[2]],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax5.set_title("Train / Soundscape Species Overlap")

    plt.suptitle("BirdCLEF 2026 EDA: Rating, Taxonomy, Soundscape", fontsize=15, y=1.01)
    out_path = cfg.output_dir / "eda_rating_taxonomy.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_audio_examples(train_df, species_counts, cfg: CFG) -> None:
    sample_labels = species_counts.head(8).index.tolist()
    random.seed(cfg.seed)
    random.shuffle(sample_labels)
    sample_labels = sample_labels[:4]

    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor("#0d1117")

    shown = 0
    for label in sample_labels:
        rows = train_df[train_df["primary_label"] == label]
        if rows.empty:
            continue
        row = rows.iloc[0]
        path = cfg.train_datadir / row["filename"]
        try:
            y, _ = librosa.load(str(path), sr=cfg.fs, duration=cfg.target_duration)
        except Exception:
            continue

        shown += 1
        if len(y) < cfg.target_samples:
            y = np.tile(y, math.ceil(cfg.target_samples / max(1, len(y))))[: cfg.target_samples]
        else:
            y = y[: cfg.target_samples]

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=cfg.fs,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            power=cfg.mel_power,
            norm=cfg.mel_norm,
            htk=cfg.mel_scale == "htk",
        )
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=cfg.mel_top_db)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        ax_wave = fig.add_subplot(2, 4, shown)
        times = np.linspace(0, cfg.target_duration, len(y))
        ax_wave.plot(times, y, color=PALETTE[(shown - 1) % len(PALETTE)], linewidth=0.5)
        ax_wave.set_title(f"{label}\n{path.name}", fontsize=9)

        ax_mel = fig.add_subplot(2, 4, 4 + shown)
        ax_mel.imshow(
            mel_norm,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, cfg.target_duration, cfg.fmin, cfg.fmax],
        )
        ax_mel.set_title("LogMel dB (normalized)", fontsize=9)

    plt.suptitle("BirdCLEF 2026 EDA: Audio Examples", fontsize=14, y=1.01)
    out_path = cfg.output_dir / "eda_logmel.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved {out_path}")
