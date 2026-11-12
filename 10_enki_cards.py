"""Enki skorlarindan temas sayfasi: en yuksek skorlu hucreleri goster."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def main(n=24, cols=6):
    cells = np.load(DATA / "cells_rgb.npy", mmap_mode="r")
    scores = np.load(DATA / "enki_scores.npy")
    order = np.argsort(-scores)[:n]

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.3), dpi=130)
    for rank, (ax, k) in enumerate(zip(axes.ravel(), order), 1):
        ax.imshow(cells[k], interpolation="bilinear")
        ax.set_title(f"#{rank}  {scores[k]:.3f}", fontsize=8)
        ax.axis("off")
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    fig.suptitle("Enki — tell benzeri en yuksek 24 hucre (350 m kare)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ROOT / "enki_top24.png")
    print("-> enki_top24.png")

    # ayrica en dusuk skorlu (kontrol) 6 hucre
    low = order[::-1][:6]
    fig, axes = plt.subplots(1, 6, figsize=(12.6, 2.4), dpi=130)
    for ax, k in zip(axes, low):
        ax.imshow(cells[k], interpolation="bilinear")
        ax.set_title(f"{scores[k]:.3f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Kontrol — en dusuk skorlu 6 hucre", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(ROOT / "enki_bottom6.png")
    print("-> enki_bottom6.png")


if __name__ == "__main__":
    main()
