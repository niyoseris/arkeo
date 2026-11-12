"""Aday hucreler icin gorsel kart destesi + genel bakis haritasi."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from rasterio.transform import from_origin

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CARDS = ROOT / "cards"
CARDS.mkdir(exist_ok=True)

X0, Y0, NX, NY, RES = 544650, 3911216, 2000, 2000, 10
CELLPX = 100  # 1 km
BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
TOPN = 60


def load_comp(season):
    with rasterio.open(DATA / f"comp_{season}.tif") as s:
        return {b: s.read(i + 1) for i, b in enumerate(BANDS)}


def truecolor(d, lo=2, hi=98):
    rgb = np.dstack([d["red"], d["green"], d["blue"]])
    out = np.zeros_like(rgb, dtype="float32")
    for i in range(3):
        ch = rgb[..., i]
        v = ch[np.isfinite(ch)]
        if v.size == 0:
            continue
        a, b = np.percentile(v, [lo, hi])
        out[..., i] = np.clip((ch - a) / max(b - a, 1e-6), 0, 1)
    return out


def main():
    ch = np.load(DATA / "channels.npz")
    cells = json.loads((DATA / "cells.json").read_text())
    score = np.load(DATA / "score.npy")
    dry, grn = load_comp("dry"), load_comp("green")
    tc_dry, tc_grn = truecolor(dry), truecolor(grn)

    div = LinearSegmentedColormap.from_list("div", ["#2c5f8a", "#f2f2f2", "#a8342a"])
    seq = LinearSegmentedColormap.from_list("seq", ["#f7f7f7", "#f0c060", "#8a2b1e"])

    made = []
    for c in cells[:TOPN]:
        cy, cx = c["cell_y"], c["cell_x"]
        ys, xs = cy * CELLPX, cx * CELLPX
        sl = (slice(ys, ys + CELLPX), slice(xs, xs + CELLPX))

        panels = [
            ("Gercek renk - kurak (May-Tem)", tc_dry[sl], None, None),
            ("Gercek renk - yesil (Sub-Nis)", tc_grn[sl], None, None),
            ("NDVI genligi (yesil - kurak)", ch["amp"][sl], seq, (-0.15, 0.45)),
            ("BSI kurak (toprak acikligi)", ch["bsi_dry"][sl], div, (-0.25, 0.45)),
            ("NDVI yerel sapmasi z", ch["z_amp"][sl], div, (-3, 3)),
            ("Yapisal tutarlilik (coherence)", ch["coh"][sl], seq, (0.1, 0.9)),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8.4), dpi=110)
        for ax, (title, img, cmap, vlim) in zip(axes.ravel(), panels):
            if cmap is None:
                ax.imshow(img, interpolation="nearest")
            else:
                im = ax.imshow(img, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                               interpolation="nearest")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(
            f"#{c['rank']:02d}   E={c['utm_e']}  N={c['utm_n']} (EPSG:32636)   "
            f"skor={c['score']:.3f}   z_amp={c['z_amp']:.1f}   "
            f"coherence={c['coherence']:.2f}   rakim={c['elev']:.0f} m   "
            f"egim={c['slope']:.1f}deg",
            fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fn = CARDS / f"card_{c['rank']:03d}.png"
        fig.savefig(fn); plt.close(fig)
        made.append((c, fn.name))
        print(f"kart {c['rank']:3d} yazildi")

    # --- genel bakis ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), dpi=120)
    hs = np.gradient(ch["dem"])
    shade = np.clip(0.5 + (hs[0] - hs[1]) * 0.02, 0, 1)
    axes[0].imshow(shade, cmap="gray", interpolation="nearest")
    axes[0].set_title("Kabartma (DEM) + ilk 60 aday", fontsize=10)
    for i, (c, _) in enumerate(made):
        axes[0].plot(c["cell_x"] * CELLPX + CELLPX / 2,
                     c["cell_y"] * CELLPX + CELLPX / 2,
                     "o", ms=5, mfc="none", mec="#c0392b", mew=1.2)
        if i < 12:
            axes[0].text(c["cell_x"] * CELLPX + CELLPX / 2,
                         c["cell_y"] * CELLPX + CELLPX / 2 - 24,
                         str(c["rank"]), color="#c0392b", fontsize=7,
                         ha="center")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    im = axes[1].imshow(score, cmap=seq, interpolation="nearest")
    plt.colorbar(im, ax=axes[1], fraction=0.04)
    axes[1].set_title("Bilesik anomali skoru (1 km hucre)", fontsize=10)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(ROOT / "overview.png")
    plt.close(fig)
    print("-> overview.png")

    # --- HTML ---
    rows = "\n".join(
        f'<figure><img src="cards/{fn}" loading="lazy">'
        f'<figcaption>#{c["rank"]} &middot; E {c["utm_e"]} / N {c["utm_n"]} '
        f'&middot; skor {c["score"]:.2f}</figcaption></figure>'
        for c, fn in made)
    (ROOT / "cards.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>Serdarli aday kartlari</title>
<style>
body{{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:24px;background:#faf9f6;color:#1c1c1c}}
h1{{font-size:20px;margin:0 0 4px}} p{{color:#555;margin:0 0 20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}}
figure{{margin:0;background:#fff;border:1px solid #e3e0d8;border-radius:8px;overflow:hidden}}
img{{width:100%;display:block}} figcaption{{padding:7px 10px;font-size:12px;color:#444}}
</style>
<h1>Serdarli &mdash; {len(made)} aday</h1>
<p>20 &times; 20 km, 1 km hucreler, 2019&ndash;2025 Sentinel-2 medyan kompozitleri. Skor siralama tabanlidir; dogrulama degildir.</p>
<div class="grid">{rows}</div>""")
    print("-> cards.html")


if __name__ == "__main__":
    main()
