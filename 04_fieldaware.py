"""v2: tarla-farkinda normalizasyon + sinir-gecen cizgisellik.

Fikir: arkeolojik iz bir tarlanin icinde baslar, tarla sinirini umursamaz,
komsu tarlada devam eder. Modern tarla siniri ise sinirda biter.
Bu yuzden olcut "duzenlilik" degil, "parsel sinirini asan ince cizgi".
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, segmentation, transform

ROOT = Path(__file__).parent
DATA = ROOT / "data"

X0, Y0, NX, NY, RES = 544650, 3911216, 2000, 2000, 10
CELLPX = 100
NCX, NCY = NX // CELLPX, NY // CELLPX


def within_parcel_residual(a, labels):
    idx = np.unique(labels)
    sums = ndimage.sum(a, labels, idx)
    cnts = ndimage.sum(np.ones_like(a), labels, idx)
    lut = np.zeros(labels.max() + 1, "float32")
    lut[idx] = (sums / np.maximum(cnts, 1)).astype("float32")
    return a - lut[labels]


def norm_rank(a):
    v = a[np.isfinite(a)]
    out = np.full(a.shape, np.nan, "float32")
    out[np.isfinite(a)] = np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    return out


def main():
    ch = np.load(DATA / "channels.npz")
    amp, bsi = ch["amp"], ch["bsi_dry"]
    nd_grn, dem, slope = ch["nd_grn"], ch["dem"], ch["slope"]

    # --- 1) tarla/parsel segmentasyonu ---
    base = np.dstack([ndimage.gaussian_filter(nd_grn, 1.5),
                      ndimage.gaussian_filter(amp, 1.5),
                      ndimage.gaussian_filter(bsi, 1.5)]).astype("float32")
    labels = segmentation.felzenszwalb(base, scale=260, sigma=1.6,
                                       min_size=120, channel_axis=-1)
    nseg = labels.max() + 1
    sizes = np.bincount(labels.ravel())[1:]
    ha = sizes * (RES * RES) / 10000.0
    print(f"parsel: {nseg}, medyan {np.median(ha):.2f} ha, "
          f"p10={np.percentile(ha,10):.2f} p90={np.percentile(ha,90):.2f} ha")

    # --- 2) parsel-ici artik ---
    r_amp = within_parcel_residual(amp, labels)
    r_bsi = within_parcel_residual(bsi, labels)
    r_amp_s = ndimage.gaussian_filter(r_amp, 1.0)

    # --- 3) ridge (cizgi) filtresi: ince sirtlar, kenar degil ---
    print("ridge filtresi...")
    ridge = filters.meijering(r_amp_s, sigmas=[1.0, 1.8], black_ridges=False)
    ridge = np.nan_to_num(ridge, nan=0.0)
    ridge /= np.percentile(ridge, 99.5) + 1e-9
    print(f"ridge: p50={np.percentile(ridge,50):.3f} p99={np.percentile(ridge,99):.3f}")

    strong = ndimage.gaussian_filter(ridge, 2) > np.percentile(ridge, 99.7)

    # --- 4) hucre basina: parsel sinirini asan cizgiler ---
    print("Hough + parsel gecisi...")
    cross = np.zeros((NCY, NCX), "float32")
    rdens = np.zeros((NCY, NCX), "float32")
    nl = np.zeros((NCY, NCX), "int32")
    for cy in range(NCY):
        for cx in range(NCX):
            ys, xs = cy * CELLPX, cx * CELLPX
            r = ridge[ys:ys + CELLPX, xs:xs + CELLPX]
            lab = labels[ys:ys + CELLPX, xs:xs + CELLPX]
            st = strong[ys:ys + CELLPX, xs:xs + CELLPX]
            rdens[cy, cx] = float(r.mean())
            if r.size == 0:
                continue
            thr = np.percentile(r, 96.0)
            edges = r > max(thr, 0.15)
            if edges.sum() < 15:
                continue
            try:
                lines = transform.probabilistic_hough_line(
                    edges, threshold=4, line_length=9, line_gap=2)
            except Exception:
                lines = []
            tot = 0.0
            for (x0_, y0_), (x1_, y1_) in lines:
                n = int(max(abs(x1_ - x0_), abs(y1_ - y0_))) + 1
                if n < 8:
                    continue
                xx = np.clip(np.linspace(x0_, x1_, n).astype(int), 0, CELLPX - 1)
                yy = np.clip(np.linspace(y0_, y1_, n).astype(int), 0, CELLPX - 1)
                ids = lab[yy, xx]
                uq = len(np.unique(ids))
                if uq < 2:
                    continue
                # cok guclu/genis cizgiler (yol, kanal) elenir
                if st[yy, xx].mean() > 0.5 and uq <= 2:
                    continue
                length = float(np.hypot(x1_ - x0_, y1_ - y0_)) * RES
                tot += length * (uq - 1)
                nl[cy, cx] += 1
            cross[cy, cx] = tot / 1e4   # m * parsel / hektar

    print(f"cizgi: hucre basina medyan {np.median(nl):.0f}, "
          f"cross medyan {np.median(cross):.1f}")

    # --- 5) skor ---
    sh = (NCY, CELLPX, NCX, CELLPX)
    dem_cell = np.nanmean(dem.reshape(sh), axis=(1, 3))
    slope_cell = np.nanmean(slope.reshape(sh), axis=(1, 3))
    rbsi_cell = np.abs(r_bsi).reshape(sh).mean(axis=(1, 3))

    score = (0.62 * norm_rank(cross)
             + 0.23 * norm_rank(rdens)
             + 0.15 * norm_rank(rbsi_cell))
    score = np.where(np.isfinite(dem_cell), score, np.nan)

    order = np.argsort(-np.nan_to_num(score, nan=-1).ravel())
    rows = []
    for flat in order[:400]:
        cy, cx = divmod(int(flat), NCX)
        rows.append({
            "rank": len(rows) + 1, "cell_y": cy, "cell_x": cx,
            "utm_e": X0 + cx * CELLPX * RES + CELLPX * RES // 2,
            "utm_n": Y0 - cy * CELLPX * RES - CELLPX * RES // 2,
            "score": float(score[cy, cx]),
            "cross": float(cross[cy, cx]),
            "lines": int(nl[cy, cx]),
            "ridge": float(rdens[cy, cx]),
            "slope": float(slope_cell[cy, cx]),
            "elev": float(dem_cell[cy, cx]),
        })
    (DATA / "cells_v2.json").write_text(json.dumps(rows, indent=1))
    np.save(DATA / "score_v2.npy", score)
    np.savez_compressed(DATA / "v2_extra.npz",
                        labels=labels.astype("int32"), ridge=ridge.astype("float32"),
                        r_amp=r_amp.astype("float32"), r_bsi=r_bsi.astype("float32"))

    e = np.array([r["elev"] for r in rows[:60]])
    s = np.array([r["slope"] for r in rows[:60]])
    print(f"\nilk 60 aday: rakim medyan {np.median(e):.0f} m "
          f"(min {e.min():.0f}, max {e.max():.0f}), "
          f"egim>5deg: {(s>5).sum()}/60, rakim>200m: {(e>200).sum()}/60")
    print("en iyi 10:")
    for r in rows[:10]:
        print(f"  #{r['rank']:2d} ({r['utm_e']},{r['utm_n']}) skor={r['score']:.3f} "
              f"cross={r['cross']:.1f} cizgi={r['lines']:3d} "
              f"egim={r['slope']:.1f} rakim={r['elev']:.0f}")


if __name__ == "__main__":
    main()
