"""Kompozitlerden yapisal/bitkisel anomali kanallari ve hucre skorlamasi."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage

ROOT = Path(__file__).parent
DATA = ROOT / "data"

X0, Y0, NX, NY, RES = 544650, 3911216, 2000, 2000, 10
TRANSFORM = from_origin(X0, Y0, RES, RES)
CRS = "EPSG:32636"
CELL = 1000  # m
CELLPX = CELL // RES  # 100 px
NCX, NCY = NX // CELLPX, NY // CELLPX

BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]


def load(season):
    with rasterio.open(DATA / f"comp_{season}.tif") as s:
        return {b: s.read(i + 1).astype("float32")
                for i, b in enumerate(BANDS)}


def ndvi(d):
    n, r = d["nir"], d["red"]
    return (n - r) / (n + r + 1e-6)


def bsi(d):
    b, g, r, n, s1, s2 = (d[k] for k in BANDS)
    num = (s1 + r) - (n + b)
    den = (s1 + r) + (n + b)
    return num / (den + 1e-6)


def local_z(a, win):
    m = ndimage.uniform_filter(a, win, mode="reflect")
    v = ndimage.uniform_filter(a * a, win, mode="reflect") - m * m
    return (a - m) / np.sqrt(np.maximum(v, 1e-8))


def structure(a, win=9):
    gx = ndimage.sobel(a, axis=1)
    gy = ndimage.sobel(a, axis=0)
    jxx = ndimage.uniform_filter(gx * gx, win, mode="reflect")
    jyy = ndimage.uniform_filter(gy * gy, win, mode="reflect")
    jxy = ndimage.uniform_filter(gx * gy, win, mode="reflect")
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-8)
    return np.sqrt(tr), coh


def dem_on_grid():
    with rasterio.open(DATA / "dem_serdarli.tif") as s:
        src = s.read(1).astype("float32")
        dem = np.full((NY, NX), np.nan, "float32")
        reproject(src, dem, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=TRANSFORM, dst_crs=CRS,
                  resampling=Resampling.bilinear,
                  src_nodata=None, dst_nodata=np.nan)
    dem = np.where(np.isfinite(dem) & (dem > 0.5), dem, np.nan)
    filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    smooth = ndimage.gaussian_filter(filled, sigma=500 / RES)
    lrm = filled - smooth
    slope = np.hypot(ndimage.sobel(filled, 1), ndimage.sobel(filled, 0)) / 8.0
    return dem, lrm, slope


def cellstats(a):
    """(NCY,NCX) hucre bazli ortalama, nan-guvenli."""
    sh = (NCY, CELLPX, NCX, CELLPX)
    m = np.nanmean(a.reshape(sh), axis=(1, 3))
    return m


def block_max(a):
    sh = (NCY, CELLPX, NCX, CELLPX)
    return np.nanmax(a.reshape(sh), axis=(1, 3))


def norm_rank(a):
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a)
    order = np.argsort(np.argsort(v))
    out = np.full(a.shape, np.nan, "float32")
    out[np.isfinite(a)] = order / max(len(v) - 1, 1)
    return out


def main():
    dry, green = load("dry"), load("green")
    print("kompozitler yuklendi")

    nd_dry, nd_grn = ndvi(dry), ndvi(green)
    bsi_dry, bsi_grn = bsi(dry), bsi(green)
    amp = nd_grn - nd_dry                      # mevsimsel NDVI genligi

    print("anomali kanallari...")
    z_amp_max = block_max(np.abs(local_z(amp, 201)))     # 2 km olcekte sapma
    z_bsi_max = block_max(np.abs(local_z(bsi_dry, 201)))
    z_ndvi_max = block_max(np.abs(local_z(nd_dry, 201)))

    emag, coh = structure(amp if amp.std() > 0 else bsi_dry, win=9)
    coh_cell = cellstats(coh)
    edge_cell = cellstats(np.log1p(emag))

    dem, lrm, slope = dem_on_grid()
    lrm_cell = cellstats(np.abs(lrm))
    slope_cell = cellstats(slope)
    elev_cell = cellstats(dem)

    feats = {
        "z_amp": z_amp_max,
        "z_bsi": z_bsi_max,
        "z_ndvi": z_ndvi_max,
        "coherence": coh_cell,
        "edge": edge_cell,
        "lrm": lrm_cell,
        "slope": slope_cell,
        "elev": elev_cell,
    }
    print({k: float(np.nanmedian(v)) for k, v in feats.items()})

    # --- skor: siralama tabanli birlestirme (olcek bagimsiz) ---
    w = {"z_amp": 0.30, "z_bsi": 0.20, "z_ndvi": 0.15,
         "coherence": 0.20, "edge": 0.15}
    score = np.zeros((NCY, NCX), "float32")
    wsum = 0.0
    for k, wk in w.items():
        score += wk * norm_rank(feats[k])
        wsum += wk
    score /= wsum

    # ova disi (yamaç/orman) hucrelerde bitkisel sinyaller zayif:
    # egim > 8 deg ise arazi kanalina agirlik kaydir
    steep = (slope_cell > 8) & np.isfinite(slope_cell)
    score_terrain = norm_rank(feats["lrm"]) * 0.6 + norm_rank(feats["coherence"]) * 0.4
    score = np.where(steep, 0.5 * score + 0.5 * score_terrain, score)
    score = np.where(np.isfinite(elev_cell), score, np.nan)

    order = np.argsort(-np.nan_to_num(score, nan=-1).ravel())
    rows = []
    for flat in order[:400]:
        cy, cx = divmod(int(flat), NCX)
        rows.append({
            "rank": len(rows) + 1,
            "cell_y": cy, "cell_x": cx,
            "utm_e": X0 + cx * CELL + CELL // 2,
            "utm_n": Y0 - cy * CELL - CELL // 2,
            "score": float(score[cy, cx]),
            "z_amp": float(z_amp_max[cy, cx]),
            "z_bsi": float(z_bsi_max[cy, cx]),
            "coherence": float(coh_cell[cy, cx]),
            "slope": float(slope_cell[cy, cx]),
            "elev": float(elev_cell[cy, cx]),
        })
    (DATA / "cells.json").write_text(json.dumps(rows, indent=1))
    np.save(DATA / "score.npy", score)

    # kanal dokumleri (goruntuleme icin)
    np.savez_compressed(
        DATA / "channels.npz",
        nd_dry=nd_dry.astype("float32"), nd_grn=nd_grn.astype("float32"),
        amp=amp.astype("float32"), bsi_dry=bsi_dry.astype("float32"),
        z_amp=local_z(amp, 201).astype("float32"),
        z_bsi=local_z(bsi_dry, 201).astype("float32"),
        coh=coh.astype("float32"), lrm=lrm.astype("float32"),
        slope=slope.astype("float32"), dem=dem.astype("float32"),
    )
    print(f"\ntoplam hucre: {NCX*NCY}, skor medyan={np.nanmedian(score):.3f}")
    print("en iyi 10:")
    for r in rows[:10]:
        print(f"  #{r['rank']:2d} ({r['utm_e']},{r['utm_n']}) skor={r['score']:.3f} "
              f"z_amp={r['z_amp']:.1f} coh={r['coherence']:.2f} "
              f"egim={r['slope']:.1f} rakim={r['elev']:.0f}")


if __name__ == "__main__":
    main()
