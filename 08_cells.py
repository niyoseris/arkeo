"""z=17 karolarindan 350x350 m hucre dizisi olusturur.

Cikti: data/cells_rgb.npy  (N,350,350,3) uint8
       data/cells_grid.json (hucre merkezleri, EPSG:32636)
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
TILES = ROOT / "datasets" / "tiles"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

Z = 17
X0, Y0 = 77729, 51764
NX, NY = 81, 82
LON_LEFT = X0 / 2 ** Z * 360 - 180
LAT_TOP = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * Y0 / 2 ** Z))))
M_PER_PX = (156543.03392 * math.cos(math.radians(35.25))) / 2 ** Z

CELL_M = 350.0
CELL_PX = 350
STEP = int(round(CELL_M / M_PER_PX))     # ~359 px adim


def build_mosaic():
    H, W = NY * 256, NX * 256
    m = np.zeros((H, W, 3), np.uint8)
    miss = 0
    for j in range(NY):
        for i in range(NX):
            p = TILES / f"{Z}_{X0+i}_{Y0+j}.jpg"
            if not p.exists():
                miss += 1
                continue
            try:
                a = np.array(Image.open(p).convert("RGB"))
                m[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = a
            except Exception:
                miss += 1
    print(f"mozaik {m.shape}, eksik {miss}, {M_PER_PX:.3f} m/px")
    return m


def main():
    m = build_mosaic()
    H, W = m.shape[:2]
    ncy, ncx = H // STEP, W // STEP
    cells, grid = [], []
    for cy in range(ncy):
        for cx in range(ncx):
            y = cy * STEP
            x = cx * STEP
            chip = m[y:y + CELL_PX, x:x + CELL_PX]
            if chip.shape[:2] != (CELL_PX, CELL_PX):
                continue
            if chip.mean() < 8:          # bos/deniz
                continue
            cells.append(chip)
            grid.append({"cy": cy, "cx": cx,
                         "px": x + CELL_PX // 2, "py": y + CELL_PX // 2})
    arr = np.stack(cells)
    np.save(DATA / "cells_rgb.npy", arr)
    (DATA / "cells_grid.json").write_text(json.dumps({
        "z": Z, "m_per_px": M_PER_PX, "step_px": STEP,
        "lon_left": LON_LEFT, "lat_top": LAT_TOP,
        "mosaic_h": H, "mosaic_w": W, "cells": grid}, indent=1))
    print(f"{len(cells)} hucre -> data/cells_rgb.npy "
          f"({arr.nbytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
