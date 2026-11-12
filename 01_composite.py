"""Sentinel-2 cok-yilli, cok-mevsimli medyan kompozit - Serdarli AOI.

Cikti: data/comp_<season>_<band>.tif  (yansima 0-1, EPSG:32636, 10 m)
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.env import Env
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# --- ortak UTM izgarasi (Serdarli merkezli 20 x 20 km) ---
X0, Y0, NX, NY, RES = 544650, 3911216, 2000, 2000, 10
BBOX_UTM = (X0, Y0 - NY * RES, X0 + NX * RES, Y0)
TRANSFORM = from_origin(X0, Y0, RES, RES)
BBOX_WGS = (33.4908, 35.16233, 33.7108, 35.34267)
CRS = "EPSG:32636"

BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
SEASONS = {"dry": ((5, 10), (7, 31)), "green": ((2, 15), (4, 20))}
YEARS = list(range(2019, 2026))
MAX_PER_YEAR = 3
CLOUD = 12.0

GDAL_CFG = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_VERSION="2",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="134217728",
    CPL_VSIL_CURL_USE_HEAD="NO",
)


def season_window(year, season):
    (m0, d0), (m1, d1) = SEASONS[season]
    return date(year, m0, d0), date(year, m1, d1)


def select_scenes(season):
    """Her yil icin bulutu en az, birbirinden >=20 gun uzak sahneler."""
    cat = Client.open("https://earth-search.aws.element84.com/v1")
    picked = []
    for year in YEARS:
        a, b = season_window(year, season)
        items = list(cat.search(
            collections=["sentinel-2-l2a"],
            bbox=BBOX_WGS,
            datetime=f"{a}/{b}",
            query={"eo:cloud_cover": {"lt": CLOUD}},
            max_items=300,
        ).items())
        items = [i for i in items
                 if i.properties.get("grid:code") == "MGRS-36SWE"]
        items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 99))
        chosen = []
        for it in items:
            if all(abs((it.datetime.date() - c.datetime.date()).days) >= 20 for c in chosen):
                chosen.append(it)
            if len(chosen) == MAX_PER_YEAR:
                break
        picked.extend(chosen)
        print(f"  {year}: {len(items):3d} uygun -> {len(chosen)} secildi")
    return picked


def read_band(href):
    last = None
    for _ in range(3):
        try:
            with rasterio.open(f"/vsicurl/{href}") as s:
                win = from_bounds(*BBOX_UTM, transform=s.transform)
                a = s.read(1, window=win, boundless=True, fill_value=0,
                           out_dtype="float32")
            return a
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5)
    print(f"    HATA {href[-60:]}: {last}")
    return None


def main(season):
    print(f"\n=== {season.upper()} mevsimi ===")
    t_all = time.time()
    scenes = select_scenes(season)
    print(f"toplam secilen: {len(scenes)} sahne")
    if not scenes:
        sys.exit(f"{season} icin sahne bulunamadi")

    (DATA / f"scenes_{season}.json").write_text(json.dumps(
        [{"id": s.id, "datetime": s.datetime.isoformat(),
          "cloud": s.properties.get("eo:cloud_cover"),
          "assets": {k: s.assets[k].href for k in BANDS}} for s in scenes], indent=1))

    stacks = {b: [] for b in BANDS}
    with Env(**GDAL_CFG):
        for bi, band in enumerate(BANDS, 1):
            t0 = time.time()
            hrefs = [s.assets[band].href for s in scenes]
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = {ex.submit(read_band, h): i for i, h in enumerate(hrefs)}
                got = []
                for f in as_completed(futs):
                    i = futs[f]
                    a = f.result()
                    if a is not None:
                        a = np.where(a > 0, a, np.nan)
                        got.append((i, a))
            got.sort(key=lambda x: x[0])
            arr = np.stack([g[1] for g in got])
            comp = np.nanmedian(arr, axis=0).astype("float32") / 10000.0
            np.clip(comp, 0, 1.2, out=comp)
            stacks[band] = comp
            cnt = np.sum(np.isfinite(arr), axis=0)
            print(f"  [{bi}/{len(BANDS)}] {band:7s}: {arr.shape[0]} sahne, "
                  f"gecerli medyan piksel={np.median(cnt):.0f}, "
                  f"medyan={np.nanmedian(comp):.4f}, {time.time()-t0:.0f}s")

    prof = dict(driver="GTiff", height=NY, width=NX, count=len(BANDS),
                dtype="float32", crs=CRS, transform=TRANSFORM,
                nodata=-9999.0, compress="deflate", tiled=True,
                blockxsize=256, blockysize=256)
    out = DATA / f"comp_{season}.tif"
    with rasterio.open(out, "w", **prof) as dst:
        for i, band in enumerate(BANDS, 1):
            dst.write(stacks[band], i)
            dst.set_band_description(i, band)
    print(f"-> {out}  ({time.time()-t_all:.0f}s toplam)")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["dry", "green"]):
        main(s)
