"""Serdarli AOI - kesif: DEM profili ve Sentinel-2 arsiv yogunlugu."""
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# --- AOI: Serdarli merkezli 20 x 20 km ---
LAT0, LON0 = 35.2525, 33.6008
HALF_KM = 10.0
DLAT = HALF_KM / 110.9
DLON = HALF_KM / (111.32 * math.cos(math.radians(LAT0)))
BBOX = (LON0 - DLON, LAT0 - DLAT, LON0 + DLON, LAT0 + DLAT)  # W, S, E, N
print(f"AOI bbox (W,S,E,N): {tuple(round(v, 5) for v in BBOX)}")

DEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_N35_00_E033_00_DEM/"
    "Copernicus_DSM_COG_10_N35_00_E033_00_DEM.tif"
)

out = DATA / "dem_serdarli.tif"
if not out.exists():
    with rasterio.open(DEM_URL) as src:
        win = from_bounds(*BBOX, transform=src.transform)
        dem = src.read(1, window=win, boundless=True)
        prof = src.profile.copy()
        prof.update(
            height=dem.shape[0],
            width=dem.shape[1],
            transform=src.window_transform(win),
            compress="deflate",
        )
        if src.nodata is not None:
            dem = np.where(dem == src.nodata, np.nan, dem).astype("float32")
            prof.update(dtype="float32", nodata=np.nan)
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(dem, 1)
    print(f"DEM yazildi: {out}")

with rasterio.open(out) as src:
    z = src.read(1).astype("float32")
    res = src.res
valid = z[np.isfinite(z)]
print(f"\n--- DEM ---")
print(f"piksel: {z.shape}, cozunurluk ~{res[0]*111320:.1f} m")
print(f"rakim: min={valid.min():.1f}  p05={np.percentile(valid,5):.1f}  "
      f"medyan={np.median(valid):.1f}  p95={np.percentile(valid,95):.1f}  max={valid.max():.1f} m")
print(f"<50 m altinda: {np.mean(valid < 50)*100:.1f}%   >200 m: {np.mean(valid > 200)*100:.1f}%")
print(f"yukseklik std: {valid.std():.1f} m")

# --- Sentinel-2 arsiv ---
from pystac_client import Client

cat = Client.open("https://earth-search.aws.element84.com/v1")
search = cat.search(
    collections=["sentinel-2-l2a"],
    bbox=BBOX,
    datetime="2018-01-01/2026-01-01",
    query={"eo:cloud_cover": {"lt": 25}},
    max_items=2000,
)
items = list(search.items())
print(f"\n--- Sentinel-2 L2A (bulut < %25) ---")
print(f"toplam sahne: {len(items)}")

by_year = {}
for it in items:
    y = it.datetime.year
    by_year.setdefault(y, []).append(it.properties.get("eo:cloud_cover", 999))
for y in sorted(by_year):
    cc = np.array(by_year[y])
    print(f"  {y}: {len(cc):3d} sahne, medyan bulut %{np.median(cc):.1f}")

if items:
    s = items[0]
    print(f"\nornek sahne: {s.id}")
    print(f"  tile: {s.properties.get('grid:code')}")
    print(f"  assets: {sorted(s.assets)[:14]}")
    (DATA / "s2_items.json").write_text(json.dumps(
        [{"id": i.id,
          "datetime": i.datetime.isoformat(),
          "cloud": i.properties.get("eo:cloud_cover"),
          "tile": i.properties.get("grid:code"),
          "assets": {k: v.href for k, v in i.assets.items()}}
         for i in items], indent=1))
    print(f"  -> {DATA/'s2_items.json'} yazildi")
