"""AOI icin z=17 (~0.98 m/px) Esri World Imagery karolarini indirir."""
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).parent
TILES = ROOT / "datasets" / "tiles"
TILES.mkdir(parents=True, exist_ok=True)

Z = int(sys.argv[1]) if len(sys.argv) > 1 else 17
LON0, LAT0, LON1, LAT1 = 33.4908, 35.16233, 33.7108, 35.34267
URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
       "World_Imagery/MapServer/tile/{z}/{y}/{x}")
HDR = {"User-Agent": "arkeo-research/0.1 (archaeological prospection study)"}


def lonlat2tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat)))
             / math.pi) / 2 * n)
    return x, y


def fetch(x, y):
    p = TILES / f"{Z}_{x}_{y}.jpg"
    if p.exists() and p.stat().st_size > 500:
        return 0
    for attempt in range(3):
        try:
            r = requests.get(URL.format(z=Z, x=x, y=y), headers=HDR, timeout=25)
            if r.status_code == 200 and len(r.content) > 500:
                p.write_bytes(r.content)
                return len(r.content)
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return -1


def main():
    x0, y0 = lonlat2tile(LON0, LAT1, Z)
    x1, y1 = lonlat2tile(LON1, LAT0, Z)
    xs = range(x0, x1 + 1)
    ys = range(y0, y1 + 1)
    jobs = [(x, y) for x in xs for y in ys]
    print(f"z={Z}: {len(jobs)} karo ({len(list(xs))}x{len(list(ys))})", flush=True)

    t0 = time.time()
    got = fail = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(fetch, x, y) for x, y in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            n = f.result()
            if n < 0:
                fail += 1
            else:
                got += n
            if i % 250 == 0:
                el = time.time() - t0
                print(f"\r{i}/{len(jobs)}  {got/1e6:.1f} MB  "
                      f"{i/el:.1f} karo/s  hata {fail}", end="", flush=True)
    print(f"\nbitti {got/1e6:.1f} MB, hata {fail}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
