"""Yuksek cozunurluklu (Esri World Imagery) hedefli goruntu cekici.

NOT: Bu katman yalnizca GORSEL INCELEME icindir. Analitik/sayisal katman
olarak kullanilmaz ve yeniden dagitilmaz (Esri/Maxar lisansi).
"""
import io
import math
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from pyproj import Transformer

ROOT = Path(__file__).parent
T = Transformer.from_crs("EPSG:32636", "EPSG:4326", always_xy=True)
URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
       "World_Imagery/MapServer/tile/{z}/{y}/{x}")


def deg2num(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat)))
         / math.pi) / 2 * n
    return x, y


def m_per_px(lat, z):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)


def fetch(e_utm, n_utm, span_tiles=4, z=18):
    lon, lat = T.transform(e_utm, n_utm)
    fx, fy = deg2num(lon, lat, z)
    cx, cy = int(fx), int(fy)
    size = (2 * span_tiles + 1) * 256
    canvas = Image.new("RGB", (size, size), (32, 32, 32))
    ok = 0
    for dy in range(-span_tiles, span_tiles + 1):
        for dx in range(-span_tiles, span_tiles + 1):
            u = URL.format(z=z, y=cy + dy, x=cx + dx)
            try:
                r = requests.get(u, timeout=20)
                if r.status_code == 200:
                    canvas.paste(Image.open(io.BytesIO(r.content)),
                                 ((dx + span_tiles) * 256, (dy + span_tiles) * 256))
                    ok += 1
            except Exception:
                pass
    mpp = m_per_px(lat, z)
    return canvas, ok, mpp, lon, lat


if __name__ == "__main__":
    import sys
    e, n = float(sys.argv[1]), float(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else "hires.jpg"
    img, ok, mpp, lon, lat = fetch(e, n, span_tiles=int(sys.argv[4]) if len(sys.argv) > 4 else 4)
    img.save(ROOT / out, quality=92)
    print(f"{out}: {ok} tile, {img.width}px, "
          f"{img.width*mpp/1000:.2f} km, {mpp:.2f} m/px, "
          f"lon={lon:.4f} lat={lat:.4f}")
