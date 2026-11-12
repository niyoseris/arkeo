"""z=17 karolarindan alt zoom seviyeleri (16,15,14) uretir."""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
SRC = ROOT / "datasets" / "tiles"
OUT = ROOT / "datasets" / "pyramid"
OUT.mkdir(parents=True, exist_ok=True)

X0, Y0, NX, NY = 77729, 51764, 81, 82


def parent_tile(x, y, z):
    """z-1 seviyesindeki karo: z seviyesindeki 4 cocugunu 128'e indirip birlestir."""
    canvas = Image.new("RGB", (256, 256), (18, 18, 18))
    for dy in range(2):
        for dx in range(2):
            cx, cy = x * 2 + dx, y * 2 + dy
            p = SRC / f"17_{cx}_{cy}.jpg"
            if not p.exists():
                continue
            try:
                im = Image.open(p).convert("RGB").resize((128, 128), Image.LANCZOS)
                canvas.paste(im, (dx * 128, dy * 128))
            except Exception:
                pass
    return canvas


def build():
    # z=16
    xs = range(X0 // 2, (X0 + NX - 1) // 2 + 1)
    ys = range(Y0 // 2, (Y0 + NY - 1) // 2 + 1)
    made = 0
    for x in xs:
        for y in ys:
            parent_tile(x, y, 17).save(OUT / f"16_{x}_{y}.jpg", quality=88)
            made += 1
    print(f"z16: {made} karo")

    # z=15, 14: bir onceki seviyeden
    for z in (15, 14):
        prev = z + 1
        xs = range(X0 // (2 ** (17 - z)), (X0 + NX - 1) // (2 ** (17 - z)) + 1)
        ys = range(Y0 // (2 ** (17 - z)), (Y0 + NY - 1) // (2 ** (17 - z)) + 1)
        made = 0
        for x in xs:
            for y in ys:
                canvas = Image.new("RGB", (256, 256), (18, 18, 18))
                for dy in range(2):
                    for dx in range(2):
                        cx, cy = x * 2 + dx, y * 2 + dy
                        p = OUT / f"{prev}_{cx}_{cy}.jpg"
                        if not p.exists():
                            continue
                        im = Image.open(p).resize((128, 128), Image.LANCZOS)
                        canvas.paste(im, (dx * 128, dy * 128))
                canvas.save(OUT / f"{z}_{x}_{y}.jpg", quality=88)
                made += 1
        print(f"z{z}: {made} karo")


if __name__ == "__main__":
    build()
