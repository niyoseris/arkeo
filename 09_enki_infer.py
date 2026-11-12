"""Enki hazir modelini (elle yeniden kurulmus) AOI hucrelerine uygular."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODEL = ROOT / "model"


def build_model(keras, layers):
    cfg = json.loads((MODEL / "enki0_config.json").read_text())
    base = keras.models.model_from_json(json.dumps(cfg))
    inp = keras.Input((350, 350, 3), name="input_layer_1")
    t = base(inp)
    t = t[0] if isinstance(t, list) else t      # Keras 3 liste donduruyor
    y = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(t)
    y = layers.Dropout(0.3, name="dropout")(y)
    y = layers.Dense(256, activation="relu", name="dense")(y)
    y = layers.Dense(2, activation="softmax", name="dense_1")(y)
    m = keras.Model(inp, y)
    m.load_weights(str(MODEL / "model.weights.h5"))
    return m


def main():
    import keras
    from keras import layers

    cells = np.load(DATA / "cells_rgb.npy", mmap_mode="r")
    grid = json.loads((DATA / "cells_grid.json").read_text())
    m = build_model(keras, layers)
    print(f"{len(cells)} hucre, model {m.count_params()} parametre")

    B = 32
    out = np.zeros(len(cells), "float32")
    for i in range(0, len(cells), B):
        b = np.asarray(cells[i:i + B], dtype="float32")   # 0-255, /255 yok
        out[i:i + len(b)] = np.asarray(m.predict(b, verbose=0))[:, 1]
        if i % 960 == 0:
            print(f"\r{i}/{len(cells)}", end="", flush=True)
    print()

    np.save(DATA / "enki_scores.npy", out)
    meta = grid["cells"]
    order = np.argsort(-out)
    print(f"skor dagilimi: medyan={np.median(out):.4f} "
          f"p95={np.percentile(out,95):.4f} p99={np.percentile(out,99):.4f} "
          f"max={out.max():.4f}")
    print(f">0.5 olan hucre: {(out > 0.5).sum()} / {len(out)}")
    print("ilk 15:")
    for rank, k in enumerate(order[:15], 1):
        g = meta[k]
        print(f"  #{rank:2d} hucre {k:4d} skor={out[k]:.4f} (px {g['px']},{g['py']})")

    (DATA / "enki_top.json").write_text(json.dumps(
        [{"cell": int(k), "score": float(out[k]),
          "px": meta[k]["px"], "py": meta[k]["py"]} for k in order[:200]],
        indent=1))


if __name__ == "__main__":
    main()
