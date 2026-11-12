"""Paralel aralik indirici (yeniden baslatilabilir)."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

CHUNK = 8 * 1024 * 1024
WORKERS = 10


def head_size(url):
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def main(url, out, workers=WORKERS):
    out = Path(out)
    total = head_size(url)
    have = out.stat().st_size if out.exists() else 0
    if have > total:
        have = 0
    print(f"boyut {total/1e9:.2f} GB, mevcut {have/1e6:.0f} MB, "
          f"kalan {(total-have)/1e6:.0f} MB", flush=True)

    with open(out, "r+b" if have else "wb") as f:
        f.truncate(total)

    ranges = [(a, min(a + CHUNK - 1, total - 1))
              for a in range(have, total, CHUNK)]
    done = have
    t0 = time.time()

    def grab(rng):
        a, b = rng
        for attempt in range(4):
            try:
                r = requests.get(url, headers={"Range": f"bytes={a}-{b}"},
                                 timeout=180, stream=True)
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                buf = bytearray()
                for c in r.iter_content(262144):
                    buf += c
                return a, bytes(buf)
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(grab, r): r for r in ranges}
        for fut in as_completed(futs):
            a, data = fut.result()
            with open(out, "r+b") as f:
                f.seek(a)
                f.write(data)
            done += len(data)
            el = time.time() - t0
            rate = (done - have) / max(el, 1e-6)
            eta = (total - done) / max(rate, 1)
            print(f"\r{done/1e6:7.0f}/{total/1e6:.0f} MB  "
                  f"{rate/1024:6.0f} KB/s  ETA {eta/60:5.1f} dk",
                  end="", flush=True)
    print(f"\nbitti: {out}  ({out.stat().st_size/1e9:.2f} GB, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else WORKERS)
