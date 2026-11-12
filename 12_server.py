"""Yerel karo sunucusu + isaretleme API'si (yalnizca 127.0.0.1)."""
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pyproj import Transformer

ROOT = Path(__file__).parent
TILES = ROOT / "datasets" / "tiles"
PYR = ROOT / "datasets" / "pyramid"
MARKERS = ROOT / "markers.json"
MARKERS.touch()
T = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)

TILE_RE = re.compile(r"^/tiles/(\d+)/(\d+)/(\d+)\.jpg$")


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (ROOT / "viewer.html").read_bytes()
            return self._send(200, body, "text/html; charset=utf-8")
        if self.path == "/markers":
            return self._send(200, MARKERS.read_bytes() or b"[]")
        m = TILE_RE.match(self.path)
        if m:
            z, x, y = (int(v) for v in m.groups())
            p = (TILES / f"17_{x}_{y}.jpg") if z == 17 else (PYR / f"{z}_{x}_{y}.jpg")
            if p.exists():
                return self._send(200, p.read_bytes(), "image/jpeg")
            return self._send(404, b"{}")
        self._send(404, b"{}")

    def do_POST(self):
        if self.path != "/mark":
            return self._send(404, b"{}")
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        lat, lon = float(data["lat"]), float(data["lon"])
        e, nn = T.transform(lon, lat)
        rec = {"lat": lat, "lon": lon,
               "utm_e": round(e, 1), "utm_n": round(nn, 1),
               "note": data.get("note", ""),
               "type": data.get("type", ""),
               "id": data.get("id", "")}
        try:
            cur = json.loads(MARKERS.read_text() or "[]")
        except Exception:
            cur = []
        cur.append(rec)
        MARKERS.write_text(json.dumps(cur, indent=1, ensure_ascii=False))
        return self._send(200, json.dumps({"ok": True, "utm_e": rec["utm_e"],
                                           "utm_n": rec["utm_n"]}).encode())

    def log_message(self, *a):
        pass


PORT = 8787

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"goruntuleyici: http://127.0.0.1:{PORT}", flush=True)
    print(f"isaretler: {MARKERS}", flush=True)
    srv.serve_forever()
