"""Boot beacon — minimal status + static server for the LTNT job's STAGING phase.

The job script stages multi-GB model snapshots to node-local /scratch BEFORE
launching the real server, which used to mean minutes of totally dead air (the
port serves nothing). The beacon runs on the app port during that window:

  * serves the built frontend (ltnt_frontend/dist) so the page always loads
  * GET /api/status + /api/health report a REAL staging fraction:
        bytes present under the /scratch dest dir  /  expected total bytes
    scaled into [0, LTNT_BOOT_BASE) so the bar stays monotone when the real
    server takes over (the real server offsets its own boot fractions by
    LTNT_BOOT_BASE — see server.py _set_boot).

The job script kills the beacon right before exec'ing the real server; the
frontend boot gate shows a brief "reconnecting" during the port handoff.

Usage: boot_beacon.py <port> <watch_dir> <expected_total_bytes> [boot_base]
"""
import http.server
import json
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
WATCH_DIR = sys.argv[2] if len(sys.argv) > 2 else "/scratch/jerryhua/ltnt_models"
TOTAL = float(sys.argv[3]) if len(sys.argv) > 3 else 20e9
BOOT_BASE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "ltnt_frontend", "dist")


def _dir_bytes(d):
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=DIST, **kw)

    def do_GET(self):
        path = self.path.split("?")[0].split("#")[0]
        if path in ("/api/status", "/api/health"):
            sz = _dir_bytes(WATCH_DIR)
            frac = min(sz / max(TOTAL, 1), 0.999)
            body = json.dumps({
                "ready": False,
                "phase": "copying model files to this machine",
                "fraction": round(frac * BOOT_BASE, 3),
                "detail": f"{sz / 2**30:.1f} / {TOTAL / 2**30:.0f} GB copied",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/api/"):
            # Any other API during staging: not ready yet.
            self.send_response(503)
            self.end_headers()
            return
        # Static SPA: real file if present, else index.html (SPA fallback).
        fs_path = os.path.join(DIST, path.lstrip("/"))
        if path == "/" or not os.path.isfile(fs_path):
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, *a):  # keep the job log clean
        pass


if __name__ == "__main__":
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[beacon] boot beacon on :{PORT} watching {WATCH_DIR} "
          f"(total={TOTAL/2**30:.0f} GB, base={BOOT_BASE})", flush=True)
    srv.serve_forever()
