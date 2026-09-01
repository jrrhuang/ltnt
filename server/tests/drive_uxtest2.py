"""Fast-poll driver: 250ms polling to capture per-step sketch ticks + refine %."""
import json, sys, time, urllib.request

BASE = sys.argv[1]

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())

params = dict(prompt="a lighthouse on a rocky cliff at golden hour, dramatic sky",
              model="sana", n_images=6, num_steps=28, num_clones=3,
              num_resampling_steps=2, guidance_scale=4.0, rho=0.4,
              schedule_mode="aggressive", height=512, width=512, seed=1234)
job = post("/api/generate", params)["job_id"]
print(f"[drive] job={job}", flush=True)
t0 = time.time(); last = None
# Phase 1: to round 1, fast-poll
while True:
    d = get(f"/api/jobs/{job}")
    key = (d["status"], d["progress"], d["stage"])
    if key != last:
        print(f"{time.time()-t0:7.2f}s  {d['progress']:3d}%  {d['stage']}", flush=True)
        last = key
    if d["status"] == "waiting_selection":
        break
    if d["status"] in ("done", "error", "cancelled"):
        print("[drive] UNEXPECTED terminal", d.get("error")); sys.exit(1)
    time.sleep(0.25)
print(f"[drive] round 1 at {time.time()-t0:.1f}s — requesting REFINE of particle 0", flush=True)
post(f"/api/jobs/{job}/refine", {"indices": [0]})
# Phase 2: refine_progress series
rp_last = None
while True:
    d = get(f"/api/jobs/{job}")
    rp = d.get("refine_progress")
    if rp != rp_last:
        print(f"{time.time()-t0:7.2f}s  refine_progress={rp}  stage={d['stage']}", flush=True)
        rp_last = rp
    if d.get("refined", {}).get("0"):
        print(f"[drive] refined url: {d['refined']['0']}", flush=True)
        break
    time.sleep(0.25)
post(f"/api/jobs/{job}/select", {"indices": [0, 1]})
print("[drive] selected; done", flush=True)
