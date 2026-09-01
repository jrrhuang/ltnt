"""Drive a CREATE MAP precompute on the isolated server, logging the % series."""
import json, sys, time, urllib.request

BASE = sys.argv[1]
EXTRA = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())

def get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())

params = dict(prompt="a lighthouse on a rocky cliff at golden hour",
              model="sana", num_particles=2, num_clones=2,
              num_resampling_steps=2, num_steps=16, guidance_scale=4.0,
              rho=0.0, seed=42, height=512, width=512, augment=False)
params.update(EXTRA)
job = post("/api/create_map/generate", params)["job_id"]
print(f"[drive] map job={job}", flush=True)
t0 = time.time(); last = None
while True:
    d = get(f"/api/create_map/{job}")
    key = (d["status"], round(d["progress"]), d["stage"])
    if key != last:
        print(f"{time.time()-t0:7.1f}s  {d['progress']:5.1f}%  {d['status']:8s}  {d['stage']}", flush=True)
        last = key
    if d["status"] in ("done", "error"):
        print(f"[drive] terminal={d['status']} error={d.get('error')}", flush=True)
        if d["status"] == "done":
            print(f"[drive] images: {len(d['result']['images'])}", flush=True)
        break
    time.sleep(1)
