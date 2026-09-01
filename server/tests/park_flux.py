import json, time, urllib.request, sys
BASE = "http://babel-s5-32:8003"
def post(p, b):
    r = urllib.request.Request(BASE+p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=30).read())
def get(p):
    return json.loads(urllib.request.urlopen(BASE+p, timeout=30).read())
job = post("/api/generate", dict(prompt="a lighthouse on a cliff", model="flux", n_images=2,
           num_steps=8, num_clones=2, num_resampling_steps=1, guidance_scale=3.5, rho=0.4,
           schedule_mode="aggressive", height=512, width=512, seed=7, augment=False))["job_id"]
print("gen job", job, flush=True)
t0 = time.time(); last = None
while True:
    d = get(f"/api/jobs/{job}")
    k = (d["status"], d["progress"], d["stage"])
    if k != last:
        print(f"{time.time()-t0:6.1f}s {d['progress']:3d}% {d['stage']}", flush=True)
        last = k
    if d["status"] == "waiting_selection":
        print("PARKED at round 1 (holding model lock)", flush=True); break
    if d["status"] in ("done", "error", "cancelled"):
        print("terminal", d["status"], d.get("error")); sys.exit(1)
    time.sleep(2)
