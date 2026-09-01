"""Drive an isolated LTNT server through a SANA session, logging the % series."""
import json, sys, time, urllib.request

BASE = sys.argv[1]              # e.g. http://babel-s5-32:8003
EXTRA = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

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
params.update(EXTRA)
job = post("/api/generate", params)["job_id"]
print(f"[drive] job_id={job}", flush=True)

t0 = time.time()
last = None
selections_done = 0
round_start = {}
while True:
    d = get(f"/api/jobs/{job}")
    key = (d["status"], d["progress"], d["stage"])
    if key != last:
        print(f"{time.time()-t0:8.1f}s  {d['progress']:3d}%  {d['status']:18s}  {d['stage']}", flush=True)
        last = key
    if d["status"] == "waiting_selection":
        rnd = d["interval"]
        if rnd == round_start.get("last_selected"):
            time.sleep(1); continue
        print(f"[drive] round {rnd} ready at {time.time()-t0:.1f}s with {len(d['images'])} images", flush=True)
        time.sleep(2)
        post(f"/api/jobs/{job}/select", {"indices": [0, 1]})
        selections_done += 1
        round_start["last_selected"] = rnd
    elif d["status"] in ("done", "error", "cancelled"):
        print(f"[drive] terminal: {d['status']} err={d.get('error')}", flush=True)
        break
    time.sleep(1)
