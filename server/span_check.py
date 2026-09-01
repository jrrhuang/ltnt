"""Span verification driver (checks 1+2). Usage:
  python span_check.py BASE flux_gate   # FLUX span, let it run to auto-stop/cap
  python span_check.py BASE finishtest  # SANA span, POST finish_round mid-stream
Records EVERY distinct span-state poll (message/saturating transitions)."""
import json, sys, time, urllib.request

BASE, MODE = sys.argv[1], sys.argv[2]
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time() - T0, m), flush=True)
def http(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
        data=(json.dumps(body).encode() if body is not None else None),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())

while True:
    try:
        st = http("GET", "/api/status")
        if st.get("ready"): break
        log("status: %s" % json.dumps(st)[:200])
    except Exception as e:
        log("waiting for server (%s)" % e)
    time.sleep(10)
log("server ready")

model = "flux" if MODE == "flux_gate" else "sana"
if MODE == "sana_gate": model = "sana"
body = {"prompt": "a lighthouse on a cliff at dusk, oil painting",
        "model": model, "span_round1": True, "n_images": 6,
        "num_steps": 28, "num_clones": 3, "num_resampling_steps": 1,
        "guidance_scale": 3.5, "rho": 0.4, "height": 512, "width": 512,
        "span_batch_size": 8, "span_max_images": 64, "span_num_steps": 14}
r = http("POST", "/api/generate", body)
job, sess = r["job_id"], r["session_id"]
log("job=%s session=%s model=%s" % (job, sess, model))

last_sig = None
finish_sent = False
while True:
    d = http("GET", "/api/jobs/%s" % job)
    span = d.get("span") or {}
    sig = (span.get("batch_n"), span.get("saturating"), span.get("message"),
           span.get("n_prompt_variations"), d.get("status"))
    if sig != last_sig:
        last_sig = sig
        log("POLL status=%s batch_n=%s nov=%s sat=%s nvar=%s msg=%r series=%s" % (
            d.get("status"), span.get("batch_n"), span.get("novelty"),
            span.get("saturating"), span.get("n_prompt_variations"),
            span.get("message"), span.get("span_series")))
    if MODE == "finishtest" and not finish_sent and (span.get("batch_n") or 0) >= 3:
        n_at_send = len(d.get("images") or [])
        log("SENDING finish_round at batch_n=%s n_images=%d" % (span.get("batch_n"), n_at_send))
        http("POST", "/api/jobs/%s/finish_round" % job)
        finish_sent = True
    if d.get("status") == "waiting_selection":
        log("SPAN DONE: n=%d stop_reason=%s series=%s first_batch_s=%s" % (
            len(d.get("images") or []), span.get("stop_reason"),
            span.get("span_series"), span.get("first_batch_seconds")))
        break
    if d.get("status") in ("error", "cancelled"):
        raise SystemExit("FAIL: %s" % json.dumps(d)[:500])
    time.sleep(1.5)

# Leave the job at waiting_selection: the next /api/generate takes over
# (server cancels the previous in-flight session on takeover).
print("CHECK COMPLETE session=%s" % sess)
