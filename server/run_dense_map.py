import os, sys, traceback, time
sys.path.insert(0, os.path.expanduser("~/diffusion/model_inference"))

prompt = sys.argv[1]
out_dir = sys.argv[2]   # absolute path under ~/diffusion/model_inference/generated/dense_<slug>
os.makedirs(out_dir, exist_ok=True)

t0 = time.time()
print(f"[dense] prompt={prompt!r} out={out_dir}", flush=True)
try:
    from explore_map.precompute import precompute_radial_map
    mp = precompute_radial_map(
        prompt,
        model="krea2",
        num_particles=12,
        num_clones=3,
        num_resampling_steps=3,
        num_steps=20,
        height=512,
        width=512,
        augment=False,
        seed=7,
        output_dir=out_dir,
    )
    print(f"[dense] DONE manifest={mp} elapsed={time.time()-t0:.0f}s", flush=True)
except Exception:
    print("[dense] ERROR", flush=True)
    traceback.print_exc()
    sys.exit(1)
