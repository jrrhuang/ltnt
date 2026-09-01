import os, sys, json, traceback, time
sys.path.insert(0, os.path.expanduser("~/diffusion/model_inference"))

prompt = sys.argv[1]
slug = sys.argv[2]
out_dir = os.path.join(os.environ["RIKA_DATA_DIR"], "scout_maps", slug)
os.makedirs(out_dir, exist_ok=True)

t0 = time.time()
print(f"[scout] prompt={prompt!r} slug={slug} out={out_dir}", flush=True)
try:
    from explore_map.precompute import precompute_radial_map
    mp = precompute_radial_map(
        prompt,
        model="krea2",
        num_particles=6,
        num_clones=3,
        num_resampling_steps=2,
        num_steps=20,
        height=512,
        width=512,
        augment=False,
        seed=7,
        output_dir=out_dir,
    )
    print(f"[scout] DONE manifest={mp} elapsed={time.time()-t0:.0f}s", flush=True)
except Exception:
    print("[scout] ERROR", flush=True)
    traceback.print_exc()
    sys.exit(1)
