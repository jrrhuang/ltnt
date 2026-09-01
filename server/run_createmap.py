#!/usr/bin/env python3
import os, sys, json, argparse, time
os.environ.setdefault("HF_HOME", "/data/user_data/jerryhua/hfcache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data/user_data/jerryhua/hfcache/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data/user_data/jerryhua/hfcache/hub")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flux_explore_precompute import precompute_explore_map
ap = argparse.ArgumentParser()
ap.add_argument("--prompt", required=True); ap.add_argument("--model", default="krea2")
ap.add_argument("--out", required=True); ap.add_argument("--R", type=int, default=4)
ap.add_argument("--steps", type=int, default=20); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--height", type=int, default=512); ap.add_argument("--width", type=int, default=512); ap.add_argument("--particles", type=int, default=6)
a = ap.parse_args(); t0=time.time()
mp = precompute_explore_map(prompt=a.prompt, model=a.model, num_particles=a.particles, num_clones=3,
    augment=(a.model=="flux"), num_resampling_steps=a.R, num_steps=a.steps,
    height=a.height, width=a.width, seed=a.seed, output_dir=a.out)
m=json.load(open(mp)); print("DONE total_images="+str(m["total_images"])+" time="+str(int(time.time()-t0))+"s out="+a.out)
