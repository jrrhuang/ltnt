"""Refine-calibration experiment for LTNT stage-adaptive refinement.

Measures how much a "refined" image diverges from the preview the artist saw,
as a function of pick stage t', for three completion strategies:

  A. PREVIEW           : coarse Euler lookahead x_{t'} -> 0, 6 steps (app preview)
  B. NAIVE-REFINE      : fine solve from x_{t'} over the remaining fine schedule
                         (deterministic Euler => identical to the full fine solve,
                         so B is the same image at every stage)
  C. PATH-FAITHFUL     : continue the coarse (6-step-spaced, preview-identical)
                         path from x_{t'} down to t''=0.25, then fine solve below
  D. POSTERIOR-MEAN    : x_hat = x - t' * v(x, t'), decoded (honest preview)

Metrics: LPIPS(alex) for (A,B), (A,C), (B,C), (D,A) per stage, mean+/-std over seeds.
Outputs: results.json + one montage per stage (seed 0: A|B|C|D) in
/home/jerryhua/diffusion/model_inference/outputs/refine_calibration/
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = "/home/jerryhua/diffusion/model_inference"
sys.path.insert(0, SCRIPT_DIR)

OUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "refine_calibration")
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT = "bioluminescent deep-sea creatures in a dark ocean, cinematic"
N_SEEDS = 8
NUM_STEPS = 20            # fine schedule
PREVIEW_STEPS = 6         # app's coarse lookahead
T_HANDOFF = 0.25          # path-faithful switch point (in shifted-sigma time)
TARGET_STAGES = [0.9, 0.8, 0.7, 0.6, 0.45, 0.3, 0.15]
GUIDANCE = 4.0
IMG = 1024

from sana_fmtt import SanaInteractive  # noqa: E402


def build_fine_schedule(sampler, num_steps):
    """Same construction as SanaInteractive.sample_interactive."""
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    shift = float(getattr(sampler.pipeline.scheduler.config, "shift", 1.0) or 1.0)
    if shift <= 1.0:
        shift = 3.0
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    sampler._flow_shift = shift
    return np.concatenate([sigmas, [0.0]]), shift


def euler_path(sampler, z, pts):
    """Euler-integrate along the given time points (list high->low)."""
    z_cur = z.clone()
    for i in range(len(pts) - 1):
        t, t_next = float(pts[i]), float(pts[i + 1])
        v = sampler.predict_velocity(z_cur, t, GUIDANCE)
        z_cur = z_cur + (t_next - t) * v
    return z_cur


def lookahead_pts(shift, t_cur, num_steps):
    """Same grid euler_lookahead uses: uniform in naive sigma, re-shifted."""
    inv = t_cur / max(shift - (shift - 1.0) * t_cur, 1e-8)
    naive = np.linspace(inv, 0.0, num_steps + 1)
    return shift * naive / (1.0 + (shift - 1.0) * naive)


def to_pil(img_t):
    arr = (img_t[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def main():
    # CRITICAL: no autograd anywhere — predict_velocity is a raw transformer
    # forward; without this the Euler loop accumulates a 40GB+ graph and OOMs
    # (cause of the job-8900806 failure).
    torch.set_grad_enabled(False)
    device = "cuda"
    sampler = SanaInteractive(PROMPT, device=device)
    fine_pts, shift = build_fine_schedule(sampler, NUM_STEPS)
    print(f"[calib] shift={shift} fine_pts={np.round(fine_pts, 4).tolist()}")

    # Map each target stage to the nearest fine-schedule index (snapshot AFTER
    # integrating to that point, i.e. state at time fine_pts[k]).
    stage_idx = []
    for tgt in TARGET_STAGES:
        k = int(np.argmin(np.abs(fine_pts[:-1] - tgt)))
        stage_idx.append(k)
    stages = [(k, float(fine_pts[k])) for k in stage_idx]
    print(f"[calib] stages (idx, t'): {stages}")

    # Fine points strictly below the handoff (for path-faithful tail).
    below = [float(p) for p in fine_pts if p < T_HANDOFF - 1e-6]
    print(f"[calib] path-faithful tail: {T_HANDOFF} -> {below}")

    import lpips
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    def lp(a, b):
        with torch.no_grad():
            return float(loss_fn(a.to(device) * 2 - 1, b.to(device) * 2 - 1).item())

    # results[t'][pair] = list over seeds
    results = {f"{t:.4f}": {p: [] for p in ["A_B", "A_C", "B_C", "D_A"]}
               for _, t in stages}
    montage_imgs = {}  # stage_t -> [A, B, C, D] PIL (seed 0)

    lh = IMG // sampler.vae_scale_factor
    for s in range(N_SEEDS):
        print(f"[calib] === seed {s} ===", flush=True)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        z = torch.randn(1, sampler.latent_channels, lh, lh,
                        device=device, dtype=sampler.dtype)

        # Full fine trajectory with snapshots.
        snaps = {}
        z_cur = z.clone()
        for k in range(NUM_STEPS):
            if k in stage_idx:
                snaps[k] = z_cur.clone()
            t, t_next = float(fine_pts[k]), float(fine_pts[k + 1])
            v = sampler.predict_velocity(z_cur, t, GUIDANCE)
            z_cur = z_cur + (t_next - t) * v
        with torch.no_grad():
            img_B = sampler.decode(z_cur).cpu()  # NAIVE-REFINE == full fine solve

        for k, t_stage in stages:
            z_snap = snaps[k]
            key = f"{t_stage:.4f}"

            # A: preview (6-step coarse lookahead, app code path)
            zA = sampler.euler_lookahead(z_snap, t_stage, GUIDANCE,
                                         num_steps=PREVIEW_STEPS)
            img_A = sampler.decode(zA).cpu()

            # C: follow the preview's coarse grid until it would cross the
            # handoff, step onto T_HANDOFF, then fine steps below.
            la = lookahead_pts(shift, t_stage, PREVIEW_STEPS)
            coarse_seg = [float(p) for p in la if p > T_HANDOFF + 1e-6]
            pts_C = coarse_seg + [T_HANDOFF] + below
            zC = euler_path(sampler, z_snap, pts_C)
            img_C = sampler.decode(zC).cpu()

            # D: posterior mean
            v = sampler.predict_velocity(z_snap, t_stage, GUIDANCE)
            zD = z_snap - t_stage * v
            img_D = sampler.decode(zD).cpu()

            results[key]["A_B"].append(lp(img_A, img_B))
            results[key]["A_C"].append(lp(img_A, img_C))
            results[key]["B_C"].append(lp(img_B, img_C))
            results[key]["D_A"].append(lp(img_D, img_A))

            if s == 0:
                montage_imgs[key] = [to_pil(x) for x in (img_A, img_B, img_C, img_D)]
            del zA, zC, zD, img_A, img_C, img_D
        del snaps
        torch.cuda.empty_cache()
        print(f"[calib] seed {s} done", flush=True)

    # Aggregate.
    table = {}
    for key, pairs in results.items():
        table[key] = {}
        for p, vals in pairs.items():
            table[key][p] = {"mean": float(np.mean(vals)),
                             "std": float(np.std(vals)),
                             "values": vals}
    out = {
        "prompt": PROMPT, "n_seeds": N_SEEDS, "num_steps": NUM_STEPS,
        "preview_steps": PREVIEW_STEPS, "t_handoff": T_HANDOFF,
        "guidance": GUIDANCE, "shift": shift,
        "stages": {f"{t:.4f}": k for k, t in stages},
        "metric": "LPIPS(alex)",
        "note_B": "NAIVE-REFINE is the deterministic continuation of the fine "
                  "schedule, hence identical to the full 20-step fine solve at "
                  "every stage.",
        "table": table,
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: {p: round(v["mean"], 4) for p, v in row.items()}
                      for k, row in table.items()}, indent=2))

    # Montages (seed 0): columns A|B|C|D, downscaled to 384.
    for key, imgs in montage_imgs.items():
        side = 384
        grid = Image.new("RGB", (side * 4, side + 24), "black")
        for i, im in enumerate(imgs):
            grid.paste(im.resize((side, side), Image.LANCZOS), (i * side, 24))
        try:
            from PIL import ImageDraw
            d = ImageDraw.Draw(grid)
            for i, lab in enumerate(["A preview", "B naive-refine",
                                     "C path-faithful", "D posterior-mean"]):
                d.text((i * side + 8, 4), f"{lab}  t'={key}", fill="white")
        except Exception:
            pass
        grid.save(os.path.join(OUT_DIR, f"montage_t{key}.png"))
    print("[calib] DONE")


if __name__ == "__main__":
    main()
