import os, sys
os.environ.setdefault("HF_HOME", "/data/user_data/jerryhua/hfcache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"] + "/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"] + "/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "/home/jerryhua/diffusion/model_inference")
import prompt_augment

CASES = [
    ("a village below mount fuji, ukiyo-e woodblock print", ["ukiyo-e", "woodblock"]),
    ("portrait of an old fisherman, oil painting on canvas", ["oil painting"]),
    ("a cyberpunk street market in the rain, pixel art", ["pixel art"]),
]
all_ok = True
for prompt, markers in CASES:
    ticks = []
    vs = prompt_augment.augment_prompt(prompt, n=4, device="cuda",
                                       token_cb=lambda d, t: ticks.append(d))
    print(f"\n=== {prompt!r}  (token ticks fired: {len(ticks)}, last={ticks[-1] if ticks else 0})")
    for i, v in enumerate(vs, 1):
        missing = [m for m in markers if m not in v.lower()]
        status = "OK " if not missing else f"MISSING {missing}"
        if missing:
            all_ok = False
        print(f"  {i}. [{status}] {v}")
print("\nRESULT:", "ALL STYLE MARKERS PRESERVED" if all_ok else "FAILURES ABOVE")
