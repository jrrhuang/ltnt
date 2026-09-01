"""Self-labeling session board (Jerry 2026-08-28: every board must state
parent vs children, stage, noise level, and per-tile prompt).

Usage: python make_board.py <session_dir> [out.jpg]
Reads interval_*/meta.json when present (roles + prompts + spawn level);
degrades gracefully for old sessions (positional labels only).
"""
import glob
import json
import os
import sys

from PIL import Image, ImageDraw

TILE = 256
HDR = 34          # per-row header strip
CAP = 30          # per-tile caption strip


def _cap(draw, x, y, w, text, fill=(210, 210, 210)):
    words, line, lines = text.split(), "", []
    for wd in words:
        t = (line + " " + wd).strip()
        if draw.textlength(t) > w - 8 and line:
            lines.append(line)
            line = wd
        else:
            line = t
    lines.append(line)
    for i, ln in enumerate(lines[:2]):
        draw.text((x + 4, y + 2 + i * 13), ln, fill=fill)


def main(sdir, out=None):
    ivs = sorted(glob.glob(os.path.join(sdir, "interval_*")),
                 key=lambda p: int(p.rsplit("_", 1)[1]))
    rows = []
    for iv in ivs:
        files = sorted(glob.glob(os.path.join(iv, "particle_*.png")))
        meta = None
        mp = os.path.join(iv, "meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
        rows.append((iv, files, meta))
    ncol = max(len(f) for _, f, _ in rows)
    W = ncol * TILE
    H = sum(HDR + TILE + CAP for _ in rows)
    board = Image.new("RGB", (W, H), (10, 10, 12))
    d = ImageDraw.Draw(board)
    y = 0
    for iv, files, meta in rows:
        stage = int(iv.rsplit("_", 1)[1])
        if stage == 1:
            hdr = f"ROUND 1 - initial cast from pure noise (no parent)"
        elif meta and meta.get("spawn_t_RN"):
            hdr = (f"ROUND {stage} - spawned from the SELECTED parent; "
                   f"renoise level t={meta['spawn_t_RN']:.2f} "
                   f"({'global/structural' if meta['spawn_t_RN'] > 0.6 else 'local/detail'} band)")
        else:
            hdr = f"ROUND {stage} - spawned from the selected parent"
        d.rectangle([0, y, W, y + HDR], fill=(28, 28, 34))
        d.text((6, y + 9), hdr, fill=(255, 220, 120))
        y += HDR
        pm = {p["idx"]: p for p in (meta or {}).get("particles", [])}
        for i, f in enumerate(files):
            x = i * TILE
            board.paste(Image.open(f).resize((TILE, TILE)), (x, y))
            info = pm.get(i)
            if info:
                role = info["role"]
                if stage == 1:
                    label = f"root {i} - {info['prompt'][:60]}"
                elif role == "anchor":
                    label = "PARENT (your pick, continued)"
                else:
                    label = f"child - reading: {info['prompt'][:55]}"
            else:
                label = ("root" if stage == 1 else
                         ("PARENT (kept)" if i == 0 else "child"))
            col = (255, 190, 90) if "PARENT" in label else (200, 200, 205)
            d.rectangle([x, y + TILE, x + TILE, y + TILE + CAP],
                        fill=(16, 16, 20))
            _cap(d, x, y + TILE, TILE, label, fill=col)
            if "PARENT" in label:
                d.rectangle([x + 1, y + 1, x + TILE - 2, y + TILE - 2],
                            outline=(255, 190, 90), width=3)
        y += TILE + CAP
    out = out or os.path.join(sdir, "board_labeled.jpg")
    board.save(out, quality=90)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
