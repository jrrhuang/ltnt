"""Method taste-board: same parent, one row per spawn method/config.
Usage: python method_board.py out.jpg  label1:session1  label2:session2 ...
Row 0 = the shared parent (from the first session's interval_1 root 0).
Each later row = that session's interval_2 children (anchor dropped).
"""
import glob
import os
import sys

from PIL import Image, ImageDraw

TILE = 232
HDR = 30
GEN = "/home/jerryhua/diffusion/model_inference/generated"


def row_files(sid):
    fs = sorted(glob.glob(f"{GEN}/{sid}/interval_2/particle_*.png"))
    return fs[1:]           # drop the anchor (parent continuation)


def main(out, pairs):
    labels = [p.split(":", 1)[0] for p in pairs]
    sids = [p.split(":", 1)[1] for p in pairs]
    parent = sorted(glob.glob(f"{GEN}/{sids[0]}/interval_1/particle_*.png"))[0]
    rows = [("PARENT (shared, seed 7777)", [parent])] + \
           [(lab, row_files(sid)) for lab, sid in zip(labels, sids)]
    ncol = max(len(f) for _, f in rows)
    W, H = ncol * TILE, sum(HDR + TILE for _ in rows)
    board = Image.new("RGB", (W, H), (10, 10, 12))
    d = ImageDraw.Draw(board)
    y = 0
    for lab, fs in rows:
        d.rectangle([0, y, W, y + HDR], fill=(28, 28, 34))
        d.text((6, y + 8), lab, fill=(255, 220, 120))
        y += HDR
        for i, f in enumerate(fs):
            board.paste(Image.open(f).resize((TILE, TILE)), (i * TILE, y))
        y += TILE
    board.save(out, quality=90)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
