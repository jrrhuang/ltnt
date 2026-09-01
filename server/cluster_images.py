"""Cluster generated images using DINOv2, DINOv3, and CLIP embeddings.

Uses UMAP for 2D projection and renders image thumbnails in embedding space.
"""

import os

# Set HF cache to scratch BEFORE importing transformers
SCRATCH_HF_HOME = "/scratch/jerryhua/.cache/huggingface"
if os.path.exists("/scratch/jerryhua"):
    os.makedirs(SCRATCH_HF_HOME, exist_ok=True)
    os.environ.setdefault("HF_HOME", SCRATCH_HF_HOME)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(SCRATCH_HF_HOME, "hub"))

import argparse
import gc
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from PIL import Image


def load_images(image_dir):
    """Load all PNG images from directory (excluding generated plots/grids)."""
    all_pngs = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    skip = {"grid.png", "similarity_dinov2_b.png", "similarity_dinov3_b.png",
            "similarity_clip_l.png", "projection_2d.png", "projection_images.png",
            "cluster_dinov2_b.png", "cluster_dinov3_b.png", "cluster_clip_l.png"}
    all_pngs = [f for f in all_pngs if os.path.basename(f) not in skip]

    if not all_pngs:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    images = []
    labels = []
    for f in all_pngs:
        img = Image.open(f).convert("RGB")
        images.append(img)
        labels.append(os.path.splitext(os.path.basename(f))[0])

    print(f"Loaded {len(images)} images from {image_dir}")
    return images, labels


def l2_normalize(features):
    """L2-normalize each row."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / (norms + 1e-8)


def extract_dinov2(images, device="cuda"):
    """Extract CLS token embeddings from DINOv2-B."""
    from transformers import AutoImageProcessor, AutoModel

    print("\n--- DINOv2-B ---")
    model_id = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    embeddings = []
    with torch.no_grad():
        for i, img in enumerate(images):
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_token.cpu().numpy())
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(images)}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return l2_normalize(np.concatenate(embeddings, axis=0))


def extract_dinov3(images, device="cuda"):
    """Extract pooler_output embeddings from DINOv3-B."""
    from transformers import AutoImageProcessor, AutoModel

    print("\n--- DINOv3-B ---")
    model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    embeddings = []
    with torch.no_grad():
        for i, img in enumerate(images):
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
            emb = outputs.pooler_output
            embeddings.append(emb.cpu().numpy())
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(images)}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return l2_normalize(np.concatenate(embeddings, axis=0))


def extract_clip(images, device="cuda"):
    """Extract image embeddings from CLIP-L."""
    from transformers import CLIPModel, CLIPProcessor

    print("\n--- CLIP-L ---")
    model_id = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()

    embeddings = []
    with torch.no_grad():
        for i, img in enumerate(images):
            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            out = model.get_image_features(pixel_values=pixel_values)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            embeddings.append(emb.cpu().numpy())
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(images)}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return l2_normalize(np.concatenate(embeddings, axis=0))


def _plot_thumbnails_2d(ax, coords, images, thumb_size):
    """Place image thumbnails at 2D coordinates on an axes."""
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    hw = max(min(x_range, y_range) * 0.04, 1e-6)

    for i, (x, y) in enumerate(coords):
        thumb = images[i].resize((thumb_size, thumb_size), Image.LANCZOS)
        ax.imshow(np.array(thumb),
                  extent=(x - hw, x + hw, y - hw, y + hw),
                  zorder=2)

    pad = hw * 3
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_image_cluster(embeddings, images, title, save_path,
                       n_neighbors=15, min_dist=0.1, thumb_size=40):
    """Plot image cluster map.

    If < 16 images: side-by-side MDS (left) + cosine similarity heatmap (right).
    If >= 16 images: UMAP only.
    """
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_similarity

    n_samples = embeddings.shape[0]

    if n_samples >= 16:
        # UMAP only
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            metric="cosine", random_state=42)
        coords = reducer.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(14, 14))
        _plot_thumbnails_2d(ax, coords, images, thumb_size)
        ax.set_title(f"{title} (UMAP)", fontsize=18)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        # MDS + cosine similarity heatmap
        cos_sim = cosine_similarity(embeddings)

        # MDS from cosine distance
        cos_dist = 1.0 - cos_sim
        np.fill_diagonal(cos_dist, 0.0)
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                  normalized_stress="auto")
        coords = mds.fit_transform(cos_dist)

        fig, axes = plt.subplots(1, 2, figsize=(24, 12))

        # Left: MDS with thumbnails
        _plot_thumbnails_2d(axes[0], coords, images, thumb_size)
        axes[0].set_title(f"{title} (MDS)", fontsize=16)

        # Right: cosine similarity heatmap with thumbnails on axes
        im = axes[1].imshow(cos_sim, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
        axes[1].set_title(f"Cosine Similarity", fontsize=16)

        # Add thumbnail labels on axes
        thumb_ax_size = max(thumb_size // 2, 20)
        for i in range(n_samples):
            thumb = images[i].resize((thumb_ax_size, thumb_ax_size), Image.LANCZOS)
            thumb_arr = np.array(thumb)
            # Place on x-axis (top)
            axes[1].imshow(thumb_arr,
                           extent=(i - 0.4, i + 0.4, -1.8, -0.6),
                           clip_on=False, zorder=5)
            # Place on y-axis (left)
            axes[1].imshow(thumb_arr,
                           extent=(-1.8, -0.6, n_samples - 1 - i - 0.4, n_samples - 1 - i + 0.4),
                           clip_on=False, zorder=5)

        # Annotate similarity values
        for i in range(n_samples):
            for j in range(n_samples):
                axes[1].text(j, i, f"{cos_sim[i, j]:.2f}",
                             ha="center", va="center", fontsize=8,
                             color="black" if cos_sim[i, j] > 0.3 else "white")

        axes[1].set_xlim(-2, n_samples)
        axes[1].set_ylim(n_samples, -2)
        axes[1].set_xticks(range(n_samples))
        axes[1].set_yticks(range(n_samples))
        axes[1].set_xticklabels([str(i) for i in range(n_samples)], fontsize=10)
        axes[1].set_yticklabels([str(i) for i in range(n_samples)], fontsize=10)
        fig.colorbar(im, ax=axes[1], shrink=0.6)

        fig.suptitle(title, fontsize=18, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Cluster images with vision encoder embeddings")
    parser.add_argument("--image_dir", required=True, help="Directory with PNG images")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--thumb_size", type=int, default=40, help="Thumbnail size in pixels")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    args = parser.parse_args()

    images, labels = load_images(args.image_dir)

    extractors = [
        ("DINOv2-B", extract_dinov2, "cluster_dinov2_b.png"),
        ("DINOv3-B", extract_dinov3, "cluster_dinov3_b.png"),
        ("CLIP-L", extract_clip, "cluster_clip_l.png"),
    ]

    for model_name, extract_fn, filename in extractors:
        emb = extract_fn(images, args.device)
        plot_image_cluster(
            emb, images,
            title=f"Image Clusters — {model_name} (UMAP)",
            save_path=os.path.join(args.image_dir, filename),
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            thumb_size=args.thumb_size,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
