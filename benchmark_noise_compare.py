"""
benchmark_noise_compare.py

Measure the impact of Gaussian noise on BRUTE-FORCE retrieval using Euclidean vs Cosine.

Dataset:
  DATASET_ROOT/
    gallery/   <label>/*.jpg
    queries/   <label>/*.jpg

Typical run (from project root):
  PYTHONPATH=. python benchmark_noise_compare.py \
    --dataset-root storage/mi_eval \
    --k 5 \
    --model vggface2 --device auto --dim 512 \
    --noise-stds 0,5,10,15,20 \
    --noise-where queries \
    --seed 42 \
    --out-json storage/mi_eval/noise_compare.json \
    --out-dir  storage/mi_eval/noise_plots

Notes
-----
- Gaussian noise is applied in **pixel space** (uint8 0..255) to PIL images before preprocessing.
- `--noise-where` controls where noise is injected: {queries, gallery, both}.
- For **Cosine**, embeddings are L2-normalized (gallery + query). For **Euclidean**, raw embeddings are used.
- If your model already outputs unit-norm embeddings, Euclidean and Cosine rankings can match closely.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Project imports
from ironclad.modules.extraction.embedding import Embedding
from ironclad.modules.extraction.preprocessing import Preprocessing


def list_images_with_labels(root_dir: Path) -> List[Tuple[Path, str]]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    items = []
    if not root_dir.exists():
        return items
    for label_dir in sorted(root_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for p in sorted(label_dir.rglob('*')):
            if p.is_file() and p.suffix.lower() in exts:
                items.append((p, label))
    return items


def add_gaussian_noise_rgb(pil_img: Image.Image, std: float, rng: np.random.Generator) -> Image.Image:
    if std <= 0:
        return pil_img
    arr = np.array(pil_img).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=std, size=arr.shape).astype(np.float32)
    arr_noisy = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr_noisy)


def precision_at_k(rels: List[int], k: int) -> float:
    return float(sum(rels[:k])) / float(max(1, k))


def average_precision_at_k(rels: List[int], k: int, R: int) -> float:
    if R <= 0:
        return 0.0
    denom = float(min(R, k))
    hits = 0
    ap_sum = 0.0
    for i in range(1, k + 1):
        if rels[i - 1] == 1:
            hits += 1
            ap_sum += (hits / i)
    return ap_sum / denom


def l2_topk(G: np.ndarray, G_sq: np.ndarray, q: np.ndarray, k: int):
    q_sq = float(np.dot(q, q))
    dot = G @ q
    dists = G_sq + q_sq - 2.0 * dot
    k = min(k, dists.shape[0])
    if k <= 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    idx = np.argpartition(dists, k - 1)[:k]
    order = np.argsort(dists[idx], kind="mergesort")
    idx_sorted = idx[order]
    return dists[idx_sorted], idx_sorted.astype(np.int64)


def cosine_topk(Gn: np.ndarray, qn: np.ndarray, k: int):
    sims = Gn @ qn
    dists = 1.0 - sims
    k = min(k, dists.shape[0])
    if k <= 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    idx = np.argpartition(dists, k - 1)[:k]
    order = np.argsort(dists[idx], kind="mergesort")
    idx_sorted = idx[order]
    return dists[idx_sorted], idx_sorted.astype(np.int64)


def eval_variant(
    variant_name: str,
    gallery_emb: np.ndarray,
    gallery_emb_norm: np.ndarray,
    gallery_labels: List[str],
    gallery_label_counts: Dict[str, int],
    queries: List[Tuple[str, np.ndarray, np.ndarray, str]],
    k: int,
) -> Dict[str, float]:
    G_sq = np.sum(gallery_emb * gallery_emb, axis=1)
    top1_hits = 0
    recall_at_k = 0
    ranks = []
    precisions = []
    apks = []

    for (_, z_raw, z_norm, gt) in queries:
        R = gallery_label_counts.get(gt, 0)

        if variant_name == "euclidean":
            _, idx = l2_topk(gallery_emb, G_sq, z_raw, k)
        else:
            _, idx = cosine_topk(gallery_emb_norm, z_norm, k)

        metas = [gallery_labels[i] for i in idx]
        rels = [1 if (m == gt) else 0 for m in metas[:k]] + [0] * max(0, k - len(metas))

        rank = 0
        for j, r in enumerate(rels, start=1):
            if r == 1:
                rank = j
                break
        ranks.append(rank)
        if rank == 1:
            top1_hits += 1
        if 1 <= rank <= k:
            recall_at_k += 1

        precisions.append(precision_at_k(rels, k))
        apks.append(average_precision_at_k(rels, k, R))

    nQ = max(1, len(queries))
    mrr = float(np.mean([1.0/r if r > 0 else 0.0 for r in ranks]))
    mean_prec_k = float(np.mean(precisions)) if precisions else 0.0
    mean_ap_k = float(np.mean(apks)) if apks else 0.0

    return {
        "top1_acc": float(top1_hits) / nQ,
        "recall_at_k": float(recall_at_k) / nQ,
        "mrr": mrr,
        "precision_at_k": mean_prec_k,
        "average_precision_at_k": mean_ap_k,
        "mean_average_precision_at_k": mean_ap_k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", type=str, default="vggface2")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--noise-stds", type=str, default="0,5,10,15,20",
                    help="Comma-separated std in pixel space (0..255), e.g., '0,5,10'")
    ap.add_argument("--noise-where", type=str, default="queries", choices=["queries", "gallery", "both"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None, help="Directory to save per-metric charts")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    std_list = [float(s) for s in args.noise_stds.split(",") if s.strip() != ""]

    dataset_root = Path(args.dataset_root)
    gallery_dir = dataset_root / "gallery"
    query_dir = dataset_root / "queries"

    gallery_items = list_images_with_labels(gallery_dir)
    query_items = list_images_with_labels(query_dir)

    if not gallery_items or not query_items:
        print(f"[ERROR] No images found. Expected structure under {dataset_root}/(gallery|queries)/<label>/*.jpg", file=sys.stderr)
        sys.exit(2)

    # Map label -> count in gallery
    gallery_label_counts: Dict[str, int] = {}
    for _, lbl in gallery_items:
        gallery_label_counts[lbl] = gallery_label_counts.get(lbl, 0) + 1

    # Device resolution
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    pre = Preprocessing()
    model = Embedding(pretrained=args.model, device=device)

    # Baseline clean gallery embeddings (we will overwrite per-std if noise applied to gallery)
    def build_gallery(std: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        G_labels, G_vecs = [], []
        for p, label in gallery_items:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            if args.noise_where in ("gallery", "both") and std > 0:
                img = add_gaussian_noise_rgb(img, std, rng)
            x = pre.process(img)
            z = model.encode(x).astype(np.float32)
            G_labels.append(label)
            G_vecs.append(z)
        if not G_vecs:
            print("[ERROR] No gallery embeddings computed.", file=sys.stderr)
            sys.exit(2)
        G = np.stack(G_vecs, axis=0)
        Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
        return G, Gn, G_labels

    # Precompute clean gallery if queries-only noise
    cache_gallery = {}
    if args.noise_where == "queries":
        G, Gn, G_labels = build_gallery(0.0)
        cache_gallery[0.0] = (G, Gn, G_labels)

    results = []
    for std in std_list:
        # Build gallery for this std if needed
        if args.noise_where == "queries":
            G, Gn, G_labels = cache_gallery[0.0]
        else:
            G, Gn, G_labels = build_gallery(std)

        # Build queries for this std
        queries = []
        for p, gt in query_items:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            if args.noise_where in ("queries", "both") and std > 0:
                img = add_gaussian_noise_rgb(img, std, rng)
            x = pre.process(img)
            z = model.encode(x).astype(np.float32)
            z_norm = z / (np.linalg.norm(z) + 1e-12)
            queries.append((str(p), z, z_norm, gt))

        eu = eval_variant("euclidean", G, Gn, G_labels, gallery_label_counts, queries, args.k)
        co = eval_variant("cosine",    G, Gn, G_labels, gallery_label_counts, queries, args.k)

        results.append({
            "std": std,
            "euclidean": eu,
            "cosine": co
        })

    report = {
        "config": {
            "distance_variants": ["euclidean", "cosine"],
            "k": args.k,
            "dim": args.dim,
            "model": args.model,
            "device": device,
            "dataset_root": str(dataset_root),
            "noise_where": args.noise_where,
            "noise_stds": std_list,
            "seed": args.seed,
        },
        "sizes": {
            "gallery": len(gallery_items),
            "queries": len(query_items),
        },
        "results": results,
        "note": "Gaussian noise std in pixel units (0..255). Cosine uses L2-normalized embeddings; Euclidean uses raw."
    }

    print(json.dumps(report, indent=2))

    # Markdown table (rows = std, columns = metrics per distance)
    metrics = ["top1_acc", "recall_at_k", "mrr", "precision_at_k", "average_precision_at_k", "mean_average_precision_at_k"]
    header = ["Std"] + [f"EU-{m}" for m in metrics] + [f"CO-{m}" for m in metrics]
    md = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in results:
        std = row["std"]
        eu = row["euclidean"]
        co = row["cosine"]
        fields = [f"{std:g}"] + [f"{eu[m]:.4f}" for m in metrics] + [f"{co[m]:.4f}" for m in metrics]
        md.append("| " + " | ".join(fields) + " |")

    print("\n# Markdown Table (Noise Impact)\n")
    print("\n".join(md))

    # Optional: save charts, one plot per metric (vs std), both variants
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        xs = [r["std"] for r in results]
        for m in metrics:
            eu_vals = [r["euclidean"][m] for r in results]
            co_vals = [r["cosine"][m] for r in results]

            plt.figure(figsize=(8, 4.5))
            plt.plot(xs, eu_vals, marker="o", label="euclidean")
            plt.plot(xs, co_vals, marker="o", label="cosine")
            plt.xlabel("Gaussian noise std (pixel units)")
            plt.ylabel(m)
            plt.title(f"{m} vs noise (k={args.k})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{m}_vs_noise.png", dpi=150)
            plt.close()

if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)