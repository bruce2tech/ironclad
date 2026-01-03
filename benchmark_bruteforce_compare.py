"""
benchmark_bruteforce_compare.py

Compare BRUTE-FORCE retrieval using Euclidean distance vs Cosine distance on your dataset.

Dataset directory must look like:
  DATASET_ROOT/
    gallery/
      Alice/ img1.jpg img2.jpg ...
      Bob/   ...
    queries/
      Alice/ q1.jpg ...
      Bob/   ...

Run (from project root):
  PYTHONPATH=. python benchmark_bruteforce_compare.py \
    --dataset-root storage/mi_eval \
    --k 5 \
    --model vggface2 --device auto --dim 512 \
    --per-query \
    --out storage/mi_eval/bruteforce_compare.json
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

# Project imports (run with PYTHONPATH=. from project root)
from ironclad.modules.extraction.embedding import Embedding
from ironclad.modules.extraction.preprocessing import Preprocessing


def list_images_with_labels(root_dir: Path) -> List[Tuple[Path, str]]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    items = []
    if not root_dir.exists():
        return items
    for label_dir in root_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for p in label_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                items.append((p, label))
    return items


def now() -> float:
    return time.perf_counter()


def summarize_times(times: List[float]) -> Dict[str, float]:
    if not times:
        return {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    arr = np.array(times) * 1e3
    return {
        "count": int(len(times)),
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def precision_at_k(rels: List[int], k: int) -> float:
    return float(sum(rels[:k])) / float(max(1, k))


def average_precision_at_k(rels: List[int], k: int, R: int) -> float:
    """
    AP@K = (1 / min(R, K)) * sum_{i=1..K} [ P@i * rel_i ]
    If R == 0, AP@K = 0.0.
    """
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


def l2_topk(G: np.ndarray, G_sq: np.ndarray, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Squared L2 distances: top-k indices and distances.
    """
    q_sq = float(np.dot(q, q))
    dot = G @ q  # (N,)
    dists = G_sq + q_sq - 2.0 * dot  # (N,)
    k = min(k, dists.shape[0])
    if k <= 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    idx = np.argpartition(dists, k - 1)[:k]
    order = np.argsort(dists[idx], kind="mergesort")
    idx_sorted = idx[order]
    return dists[idx_sorted], idx_sorted.astype(np.int64)


def cosine_topk(Gn: np.ndarray, qn: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine distance = 1 - cosine similarity.
    """
    sims = Gn @ qn  # (N,)
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
    """
    Compute aggregate metrics for one variant ('euclidean' or 'cosine').
    Returns dict of metrics and search timing summary.
    """
    G_sq = np.sum(gallery_emb * gallery_emb, axis=1)  # for euclidean

    top1_hits = 0
    recall_at_k = 0
    ranks: List[int] = []
    precisions: List[float] = []
    apks: List[float] = []
    t_search: List[float] = []

    for (_, z_raw, z_norm, gt) in queries:
        R = gallery_label_counts.get(gt, 0)

        t0 = now()
        if variant_name == "euclidean":
            _, idx = l2_topk(gallery_emb, G_sq, z_raw, k)
        else:
            _, idx = cosine_topk(gallery_emb_norm, z_norm, k)
        t1 = now()
        t_search.append(t1 - t0)

        metas = [gallery_labels[i] for i in idx]
        rels = [1 if (m == gt) else 0 for m in metas[:k]] + [0] * max(0, k - len(metas))

        # rank of first relevant (for mrr, top1, recall@k)
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
        "search_timings_ms": summarize_times(t_search)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True, help="ROOT with gallery/ and queries/ subfolders")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", type=str, default="vggface2")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dim", type=int, default=512, help="Embedding dimension (must match your model)")
    ap.add_argument("--out-json", type=str, default=None, help="Optional path to save JSON report")
    ap.add_argument("--out-fig", type=str, default=None, help="Optional path to save comparison bar chart (.png)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    gallery_dir = dataset_root / "gallery"
    query_dir = dataset_root / "queries"

    gallery_items = list_images_with_labels(gallery_dir)
    query_items = list_images_with_labels(query_dir)

    if not gallery_items or not query_items:
        print(f"[ERROR] No images found. Expected structure under {dataset_root}/(gallery|queries)/<label>/*.jpg", file=sys.stderr)
        sys.exit(2)

    # Count relevant per label in gallery (for AP@K)
    gallery_label_counts: Dict[str, int] = {}
    for _, lbl in gallery_items:
        gallery_label_counts[lbl] = gallery_label_counts.get(lbl, 0) + 1

    # Resolve device
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    pre = Preprocessing()
    model = Embedding(pretrained=args.model, device=device)

    # --- Build gallery embeddings ---
    G_labels: List[str] = []
    G_vecs = []
    t_g_pre, t_g_emb = [], []

    for p, label in gallery_items:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        t0 = now(); x = pre.process(img); t1 = now()
        z = model.encode(x).astype(np.float32); t2 = now()
        t_g_pre.append(t1 - t0); t_g_emb.append(t2 - t1)
        G_vecs.append(z)
        G_labels.append(label)

    if not G_vecs:
        print("[ERROR] No gallery embeddings computed.", file=sys.stderr)
        sys.exit(2)

    G = np.stack(G_vecs, axis=0)  # (N,D)
    G_norm = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)

    # --- Build query embeddings ---
    queries = []
    t_q_pre, t_q_emb = [], []
    for p, gt in query_items:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        t0 = now(); x = pre.process(img); t1 = now()
        z = model.encode(x).astype(np.float32); t2 = now()
        t_q_pre.append(t1 - t0); t_q_emb.append(t2 - t1)
        z_norm = z / (np.linalg.norm(z) + 1e-12)
        queries.append((str(p), z, z_norm, gt))

    # --- Evaluate both variants ---
    eu_metrics = eval_variant("euclidean", G, G_norm, G_labels, gallery_label_counts, queries, args.k)
    co_metrics = eval_variant("cosine",    G, G_norm, G_labels, gallery_label_counts, queries, args.k)

    report = {
        "config": {
            "distance_variants": ["euclidean", "cosine"],
            "k": args.k,
            "dim": args.dim,
            "model": args.model,
            "device": device,
            "dataset_root": str(dataset_root),
        },
        "sizes": {
            "gallery": int(G.shape[0]),
            "queries": int(len(queries)),
        },
        "gallery_timings_ms": {
            "preprocess": summarize_times(t_g_pre),
            "embed": summarize_times(t_g_emb),
        },
        "query_timings_ms": {
            "preprocess": summarize_times(t_q_pre),
            "embed": summarize_times(t_q_emb),
        },
        "euclidean": eu_metrics,
        "cosine":    co_metrics,
        "notes": "Cosine uses L2-normalized embeddings; Euclidean uses raw embeddings. Both are brute-force."
    }

    print(json.dumps(report, indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)

    if args.out_fig:
        # Build a grouped bar chart comparing both variants on the requested metrics.
        metrics = [
            "top1_acc",
            "recall_at_k",
            "mrr",
            "precision_at_k",
            "average_precision_at_k",
            "mean_average_precision_at_k",
        ]

        eu_vals = [eu_metrics[m] for m in metrics]
        co_vals = [co_metrics[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width/2, eu_vals, width, label="euclidean")
        plt.bar(x + width/2, co_vals, width, label="cosine")
        plt.xticks(x, metrics, rotation=30, ha="right")
        plt.ylabel("score")
        plt.title(f"Brute-force comparison (k={args.k})")
        plt.legend()
        plt.tight_layout()
        Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_fig, dpi=150)
        plt.close()

if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)