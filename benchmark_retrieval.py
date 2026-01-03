"""
benchmark_retrieval.py

Benchmarks identification performance + latency for your Ironclad project.

It expects a labeled directory structure like:

  DATASET_ROOT/
    gallery/
      Alice/ img1.jpg img2.jpg ...
      Bob/   img3.jpg ...
      ...
    queries/
      Alice/ imgQ1.jpg imgQ2.jpg ...
      Bob/   imgQ3.jpg ...
      ...

For each query image, we compute the embedding, search top-K against the gallery index,
and measure accuracy/recall and latency.

It also has an optional synthetic speed benchmark (random vectors) if you don't
have a labeled dataset yet.

Usage (from project root):
  PYTHONPATH=. python benchmark_retrieval.py \
    --dataset-root /path/to/DATASET_ROOT \
    --index hnsw --metric cosine --k 5 \
    --M 32 --efC 40 --efS 64 \
    --model vggface2 --device auto

Synthetic benchmark only:
  PYTHONPATH=. python benchmark_retrieval.py --synthetic --N 20000 --D 512 --k 10 --index hnsw --metric cosine

Report is printed to stdout and also saved to JSON via --out.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Project imports (run with PYTHONPATH=. from project root)
from ironclad.modules.extraction.embedding import Embedding
from ironclad.modules.extraction.preprocessing import Preprocessing
from ironclad.modules.retrieval.index.hnsw import FaissHNSW
from ironclad.modules.retrieval.index.lsh import FaissLSH
from ironclad.modules.retrieval.search import FaissSearch


def list_images_with_labels(root_dir: Path) -> List[Tuple[Path, str]]:
    """Return [(path, label)] for images in structure <root>/<label>/*.jpg|png|jpeg"""
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


def build_index(index_name: str, dim: int, metric: str, args) -> object:
    index_name = index_name.lower()
    metric = metric.lower()
    if index_name == "hnsw":
        return FaissHNSW(dim=dim, metric=metric, M=args.M, efConstruction=args.efC, efSearch=args.efS)
    elif index_name == "lsh":
        return FaissLSH(dim=dim, nbits=args.nbits)
    else:
        raise ValueError(f"Unknown index type: {index_name}")


def device_string(arg: str) -> str:
    if arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return arg


def precision_at_k(rels: List[int], k: int) -> float:
    # rels: binary relevance list for the top-k list (len==k), 1 if item is relevant
    return float(sum(rels[:k])) / float(max(1, k))


def average_precision_at_k(rels: List[int], k: int, R: int) -> float:
    """
    AP@K = (1 / min(R, K)) * sum_{i=1..K} [ P@i * rel_i ]
    where:
      - rel_i is 1 if item at rank i is relevant, else 0
      - R is the number of relevant items in the entire gallery for this query's label
    If R == 0, AP@K is defined as 0.0.
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




def benchmark_dataset(args):
    dataset_root = Path(args.dataset_root)
    gallery_dir = dataset_root / "gallery"
    query_dir = dataset_root / "queries"

    gallery_items = list_images_with_labels(gallery_dir)
    query_items = list_images_with_labels(query_dir)

    if not gallery_items or not query_items:
        print(f"[ERROR] No images found. Expected structure under {dataset_root}/(gallery|queries)/<label>/*.jpg", file=sys.stderr)
        sys.exit(2)

    # Build a map of label -> count in gallery (for AP@K and better recall estimates)
    gallery_label_counts: Dict[str, int] = {}
    for _, lbl in gallery_items:
        gallery_label_counts[lbl] = gallery_label_counts.get(lbl, 0) + 1

    device = device_string(args.device)
    pre = Preprocessing()
    model = Embedding(pretrained=args.model, device=device)

    # Build index
    dim = args.dim
    index_impl = build_index(args.index, dim, args.metric, args)
    search = FaissSearch(index_impl, metric=args.metric)

    # --- Build gallery ---
    t_pre, t_emb, t_add = [], [], []
    n_gallery = 0
    for p, label in gallery_items:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        t0 = now()
        x = pre.process(img)
        t1 = now()
        z = model.encode(x)
        t2 = now()
        index_impl.add_embeddings([z], [label])
        t3 = now()

        t_pre.append(t1 - t0)
        t_emb.append(t2 - t1)
        t_add.append(t3 - t2)
        n_gallery += 1

    # --- Query ---
    top1 = 0
    recall_at_k = 0
    ranks: List[int] = []
    t_q_pre, t_q_emb, t_q_search = [], [], []

    precisions: List[float] = []
    apks: List[float] = []

    for p, gt_label in query_items:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue

        R = gallery_label_counts.get(gt_label, 0)  # number of relevant items in gallery

        t0 = now()
        x = pre.process(img)
        t1 = now()
        z = model.encode(x)
        t2 = now()
        D, I, M = search.search(z, k=args.k)
        t3 = now()

        # collect timings
        t_q_pre.append(t1 - t0)
        t_q_emb.append(t2 - t1)
        t_q_search.append(t3 - t2)

        # M is List[List[metadata]]; first row
        metas = M[0] if M is not None and len(M) > 0 else []
        # Compute rank of gt_label

        # binary relevance for top-k
        rels = [1 if (m == gt_label) else 0 for m in metas[:args.k]] + [0] * max(0, args.k - len(metas))
        
        rank = None
        for j, r in enumerate(rels, start=1):
            if r == 1:
                rank = j
                break

        ranks.append(rank if rank is not None else 0)
        if rank == 1:
            top1 += 1
        if rank is not None and rank <= args.k:
            recall_at_k += 1

                # --- new metrics ---
        precisions.append(precision_at_k(rels, args.k))
        apks.append(average_precision_at_k(rels, args.k, R))
        p_at_k = precision_at_k(rels, args.k)
        ap_at_k = average_precision_at_k(rels, args.k, R)
        precisions.append(p_at_k)
        apks.append(ap_at_k)

        if args.per_query:
            per_query.append({
                "query": str(p),
                "label": gt_label,
                "rank": rank if rank is not None else 0,
                "R_in_gallery": R,
                "precision_at_k": p_at_k,
                "ap_at_k": ap_at_k,
                "hits_in_top_k": int(sum(rels[:args.k])),
                "top_k_metas": metas[:args.k],
            })

    # mean(AP@K) is the same as mAP@K at the corpus level
    mean_ap_k = float(np.mean(apks)) if apks else 0.0    
    n_queries = len(query_items)
    report = {
        "config": {
            "index": args.index,
            "metric": args.metric,
            "k": args.k,
            "dim": dim,
            "M": args.M,
            "efConstruction": args.efC,
            "efSearch": args.efS,
            "nbits": args.nbits,
            "model": args.model,
            "device": device,
            "dataset_root": str(dataset_root),
        },
        "sizes": {
            "gallery": n_gallery,
            "queries": n_queries,
        },
        "accuracy": {
            "top1_acc": float(top1) / max(1, n_queries),
            "recall_at_k": float(recall_at_k) / max(1, n_queries),
            "mrr": float(np.mean([1.0/r if r and r > 0 else 0.0 for r in ranks])),
            "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "average_precision_at_k": mean_ap_k,            # <-- added explicit AP@K (mean over queries)
            "mean_average_precision_at_k": mean_ap_k        # <-- same value; kept for clarity
        },
        "timings_ms": {
            "gallery_preprocess": summarize_times(t_pre),
            "gallery_embed": summarize_times(t_emb),
            "gallery_add": summarize_times(t_add),
            "query_preprocess": summarize_times(t_q_pre),
            "query_embed": summarize_times(t_q_emb),
            "query_search": summarize_times(t_q_search),
        },
    }
    return report


def benchmark_synthetic(args):
    rng = np.random.default_rng(123)
    dim = args.D
    N = args.N
    K = args.k

    # random unit vectors to simulate cosine embeddings
    X = rng.standard_normal((N, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

    # half gallery, half queries
    half = N // 2
    Xg, Xq = X[:half], X[half:half + min(half, 5000)]  # cap queries for speed
    labels = [f"id_{i%1000}" for i in range(half)]  # many collisions intentionally

    index_impl = build_index(args.index, dim, args.metric, args)
    search = FaissSearch(index_impl, metric=args.metric)

    t_add = []
    for i in range(len(Xg)):
        t0 = now()
        index_impl.add_embeddings([Xg[i]], [labels[i]])
        t1 = now()
        t_add.append(t1 - t0)

    t_search = []
    for i in range(len(Xq)):
        t0 = now()
        D, I, M = search.search(Xq[i], k=K)
        t1 = now()
        t_search.append(t1 - t0)

    return {
        "config": {
            "synthetic": True,
            "N": N,
            "D": dim,
            "k": K,
            "index": args.index,
            "metric": args.metric,
            "M": args.M,
            "efConstruction": args.efC,
            "efSearch": args.efS,
            "nbits": args.nbits,
        },
        "sizes": {"gallery": len(Xg), "queries": len(Xq)},
        "timings_ms": {
            "add": summarize_times(t_add),
            "search": summarize_times(t_search),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default=None, help="ROOT with gallery/ and queries/ subfolders")
    ap.add_argument("--index", type=str, default="hnsw", choices=["hnsw", "lsh"])
    ap.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean", "dot_product"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", type=str, default="vggface2")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dim", type=int, default=512, help="Embedding dimension (must match your model)")
    # HNSW params
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efC", type=int, default=40, help="efConstruction")
    ap.add_argument("--efS", type=int, default=64, help="efSearch")
    # LSH params
    ap.add_argument("--nbits", type=int, default=128)
    # Synthetic
    ap.add_argument("--synthetic", action="store_true", help="Run synthetic speed benchmark instead of dataset benchmark")
    ap.add_argument("--N", type=int, default=10000, help="Total synthetic vectors (half gallery, half queries<=5000)")
    ap.add_argument("--D", type=int, default=512, help="Dim for synthetic vectors")
    ap.add_argument("--out", type=str, default=None, help="Optional path to save JSON report")
    ap.add_argument("--per-query", action="store_true", help="Include per-query metrics in the JSON output")

    args = ap.parse_args()

    if args.synthetic and args.dataset_root:
        print("[WARN] --synthetic ignores --dataset-root", file=sys.stderr)

    if args.synthetic:
        report = benchmark_synthetic(args)
    else:
        if not args.dataset_root:
            print("[ERROR] Provide --dataset-root or use --synthetic", file=sys.stderr)
            sys.exit(2)
        report = benchmark_dataset(args)

    # Print and optionally save
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
# Path('/mnt/data/benchmark_retrieval.py').write_text(updated)
# print("Updated /mnt/data/benchmark_retrieval.py with Precision@K, AP@K, and mAP@K.")

if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)
    
'''
with open('/mnt/data/benchmark_retrieval.py', 'w') as f:
    f.write(script)

print("Wrote benchmark script to /mnt/data/benchmark_retrieval.py")
'''