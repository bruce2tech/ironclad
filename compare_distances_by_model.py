"""
compare_distances_by_model.py

Compare distance measures (euclidean, cosine, dot_product, minkowski Lp)
on the same evaluation dataset using two FaceNet variants:
  - pretrained='vggface2'
  - pretrained='casia-webface'

Metrics:
  - mean Average Precision at K (mAP@K)
  - Recall@K
  - Baseline contrast: Mean Reciprocal Rank (MRR) for Euclidean

Exact brute-force (NumPy) — no FAISS index dependence.
Cosine uses L2-normalized embeddings; others use raw embeddings.
Minkowski is exact Lp (default p=1).

Dataset structure:
  DATASET_ROOT/
    gallery/<label>/*.jpg
    queries/<label>/*.jpg

Run (from project root):
  PYTHONPATH=. python compare_distances_by_model.py \
    --dataset-root storage/mi_eval \
    --k 5 --p 1 \
    --device auto \
    --out-json storage/mi_eval/compare_distances_by_model.json \
    --out-dir  storage/mi_eval/compare_distances_charts
"""

import argparse, json, sys, time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ironclad.modules.extraction.preprocessing import Preprocessing
from ironclad.modules.extraction.embedding import Embedding


def list_images_with_labels(root: Path) -> List[Tuple[Path, str]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir():
                label = d.name
                for p in sorted(d.rglob("*")):
                    if p.is_file() and p.suffix.lower() in exts:
                        items.append((p, label))
    return items


# ---------- metrics ----------
def precision_at_k(rels: List[int], k: int) -> float:
    return float(sum(rels[:k])) / float(max(1, k))


def average_precision_at_k(rels: List[int], k: int, R: int) -> float:
    if R <= 0:
        return 0.0
    denom = float(min(R, k))
    hits = 0
    ap_sum = 0.0
    for i in range(1, k + 1):
        if rels[i-1] == 1:
            hits += 1
            ap_sum += (hits / i)
    return ap_sum / denom


# ---------- top-k (distances returned; lower=better) ----------
def topk_euclidean_with_Gsq(G: np.ndarray, G_sq: np.ndarray, q: np.ndarray, k: int):
    qsq = float(np.dot(q, q))
    d = G_sq + qsq - 2.0 * (G @ q)
    k = min(k, d.shape[0])
    idx = np.argpartition(d, k-1)[:k]
    order = np.argsort(d[idx])
    return d[idx][order], idx[order]


def topk_cosine(Gn: np.ndarray, qn: np.ndarray, k: int):
    sims = Gn @ qn
    d = 1.0 - sims
    k = min(k, d.shape[0])
    idx = np.argpartition(d, k-1)[:k]
    order = np.argsort(d[idx])
    return d[idx][order], idx[order]


def topk_dot(G: np.ndarray, q: np.ndarray, k: int):
    s = G @ q
    d = -s  # convert to distance
    k = min(k, d.shape[0])
    idx = np.argpartition(d, k-1)[:k]
    order = np.argsort(d[idx])
    return d[idx][order], idx[order]


def topk_minkowski(G: np.ndarray, q: np.ndarray, p: float, k: int):
    # exact Lp (we omit the final root since it doesn't change ranking)
    d = np.sum(np.abs(G - q) ** p, axis=1)
    k = min(k, d.shape[0])
    idx = np.argpartition(d, k-1)[:k]
    order = np.argsort(d[idx])
    return d[idx][order], idx[order]


def eval_distances_for_model(pretrained: str,
                             device: str,
                             dataset_root: Path,
                             k: int,
                             p: float) -> Dict[str, Dict[str, float]]:
    """Return metrics per distance for a single model."""
    pre = Preprocessing()
    emb = Embedding(pretrained=pretrained, device=device)

    gal_items = list_images_with_labels(dataset_root / "gallery")
    qry_items = list_images_with_labels(dataset_root / "queries")
    if not gal_items or not qry_items:
        raise RuntimeError(f"Expected images under {dataset_root}/gallery and /queries")

    # label counts (for AP@K normalization)
    label_counts: Dict[str, int] = {}
    for _, lbl in gal_items:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # gallery embeddings
    G_vecs, labels = [], []
    for pth, lbl in gal_items:
        try:
            img = Image.open(pth).convert("RGB")
        except Exception:
            continue
        x = pre.process(img)
        z = emb.encode(x).astype(np.float32)
        G_vecs.append(z); labels.append(lbl)
    if not G_vecs:
        raise RuntimeError("No gallery embeddings computed.")
    G = np.stack(G_vecs, axis=0)
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    G_sq = np.sum(G * G, axis=1)

    # queries
    queries = []
    for pth, gt in qry_items:
        try:
            img = Image.open(pth).convert("RGB")
        except Exception:
            continue
        x = pre.process(img)
        z = emb.encode(x).astype(np.float32)
        zn = z / (np.linalg.norm(z) + 1e-12)
        queries.append((str(pth), z, zn, gt))

    metrics_by_distance: Dict[str, Dict[str, float]] = {}
    distances = ["euclidean", "cosine", "dot_product", f"minkowski_p{int(p) if float(p).is_integer() else p}"]
    for name in distances:
        top1, hit_at_k, ranks = 0, 0, []
        apks = []
        for (_qp, q_raw, q_norm, gt) in queries:
            if name == "euclidean":
                _, idx = topk_euclidean_with_Gsq(G, G_sq, q_raw, k)
            elif name == "cosine":
                _, idx = topk_cosine(Gn, q_norm, k)
            elif name == "dot_product":
                _, idx = topk_dot(G, q_raw, k)
            else:
                _, idx = topk_minkowski(G, q_raw, float(p), k)

            topk_labels = [labels[i] for i in idx]
            rels = [1 if (lab == gt) else 0 for lab in topk_labels]

            # MRR rank
            rank = 0
            for j, r in enumerate(rels, start=1):
                if r == 1:
                    rank = j; break
            ranks.append(rank)
            if rank == 1: top1 += 1
            if 1 <= rank <= k: hit_at_k += 1

            R = label_counts.get(gt, 0)
            apks.append(average_precision_at_k(rels, k, R))

        nQ = max(1, len(queries))
        mrr = float(np.mean([1.0/r if r > 0 else 0.0 for r in ranks]))
        metrics_by_distance[name] = {
            "recall_at_k": float(hit_at_k) / nQ,
            "mean_average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
            "mrr": mrr,  # will use Euclidean MRR as "baseline" in the report
        }

    return metrics_by_distance


def bar_charts_for_model(model_name: str,
                         metrics_by_distance: Dict[str, Dict[str, float]],
                         out_dir: Path):
    # Order distances for display
    order = ["euclidean", "cosine", "dot_product"] + [d for d in metrics_by_distance.keys() if d.startswith("minkowski_")]
    order = [d for d in order if d in metrics_by_distance]

    # Plot mAP@K and Recall@K side-by-side for this model
    metrics = ["mean_average_precision_at_k", "recall_at_k"]
    labels = ["mAP@K", "Recall@K"]

    X = np.arange(len(order))
    W = 0.35

    for m, label in zip(metrics, labels):
        plt.figure(figsize=(8,4.5))
        vals = [metrics_by_distance[d][m] for d in order]
        plt.bar(X, vals, width=W)
        plt.xticks(X, order, rotation=20, ha="right")
        plt.ylim(0, 1.0)
        plt.ylabel("score")
        plt.title(f"{model_name}: {label} by distance")
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"{model_name}_{m}.png", dpi=150)
        plt.close()


def print_markdown(model_name: str, metrics_by_distance: Dict[str, Dict[str, float]]):
    order = ["euclidean", "cosine", "dot_product"] + [d for d in metrics_by_distance.keys() if d.startswith("minkowski_")]
    order = [d for d in order if d in metrics_by_distance]

    header = "| Metric | " + " | ".join(order) + " |\n"
    header += "|" + "|".join(["---"]*(len(order)+1)) + "|\n"

    # mAP
    row_map = ["mAP@K"] + [f"{metrics_by_distance[d]['mean_average_precision_at_k']:.4f}" for d in order]
    # Recall
    row_rec = ["Recall@K"] + [f"{metrics_by_distance[d]['recall_at_k']:.4f}" for d in order]
    # MRR (we'll call Euclidean the baseline)
    row_mrr = ["MRR (baseline=Euclidean)"]
    base_mrr = metrics_by_distance.get("euclidean", {}).get("mrr", 0.0)
    for d in order:
        if d == "euclidean":
            row_mrr.append(f"{base_mrr:.4f}")
        else:
            row_mrr.append("—")

    print(f"\n## {model_name}\n")
    print(header + "| " + " | ".join(row_map) + " |\n| " + " | ".join(row_rec) + " |\n| " + " | ".join(row_mrr) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="Path containing gallery/ and queries/")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--p", type=float, default=1.0, help="Minkowski p (1=L1, 2=L2, ...)")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    models = ["vggface2", "casia-webface"]
    results = {}
    for m in models:
        metrics_by_distance = eval_distances_for_model(
            pretrained=m,
            device=device,
            dataset_root=dataset_root,
            k=args.k,
            p=args.p
        )
        results[m] = metrics_by_distance
        print_markdown(m, metrics_by_distance)
        if args.out_dir:
            bar_charts_for_model(m, metrics_by_distance, Path(args.out_dir))

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump({
                "config": {
                    "dataset_root": str(dataset_root),
                    "k": args.k,
                    "p": args.p,
                    "device": device,
                    "models": models
                },
                "results": results
            }, f, indent=2)

if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)