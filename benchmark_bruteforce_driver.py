"""
benchmark_bruteforce_driver.py

Exact brute-force comparison of Euclidean, Cosine, Dot Product, and Minkowski(p)
on your labeled dataset.

Expected structure:
  DATASET_ROOT/
    gallery/<label>/*.jpg
    queries/<label>/*.jpg

Run (from project root):
  PYTHONPATH=. python benchmark_bruteforce_driver.py \
    --dataset-root storage/mi_eval \
    --k 5 \
    --p 1 \
    --model vggface2 --device auto --dim 512 \
    --out-json storage/mi_eval/bruteforce_multi_metric.json \
    --out-fig  storage/mi_eval/bruteforce_multi_metric.png
"""

import argparse, json, sys, time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from ironclad.modules.extraction.preprocessing import Preprocessing
from ironclad.modules.extraction.embedding import Embedding


# ---------- helpers ----------
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


# ---------- top-k (all return *distances* lower is better) ----------

# Then use a variant that takes G_sq as input:
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
    d = -s
    k = min(k, d.shape[0])
    idx = np.argpartition(d, k-1)[:k]
    order = np.argsort(d[idx])
    return d[idx][order], idx[order]


def topk_minkowski(G: np.ndarray, q: np.ndarray, p: float, k: int):
    diff = np.abs(G - q) ** p
    d = np.sum(diff, axis=1) ** (1.0 / p)
    k = min(k, d.shape[0])
    idx = np.argpartition(d, k-1)[:k]
    order = np.argsort(d[idx])
    return d[idx][order], idx[order]


# ---------- evaluation ----------
def evaluate_variants(G: np.ndarray,
                      Gn: np.ndarray,
                      G_sq: np.ndarray,
                      labels: List[str],
                      label_counts: Dict[str, int],
                      queries: List[Tuple[str, np.ndarray, np.ndarray, str]],
                      k: int,
                      p_minkowski: float) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Returns (metrics_by_variant, avg_search_time_ms_by_variant)
    """
    variants = ["euclidean", "cosine", "dot_product", f"minkowski_p{int(p_minkowski) if float(p_minkowski).is_integer() else p_minkowski}"]
    metrics: Dict[str, Dict[str, float]] = {}
    search_ms: Dict[str, float] = {}

    for name in variants:
        top1, hit_at_k, ranks = 0, 0, []
        precisions, apks = [], []
        t_total = 0.0

        for (_qp, q_raw, q_norm, gt) in queries:
            R = label_counts.get(gt, 0)

            t0 = time.perf_counter()
            if name == "euclidean":
                _, idx = topk_euclidean_with_Gsq(G, G_sq, q_raw, k)
            elif name == "cosine":
                _, idx = topk_cosine(Gn, q_norm, k)
            elif name == "dot_product":
                _, idx = topk_dot(G, q_raw, k)
            else:  # minkowski
                _, idx = topk_minkowski(G, q_raw, float(p_minkowski), k)
            t1 = time.perf_counter()
            t_total += (t1 - t0)

            topk_labels = [labels[i] for i in idx]
            rels = [1 if (lab == gt) else 0 for lab in topk_labels]

            rank = 0
            for j, r in enumerate(rels, start=1):
                if r == 1:
                    rank = j; break
            ranks.append(rank)
            if rank == 1: top1 += 1
            if 1 <= rank <= k: hit_at_k += 1

            precisions.append(precision_at_k(rels, k))
            apks.append(average_precision_at_k(rels, k, R))

        nQ = max(1, len(queries))
        mrr = float(np.mean([1.0/r if r > 0 else 0.0 for r in ranks]))
        metrics[name] = {
            "top1_acc": float(top1) / nQ,
            "recall_at_k": float(hit_at_k) / nQ,
            "mrr": mrr,
            "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
            "mean_average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
        }
        search_ms[name] = (t_total / nQ) * 1e3  # avg per query (ms)

    return metrics, search_ms


def accuracy_markdown_table(results: Dict[str, Dict[str, float]]) -> str:
    metrics = [
        ("top1_acc", "Top-1 Acc"),
        ("recall_at_k", "Recall@K"),
        ("mrr", "MRR"),
        ("precision_at_k", "Precision@K"),
        ("average_precision_at_k", "AP@K"),
        ("mean_average_precision_at_k", "mAP@K"),
    ]
    variants = list(results.keys())
    header = "| Metric | " + " | ".join(variants) + " |\n"
    header += "|" + "|".join(["---"]*(len(variants)+1)) + "|\n"
    rows = []
    for key, label in metrics:
        row = [label] + [f"{results[v][key]:.4f}" for v in variants]
        rows.append("| " + " | ".join(row) + " |")
    return header + "\n".join(rows)


def timing_markdown_table(q_pre_ms: float, q_emb_ms: float, search_ms: Dict[str, float]) -> str:
    variants = list(search_ms.keys())
    header = "| Metric | " + " | ".join(variants) + " |\n"
    header += "|" + "|".join(["---"]*(len(variants)+1)) + "|\n"
    rows = []
    rows.append("| query_pre_ms | " + " | ".join([f"{q_pre_ms:.2f}"]*len(variants)) + " |")
    rows.append("| query_embed_ms | " + " | ".join([f"{q_emb_ms:.2f}"]*len(variants)) + " |")
    rows.append("| search_avg_ms | " + " | ".join([f"{search_ms[v]:.2f}" for v in variants]) + " |")
    return header + "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--p", type=float, default=1.0, help="Minkowski p (1=L1, 2=L2, etc.)")
    ap.add_argument("--model", default="vggface2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-fig", default=None)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    gal_items = list_images_with_labels(dataset_root / "gallery")
    qry_items = list_images_with_labels(dataset_root / "queries")
    if not gal_items or not qry_items:
        print(f"[ERROR] Expected images under {dataset_root}/gallery and /queries", file=sys.stderr)
        sys.exit(2)

    # Count gallery per label (for AP@K)
    label_counts: Dict[str, int] = {}
    for _, lbl in gal_items:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # Device
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    pre = Preprocessing()
    model = Embedding(pretrained=args.model, device=device)

    # Build gallery embeddings (raw + normalized); also log norms
    G_vecs, labels = [], []
    t_g_pre, t_g_emb = [], []
    for p, lbl in gal_items:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        t0 = time.perf_counter(); x = pre.process(img); t1 = time.perf_counter()
        z = model.encode(x).astype(np.float32); t2 = time.perf_counter()
        t_g_pre.append((t1 - t0) * 1e3)
        t_g_emb.append((t2 - t1) * 1e3)
        G_vecs.append(z); labels.append(lbl)
    if not G_vecs:
        print("[ERROR] No gallery embeddings computed.", file=sys.stderr)
        sys.exit(2)
    G = np.stack(G_vecs, axis=0)
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    G_sq = np.sum(G * G, axis=1)

    norms = np.linalg.norm(G, axis=1)
    print(f"[diag] gallery norms: mean={norms.mean():.4f} std={norms.std():.4f} min={norms.min():.4f} max={norms.max():.4f}")

    # Build queries (measure preprocess/embed timing)
    queries = []
    t_q_pre, t_q_emb = [], []
    for p, gt in qry_items:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        t0 = time.perf_counter(); x = pre.process(img); t1 = time.perf_counter()
        z = model.encode(x).astype(np.float32); t2 = time.perf_counter()
        t_q_pre.append((t1 - t0) * 1e3)
        t_q_emb.append((t2 - t1) * 1e3)
        zn = z / (np.linalg.norm(z) + 1e-12)
        queries.append((str(p), z, zn, gt))

    # Evaluate all variants (and measure search avg ms)
    metrics, search_ms = evaluate_variants(G, Gn, G_sq, labels, label_counts, queries, args.k, args.p)

    # Summaries
    q_pre_ms = float(np.mean(t_q_pre)) if t_q_pre else 0.0
    q_emb_ms = float(np.mean(t_q_emb)) if t_q_emb else 0.0

    report = {
        "config": {
            "k": args.k, "p": args.p, "model": args.model,
            "device": device, "dataset_root": str(dataset_root),
        },
        "sizes": {
            "gallery": int(G.shape[0]), "queries": int(len(queries)),
        },
        "timings_ms": {
            "gallery_preprocess_avg_ms": float(np.mean(t_g_pre)) if t_g_pre else 0.0,
            "gallery_embed_avg_ms": float(np.mean(t_g_emb)) if t_g_emb else 0.0,
            "query_preprocess_avg_ms": q_pre_ms,
            "query_embed_avg_ms": q_emb_ms,
            "search_avg_ms_by_variant": search_ms,
        },
        "results": metrics,
    }

    print(json.dumps(report, indent=2))

    # Markdown tables
    print("\n# Accuracy (Baseline)\n")
    print(accuracy_markdown_table(metrics))

    print("\n# Timing (Baseline, averages per query in ms)\n")
    print(timing_markdown_table(q_pre_ms, q_emb_ms, search_ms))

    # Optional grouped bar chart (accuracy only)
    if args.out_fig:
        metrics_list = ["top1_acc","recall_at_k","mrr","precision_at_k","average_precision_at_k","mean_average_precision_at_k"]
        names = list(metrics.keys())
        X = np.arange(len(metrics_list))
        W = 0.8 / max(1, len(names))
        plt.figure(figsize=(12, 5))
        for i, n in enumerate(names):
            vals = [metrics[n][m] for m in metrics_list]
            plt.bar(X + (i - (len(names)-1)/2)*W, vals, width=W, label=n)
        plt.xticks(X, metrics_list, rotation=25, ha="right")
        plt.ylabel("score")
        plt.title(f"Exact brute-force (k={args.k}, p={args.p}) — Baseline")
        plt.legend()
        plt.tight_layout()
        Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_fig, dpi=150)
        plt.close()

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)

# import argparse, json, sys, time
# from pathlib import Path
# from typing import List, Tuple, Dict
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import time

# # Your modules
# from ironclad.modules.extraction.preprocessing import Preprocessing
# from ironclad.modules.extraction.embedding import Embedding

# # ---------- helpers ----------
# def list_images_with_labels(root: Path) -> List[Tuple[Path, str]]:
#     exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
#     items = []
#     if root.exists():
#         for d in sorted(root.iterdir()):
#             if d.is_dir():
#                 label = d.name
#                 for p in sorted(d.rglob("*")):
#                     if p.is_file() and p.suffix.lower() in exts:
#                         items.append((p, label))
#     return items

# def precision_at_k(rels: List[int], k: int) -> float:
#     return float(sum(rels[:k])) / float(max(1, k))

# def average_precision_at_k(rels: List[int], k: int, R: int) -> float:
#     # AP@K = (1/min(R,K)) * sum_{i=1..K} [ P@i * rel_i ]
#     if R <= 0:
#         return 0.0
#     denom = float(min(R, k))
#     hits = 0
#     ap_sum = 0.0
#     for i in range(1, k + 1):
#         if rels[i-1] == 1:
#             hits += 1
#             ap_sum += (hits / i)
#     return ap_sum / denom

# # top-k helpers (all return *distances* where lower is better, and indices)
# def topk_euclidean(G: np.ndarray, q: np.ndarray, k: int):
#     # squared L2: ||g||^2 + ||q||^2 - 2*g·q
#     Gsq = np.sum(G * G, axis=1)
#     qsq = float(np.dot(q, q))
#     d = Gsq + qsq - 2.0 * (G @ q)
#     k = min(k, d.shape[0])
#     idx = np.argpartition(d, k-1)[:k]
#     order = np.argsort(d[idx])
#     return d[idx][order], idx[order]

# def topk_cosine(Gn: np.ndarray, qn: np.ndarray, k: int):
#     # cosine distance = 1 - cos_sim
#     sims = Gn @ qn
#     d = 1.0 - sims
#     k = min(k, d.shape[0])
#     idx = np.argpartition(d, k-1)[:k]
#     order = np.argsort(d[idx])
#     return d[idx][order], idx[order]

# def topk_dot(G: np.ndarray, q: np.ndarray, k: int):
#     # dot-product as a *score* (higher is better) -> convert to distance = -score
#     s = G @ q
#     d = -s
#     k = min(k, d.shape[0])
#     idx = np.argpartition(d, k-1)[:k]
#     order = np.argsort(d[idx])
#     return d[idx][order], idx[order]

# def topk_minkowski(G: np.ndarray, q: np.ndarray, p: float, k: int):
#     # exact Lp
#     # distances = (sum |g - q|^p)^(1/p)
#     diff = np.abs(G - q) ** p
#     d = np.sum(diff, axis=1) ** (1.0 / p)
#     k = min(k, d.shape[0])
#     idx = np.argpartition(d, k-1)[:k]
#     order = np.argsort(d[idx])
#     return d[idx][order], idx[order]

# def eval_variant(name: str,
#                  gallery_raw: np.ndarray,
#                  gallery_norm: np.ndarray,
#                  labels: List[str],
#                  label_counts: Dict[str, int],
#                  queries: List[Tuple[str, np.ndarray, np.ndarray, str]],
#                  k: int,
#                  p_minkowski: float = 1.0) -> Dict[str, float]:
#     top1, hit_at_k, ranks = 0, 0, []
#     precisions, apks = [], []
#     for (_qp, q_raw, q_norm, gt) in queries:
#         R = label_counts.get(gt, 0)
#         if name == "euclidean":
#             _, idx = topk_euclidean(gallery_raw, q_raw, k)
#         elif name == "cosine":
#             _, idx = topk_cosine(gallery_norm, q_norm, k)
#         elif name == "dot_product":
#             _, idx = topk_dot(gallery_raw, q_raw, k)
#         elif name.startswith("minkowski"):
#             _, idx = topk_minkowski(gallery_raw, q_raw, p_minkowski, k)
#         else:
#             raise ValueError(name)

#         topk_labels = [labels[i] for i in idx]
#         rels = [1 if (lab == gt) else 0 for lab in topk_labels]
#         # first relevant rank for MRR / top1 / recall@k
#         rank = 0
#         for j, r in enumerate(rels, start=1):
#             if r == 1:
#                 rank = j
#                 break
#         ranks.append(rank)
#         if rank == 1: top1 += 1
#         if 1 <= rank <= k: hit_at_k += 1

#         precisions.append(precision_at_k(rels, k))
#         apks.append(average_precision_at_k(rels, k, R))

#     nQ = max(1, len(queries))
#     mrr = float(np.mean([1.0/r if r > 0 else 0.0 for r in ranks]))
#     return {
#         "top1_acc": float(top1) / nQ,
#         "recall_at_k": float(hit_at_k) / nQ,
#         "mrr": mrr,
#         "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
#         "average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
#         "mean_average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
#     }

# def markdown_table(results: Dict[str, Dict[str, float]]) -> str:
#     metrics = [
#         ("top1_acc", "Top-1 Acc"),
#         ("recall_at_k", "Recall@K"),
#         ("mrr", "MRR"),
#         ("precision_at_k", "Precision@K"),
#         ("average_precision_at_k", "AP@K"),
#         ("mean_average_precision_at_k", "mAP@K"),
#     ]
#     variants = list(results.keys())
#     header = "| Metric | " + " | ".join([v for v in variants]) + " |\n"
#     header += "|" + "|".join(["---"]*(len(variants)+1)) + "|\n"
#     rows = []
#     for key, label in metrics:
#         row = [label] + [f"{results[v][key]:.4f}" for v in variants]
#         rows.append("| " + " | ".join(row) + " |")
#     return header + "\n".join(rows)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset-root", required=True)
#     ap.add_argument("--k", type=int, default=5)
#     ap.add_argument("--p", type=float, default=1.0, help="Minkowski p (1=L1, 2=L2, etc.)")
#     ap.add_argument("--model", default="vggface2")
#     ap.add_argument("--device", default="auto")
#     ap.add_argument("--dim", type=int, default=512)
#     ap.add_argument("--out-json", default=None)
#     ap.add_argument("--out-fig", default=None)
#     args = ap.parse_args()

#     dataset_root = Path(args.dataset_root)
#     gal_items = list_images_with_labels(dataset_root / "gallery")
#     qry_items = list_images_with_labels(dataset_root / "queries")
#     if not gal_items or not qry_items:
#         print(f"[ERROR] Expected images under {dataset_root}/gallery and {dataset_root}/queries", file=sys.stderr)
#         sys.exit(2)

#     # Count gallery per label (for AP@K)
#     label_counts: Dict[str, int] = {}
#     for _, lbl in gal_items:
#         label_counts[lbl] = label_counts.get(lbl, 0) + 1

#     # Device
#     device = args.device
#     if device == "auto":
#         try:
#             import torch
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#         except Exception:
#             device = "cpu"

#     pre = Preprocessing()
#     model = Embedding(pretrained=args.model, device=device)

#     # Build gallery embeddings (raw + normalized)
#     G_vecs, labels = [], []
#     for p, lbl in gal_items:
#         try:
#             img = Image.open(p).convert("RGB")
#         except Exception:
#             continue
#         x = pre.process(img)
#         z = model.encode(x).astype(np.float32)  # raw
#         G_vecs.append(z)
#         labels.append(lbl)
#     G = np.stack(G_vecs, axis=0)
#     Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)

#     # Diagnostics: are we already unit-norm?
#     norms = np.linalg.norm(G, axis=1)
#     print(f"[diag] gallery norms: mean={norms.mean():.4f} std={norms.std():.4f} min={norms.min():.4f} max={norms.max():.4f}")

#     # Build queries
#     queries = []
#     for p, gt in qry_items:
#         try:
#             img = Image.open(p).convert("RGB")
#         except Exception:
#             continue
#         x = pre.process(img)
#         z = model.encode(x).astype(np.float32)
#         zn = z / (np.linalg.norm(z) + 1e-12)
#         queries.append((str(p), z, zn, gt))

#     # Evaluate all variants (exact brute-force)
#     variants = {
#         "euclidean": ("Euclidean (squared L2)", {}),
#         "cosine": ("Cosine (1 - cos sim)", {}),
#         "dot_product": ("Dot Product (-g·q)", {}),
#         f"minkowski_p{int(args.p) if float(args.p).is_integer() else args.p}": ("Minkowski Lp", {"p": args.p}),
#     }

#     results = {}
#     for name, (_desc, extra) in variants.items():
#         if name.startswith("minkowski"):
#             res = eval_variant("minkowski", G, Gn, labels, label_counts, queries, args.k, p_minkowski=extra["p"])
#         else:
#             res = eval_variant(name, G, Gn, labels, label_counts, queries, args.k)
#         results[name] = res

#     # Print JSON + Markdown table
#     report = {
#         "config": {
#             "k": args.k,
#             "p": args.p,
#             "model": args.model,
#             "device": device,
#             "dataset_root": str(dataset_root),
#         },
#         "sizes": {"gallery": int(G.shape[0]), "queries": int(len(queries))},
#         "results": results,
#     }
#     print(json.dumps(report, indent=2))
#     print("\n# Markdown Table\n")
#     print(markdown_table(results))

#     # Optional figure: grouped bars per metric
#     if args.out_fig:
#         metrics = ["top1_acc","recall_at_k","mrr","precision_at_k","average_precision_at_k","mean_average_precision_at_k"]
#         names = list(results.keys())
#         X = np.arange(len(metrics))
#         W = 0.8 / max(1, len(names))
#         plt.figure(figsize=(12, 5))
#         for i, n in enumerate(names):
#             vals = [results[n][m] for m in metrics]
#             plt.bar(X + (i - (len(names)-1)/2)*W, vals, width=W, label=n)
#         plt.xticks(X, metrics, rotation=25, ha="right")
#         plt.ylabel("score")
#         plt.title(f"Exact brute-force comparison (k={args.k}, p={args.p})")
#         plt.legend()
#         plt.tight_layout()
#         Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(args.out_fig, dpi=150)
#         plt.close()

#     if args.out_json:
#         Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
#         with open(args.out_json, "w") as f:
#             json.dump(report, f, indent=2)

if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)
