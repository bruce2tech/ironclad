"""
benchmark_noise_driver.py

Sweep Gaussian noise levels and measure impact on retrieval metrics AND processing time
for exact brute-force search using four distances:
  - Euclidean (squared L2)
  - Cosine (1 - cosine similarity)  [uses L2-normalized embeddings]
  - Dot Product (-g·q)              [treated as a distance by negating the score]
  - Minkowski (Lp, exact)           [default p=1]

Dataset layout:
  DATASET_ROOT/
    gallery/<label>/*.jpg
    queries/<label>/*.jpg

Run (from project root):
  PYTHONPATH=. python benchmark_noise_driver.py \
    --dataset-root storage/mi_eval \
    --k 5 \
    --noise-stds 0,5,10,15,20 \
    --noise-where queries \
    --p 1 \
    --model vggface2 --device auto --dim 512 \
    --seed 42 \
    --out-json storage/mi_eval/noise_driver_results.json \
    --out-dir  storage/mi_eval/noise_driver_plots

Outputs:
- JSON with per-noise results: all six accuracy metrics + average query preprocess/embed/search times.
- Two Markdown tables printed to stdout:
  1) Accuracy (all six metrics) by noise and distance
  2) Timing (query preprocess/embed + per-distance search avg ms) by noise
- Optional plots: one chart per metric (and one for search time), vs noise std, for all distances.
"""

import argparse, json, sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# Project modules
from ironclad.modules.extraction.preprocessing import Preprocessing
from ironclad.modules.extraction.embedding import Embedding


# ---------- I/O helpers ----------
def list_images_with_labels(root: Path) -> List[Tuple[Path, str]]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    items = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir():
                label = d.name
                for p in sorted(d.rglob('*')):
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


# ---------- top-k (return distances and indices) ----------

# Then use a variant that takes G_sq as input:
def topk_euclidean_with_Gsq(G: np.ndarray, G_sq: np.ndarray, q: np.ndarray, k: int):
    qsq = float(np.dot(q, q))
    d = G_sq + qsq - 2.0 * (G @ q)
    k = min(k, d.shape[0])
    idx = np.argpartition(d, k-1)[:k]
    order = np.argsort(d[idx])
    return d[idx][order], idx[order]

# def topk_euclidean(G: np.ndarray, q: np.ndarray, k: int):
#     Gsq = np.sum(G * G, axis=1)
#     qsq = float(np.dot(q, q))
#     d = Gsq + qsq - 2.0 * (G @ q)
#     k = min(k, d.shape[0])
#     idx = np.argpartition(d, k-1)[:k]
#     order = np.argsort(d[idx])
#     return d[idx][order], idx[order]


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
    Returns (metrics_by_variant, search_time_avg_ms_by_variant)
    """
    variants = ["euclidean", "cosine", "dot_product", f"minkowski_p{int(p_minkowski) if float(p_minkowski).is_integer() else p_minkowski}"]
    metrics: Dict[str, Dict[str, float]] = {}
    search_times_ms: Dict[str, float] = {}

    # Precompute per-variant search times
    import time
    for name in variants:
        top1, hit_at_k, ranks = 0, 0, []
        precisions, apks = [], []
        t_search_total = 0.0
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
            t_search_total += (t1 - t0)

            topk_labels = [labels[i] for i in idx]
            rels = [1 if (lab == gt) else 0 for lab in topk_labels]

            # first relevant rank
            rank = 0
            for j, r in enumerate(rels, start=1):
                if r == 1:
                    rank = j
                    break
            ranks.append(rank)
            if rank == 1: top1 += 1
            if 1 <= rank <= k: hit_at_k += 1

            precisions.append(precision_at_k(rels, k))
            apks.append(average_precision_at_k(rels, k, R))

        nQ = max(1, len(queries))
        mrr = float(np.mean([1.0/r if r > 0 else 0.0 for r in ranks]))
        res = {
            "top1_acc": float(top1) / nQ,
            "recall_at_k": float(hit_at_k) / nQ,
            "mrr": mrr,
            "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
            "mean_average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
        }
        metrics[name] = res
        search_times_ms[name] = (t_search_total / nQ) * 1e3  # avg per query in ms

    return metrics, search_times_ms


# ---------- markdown helpers ----------
def accuracy_markdown_table(noise_levels: List[float], results: List[Dict[str, Dict[str, float]]]):
    metrics = ["top1_acc","recall_at_k","mrr","precision_at_k","average_precision_at_k","mean_average_precision_at_k"]
    variants = list(next(iter(results)).keys()) if results else []
    header = ["Std"]
    for v in variants:
        for m in metrics:
            header.append(f"{v}:{m}")
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"]*len(header)) + "|"]
    for std, res in zip(noise_levels, results):
        row = [f"{std:g}"]
        for v in variants:
            for m in metrics:
                row.append(f"{res[v][m]:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def timing_markdown_table(noise_levels: List[float],
                          q_pre_ms: List[float],
                          q_emb_ms: List[float],
                          search_ms_list: List[Dict[str, float]]):
    variants = list(search_ms_list[0].keys()) if search_ms_list else []
    header = ["Std","query_pre_ms","query_emb_ms"] + [f"{v}:search_ms" for v in variants]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"]*len(header)) + "|"]
    for std, pre, emb, smap in zip(noise_levels, q_pre_ms, q_emb_ms, search_ms_list):
        row = [f"{std:g}", f"{pre:.2f}", f"{emb:.2f}"] + [f"{smap[v]:.2f}" for v in variants]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--p", type=float, default=1.0, help="Minkowski p (1=L1, 2=L2, ...)")
    ap.add_argument("--model", type=str, default="vggface2")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--noise-stds", type=str, default="0,5,10,15,20")
    ap.add_argument("--noise-where", type=str, default="queries", choices=["queries","gallery","both"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None, help="Directory to save plots (one per metric)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    noise_levels = [float(s) for s in args.noise_stds.split(",") if s.strip() != ""]

    dataset_root = Path(args.dataset_root)
    gal_items = list_images_with_labels(dataset_root / "gallery")
    qry_items = list_images_with_labels(dataset_root / "queries")
    if not gal_items or not qry_items:
        print(f"[ERROR] Expected images under {dataset_root}/gallery and /queries", file=sys.stderr)
        sys.exit(2)

    # gallery label counts (for AP@K)
    label_counts: Dict[str, int] = {}
    for _, lbl in gal_items:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # device
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    pre = Preprocessing()
    model = Embedding(pretrained=args.model, device=device)

    # storage for results
    metrics_per_std: List[Dict[str, Dict[str, float]]] = []
    q_pre_ms_list: List[float] = []
    q_emb_ms_list: List[float] = []
    search_ms_per_std: List[Dict[str, float]] = []

    import time

    for std in noise_levels:
        # --- Build gallery embeddings (with noise if requested) ---
        G_vecs, labels = [], []
        t_g_pre, t_g_emb = [], []
        for pth, lbl in gal_items:
            try:
                img = Image.open(pth).convert("RGB")
            except Exception:
                continue
            if args.noise_where in ("gallery","both") and std > 0:
                img = add_gaussian_noise_rgb(img, std, rng)
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

        # --- Build query embeddings (with noise if requested) ---
        queries = []
        t_q_pre, t_q_emb = [], []
        for pth, gt in qry_items:
            try:
                img = Image.open(pth).convert("RGB")
            except Exception:
                continue
            if args.noise_where in ("queries","both") and std > 0:
                img = add_gaussian_noise_rgb(img, std, rng)
            t0 = time.perf_counter(); x = pre.process(img); t1 = time.perf_counter()
            z = model.encode(x).astype(np.float32); t2 = time.perf_counter()
            t_q_pre.append((t1 - t0) * 1e3)
            t_q_emb.append((t2 - t1) * 1e3)
            zn = z / (np.linalg.norm(z) + 1e-12)
            queries.append((str(pth), z, zn, gt))

        # --- Evaluate all distances & measure search time ---
        metrics, search_ms = evaluate_variants(G, Gn, G_sq, labels, label_counts, queries, args.k, args.p)
        metrics_per_std.append(metrics)
        q_pre_ms_list.append(float(np.mean(t_q_pre)) if t_q_pre else 0.0)
        q_emb_ms_list.append(float(np.mean(t_q_emb)) if t_q_emb else 0.0)
        search_ms_per_std.append(search_ms)

    # --- Print Markdown tables ---
    print("\n# Accuracy vs Noise\n")
    print(accuracy_markdown_table(noise_levels, metrics_per_std))

    print("\n# Timing vs Noise (averages per query, ms)\n")
    print(timing_markdown_table(noise_levels, q_pre_ms_list, q_emb_ms_list, search_ms_per_std))

    # --- Save JSON ---
    report = {
        "config": {
            "k": args.k,
            "p": args.p,
            "model": args.model,
            "device": device,
            "dataset_root": str(dataset_root),
            "noise_where": args.noise_where,
            "noise_stds": noise_levels,
            "seed": args.seed,
        },
        "sizes": {"gallery": len(gal_items), "queries": len(qry_items)},
        "results": [
            {
                "std": std,
                "metrics": metrics_per_std[i],
                "query_preprocess_avg_ms": q_pre_ms_list[i],
                "query_embed_avg_ms": q_emb_ms_list[i],
                "search_avg_ms_by_variant": search_ms_per_std[i],
            } for i, std in enumerate(noise_levels)
        ]
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)

    # --- Plots (one per metric + one for search time) ---
    if args.out_dir:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

        # accuracy plots
        metrics_names = ["top1_acc","recall_at_k","mrr","precision_at_k","average_precision_at_k","mean_average_precision_at_k"]
        variants = list(metrics_per_std[0].keys()) if metrics_per_std else []
        xs = noise_levels

        for m in metrics_names:
            plt.figure(figsize=(8,4.5))
            for v in variants:
                ys = [metrics_per_std[i][v][m] for i in range(len(xs))]
                plt.plot(xs, ys, marker="o", label=v)
            plt.xlabel("Gaussian noise std (pixel units)")
            plt.ylabel(m)
            plt.title(f"{m} vs noise (k={args.k})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / f"{m}_vs_noise.png", dpi=150)
            plt.close()

        # search time plot
        plt.figure(figsize=(8,4.5))
        for v in variants:
            ys = [search_ms_per_std[i][v] for i in range(len(xs))]
            plt.plot(xs, ys, marker="o", label=v)
        plt.xlabel("Gaussian noise std (pixel units)")
        plt.ylabel("search avg (ms/query)")
        plt.title("Search time vs noise")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "search_time_vs_noise.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)