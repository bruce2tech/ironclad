# Create two separate, ready-to-run scripts:
# 1) compare_bruteforce_vs_hnsw_vggface2.py  (fixed to pretrained='vggface2')
# 2) compare_bruteforce_vs_hnsw_casia.py     (fixed to pretrained='casia-webface')


"""
Compare Brute Force (FAISS Flat) vs HNSW on the same dataset using the chosen FaceNet model,
reporting accuracy metrics and latency, and extrapolating to 1B vectors (very rough).
Default metric: cosine (normalize + inner-product).

Dataset structure:
  DATASET_ROOT/
    gallery/<label>/*.jpg
    queries/<label>/*.jpg

Example:
  PYTHONPATH=. python {script_name} \
    --dataset-root storage/mi_eval \
    --k 5 \
    --metric cosine \
    --hnsw-M 32 --hnsw-efC 80 --hnsw-efS 64 \
    --faiss-threads 1 \
    --out-json storage/mi_eval/{shortname}.json \
    --out-dir  storage/mi_eval/{shortname}_plots
"""

import argparse, json, sys, time, math
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import faiss

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


def compute_metrics(indices: np.ndarray, labels: List[str], gt_label: str, k: int, label_counts: Dict[str, int]):
    topk_labels = [labels[i] for i in indices]
    rels = [1 if (lab == gt_label) else 0 for lab in topk_labels]

    rank = 0
    for j, r in enumerate(rels, start=1):
        if r == 1:
            rank = j; break
    top1 = 1 if rank == 1 else 0
    hit_at_k = 1 if 1 <= rank <= k else 0

    R = label_counts.get(gt_label, 0)
    ap_k = average_precision_at_k(rels, k, R)
    return top1, hit_at_k, (1.0/rank if rank > 0 else 0.0), precision_at_k(rels, k), ap_k


def build_flat_index(dim: int, metric: str = "cosine"):
    if metric == "cosine":
        return faiss.IndexFlatIP(dim), faiss.METRIC_INNER_PRODUCT
    elif metric == "euclidean":
        return faiss.IndexFlatL2(dim), faiss.METRIC_L2
    else:
        raise ValueError("Only 'cosine' or 'euclidean' supported here.")


def build_hnsw_index(dim: int, M: int, efC: int, metric: str = "cosine"):
    if metric == "cosine":
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = efC
    return index


def normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def estimate_memory_bytes(n: int, dim: int, metric: str, M: int = 32) -> Dict[str, int]:
    """
    Rough memory estimates (bytes) for 1B points:

    - Vectors (float32): n * dim * 4
    - HNSW links: ~ between n * M * 4 and n * 2 * M * 4 (int32 per link), ignoring levels/overheads.
    - Flat index has negligible overhead beyond vectors.
    """
    vec_bytes = n * dim * 4
    graph_low = n * M * 4
    graph_high = n * 2 * M * 4
    return {"vectors_bytes": vec_bytes, "hnsw_graph_low_bytes": graph_low, "hnsw_graph_high_bytes": graph_high}


def human_bytes(num: float) -> str:
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}EB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--metric", default="cosine", choices=["cosine","euclidean"])

    ap.add_argument("--hnsw-M", type=int, default=32)
    ap.add_argument("--hnsw-efC", type=int, default=80)
    ap.add_argument("--hnsw-efS", type=int, default=64)

    ap.add_argument("--faiss-threads", type=int, default=1, help="Set FAISS OMP threads (macOS often stable at 1)")

    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    # threads (helps stability on some macOS setups)
    faiss.omp_set_num_threads(max(1, int(args.faiss_threads)))

    # device
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # load images
    dataset_root = Path(args.dataset_root)
    gal_items = list_images_with_labels(dataset_root / "gallery")
    qry_items = list_images_with_labels(dataset_root / "queries")
    if not gal_items or not qry_items:
        print(f"[ERROR] Expected images under {dataset_root}/gallery and /queries", file=sys.stderr)
        sys.exit(2)

    # counts for AP normalization
    label_counts: Dict[str, int] = {}
    for _, lbl in gal_items:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # embed
    pre = Preprocessing()
    emb = Embedding(pretrained="{pretrained}", device=device)

    G_vecs, G_labels = [], []
    t_g_pre, t_g_emb = [], []
    for p, lab in gal_items:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        t0 = time.perf_counter(); x = pre.process(img); t1 = time.perf_counter()
        z = emb.encode(x).astype(np.float32); t2 = time.perf_counter()
        t_g_pre.append((t1 - t0) * 1e3)
        t_g_emb.append((t2 - t1) * 1e3)
        G_vecs.append(z); G_labels.append(lab)

    Q_vecs, Q_labels = [], []
    t_q_pre, t_q_emb = [], []
    for p, lab in qry_items:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        t0 = time.perf_counter(); x = pre.process(img); t1 = time.perf_counter()
        z = emb.encode(x).astype(np.float32); t2 = time.perf_counter()
        t_q_pre.append((t1 - t0) * 1e3)
        t_q_emb.append((t2 - t1) * 1e3)
        Q_vecs.append(z); Q_labels.append(lab)

    if not G_vecs or not Q_vecs:
        print("[ERROR] No embeddings computed.", file=sys.stderr); sys.exit(2)

    G = np.stack(G_vecs, axis=0)
    Q = np.stack(Q_vecs, axis=0)
    dim = G.shape[1]

    # normalize for cosine
    do_norm = (args.metric == "cosine")
    if do_norm:
        def normalize_rows(X: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            return X / norms
        G = normalize_rows(G)
        Q = normalize_rows(Q)

    # ---------- Build Flat ----------
    flat, _ = build_flat_index(dim, metric=args.metric)
    t0 = time.perf_counter()
    flat.add(G)
    t1 = time.perf_counter()
    flat_build_ms = (t1 - t0) * 1e3

    # ---------- Build HNSW ----------
    hnsw = build_hnsw_index(dim, args.hnsw_M, args.hnsw_efC, metric=args.metric)
    t0 = time.perf_counter()
    hnsw.add(G)
    t1 = time.perf_counter()
    hnsw_build_ms = (t1 - t0) * 1e3
    hnsw.hnsw.efSearch = args.hnsw_efS

    # ---------- Evaluate ----------
    def eval_index(index, name: str):
        top1 = 0
        hit_at_k = 0
        mrrs = []; precisions = []; apks = []
        t_search = 0.0
        for q, gt in zip(Q, Q_labels):
            q = q.reshape(1, -1)
            t0 = time.perf_counter()
            D, I = index.search(q, args.k)
            t1 = time.perf_counter()
            t_search += (t1 - t0)

            idx = I[0].tolist()

            t1_, hit_, mrr_, p_, ap_ = compute_metrics(idx, G_labels, gt, args.k, label_counts)
            top1 += t1_; hit_at_k += hit_; mrrs.append(mrr_); precisions.append(p_); apks.append(ap_)
        nQ = len(Q_labels)
        return {
            "top1_acc": top1 / nQ if nQ else 0.0,
            "recall_at_k": hit_at_k / nQ if nQ else 0.0,
            "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
            "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
            "mean_average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
            "search_avg_ms": (t_search / nQ) * 1e3 if nQ else 0.0
        }

    flat_metrics = eval_index(flat, "flat")
    hnsw_metrics = eval_index(hnsw, "hnsw")

    # ---------- Extrapolate to 1B ----------
    N_cur = G.shape[0]
    N_target = 1_000_000_000

    # Flat scaling ~ linear in N
    per_vec_ms = flat_metrics["search_avg_ms"] / max(1, N_cur)
    est_flat_ms = per_vec_ms * N_target
    est_flat_s = est_flat_ms / 1e3

    # HNSW scaling ~ O(efSearch * log N). As a crude proxy, scale by log(N).
    def logn(n): return math.log(max(2, n), 2.0)
    scale_hnsw = logn(N_target) / logn(N_cur)
    est_hnsw_ms = hnsw_metrics["search_avg_ms"] * scale_hnsw
    est_hnsw_s = est_hnsw_ms / 1e3

    mem_est = estimate_memory_bytes(N_target, dim, args.metric, M=args.hnsw_M)

    # ---------- Report ----------
    results = {
        "config": {
            "dataset_root": str(dataset_root),
            "k": args.k,
            "model": "{pretrained}",
            "device": device,
            "metric": args.metric,
            "hnsw": {"M": args.hnsw_M, "efConstruction": args.hnsw_efC, "efSearch": args.hnsw_efS},
            "dim": dim,
            "gallery_size": int(N_cur),
            "queries": int(Q.shape[0])
        },
        "timings_ms": {
            "build_flat_ms": flat_build_ms,
            "build_hnsw_ms": hnsw_build_ms,
            "flat_search_avg_ms": flat_metrics["search_avg_ms"],
            "hnsw_search_avg_ms": hnsw_metrics["search_avg_ms"]
        },
        "metrics": {"flat": flat_metrics, "hnsw": hnsw_metrics},
        "scaling_estimates_1B": {
            "flat_search_est_s_per_query": est_flat_s,
            "hnsw_search_est_s_per_query": est_hnsw_s,
            "assumptions": {
                "flat_linear_per_vector_ms": per_vec_ms,
                "hnsw_log_scaling_factor": scale_hnsw,
                "note": "HNSW estimate assumes cost ~ efSearch*logN; real-world depends on data/params."
            },
            "memory_bytes": {
                "vectors": mem_est["vectors_bytes"],
                "hnsw_graph_low": mem_est["hnsw_graph_low_bytes"],
                "hnsw_graph_high": mem_est["hnsw_graph_high_bytes"]
            },
            "memory_human": {
                "vectors": human_bytes(mem_est["vectors_bytes"]),
                "hnsw_graph_low": human_bytes(mem_est["hnsw_graph_low_bytes"]),
                "hnsw_graph_high": human_bytes(mem_est["hnsw_graph_high_bytes"])
            }
        }
    }

    print(json.dumps(results, indent=2))

    if args.out_json:
        outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(results, f, indent=2)

    if args.out_dir:
        outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)
        # bar: accuracy vs index
        names = ["flat","hnsw"]
        metrics = ["top1_acc","recall_at_k","mrr","average_precision_at_k"]
        X = np.arange(len(metrics)); W = 0.35
        plt.figure(figsize=(8,4.5))
        for i, n in enumerate(names):
            vals = [results["metrics"][n][m] for m in metrics]
            plt.bar(X + (i - (len(names)-1)/2)*W, vals, width=W, label=n)
        plt.xticks(X, metrics, rotation=20, ha="right"); plt.ylim(0,1.0); plt.ylabel("score")
        plt.title("{title}: Brute Force vs HNSW — accuracy")
        plt.legend(); plt.tight_layout()
        plt.savefig(outd / "accuracy_flat_vs_hnsw.png", dpi=150); plt.close()

        # bar: search time (ms)
        plt.figure(figsize=(6,4))
        times = [results["timings_ms"]["flat_search_avg_ms"], results["timings_ms"]["hnsw_search_avg_ms"]]
        plt.bar(["flat","hnsw"], times)
        plt.ylabel("avg search ms/query"); plt.title("{title}: Brute Force vs HNSW — search time")
        plt.tight_layout(); plt.savefig(outd / "time_flat_vs_hnsw.png", dpi=150); plt.close()

        # text file with 1B estimate
        with open(outd / "estimate_1B.txt", "w") as f:
            f.write("=== 1B scale estimates (very rough) ===\n")
            f.write(f"Vectors memory (float32 {dim}D): {results['scaling_estimates_1B']['memory_human']['vectors']}\n")
            f.write(f"HNSW graph memory low/high (M={args.hnsw_M}): "
                    f"{results['scaling_estimates_1B']['memory_human']['hnsw_graph_low']}"
                    f" — {results['scaling_estimates_1B']['memory_human']['hnsw_graph_high']}\n")
            f.write(f"Flat search est per query: {results['scaling_estimates_1B']['flat_search_est_s_per_query']:.2f} s\n")
            f.write(f"HNSW search est per query: {results['scaling_estimates_1B']['hnsw_search_est_s_per_query']:.2f} s\n")
            f.write("Assumptions: flat ~ O(N); HNSW ~ O(efSearch*logN). I/O, caches, PQ/compression ignored.\n")


if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)

vgg_script = common.format(script_name="compare_bruteforce_vs_hnsw_vggface2.py",
                           shortname="bf_vs_hnsw_vggface2",
                           pretrained="vggface2",
                           title="VGGFace2",
                           )
casia_script = common.format(script_name="compare_bruteforce_vs_hnsw_casia.py",
                            shortname="bf_vs_hnsw_casia",
                            pretrained="casia-webface",
                            title="CASIA-WebFace",
                            )