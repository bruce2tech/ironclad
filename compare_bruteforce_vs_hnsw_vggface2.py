"""
compare_bruteforce_vs_hnsw_vggface2.py

Compare Brute Force (FAISS Flat) vs HNSW using FaceNet (InceptionResnetV1)
pretrained on VGGFace2. Supports reusing a cache (.npz) produced by the
HNSW-vs-LSH script to guarantee identical gallery/query splits and embeddings.

Outputs:
- JSON summary with metrics & timings
- Optional PNG plots
- Rough 1B-vector scaling & memory estimates

Dataset structure (used ONLY when cache is not provided/found):
  DATASET_ROOT/
    gallery/<label>/*.jpg
    queries/<label>/*.jpg

Example:
  PYTHONPATH=. python compare_bruteforce_vs_hnsw_vggface2.py \
    --dataset-root storage/mi_eval \
    --k 5 \
    --metric cosine \
    --hnsw-M 32 --hnsw-efC 100 --hnsw-efS 64 \
    --faiss-threads 1 \
    --cache storage/mi_eval/vggface2_split_embeddings.npz \
    --out-json storage/mi_eval/bf_vs_hnsw_vggface2.json \
    --out-dir  storage/mi_eval/bf_vs_hnsw_vggface2_plots
"""

# ------------------- Safety knobs for OpenMP/BLAS on macOS -------------------
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")

# ------------------------------- Imports -------------------------------------
import argparse, json, sys, time, math
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Your embedding stack (used only when cache not provided/found)
from ironclad.modules.extraction.preprocessing import Preprocessing
from ironclad.modules.extraction.embedding import Embedding

import torch, faiss
torch.set_num_threads(1)
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

# ------------------------------- Cache IO ------------------------------------
def _cache_meta(model_name, feat_dim, metric, img_size, dataset_root):
    return {
        "model": model_name,
        "dim": int(feat_dim),
        "metric": metric,
        "img_size": int(img_size),
        "dataset_root": str(Path(dataset_root).resolve())
    }

def load_cache(cache_path: Path, cache_strict: bool, expect_model: str,
               expect_metric: str, expect_dim: int, expect_img_size: int):
    data = np.load(cache_path, allow_pickle=True)
    gallery_embs = data["gallery_embs"].astype("float32")
    query_embs   = data["query_embs"].astype("float32")
    labels_gallery = data["labels_gallery"]
    labels_query   = data["labels_query"]

    meta = {}
    if "meta_json" in data:
        raw = data["meta_json"]
        # np.savez_compressed sometimes stores a 0-d array for strings
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        meta = json.loads(raw)

    mismatches = []
    if meta:
        if meta.get("model") != expect_model: mismatches.append("model")
        if int(meta.get("dim", expect_dim)) != int(gallery_embs.shape[1]): mismatches.append("dim")
        if meta.get("metric") != expect_metric: mismatches.append("metric")
        if int(meta.get("img_size", expect_img_size)) != int(expect_img_size): mismatches.append("img_size")
    if cache_strict and mismatches:
        raise ValueError(f"[cache] Metadata mismatch on: {', '.join(mismatches)}")

    print(f"[cache] Loaded {cache_path} | G={len(gallery_embs)} Q={len(query_embs)} | meta={meta or 'n/a'}")
    return gallery_embs, query_embs, labels_gallery, labels_query, meta

def save_cache(cache_path: Path, gallery_embs, query_embs, labels_gallery, labels_query, meta: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path,
        gallery_embs=gallery_embs, query_embs=query_embs,
        labels_gallery=labels_gallery, labels_query=labels_query,
        meta_json=json.dumps(meta))
    print(f"[cache] Saved split+embeddings -> {cache_path}")

# ------------------------------- Data utils ----------------------------------
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

# ------------------------------- Math/utils ----------------------------------
def normalize_if_cosine(x: np.ndarray, metric: str) -> np.ndarray:
    if metric.lower() == "cosine":
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        x = x / n
    return x.astype("float32")

def human_bytes(num: float) -> str:
    for unit in ["B","KB","MB","GB","TB","PB","EB"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}ZB"

def estimate_memory_bytes(n: int, dim: int, M: int = 32) -> Dict[str, int]:
    # Float32 vector store + crude HNSW graph bounds (ignore overheads/levels).
    vec_bytes = n * dim * 4
    graph_low = n * M * 4
    graph_high = n * M * 8
    return {"vectors_bytes": vec_bytes, "hnsw_graph_low_bytes": graph_low, "hnsw_graph_high_bytes": graph_high}

def index_mem_bytes(index) -> int:
    try:
        b = faiss.serialize_index(index)
        return len(b)
    except Exception:
        try:
            b = faiss.serialize_index_binary(index)
            return len(b)
        except Exception:
            return -1

# ------------------------------- FAISS builders ------------------------------
def build_flat_index(dim: int, metric: str = "cosine"):
    if metric.lower() == "cosine":
        return faiss.IndexFlatIP(dim)
    elif metric.lower() == "euclidean":
        return faiss.IndexFlatL2(dim)
    else:
        raise ValueError("Only 'cosine' or 'euclidean' supported.")

def build_hnsw_index(dim: int, M: int, efC: int, metric: str = "cosine"):
    # index_factory is the most version-proof way to set metric for HNSWFlat
    if metric.lower() == "cosine":
        metric_type = faiss.METRIC_INNER_PRODUCT
        index = faiss.index_factory(dim, f"HNSW{M},Flat", metric_type)
        desc = f"IndexHNSWFlat(M={M}, efC={efC}, metric=IP)"
    else:
        metric_type = faiss.METRIC_L2
        index = faiss.index_factory(dim, f"HNSW{M},Flat", metric_type)
        desc = f"IndexHNSWFlat(M={M}, efC={efC}, metric=L2)"
    # set efConstruction
    hnsw = faiss.downcast_index(index).hnsw
    hnsw.efConstruction = int(efC)
    return index, desc

# ------------------------------- Metrics -------------------------------------
def rank_metrics(labels_query: np.ndarray,
                 labels_gallery: np.ndarray,
                 indices: np.ndarray,
                 k: int) -> Dict[str, float]:
    nq, K = indices.shape
    assert K >= k
    top1 = 0
    any_hit = 0
    precisions, ap_list, rr_list = [], [], []
    for qi in range(nq):
        qlab = labels_query[qi]
        cand = indices[qi, :k]
        rel = (labels_gallery[cand] == qlab).astype(np.int32)
        if rel[0] == 1:
            top1 += 1
        if rel.any():
            any_hit += 1
        precisions.append(rel.sum() / k)
        if rel.sum() > 0:
            cum = np.cumsum(rel)
            prec_at_i = cum / (np.arange(k) + 1)
            ap = (prec_at_i * rel).sum() / rel.sum()
        else:
            ap = 0.0
        ap_list.append(ap)
        pos = np.where(rel == 1)[0]
        rr_list.append(1.0 / (pos[0] + 1) if pos.size > 0 else 0.0)
    return {
        "top1_acc": top1 / nq,
        "recall_at_k": any_hit / nq,
        "precision_at_k": float(np.mean(precisions)),
        "average_precision_at_k": float(np.mean(ap_list)),
        "mean_average_precision_at_k": float(np.mean(ap_list)),
        "mrr": float(np.mean(rr_list)),
    }

def evaluate_index(index, gallery: np.ndarray, labels_gallery: np.ndarray,
                   query: np.ndarray, labels_query: np.ndarray, k: int) -> Dict[str, float]:
    t0 = time.perf_counter()
    index.add(gallery)
    build_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    D, I = index.search(query, k)
    t_search = time.perf_counter() - t1
    avg_search_ms = (t_search / len(query)) * 1000.0

    m = rank_metrics(labels_query, labels_gallery, I, k)
    m["avg_search_ms"] = avg_search_ms
    m["build_time_s"] = build_time_s
    mem = index_mem_bytes(index)
    if mem >= 0:
        m["index_mem_mb"] = mem / (1024 * 1024)
    return m

# ------------------------------- Fallback embed ------------------------------
def embed_from_folders(dataset_root: Path, device: str):
    """Used only when cache is missing. Returns (G, Q, labels_gallery_int, labels_query_int)."""
    gal_items = list_images_with_labels(dataset_root / "gallery")
    qry_items = list_images_with_labels(dataset_root / "queries")
    if not gal_items or not qry_items:
        print(f"[ERROR] Expected images under {dataset_root}/gallery and /queries", file=sys.stderr)
        sys.exit(2)

    pre = Preprocessing()
    emb = Embedding(pretrained="vggface2", device=device)

    G_vecs, G_labels_str = [], []
    for p, lab in gal_items:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        x = pre.process(img)
        z = emb.encode(x).astype(np.float32)
        G_vecs.append(z); G_labels_str.append(lab)

    Q_vecs, Q_labels_str = [], []
    for p, lab in qry_items:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        x = pre.process(img)
        z = emb.encode(x).astype(np.float32)
        Q_vecs.append(z); Q_labels_str.append(lab)

    if not G_vecs or not Q_vecs:
        print("[ERROR] No embeddings computed.", file=sys.stderr); sys.exit(2)

    G = np.stack(G_vecs, axis=0)
    Q = np.stack(Q_vecs, axis=0)

    # Map string labels -> int ids (consistent across gallery+query)
    uniq = sorted(set(G_labels_str + Q_labels_str))
    lut = {lab: i for i, lab in enumerate(uniq)}
    labels_gallery = np.array([lut[x] for x in G_labels_str], dtype=np.int32)
    labels_query   = np.array([lut[x] for x in Q_labels_str], dtype=np.int32)

    return G, Q, labels_gallery, labels_query

# --------------------------------- Main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--metric", default="cosine", choices=["cosine","euclidean"])

    ap.add_argument("--hnsw-M", type=int, default=32)
    ap.add_argument("--hnsw-efC", type=int, default=80)
    ap.add_argument("--hnsw-efS", type=int, default=64)

    ap.add_argument("--faiss-threads", type=int, default=1,
                    help="FAISS OMP threads (1 is safest on macOS)")

    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-dir", default=None)

    ap.add_argument("--cache", type=str, default=None,
        help="Path to .npz with gallery/query embeddings & labels (from HNSW-vs-LSH).")
    ap.add_argument("--cache-strict", action="store_true",
        help="Refuse to load cache if metadata mismatches.")
    ap.add_argument("--img-size", type=int, default=160,
        help="Only used for cache metadata sanity; not for embedding here.")
    args = ap.parse_args()

    # Consistent metadata with your HNSW-vs-LSH script
    expect_model = "facenet_vggface2"
    expect_dim   = 512

    # Threads / device
    try:
        faiss.omp_set_num_threads(max(1, int(args.faiss_threads)))
    except Exception:
        pass
    device = args.device

    # ---- Load from cache if available ----
    cache_path = Path(args.cache) if args.cache else None
    loaded_from_cache = False
    if cache_path and cache_path.exists():
        gallery_embs, query_embs, labels_gallery, labels_query, _meta = load_cache(
            cache_path=cache_path,
            cache_strict=args.cache_strict,
            expect_model=expect_model,
            expect_metric=args.metric,
            expect_dim=expect_dim,
            expect_img_size=args.img_size,
        )
        G = gallery_embs
        Q = query_embs
        # labels from cache are already numeric (np.int32), but normalize dtype anyway
        labels_gallery = labels_gallery.astype(np.int32)
        labels_query   = labels_query.astype(np.int32)
        loaded_from_cache = True
    else:
        # Fallback: embed from folder structure
        dataset_root = Path(args.dataset_root)
        G, Q, labels_gallery, labels_query = embed_from_folders(dataset_root, device)

    # Normalize if cosine
    G = normalize_if_cosine(G, args.metric)
    Q = normalize_if_cosine(Q, args.metric)
    dim = int(G.shape[1])

    # --------------------- Build & Evaluate: Flat ----------------------------
    flat = build_flat_index(dim, metric=args.metric)
    flat_metrics = evaluate_index(flat, G, labels_gallery, Q, labels_query, args.k)
    flat_build_ms = flat_metrics["build_time_s"] * 1e3

    # --------------------- Build & Evaluate: HNSW ----------------------------
    hnsw, hnsw_desc = build_hnsw_index(dim, args.hnsw_M, args.hnsw_efC, metric=args.metric)
    hnsw.hnsw.efSearch = args.hnsw_efS
    hnsw_metrics = evaluate_index(hnsw, G, labels_gallery, Q, labels_query, args.k)
    hnsw_build_ms = hnsw_metrics["build_time_s"] * 1e3

    # --------------------- 1B scaling (very rough) ---------------------------
    N_cur = int(G.shape[0])
    N_target = 1_000_000_000

    # Flat: linear
    per_vec_ms = flat_metrics["avg_search_ms"] / max(1, N_cur)
    est_flat_s = (per_vec_ms * N_target) / 1e3

    # HNSW: ~ log N (natural log ratio; matches your earlier runs)
    log_ratio = math.log(N_target) / max(math.log(max(N_cur, 2)), 1.0)
    est_hnsw_s = (hnsw_metrics["avg_search_ms"] * log_ratio) / 1e3

    mem_est = estimate_memory_bytes(N_target, dim, M=args.hnsw_M)

    # ----------------------------- Results -----------------------------------
    dataset_root = str(Path(args.dataset_root).resolve())
    results = {
        "config": {
            "dataset_root": dataset_root,
            "k": args.k,
            "model": "vggface2",
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
            "flat_search_avg_ms": flat_metrics["avg_search_ms"],
            "hnsw_search_avg_ms": hnsw_metrics["avg_search_ms"]
        },
        "metrics": {
            "flat": {k: v for k, v in flat_metrics.items()},
            "hnsw": {k: v for k, v in hnsw_metrics.items()},
        },
        "hnsw_desc": hnsw_desc,
        "scaling_estimates_1B": {
            "flat_search_est_s_per_query": est_flat_s,
            "hnsw_search_est_s_per_query": est_hnsw_s,
            "assumptions": {
                "flat_linear_per_vector_ms": per_vec_ms,
                "hnsw_log_scaling_factor": log_ratio,
                "note": "HNSW assumes ~ efSearch*logN; real performance depends on data/params."
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

    # ----------------------------- Save JSON ---------------------------------
    if args.out_json:
        outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[write] {outp.resolve()}")

    # ----------------------------- Plots -------------------------------------
    if args.out_dir:
        outd = Path(args.out_dir); outd.mkdir(parents=True, exist_ok=True)

        # Accuracy bars
        names = ["flat","hnsw"]
        mnames = ["top1_acc","recall_at_k","mrr","average_precision_at_k"]
        X = np.arange(len(mnames)); W = 0.35
        plt.figure(figsize=(8,4.5))
        for i, n in enumerate(names):
            vals = [results["metrics"][n][m] for m in mnames]
            plt.bar(X + (i - (len(names)-1)/2)*W, vals, width=W, label=n)
        plt.xticks(X, mnames, rotation=20, ha="right"); plt.ylim(0,1.0); plt.ylabel("score")
        plt.title("VGGFace2: Brute Force vs HNSW — accuracy")
        plt.legend(); plt.tight_layout()
        plt.savefig(outd / "accuracy_flat_vs_hnsw.png", dpi=150); plt.close()

        # Search time bars
        plt.figure(figsize=(6,4))
        times = [results["timings_ms"]["flat_search_avg_ms"], results["timings_ms"]["hnsw_search_avg_ms"]]
        plt.bar(["flat","hnsw"], times)
        plt.ylabel("avg search ms/query"); plt.title("VGGFace2: Brute Force vs HNSW — search time")
        plt.tight_layout(); plt.savefig(outd / "time_flat_vs_hnsw.png", dpi=150); plt.close()

        # 1B estimate text
        with open(outd / "estimate_1B.txt", "w") as f:
            f.write("=== 1B scale estimates (very rough) ===\n")
            f.write(f"Vectors memory (float32 {dim}D): {results['scaling_estimates_1B']['memory_human']['vectors']}\n")
            f.write(f"HNSW graph memory low/high (M={args.hnsw_M}): "
                    f"{results['scaling_estimates_1B']['memory_human']['hnsw_graph_low']}"
                    f" — {results['scaling_estimates_1B']['memory_human']['hnsw_graph_high']}\n")
            f.write(f"Flat search est per query: {results['scaling_estimates_1B']['flat_search_est_s_per_query']:.6f} s\n")
            f.write(f"HNSW search est per query: {results['scaling_estimates_1B']['hnsw_search_est_s_per_query']:.6f} s\n")
            f.write("Assumptions: flat ~ O(N); HNSW ~ O(efSearch*logN). Overheads/compression/caches ignored.\n")
        print(f"[write] {outd.resolve()}/(plots + estimate_1B.txt)")

if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"[script] runtime={time.perf_counter()-t0:.2f}s")