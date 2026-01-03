"""
Compare Flat (brute force) vs HNSW vs LSH on VGGFace2 features.

- Metrics: Top-1 Acc (Hit@1), Recall@K (any-hit), MRR, Precision@K, AP@K, mAP@K
- Distance: cosine (default, via L2-normalize + inner product) or euclidean
- Indexes: FAISS IndexFlat (brute force) vs IndexHNSWFlat (approx) vs LSH (FAISS IndexLSH or binary fallback)
- LSH options:
    * --lsh-bits: number of random-hyperplane bits (default 256)
    * --lsh-rerank: re-rank top-R LSH candidates using exact cosine for better accuracy (default 0 = no rerank)
- Scaling estimates for N=1e9 (configurable via --scaling-target)

Example:
    PYTHONPATH=. python compare_bruteforce_vs_hnsw_vggface2.py \
        --dataset-root storage/mi_eval \
        --k 5 --metric cosine --batch-size 64 --device cpu \
        --hnsw-m 32 --hnsw-efc 100 --hnsw-efs 64 \
        --lsh-bits 256 --lsh-rerank 200 --scaling-target 1000000000
"""

import os
# ---- Safety band-aids for OpenMP / BLAS thread conflicts (pre-import) ----
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import argparse
import gc
import io
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

import faiss


# --------------------------- Embedding model -------------------------------

def build_vggface2_embedder(device: torch.device):
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    feat_dim = 512
    def _embed(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(batch)
    return _embed, feat_dim, "facenet_vggface2"


# --------------------------- Dataset utilities -----------------------------

def make_transforms(img_size=160):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])


def split_gallery_query_by_id(dataset, max_images_per_id=10, seed=1234):
    rng = random.Random(seed)
    id_to_indices: Dict[int, List[int]] = {}
    for idx, (_, cls) in enumerate(dataset.imgs):
        id_to_indices.setdefault(cls, []).append(idx)

    gallery_indices, query_indices = [], []
    labels_gallery, labels_query = [], []

    for cls, idxs in id_to_indices.items():
        if max_images_per_id is not None:
            idxs = idxs[:max_images_per_id]
        if len(idxs) < 2:
            continue
        q_idx = rng.choice(idxs)
        g_idxs = [i for i in idxs if i != q_idx]
        if len(g_idxs) == 0:
            continue
        query_indices.append(q_idx)
        gallery_indices.extend(g_idxs)
        labels_query.append(cls)
        labels_gallery.extend([cls] * len(g_idxs))

    return (Subset(dataset, gallery_indices),
            Subset(dataset, query_indices),
            np.array(labels_gallery, dtype=np.int32),
            np.array(labels_query, dtype=np.int32))


def collate_keep_path(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(targets, dtype=torch.long)


# --------------------------- Metric utilities ------------------------------

def rank_metrics(labels_query: np.ndarray,
                 labels_gallery: np.ndarray,
                 indices: np.ndarray,
                 k: int) -> Dict[str, float]:
    nq, K = indices.shape
    assert K >= k
    top1_hits = 0
    recall_any_hits = 0
    precisions, ap_list, rr_list = [], [], []

    for qi in range(nq):
        qlab = labels_query[qi]
        retrieved = indices[qi, :k]
        retrieved_labels = labels_gallery[retrieved]
        rel = (retrieved_labels == qlab).astype(np.int32)

        if rel[0] == 1:
            top1_hits += 1
        if rel.any():
            recall_any_hits += 1
        precisions.append(rel.sum() / k)

        if rel.sum() > 0:
            cum_rel = np.cumsum(rel)
            precisions_at_i = cum_rel / (np.arange(k) + 1)
            ap = (precisions_at_i * rel).sum() / rel.sum()
        else:
            ap = 0.0
        ap_list.append(ap)

        ranks = np.where(rel == 1)[0]
        rr_list.append(1.0 / (ranks[0] + 1) if ranks.size > 0 else 0.0)

    return {
        "top1_acc": top1_hits / nq,
        "recall_at_k": recall_any_hits / nq,
        "precision_at_k": float(np.mean(precisions)),
        "average_precision_at_k": float(np.mean(ap_list)),
        "mean_average_precision_at_k": float(np.mean(ap_list)),
        "mrr": float(np.mean(rr_list)),
    }


# --------------------------- FAISS helpers ---------------------------------

def normalize_if_cosine(x: np.ndarray, metric: str) -> np.ndarray:
    if metric.lower() == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        x = x / norms
    return x.astype('float32')


def make_index_flat(d: int, metric: str):
    if metric.lower() == "cosine":
        return faiss.IndexFlatIP(d), "IndexFlatIP (cosine via IP)"
    elif metric.lower() == "euclidean":
        return faiss.IndexFlatL2(d), "IndexFlatL2"
    else:
        raise ValueError("metric must be one of {'cosine','euclidean'}")


def make_index_hnsw(d: int, metric: str, M: int, efC: int, efS: int):
    if metric.lower() == "cosine":
        metric_type = faiss.METRIC_INNER_PRODUCT
        index = faiss.index_factory(d, f"HNSW{M},Flat", metric_type)
        desc = f"IndexHNSWFlat(M={M}, efC={efC}, efS={efS}, metric=IP)"
    elif metric.lower() == "euclidean":
        metric_type = faiss.METRIC_L2
        index = faiss.index_factory(d, f"HNSW{M},Flat", metric_type)
        desc = f"IndexHNSWFlat(M={M}, efC={efC}, efS={efS}, metric=L2)"
    else:
        raise ValueError("metric must be one of {'cosine','euclidean'}")

    hnsw = faiss.downcast_index(index).hnsw
    hnsw.efConstruction = int(efC)
    hnsw.efSearch = int(efS)
    return index, desc


# ---- LSH: FAISS IndexLSH or binary fallback (IndexBinaryFlat) --------------

def pack_bits(bits: np.ndarray) -> np.ndarray:
    """
    bits: (N, nbits) uint8 {0,1} -> packed (N, ceil(nbits/8)) uint8
    """
    N, nbits = bits.shape
    nbytes = (nbits + 7) // 8
    out = np.zeros((N, nbytes), dtype='uint8')
    for i in range(nbits):
        out[:, i // 8] |= (bits[:, i] << (i % 8))
    return out


def lsh_hash(x: np.ndarray, rp: np.ndarray) -> np.ndarray:
    """
    x: (N, d) float32 normalized vectors
    rp: (nbits, d) float32 random hyperplanes ~ N(0,1)
    returns packed binary codes (N, nbytes)
    """
    proj = x @ rp.T  # (N, nbits)
    bits = (proj >= 0).astype('uint8')
    return pack_bits(bits)


def make_index_lsh(d: int, nbits: int):
    """
    Try FAISS IndexLSH (stores/transforms floats internally). If not available,
    fall back to manual random-hyperplane hashing + IndexBinaryFlat on codes.
    Returns:
        index, desc, rp_or_None, mode
        mode in {"faiss_lsh","binary_lsh"}
    """
    try:
        index = faiss.IndexLSH(d, nbits)
        desc = f"IndexLSH(nbits={nbits})"
        return index, desc, None, "faiss_lsh"
    except Exception:
        rp = np.random.randn(nbits, d).astype('float32')
        index = faiss.IndexBinaryFlat(nbits)  # Hamming distance
        desc = f"BinaryLSH(nbits={nbits})"
        return index, desc, rp, "binary_lsh"


def index_mem_bytes(index) -> int:
    try:
        b = faiss.serialize_index(index)
        return len(b)
    except Exception:
        try:
            b = faiss.serialize_index_binary(index)  # for binary indices
            return len(b)
        except Exception:
            return -1


# --------------------------- Embedding & eval -------------------------------

def embed_loader(dataloader: DataLoader, embed_fn, device: torch.device) -> np.ndarray:
    embs: List[np.ndarray] = []
    for xb, _ in tqdm(dataloader, desc="Embedding", leave=False):
        xb = xb.to(device, non_blocking=False)
        feats = embed_fn(xb)
        embs.append(feats.detach().cpu().numpy().astype('float32'))
    return np.concatenate(embs, axis=0)


def evaluate_index_float(index, gallery: np.ndarray, labels_gallery: np.ndarray,
                         query: np.ndarray, labels_query: np.ndarray, k: int) -> Tuple[Dict, float]:
    t0 = time.perf_counter()
    index.add(gallery)
    build_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    D, I = index.search(query, k)
    search_time = time.perf_counter() - t1
    avg_search_ms = (search_time / len(query)) * 1000.0

    metrics = rank_metrics(labels_query, labels_gallery, I, k)
    metrics["avg_search_ms"] = avg_search_ms
    metrics["build_time_s"] = build_time
    mem = index_mem_bytes(index)
    if mem >= 0:
        metrics["index_mem_mb"] = mem / (1024 * 1024)
    return metrics, avg_search_ms, I


def evaluate_index_binary(index, gallery_codes: np.ndarray, labels_gallery: np.ndarray,
                          query_codes: np.ndarray, labels_query: np.ndarray,
                          k: int) -> Tuple[Dict, float, np.ndarray]:
    t0 = time.perf_counter()
    index.add(gallery_codes)
    build_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    D, I = index.search(query_codes, k)
    search_time = time.perf_counter() - t1
    avg_search_ms = (search_time / len(query_codes)) * 1000.0

    metrics = rank_metrics(labels_query, labels_gallery, I, k)
    metrics["avg_search_ms"] = avg_search_ms
    metrics["build_time_s"] = build_time
    try:
        mem = index_mem_bytes(index)
        if mem >= 0:
            metrics["index_mem_mb"] = mem / (1024 * 1024)
    except Exception:
        pass
    return metrics, avg_search_ms, I


def lsh_rerank(indices: np.ndarray, query: np.ndarray, gallery: np.ndarray,
               k: int, rerank_k: int, metric: str) -> np.ndarray:
    """
    Given LSH top-candidate indices per query, optionally re-rank the first R using exact scores.
    Returns final indices (nq, k).
    """
    nq, Kcand = indices.shape
    R = min(rerank_k, Kcand)
    if R <= 0:
        return indices[:, :k]

    out = np.empty((nq, k), dtype=np.int64)
    for qi in range(nq):
        cand = indices[qi, :R]
        if metric.lower() == "cosine":
            # gallery/query are normalized already: cosine == dot
            sims = gallery[cand] @ query[qi]
            order = np.argsort(-sims)[:k]
        else:
            # euclidean: compute distances, sort ascending
            diffs = gallery[cand] - query[qi]
            dists = np.sum(diffs * diffs, axis=1)
            order = np.argsort(dists)[:k]
        out[qi] = cand[order]
    return out


# --------------------------- Scaling estimates ------------------------------

def human_bytes(b: int) -> str:
    units = [("TB", 1<<40), ("GB", 1<<30), ("MB", 1<<20)]
    for name, size in units:
        if b >= size:
            return f"{b/size:.1f}{name}"
    return f"{b}B"


def estimate_scaling(N_target: int,
                     flat_avg_ms: float,
                     hnsw_avg_ms: float,
                     lsh_avg_ms: float,
                     n: int,
                     dim: int,
                     M: int,
                     efSearch: int,
                     lsh_bits: int,
                     dtype_bytes: int = 4):
    # Flat (linear)
    per_vec_ms = flat_avg_ms / max(n, 1)
    flat_search_est_s = (per_vec_ms * N_target) / 1000.0

    # HNSW (~ log N)
    log_ratio = math.log(N_target) / max(math.log(max(n, 2)), 1.0)
    hnsw_search_est_s = (hnsw_avg_ms / 1000.0) * log_ratio

    # LSH baseline (linear over hash codes; very fast Hamming but still ~N unless you bucket)
    per_code_ms = lsh_avg_ms / max(n, 1)
    lsh_search_est_s = (per_code_ms * N_target) / 1000.0

    # Memory
    vectors_bytes = N_target * dim * dtype_bytes  # float32 vectors
    hnsw_graph_low = N_target * M * 4
    hnsw_graph_high = N_target * M * 8
    lsh_codes_bytes = N_target * (lsh_bits // 8)

    return {
        "flat": {
            "search_est_s_per_query": flat_search_est_s,
            "assumptions": {"linear_per_vector_ms": per_vec_ms}
        },
        "hnsw": {
            "search_est_s_per_query": hnsw_search_est_s,
            "assumptions": {"log_scaling_factor": log_ratio, "efSearch": efSearch}
        },
        "lsh": {
            "search_est_s_per_query": lsh_search_est_s,
            "assumptions": {
                "linear_per_code_ms": per_code_ms,
                "nbits": lsh_bits,
                "note": "Estimate is for flat Hamming scan. Bucketed LSH can be sublinear but depends on tables/probes."
            }
        },
        "memory_bytes": {
            "vectors": vectors_bytes,
            "hnsw_graph_low": hnsw_graph_low,
            "hnsw_graph_high": hnsw_graph_high,
            "lsh_codes": lsh_codes_bytes
        },
        "memory_human": {
            "vectors": human_bytes(vectors_bytes),
            "hnsw_graph_low": human_bytes(hnsw_graph_low),
            "hnsw_graph_high": human_bytes(hnsw_graph_high),
            "lsh_codes": human_bytes(lsh_codes_bytes)
        }
    }


# --------------------------- Main pipeline ---------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare Flat vs HNSW vs LSH on VGGFace2 embeddings")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit-identities", type=int, default=None)
    parser.add_argument("--max-images-per-id", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)

    # HNSW params
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-efc", type=int, default=100)
    parser.add_argument("--hnsw-efs", type=int, default=64)

    # LSH params
    parser.add_argument("--lsh-bits", type=int, default=256)
    parser.add_argument("--lsh-rerank", type=int, default=0, help="re-rank top-R per query with exact metric (0 disables)")

    # Scaling
    parser.add_argument("--scaling-target", type=int, default=1_000_000_000, help="N for scaling estimates")

    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    device = torch.device(args.device)

    # Dataset
    tfm = make_transforms(args.img_size)
    ds = datasets.ImageFolder(args.dataset_root, transform=tfm)

    # Optionally limit identities
    if args.limit_identities is not None:
        keep_classes = sorted(ds.class_to_idx.values())[:args.limit_identities]
        keep_idx = [i for i, (_, cls) in enumerate(ds.imgs) if cls in keep_classes]
        ds = Subset(ds, keep_idx)

        class SubsetWithAttrs(Subset): pass
        ds = SubsetWithAttrs(ds.dataset, ds.indices)
        imgs = [ds.dataset.imgs[i] for i in ds.indices]
        class_names = sorted(list({Path(p).parent.name for p, _ in imgs}))
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        remapped = []
        for p, _ in imgs:
            name = Path(p).parent.name
            remapped.append((p, class_to_idx[name]))
        ds.imgs = remapped
        ds.classes = class_names
        ds.class_to_idx = class_to_idx

    # Split G/Q
    gallery_ds, query_ds, labels_gallery, labels_query = split_gallery_query_by_id(
        ds, max_images_per_id=args.max_images_per_id, seed=args.seed
    )
    if len(query_ds) == 0 or len(gallery_ds) == 0:
        raise RuntimeError("Empty gallery or query after split; tweak --max-images-per-id or --limit-identities.")

    # Loaders
    g_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False, persistent_workers=False,
                          collate_fn=collate_keep_path)
    q_loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False, persistent_workers=False,
                          collate_fn=collate_keep_path)

    # Embed
    embed_fn, feat_dim, model_name = build_vggface2_embedder(device)
    print(f"[embed] Model: {model_name} | feat_dim={feat_dim} | device={device}")
    t0 = time.perf_counter()
    gallery_embs = embed_loader(g_loader, embed_fn, device)
    query_embs   = embed_loader(q_loader, embed_fn, device)
    embed_time = time.perf_counter() - t0
    print(f"[embed] Done in {embed_time:.2f}s | G={len(gallery_embs)} Q={len(query_embs)}")

    # Normalize if cosine
    gallery = normalize_if_cosine(gallery_embs, args.metric)
    query   = normalize_if_cosine(query_embs, args.metric)

    # ---------------- Flat ----------------
    flat_index, flat_desc = make_index_flat(gallery.shape[1], args.metric)
    print(f"\n[FLAT] Building {flat_desc}")
    flat_metrics, flat_avg_ms, _ = evaluate_index_float(flat_index, gallery, labels_gallery,
                                                        query, labels_query, args.k)
    del flat_index; gc.collect()

    # ---------------- HNSW ----------------
    hnsw_index, hnsw_desc = make_index_hnsw(gallery.shape[1], args.metric,
                                            args.hnsw_m, args.hnsw_efc, args.hnsw_efs)
    print(f"\n[HNSW] Building {hnsw_desc}")
    hnsw_metrics, hnsw_avg_ms, _ = evaluate_index_float(hnsw_index, gallery, labels_gallery,
                                                        query, labels_query, args.k)
    del hnsw_index; gc.collect()

    # ---------------- LSH -----------------
    lsh_index, lsh_desc, rp, mode = make_index_lsh(gallery.shape[1], args.lsh_bits)
    print(f"\n[LSH] Building {lsh_desc} (mode={mode}, rerank={args.lsh_rerank})")

    if mode == "faiss_lsh":
        # FAISS transforms floats internally
        lsh_metrics_raw, lsh_avg_ms_raw, I = evaluate_index_float(lsh_index, gallery, labels_gallery,
                                                                  query, labels_query, max(args.k, args.lsh_rerank, args.k))
        # Optional re-rank
        if args.lsh_rerank > 0:
            t_rr0 = time.perf_counter()
            I2 = lsh_rerank(I, query, gallery, args.k, args.lsh_rerank, args.metric)
            t_rr = time.perf_counter() - t_rr0
            # Recompute metrics on re-ranked indices
            lsh_metrics = rank_metrics(labels_query, labels_gallery, I2, args.k)
            lsh_metrics["avg_search_ms"] = lsh_metrics_raw["avg_search_ms"] + (t_rr / len(query)) * 1000.0
            lsh_metrics["build_time_s"] = lsh_metrics_raw["build_time_s"]
            lsh_metrics["index_mem_mb"] = lsh_metrics_raw.get("index_mem_mb", None)
            lsh_avg_ms = lsh_metrics["avg_search_ms"]
        else:
            lsh_metrics = lsh_metrics_raw
            lsh_avg_ms = lsh_metrics_raw["avg_search_ms"]
    else:
        # Binary fallback: pre-hash
        gallery_codes = lsh_hash(gallery, rp)
        query_codes   = lsh_hash(query, rp)
        # search at least rerank_k
        search_k = max(args.k, args.lsh_rerank, args.k)
        lsh_metrics_raw, lsh_avg_ms_raw, I = evaluate_index_binary(lsh_index, gallery_codes, labels_gallery,
                                                                   query_codes, labels_query, search_k)
        if args.lsh_rerank > 0:
            t_rr0 = time.perf_counter()
            I2 = lsh_rerank(I, query, gallery, args.k, args.lsh_rerank, args.metric)
            t_rr = time.perf_counter() - t_rr0
            lsh_metrics = rank_metrics(labels_query, labels_gallery, I2, args.k)
            lsh_metrics["avg_search_ms"] = lsh_metrics_raw["avg_search_ms"] + (t_rr / len(query)) * 1000.0
            lsh_metrics["build_time_s"] = lsh_metrics_raw["build_time_s"]
            lsh_metrics["index_mem_mb"] = lsh_metrics_raw.get("index_mem_mb", None)
            lsh_avg_ms = lsh_metrics["avg_search_ms"]
        else:
            lsh_metrics = lsh_metrics_raw
            lsh_avg_ms = lsh_metrics_raw["avg_search_ms"]

    # ---------------- Summary + scaling ----------------
    def short(d):
        keys = ["top1_acc", "recall_at_k", "mrr", "precision_at_k",
                "average_precision_at_k", "mean_average_precision_at_k",
                "avg_search_ms", "build_time_s", "index_mem_mb"]
        out = {}
        for k in keys:
            v = d.get(k, None)
            if v is None:
                continue
            out[k] = round(float(v), 6)
        return out

    scaling = estimate_scaling(args.scaling_target,
                               flat_avg_ms, hnsw_avg_ms, lsh_avg_ms,
                               n=int(gallery.shape[0]),
                               dim=int(gallery.shape[1]),
                               M=args.hnsw_m,
                               efSearch=args.hnsw_efs,
                               lsh_bits=args.lsh_bits)

    summary = {
        "dataset": "VGGFace2-derived (features from facenet_pytorch)",
        "metric": args.metric,
        "k": args.k,
        "gallery_size": int(gallery.shape[0]),
        "query_size": int(query.shape[0]),
        "model": model_name,
        "flat_desc": flat_desc,
        "hnsw_desc": hnsw_desc,
        "lsh_desc": lsh_desc,
        "flat": short(flat_metrics),
        "hnsw": short(hnsw_metrics),
        "lsh": short(lsh_metrics),
        "hnsw_params": {"M": args.hnsw_m, "efConstruction": args.hnsw_efc, "efSearch": args.hnsw_efs},
        "lsh_params": {"nbits": args.lsh_bits, "rerank": args.lsh_rerank, "mode": mode},
        "scaling_estimates": scaling
    }

    print("\n==== RESULTS ====")
    print(json.dumps(summary, indent=2))

    # Optional TSV log
    out_tsv = Path("vggface2_flat_hnsw_lsh.tsv")
    with io.open(out_tsv, "w", encoding="utf-8") as f:
        f.write("index\tmetric\tk\tgallery\tquery\t"
                "top1_acc\trecall_at_k\tmrr\tprecision_at_k\tap_at_k\tmap_at_k\tavg_search_ms\tbuild_time_s\tindex_mem_mb\n")
        def row(name, m):
            f.write(f"{name}\t{args.metric}\t{args.k}\t{gallery.shape[0]}\t{query.shape[0]}\t"
                    f"{m.get('top1_acc',-1)}\t{m.get('recall_at_k',-1)}\t{m.get('mrr',-1)}\t"
                    f"{m.get('precision_at_k',-1)}\t{m.get('average_precision_at_k',-1)}\t{m.get('mean_average_precision_at_k',-1)}\t"
                    f"{m.get('avg_search_ms',-1)}\t{m.get('build_time_s',-1)}\t{m.get('index_mem_mb',-1)}\n")
        row("FLAT", flat_metrics)
        row("HNSW", hnsw_metrics)
        row("LSH",  lsh_metrics)
    print(f"[write] Wrote {out_tsv.resolve()}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"[script] runtime={time.perf_counter()-t0:.2f}s")
