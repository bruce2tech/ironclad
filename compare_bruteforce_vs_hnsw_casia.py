"""
Compare Brute Force (Flat) vs HNSW on CASIA-WebFace for face retrieval.

- Metrics: Top-1 Acc (Hit@1), Recall@K (any-hit), MRR, Precision@K, AP@K, mAP@K
- Distance: cosine (default, via L2-normalize + inner product) or euclidean
- Indexes: FAISS IndexFlat (brute force) vs IndexHNSWFlat (approximate)
- Robust to Apple Silicon OpenMP/libomp conflicts (safe env & single-threading)

Example:
    PYTHONPATH=. python compare_bruteforce_vs_hnsw_casia.py \
        --dataset-root /data/CASIA-WebFace \
        --k 5 --limit-identities 500 --max-images-per-id 10 \
        --metric cosine --hnsw-m 32 --hnsw-efc 100 --hnsw-efs 64 \
        --batch-size 64 --device cpu
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
import random
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from facenet_pytorch import InceptionResnetV1
# Optional, nicer progress
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# FAISS
import faiss


# --------------------------- Embedding model -------------------------------

class IdentityEncoder(nn.Module):
    """Fallback: global-average-pool features from torchvision ResNet50 (ImageNet)."""
    def __init__(self, arch="resnet50", pretrained=True):
        super().__init__()
        if arch == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
        else:
            raise ValueError("Unsupported arch for IdentityEncoder")
    def forward(self, x):
        return self.backbone(x)


def build_embedder(device: torch.device, requested: str = "auto"):
    """
    Try facenet-pytorch InceptionResnetV1 (VGGFace2 weights). Fallback to ResNet50.
    """
    if requested in ("facenet", "auto"):
        try:
            from facenet_pytorch import InceptionResnetV1
            model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            feat_dim = 512
            def _embed(batch: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    return model(batch)
            return _embed, feat_dim, "facenet_vggface2"
        except Exception as e:
            if requested == "facenet":
                raise
            print(f"[WARN] facenet_pytorch unavailable ({e}). Falling back to ResNet50 ImageNet features.")
    # Fallback
    net = IdentityEncoder().eval().to(device)
    feat_dim = net.feature_dim
    def _embed(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return net(batch)
    return _embed, feat_dim, "resnet50_imagenet"

def load_embedder(model_name: str, device: str):
    # model_name is "casia-webface" or "vggface2"
    embedder = InceptionResnetV1(pretrained=model_name).eval().to(device)
    feat_dim = 512
    printed_name = f"facenet_{model_name.replace('-', '')}"
    return embedder, feat_dim, printed_name


# --------------------------- Dataset utilities -----------------------------

def make_transforms(img_size=160):
    # Simple center crop & normalization; CASIA images vary in alignment/quality.
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])


def split_gallery_query_by_id(dataset, max_images_per_id=10, seed=1234):
    """
    For each identity, choose 1 query image at random, and up to (max_images_per_id-1) as gallery.
    Filter out identities with <2 images after capping (so query & gallery are disjoint).
    Returns: (gallery_subset, query_subset, labels_gallery, labels_query)
    """
    rng = random.Random(seed)
    # Build mapping: class_idx -> list of dataset indices
    id_to_indices: Dict[int, List[int]] = {}
    for idx, (_, cls) in enumerate(dataset.imgs):  # imgs is a list[(path, class)]
        id_to_indices.setdefault(cls, []).append(idx)

    gallery_indices, query_indices = [], []
    labels_gallery, labels_query = [], []

    for cls, idxs in id_to_indices.items():
        # cap per identity
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
    # For ImageFolder, items are (image, class). Keep tensors; labels come from subset indices.
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(targets, dtype=torch.long)


# --------------------------- Metric utilities ------------------------------

def rank_metrics(labels_query: np.ndarray,
                 labels_gallery: np.ndarray,
                 indices: np.ndarray,
                 k: int) -> Dict[str, float]:
    """
    Compute metrics given FAISS top-k indices for each query.
    - labels_query: (nq,)
    - labels_gallery: (ng,)
    - indices: (nq, k) indices into gallery
    """
    nq, K = indices.shape
    assert K >= k
    top1_hits = 0
    recall_any_hits = 0
    precisions, ap_list, rr_list = [], [], []

    for qi in range(nq):
        qlab = labels_query[qi]
        retrieved = indices[qi, :k]
        retrieved_labels = labels_gallery[retrieved]

        # Binary relevance
        rel = (retrieved_labels == qlab).astype(np.int32)

        # Top1
        if rel[0] == 1:
            top1_hits += 1

        # Recall@K as "any-hit"
        if rel.any():
            recall_any_hits += 1

        # Precision@K
        precisions.append(rel.sum() / k)

        # AP@K (average precision over the first K results)
        # classic AP: sum_{i} P@i * rel_i / (#relevant in entire gallery)
        # Here, we approximate with AP@K using #relevant_in_topK for normalization
        # Alternatively, exact AP uses total relevant in gallery for that class.
        # To be consistent with earlier numbers, we normalize by relevant_in_topK>0.
        if rel.sum() > 0:
            cum_rel = np.cumsum(rel)
            precisions_at_i = cum_rel / (np.arange(k) + 1)
            ap = (precisions_at_i * rel).sum() / rel.sum()
        else:
            ap = 0.0
        ap_list.append(ap)

        # MRR (first relevant)
        ranks = np.where(rel == 1)[0]
        rr_list.append(1.0 / (ranks[0] + 1) if ranks.size > 0 else 0.0)

    return {
        "top1_acc": top1_hits / nq,
        "recall_at_k": recall_any_hits / nq,
        "precision_at_k": float(np.mean(precisions)),
        "average_precision_at_k": float(np.mean(ap_list)),
        "mean_average_precision_at_k": float(np.mean(ap_list)),  # identical here
        "mrr": float(np.mean(rr_list)),
    }


# --------------------------- FAISS helpers ---------------------------------

def normalize_if_cosine(x: np.ndarray, metric: str) -> np.ndarray:
    if metric.lower() == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        x = x / norms
    return x


def make_index_flat(d: int, metric: str):
    if metric.lower() == "cosine":
        return faiss.IndexFlatIP(d), "IndexFlatIP (cosine via IP)"
    elif metric.lower() == "euclidean":
        return faiss.IndexFlatL2(d), "IndexFlatL2"
    else:
        raise ValueError("metric must be one of {'cosine','euclidean'}")


def make_index_hnsw(d: int, metric: str, M: int, efC: int, efS: int):
    # Use index_factory for robust metric support across faiss versions
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

    # efConstruction and efSearch
    # (IndexHNSWFlat created via factory returns .hnsw)
    hnsw = faiss.downcast_index(index).hnsw
    hnsw.efConstruction = int(efC)
    hnsw.efSearch = int(efS)
    return index, desc


def index_mem_bytes(index) -> int:
    try:
        # faiss >= 1.7: serialize_index returns Python bytes
        b = faiss.serialize_index(index)
        return len(b)
    except Exception:
        return -1


# --------------------------- Embedding & eval -------------------------------

def embed_loader(dataloader: DataLoader, embed_fn, device: torch.device) -> np.ndarray:
    embs: List[np.ndarray] = []
    for xb, _ in tqdm(dataloader, desc="Embedding", leave=False):
        xb = xb.to(device, non_blocking=False)
        feats = embed_fn(xb)  # (B, D)
        feats = feats.detach().cpu().numpy().astype('float32')
        embs.append(feats)
    return np.concatenate(embs, axis=0)


def evaluate_index(index, gallery_embs: np.ndarray, gallery_labels: np.ndarray,
                   query_embs: np.ndarray, query_labels: np.ndarray, k: int,
                   metric: str) -> Tuple[Dict, float]:
    t0 = time.perf_counter()
    # Add to index (for Flat we add all at once; for HNSW it's fine too)
    index.add(gallery_embs)
    build_time = time.perf_counter() - t0

    # Search
    t1 = time.perf_counter()
    D, I = index.search(query_embs, k)
    search_time = time.perf_counter() - t1
    avg_search_ms = (search_time / len(query_embs)) * 1000.0

    # Metrics
    metrics = rank_metrics(query_labels, gallery_labels, I, k)
    metrics["avg_search_ms"] = avg_search_ms
    metrics["build_time_s"] = build_time
    mem = index_mem_bytes(index)
    if mem >= 0:
        metrics["index_mem_mb"] = mem / (1024 * 1024)

    # Clean up index memory when caller discards
    return metrics, avg_search_ms


# --------------------------- Main pipeline ---------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare Brute Force vs HNSW on CASIA-WebFace")
    parser.add_argument("--dataset-root", type=str, required=True,
                        help="Path to CASIA-WebFace root (ImageFolder layout).")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)  # keep 0 on macOS to be safe
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit-identities", type=int, default=None,
                        help="Optional: cap number of identities (for faster dev runs).")
    parser.add_argument("--max-images-per-id", type=int, default=10,
                        help="Cap images per identity before splitting into gallery/query.")
    parser.add_argument("--seed", type=int, default=1234)

    # HNSW params
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-efc", type=int, default=100)
    parser.add_argument("--hnsw-efs", type=int, default=64)

    # Embedding options
    parser.add_argument("--model", default="casia-webface",choices=["casia-webface", "vggface2"],
                        help="Which FaceNet weights to use (facenet-pytorch).")

    # parser.add_argument("--model", type=str, default="auto", choices=["auto", "facenet", "resnet50"])
    parser.add_argument("--cache", type=str, default=None,
                        help="Optional .npz to cache embeddings (saves/loads).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Single-thread for stability
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
        # remap: keep only the first N identities in class_to_idx order
        keep_classes = sorted(ds.class_to_idx.values())[:args.limit_identities]
        keep_idx = [i for i, (_, cls) in enumerate(ds.imgs) if cls in keep_classes]
        ds = Subset(ds, keep_idx)

        # Need a tiny shim so downstream still sees .imgs and .class_to_idx fields
        # Build shim attributes:
        class SubsetWithAttrs(Subset):
            pass
        ds = SubsetWithAttrs(ds.dataset, ds.indices)  # re-wrap
        # Recompute imgs & class_to_idx on the fly
        imgs = [ds.dataset.imgs[i] for i in ds.indices]
        class_names = sorted(list({Path(p).parent.name for p, _ in imgs}))
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        # Remap labels to 0..C-1
        remapped = []
        for p, _ in imgs:
            name = Path(p).parent.name
            remapped.append((p, class_to_idx[name]))
        ds.imgs = remapped
        ds.classes = class_names
        ds.class_to_idx = class_to_idx

    # Split into gallery/query (1 query per identity)
    gallery_ds, query_ds, labels_gallery, labels_query = split_gallery_query_by_id(
        ds, max_images_per_id=args.max_images_per_id, seed=args.seed
    )

    if len(query_ds) == 0 or len(gallery_ds) == 0:
        raise RuntimeError("After splitting, got empty gallery or query. "
                           "Try increasing --max-images-per-id or removing --limit-identities.")

    # Dataloaders (macOS-safe defaults)
    g_loader = DataLoader(gallery_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False, persistent_workers=False,
                          collate_fn=collate_keep_path)
    q_loader = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=False, persistent_workers=False,
                          collate_fn=collate_keep_path)

    # Build/embed
    # embed_fn, feat_dim, model_name = build_embedder(device, requested=("facenet" if args.model == "facenet" else "auto"))
    embedder, feat_dim, model_name = load_embedder(args.model, args.device)
    def embed_fn(xb):
        with torch.no_grad():
            return embedder(xb)
        
    cache_path = Path(args.cache) if args.cache else None
    if cache_path and cache_path.exists():
        data = np.load(cache_path)
        gallery_embs = data["gallery_embs"]
        query_embs = data["query_embs"]
        labels_gallery = data["labels_gallery"]
        labels_query = data["labels_query"]
        feat_dim = gallery_embs.shape[1]
        print(f"[cache] Loaded embeddings from {cache_path}")
    else:
        print(f"[embed] Model: {model_name} | feat_dim={feat_dim} | device={device}")
        t0 = time.perf_counter()
        gallery_embs = embed_loader(g_loader, embed_fn, device)
        query_embs = embed_loader(q_loader, embed_fn, device)
        embed_time = time.perf_counter() - t0
        print(f"[embed] Done in {embed_time:.2f}s | G={len(gallery_embs)} Q={len(query_embs)}")

        if cache_path:
            np.savez_compressed(cache_path,
                                gallery_embs=gallery_embs,
                                query_embs=query_embs,
                                labels_gallery=labels_gallery,
                                labels_query=labels_query)
            print(f"[cache] Saved to {cache_path}")

    # Normalize if cosine
    gallery = normalize_if_cosine(gallery_embs.astype('float32'), args.metric)
    query = normalize_if_cosine(query_embs.astype('float32'), args.metric)

    # ---------------- Brute Force ----------------
    flat_index, flat_desc = make_index_flat(gallery.shape[1], args.metric)
    print(f"\n[BRUTE] Building {flat_desc}")
    flat_metrics, flat_avg_ms = evaluate_index(flat_index, gallery, labels_gallery, query, labels_query, args.k, args.metric)
    # Release index early
    del flat_index
    gc.collect()

    # ---------------- HNSW ----------------
    hnsw_index, hnsw_desc = make_index_hnsw(gallery.shape[1], args.metric, args.hnsw_m, args.hnsw_efc, args.hnsw_efs)
    print(f"\n[HNSW] Building {hnsw_desc}")
    hnsw_metrics, hnsw_avg_ms = evaluate_index(hnsw_index, gallery, labels_gallery, query, labels_query, args.k, args.metric)
    del hnsw_index
    gc.collect()

    # Summary
    def short(d):
        keys = ["top1_acc", "recall_at_k", "mrr", "precision_at_k",
                "average_precision_at_k", "mean_average_precision_at_k",
                "avg_search_ms", "build_time_s", "index_mem_mb"]
        return {k: round(float(d.get(k, -1)), 6) for k in keys}

    summary = {
        "dataset": "CASIA-WebFace",
        "metric": args.metric,
        "k": args.k,
        "gallery_size": int(gallery.shape[0]),
        "query_size": int(query.shape[0]),
        "model": model_name,
        "flat_desc": flat_desc,
        "hnsw_desc": hnsw_desc,
        "flat": short(flat_metrics),
        "hnsw": short(hnsw_metrics),
        "hnsw_params": {"M": args.hnsw_m, "efConstruction": args.hnsw_efc, "efSearch": args.hnsw_efs},
    }
# 1B scaling estimates (comparable to your earlier output)
    summary["scaling_estimates_1B"] = estimate_scaling_1B(
        flat_avg_ms=flat_avg_ms,
        hnsw_avg_ms=hnsw_avg_ms,
        n=int(gallery.shape[0]),
        dim=int(gallery.shape[1]),
        M=args.hnsw_m,
        efSearch=args.hnsw_efs,
    )
    print("\n==== RESULTS ====")
    print(json.dumps(summary, indent=2))

    # Also save a CSV-like TSV if you want to log
    out_tsv = Path("casia_bruteforce_vs_hnsw.tsv")
    with io.open(out_tsv, "w", encoding="utf-8") as f:
        f.write("index\tmetric\tk\tgallery\tquery\t"
                "top1_acc\trecall_at_k\tmrr\tprecision_at_k\tap_at_k\tmap_at_k\tavg_search_ms\tbuild_time_s\tindex_mem_mb\n")
        f.write(f"BRUTE\t{args.metric}\t{args.k}\t{gallery.shape[0]}\t{query.shape[0]}\t"
                f"{flat_metrics['top1_acc']}\t{flat_metrics['recall_at_k']}\t{flat_metrics['mrr']}\t"
                f"{flat_metrics['precision_at_k']}\t{flat_metrics['average_precision_at_k']}\t{flat_metrics['mean_average_precision_at_k']}\t"
                f"{flat_metrics.get('avg_search_ms', -1)}\t{flat_metrics.get('build_time_s', -1)}\t{flat_metrics.get('index_mem_mb', -1)}\n")
        f.write(f"HNSW\t{args.metric}\t{args.k}\t{gallery.shape[0]}\t{query.shape[0]}\t"
                f"{hnsw_metrics['top1_acc']}\t{hnsw_metrics['recall_at_k']}\t{hnsw_metrics['mrr']}\t"
                f"{hnsw_metrics['precision_at_k']}\t{hnsw_metrics['average_precision_at_k']}\t{hnsw_metrics['mean_average_precision_at_k']}\t"
                f"{hnsw_metrics.get('avg_search_ms', -1)}\t{hnsw_metrics.get('build_time_s', -1)}\t{hnsw_metrics.get('index_mem_mb', -1)}\n")
    print(f"[write] Wrote {out_tsv.resolve()}")


def estimate_scaling_1B(flat_avg_ms: float,
                        hnsw_avg_ms: float,
                        n: int,
                        dim: int,
                        M: int,
                        efSearch: int,
                        dtype_bytes: int = 4):
    """
    Returns a dict with per-query time estimates at N=1e9 and memory needs.
    flat_avg_ms/hnsw_avg_ms are the measured averages at current n.
    """
    N_target = 1_000_000_000

    # ---- Flat (linear) ----
    per_vec_ms = flat_avg_ms / max(n, 1)
    flat_search_est_s = (per_vec_ms * N_target) / 1000.0

    # ---- HNSW (heuristic): ~ efSearch * log N scaling relative to current n ----
    # Scale the measured time by the ratio of logs (natural log is fine).
    log_ratio = math.log(N_target) / max(math.log(max(n, 2)), 1.0)
    hnsw_search_est_s = (hnsw_avg_ms / 1000.0) * log_ratio

    # ---- Memory ----
    # Vectors (float32): N * dim * 4 bytes
    vectors_bytes = N_target * dim * dtype_bytes
    # HNSW graph links: ~ N * M * (4..8 bytes) depending on build
    hnsw_graph_low = N_target * M * 4
    hnsw_graph_high = N_target * M * 8

    def human(b):
        # simple humanizer matching your previous format
        units = [("TB", 1<<40), ("GB", 1<<30), ("MB", 1<<20)]
        for name, size in units:
            if b >= size:
                return f"{b/size:.1f}{name}"
        return f"{b}B"

    return {
        "flat_search_est_s_per_query": flat_search_est_s,
        "hnsw_search_est_s_per_query": hnsw_search_est_s,
        "assumptions": {
            "flat_linear_per_vector_ms": per_vec_ms,
            "hnsw_log_scaling_factor": log_ratio,
            "note": "HNSW estimate assumes ~ log N scaling; real-world depends on data/params."
        },
        "memory_bytes": {
            "vectors": vectors_bytes,
            "hnsw_graph_low": hnsw_graph_low,
            "hnsw_graph_high": hnsw_graph_high
        },
        "memory_human": {
            "vectors": human(vectors_bytes),
            "hnsw_graph_low": human(hnsw_graph_low),
            "hnsw_graph_high": human(hnsw_graph_high)
        }
    }


if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)