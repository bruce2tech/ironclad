# evaluate_retrieval.py
from __future__ import annotations
import argparse, json, time, os
import numpy as np
import faiss

def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    X = d["feats"].astype("float32")
    y = d["labels"].astype(str)
    return X, y

def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def bruteforce_cosine_search(gF: np.ndarray, qF: np.ndarray, k: int):
    d = gF.shape[1]
    index = faiss.IndexFlatIP(d)  # brute-force IP; cosine when inputs are unit-norm
    index.add(gF)
    t0 = time.perf_counter()
    sims, I = index.search(qF, k)
    t1 = time.perf_counter()
    search_avg_ms = (t1 - t0) * 1000.0 / max(1, qF.shape[0])
    return sims, I, search_avg_ms

def compute_core_metrics(topk_lbls: np.ndarray, true_lbls: np.ndarray, k: int):
    nq = true_lbls.shape[0]
    top1 = float((topk_lbls[:, 0] == true_lbls).mean())
    recall_k = float((topk_lbls == true_lbls[:, None]).any(axis=1).mean())
    precision_k = float(((topk_lbls == true_lbls[:, None]).sum(axis=1) / k).mean())
    ranks = []
    for i in range(nq):
        m = np.where(topk_lbls[i] == true_lbls[i])[0]
        ranks.append(1.0 / (m[0] + 1) if len(m) else 0.0)
    mrr = float(np.mean(ranks))
    return top1, recall_k, precision_k, mrr

def compute_ap_map_at_k(topk_lbls: np.ndarray, true_lbls: np.ndarray, gallery_lbls: np.ndarray, k: int):
    ap_per_query, per_id_aps = [], {}
    ids, counts = np.unique(gallery_lbls, return_counts=True)
    rel_count = {i: int(c) for i, c in zip(ids.tolist(), counts.tolist())}
    for i in range(true_lbls.shape[0]):
        y = true_lbls[i]
        denom = min(k, rel_count.get(y, 0))
        if denom == 0:
            ap_per_query.append(0.0); continue
        hits = (topk_lbls[i] == y)
        seen, prec_sum = 0, 0.0
        for r in range(k):
            if hits[r]:
                seen += 1
                prec_sum += seen / float(r + 1)
        ap_i = prec_sum / float(denom)
        ap_per_query.append(ap_i)
        per_id_aps.setdefault(y, []).append(ap_i)
    APk = float(np.mean(ap_per_query)) if ap_per_query else 0.0
    mAPk = float(np.mean([np.mean(v) for v in per_id_aps.values()])) if per_id_aps else 0.0
    return APk, mAPk

# ---------- Optional: end-to-end embed timing ----------
def time_query_embedding(backend: str, queries_root: str, limit: int = 50, det_size: int = 320, prefer_coreml: bool = True):
    """Return avg per-query times (ms): query_pre_ms, query_embed_ms, n_timed."""
    from PIL import Image
    from ironclad.modules.extraction.embedders import ArcFaceEmbedder, VGGFace2Embedder

    if backend.lower() in {"arcface","insightface"}:
        emb = ArcFaceEmbedder(ctx_id=-1, det_size=(det_size, det_size), prefer_coreml=prefer_coreml)
    elif backend.lower() in {"vggface2","facenet"}:
        emb = VGGFace2Embedder()
    else:
        raise ValueError("backend must be arcface|vggface2")

    # collect image paths
    img_exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    paths = []
    for root,_,files in os.walk(queries_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in img_exts:
                paths.append(os.path.join(root,f))
    paths.sort()
    if limit: paths = paths[:limit]

    pre_sum = embed_sum = 0.0
    n_ok = 0
    for i, p in enumerate(paths):
        if i % 25 == 0:
            print(f"[timing] {i}/{len(paths)}...", flush=True)
        t0 = time.perf_counter(); img = Image.open(p).convert("RGB"); t1 = time.perf_counter()
        vec = emb.embed(img); t2 = time.perf_counter()
        if vec is None: continue
        pre_sum += (t1 - t0); embed_sum += (t2 - t1); n_ok += 1
    if n_ok == 0:
        return 0.0, 0.0, 0
    return pre_sum*1000.0/n_ok, embed_sum*1000.0/n_ok, n_ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery_npz", required=True)
    ap.add_argument("--queries_npz", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--out_json", default="")

    # timing options
    ap.add_argument("--report_search_time", action="store_true",
                    help="Include average FAISS search time per query (ms).")
    ap.add_argument("--time_embed_backend", choices=["arcface","vggface2"], default=None,
                    help="Also measure average per-query embedding time for this backend.")
    ap.add_argument("--queries_root", default=None, help="Root folder of raw query images (for embed timing).")
    ap.add_argument("--limit_embed", type=int, default=50, help="Cap number of queries for embed timing.")
    ap.add_argument("--det_size", type=int, default=320, help="ArcFace detector size (e.g. 256/320/640).")
    ap.add_argument("--prefer_coreml", action="store_true", help="Ask InsightFace to use CoreML if available.")
    args = ap.parse_args()

    # Load & filter
    gF, gY = load_npz(args.gallery_npz)
    qF, qY = load_npz(args.queries_npz)
    gal_set = set(gY.tolist())
    keep = np.array([lbl in gal_set for lbl in qY], dtype=bool)
    qF, qY = qF[keep], qY[keep]

    # Normalize (cosine)
    gF = l2_normalize(gF); qF = l2_normalize(qF)

    # Search (+ optional timing)
    sims, I, search_avg_ms = bruteforce_cosine_search(gF, qF, args.k)
    pred_lbls = gY[I]

    # Metrics
    top1, recall_k, precision_k, mrr = compute_core_metrics(pred_lbls, qY, args.k)
    ap_k, map_k = compute_ap_map_at_k(pred_lbls, qY, gY, args.k)

    report = {
        "dataset": "MI-Eval",
        "distance": "cosine",
        "index": "bruteforce_flatip",
        "k": int(args.k),
        "gallery_size": int(gF.shape[0]),
        "query_size": int(qF.shape[0]),
        "n_identities_gallery": int(len(set(gY.tolist()))),
        "model": args.model_name,
        "metrics": {
            "top1_acc": top1,
            "recall_at_k": recall_k,
            "precision_at_k": precision_k,
            "mrr": mrr,
            "ap_at_k": ap_k,
            "map_at_k": map_k
        }
    }

    if args.report_search_time:
        report["timing"] = {"search_avg_ms": search_avg_ms}

    if args.time_embed_backend:
        if not args.queries_root:
            raise SystemExit("--queries_root is required when --time_embed_backend is set.")
        qpre_ms, qemb_ms, n_timed = time_query_embedding(
            backend=args.time_embed_backend,
            queries_root=args.queries_root,
            limit=args.limit_embed,
            det_size=args.det_size,
            prefer_coreml=args.prefer_coreml
        )
        report.setdefault("timing", {}).update({
            "query_pre_ms": qpre_ms,
            "query_embed_ms": qemb_ms,
            "n_timed": n_timed
        })

    print(json.dumps(report, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)

if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)