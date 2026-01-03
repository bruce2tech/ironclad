#!/usr/bin/env python3
# evaluate_retrieval.py  — prints results to terminal (stdout) only
from __future__ import annotations
import argparse, json, time
import numpy as np
import faiss

# ---------- IO ----------
def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    X = d["feats"].astype("float32")
    y = d["labels"].astype(str)
    p = d["paths"].astype(object) if "paths" in d.files else None
    return X, y, p

def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

# ---------- Search & metrics ----------
def bruteforce_cosine_search(gF: np.ndarray, qF: np.ndarray, k: int):
    d = gF.shape[1]
    index = faiss.IndexFlatIP(d)   # brute-force IP; cosine on unit-norm inputs
    t0 = time.perf_counter()
    index.add(gF)
    sims, I = index.search(qF, k)
    t1 = time.perf_counter()
    return sims, I, (t1 - t0) * 1000.0 / max(1, qF.shape[0])  # ms/query

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

def pack_metrics(pred_lbls, qY, gY, k):
    top1, recall_k, precision_k, mrr = compute_core_metrics(pred_lbls, qY, k)
    ap_k, map_k = compute_ap_map_at_k(pred_lbls, qY, gY, k)
    return dict(top1_acc=top1, recall_at_k=recall_k, precision_at_k=precision_k,
                mrr=mrr, ap_at_k=ap_k, map_at_k=map_k)

# ---------- Noise helpers ----------
def parse_sigmas(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]

def add_noise_embeddings(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = X + sigma * rng.standard_normal(size=X.shape).astype("float32")
    return l2_normalize(noisy)

def add_noise_image_np(img_np_uint8: np.ndarray, sigma_frac: float, rng: np.random.Generator) -> np.ndarray:
    std = sigma_frac * 255.0
    noisy = img_np_uint8.astype(np.float32) + rng.normal(0.0, std, img_np_uint8.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 255.0).astype(np.uint8)
    return noisy

def reembed_with_noise(paths: np.ndarray, backend: str, sigma_frac: float,
                       det_size: int = 320, prefer_coreml: bool = True, limit: int = 0,
                       seed: int = 0) -> np.ndarray:
    """Re-embed images after adding Gaussian noise in image space (slower)."""
    from PIL import Image
    import cv2  # noqa: F401  (used implicitly by embedders)
    rng = np.random.default_rng(seed)
    from ironclad.modules.extraction.embedders import ArcFaceEmbedder, VGGFace2Embedder

    if backend.lower() in {"arcface","insightface"}:
        emb = ArcFaceEmbedder(ctx_id=-1, det_size=(det_size, det_size), prefer_coreml=prefer_coreml)
    elif backend.lower() in {"vggface2","facenet"}:
        emb = VGGFace2Embedder()
    else:
        raise ValueError("reembed backend must be arcface|vggface2")

    feats, n = [], 0
    for p in paths:
        if limit and n >= limit: break
        rgb = np.array(Image.open(p).convert("RGB"))
        noisy = add_noise_image_np(rgb, sigma_frac=sigma_frac, rng=rng)
        img = Image.fromarray(noisy)
        vec = emb.embed(img)
        if vec is None: continue
        feats.append(vec); n += 1
    if not feats:
        return np.empty((0, 512), dtype="float32")
    feats = np.stack(feats).astype("float32")
    return l2_normalize(feats)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Print retrieval metrics (and optional noise sweep) to stdout only.")
    ap.add_argument("--gallery_npz", required=True)
    ap.add_argument("--queries_npz", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--filter_orphans", action="store_true",
                    help="Drop queries whose label is not present in gallery (recommended).")
    ap.add_argument("--report_search_time", action="store_true",
                    help="Also report average FAISS search time per query (ms).")

    # Noise sweep (fast = embedding; slow = image)
    ap.add_argument("--noise_sigmas", default="", help="Comma-separated sigmas. "
                    "Embedding mode: std in embedding space (e.g., 0,0.02,0.05). "
                    "Image mode: fraction of 255 (e.g., 0,0.02,0.05).")
    ap.add_argument("--noise_mode", choices=["embedding","image"], default="embedding")
    ap.add_argument("--noise_apply", choices=["queries","gallery","both"], default="queries")
    ap.add_argument("--noise_seed", type=int, default=0)

    # Image-mode re-embed options (only used if --noise_mode image)
    ap.add_argument("--reembed_backend", choices=["arcface","vggface2"], default=None)
    ap.add_argument("--prefer_coreml", action="store_true")
    ap.add_argument("--det_size", type=int, default=320)
    ap.add_argument("--reembed_limit", type=int, default=0)

    args = ap.parse_args()

    # Load & optionally filter
    gF, gY, gP = load_npz(args.gallery_npz)
    qF, qY, qP = load_npz(args.queries_npz)

    if args.filter_orphans:
        gal_set = set(gY.tolist())
        keep = np.array([lbl in gal_set for lbl in qY], dtype=bool)
        qF, qY = qF[keep], qY[keep]
        if qP is not None:
            qP = qP[keep]

    # Normalize → cosine
    gF = l2_normalize(gF)
    qF = l2_normalize(qF)

    # Baseline (no noise)
    sims, I, search_ms = bruteforce_cosine_search(gF, qF, args.k)
    base_metrics = pack_metrics(gY[I], qY, gY, args.k)

    report = {
        "dataset": "MI-Eval",
        "distance": "cosine",
        "index": "bruteforce_flatip",
        "k": int(args.k),
        "gallery_size": int(gF.shape[0]),
        "query_size": int(qF.shape[0]),
        "n_identities_gallery": int(len(set(gY.tolist()))),
        "model": args.model_name,
        "metrics_baseline": base_metrics
    }
    if args.report_search_time:
        report["timing"] = {"search_avg_ms": search_ms}

    # Noise sweep (prints as part of the same JSON)
    if args.noise_sigmas:
        sigmas = parse_sigmas(args.noise_sigmas)
        rng = np.random.default_rng(args.noise_seed)
        results = {}
        for sigma in sigmas:
            if sigma == 0.0:
                results[f"{sigma:.4f}"] = base_metrics
                continue

            if args.noise_mode == "embedding":
                gF_noisy = add_noise_embeddings(gF, sigma, rng) if args.noise_apply in ("gallery","both") else gF
                qF_noisy = add_noise_embeddings(qF, sigma, rng) if args.noise_apply in ("queries","both") else qF
                sims2, I2, _ = bruteforce_cosine_search(gF_noisy, qF_noisy, args.k)
                metrics = pack_metrics(gY[I2], qY, gY, args.k)

            else:
                if args.reembed_backend is None:
                    raise SystemExit("--reembed_backend is required for noise_mode=image.")
                gF2, qF2 = gF, qF
                if args.noise_apply in ("gallery","both"):
                    if gP is None:
                        raise SystemExit("Gallery NPZ needs 'paths' to re-embed with image noise.")
                    gF2 = reembed_with_noise(gP, args.reembed_backend, sigma,
                                             det_size=args.det_size, prefer_coreml=args.prefer_coreml,
                                             limit=args.reembed_limit, seed=args.noise_seed)
                    if gF2.shape[0] != gF.shape[0]:
                        raise SystemExit("Re-embedded gallery count != original; set --reembed_limit=0.")
                if args.noise_apply in ("queries","both"):
                    if qP is None:
                        raise SystemExit("Queries NPZ needs 'paths' to re-embed with image noise.")
                    qF2 = reembed_with_noise(qP, args.reembed_backend, sigma,
                                             det_size=args.det_size, prefer_coreml=args.prefer_coreml,
                                             limit=args.reembed_limit, seed=args.noise_seed)
                    if qF2.shape[0] != qF.shape[0]:
                        raise SystemExit("Re-embedded queries count != original; set --reembed_limit=0.")
                sims2, I2, _ = bruteforce_cosine_search(gF2, qF2, args.k)
                metrics = pack_metrics(gY[I2], qY, gY, args.k)

            results[f"{sigma:.4f}"] = metrics

        report["noise"] = {
            "mode": args.noise_mode,
            "apply_to": args.noise_apply,
            "sigmas": sigmas,
            "sigma_unit": "embedding_std" if args.noise_mode == "embedding" else "fraction_of_255_pixels",
            "results_by_sigma": results
        }

    # === PRINT TO TERMINAL ===
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)