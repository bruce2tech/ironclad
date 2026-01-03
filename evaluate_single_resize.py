#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time, os, sys
import numpy as np
import faiss
from PIL import Image

# ---------------- IO ----------------
def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    X = d["feats"].astype("float32")
    y = d["labels"].astype(str)
    p = d["paths"].astype(object) if "paths" in d.files else None
    return X, y, p

def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

# ------------- Search & metrics -------------
def bruteforce_cosine_search(gF: np.ndarray, qF: np.ndarray, k: int):
    d = gF.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine if inputs are L2-normalized
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

def compute_ap_at_k(topk_lbls: np.ndarray, true_lbls: np.ndarray, gallery_lbls: np.ndarray, k: int):
    ap_per_query = []
    ids, counts = np.unique(gallery_lbls, return_counts=True)
    rel_count = {i: int(c) for i, c in zip(ids.tolist(), counts.tolist())}
    for i in range(true_lbls.shape[0]):
        y = true_lbls[i]
        denom = min(k, rel_count.get(y, 0))
        if denom == 0:
            ap_per_query.append(0.0); continue
        hits = (topk_lbls[i] == y)
        seen = prec_sum = 0.0
        for r in range(k):
            if hits[r]:
                seen += 1
                prec_sum += seen / float(r + 1)
        ap_per_query.append(prec_sum / float(denom))
    return float(np.mean(ap_per_query)) if ap_per_query else 0.0

def pack_metrics(pred_lbls, qY, gY, k):
    top1, r, p, mrr = compute_core_metrics(pred_lbls, qY, k)
    ap = compute_ap_at_k(pred_lbls, qY, gY, k)
    return dict(top1_acc=top1, recall_at_k=r, precision_at_k=p, mrr=mrr, ap_at_k=ap)

# ------------- Resize helpers -------------
def _interp_from_name(name: str):
    from PIL import Image
    name = name.lower()
    if name == "nearest": return Image.NEAREST
    if name == "bilinear": return Image.BILINEAR
    if name == "bicubic": return Image.BICUBIC
    if name == "lanczos": return Image.LANCZOS
    raise ValueError("Unknown interpolation: " + name)

def reembed_once(paths: np.ndarray, backend: str, scale: float,
                 mode: str = "down_up", interpolation: str = "bilinear",
                 det_size: int = 320, prefer_coreml: bool = True,
                 limit: int = 0, assume_aligned: bool = False, verbose: bool = True):
    """
    Re-embed images after resizing.
      mode='scale'   : resize to (round(w*scale), round(h*scale)) and feed as-is.
      mode='down_up' : downscale by 'scale', then resize back to original (simulates blur/quality loss).
    If 'paths' point to pre-aligned face chips, set assume_aligned=True to skip detection (VGG fast path).
    """
    from ironclad.modules.extraction.embedders import ArcFaceEmbedder, VGGFace2Embedder

    if backend.lower() in {"arcface","insightface"}:
        if verbose: print(f"[ArcFace] prefer_coreml={prefer_coreml} det_size={det_size}", flush=True)
        emb = ArcFaceEmbedder(ctx_id=-1, det_size=(det_size, det_size), prefer_coreml=prefer_coreml)
    elif backend.lower() in {"vggface2","facenet"}:
        emb = VGGFace2Embedder()
        if verbose:
            try:
                import torch
                print(f"[VGG] device={getattr(emb,'device','?')} mps={torch.backends.mps.is_available()} assume_aligned={assume_aligned}", flush=True)
            except Exception:
                print(f"[VGG] assume_aligned={assume_aligned}", flush=True)
    else:
        raise ValueError("backend must be arcface|vggface2")

    interp = _interp_from_name(interpolation)
    n = len(paths) if not limit else min(limit, len(paths))
    feats = []
    t0 = time.perf_counter()
    for i in range(n):
        p = paths[i]
        if verbose and (i % 25 == 0):
            print(f"[reembed] {i}/{n} scale={scale:.2f}", flush=True)
        img = Image.open(p).convert("RGB")
        w, h = img.size
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        if mode == "scale":
            resized = img.resize((new_w, new_h), resample=interp)
        elif mode == "down_up":
            resized = img.resize((new_w, new_h), resample=interp).resize((w, h), resample=interp)
        else:
            raise ValueError("resize mode must be 'scale' or 'down_up'")

        # Fast path only for VGG when chips are passed
        if backend.lower() in {"vggface2","facenet"} and assume_aligned:
            vec = emb.embed(resized, assume_aligned=True)
        else:
            vec = emb.embed(resized)

        if vec is None:
            vec = np.zeros(512, dtype="float32")  # keep shapes consistent
        feats.append(vec)

    t1 = time.perf_counter()
    feats = np.stack(feats).astype("float32")
    return l2_normalize(feats), (t1 - t0) / max(1, n)  # avg seconds per image

def map_to_chips(paths, orig_root, chips_root):
    mapped = []
    for p in paths:
        rel = os.path.relpath(p, orig_root)
        cp = os.path.join(chips_root, rel)
        mapped.append(cp if os.path.exists(cp) else p)  # fallback if missing
    return np.array(mapped, dtype=object)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Single-scale resize evaluation (prints metrics).")
    ap.add_argument("--gallery_npz", required=True)
    ap.add_argument("--queries_npz", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--filter_orphans", action="store_true")

    # One scale only
    ap.add_argument("--scale", type=float, default=0.75, help="Resize factor (e.g., 0.75)")
    ap.add_argument("--resize_mode", choices=["scale","down_up"], default="down_up")
    ap.add_argument("--resize_apply", choices=["queries","gallery","both"], default="queries")
    ap.add_argument("--interpolation", choices=["nearest","bilinear","bicubic","lanczos"], default="bilinear")

    ap.add_argument("--reembed_backend", choices=["arcface","vggface2"], required=True)
    ap.add_argument("--assume_aligned", action="store_true", help="Use when paths are chips (VGG fast path).")
    ap.add_argument("--chips_queries_root", default=None)
    ap.add_argument("--chips_gallery_root", default=None)

    ap.add_argument("--det_size", type=int, default=320)
    ap.add_argument("--prefer_coreml", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0=all; N=first N images (for quick tests)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Load
    gF, gY, gP = load_npz(args.gallery_npz)
    qF, qY, qP = load_npz(args.queries_npz)

    # Optional filter (drop queries whose label not in gallery)
    if args.filter_orphans:
        gal_set = set(gY.tolist())
        keep = np.array([lbl in gal_set for lbl in qY], dtype=bool)
        qF, qY = qF[keep], qY[keep]
        if qP is not None: qP = qP[keep]

    # Normalize for cosine
    gF = l2_normalize(gF); qF = l2_normalize(qF)

    # Baseline metrics (original embeddings)
    _, I0, search_ms0 = bruteforce_cosine_search(gF, qF, args.k)
    base_metrics = pack_metrics(gY[I0], qY, gY, args.k)

    # Map to chips (optional)
    q_src, g_src = qP, gP
    if args.chips_queries_root:
        q_src = map_to_chips(qP, orig_root="storage/mi_eval/queries", chips_root=args.chips_queries_root)
    if args.chips_gallery_root:
        g_src = map_to_chips(gP, orig_root="storage/mi_eval/gallery", chips_root=args.chips_gallery_root)

    # Re-embed with a single scale
    gF2, qF2 = gF, qF
    gY_arr, qY_arr = gY, qY
    avg_embed_sec = None

    if args.resize_apply in ("gallery","both"):
        if g_src is None: raise SystemExit("Gallery NPZ missing 'paths' and no chips path provided.")
        gF2, avg_embed_sec = reembed_once(
            g_src, args.reembed_backend, args.scale, args.resize_mode, args.interpolation,
            args.det_size, args.prefer_coreml, args.limit, args.assume_aligned, args.verbose
        )
        if args.limit and gF2.shape[0] != gF.shape[0]:
            gY_arr = gY[:gF2.shape[0]]

    if args.resize_apply in ("queries","both"):
        if q_src is None: raise SystemExit("Queries NPZ missing 'paths' and no chips path provided.")
        qF2, avg_embed_sec = reembed_once(
            q_src, args.reembed_backend, args.scale, args.resize_mode, args.interpolation,
            args.det_size, args.prefer_coreml, args.limit, args.assume_aligned, args.verbose
        )
        if args.limit and qF2.shape[0] != qF.shape[0]:
            qY_arr = qY[:qF2.shape[0]]

    # Search with resized split(s)
    _, I2, search_ms2 = bruteforce_cosine_search(gF2, qF2, args.k)
    resized_metrics = pack_metrics(gY_arr[I2], qY_arr, gY_arr, args.k)

    # Print: markdown table
    print("\n| Setting        | Top-1 Acc | Recall@k | Precision@k | MRR    | AP@k   |")
    print("| ---            | ---:      | ---:     | ---:        | ---:   | ---:   |")
    print(f"| Baseline (σ=0) | {base_metrics['top1_acc']:.4f} | {base_metrics['recall_at_k']:.4f} | "
          f"{base_metrics['precision_at_k']:.4f} | {base_metrics['mrr']:.4f} | {base_metrics['ap_at_k']:.4f} |")
    print(f"| Resize s={args.scale:.2f} | {resized_metrics['top1_acc']:.4f} | {resized_metrics['recall_at_k']:.4f} | "
          f"{resized_metrics['precision_at_k']:.4f} | {resized_metrics['mrr']:.4f} | {resized_metrics['ap_at_k']:.4f} |")

    # Print: compact JSON
    out = {
        "dataset": "MI-Eval",
        "distance": "cosine",
        "index": "bruteforce_flatip",
        "k": args.k,
        "gallery_size": int(gF.shape[0]),
        "query_size": int(qF.shape[0]),
        "model": args.model_name,
        "apply": args.resize_apply,
        "resize": dict(scale=args.scale, mode=args.resize_mode, interpolation=args.interpolation),
        "assume_aligned": bool(args.assume_aligned),
        "limit": int(args.limit),
        "timing": {
            "search_avg_ms_baseline": search_ms0,
            "search_avg_ms_resized": search_ms2,
            **({"avg_embed_sec": avg_embed_sec} if avg_embed_sec is not None else {})
        },
        "metrics": {
            "baseline": base_metrics,
            "resized": resized_metrics
        }
    }
    print("\n" + json.dumps(out, indent=2))

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
    