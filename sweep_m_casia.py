"""
sweep_m_casia.py

Investigate how retrieval performance varies as the number of gallery images
per identity increases (m = 1, 2, 3, ...), using **FaceNet (CASIA-WebFace)**
embeddings and **cosine similarity (exact brute-force)**.

Dataset layout:
  DATASET_ROOT/
    gallery/<identity>/*.jpg     # multiple images per identity
    queries/<identity>/*.jpg     # usually 1 image per identity

For each m:
  - Choose m images per identity (random w/ seed or first m).
  - Build **centroid** per identity: mean embedding of the m images, then L2-normalize.
  - Evaluate queries (L2-normalized) against centroids with cosine (IP) and compute metrics.

Metrics (per m): top1_acc, recall_at_k, mrr, precision_at_k, average_precision_at_k, mean_average_precision_at_k

Options:
  --m-list "1,2,3,4" or --m-max N to sweep [1..N]
  --fixed-identities true/false
     true: use only identities that have >= max(m-list) images **and** appear in queries (fair comparison across m)
     false: at each m, include any identity with >= m images (identity set may change across m)
  --sampling {random,first} and --seed for random
  --device {auto,cuda,cpu}
  --out-json, --out-dir for plots

Run example:
  PYTHONPATH=. python sweep_m_casia.py \
    --dataset-root storage/mi_eval \
    --k 5 \
    --m-list 1,2,3,4,5 \
    --fixed-identities true \
    --sampling random --seed 42 \
    --device auto \
    --out-json storage/mi_eval/sweep_casia_m.json \
    --out-dir  storage/mi_eval/sweep_casia_plots
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


def add_identity_map(items: List[Tuple[Path, str]]) -> Dict[str, List[Path]]:
    by_id: Dict[str, List[Path]] = {}
    for p, lab in items:
        by_id.setdefault(lab, []).append(p)
    for lab in by_id:
        by_id[lab] = sorted(by_id[lab])
    return by_id


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


# ---------- core helpers ----------
def embed_image(pil_img, pre: Preprocessing, emb: Embedding) -> np.ndarray:
    x = pre.process(pil_img)
    z = emb.encode(x).astype(np.float32)
    return z


def build_centroid(emb_list: List[np.ndarray]) -> np.ndarray:
    c = np.mean(np.stack(emb_list, axis=0), axis=0)
    n = np.linalg.norm(c) + 1e-12
    return (c / n).astype(np.float32)


def cosine_topk(Gn: np.ndarray, qn: np.ndarray, k: int):
    sims = Gn @ qn  # (N,)(D) -> (N,)
    # top-k by similarity (higher better) -> turn into distances if needed: d=1-sim
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, k-1)[:k]
    order = np.argsort(-sims[idx])
    return sims[idx][order], idx[order]


def evaluate_at_m(m: int,
                  pre: Preprocessing,
                  emb: Embedding,
                  gallery_by_id: Dict[str, List[Path]],
                  queries_by_id: Dict[str, List[Path]],
                  k: int,
                  sampling: str,
                  rng: np.random.Generator,
                  fixed_identities: bool,
                  eligible_ids_fixed: List[str] = None) -> Dict[str, float]:
    # choose identities
    if fixed_identities and eligible_ids_fixed is not None:
        ids = eligible_ids_fixed
    else:
        ids = [lab for lab, paths in gallery_by_id.items() if len(paths) >= m and lab in queries_by_id]

    if not ids:
        raise RuntimeError(f"No identities have >= {m} images and a query.")

    # gallery centroids (normalized)
    G_centroids, labels = [], []
    for lab in ids:
        paths = gallery_by_id[lab]
        if sampling == "random":
            sel = rng.choice(len(paths), size=m, replace=False)
            chosen = [paths[i] for i in sel]
        else:
            chosen = paths[:m]

        embs = []
        for p in chosen:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            embs.append(embed_image(img, pre, emb))
        if not embs:
            continue
        c = build_centroid(embs)
        G_centroids.append(c); labels.append(lab)

    if not G_centroids:
        raise RuntimeError("No gallery centroids were built.")
    G = np.stack(G_centroids, axis=0)  # already normalized

    # queries (normalized)
    queries = []
    for lab in ids:
        qpaths = queries_by_id.get(lab, [])
        if not qpaths:
            continue
        p = qpaths[0]  # assume one per identity
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        z = embed_image(img, pre, emb)
        zn = z / (np.linalg.norm(z) + 1e-12)
        queries.append((str(p), zn, lab))

    if not queries:
        raise RuntimeError("No queries available after filtering identities.")

    # evaluate
    top1, hit_at_k, ranks = 0, 0, []
    precisions, apks = [], []
    for (_qp, qn, gt) in queries:
        _, idx = cosine_topk(G, qn, k)
        topk_labels = [labels[i] for i in idx]
        rels = [1 if (lab == gt) else 0 for lab in topk_labels]

        # rank of first relevant
        rank = 0
        for j, r in enumerate(rels, start=1):
            if r == 1:
                rank = j; break
        ranks.append(rank)
        if rank == 1: top1 += 1
        if 1 <= rank <= k: hit_at_k += 1

        # for AP@K, R is 1 when using centroids (one relevant per identity)
        R = 1
        precisions.append(precision_at_k(rels, k))
        apks.append(average_precision_at_k(rels, k, R))

    nQ = max(1, len(queries))
    mrr = float(np.mean([1.0/r if r > 0 else 0.0 for r in ranks]))
    return {
        "num_identities": len(ids),
        "top1_acc": float(top1) / nQ,
        "recall_at_k": float(hit_at_k) / nQ,
        "mrr": mrr,
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
        "mean_average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
    }


def plot_metric_vs_m(m_values: List[int], ys: List[float], metric_name: str, out_dir: Path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4.5))
    plt.plot(m_values, ys, marker="o")
    plt.xlabel("m (images per identity in gallery)")
    plt.ylabel(metric_name)
    plt.title(f"CASIA-WebFace — {metric_name} vs m (cosine, centroids)")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"{metric_name}_vs_m.png", dpi=150)
    plt.close()


def markdown_table(results_by_m: Dict[int, Dict[str, float]]) -> str:
    keys = ["num_identities","top1_acc","recall_at_k","mrr","precision_at_k","average_precision_at_k","mean_average_precision_at_k"]
    header = "| m | " + " | ".join(keys) + " |\n"
    header += "|" + "|".join(["---"]*(len(keys)+1)) + "|\n"
    lines = []
    for m in sorted(results_by_m.keys()):
        row = [str(m)] + [f"{results_by_m[m][k]:.4f}" if k!="num_identities" else str(results_by_m[m][k]) for k in keys]
        lines.append("| " + " | ".join(row) + " |")
    return header + "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="Path containing gallery/ and queries/")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--m-list", type=str, default=None, help="Comma list of m values, e.g., 1,2,3,4")
    ap.add_argument("--m-max", type=int, default=None, help="If set, use m=1..m_max")
    ap.add_argument("--fixed-identities", type=str, default="true", choices=["true","false"])
    ap.add_argument("--sampling", type=str, default="random", choices=["random","first"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    if (args.m_list is None) == (args.m_max is None):
        print("[ERROR] Provide exactly one of --m-list or --m-max", file=sys.stderr)
        sys.exit(2)

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    # m sweep values
    if args.m_list is not None:
        m_values = [int(s) for s in args.m_list.split(",") if s.strip()!=""]
    else:
        m_values = list(range(1, int(args.m_max)+1))

    dataset_root = Path(args.dataset_root)
    gal_items = list_images_with_labels(dataset_root / "gallery")
    qry_items = list_images_with_labels(dataset_root / "queries")
    if not gal_items or not qry_items:
        print(f"[ERROR] Expected images under {dataset_root}/gallery and /queries", file=sys.stderr)
        sys.exit(2)

    gallery_by_id = add_identity_map(gal_items)
    queries_by_id = add_identity_map(qry_items)

    # determine eligible identities if fixed-identities
    fixed = (args.fixed_identities.lower() == "true")
    eligible_fixed = None
    if fixed:
        # identities with at least max(m_values) gallery images and a query
        mmax = max(m_values)
        eligible_fixed = sorted([lab for lab, paths in gallery_by_id.items()
                                 if len(paths) >= mmax and lab in queries_by_id])

        if not eligible_fixed:
            print(f"[ERROR] No identities have >= {mmax} gallery images and a query.", file=sys.stderr)
            sys.exit(2)

    pre = Preprocessing()
    emb = Embedding(pretrained="casia-webface", device=device)
    rng = np.random.default_rng(args.seed)

    results_by_m: Dict[int, Dict[str, float]] = {}
    for m in m_values:
        res = evaluate_at_m(
            m=m, pre=pre, emb=emb,
            gallery_by_id=gallery_by_id,
            queries_by_id=queries_by_id,
            k=args.k,
            sampling=args.sampling, rng=rng,
            fixed_identities=fixed, eligible_ids_fixed=eligible_fixed
        )
        results_by_m[m] = res

    # Print Markdown table
    print("\n# CASIA-WebFace — Cosine (centroids) — Metrics vs m\n")
    print(markdown_table(results_by_m))

    # Save JSON
    if args.out_json:
        outp = Path(args.out_json); outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump({
                "config": {
                    "device": device,
                    "k": args.k,
                    "m_values": m_values,
                    "fixed_identities": fixed,
                    "sampling": args.sampling,
                    "seed": args.seed,
                },
                "sizes": {
                    "gallery_identities_total": len(gallery_by_id),
                    "query_identities_total": len(queries_by_id),
                },
                "results_by_m": results_by_m
            }, f, indent=2)

    # Plots
    if args.out_dir:
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for metric in ["top1_acc","recall_at_k","mrr","average_precision_at_k","mean_average_precision_at_k","precision_at_k"]:
            ys = [results_by_m[m][metric] for m in m_values]
            plot_metric_vs_m(m_values, ys, metric, out_dir)


if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)