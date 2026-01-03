"""
compare_models_euclidean.py

Compare VGGFace2 vs CASIA-WebFace (both FaceNet/InceptionResnetV1 backends)
using **exact brute-force Euclidean** retrieval on the same dataset.

Outputs:
- Accuracy metrics: top1_acc, recall_at_k, mrr, precision_at_k, average_precision_at_k, mean_average_precision_at_k
- Baseline & noisy performance (Gaussian noise on queries by default)
- Side-by-side bar charts (baseline, and one per noise std) comparing the two models

Dataset structure:
  DATASET_ROOT/
    gallery/<label>/*.jpg
    queries/<label>/*.jpg

Run (from project root):
  PYTHONPATH=. python compare_models_euclidean.py \
    --dataset-root storage/mi_eval \
    --k 5 \
    --noise-stds 0,5,10,15,20 \
    --noise-where queries \
    --device auto \
    --out-dir storage/mi_eval/compare_models \
    --out-json storage/mi_eval/compare_models/results.json
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


# ---------- euclidean top-k (squared L2) ----------
def topk_euclidean_with_Gsq(G: np.ndarray, G_sq: np.ndarray, q: np.ndarray, k: int):
    qsq = float(np.dot(q, q))
    d = G_sq + qsq - 2.0 * (G @ q)
    k = min(k, d.shape[0])
    idx = np.argpartition(d, k-1)[:k]
    order = np.argsort(d[idx])
    return d[idx][order], idx[order]


def add_gaussian_noise_rgb(pil_img: Image.Image, std: float, rng: np.random.Generator) -> Image.Image:
    if std <= 0:
        return pil_img
    arr = np.array(pil_img).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=std, size=arr.shape).astype(np.float32)
    arr_noisy = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr_noisy)


# ---------- evaluation for one model ----------
def evaluate_model_euclidean(model_name: str,
                             pre: Preprocessing,
                             device: str,
                             gallery_items: List[Tuple[Path, str]],
                             query_items: List[Tuple[Path, str]],
                             k: int,
                             noise_std: float,
                             noise_where: str,
                             rng: np.random.Generator,
                             label_counts: Dict[str, int]) -> Dict[str, float]:
    emb = Embedding(pretrained=model_name, device=device)

    # Gallery
    G_vecs, labels = [], []
    for pth, lbl in gallery_items:
        try:
            img = Image.open(pth).convert("RGB")
        except Exception:
            continue
        if noise_std > 0 and noise_where in ("gallery", "both"):
            img = add_gaussian_noise_rgb(img, noise_std, rng)
        x = pre.process(img)
        z = emb.encode(x).astype(np.float32)
        G_vecs.append(z); labels.append(lbl)
    if not G_vecs:
        raise RuntimeError("No gallery embeddings computed.")
    G = np.stack(G_vecs, axis=0)
    G_sq = np.sum(G * G, axis=1)

    # Queries
    queries = []
    for pth, gt in query_items:
        try:
            img = Image.open(pth).convert("RGB")
        except Exception:
            continue
        if noise_std > 0 and noise_where in ("queries", "both"):
            img = add_gaussian_noise_rgb(img, noise_std, rng)
        x = pre.process(img)
        z = emb.encode(x).astype(np.float32)
        queries.append((str(pth), z, gt))

    # Evaluate
    top1, hit_at_k, ranks = 0, 0, []
    precisions, apks = [], []
    for (_qp, q_raw, gt) in queries:
        _, idx = topk_euclidean_with_Gsq(G, G_sq, q_raw, k)

        topk_labels = [labels[i] for i in idx]
        rels = [1 if (lab == gt) else 0 for lab in topk_labels]

        rank = 0
        for j, r in enumerate(rels, start=1):
            if r == 1:
                rank = j; break
        ranks.append(rank)
        if rank == 1: top1 += 1
        if 1 <= rank <= k: hit_at_k += 1

        R = label_counts.get(gt, 0)
        precisions.append(precision_at_k(rels, k))
        apks.append(average_precision_at_k(rels, k, R))

    nQ = max(1, len(queries))
    mrr = float(np.mean([1.0/r if r > 0 else 0.0 for r in ranks]))
    return {
        "top1_acc": float(top1) / nQ,
        "recall_at_k": float(hit_at_k) / nQ,
        "mrr": mrr,
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
        "mean_average_precision_at_k": float(np.mean(apks)) if apks else 0.0,
    }


def make_side_by_side_bar(results_for_models: Dict[str, Dict[str, float]], title: str, save_path: Path):
    metrics = ["top1_acc","recall_at_k","mrr","precision_at_k","average_precision_at_k","mean_average_precision_at_k"]
    names = list(results_for_models.keys())
    X = np.arange(len(metrics))
    W = 0.8 / max(1, len(names))

    plt.figure(figsize=(12,5))
    for i, n in enumerate(names):
        vals = [results_for_models[n][m] for m in metrics]
        plt.bar(X + (i - (len(names)-1)/2)*W, vals, width=W, label=n)
    plt.xticks(X, metrics, rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--noise-stds", type=str, default="0,5,10,15,20")
    ap.add_argument("--noise-where", type=str, default="queries", choices=["queries","gallery","both"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    gal_items = list_images_with_labels(dataset_root / "gallery")
    qry_items = list_images_with_labels(dataset_root / "queries")
    if not gal_items or not qry_items:
        print(f"[ERROR] Expected images under {dataset_root}/gallery and /queries", file=sys.stderr)
        sys.exit(2)

    label_counts: Dict[str, int] = {}
    for _, lbl in gal_items:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    pre = Preprocessing()
    rng = np.random.default_rng(42)
    noise_levels = [float(s) for s in args.noise_stds.split(",") if s.strip() != ""]

    models = ["vggface2", "casia-webface"]
    all_results = []  # list of {"std": x, "metrics": {model: metrics}}

    for std in noise_levels:
        metrics_by_model: Dict[str, Dict[str, float]] = {}
        for m in models:
            res = evaluate_model_euclidean(
                model_name=m,
                pre=pre, device=device,
                gallery_items=gal_items, query_items=qry_items,
                k=args.k,
                noise_std=std, noise_where=args.noise_where, rng=rng,
                label_counts=label_counts
            )
            metrics_by_model[m] = res

        all_results.append({"std": std, "metrics": metrics_by_model})

        # save chart for this std (baseline when std==0)
        if args.out_dir:
            title = f"Euclidean — {'Baseline' if std==0 else f'Noise std={std:g}'}"
            out_png = Path(args.out_dir) / f"euclidean_{'baseline' if std==0 else f'noise_std_{int(std) if std.is_integer() else std}'} .png"
            # sanitize path (remove space if any)
            out_png = Path(str(out_png).replace(' ', ''))
            make_side_by_side_bar(metrics_by_model, title, out_png)

    # save json
    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump({
                "config": {
                    "k": args.k,
                    "device": device,
                    "dataset_root": str(dataset_root),
                    "noise_where": args.noise_where,
                    "noise_stds": noise_levels,
                    "models": models
                },
                "results": all_results
            }, f, indent=2)

    # Also print a compact baseline table for convenience
    baseline = next((r for r in all_results if r["std"] == 0), None)
    if baseline:
        print("\n# Baseline (Euclidean) — VGGFace2 vs CASIA-WebFace\n")
        metrics = ["top1_acc","recall_at_k","mrr","precision_at_k","average_precision_at_k","mean_average_precision_at_k"]
        header = "| Metric | VGGFace2 | CASIA-WebFace |\n|---|---:|---:|"
        print(header)
        for m in metrics:
            vgg = baseline["metrics"]["vggface2"][m]
            cas = baseline["metrics"]["casia-webface"][m]
            print(f"| {m} | {vgg:.4f} | {cas:.4f} |")

if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)
