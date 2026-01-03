#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, time, json
import numpy as np
import faiss
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

# Import embedders
from ironclad.modules.extraction.embedders import ArcFaceEmbedder, VGGFace2Embedder

def load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    X = d["feats"].astype("float32")
    y = d["labels"].astype(object)
    p = d["paths"].astype(object)
    # safety: normalize for cosine==IP
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X, y, p

def build_flatip(dims):
    return faiss.IndexFlatIP(dims)

def topk_search(gF, gY, qF, k):
    index = build_flatip(gF.shape[1])
    index.add(gF)
    t0 = time.perf_counter()
    sims, I = index.search(qF, k)
    t1 = time.perf_counter()
    pred_lbls = gY[I]
    search_avg_ms = (t1 - t0) * 1000.0 / qF.shape[0]
    return pred_lbls, search_avg_ms

def compute_metrics(topk_lbls: np.ndarray, true_lbls: np.ndarray, gY: np.ndarray, k: int):
    nq = true_lbls.shape[0]
    # Top1, Recall@K, Precision@K, MRR
    top1 = float((topk_lbls[:,0] == true_lbls).mean())
    hit = float((topk_lbls == true_lbls[:,None]).any(axis=1).mean())
    prec = float(((topk_lbls == true_lbls[:,None]).sum(axis=1) / k).mean())
    # MRR
    ranks = []
    for i in range(nq):
        m = np.where(topk_lbls[i] == true_lbls[i])[0]
        ranks.append(1.0/(m[0]+1) if len(m) else 0.0)
    mrr = float(np.mean(ranks))

    # AP@k and mAP@k (macro over identities)
    ap_per_query = []
    id_to_aps = defaultdict(list)
    for i in range(nq):
        y_true = true_lbls[i]
        # total relevant in gallery for this id (cap at k for AP@k denominator)
        rel_total = min(k, int(np.sum(gY == y_true)))
        if rel_total == 0:
            ap_per_query.append(0.0)
            continue
        hits = (topk_lbls[i] == y_true)
        num_hits = 0
        prec_sum = 0.0
        for r in range(k):
            if hits[r]:
                num_hits += 1
                prec_sum += num_hits / float(r+1)
        ap_i = prec_sum / float(rel_total)
        ap_per_query.append(ap_i)
        id_to_aps[y_true].append(ap_i)

    APk = float(np.mean(ap_per_query)) if ap_per_query else 0.0
    # macro over identities
    id_means = [np.mean(v) for v in id_to_aps.values()] or [0.0]
    mAPk = float(np.mean(id_means))
    return dict(
        top1_acc=top1,
        recall_at_k=hit,
        precision_at_k=prec,
        mrr=mrr,
        ap_at_k=APk,
        map_at_k=mAPk,
    )

# def time_query_embedding(paths, embedder, limit=0):
#     """Return (query_pre_ms, query_embed_ms, n_ok) averaged over successful embeddings."""
#     pre_sum = 0.0
#     embed_sum = 0.0
#     n_ok = 0
#     for i, p in enumerate(paths):
#         if limit and n_ok >= limit:
#             break
#         t0 = time.perf_counter()
#         img = Image.open(p).convert("RGB")
#         t1 = time.perf_counter()
#         vec = embedder.embed(img)
#         t2 = time.perf_counter()
#         if vec is None:
#             continue
#         pre_sum += (t1 - t0)
#         embed_sum += (t2 - t1)
#         n_ok += 1
#     if n_ok == 0:
#         return 0.0, 0.0, 0
#     return pre_sum * 1000.0 / n_ok, embed_sum * 1000.0 / n_ok, n_ok

def time_query_embedding(paths, embedder, limit=0, ping_every=50):
    import time
    from PIL import Image
    pre_sum = embed_sum = 0.0
    n_ok = n_seen = 0
    total = len(paths) if not limit else min(limit, len(paths))
    for i, p in enumerate(paths):
        if limit and n_seen >= limit: break
        n_seen += 1
        if i % ping_every == 0:
            print(f"[timing] {n_seen}/{total}...", flush=True)
        t0 = time.perf_counter(); img = Image.open(p).convert("RGB"); t1 = time.perf_counter()
        vec = embedder.embed(img); t2 = time.perf_counter()
        if vec is None: continue
        pre_sum += (t1 - t0); embed_sum += (t2 - t1); n_ok += 1
    if n_ok == 0: return 0.0, 0.0, 0
    return pre_sum*1000.0/n_ok, embed_sum*1000.0/n_ok, n_ok

def format_markdown_table(headers, rows):
    # headers: list[str]; rows: list[list]
    col_widths = [max(len(h), *(len(f"{r[c]}") for r in rows)) for c, h in enumerate(headers)]
    def fmt_row(vals):
        return "| " + " | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(vals)) + " |"
    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"
    table = [fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    return "\n".join(table)

def save_table_md(out_path, md):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md + "\n")

def plot_grouped_bars(title, categories, series_names, series_values, out_path, ylabel=None):
    """
    categories: list[str]
    series_names: list[str] e.g. ["ArcFace", "VGGFace2"]
    series_values: list[list[float]] shape (len(series_names), len(categories))
    """
    import numpy as np
    x = np.arange(len(categories))
    width = 0.38
    fig, ax = plt.subplots()
    for i, (name, vals) in enumerate(zip(series_names, series_values)):
        ax.bar(x + (i - 0.5)*width, vals, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--arc_gallery_npz", required=True)
    ap.add_argument("--arc_queries_npz", required=True)
    ap.add_argument("--vgg_gallery_npz", required=True)
    ap.add_argument("--vgg_queries_npz", required=True)
    ap.add_argument("--save_dir", default="storage/mi_eval_feats/compare")
    ap.add_argument("--arcface_ctx_id", type=int, default=-1, help="-1=CPU, 0=GPU")
    ap.add_argument("--limit_embed", type=int, default=0, help="Optional cap for timing runs")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load ArcFace data
    arc_gF, arc_gY, arc_gP = load_npz(args.arc_gallery_npz)
    arc_qF, arc_qY, arc_qP = load_npz(args.arc_queries_npz)
    # keep only queries whose label exists in gallery
    arc_keep = np.array([y in set(arc_gY.tolist()) for y in arc_qY], dtype=bool)
    arc_qF, arc_qY, arc_qP = arc_qF[arc_keep], arc_qY[arc_keep], arc_qP[arc_keep]

    # Load VGGFace2 data
    vgg_gF, vgg_gY, vgg_gP = load_npz(args.vgg_gallery_npz)
    vgg_qF, vgg_qY, vgg_qP = load_npz(args.vgg_queries_npz)
    vgg_keep = np.array([y in set(vgg_gY.tolist()) for y in vgg_qY], dtype=bool)
    vgg_qF, vgg_qY, vgg_qP = vgg_qF[vgg_keep], vgg_qY[vgg_keep], vgg_qP[vgg_keep]

    k = args.k

    # ----- Retrieval & metrics (bruteforce cosine via IP) -----
    arc_pred, arc_search_ms = topk_search(arc_gF, arc_gY, arc_qF, k)
    vgg_pred, vgg_search_ms = topk_search(vgg_gF, vgg_gY, vgg_qF, k)

    arc_perf = compute_metrics(arc_pred, arc_qY, arc_gY, k)
    vgg_perf = compute_metrics(vgg_pred, vgg_qY, vgg_gY, k)

    # ----- Timing: preprocessing & embedding on queries -----
    arc_embedder = ArcFaceEmbedder(ctx_id=args.arcface_ctx_id)
    vgg_embedder = VGGFace2Embedder()

    arc_pre_ms, arc_embed_ms, arc_n = time_query_embedding(arc_qP, arc_embedder, limit=args.limit_embed)
    vgg_pre_ms, vgg_embed_ms, vgg_n = time_query_embedding(vgg_qP, vgg_embedder, limit=args.limit_embed)

    timing = {
        "arcface": dict(query_pre_ms=arc_pre_ms, query_embed_ms=arc_embed_ms, search_avg_ms=arc_search_ms, n_timed=arc_n),
        "vggface2": dict(query_pre_ms=vgg_pre_ms, query_embed_ms=vgg_embed_ms, search_avg_ms=vgg_search_ms, n_timed=vgg_n),
    }

    # ----- Markdown tables -----
    perf_headers = ["Model","Top-1 Acc","Recall@k","MRR","Precision@k","AP@k","mAP@k"]
    perf_rows = [
        ["ArcFace", f"{arc_perf['top1_acc']:.4f}", f"{arc_perf['recall_at_k']:.4f}",
         f"{arc_perf['mrr']:.4f}", f"{arc_perf['precision_at_k']:.4f}",
         f"{arc_perf['ap_at_k']:.4f}", f"{arc_perf['map_at_k']:.4f}"],
        ["VGGFace2", f"{vgg_perf['top1_acc']:.4f}", f"{vgg_perf['recall_at_k']:.4f}",
         f"{vgg_perf['mrr']:.4f}", f"{vgg_perf['precision_at_k']:.4f}",
         f"{vgg_perf['ap_at_k']:.4f}", f"{vgg_perf['map_at_k']:.4f}"],
    ]
    perf_md = format_markdown_table(perf_headers, perf_rows)
    save_table_md(os.path.join(args.save_dir, "performance_table.md"), perf_md)

    time_headers = ["Model","query_pre_ms","query_embed_ms","search_avg_ms","n_timed"]
    time_rows = [
        ["ArcFace", f"{timing['arcface']['query_pre_ms']:.2f}", f"{timing['arcface']['query_embed_ms']:.2f}",
         f"{timing['arcface']['search_avg_ms']:.2f}", timing["arcface"]["n_timed"]],
        ["VGGFace2", f"{timing['vggface2']['query_pre_ms']:.2f}", f"{timing['vggface2']['query_embed_ms']:.2f}",
         f"{timing['vggface2']['search_avg_ms']:.2f}", timing["vggface2"]["n_timed"]],
    ]
    timing_md = format_markdown_table(time_headers, time_rows)
    save_table_md(os.path.join(args.save_dir, "timing_table.md"), timing_md)

    # Print to stdout as well (so you can copy/paste)
    print("\nPerformance metrics (k={}):\n".format(k))
    print(perf_md)
    print("\nTiming metrics (averages):\n")
    print(timing_md)

    # ----- Bar charts -----
    # Performance chart
    perf_categories = ["Top-1 Acc","Recall@k","MRR","Precision@k","AP@k","mAP@k"]
    arc_vals = [arc_perf["top1_acc"], arc_perf["recall_at_k"], arc_perf["mrr"],
                arc_perf["precision_at_k"], arc_perf["ap_at_k"], arc_perf["map_at_k"]]
    vgg_vals = [vgg_perf["top1_acc"], vgg_perf["recall_at_k"], vgg_perf["mrr"],
                vgg_perf["precision_at_k"], vgg_perf["ap_at_k"], vgg_perf["map_at_k"]]
    plot_grouped_bars(
        title=f"Performance (k={k})",
        categories=perf_categories,
        series_names=["ArcFace","VGGFace2"],
        series_values=[arc_vals, vgg_vals],
        out_path=os.path.join(args.save_dir, "performance_bars.png"),
        ylabel="score (0–1)"
    )

    # Timing chart
    time_categories = ["query_pre_ms","query_embed_ms","search_avg_ms"]
    arc_tvals = [timing["arcface"]["query_pre_ms"], timing["arcface"]["query_embed_ms"], timing["arcface"]["search_avg_ms"]]
    vgg_tvals = [timing["vggface2"]["query_pre_ms"], timing["vggface2"]["query_embed_ms"], timing["vggface2"]["search_avg_ms"]]
    plot_grouped_bars(
        title=f"Timing (averages)",
        categories=time_categories,
        series_names=["ArcFace","VGGFace2"],
        series_values=[arc_tvals, vgg_tvals],
        out_path=os.path.join(args.save_dir, "timing_bars.png"),
        ylabel="milliseconds"
    )

    # Save a JSON blob too
    out_json = os.path.join(args.save_dir, "compare_summary.json")
    with open(out_json, "w") as f:
        json.dump({
            "k": k,
            "arcface": {"performance": arc_perf, "timing": timing["arcface"]},
            "vggface2": {"performance": vgg_perf, "timing": timing["vggface2"]},
        }, f, indent=2)
    print(f"\nSaved tables & charts to: {args.save_dir}")

if __name__ == "__main__":
    start_script_timer = time.perf_counter()
    main()
    end_script_timer = time.perf_counter()
    script_runtime = end_script_timer - start_script_timer
    print(script_runtime)