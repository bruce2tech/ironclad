#!/usr/bin/env python3
import os, argparse, numpy as np
from PIL import Image

def find_images(root, exts):
    found = []
    for r, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                found.append(os.path.join(r, f))
    return sorted(found)

def make_label(path, mode):
    if mode == "dirname":
        return os.path.basename(os.path.dirname(path))
    elif mode == "stem":
        return os.path.splitext(os.path.basename(path))[0]
    else:
        raise ValueError("--label_mode must be dirname or stem")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root of images.")
    ap.add_argument("--backend", choices=["arcface","vggface2"], required=True)
    ap.add_argument("--out", required=True, help="Output .npz path (parent dir will be created).")
    ap.add_argument("--label_mode", choices=["dirname","stem"], default="stem",
                    help="How to derive labels. Your layout wants 'stem'.")
    ap.add_argument("--det_size", type=int, default=320)
    ap.add_argument("--prefer_coreml", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Debug: only embed first N images.")
    args = ap.parse_args()

    # Tame OpenMP/BLAS on macOS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    # Import embedders lazily
    from ironclad.modules.extraction.embedders import ArcFaceEmbedder, VGGFace2Embedder
    if args.backend == "arcface":
        emb = ArcFaceEmbedder(ctx_id=-1, det_size=(args.det_size, args.det_size),
                              prefer_coreml=args.prefer_coreml)
    else:
        emb = VGGFace2Embedder()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not os.path.isdir(args.root):
        raise SystemExit(f"[error] --root not a directory: {args.root}")

    paths = find_images(args.root, exts)
    if args.limit:
        paths = paths[:args.limit]

    print(f"[info] root={args.root}")
    print(f"[info] found {len(paths)} images")
    if paths[:2]:
        for p in paths[:2]:
            print(f"[info] sample: {p}")

    if not paths:
        raise SystemExit("No images found under --root (check path and extensions).")

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **k: x

    feats, labels = [], []
    for p in tqdm(paths, desc=f"Embedding ({args.backend})"):
        try:

            img = Image.open(p).convert("RGB")
            if args.backend.lower() == "vggface2":
                # ensure classic VGGFace2/ResNet input; avoids MPS adaptive-pool error
                img = img.resize((224, 224), Image.BILINEAR)
            vec = emb.embed(img)

            # img = Image.open(p).convert("RGB")
            # vec = emb.embed(img)
            if vec is None:
                continue
            feats.append(np.asarray(vec, dtype="float32"))
            labels.append(make_label(p, args.label_mode))
        except Exception as e:
            print(f"[warn] failed on {p}: {e}")

    if not feats:
        raise SystemExit("No features produced (all embeds failed?).")

    X = np.vstack(feats).astype("float32")
    y = np.array(labels, dtype=object)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(args.out, feats=X, labels=y)
    print(f"[done] saved {args.out}  feats={X.shape}  n_labels={len(y)}")

if __name__ == "__main__":
    main()