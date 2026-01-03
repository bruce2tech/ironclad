# extract_features.py
from __future__ import annotations
import argparse, os, glob, re
from typing import Iterable, List, Tuple, Optional
import numpy as np
from PIL import Image

from .embedders import VGGFace2Embedder, ArcFaceEmbedder, BaseEmbedder

IMG_EXTS = (".jpg",".jpeg",".png",".bmp",".webp")

def walk_images(root: str) -> List[str]:
    paths = []
    for p, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(p, f))
    paths.sort()
    return paths

def derive_label(path: str, split_root: str, label_regex: Optional[str]) -> str:
    if label_regex:
        m = re.search(label_regex, os.path.basename(path))
        if m:
            return m.group(1)
    # Prefer subfolder name under split_root
    parent = os.path.basename(os.path.dirname(path))
    if os.path.abspath(os.path.dirname(path)) != os.path.abspath(split_root):
        return parent
    # Fallback: filename prefix before first underscore
    return os.path.basename(path).split("_")[0]

def build_embedder(name: str) -> BaseEmbedder:
    if name.lower() in {"vggface2", "facenet"}:
        return VGGFace2Embedder()
    elif name.lower() in {"arcface", "insightface"}:
        return ArcFaceEmbedder(ctx_id=-1)
    raise ValueError(f"Unknown backend {name}")

def extract_one_split(split_root: str, backend: str, label_regex: Optional[str], limit: int=0) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    embedder = build_embedder(backend)
    paths = walk_images(split_root)
    feats, labels, keep_paths = [], [], []
    n = 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        vec = embedder.embed(img)
        if vec is None:
            continue
        feats.append(vec)
        labels.append(derive_label(p, split_root, label_regex))
        keep_paths.append(p)
        n += 1
        if limit and n >= limit:
            break
    return (np.asarray(feats, dtype="float32"),
            np.asarray(labels, dtype=object),
            np.asarray(keep_paths, dtype=object))

def save_npz(out_path: str, feats: np.ndarray, labels: np.ndarray, paths: np.ndarray):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, feats=feats, labels=labels, paths=paths)
    print(f"Wrote {out_path}: feats={feats.shape}, ids={len(set(labels.tolist()))}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery_root", required=True)
    ap.add_argument("--queries_root", required=True)
    ap.add_argument("--backend", choices=["vggface2","arcface"], required=True)
    ap.add_argument("--out_dir", required=True, help="Where .npz files go")
    ap.add_argument("--label_regex", default=None, help="Optional regex with one capture group for label")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gF, gY, gP = extract_one_split(args.gallery_root, args.backend, args.label_regex, args.limit)
    qF, qY, qP = extract_one_split(args.queries_root, args.backend, args.label_regex, args.limit)

    # safety: ensure L2-normalized
    gF = gF / (np.linalg.norm(gF, axis=1, keepdims=True) + 1e-12)
    qF = qF / (np.linalg.norm(qF, axis=1, keepdims=True) + 1e-12)

    save_npz(os.path.join(args.out_dir, f"{args.backend}_gallery.npz"), gF, gY, gP)
    save_npz(os.path.join(args.out_dir, f"{args.backend}_queries.npz"), qF, qY, qP)

if __name__ == "__main__":
    main()
