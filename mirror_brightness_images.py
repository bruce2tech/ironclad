#!/usr/bin/env python3
import argparse, os
from PIL import Image, ImageEnhance
from pathlib import Path

def find_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    root = Path(root)
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])

def factor_tag(f):
    # 0.5 -> b050, 1.25 -> b125
    return f"b{int(round(float(f)*100)):03d}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="Directory containing images (tree).")
    ap.add_argument("--dst_root", required=True, help="Parent directory to write the mirrored tree.")
    ap.add_argument("--factor", type=float, required=True, help="Brightness factor, e.g. 0.50, 0.75, 1.25, 1.50")
    ap.add_argument("--dry_run", action="store_true", help="List a few planned writes, then exit.")
    args = ap.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_parent = Path(args.dst_root).resolve()

    if not src_root.is_dir():
        raise SystemExit(f"[error] --src_root not found: {src_root}")

    files = find_images(src_root)
    print(f"brightness {args.factor}: found {len(files)} images under {src_root}")

    # create sibling like {dst_root}/{basename(src_root)}_{bXXX}
    out_root = dst_parent / f"{src_root.name}_{factor_tag(args.factor)}"
    print(f"[info] output will go to: {out_root}")

    if args.dry_run:
        for p in files[:5]:
            rel = p.relative_to(src_root)
            q = out_root / rel
            print(f"DRY-RUN  {p}  ->  {q}")
        print("[dry_run] done.")
        return

    # tqdm is optional
    try:
        from tqdm import tqdm
        it = tqdm(files, desc=f"brightness {args.factor}")
    except Exception:
        it = files

    n_ok = 0
    for p in it:
        rel = p.relative_to(src_root)
        q = out_root / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        try:
            im = Image.open(p).convert("RGB")
            im = ImageEnhance.Brightness(im).enhance(args.factor)
            im.save(q)
            n_ok += 1
        except Exception as e:
            print(f"[warn] failed {p}: {e}")

    print(f"[done] wrote {n_ok} files to {out_root}")

if __name__ == "__main__":
    main()