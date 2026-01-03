#!/usr/bin/env python3
import argparse, sys, shutil
from pathlib import Path
from PIL import Image, ImageOps

# Try tqdm; fall back to a tiny built-in progress bar if not installed
try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

INTERP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def resize_one(src: Path, dst: Path, scale: float, interp):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        # respect EXIF orientation
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        if (nw, nh) != (w, h):
            im = im.resize((nw, nh), interp)

        ext = src.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            # JPEG must be L or RGB
            if im.mode not in ("L", "RGB"):
                im = im.convert("RGB")
            im.save(dst, quality=95, subsampling=1, optimize=True)
        elif ext == ".png" and im.mode == "P":
            im.save(dst, optimize=True)
        else:
            im.save(dst)

def _fallback_progress(total):
    # Minimal progress bar that works without tqdm
    class _Bar:
        def __init__(self, total):
            self.total = total
            self.count = 0
            self.postfix = ""

        def update(self, n=1):
            self.count += n
            width = shutil.get_terminal_size((80, 20)).columns
            width = max(20, min(width, 120))
            frac = 0 if self.total == 0 else self.count / self.total
            filled = int((width - 20) * frac)
            bar = "#" * filled + "-" * ((width - 20) - filled)
            pct = f"{frac*100:6.2f}%"
            line = f"[{bar}] {pct} {self.postfix}"
            print("\r" + line[:width], end="", flush=True)

        def set_postfix(self, **kwargs):
            parts = [f"{k}={v}" for k, v in kwargs.items()]
            self.postfix = " ".join(parts)

        def close(self):
            self.update(0)
            print()
    return _Bar(total)

def main():
    ap = argparse.ArgumentParser(description="Mirror + resize an image tree by a scale factor.")
    ap.add_argument("--src", required=True, help="Source root directory")
    ap.add_argument("--dst", required=True, help="Destination root directory (created if missing)")
    ap.add_argument("--scale", type=float, required=True, help="Scale factor (e.g. 0.75)")
    ap.add_argument("--interpolation", choices=list(INTERP.keys()), default="lanczos")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                    help="Skip files that already exist at dst")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N images (for testing)")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.exists():
        print(f"[error] src not found: {src}", file=sys.stderr); sys.exit(1)
    dst.mkdir(parents=True, exist_ok=True)

    interp = INTERP[args.interpolation]
    all_imgs = [p for p in src.rglob("*") if p.is_file() and is_image(p)]
    total = len(all_imgs)
    print(f"[info] found {total} images under {src}")

    wrote = 0
    skipped = 0
    errors = 0

    # Choose progress bar
    if tqdm is not None:
        pbar = tqdm(total=total, unit="img", dynamic_ncols=True)
        def tick(): pbar.update(1)  # noqa: E306
        def set_postfix(**kw): pbar.set_postfix(**kw)  # noqa: E306
        def close(): pbar.close()  # noqa: E306
    else:
        pbar = _fallback_progress(total)
        def tick(): pbar.update(1)  # noqa: E306
        def set_postfix(**kw): pbar.set_postfix(**kw)  # noqa: E306
        def close(): pbar.close()  # noqa: E306

    for p in all_imgs:
        rel = p.relative_to(src)
        outp = dst / rel
        if args.skip_existing and outp.exists():
            skipped += 1
            set_postfix(wrote=wrote, skipped=skipped, errors=errors)
            tick()
            continue

        try:
            resize_one(p, outp, args.scale, interp)
            wrote += 1
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"\n[warn] failed {p}: {e}", file=sys.stderr)

        set_postfix(wrote=wrote, skipped=skipped, errors=errors)
        tick()

        if args.limit and wrote >= args.limit:
            break

    close()
    print(f"[done] wrote={wrote} skipped={skipped} errors={errors} -> {dst}")

if __name__ == "__main__":
    main()
