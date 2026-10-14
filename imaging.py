"""Local image prep with Pillow: resizing to model-friendly sizes, masks, contact sheets."""
import io
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps


def fit_dims(w, h, megapixels, multiple=16):
    """Scale (w, h) to ~megapixels keeping aspect ratio, rounded to `multiple`."""
    scale = (megapixels * 1_000_000 / (w * h)) ** 0.5
    return (max(multiple, round(w * scale / multiple) * multiple),
            max(multiple, round(h * scale / multiple) * multiple))


def prepare(path, work_dir, megapixels=1.0):
    """Resize an image to ~megapixels (multiple-of-16 dims). Returns (new_path, w, h)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = fit_dims(*img.size, megapixels)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{Path(path).stem}_{w}x{h}.png"
    img.resize((w, h), Image.LANCZOS).save(out)
    return str(out), w, h


def size_of(path):
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).size


def region_mask(w, h, region, work_dir, name="mask", ellipse=False, feather=0):
    """White = area to change. region = [x0, y0, x1, y1] as fractions (0-1) or pixels."""
    if all(0 <= v <= 1 for v in region):
        region = [region[0] * w, region[1] * h, region[2] * w, region[3] * h]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    (draw.ellipse if ellipse else draw.rectangle)([round(v) for v in region], fill=255)
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    out = Path(work_dir) / f"{name}_{w}x{h}.png"
    mask.save(out)
    return str(out)


def resize_mask(path, w, h, work_dir):
    """Resize a user-supplied mask (white = change) to w x h."""
    m = Image.open(path).convert("L").resize((w, h), Image.LANCZOS)
    out = Path(work_dir) / f"{Path(path).stem}_mask_{w}x{h}.png"
    m.save(out)
    return str(out)


def contact_sheet(paths, out_path, labels=None, thumb=384):
    """Grid of thumbnails with labels, so several drafts can be compared in one view."""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    cols = min(len(imgs), 4)
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, img in enumerate(imgs):
        img.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 24)
        sheet.paste(img, (x + (thumb - img.width) // 2, y))
        draw.text((x + 6, y + thumb + 4), labels[i] if labels else str(i + 1), fill="black")
    sheet.save(out_path)
    return str(out_path)


def prepare_exact(path, w, h, work_dir):
    """Resize (center-crop to aspect) an image to exactly w x h, e.g. a control image matching a reference."""
    img = ImageOps.fit(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), (w, h), Image.LANCZOS)
    out = Path(work_dir) / f"{Path(path).stem}_fit_{w}x{h}.png"
    img.save(out)
    return str(out)


def preview_jpeg(path, max_side=768, quality=80):
    """Small JPEG bytes of an image (transparent areas on a checkerboard) for showing to Claude."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        board = Image.new("RGBA", img.size, (255, 255, 255, 255))
        draw = ImageDraw.Draw(board)
        for y in range(0, img.height, 32):
            for x in range(0, img.width, 32):
                if (x // 32 + y // 32) % 2:
                    draw.rectangle([x, y, x + 31, y + 31], fill=(204, 204, 204, 255))
        board.alpha_composite(img.convert("RGBA"))
        img = board
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def clean_old(work_dir, max_age_hours=24):
    """Delete temporary resized inputs/masks older than max_age_hours."""
    cutoff = time.time() - max_age_hours * 3600
    for p in Path(work_dir).glob("*") if Path(work_dir).exists() else []:
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass
