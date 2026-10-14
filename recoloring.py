"""Masked colour changes in Oklab: retain perceived lightness and source geometry.

Colour transforms follow Bjorn Ottosson's public-domain/MIT implementation:
https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
"""
import math

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageOps


def _linear(value):
    value /= 255
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def to_oklab(rgb):
    r, g, b = map(_linear, rgb)
    l = (0.4122214708*r + 0.5363325363*g + 0.0514459929*b) ** (1/3)
    m = (0.2119034982*r + 0.6806995451*g + 0.1073969566*b) ** (1/3)
    s = (0.0883024619*r + 0.2817188376*g + 0.6299787005*b) ** (1/3)
    return (0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            0.0259040371*l + 0.7827717662*m - 0.8086757660*s)


def _linear_rgb(lightness, a, b):
    l = (lightness + 0.3963377774*a + 0.2158037573*b) ** 3
    m = (lightness - 0.1055613458*a - 0.0638541728*b) ** 3
    s = (lightness - 0.0894841775*a - 1.2914855480*b) ** 3
    return (4.0767416621*l - 3.3077115913*m + 0.2309699292*s,
            -1.2684380046*l + 2.6097574011*m - 0.3413193965*s,
            -0.0041960863*l - 0.7034186147*m + 1.7076147010*s)


def _in_gamut(rgb):
    return all(-1e-7 <= value <= 1 + 1e-7 for value in rgb)


def _encode(value):
    value = max(0.0, min(1.0, value))
    value = 12.92*value if value <= 0.0031308 else 1.055*value**(1/2.4) - 0.055
    return max(0, min(255, round(value * 255)))


def from_oklab(lightness, a, b):
    """Compress chroma into sRGB rather than clipping channels and losing shading."""
    rgb = _linear_rgb(lightness, a, b)
    if not _in_gamut(rgb):
        low, high = 0.0, 1.0
        for _ in range(14):
            mid = (low + high) / 2
            if _in_gamut(_linear_rgb(lightness, a*mid, b*mid)):
                low = mid
            else:
                high = mid
        rgb = _linear_rgb(lightness, a*low, b*low)
    return tuple(map(_encode, rgb))


def load_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        mode = 'RGBA' if 'A' in image.getbands() or 'transparency' in image.info else 'RGB'
        return image.convert(mode)


def selection(size, mask_path='', region=None):
    """Intersect an optional region with a white=change mask at the original size."""
    if not mask_path and region is None:
        raise ValueError('Provide mask_path or region to select the area to recolour')
    mask = Image.new('L', size, 255)
    if mask_path:
        with Image.open(mask_path) as image:
            image = ImageOps.exif_transpose(image)
            mask = image.convert('L')
            if 'A' in image.getbands() or 'transparency' in image.info:
                mask = ImageChops.multiply(mask, image.convert('RGBA').getchannel('A'))
            mask = mask.resize(size, Image.Resampling.LANCZOS)
    if region is not None:
        if len(region) != 4 or not all(math.isfinite(v) for v in region):
            raise ValueError('region must contain four finite coordinates')
        w, h = size
        coords = list(region)
        if all(0 <= v <= 1 for v in coords):
            coords = [coords[0]*w, coords[1]*h, coords[2]*w, coords[3]*h]
        x0, y0, x1, y1 = coords
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('region must be ordered and lie inside the displayed source image')
        box = (math.floor(x0), math.floor(y0), math.ceil(x1)-1, math.ceil(y1)-1)
        region_mask = Image.new('L', size, 0)
        ImageDraw.Draw(region_mask).rectangle(box, fill=255)
        mask = ImageChops.multiply(mask, region_mask)
    return mask


def recolor(image, mask, color, strength=1.0, saturation=1.0, feather=0):
    """Keep Oklab L and spatial detail; change only selected pixels' hue/chroma.

    The target supplies hue/chroma, not exposure. Neutral targets desaturate.
    Feathering occurs inside the supplied selection, never outside its support.
    """
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must be between 0 and 1')
    if not math.isfinite(saturation) or not 0 <= saturation <= 2:
        raise ValueError('saturation must be between 0 and 2')
    if not math.isfinite(feather) or not 0 <= feather <= 64:
        raise ValueError('feather must be between 0 and 64 pixels')
    try:
        target = ImageColor.getrgb(color)
    except (ValueError, TypeError) as error:
        raise ValueError('color must be a named colour or an opaque hex colour, e.g. #3267C8') from error
    if len(target) == 4 and target[3] == 255:
        target = target[:3]
    if len(target) != 3:
        raise ValueError('color must be opaque; source transparency is preserved separately')
    if mask.size != image.size:
        raise ValueError('mask must match source dimensions')
    mask = mask.convert('L')
    if feather:
        mask = ImageChops.multiply(mask, mask.filter(ImageFilter.GaussianBlur(feather)))
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError('The selection is empty')
    result = image.copy()
    source, output, selected = image.load(), result.load(), mask.load()
    weight, total_chroma, count = 0, 0.0, 0
    for y in range(bounds[1], bounds[3]):
        for x in range(bounds[0], bounds[2]):
            alpha = selected[x, y]
            if alpha:
                _, a, b = to_oklab(source[x, y][:3])
                total_chroma += math.hypot(a, b) * alpha
                weight += alpha
                count += 1
    reference_chroma = total_chroma / weight
    _, ta, tb = to_oklab(target)
    changed, max_lightness_error = 0, 0.0
    for y in range(bounds[1], bounds[3]):
        for x in range(bounds[0], bounds[2]):
            alpha = selected[x, y] / 255 * strength
            if not alpha:
                continue
            pixel = source[x, y]
            lightness, a, b = to_oklab(pixel[:3])
            # Preserve chroma variation in coloured fabric; support grey fabric too.
            scale = math.hypot(a, b)/reference_chroma if reference_chroma >= .01 else 1.0
            new = from_oklab(lightness, a + (ta*scale*saturation-a)*alpha,
                            b + (tb*scale*saturation-b)*alpha)
            if new != pixel[:3]:
                output[x, y] = new + pixel[3:]  # Preserve alpha exactly.
                changed += 1
                max_lightness_error = max(max_lightness_error, abs(to_oklab(new)[0]-lightness))
    return result, mask, {'selected_pixels': count, 'changed_pixels': changed,
                          'max_lightness_error': round(max_lightness_error, 6)}
