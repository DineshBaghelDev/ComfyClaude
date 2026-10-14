import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops

import recoloring


class RecoloringTest(unittest.TestCase):
    def fabric(self, mode='RGB'):
        image = Image.new(mode, (12, 8), (220, 181, 144) if mode == 'RGB' else (220, 181, 144, 77))
        shades = [(39, 22, 12), (82, 45, 25), (145, 78, 35), (196, 112, 56), (231, 155, 93)]
        for y in range(2, 6):
            for x in range(3, 9):
                rgb = (0, 0, 0) if x == 5 else shades[(x+y) % len(shades)]
                image.putpixel((x, y), rgb if mode == 'RGB' else rgb + ((x*29+y*13) % 256,))
        return image

    def test_changes_hue_preserves_lightness_and_protected_pixels(self):
        original = self.fabric()
        mask = recoloring.selection(original.size, region=[3, 2, 9, 6])
        result, _, stats = recoloring.recolor(original, mask, '#3267C8')
        for y in range(original.height):
            for x in range(original.width):
                old, new = original.getpixel((x, y)), result.getpixel((x, y))
                if not mask.getpixel((x, y)) or old == (0, 0, 0):
                    self.assertEqual(new, old)
                else:
                    self.assertGreater(new[2], new[0], (old, new))
                    self.assertLess(abs(recoloring.to_oklab(old)[0]-recoloring.to_oklab(new)[0]), .003)
        self.assertGreater(stats['changed_pixels'], 0)
        self.assertEqual(result.size, original.size)

    def test_inside_feather_never_changes_unselected_pixels(self):
        original = self.fabric()
        mask = recoloring.selection(original.size, region=[3, 2, 9, 6])
        result, applied, _ = recoloring.recolor(original, mask, 'royalblue', feather=3)
        for y in range(original.height):
            for x in range(original.width):
                if mask.getpixel((x, y)) == 0:
                    self.assertEqual(applied.getpixel((x, y)), 0)
                    self.assertEqual(result.getpixel((x, y)), original.getpixel((x, y)))

    def test_zero_strength_is_exact_identity_and_alpha_survives(self):
        original = self.fabric('RGBA')
        mask = Image.new('L', original.size, 255)
        unchanged, _, stats = recoloring.recolor(original, mask, 'blue', strength=0)
        self.assertEqual(unchanged.tobytes(), original.tobytes())
        self.assertEqual(stats['changed_pixels'], 0)
        edited, _, _ = recoloring.recolor(original, mask, 'blue')
        self.assertEqual(edited.getchannel('A').tobytes(), original.getchannel('A').tobytes())

    def test_gamut_compression_preserves_lightness_at_extremes(self):
        for rgb in [(255, 240, 10), (8, 5, 2), (245, 180, 160), (255, 255, 255), (0, 0, 0)]:
            image = Image.new('RGB', (1, 1), rgb)
            edited, _, _ = recoloring.recolor(image, Image.new('L', (1, 1), 255), 'blue', saturation=2)
            self.assertLess(abs(recoloring.to_oklab(rgb)[0]-recoloring.to_oklab(edited.getpixel((0, 0)))[0]), .003)

    def test_grey_fabric_can_be_coloured_and_zero_saturation_desaturates(self):
        image = Image.new('RGB', (1, 1), (140, 140, 140))
        mask = Image.new('L', image.size, 255)
        blue, _, _ = recoloring.recolor(image, mask, 'royalblue')
        self.assertGreater(blue.getpixel((0, 0))[2], blue.getpixel((0, 0))[0])
        grey, _, _ = recoloring.recolor(self.fabric(), Image.new('L', (12, 8), 255), 'blue', saturation=0)
        for y in range(grey.height):
            for x in range(grey.width):
                rgb = grey.getpixel((x, y))
                self.assertLessEqual(max(rgb)-min(rgb), 1)

    def test_mask_alpha_and_region_intersection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mask.png'
            mask = Image.new('RGBA', (4, 4), (255, 255, 255, 255))
            mask.putpixel((1, 1), (255, 255, 255, 0))
            mask.save(path)
            selected = recoloring.selection((4, 4), path, [1, 1, 3, 3])
            self.assertEqual(selected.getpixel((1, 1)), 0)
            self.assertEqual(selected.getpixel((2, 2)), 255)
            self.assertEqual(selected.getpixel((0, 0)), 0)
            self.assertEqual(selected.getpixel((3, 3)), 0)

    def test_invalid_selection_and_parameters_are_rejected(self):
        for region in ([.8, .1, .2, .9], [0, 0, 20, 3], [0, 0, float('nan'), 1]):
            with self.assertRaises(ValueError):
                recoloring.selection((12, 8), region=region)
        with self.assertRaises(ValueError):
            recoloring.selection((12, 8))
        image = self.fabric()
        mask = Image.new('L', image.size, 255)
        for params in ({'strength': 2}, {'strength': float('nan')}, {'saturation': -1}, {'feather': 65}):
            with self.assertRaises(ValueError):
                recoloring.recolor(image, mask, 'blue', **params)
        for color in ('not-a-colour', '#3267c880'):
            with self.assertRaises(ValueError):
                recoloring.recolor(image, mask, color)
        with self.assertRaises(ValueError):
            recoloring.recolor(image, Image.new('L', image.size, 0), 'blue')


if __name__ == '__main__':
    unittest.main()
