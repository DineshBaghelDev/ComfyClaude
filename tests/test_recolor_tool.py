import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import server


class RecolorToolTest(unittest.TestCase):
    def test_cpu_tool_keeps_alpha_and_saves_png_without_contacting_comfyui(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, mask = root/'source.png', root/'mask.png'
            original = Image.new('RGBA', (8, 6), (190, 105, 50, 123))
            original.save(source)
            Image.new('L', original.size, 255).save(mask)
            saved = root/'project'/'blue.png'
            with patch.object(server, 'OUT', root/'outputs'), \
                    patch.object(server, 'ensure_comfy', side_effect=AssertionError('ComfyUI contacted')), \
                    patch.object(server.comfy, 'queue_prompt', side_effect=AssertionError('Diffusion invoked')):
                response = server.recolor_image(str(source), 'royalblue', mask_path=str(mask), save_to=str(saved))
            self.assertIn('Original resolution 8x6', response[0])
            self.assertTrue(saved.exists())
            with Image.open(saved) as edited:
                self.assertEqual(edited.size, original.size)
                self.assertEqual(edited.getchannel('A').tobytes(), original.getchannel('A').tobytes())
            metadata = json.loads(next((root/'outputs').glob('*.json')).read_text())
            self.assertEqual(metadata['changed_pixels'], 48)
            self.assertEqual(metadata['saved_to'], [str(saved)])

    def test_failed_project_copy_still_returns_the_generated_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.png'
            Image.new('RGB', (4, 4), (190, 105, 50)).save(source)
            with patch.object(server, 'OUT', root/'outputs'), \
                    patch.object(server, '_copy_to', side_effect=OSError('destination unavailable')):
                response = server.recolor_image(str(source), 'blue', region=[0, 0, 1, 1], save_to=str(root/'blue.png'))
            self.assertIn('Could not save', response[0])
            self.assertIn('Images:', response[0])
            self.assertEqual(len(list((root/'outputs').glob('*.json'))), 1)

    def test_capabilities_reports_local_tool_when_comfyui_is_unavailable(self):
        with patch.object(server, 'ensure_comfy', side_effect=ConnectionError('offline')):
            self.assertIn('recolor_image [ready]', server.list_capabilities())


if __name__ == '__main__':
    unittest.main()
