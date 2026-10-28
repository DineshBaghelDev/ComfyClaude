"""CPU checks for Studio helpers, Z-Image LoRA wiring and custom workflow validation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import server
import settings
import ui
import workflows


class ZImageLoraTest(unittest.TestCase):
    def setUp(self):
        self.cfg, _, _ = settings.load()
        self.q = self.cfg['qualities']['balanced']

    def test_loras_chain_between_unet_and_sampling_patch(self):
        graph = workflows.text_to_image(self.q, self.cfg, 'a fox', 1024, 1024, 1, 8)
        workflows.add_model_loras(graph, [{'name': 'a.safetensors', 'strength_model': 0.55},
                                          {'name': 'b.safetensors'}])
        self.assertEqual(graph['lora_0']['inputs']['model'], ['unet', 0])
        self.assertEqual(graph['lora_0']['inputs']['strength_model'], 0.55)
        self.assertEqual(graph['lora_1']['inputs']['model'], ['lora_0', 0])
        self.assertEqual(graph['lora_1']['inputs']['strength_model'], 1.0)
        self.assertEqual(graph['model']['inputs']['model'], ['lora_1', 0])

    def test_no_loras_leaves_graph_unchanged(self):
        graph = workflows.text_to_image(self.q, self.cfg, 'a fox', 1024, 1024, 1, 8)
        before = json.dumps(graph, sort_keys=True)
        workflows.add_model_loras(graph, None)
        self.assertEqual(json.dumps(graph, sort_keys=True), before)


class CustomWorkflowValidationTest(unittest.TestCase):
    def test_missing_placeholder_is_rejected_before_contacting_comfyui(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'needs_image.json'
            path.write_text(json.dumps({'load': {'class_type': 'LoadImage', 'inputs': {'image': '{{image_layout}}'}},
                                        'pos': {'class_type': 'CLIPTextEncode', 'inputs': {'text': '{{prompt}}'}}}))
            with mock.patch.object(server, '_custom_workflows', return_value={'needs_image': path}), \
                    mock.patch.object(server, 'ensure_comfy') as ensure:
                with self.assertRaisesRegex(ValueError, r"image_layout"):
                    server.run_custom_workflow('needs_image', {'prompt': 'x'})
                ensure.assert_not_called()


class StudioHelpersTest(unittest.TestCase):
    def test_result_text_is_parsed_into_labelled_images_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, mask = Path(tmp) / 'out.png', Path(tmp) / 'out_mask.png'
            for p in (out, mask):
                Image.new('RGB', (4, 4)).save(p)
            out.with_suffix('.json').write_text(json.dumps({'seed': 7}))
            text = f"Done.\nImages:\n{out}\nApplied mask: {mask}\nSaved to: {out}"
            images = ui._images(text)
        self.assertEqual([i['label'] for i in images], ['', 'Applied mask'])
        self.assertEqual(images[0]['meta'], {'seed': 7})

    def test_only_images_inside_the_output_folder_are_served(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (4, 4)).save(root / 'a.png')
            (root / 'jobs.json').write_text('{}')
            with mock.patch.object(ui.S, 'OUT', root):
                self.assertTrue(ui._allowed(str(root / 'a.png')))
                self.assertFalse(ui._allowed(str(root / 'jobs.json')))
                self.assertFalse(ui._allowed(str(root / '..' / 'a.png')))


if __name__ == '__main__':
    unittest.main()
