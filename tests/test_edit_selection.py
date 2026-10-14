import unittest
from unittest.mock import patch

import server


class EditSelectionTest(unittest.TestCase):
    def test_auto_prefers_installed_content_aware_model(self):
        with patch.object(server, 'ensure_comfy'), patch.object(server.comfy, 'object_info'), \
                patch.object(server, '_has', return_value=True), patch.object(server, '_pick_quality') as legacy:
            name, quality, model, _ = server._pick_edit_model('auto', 'balanced', 'inpaint')
            self.assertEqual(name, 'sd15_inpaint')
            self.assertIsNone(quality)
            self.assertEqual(model['checkpoint'], 'sd-v1-5-inpainting.ckpt')
            legacy.assert_not_called()

    def test_explicit_missing_model_does_not_silently_fall_back(self):
        with patch.object(server, 'ensure_comfy'), patch.object(server.comfy, 'object_info'), \
                patch.object(server, '_has', return_value=False), patch.object(server, '_pick_quality') as legacy:
            with self.assertRaises(FileNotFoundError):
                server._pick_edit_model('sd15_inpaint', '', 'outpaint')
            legacy.assert_not_called()

    def test_auto_missing_model_reports_compatibility_fallback(self):
        with patch.object(server, 'ensure_comfy'), patch.object(server.comfy, 'object_info'), \
                patch.object(server, '_has', return_value=False), \
                patch.object(server, '_pick_quality', return_value=('balanced', {}, '')):
            _, _, model, note = server._pick_edit_model('auto', '', 'outpaint')
            self.assertIsNone(model)
            self.assertIn('Dedicated inpainting model unavailable', note)

    def test_explicit_z_image_keeps_requested_quality(self):
        with patch.object(server, 'ensure_comfy'), patch.object(server, '_pick_quality',
                return_value=('max', {}, '')) as legacy:
            _, _, model, _ = server._pick_edit_model('z_image', 'max', 'inpaint')
            self.assertIsNone(model)
            legacy.assert_called_once_with('max', 'inpaint')


if __name__ == '__main__':
    unittest.main()
