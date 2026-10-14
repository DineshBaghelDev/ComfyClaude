"""Regression checks for edit graph wiring and bounded MCP parameters."""
import unittest

import settings
import workflows


class EditWorkflowsTest(unittest.TestCase):
    def setUp(self):
        self.cfg, _, _ = settings.load()
        self.q = self.cfg['qualities']['balanced']

    def test_all_inpaint_mask_sources_use_separate_sampling_and_blending(self):
        for mode in ('region', 'file', 'subject', 'background'):
            with self.subTest(mode=mode):
                graph = workflows.inpaint(self.q, self.cfg, 'scene', 'source.png', mode,
                                          'mask.png', 1.0, 16, 42, feather=31)
                sampling = graph['latent']['inputs']['mask']
                blending = graph['composite']['inputs']['mask']
                self.assertNotEqual(sampling, blending)
                self.assertEqual(blending, ['blend_mask', 0])
                # Gaussian support includes diagonal pixels: square growth must cover them.
                self.assertFalse(graph[sampling[0]]['inputs']['tapered_corners'])
                self.assertGreater(graph[sampling[0]]['inputs']['expand'], 31)
                self.assertEqual(graph['save']['inputs']['images'], ['composite', 0])

    def test_hard_paste_has_no_blur_nodes(self):
        graph = workflows.inpaint(self.q, self.cfg, 'scene', 'source.png', 'file',
                                  'mask.png', 1.0, 0, 42, feather=0)
        self.assertNotIn('mask_blur', graph)
        self.assertEqual(graph['composite']['inputs']['mask'], ['mask', 0])

    def test_outpaint_preserves_source_through_masked_composite(self):
        for padding in ((192, 0, 192, 0), (0, 192, 0, 192), (96, 96, 96, 96)):
            with self.subTest(padding=padding):
                graph = workflows.outpaint(self.q, self.cfg, 'scene', 'source.png',
                                           *padding, 42, feather=40)
                self.assertEqual(graph['composite']['inputs']['destination'], ['pad', 0])
                self.assertEqual(graph['composite']['inputs']['mask'], ['pad', 1])
                self.assertEqual(graph['latent']['inputs']['mask'], ['pad', 1])
                self.assertEqual(graph['pad']['inputs']['feathering'], 40)
                self.assertEqual(graph['save']['inputs']['images'], ['composite', 0])

    def test_invalid_feather_is_rejected(self):
        for feather in (-1, 32):
            with self.assertRaises(ValueError):
                workflows.inpaint(self.q, self.cfg, 'scene', 'source.png', 'region',
                                  'mask.png', 1.0, 16, 42, feather=feather)
        for feather in (-1, 16369):
            with self.assertRaises(ValueError):
                workflows.outpaint(self.q, self.cfg, 'scene', 'source.png',
                                   192, 0, 0, 0, 42, feather=feather)

    def test_content_aware_fill_conditions_on_source_and_mask(self):
        model = self.cfg['editing']['models']['sd15_inpaint']
        for tool in ('inpaint', 'outpaint'):
            with self.subTest(tool=tool):
                if tool == 'inpaint':
                    graph = workflows.inpaint(None, self.cfg, 'boat', 'source.png', 'region',
                                              'mask.png', 1.0, 16, 42, edit_model=model)
                    expected_pixels, expected_mask = ['load', 0], ['sample_mask', 0]
                else:
                    graph = workflows.outpaint(None, self.cfg, 'lake', 'source.png',
                                               128, 0, 128, 0, 42, edit_model=model)
                    expected_pixels, expected_mask = ['pad', 0], ['pad', 1]
                self.assertEqual(graph['latent']['class_type'], 'InpaintModelConditioning')
                self.assertEqual(graph['latent']['inputs']['pixels'], expected_pixels)
                self.assertEqual(graph['latent']['inputs']['mask'], expected_mask)
                self.assertEqual(graph['sampler']['inputs']['positive'], ['latent', 0])
                self.assertEqual(graph['sampler']['inputs']['negative'], ['latent', 1])
                self.assertEqual(graph['sampler']['inputs']['latent_image'], ['latent', 2])
                self.assertEqual(graph['sampler']['inputs']['steps'], 24)
                self.assertNotIn('unet', graph)
                self.assertNotIn('ModelSamplingAuraFlow', [n['class_type'] for n in graph.values()])


if __name__ == '__main__':
    unittest.main()
