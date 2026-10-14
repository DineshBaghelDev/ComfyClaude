import copy
import unittest
from unittest.mock import patch

import server
import workflows


class CheckpointModelsTest(unittest.TestCase):
    def setUp(self):
        self.profile = copy.deepcopy(server.CFG['generation_models']['noobai_vpred'])

    def test_vpred_uses_checkpoint_vae_clip_and_correct_latent(self):
        graph = workflows.checkpoint_image(self.profile, 'panel', 'bad', 832, 1216, 42)
        self.assertEqual(graph['prediction']['inputs']['sampling'], 'v_prediction')
        self.assertTrue(graph['prediction']['inputs']['zsnr'])
        self.assertEqual(graph['latent']['class_type'], 'EmptyLatentImage')
        self.assertEqual(graph['decode']['inputs']['vae'], ['checkpoint', 2])
        self.assertEqual(graph['neg']['inputs']['text'], 'bad')
        self.assertEqual(graph['sampler']['inputs']['sampler_name'], 'euler')
        self.assertNotIn('unet', graph)

    def test_loras_chain_model_and_clip_before_prediction_and_rescale(self):
        self.profile['cfg_rescale'] = .2
        graph = workflows.checkpoint_image(self.profile, 'panel', '', 832, 1216, 42,
            loras=[{'name': 'character.safetensors', 'strength_model': .8}, {'name': 'style.safetensors'}])
        self.assertEqual(graph['lora_1']['inputs']['model'], ['lora_0', 0])
        self.assertEqual(graph['lora_1']['inputs']['clip'], ['lora_0', 1])
        self.assertEqual(graph['pos']['inputs']['clip'], ['lora_1', 1])
        self.assertEqual(graph['prediction']['inputs']['model'], ['lora_1', 0])
        self.assertEqual(graph['sampler']['inputs']['model'], ['rescale', 0])

    def test_epsilon_reference_does_not_use_z_image_or_mask_conditioning(self):
        self.profile.update(prediction_type='eps', zsnr=False)
        graph = workflows.checkpoint_image(self.profile, 'panel', '', 832, 1216, 42, 12, 'reference.png', .4)
        self.assertEqual(graph['prediction']['inputs']['sampling'], 'eps')
        self.assertFalse(graph['prediction']['inputs']['zsnr'])
        self.assertEqual(graph['latent']['class_type'], 'VAEEncode')
        self.assertEqual(graph['sampler']['inputs']['denoise'], .4)
        self.assertEqual(graph['sampler']['inputs']['steps'], 12)

    def test_missing_checkpoint_does_not_fall_back_or_submit(self):
        with patch.object(server, 'ensure_comfy'), patch.object(server.comfy, 'object_info',
                return_value={n: {} for n in ['CheckpointLoaderSimple', 'ModelSamplingDiscrete',
                    'CLIPTextEncode', 'KSampler', 'VAEDecode', 'SaveImage', 'EmptyLatentImage', 'RescaleCFG']}), \
                patch.object(server, '_has', return_value=False), patch.object(server, '_run') as run:
            with self.assertRaises(FileNotFoundError):
                server.generate_with_model('panel')
            run.assert_not_called()

    def test_tool_preserves_empty_negative_and_records_profile(self):
        with patch.object(server, 'ensure_comfy'), patch.object(server.comfy, 'object_info',
                return_value={n: {} for n in ['CheckpointLoaderSimple', 'ModelSamplingDiscrete',
                    'CLIPTextEncode', 'KSampler', 'VAEDecode', 'SaveImage', 'EmptyLatentImage', 'RescaleCFG']}), \
                patch.object(server, '_has', return_value=True), patch.object(server, '_run') as run:
            server.generate_with_model('panel', negative_prompt='', seed=42, steps=12)
            graph, meta = run.call_args.args[:2]
            self.assertEqual(graph['neg']['inputs']['text'], '')
            self.assertEqual(meta['model'], 'noobai_vpred')
            self.assertEqual(meta['profile']['prediction_type'], 'v_prediction')
            self.assertEqual(meta['steps'], 12)

    def test_invalid_model_and_dimensions_fail_before_comfy(self):
        with patch.object(server, 'ensure_comfy') as connect:
            for params in ({'model': 'unknown'}, {'width': 833}, {'strength': 0}):
                with self.assertRaises(ValueError):
                    server.generate_with_model('panel', **params)
            connect.assert_not_called()

    def test_noobai_setup_is_checkpoint_only(self):
        import setup_models
        with patch('sys.argv', ['setup_models.py', '--models-dir', 'unused', '--tiers', 'noobai']), \
                patch.object(setup_models, 'download') as download, \
                patch.object(setup_models, 'install_gguf_node') as gguf:
            setup_models.main()
            self.assertEqual([call.args[0] for call in download.call_args_list], ['noobai'])
            gguf.assert_not_called()


if __name__ == '__main__':
    unittest.main()
