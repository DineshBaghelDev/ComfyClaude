# Optional SDXL / NoobAI models

The plugin supplies reusable checkpoint loading, prediction settings, reference img2img and LoRA
loading through `generate_with_model`. Your comic extension holds character descriptions,
approved character/style LoRAs, panel conventions and project prompts. Model weights are separate
and are never automatically downloaded by MCP calls. Installing the plugin does not install NoobAI.

The optional `noobai_vpred` profile uses the author's NoobAI XL V-pred 1.0 checkpoint, bundled
CLIP/VAE, Euler, 30 steps, CFG 5 and zero-terminal-SNR V-prediction. It uses standard ComfyUI nodes.
CFG rescaling is configurable through `cfg_rescale` (0.2 in the NoobAI profile); there is no CLIP skip.
The [author's model card](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/main/README.md)
documents settings and prohibits commercial use of generated products. Review the exact version's
terms before publishing a commercial comic; the plugin's MIT license does not license model outputs.

After agreeing to the approximately 7 GB download:

```powershell
uv run python setup_models.py --models-dir "D:/ComfyUI/models" --tiers noobai
```

Use your actual shared model folder for ComfyUI Desktop. This downloads only the checkpoint, not
Z-Image, GGUF nodes, ControlNet or other extras. Existing files are reused. Restart the MCP server
after changing config; call `list_capabilities` to refresh available ComfyUI model filenames.

```python
generate_with_model(model="noobai_vpred", prompt="1boy, adult, short black hair, blue shirt, comic panel, clean lineart, stone alley", width=832, height=1216, seed=42)
```

Optional arguments: `negative_prompt` (None uses the profile default; "" disables it), `steps`,
`image_path`, `strength` (reference denoising), `save_to`, and
`loras=[{"name": "my_character.safetensors", "strength_model": 0.8, "strength_clip": 0.8}]`.
Install compatible LoRAs into `models/loras`; compatibility cannot be inferred from filenames.
Img2img center-crops the reference to the requested aspect ratio. It does not guarantee identity.
NoobAI generation is separate from the dedicated inpaint/outpaint backend and Z-Image ControlNet.

For another SDXL checkpoint, merge a profile into `~/.comfyclaude/config.json`:

```json
{
  "generation_models": {
    "my_sdxl": {
      "architecture": "sdxl",
      "description": "My SDXL model",
      "checkpoint": "my-model.safetensors",
      "prediction_type": "eps",
      "zsnr": false,
      "steps": 30,
      "cfg": 5.0,
      "sampler_name": "euler",
      "scheduler": "normal",
      "prompt_prefix": "",
      "negative_prompt": "low quality"
    }
  }
}
```

Then pass `model="my_sdxl"`. Follow that version's settings; epsilon and V-pred weights are not
interchangeable. A missing selected checkpoint returns an error rather than falling back to another
model. The default Z-Image tools and quality tiers are unchanged.

Use `/comfyclaude:comic-panels` for the bundled comic skill. Its reference templates and the
`noobai_character_portrait`, `noobai_dialogue_closeup`, `noobai_establishing_scene` custom workflows
ship with this plugin. No separate extension is required. Personal character sheets, recurring
prompts and panel manifests stay in the user's comic project. A personal `comic-noobai` skill can
also be installed in Claude/Codex; it uses its own relative template references.

Example custom workflow:

```python
run_custom_workflow(name="noobai_character_portrait", params={"character": "adult man, short black hair, blue collared shirt", "expression": "calm", "scene": "sunlit town square", "seed": 42})
```

Bundled workflow filenames/settings are examples for V-pred 1.0. User JSON in
`~/.comfyclaude/workflows` overrides bundled workflows with the same name. Use
`generate_with_model` for configured model selection and LoRA chains.
