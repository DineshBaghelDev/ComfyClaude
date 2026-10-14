---
name: setup
description: Set up or troubleshoot ComfyClaude — find the user's ComfyUI, pick model tiers for their GPU/RAM, download models, configure auto-start, and verify. Use when the user installs ComfyClaude, when image tools report ComfyUI unavailable or missing models, or when generation is unexpectedly slow.
---

# ComfyClaude setup

Check installed models first. Before downloading missing weights, explain the specific model,
approximate size and purpose, then ask whether to download it unless already authorized.
Install only the requested capabilities; `--extras` is optional, not part of the default setup.

For NoobAI XL V-pred 1.0, use `--tiers noobai` (~7 GB checkpoint only, bundled CLIP/VAE,
no GGUF node or Z-Image weights). Its author's model card prohibits commercial use of generated
products: https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/main/README.md.
Call `generate_with_model(model="noobai_vpred", ...)` after checking profile readiness.
If an existing checkpoint has a different filename, override `generation_models.noobai_vpred.checkpoint`
in the user's config. Do not apply V-pred settings to epsilon checkpoints; use a separate SDXL
profile with `prediction_type="eps"` and `zsnr=false`, following that version's model card.

For colour-only edits with `recolor_image`, no ComfyUI, GPU or model download is required.
Use a precise local mask and install the plugin's Python dependencies normally. If ComfyUI is
unavailable, `list_capabilities` still reports this local tool as ready.

For content-aware inpaint/outpaint, also install the SD 1.5 inpainting checkpoint:
`uv run --directory <plugin root> python setup_models.py --models-dir <actual models folder> --tiers inpaint`.
This installs the editing checkpoint only (~4.3 GB), with no GGUF node required. Alternatively
add `--inpaint` to the normal Z-Image setup command. Check `list_capabilities` afterwards and
select `edit_model="sd15_inpaint"` to require the dedicated backend.

The plugin root is two folders above this skill's directory (it contains `setup_models.py`).
Run its scripts with `uv run --directory <plugin root> python <script>`.

## 1. Check prerequisites
- `uv --version` — needed to run the server. If missing, point the user to https://docs.astral.sh/uv/.
- `git --version` — needed to install the ComfyUI-GGUF node.
- An NVIDIA GPU is strongly recommended (`nvidia-smi` shows VRAM). Ask for RAM if unsure.

## 2. Find ComfyUI
Ask the user, or look in common places: `ComfyUI_windows_portable/ComfyUI`, `~/ComfyUI`,
ComfyUI Desktop installs (Windows: the folder chosen at install, look for `ComfyUI/main.py`;
the Desktop app often keeps models in a separate shared folder — check its `extra_model_paths`
settings). The right folder contains `main.py`. If ComfyUI is not installed, send them to
https://www.comfy.org/download (Desktop) or https://github.com/comfyanonymous/ComfyUI.

## 3. Choose tiers by hardware
| Hardware | Install | Notes |
|---|---|---|
| 6–8 GB VRAM, 16 GB RAM | `balanced` | default; ~90 s/image on an RTX 4050 laptop |
| 6–8 GB VRAM, 8–12 GB RAM | `lowmem` | smaller files; may sample slower |
| 12–16 GB VRAM, 32 GB RAM | `balanced,max` | `max` becomes practical |
| 24 GB+ VRAM | `max` | set `default_quality` to `max` |
Add `--extras` (~4.5 GB) for ControlNet, pose, background removal and upscaling.

## 4. Download
```
uv run --directory <plugin root> python setup_models.py --comfy-path "<ComfyUI folder>" --tiers balanced
```
If their models live elsewhere (Desktop shared folder, extra_model_paths), add `--models-dir "<models folder>"`.
Downloads are ~11–30 GB and resume if interrupted — run it in the background and keep the user informed.
`--check` lists what is already installed.

## 5. Configure
Plugin settings (`/plugin` → comfyclaude → configure) set the ComfyUI URL, the ComfyUI folder
(enables auto-start) and the output folder. Anything else goes in `~/.comfyclaude/config.json`,
which overrides `config/defaults.json` key by key, for example:
```json
{
  "comfy_path": "D:/ComfyUI",
  "comfy_args": ["--cache-classic", "--disable-pinned-memory", "--extra-model-paths-config", "D:/models.yaml"],
  "default_quality": "max",
  "qualities": {"balanced": {"unet": "my-finetune-Q8_0.gguf"}}
}
```
`comfy_args` are only used when ComfyClaude starts ComfyUI itself.

## 6. Verify
Restart ComfyUI (new node/models), then call `list_capabilities` and generate one test image.
The requested model profiles/tools should show `ready`; unused optional models may remain missing.

## Troubleshooting
- **Slow (model reloads every image)**: start ComfyUI with `--cache-classic --disable-pinned-memory`
  (the default `comfy_args`); with the Desktop app add them to its launch arguments.
- **"ComfyUI-GGUF custom node not installed"**: rerun setup without `--no-node`, or install it with ComfyUI Manager, then restart ComfyUI.
- **Not reachable**: check the port in the ComfyUI window/log and set the ComfyUI URL setting.
- **Out of memory**: use `lowmem`, smaller sizes (768px) then `upscale`, close other GPU apps.
