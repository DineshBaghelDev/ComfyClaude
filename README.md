# ComfyClaude

Gives Claude (Claude Code / any MCP client) image generation using **ComfyUI + Z-Image Turbo running locally**.
Built for a 6 GB VRAM / 16 GB RAM laptop (RTX 4050), using GGUF-quantized models to fit in memory.

## Features

| Tool | What it does | Models |
|---|---|---|
| `generate_image` | Text → image, style presets, 3 quality tiers | Z-Image Turbo |
| `generate_drafts` | Several fast low-res drafts + contact sheet to pick a seed | Z-Image Turbo |
| `generate_from_reference` | Reference image → new image (img2img) | Z-Image Turbo |
| `generate_with_control` | Follow edges (canny) or a person's pose of a reference | Z-Image Fun ControlNet Union 2.1 lite, SDPose |
| `inpaint` | Regenerate a region / the subject / the background only | Z-Image Turbo, BiRefNet |
| `outpaint` | Extend the canvas | Z-Image Turbo |
| `upscale` | 4x-UltraSharp upscale, optional detail-refine pass | 4x-UltraSharp |
| `remove_background` | Transparent PNG cut-out | BiRefNet |
| `run_custom_workflow` | Run your own exported ComfyUI workflow with `{{placeholders}}` | any |
| `job_status`, `list_capabilities`, `comfy_status`, `free_memory` | Utilities | |

ComfyUI is **started automatically** (headless) when a tool needs it.

### Quality tiers (`config.json`)
| Tier | Model files | Use |
|---|---|---|
| `max` | bf16 Z-Image + bf16 Qwen3-4B (~19 GB) | best quality, slowest |
| `balanced` (default) | GGUF Q8 (~11 GB), 8 steps | practically identical to max |
| `fast` | same Q8 files, 6 steps | drafts / throwaway (no model reload) |
| `lowmem` | GGUF Q5 (~8.5 GB) | only if RAM is tight; K-quants sample slower on small GPUs |

### Performance (RTX 4050 6 GB, 16 GB RAM, 1024x1024)
ComfyUI is started with `--cache-classic --disable-pinned-memory` (`comfy_args` in `config.json`):
the default RAM-pressure cache evicts the diffusion model whenever the text encoder loads, forcing a
full reload every prompt. Measured:

| Setup | Cold | Warm |
|---|---|---|
| bf16, default ComfyUI | 150–330 s | — |
| Q8, default cache | ~155 s | ~125 s |
| Q8, `--cache-classic --disable-pinned-memory` | ~136 s | **~90 s** |
| Q5, default cache | ~185 s | ~150–177 s |

## Setup

1. Install [ComfyUI Desktop](https://www.comfy.org/) with Z-Image Turbo (bf16) and adjust paths in `config.json`,
   `comfy_paths.yaml` and `start_comfy.ps1`.
2. Install the GGUF node and download all models (~30 GB):
   ```
   python setup_models.py
   ```
3. Create the server venv:
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
4. Register with Claude Code (all projects):
   ```
   claude mcp add --scope user comfyclaude -- E:/Tools/ComfyClaude/.venv/Scripts/python.exe E:/Tools/ComfyClaude/server.py
   ```
   Restart Claude Code; the tools appear as `mcp__comfyclaude__*`.

## Custom workflows
In ComfyUI: build a workflow → *Export (API)* → save to `workflows_custom/<name>.json`.
Replace values you want Claude to control with placeholders like `"{{prompt}}"`, `"{{seed}}"`, `"{{image}}"`.
See `workflows_custom/example_t2i_bf16.json`.

## Output
Images are saved to `outputs/`, each with a `.json` sidecar (prompt, seed, quality, time) for reproducibility.
