# ComfyClaude

**Give Claude the ability to create and edit images — locally, privately, for free — using your own
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).**

ComfyClaude is a Claude Code plugin that connects Claude to image tools running on your PC.
Ask in plain language — *"make a poster for my bake sale"*, *"put this product
on a marble table"*, *"give me four logo ideas"* — and Claude writes the prompt, runs the right workflow,
**looks at the result itself**, fixes what's wrong, and hands you the file.

Generation uses Z-Image Turbo through ComfyUI; content-aware fill can use a dedicated SD 1.5
inpainting checkpoint. Colour-only edits use a CPU recolour path that preserves the original
shading and linework. [Version history](CHANGELOG.md) · [GitHub releases](https://github.com/DineshBaghelDev/ComfyClaude/releases)

It is tuned to run on modest hardware: everything here was built and benchmarked on a laptop RTX 4050
with 6 GB VRAM and 16 GB RAM.

<p align="center">
  <img src="docs/ramen.jpg" width="32%" alt="Ramen shop with a neon sign reading COMFY RAMEN">
  <img src="docs/fox.jpg" width="32%" alt="Watercolor fox reading a book next to an OPEN SOURCE sign">
</p>
<p align="center">
  <img src="docs/controlnet.jpg" width="65%" alt="ControlNet: edge map of the ramen shop and a snowy daytime version">
</p>
<p align="center">
  <img src="docs/outpaint.jpg" width="48%" alt="The ramen shop extended into a wide street scene">
  <img src="docs/remove_background.jpg" width="48%" alt="Sneakers product shot and its transparent cut-out">
</p>
<p align="center"><sub>All generated on a 6 GB laptop GPU. Top: text-to-image. Middle: ControlNet (same layout, new scene).
Bottom: outpaint and background removal.</sub></p>

## What it can do

| Tool | What it does |
|---|---|
| `generate_image` | Text → image, 12 style presets, accurate text rendering (signs, posters) |
| `generate_drafts` | Several quick drafts side by side; pick one and finalize it |
| `generate_from_reference` | Use an image as a reference: variations, restyling (img2img), optionally plus a second image for pose/outlines |
| `generate_with_control` | Follow a reference's exact outlines (canny) or a person's pose |
| `inpaint` | Content-aware fill / replace a region, subject, or background with a dedicated inpainting model |
| `outpaint` | Content-aware extension beyond image borders (e.g. square → widescreen) |
| `recolor_image` | Change garment/prop colour while preserving shading, linework and original resolution; no models |
| `upscale` | 2–4× upscale, optional detail-adding refine pass |
| `remove_background` | Transparent PNG cut-outs |
| `run_custom_workflow` | Run any workflow you export from ComfyUI |
| `list_capabilities`, `job_status`, `comfy_status`, `free_memory` | Utilities |

## How it works

```
 You ──► Claude ──(MCP tools)──► ComfyClaude server ──(HTTP API)──► ComfyUI on your GPU
           ▲                          │  picks workflow + quality,       │
           │                          │  resizes inputs, builds masks    │
           └──── Claude opens the ◄───┴──── image saved to disk ◄────────┘
                 image and checks it
```

- **Plugin** = the package you install. It bundles an **MCP server** (gives Claude the tools) and
  **skills** (teach Claude when to use which tool, how to prompt Z-Image, and to always check its work).
- The server builds ComfyUI workflows in code, so one graph works across all quality tiers.
- `recolor_image` runs directly on CPU at the original displayed resolution and needs no ComfyUI connection.
- It asks ComfyUI which model files exist (`/object_info`), so it works with any install layout.
- If ComfyUI isn't running and you've told it where ComfyUI lives, it starts ComfyUI in the background.
- Results come back to Claude with a small preview, so it can check them immediately.
- Images are saved to the output folder with a `.json` next to each (prompt, seed, settings, time), and can
  also be saved straight into your project: *"save it as assets/hero.webp"* (`save_to`).
- If ComfyUI is already running (port 8188 or the Desktop app's 8000) it's reused; otherwise it's started.

## Requirements

These requirements cover image generation and diffusion editing. Masked recolouring only needs
the plugin's Python dependencies; it can run without ComfyUI or a GPU.

- [Claude Code](https://code.claude.com)
- [ComfyUI](https://www.comfy.org/download) (Desktop, portable, or manual install), recent version
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (runs the server; installs its Python deps automatically) and `git`
- An NVIDIA GPU with **6 GB+ VRAM** and **16 GB+ RAM** recommended. A big pagefile/swap helps on 16 GB.
- Disk: ~11 GB for the default models, ~30 GB for everything

## Install

**1. Install the plugin** (in Claude Code 2.1.275 or later):
```
/plugin install comfyclaude --marketplace DineshBaghelDev/ComfyClaude
```
On older versions:
```
/plugin marketplace add DineshBaghelDev/ComfyClaude
/plugin install comfyclaude@comfyclaude
```

**2. Let Claude set it up.** Just say:
> *"Set up ComfyClaude"*

The setup skill finds your ComfyUI, asks about your GPU, installs the
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) node and downloads the right models. Or do it yourself:
```
uv run --directory <plugin folder> python setup_models.py --comfy-path "C:/path/to/ComfyUI" --tiers balanced --extras
```

**3. Configure** (`/plugin` → comfyclaude → configure):
| Setting | Meaning |
|---|---|
| ComfyUI URL | Default `http://127.0.0.1:8188`. ComfyUI Desktop often uses `8000`. |
| ComfyUI folder | Optional. Folder with `main.py` — enables auto-start. |
| Output folder | Optional. Default `~/ComfyClaude/outputs`. |

**4. Restart ComfyUI and try it:** *"Generate a cozy ramen shop at night with a neon sign reading 'COMFY RAMEN'."*

<details>
<summary>Without the plugin system (any MCP client)</summary>

```
git clone https://github.com/DineshBaghelDev/ComfyClaude
cd ComfyClaude
claude mcp add --scope user comfyclaude -- uv run --directory "$PWD" python server.py
```
Set `COMFYUI_URL` / `COMFYUI_PATH` / `COMFYCLAUDE_OUTPUT_DIR` env vars or use `~/.comfyclaude/config.json`.
</details>

## Quality tiers

| Tier | Models | Memory | Use for |
|---|---|---|---|
| `balanced` *(default)* | Z-Image Turbo GGUF Q8 + Qwen3-4B Q8, 8 steps | ~11 GB | Everything. Visually near-identical to `max`. |
| `fast` | same files, 6 steps | same | Drafts and throwaway images (no model reload) |
| `lowmem` | GGUF Q5 | ~8.5 GB | Low RAM. Note: K-quants can sample *slower* on small GPUs. |
| `max` | original bf16 | ~19 GB | Best possible quality; practical with 12 GB+ VRAM / 32 GB RAM |

Claude picks the tier per task (drafts → `fast`, finals → `balanced`) unless you ask for one.

### Which tiers should I install?
| Hardware | Tiers |
|---|---|
| 6–8 GB VRAM, 16 GB RAM | `balanced` |
| 6–8 GB VRAM, 8–12 GB RAM | `lowmem` |
| 12–16 GB VRAM, 32 GB RAM | `balanced,max` |
| 24 GB+ VRAM | `max` (set `default_quality` to `max`) |

### Performance (RTX 4050 Laptop 6 GB, 16 GB RAM, 1024×1024)
| Setup | First image | Next images |
|---|---|---|
| bf16, ComfyUI defaults | 150–330 s | — |
| Q8, ComfyUI defaults | ~155 s | ~125 s |
| **Q8, `--cache-classic --disable-pinned-memory`** | ~135 s | **~90 s** |
| Q8 drafts at 768 px, 6 steps | — | **~20 s** each |

On 16 GB RAM, ComfyUI's default cache evicts the diffusion model every time the text encoder loads, so
every image reloads ~7 GB from disk. ComfyClaude starts ComfyUI with `--cache-classic --disable-pinned-memory`
to keep models loaded. **If you start ComfyUI yourself, add those flags** (ComfyUI Desktop: server launch arguments).
Please share numbers from your GPU in an issue!

## Customize

Everything is overridable in **`~/.comfyclaude/config.json`** — it's merged over
[`config/defaults.json`](config/defaults.json) key by key:

```json
{
  "comfy_path": "D:/ComfyUI",
  "comfy_args": ["--cache-classic", "--disable-pinned-memory", "--extra-model-paths-config", "D:/models.yaml"],
  "default_quality": "max",
  "qualities": {
    "balanced": { "unet": "my-z-image-finetune-Q8_0.gguf" },
    "int8": {
      "description": "Official int8 build",
      "loader": "standard",
      "unet": "z_image_turbo_int8_convrot.safetensors",
      "clip": "qwen_3_4b_fp8_mixed.safetensors"
    }
  },
  "sampler": { "steps": 9 },
  "models": { "upscaler": "RealESRGAN_x4plus.pth" }
}
```

- **Your own tiers / fine-tunes**: add entries to `qualities`. Comfy-Org also publishes
  [int8 and nvfp4 builds](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files) of Z-Image Turbo
  that may be faster on some GPUs (nvfp4 needs RTX 50-series) — untested here, benchmarks welcome.
- **Style presets**: add `~/.comfyclaude/presets.json` (`{"noir": "Black and white film noir photo, ..."}`).
- **Custom workflows**: in ComfyUI, *Export (API)* a workflow into `~/.comfyclaude/workflows/<name>.json`, and replace
  values Claude should control with placeholders like `"{{prompt}}"`, `"{{seed}}"`, `"{{image}}"`.
  Claude runs it with `run_custom_workflow`. See [`workflows_custom/example_t2i_bf16.json`](workflows_custom/example_t2i_bf16.json).
- **Other models / new tools**: see [CONTRIBUTING.md](CONTRIBUTING.md).

### Using a different model

For **content-aware inpaint/outpaint**, install the dedicated checkpoint:

```text
uv run python setup_models.py --models-dir "C:/path/to/ComfyUI/models" --tiers inpaint
```

Use your actual shared models folder for ComfyUI Desktop. This adds the published SD 1.5
inpainting checkpoint (~4.3 GB); its license is CreativeML OpenRAIL-M. The tool's
`edit_model="auto"` prefers an installed editing checkpoint and reports the backend used.
`edit_model="sd15_inpaint"` requires that checkpoint and errors if it is unavailable.
This supplies both the masked image and mask through `InpaintModelConditioning` rather than
only sampling with a noise mask. Start at `megapixels=0.26` (~512×512 working source) on a 6 GB GPU.
`edit_model="z_image"` explicitly keeps the previous Z-Image workflow.

To use another compatible SD-style **inpainting checkpoint**, install it under
`models/checkpoints` and add a named editing model in `~/.comfyclaude/config.json`:

```json
{
  "editing": {
    "default_model": "my_fill",
    "models": {
      "my_fill": {
        "description": "My dedicated inpainting checkpoint",
        "checkpoint": "my-inpaint-model.safetensors",
        "steps": 24,
        "cfg": 7.0,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras"
      }
    }
  }
}
```

Restart the plugin's MCP server after changing config, check `list_capabilities`, then use
`edit_model="my_fill"` on `inpaint` or `outpaint`. This checkpoint backend is tested with
SD 1.5 inpainting; other families needing different loaders or conditioning use custom workflows.
Z-Image `quality` tiers do not apply when a dedicated editing model is selected.

For a **compatible Z-Image Turbo fine-tune**, install its diffusion weights in ComfyUI's
`models/diffusion_models` (or the shared folder configured in your ComfyUI installation).
Add a tier in `~/.comfyclaude/config.json`, keeping the matching text encoder and VAE:

```json
{
  "qualities": {
    "custom": {
      "description": "My Z-Image Turbo fine-tune",
      "loader": "standard",
      "unet": "my-z-image-turbo.safetensors",
      "clip": "qwen_3_4b.safetensors"
    }
  }
}
```

For compatible GGUF weights, use `"loader": "gguf"` with the corresponding GGUF diffusion
and text-encoder files. Restart the plugin's MCP server after editing its config. Call
`list_capabilities` to confirm the tier is ready, then ask Claude to use `quality="custom"`.
If its files are missing, the plugin may fall back to another installed tier and reports that
in the result. `default_quality` can also be set to `custom` for tools without their own default;
explicitly passing `quality` is the reliable way to select it for every generation/edit tool.
For editing with a Z-Image fine-tune, also pass `edit_model="z_image"` so an installed dedicated
inpainting checkpoint does not take precedence.

For **a different model family**, such as SDXL or Flux, use its own working ComfyUI workflow.
The built-in tools use Z-Image's text encoder, latent format and sampling setup; a filename
override does not make those graphs compatible with another family.

1. Install the model and any nodes its workflow needs, and verify the workflow in ComfyUI.
2. Use **File → Export Workflow (API)**, and save it as
   `~/.comfyclaude/workflows/my_model.json` (API format, rather than the editor's ordinary JSON).
3. Replace the positive prompt value with `"{{prompt}}"` and the seed value with `"{{seed}}"`.
   For editing, replace a LoadImage filename with `"{{image}}"`; use `"{{image_mask}}"` for
   a separate mask image. Keep the model-specific loader and conditioning nodes intact.
4. Ask Claude to call:

```text
run_custom_workflow(name="my_model", params={"prompt": "An alpine lake", "seed": 42})
```

For an editing workflow, also pass `"image": "C:/images/source.png"` and, when needed,
`"image_mask": "C:/images/mask.png"`. Existing local files in parameters starting with
`image` are uploaded automatically. Include a SaveImage node so the plugin can return the result.

### Blending inpaint and outpaint

For **colour-only changes**, use `recolor_image` with a precise black/white mask:

```text
recolor_image(image_path="C:/images/panel.png", mask_path="C:/images/shirt_mask.png", color="#3267C8")
```

It changes hue/chroma while retaining perceived lightness in Oklab, without regenerating any
geometry, folds or details. Source dimensions and transparency are kept. `strength` blends
the change (0–1), `saturation` adjusts colour intensity (0–2), and `feather` softens only inside
the selection (0–64 original pixels, default 0). A `region` can restrict the mask or select a
rectangle; it changes every selected pixel, so precise masks are best for clothing and hair.
The target colour's brightness is intentionally not applied; neutral targets desaturate rather
than changing exposure. This tool runs on CPU without ComfyUI or downloaded models and returns
the edited image, an applied-mask path and JSON metadata. Save as PNG to retain exact pixels.

Inpaint samples a larger area than it pastes, giving the model room to join new content
to the source. Its `grow` expands the edit area; `feather` softens its edge (0–31 working
pixels, default 16). Outpaint's `feather` blends into the source along extended edges (default
40 working pixels; less than half each working source dimension), with progressively less
sampling noise inside that overlap to retain source structure. Setting `feather=0` gives
a hard join. Pixels outside the blend remain identical to the **resized working source**;
the tools still resize inputs according to `megapixels`.

Describe the whole scene, including lighting and texture, in the edit prompt. Soft blending
reduces boundary artifacts but cannot guarantee matching content for every prompt or seed.
Give an inserted object enough room inside the mask; if the model draws it across the blend
boundary, its edges can fade or clip. Prefer the dedicated `sd15_inpaint` backend for
content-aware fill. For model families with a different conditioning setup, use their own
editing graph through `run_custom_workflow`.

To verify locally before a release, run the regression checks and the real MCP smoke test:

```text
uv run python -m unittest discover -s tests -p "test_*.py" -v
uv run python tests/live_edit_check.py
```

The live test launches this checkout's MCP server over stdio, calls its generation and editing
tools against local ComfyUI, and checks output sizes, protected pixels and invalid parameters.
It requires installed models and takes several minutes. It saves a report and a comparison
image under `outputs/mcp_edit_check.*` for visual inspection. Use `--source <image.png>` to
reuse an existing source.

The recolour path has its own MCP check, which requires an image and a precise mask but no models:

```text
uv run python tests/live_recolor_check.py --source panel.png --mask shirt_mask.png --color "#3267C8"
```

It checks original dimensions, exact pixels outside the mask, perceived-lightness preservation,
PNG project copies and input validation. Results and a comparison are saved under
`outputs/recolor_checks/`. Inspect the mask and image visually as well: a colour-based mask can
accidentally select similar-coloured skin outlines or background details. Remove those from the
mask before accepting the result.

## Troubleshooting

| Problem | Fix |
|---|---|
| "ComfyUI is not running" | Start ComfyUI, check the port in its window, or set the ComfyUI folder for auto-start |
| "ComfyUI-GGUF custom node not installed" | Run setup again or install it via ComfyUI Manager; restart ComfyUI |
| Every image is slow, model reloads | Start ComfyUI with `--cache-classic --disable-pinned-memory` |
| Out of memory | `lowmem` tier, smaller images then `upscale`, close other GPU apps, bigger pagefile |
| Missing model files | `setup_models.py --check`, then download the missing tier/extras |
| Windows access violation in ComfyUI-GGUF during model unloading | On the locally tested ComfyUI 0.40.0 / PyTorch 2.12.1 setup, the final outpaint test completed with `--disable-dynamic-vram --disable-async-offload`. Try these server launch flags if available in your ComfyUI version; reducing canvas size also reduces memory pressure. This is a runtime workaround, not a guarantee for every GGUF installation. |

Logs: `<output folder>/comfy_server.log` (when ComfyClaude started ComfyUI).

## Limitations

- Instruction-style editing ("make the sky red") isn't included: there's no public Z-Image edit model yet,
  and Qwen-Image-Edit / Flux Kontext are heavy for 6–8 GB GPUs. Use `inpaint` or `generate_from_reference`.
- Pose transfer is only as good as the extracted skeleton (limbs hidden by clothing can be missed).
- References: one image as the starting point plus one for structure/pose. Blending several images'
  subjects or styles needs a multi-reference model (e.g. a future Z-Image edit/omni release); for now
  Claude describes extra references in the prompt.
- NVIDIA GPUs are the tested path. AMD/Apple may work through ComfyUI but are untested.

## Models & licenses

ComfyClaude's code is [MIT](LICENSE). Models are downloaded from their authors and keep their own licenses:

| Model | License |
|---|---|
| [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (+ [GGUF](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF)) | Apache-2.0 |
| [SD 1.5 Inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) (optional content-aware fill) | CreativeML OpenRAIL-M |
| [Qwen3-4B](https://huggingface.co/unsloth/Qwen3-4B-GGUF) text encoder | Apache-2.0 |
| [Z-Image Turbo Fun ControlNet Union 2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1) | Apache-2.0 |
| [SDPose](https://huggingface.co/Comfy-Org/SDPose), [BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet) | MIT |
| 4x-UltraSharp upscaler | CC BY-NC-SA 4.0 (**non-commercial**) — swap via `models.upscaler` for commercial use |

Thanks to the ComfyUI, ComfyUI-GGUF, Tongyi-MAI and Alibaba PAI teams.
