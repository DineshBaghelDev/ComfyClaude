# Changelog

Changes are grouped by version. Historical entries describe the code at their original tags.

## [v1.3.0] - 2026-10-15

- Added optional SDXL checkpoint profiles and `generate_with_model` with negative prompts,
  reference img2img, compatible LoRA chains and model settings recorded in output metadata.
- Added NoobAI XL V-pred 1.0 settings and `--tiers noobai`, downloading only its checkpoint.
  Existing Z-Image defaults and dedicated editing tools are unchanged.
- Bundled a comic-panels skill, three reusable panel presets and three NoobAI custom workflows; no separate comic plugin required.
- Setup guidance now checks installed models and asks before missing-model downloads,
  with extras opt-in. Documented separate personal comic extensions and NoobAI's model license.


## [v1.2.0] - 2026-10-15

### Content-aware editing

- Added a dedicated SD 1.5 Inpainting backend for `inpaint` and `outpaint`, using the masked
  source image and mask as model conditioning.
- Added `edit_model` selection: `auto` prefers an installed editing checkpoint,
  `sd15_inpaint` requires that checkpoint, and `z_image` explicitly selects compatibility mode.
- Added configurable editing checkpoints and sampling settings under `editing.models`.
- Added `setup_models.py --tiers inpaint` for the editing checkpoint alone (~4.3 GB), or
  `--inpaint` alongside the normal Z-Image setup. The checkpoint uses CreativeML OpenRAIL-M.
- Added a softened inpaint paste boundary and extra sampling context; outpaint now composites
  the prepared source back outside its feathered overlap to avoid VAE reconstruction drift.
- Added configurable `feather` parameters, validation and edit settings in saved metadata.

### Precise colour changes

- Added `recolor_image`: CPU masked hue/chroma changes that preserve perceived lightness,
  source resolution, transparency and spatial detail, with no ComfyUI or model requirement.
- Added mask/region intersection, colour strength, saturation, inward-only feathering, mask
  output and JSON metadata. Pixels outside the selected support remain exactly unchanged.
- Updated the plugin skills to route colour-only edits to recolouring instead of diffusion,
  which can invent folds, pockets, seams or unwanted changes to linework.

### Documentation and verification

- Expanded model-selection, setup, mask-review and release guidance in README, contributor docs
  and plugin skills. Plugin, package and lockfile versions are aligned.
- Added 19 CPU regression tests and real MCP stdio checks for diffusion edits and recolouring.
- Local MCP checks covered object removal, scene extension, anime backgrounds, a small facial
  repair and an orange-to-blue shirt recolour at its original 1280x784 resolution.
- Protected-pixel, output-size, transparency, PNG copy and input-validation checks passed.

### Notes when upgrading

- Reload the plugin's MCP server so it advertises the new tool and editing parameters.
- Install the optional inpainting checkpoint to use dedicated content-aware fill. Without it,
  `auto` reports a Z-Image fallback; an explicitly requested missing checkpoint raises an error.
- Precise masks remain essential. Colour-based selection can catch skin outlines or background
  details; broad diffusion masks can introduce artifacts in fine anime facial features.
- Pixel preservation for diffusion tools applies to their resized working source. Recolouring
  keeps the original displayed dimensions. Exact preservation requires PNG output.
- The local ComfyUI-GGUF runtime had Windows access violations during weight unloading; test
  launches used disabled dynamic VRAM/asynchronous offloading. This is an installation-specific
  runtime workaround, not a guarantee for other systems.

## [v1.1.0] - 2026-10-14

- Added small image previews to MCP tool responses, including checkerboards for transparency.
  Preview size/count are configurable, and auxiliary ControlNet maps are distinguished.
- Added `save_to` to save outputs directly into a project file or folder, with PNG/JPG/WebP support.
- Added an optional second reference to `generate_from_reference`: one image supplies the
  appearance/layout while another supplies canny or pose structure. Control images are fitted
  to the working source dimensions.
- Added ComfyUI health rechecks and reconnection to a responding configured/Desktop/default port
  after a server restart, with model-list cache refresh when the connection changes.
- Expanded usage skills and documentation for previews, project outputs and multiple references.

## [v1.0.0] - 2026-10-14

- Packaged the local ComfyUI MCP server as a Claude Code plugin with a marketplace entry,
  user settings, generation guidance and setup/troubleshooting skills.
- Added Z-Image Turbo generation, draft contact sheets, image-to-image references, canny/pose
  ControlNet, region/file/subject/background inpaint, outpaint, upscale/refine and background removal.
- Added `balanced`, `fast`, `lowmem` and `max` quality tiers, style presets, seed control,
  working-size preparation and low-memory model choices.
- Added model/node setup, ComfyUI discovery and background auto-start, capability/status tools,
  persistent job metadata and image download to local outputs.
- Added user config overrides and custom exported ComfyUI API workflows with typed placeholders.
- Added installation, hardware, benchmark, model-license and contributor documentation.

[v1.2.0]: https://github.com/DineshBaghelDev/ComfyClaude/compare/v1.1.0...v1.2.0
[v1.1.0]: https://github.com/DineshBaghelDev/ComfyClaude/compare/v1.0.0...v1.1.0
[v1.0.0]: https://github.com/DineshBaghelDev/ComfyClaude/tree/v1.0.0
