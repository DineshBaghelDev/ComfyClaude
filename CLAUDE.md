# ComfyClaude

MCP server (`server.py`) that gives Claude image generation through local ComfyUI + Z-Image Turbo.
Hardware: RTX 4050 6 GB VRAM, 16 GB RAM — models are partially offloaded; generations take ~30–150 s.

## Using the image tools

Call `list_capabilities` first in a session: it shows which tools are ready, default quality per tool,
and run times measured on this PC.

### Pick the tool
| Request | Tool |
|---|---|
| New image | `generate_image` |
| "Show me some options" / unclear direction | `generate_drafts` → pick best seed → `generate_image` with that seed |
| Variation / restyle of a given image | `generate_from_reference` (strength 0.3 subtle … 0.8 loose) |
| Keep exact outlines / layout | `generate_with_control` mode=canny |
| Copy a person's pose | `generate_with_control` mode=pose |
| Change one part, fix a hand/face, swap object | `inpaint` mask_mode=region (fractions) |
| New background behind subject | `inpaint` mask_mode=background |
| Wider/taller image | `outpaint` |
| Bigger/sharper | `upscale` (refine=True for added detail, slow) |
| Transparent cut-out | `remove_background` |

### Pick the quality
- `fast` (Q8, 6 steps): drafts, throwaway, quick tests.
- `lowmem` (Q5): only when RAM is tight — it samples slower on this GPU.
- `balanced` (Q8, default): normal use — practically identical to max.
- `max` (bf16): only when the user asks for the best possible quality.

### Prompting Z-Image Turbo
- Detailed natural-language sentences: subject, setting, lighting, camera/style, mood.
- Text to render goes in quotes: sign reading "COMFY RAMEN". Keep it short.
- No negative prompts (cfg is 1). Use `style=` presets instead of repeating style boilerplate.
- ~1 megapixel sizes: 1024x1024, 1216x832 (landscape), 832x1216 (portrait).

### Always check the result
After every generation, open the image with Read and compare it to the request: subject, count of
objects, text spelling, hands/faces, composition. If something is wrong, fix it — rephrase the prompt,
change seed, or `inpaint` just the broken area — before presenting. Tell the user the seed and file path.

### Long jobs
Tools wait ~50 s; if a job is still running they return a job id → call `job_status(id)` until done.
If ComfyUI is not running the tools start it automatically (takes ~45 s).

## Files
- `config.json` quality tiers + sampler settings · `capabilities.json` tool/model manifest · `presets.json` styles
- `workflows.py` graph builders · `imaging.py` Pillow helpers · `comfy_client.py` HTTP client
- `workflows_custom/*.json` user workflows (API format, `{{placeholders}}`) for `run_custom_workflow`
- `setup_models.py` downloads models · `start_comfy.ps1` manual headless ComfyUI start
- Outputs: `outputs/` (each image has a `.json` with prompt/seed/quality/time)
