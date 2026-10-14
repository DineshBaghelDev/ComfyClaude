---
name: image-generation
description: How to generate, edit, and refine images with the ComfyClaude tools (local ComfyUI + Z-Image Turbo). Use whenever the user asks to create, draw, render, design, edit, restyle, extend, upscale, or cut out an image, or wants a logo, poster, illustration, photo, mockup, sticker or similar visual.
---

# Image generation with ComfyClaude

The `mcp__comfyclaude__*` tools run image models on the user's own GPU through ComfyUI.
Jobs take from ~10 seconds to a few minutes depending on their hardware.

## Start of a session
Call `list_capabilities` once. It shows which tools are ready, which quality tiers are installed,
and run times measured on this machine. If ComfyUI or models are missing, use the `setup` skill.

## Pick the tool
| Request | Tool |
|---|---|
| New image | `generate_image` |
| Unclear direction / "show me options" | `generate_drafts` → pick the best seed → `generate_image` with that seed |
| Variation or restyle of a given image | `generate_from_reference` (strength 0.3 subtle … 0.8 loose) |
| Keep exact outlines / layout of a reference | `generate_with_control` mode=canny |
| Copy a person's pose | `generate_with_control` mode=pose (check the saved `*_map` image) |
| Change one part, fix a hand/face, swap an object | `inpaint` mask_mode=region, region=[x0,y0,x1,y1] as fractions |
| New background behind the subject | `inpaint` mask_mode=background |
| Wider / taller image | `outpaint` |
| Bigger / sharper | `upscale` (refine=True adds real detail but is slow) |
| Transparent cut-out (stickers, products) | `remove_background` |
| User's own ComfyUI workflow | `run_custom_workflow` |

## Pick the quality
- `balanced` (default): normal use.
- `fast`: drafts, throwaway, quick tests.
- `max`: only when the user asks for the best possible quality.
- `lowmem`: only when RAM is tight.
Switching tiers reloads models (slow) — don't alternate needlessly.

## Prompting Z-Image Turbo
- Write detailed natural-language sentences: subject, setting, lighting, camera or art style, mood.
- Text to render goes in quotes and should be short: a sign reading "OPEN 24H".
- There are no negative prompts (cfg is 1); describe what you want instead of what to avoid.
- Use `style=` presets (photo, cinematic, portrait, product, anime, illustration, watercolor, 3d,
  poster, logo, pixel, sketch) instead of repeating style boilerplate.
- Stay near 1 megapixel: 1024x1024, 1216x832 (landscape), 832x1216 (portrait). Upscale afterwards.
- For a reference image, look at it first (Read) and describe in the prompt what must be kept.

## Always check your work
After every generation, open the image with Read and compare it to the request: subject, number of
objects, spelling of text, hands and faces, composition, style. If something is wrong, fix it before
presenting — new seed, clearer prompt, or `inpaint` only the broken area. Then give the user the file
path and seed (so they can reproduce or tweak it).

## Long jobs
Tools wait ~50 s. If a job is still running they return a job id: call `job_status(id)` until done.
If ComfyUI isn't running and a ComfyUI folder is configured, the tools start it (~30–60 s).
