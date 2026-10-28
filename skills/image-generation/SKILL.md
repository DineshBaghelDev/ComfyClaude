---
name: image-generation
description: Generate, fill, extend, recolour and refine images with local ComfyClaude tools (ComfyUI diffusion and CPU colour edits). Use whenever the user asks to create, draw, render, design, edit, restyle, extend, recolour, upscale, or cut out an image, or wants a logo, poster, illustration, photo, mockup, sticker or similar visual.
---

# Image generation with ComfyClaude

The `mcp__comfyclaude__*` tools run diffusion models on the user's own GPU through ComfyUI.
`recolor_image` runs on CPU without ComfyUI or models and retains the original displayed size.
Jobs take from ~10 seconds to a few minutes depending on their hardware.

## Start of a session
Call `list_capabilities` once. It shows which tools are ready, which quality tiers are installed,
and run times measured on this machine. If ComfyUI or models are missing, use the `setup` skill.

## Pick the tool
| Request | Tool |
|---|---|
| New image | `generate_image` |
| NoobAI / named SDXL checkpoint / character LoRA | `generate_with_model` |
| Unclear direction / "show me options" | `generate_drafts` → pick the best seed → `generate_image` with that seed |
| Variation or restyle of a given image | `generate_from_reference` (strength 0.3 subtle … 0.8 loose) |
| Look of image A + pose/outlines of image B | `generate_from_reference` with `control_image`=B, `control_mode`=pose/canny, strength ≥ 0.7 |
| Keep exact outlines / layout of a reference | `generate_with_control` mode=canny |
| Copy a person's pose | `generate_with_control` mode=pose (check the saved `*_map` image) |
| Change one part, fix a hand/face, swap an object | `inpaint` mask_mode=region, region=[x0,y0,x1,y1] as fractions |
| New background behind the subject | `inpaint` mask_mode=background |
| Wider / taller image | `outpaint` |
| Change colour only (shirt, hair, prop), keeping folds and linework | `recolor_image` with a precise mask and target colour |
| Bigger / sharper | `upscale` (refine=True adds real detail but is slow) |
| Transparent cut-out (stickers, products) | `remove_background` |
| User's own ComfyUI workflow | `run_custom_workflow` |

## Pick the quality
- `balanced` (default): normal use.
- `fast`: drafts, throwaway, quick tests.
- `max`: only when the user asks for the best possible quality.
- `lowmem`: only when RAM is tight.
Switching tiers reloads models (slow) — don't alternate needlessly.

## Content-aware editing
Use `recolor_image` for colour-only requests before considering diffusion. A precise file mask
selects the garment/prop; a rectangular region also changes skin/background inside it. Inspect
the mask first. It keeps the original resolution, perceived lightness and alpha; outside the
selection pixels remain identical. It cannot replace objects, redraw shapes or add new details.
Colour-based mask selection can accidentally include skin outlines, fingers or background details.
Exclude those explicitly and inspect the final image. Keep a colour-only request on the recolour
path instead of repeatedly increasing diffusion strength or prompt guidance.

For inpaint/outpaint, prefer a dedicated editing model listed as ready in `list_capabilities`.
`edit_model="auto"` selects an installed mask-conditioned checkpoint; use `edit_model="sd15_inpaint"`
to require the supplied content-aware fill model. It uses its own checkpoint, encoder, VAE,
steps and CFG; Z-Image `quality` tiers do not apply. If it is missing, install with
`setup_models.py --inpaint`. `edit_model="z_image"` explicitly selects the compatibility workflow.
Start around 512 pixels / `megapixels=0.26` for SD 1.5; upscale afterwards if needed.
Leave room for the entire object inside an inpaint mask and inspect joins after outpainting.
Use `feather` to soften the paste boundary; feathering alone does not supply image conditioning.
For content-aware removal, describe the replacement texture concisely (e.g. "blue lake water,
small gentle ripples"). Prefer a short fill prompt over a long full-scene description for SD 1.5.

## Prompting Z-Image Turbo
- Write detailed natural-language sentences: subject, setting, lighting, camera or art style, mood.
- Text to render goes in quotes and should be short: a sign reading "OPEN 24H".
- There are no negative prompts (cfg is 1); describe what you want instead of what to avoid.
- Use `style=` presets (photo, cinematic, portrait, product, anime, illustration, watercolor, 3d,
  poster, logo, pixel, sketch) instead of repeating style boilerplate.
- Stay near 1 megapixel: 1024x1024, 1216x832 (landscape), 832x1216 (portrait). Upscale afterwards.
- For a reference image, look at it first (Read) and describe in the prompt what must be kept.

## Multiple references
Z-Image takes one starting image. To combine references:
- Two images: `generate_from_reference` (look/colors) + `control_image` (structure or pose).
- More, or abstract qualities (a character's outfit, a style): look at each image with Read, then describe
  precisely what to take from each in the prompt.

## Where results go
Images are saved to the ComfyClaude output folder. When the user is working on a project and wants the
image there, pass `save_to` with an absolute path — a file (`<project>/assets/hero.png`, `.jpg`/`.webp`
converts) or a folder. Don't copy files around manually.

## Always check your work
Every result comes back with a preview image. Compare it to the request: subject, number of objects,
spelling of text, hands and faces, composition, style (open the full file with Read when fine detail
matters, e.g. small text). If something is wrong, fix it before presenting — new seed, clearer prompt,
or `inpaint` only the broken area. Then give the user the file path and seed.

## Studio (web UI) and LoRAs
If the user wants to click rather than chat, or asks for the UI/studio, call `open_ui` and give them
the URL. Z-Image tools (`generate_image`, `generate_drafts`, `generate_from_reference`,
`generate_with_control`) accept `loras=[{"name": ..., "strength_model": 0.55}]` for Z-Image-compatible
style LoRAs; list installed files with `list_capabilities`.

## Long jobs
Tools wait ~50 s. If a job is still running they return a job id: call `job_status(id)` until done.
If ComfyUI isn't running and a ComfyUI folder is configured, the tools start it (~30–60 s).

## NoobAI / SDXL comic generation

Use `generate_with_model` for named SDXL profiles, negative prompts, img2img and compatible LoRAs.
`noobai_vpred` uses V-prediction, zero-terminal-SNR, Euler, CFG 5, 30 steps and ~1 MP.
Use tag-style prompts and preserve approved character descriptions/LoRAs across panels.
Project-specific comic conventions belong in a separate project skill/extension.
Check the selected checkpoint is installed; ask before downloading missing weights.
NoobAI V-pred 1.0 prohibits commercial use of generated products per its author model card.
See [model setup and profiles](../../docs/noobai.md).
