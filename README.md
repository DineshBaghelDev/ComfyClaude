# ComfyClaude

MCP server that gives Claude image generation through local ComfyUI + Z-Image Turbo.

## Setup (once)
1. Close ComfyUI, then: `python setup_models.py`
   Installs the ComfyUI-GGUF node and downloads Q8 + Q5 models (~19 GB).
2. Start ComfyUI Desktop.
3. Open Claude Code in this folder and approve the `comfyclaude` MCP server (from `.mcp.json`).

## Quality tiers (`config.json`)
| tier | model | use |
|---|---|---|
| max | bf16 (original) | best quality, slow |
| balanced (default) | GGUF Q8 | near-identical, much faster |
| fast | GGUF Q5 | drafts / throwaway |

If GGUF files are missing, tools fall back to `max` automatically.

## Tools
generate_image, generate_from_reference, job_status, list_qualities, comfy_status, free_memory

Outputs go to `outputs/`, each image with a `.json` holding prompt, seed, quality, and timing.
