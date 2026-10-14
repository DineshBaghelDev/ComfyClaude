# Contributing

## Layout
| Path | What |
|---|---|
| `server.py` | MCP server: tool definitions, job handling, ComfyUI auto-start |
| `workflows.py` | Builds ComfyUI API-format graphs for each tool (one graph per tool, loaders swapped per tier) |
| `imaging.py` | Pillow helpers: resizing to model-friendly sizes, masks, contact sheets |
| `comfy_client.py` | Stdlib-only ComfyUI HTTP client (`/prompt`, `/history`, `/view`, `/upload/image`, `/object_info`, `/free`) |
| `settings.py` | Config loading: `config/defaults.json` ← `~/.comfyclaude/config.json` ← env vars |
| `config/capabilities.json` | Tool manifest (default tier, required models, which loader node lists each model) |
| `config/presets.json` | Style presets (users can add their own in `~/.comfyclaude/presets.json`) |
| `setup_models.py` | Installs the GGUF node and downloads models |
| `skills/` | Plugin skills: usage guide for Claude, setup/troubleshooting guide |
| `.claude-plugin/` | Plugin manifest and marketplace entry |

## Develop
```
uv run python -c "import server, asyncio; print([t.name for t in asyncio.run(server.mcp.list_tools())])"
claude plugin validate . --strict
```
Tools can be called directly from Python for testing (`server.generate_image(...)`), with ComfyUI running.

## Adding a tool
1. Add a graph builder in `workflows.py`. Check node names/inputs against your ComfyUI's `/object_info`.
2. Add the `@mcp.tool()` in `server.py`; write the docstring for Claude (when to use it, parameter ranges).
3. Register it in `config/capabilities.json` (default quality, required models) and, if it needs a
   model, add the file to `setup_models.py` and the loader node to `model_nodes`.
4. Mention it in `skills/image-generation/SKILL.md` and the README table.

## Supporting another model family
Tiers in `config/defaults.json` pick loader nodes and files. A different model (Flux, Qwen-Image, SDXL…)
needs its own graph builders because samplers, text encoders and latent formats differ. Benchmarks and
results on other GPUs are very welcome in issues/PRs.
