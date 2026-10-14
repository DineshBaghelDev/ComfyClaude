# Contributing

## Layout
| Path | What |
|---|---|
| `server.py` | MCP server: tool definitions, job handling, ComfyUI auto-start |
| `workflows.py` | Builds ComfyUI API-format graphs for each tool (one graph per tool, loaders swapped per tier) |
| `imaging.py` | Pillow helpers: resizing to model-friendly sizes, masks, contact sheets |
| `recoloring.py` | CPU colour transforms: masked Oklab hue/chroma changes with lightness and alpha preservation |
| `comfy_client.py` | Stdlib-only ComfyUI HTTP client (`/prompt`, `/history`, `/view`, `/upload/image`, `/object_info`, `/free`) |
| `settings.py` | Config loading: `config/defaults.json` ← `~/.comfyclaude/config.json` ← env vars |
| `config/capabilities.json` | Tool manifest (default tier, required models, which loader node lists each model) |
| `config/presets.json` | Style presets (users can add their own in `~/.comfyclaude/presets.json`) |
| `setup_models.py` | Installs the GGUF node and downloads models |
| `skills/` | Plugin skills: usage guide for Claude, setup/troubleshooting guide |
| `.claude-plugin/` | Plugin manifest and marketplace entry |
| `tests/` | CPU regression tests and real MCP smoke tests |
| `CHANGELOG.md` | User-facing changes for each release |

## Develop
```
uv run python -c "import server, asyncio; print([t.name for t in asyncio.run(server.mcp.list_tools())])"
claude plugin validate . --strict
```
Run the CPU regression tests without ComfyUI:

```text
uv run --locked python -m unittest discover -s tests -p "test_*.py" -v
```

Verify the actual MCP interface before shipping:

```text
uv run --locked python tests/live_edit_check.py --source reference.png --edit-model sd15_inpaint
uv run --locked python tests/live_recolor_check.py --source reference.png --mask selection.png --color "#3267C8"
uv run --locked python tests/live_checkpoint_check.py
```

The diffusion check needs local ComfyUI and its checkpoint. Recolouring requires no models.
Both use MCP stdio calls, test output dimensions and protected pixels, and save comparisons for
visual inspection. Check semantic edits and mask boundaries yourself: numerical checks alone
do not establish that an object was removed, a colour changed correctly, or a scene continued.
Direct Python calls are useful during development but do not replace the MCP release checks.

## Adding a tool
1. For diffusion tools, add a graph builder in `workflows.py` and verify node inputs against
   ComfyUI's `/object_info`. For deterministic local tools, keep the transform independent of ComfyUI.
2. Add the `@mcp.tool()` in `server.py`; write the docstring for Claude (when to use it, parameter ranges).
3. Register it in `config/capabilities.json` (default quality, required models) and, if it needs a
   model, add the file to `setup_models.py` and the loader node to `model_nodes`.
4. Mention it in `skills/image-generation/SKILL.md` and the README table.
5. Test protected pixels, input validation, transparency where applicable, and the real MCP call.

## Supporting another model family
Tiers in `config/defaults.json` pick loader nodes and files. A different model (Flux, Qwen-Image, SDXL…)
needs its own graph builders because samplers, text encoders and latent formats differ. Benchmarks and
results on other GPUs are very welcome in issues/PRs.

For compatible SD-style inpainting checkpoints, add entries under `editing.models` and select
them with `edit_model`. The native checkpoint backend is tested with SD 1.5 Inpainting.
For other architectures, start with an exported API workflow through `run_custom_workflow`.
Colour-only edits should use `recolor_image`; regenerating a garment to change its colour can
introduce seams, new pockets, distorted linework or unwanted style changes.

## Release checklist

1. Update plugin/marketplace descriptions, README, skills and `CHANGELOG.md` for the final behaviour.
2. Align `.claude-plugin/plugin.json`, `pyproject.toml` and `uv.lock` versions.
3. Run locked CPU tests, strict plugin validation and the relevant MCP checks; inspect the images.
4. Commit and push the release commit. Create and push an annotated version tag at that commit.
5. Publish GitHub release notes describing changes, setup requirements and tested limitations.
   Historical releases must describe their own tag's contents, rather than later features.
