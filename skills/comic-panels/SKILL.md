---
name: comic-panels
description: Generate comic character portraits, dialogue close-ups and establishing scenes with ComfyClaude model profiles or custom workflows. Use for comic panel creation and continuity, not unrelated image requests.
---

# Comic panels with ComfyClaude

Call `list_capabilities` to check the selected generation model, LoRAs and custom workflows.
Use `generate_with_model` for named SDXL profiles such as `noobai_vpred`; check installed
weights before asking about a download. NoobAI is optional, not the plugin default.
Its V-pred 1.0 model card prohibits commercial use of generated products; review the chosen
version's terms for a commercial comic: https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/main/README.md.

Read [panel templates](references/templates.json) for character_portrait, dialogue_closeup and
establishing_scene. Substitute the approved character, expression and scene into prompt_template,
then call generate_with_model using its prompt, negative_prompt, model, dimensions, steps and seed.
For another compatible SDXL profile, select that model and follow its model card settings.
Keep the artistic direction and character appearance blocks fixed across related panels.
For custom workflow use, the plugin bundles `noobai_character_portrait`, `noobai_dialogue_closeup`
and `noobai_establishing_scene`. Call run_custom_workflow with character, expression, scene and seed.
These examples use the published NoobAI filename; a user workflow in ~/.comfyclaude/workflows
can override a bundled name to change checkpoint/settings. Prefer generate_with_model when using
configured model profiles or LoRAs. No separate comic plugin or MCP server is needed.

Use portrait framing for readable faces. Avoid vague comic-panel tags that produced black borders
and tiny subjects in local tests. Add panel layout, speech bubbles and lettering after generation.
The portrait test yielded an unrequested forehead jewel, so the character templates exclude it.
The environment test missed a fountain: inspect scene adherence rather than accepting execution
success as visual approval. Read [template design](references/template-design.md) when refining.

Use compatible character/style LoRAs only when installed and approved; don't invent filenames.
If no approved character sheet exists, make one before producing a sequence. Matching seeds and
img2img do not guarantee identity. Keep character sheets, prompts, LoRA settings and the panel
manifest in the comic project. Inspect faces, hands, clothing, continuity, lighting and composition.
Save paths, seeds, checkpoint and settings in the manifest. Use recolor_image for colour-only edits,
dedicated inpaint for geometry changes; NoobAI is not the SD1.5 inpainting checkpoint.
Pass an absolute save_to inside the comic project when its location is known.
