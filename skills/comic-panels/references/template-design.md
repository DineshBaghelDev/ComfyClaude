# Template design and evaluation

References inspected:
- https://civitai.com/models/833294/noobai-xl-nai-xl (V-pred 1.0 release, version 1190596)
- https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/main/README.md
- https://huggingface.co/Laxhar/noobai-XL-Vpred-0.5/blob/main/comfy_ui_workflow_sample.png

The Civitai release recommends CFG rescaling around 0.2 and avoids Karras schedules for V-pred.
The older official PNG embeds a workflow with Euler/normal, CFG 5, 30 steps, V-pred and zsnr.
The downloaded checkpoint's SHA256 matches both the Hugging Face and Civitai original file.

Visual references use large expressive faces, deliberately composed subject placement, detailed
linework, layered highlights/shadows and strongly controlled colors. These features inform the
portrait and dialogue templates. Reference images are not bundled, and source prompts have not
been copied wholesale. Avoid mixing quality illustration goals with vague comic-panel tags that
can produce flat panels, black borders or miniature subjects. Add panel layout and lettering later.

Assess actual results: face visibility, eye alignment, coherent anatomy, subject scale, lighting,
clothing continuity and usable lettering space. A successful MCP run only verifies execution;
it is not visual approval. Record accepted/rejected outputs with reasons in the panel manifest.
