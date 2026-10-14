"""ComfyClaude MCP server: local Z-Image/SDXL generation and image editing."""
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image as MCPImage
from PIL import Image

import imaging
import recoloring
import settings
import workflows
from comfy_client import ComfyClient

ROOT = Path(__file__).parent
CFG, CAPS, PRESETS = settings.load()
OUT = Path(CFG["output_dir"])
WORK = OUT / "_work"
JOBS_FILE = OUT / "jobs.json"
CUSTOM_DIRS = [ROOT / "workflows_custom", settings.USER_DIR / "workflows"]
imaging.clean_old(WORK)

INSTRUCTIONS = """Local image generation through the user's own ComfyUI with Z-Image Turbo and dedicated inpainting models (jobs take ~10 s to a few minutes
depending on their GPU).
- Call list_capabilities once per session to see ready tools, default qualities and measured times on this PC.
- Quality: balanced (default) for normal use, fast for drafts/throwaway, max only when the user asks for the
  best quality. Avoid switching tiers back and forth (each switch reloads models).
- Use generate_drafts when the direction is unclear, then generate_image with the chosen seed.
- Colour-only edits (clothing, hair, props): use recolor_image with a precise mask. It preserves
  source lightness, linework and geometry at original resolution and needs no diffusion model.
- Inpaint/outpaint: edit_model=auto prefers an installed dedicated content-aware model. Require
  sd15_inpaint for masked-image conditioning; use z_image only for compatibility. Z-Image quality
  tiers do not apply to a dedicated editing checkpoint. SD 1.5 works well around 512 pixels.
- Z-Image generation: detailed natural language; text to render goes in quotes; ~1 MP sizes.
- SDXL/NoobAI: use generate_with_model with a named generation_models profile, tag-style prompts,
  negative prompts and optional compatible LoRAs. Check profile readiness; never silently change
  checkpoints. Comic-specific characters, prompts and panel conventions belong in a project extension.
- Check existing models before setup. Explain required downloads and sizes and obtain the user's
  agreement before downloading missing weights; install only models needed for the requested tools.
- Content-aware fill: describe the replacement content concisely, e.g. "blue lake water, small
  ripples" to remove a boat. Describe what should appear in the mask rather than just "remove".
- Results come back with a preview image: ALWAYS check it against the request (subject, counts, text
  spelling, hands/faces); open the full file with Read for fine detail. Fix problems (new seed, better
  prompt, or inpaint the broken area) before presenting.
- To put an image into the user's project, pass save_to (absolute file or folder path).
- If a tool returns a job id, call job_status until done. Report file paths and seeds to the user.
- If ComfyUI or models are missing, tell the user to run the setup (see the comfyclaude setup skill)."""

mcp = FastMCP("comfyclaude", instructions=INSTRUCTIONS)
# Also probe the usual ports, so an already-running ComfyUI (Desktop uses 8000) is found
# instead of starting a second instance that would run out of GPU memory.
comfy = ComfyClient(list(dict.fromkeys([CFG["comfy_url"], "http://127.0.0.1:8188", "http://127.0.0.1:8000"])))


# ---- ComfyUI process --------------------------------------------------------
def _start_comfy(log):
    """Launch ComfyUI headless. Detached so it survives MCP server restarts."""
    path = Path(CFG["comfy_path"])
    port = CFG["comfy_url"].rsplit(":", 1)[-1].split("/")[0]
    cmd = [settings.find_comfy_python(path, CFG.get("comfy_python", "")), "main.py",
           "--port", port if port.isdigit() else "8188", *CFG.get("comfy_args", [])]
    # stdin/stdout must not inherit the MCP stdio channel.
    kw = dict(cwd=path, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        for extra in (subprocess.CREATE_BREAKAWAY_FROM_JOB, 0):  # breakaway may be refused
            try:
                return subprocess.Popen(cmd, creationflags=flags | extra, **kw)
            except OSError:
                continue
    return subprocess.Popen(cmd, start_new_session=True, **kw)


def ensure_comfy():
    """Connect to ComfyUI, starting it headless if it isn't running and comfy_path is set."""
    try:
        return comfy.connect()
    except ConnectionError:
        if not CFG["comfy_path"]:
            raise ConnectionError(
                f"ComfyUI is not running at {CFG['comfy_url']}. Start ComfyUI, or set comfy_path "
                f"(plugin setting or {settings.user_config_path()}) so it can be started automatically.")
        if not (Path(CFG["comfy_path"]) / "main.py").exists():
            raise ConnectionError(f"comfy_path '{CFG['comfy_path']}' has no main.py (point it at the ComfyUI folder).")
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "comfy_server.log"
    _start_comfy(open(log_path, "ab"))
    end = time.time() + CFG["start_timeout"]
    while time.time() < end:
        time.sleep(3)
        try:
            return comfy.connect()
        except ConnectionError:
            continue
    raise ConnectionError(f"Started ComfyUI but it did not come up in {CFG['start_timeout']}s. See {log_path}")


# ---- models / quality --------------------------------------------------------
def _has(node, field, name):
    opts = comfy.options(node, field)
    return opts is not None and name in opts


def _quality_problem(q):
    """None if the tier can run, else a short reason."""
    if q["loader"] == "gguf":
        if comfy.options("UnetLoaderGGUF", "unet_name") is None:
            return "ComfyUI-GGUF custom node not installed"
        unet, clip = ("UnetLoaderGGUF", "unet_name"), ("CLIPLoaderGGUF", "clip_name")
    else:
        unet, clip = ("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name")
    missing = [f for (n, k), f in ((unet, q["unet"]), (clip, q["clip"])) if not _has(n, k, f)]
    if not _has("VAELoader", "vae_name", CFG["vae"]):
        missing.append(CFG["vae"])
    return f"missing model files: {missing}" if missing else None


def _need(*keys):
    ensure_comfy()
    missing = [CFG["models"][k] for k in keys if not _has(*CAPS["model_nodes"][k], CFG["models"][k])]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}. Run the setup: python setup_models.py --extras")


def _pick_quality(name, tool):
    """Resolve a quality tier; falls back to another installed tier if files are missing."""
    ensure_comfy()
    name = name or CAPS["tools"].get(tool, {}).get("quality") or CFG["default_quality"]
    if name not in CFG["qualities"]:
        raise ValueError(f"Unknown quality '{name}'. Options: {list(CFG['qualities'])}")
    if not _quality_problem(CFG["qualities"][name]):
        return name, CFG["qualities"][name], ""
    for alt in CFG["qualities"]:
        if alt != name and not _quality_problem(CFG["qualities"][alt]):
            return alt, CFG["qualities"][alt], f"(note: quality '{name}' unavailable, used '{alt}')\n"
    raise FileNotFoundError(f"No usable Z-Image models: {_quality_problem(CFG['qualities'][name])}. "
                            "Run the setup: python setup_models.py")


def _seed(seed):
    return seed if seed >= 0 else random.randint(0, 2**31 - 1)


def _pick_edit_model(name, quality, tool):
    """Prefer installed mask-conditioned checkpoints; allow explicit Z-Image compatibility."""
    ensure_comfy()
    editing = CFG.get("editing", {})
    models = editing.get("models", {})
    name = name or editing.get("default_model", "auto")
    if name not in ("auto", "z_image") and name not in models:
        raise ValueError(f"Unknown edit_model '{name}'. Options: {['auto', 'z_image', *models]}")
    if name != "z_image":
        comfy.object_info(refresh=True)
        ready = [key for key, model in models.items()
                 if _has("CheckpointLoaderSimple", "ckpt_name", model["checkpoint"])]
        chosen = (ready[0] if ready else None) if name == "auto" else name
        if chosen:
            if chosen not in ready:
                raise FileNotFoundError(f"Missing editing checkpoint: {models[chosen]['checkpoint']}. "
                                        "Install it in ComfyUI models/checkpoints. For sd15_inpaint run setup_models.py --inpaint.")
            model = {"steps": 24, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", **models[chosen]}
            return chosen, None, model, f"Content-aware fill model: {chosen}.\n"
        note = "Dedicated inpainting model unavailable; using Z-Image compatibility mode. Install with setup_models.py --inpaint.\n"
    else:
        note = "Z-Image compatibility mode (no dedicated masked-image conditioning).\n"
    qname, q, quality_note = _pick_quality(quality, tool)
    return qname, q, None, note + quality_note


def _styled(prompt, style):
    if not style:
        return prompt
    if style not in PRESETS:
        raise ValueError(f"Unknown style '{style}'. Options: {list(PRESETS)}")
    return f"{prompt.rstrip('. ')}. {PRESETS[style]}"


def _upload(path, megapixels=None):
    """Upload a local image (optionally resized to ~megapixels). Returns (name, w, h)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Image not found: {path}")
    ensure_comfy()
    if megapixels:
        path, w, h = imaging.prepare(path, WORK, megapixels)
    else:
        w, h = imaging.size_of(path)
    return comfy.upload_image(path), w, h


# ---- jobs --------------------------------------------------------------------
def _load_jobs():
    return json.loads(JOBS_FILE.read_text()) if JOBS_FILE.exists() else {}


def _save_job(pid, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = _load_jobs()
    jobs[pid] = meta
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))


def _submit(wf, meta):
    ensure_comfy()
    pid = comfy.queue_prompt(wf)
    meta.update({"prompt_id": pid, "submitted": time.time()})
    _save_job(pid, meta)
    return pid


def _collect(pid, timeout):
    """Wait for a job. Returns (status_text, paths or None)."""
    hist = comfy.wait(pid, timeout)
    meta = _load_jobs().get(pid, {})
    if hist is None:
        pos = comfy.queue_position(pid) or "running"
        return f"Job {pid} still {pos}. Call job_status('{pid}') to keep waiting.", None
    if hist.get("status", {}).get("status_str") == "error":
        msgs = [m[1] for m in hist["status"].get("messages", []) if m[0] == "execution_error"]
        err = msgs[0].get("exception_message", msgs) if msgs else hist["status"]
        return f"Job {pid} failed: {str(err)[:1500]}", None
    if meta.get("files"):  # already collected earlier
        saved = f"\nSaved to: {', '.join(meta['saved_to'])}" if meta.get("saved_to") else ""
        return f"Done ({meta.get('quality')}).{saved}", meta["files"]
    paths = comfy.download_outputs(hist, OUT)
    meta["seconds"] = round(time.time() - meta.get("submitted", time.time()), 1)
    meta["files"] = paths
    saved = ""
    if meta.get("save_to"):
        try:
            meta["saved_to"] = _copy_to(_main_files(paths), meta["save_to"])
            saved = f"\nSaved to: {', '.join(meta['saved_to'])}"
        except Exception as e:  # never lose the generated image because of a bad target path
            saved = f"\nCould not save to {meta['save_to']}: {e}"
    for p in paths:
        Path(p).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    _save_job(pid, meta)
    return f"Done in {meta['seconds']}s ({meta.get('quality') or meta.get('tool')}).{saved}", paths


def _main_files(paths):
    """Result images without auxiliary outputs (e.g. ControlNet maps)."""
    return [p for p in paths if "_map_" not in Path(p).name] or list(paths)


def _copy_to(paths, dest):
    """Copy results to an absolute file or folder path; .jpg/.webp targets are converted."""
    dest = Path(dest).expanduser()
    if not dest.is_absolute():
        raise ValueError(f"save_to must be an absolute path, got '{dest}'")
    if dest.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        targets = [dest] if len(paths) == 1 else \
            [dest.with_name(f"{dest.stem}_{i + 1}{dest.suffix}") for i in range(len(paths))]
    else:
        targets = [dest / Path(p).name for p in paths]
    for src, t in zip(paths, targets):
        t.parent.mkdir(parents=True, exist_ok=True)
        if t.suffix.lower() == ".png":
            shutil.copyfile(src, t)
        elif t.suffix.lower() in (".jpg", ".jpeg"):
            Image.open(src).convert("RGB").save(t, quality=95)
        else:
            Image.open(src).save(t, quality=95)
    return [str(t) for t in targets]


def _result(text, paths=None, preview_paths=None):
    """Tool result: text plus small previews so Claude can check images without opening files."""
    paths = paths or []
    shown = preview_paths if preview_paths is not None else \
        _main_files(paths) + [p for p in paths if p not in _main_files(paths)]
    content = [text]
    if CFG.get("preview_size", 768):
        for p in shown[:CFG.get("max_previews", 2)]:
            content.append(MCPImage(data=imaging.preview_jpeg(p, CFG.get("preview_size", 768)), format="jpeg"))
    return content


def _run(wf, meta, note="", save_to=""):
    if save_to:
        meta["save_to"] = save_to
    pid = _submit(wf, meta)
    status, paths = _collect(pid, CFG["wait_seconds"])
    return _result(note + status + ("\nImages:\n" + "\n".join(paths) if paths else ""), paths)


# ---- generation tools ----------------------------------------------------------
@mcp.tool(structured_output=False)
def generate_with_model(prompt: str, model: str = "noobai_vpred", negative_prompt: str | None = None,
                        width: int = 832, height: int = 1216, seed: int = -1, steps: int = 0,
                        image_path: str = "", strength: float = 0.6,
                        loras: list[dict] | None = None, save_to: str = "") -> list:
    """Generate with a named SDXL checkpoint profile (including NoobAI), optionally img2img/LoRAs.

    See list_capabilities for model profiles and installed LoRA filenames. No weights are downloaded.
    NoobAI uses tag-style prompts. Model-specific sampling is independent of Z-Image quality tiers.
    negative_prompt=None uses the profile default; an empty string disables it. LoRAs are dictionaries
    with name, strength_model and strength_clip; use LoRAs compatible with the chosen checkpoint.
    image_path supplies an optional composition reference, not guaranteed character identity.
    NoobAI is a generation checkpoint, not the dedicated inpaint/outpaint backend.
    """
    profiles = CFG.get("generation_models", {})
    if model not in profiles:
        raise ValueError(f"Unknown model '{model}'. Options: {list(profiles)}")
    profile = profiles[model]
    if profile.get("architecture") != "sdxl":
        raise ValueError("generate_with_model supports SDXL checkpoint profiles only")
    if profile.get("prediction_type", "eps") not in ("eps", "v_prediction"):
        raise ValueError("prediction_type must be eps or v_prediction")
    if width < 64 or height < 64 or width % 8 or height % 8:
        raise ValueError("width/height must be multiples of 8 and at least 64")
    if not 0 < strength <= 1 or steps < 0:
        raise ValueError("strength must be >0 and <=1; steps must be >=0")
    ensure_comfy()
    info = comfy.object_info(refresh=True)
    required = ["CheckpointLoaderSimple", "ModelSamplingDiscrete", "CLIPTextEncode", "KSampler",
                "VAEDecode", "SaveImage", "VAEEncode" if image_path else "EmptyLatentImage"]
    if profile.get("cfg_rescale", 0):
        required.append("RescaleCFG")
    if loras:
        required.append("LoraLoader")
    missing_nodes = [node for node in required if node not in info]
    if missing_nodes:
        raise RuntimeError(f"Missing ComfyUI nodes: {missing_nodes}. Update ComfyUI.")
    if not _has("CheckpointLoaderSimple", "ckpt_name", profile["checkpoint"]):
        raise FileNotFoundError(f"Missing checkpoint: {profile['checkpoint']}. Install it in ComfyUI models/checkpoints. "
                                "No automatic download or model fallback. For NoobAI V-pred: setup_models.py --tiers noobai. "
                                "Ask before downloading weights; check the model license.")
    for lora in loras or []:
        if not isinstance(lora, dict) or not isinstance(lora.get("name"), str):
            raise ValueError("Each LoRA needs a filename in name")
        if not _has("LoraLoader", "lora_name", lora["name"]):
            raise FileNotFoundError(f"Missing LoRA: {lora['name']}. Install a compatible file in models/loras.")
        for key in ("strength_model", "strength_clip"):
            if not isinstance(lora.get(key, 1.0), (int, float)) or not -10 <= lora.get(key, 1.0) <= 10:
                raise ValueError(f"LoRA {key} must be between -10 and 10")
    full = ", ".join(p for p in (profile.get("prompt_prefix", ""), prompt) if p)
    negative = profile.get("negative_prompt", "") if negative_prompt is None else negative_prompt
    seed = _seed(seed)
    image_name = ""
    if image_path:
        WORK.mkdir(parents=True, exist_ok=True)
        image_name = comfy.upload_image(imaging.prepare_exact(image_path, width, height, WORK))
    wf = workflows.checkpoint_image(profile, full, negative, width, height, seed, steps,
                                    image_name, strength, loras)
    meta = {"tool": "generate_with_model", "model": model, "checkpoint": profile["checkpoint"],
            "profile": profile, "prompt": full, "negative_prompt": negative, "width": width, "height": height,
            "seed": seed, "steps": steps or profile.get("steps", 30), "reference": image_path,
            "strength": strength if image_path else 1.0, "loras": loras or []}
    return _run(wf, meta, f"Model: {model}\n", save_to)


@mcp.tool(structured_output=False)
def generate_image(prompt: str, width: int = 1024, height: int = 1024, quality: str = "",
                   style: str = "", seed: int = -1, steps: int = 0, save_to: str = "") -> list:
    """Generate an image from text with Z-Image Turbo.

    quality: "balanced" (GGUF Q8, default, near-identical to original), "fast" (same model, 6 steps,
    for drafts), "max" (bf16 original, slowest), "lowmem" (Q5, only if RAM is tight). style: optional preset name (see list_capabilities).
    Width/height: multiples of 16, ~1 megapixel is the sweet spot (1024x1024, 1216x832, 832x1216).
    Write detailed natural-language prompts; put any text to render in quotes.
    Returns saved image paths (open them with Read to check the result) or a job id if still running.
    save_to: optional absolute path (a file like C:/proj/assets/hero.png, or a folder) to also save
    the result into, e.g. the user's project. A .jpg/.webp extension converts the format.
    """
    qname, q, note = _pick_quality(quality, "generate_image")
    seed = _seed(seed)
    full = _styled(prompt, style)
    steps = steps or q.get("steps") or CFG["sampler"]["steps"]
    wf = workflows.text_to_image(q, CFG, full, width, height, seed, steps)
    meta = {"tool": "generate_image", "prompt": full, "width": width, "height": height,
            "seed": seed, "quality": qname, "steps": steps}
    return _run(wf, meta, note, save_to)


@mcp.tool(structured_output=False)
def generate_drafts(prompt: str, count: int = 4, width: int = 768, height: int = 768,
                    quality: str = "", style: str = "", seed: int = -1) -> list:
    """Quickly generate several low-res drafts with different seeds (default: fast quality),
    plus one contact sheet to compare them. Pick the best seed, then call generate_image with that
    seed at full size / higher quality. Note: changing size changes the composition somewhat.
    """
    qname, q, note = _pick_quality(quality, "generate_drafts")
    base = _seed(seed)
    full = _styled(prompt, style)
    pids = []
    for i in range(max(1, min(count, 8))):
        s = base + i
        wf = workflows.text_to_image(q, CFG, full, width, height, s, q.get("steps"), prefix="claude/draft")
        pids.append((s, _submit(wf, {"tool": "generate_drafts", "prompt": full, "width": width,
                                     "height": height, "seed": s, "quality": qname})))
    results, pending = [], []
    end = time.time() + CFG["wait_seconds"] * 3
    for s, pid in pids:
        status, paths = _collect(pid, max(1, end - time.time()))
        if paths:
            results.append((s, paths[0]))
        else:
            pending.append(f"seed {s}: {status}")
    out, sheet = note, None
    if results:
        sheet = imaging.contact_sheet([p for _, p in results], OUT / f"drafts_{base}.png",
                                      [f"seed {s}" for s, _ in results])
        out += f"Contact sheet: {sheet}\n" + "\n".join(f"seed {s}: {p}" for s, p in results)
    if pending:
        out += "\nStill running:\n" + "\n".join(pending)
    return _result(out, preview_paths=[sheet] if sheet else [])


@mcp.tool(structured_output=False)
def generate_from_reference(prompt: str, image_path: str, strength: float = 0.6, quality: str = "",
                            style: str = "", seed: int = -1, megapixels: float = 1.0,
                            control_image: str = "", control_mode: str = "canny", control_strength: float = 0,
                            save_to: str = "") -> list:
    """Generate a new image based on a reference image (img2img). Keeps composition and colors.

    strength 0.2-0.4: small changes, stays close to the reference.
    strength 0.5-0.7: restyle/rework while keeping composition and colors.
    strength 0.8-0.95: only loose layout survives.
    Tip: look at the reference first and describe what should be kept in the prompt.
    For exact outlines or a person's pose alone use generate_with_control instead.

    Second reference (optional): control_image takes structure from ANOTHER image while image_path
    supplies colors/look — e.g. the pose of a person from photo B in the palette/style of image A.
    control_mode: canny (outlines) | pose | direct (already a control map). control_strength 0 = auto
    (canny 0.8, pose 1.0). The control image is cropped/resized to the reference's size; use strength
    >= 0.7 so the control can reshape the reference. For more references than two, look at each and
    describe what to take from them in the prompt.
    save_to: optional absolute path (a file like C:/proj/assets/hero.png, or a folder) to also save
    the result into, e.g. the user's project. A .jpg/.webp extension converts the format.
    """
    if control_image:
        _need("controlnet", *(["pose"] if control_mode == "pose" else []))
        if not Path(control_image).exists():
            raise FileNotFoundError(f"Image not found: {control_image}")
    qname, q, note = _pick_quality(quality, "generate_from_reference")
    seed = _seed(seed)
    full = _styled(prompt, style)
    name, w, h = _upload(image_path, megapixels)
    wf = workflows.image_to_image(q, CFG, full, name, strength, seed, q.get("steps"))
    meta = {"tool": "generate_from_reference", "prompt": full, "reference": image_path,
            "strength": strength, "seed": seed, "quality": qname, "width": w, "height": h}
    if control_image:
        control_strength = control_strength or (0.8 if control_mode == "canny" else 1.0)
        cname = comfy.upload_image(imaging.prepare_exact(control_image, w, h, WORK))
        workflows.add_control(wf, CFG, cname, control_mode, w, h, control_strength, "claude/i2i")
        meta.update({"control_image": control_image, "control_mode": control_mode,
                     "control_strength": control_strength})
    return _run(wf, meta, note, save_to)


@mcp.tool(structured_output=False)
def generate_with_control(prompt: str, image_path: str, mode: str = "canny", strength: float = 0,
                          quality: str = "", style: str = "", seed: int = -1, megapixels: float = 1.0, save_to: str = "") -> list:
    """Generate an image that follows the structure of a reference using Z-Image ControlNet Union.

    mode "canny": follow the outlines/edges of the reference (architecture, objects, layouts).
    mode "pose": copy the body pose of people in the reference (extracted with SDPose).
    mode "direct": image_path already IS a control map (canny/depth/pose image).
    strength 0.6-1.0 (lower = more freedom); 0 = auto (canny 0.8, pose/direct 1.0).
    Pose quality depends on the extracted skeleton: check the *_map image (limbs hidden by clothing
    can be missed). The prompt fully decides colors, style and content.
    Also saves the control map (*_map) so you can check what was extracted.
    save_to: optional absolute path (a file like C:/proj/assets/hero.png, or a folder) to also save
    the result into, e.g. the user's project. A .jpg/.webp extension converts the format.
    """
    _need("controlnet", *(["pose"] if mode == "pose" else []))
    strength = strength or (0.8 if mode == "canny" else 1.0)
    qname, q, note = _pick_quality(quality, "generate_with_control")
    seed = _seed(seed)
    full = _styled(prompt, style)
    name, w, h = _upload(image_path, megapixels)
    wf = workflows.controlnet(q, CFG, full, name, mode, w, h, strength, seed, q.get("steps"))
    meta = {"tool": "generate_with_control", "prompt": full, "reference": image_path, "mode": mode,
            "strength": strength, "seed": seed, "quality": qname, "width": w, "height": h}
    return _run(wf, meta, note, save_to)


@mcp.tool(structured_output=False)
def inpaint(image_path: str, prompt: str, mask_mode: str = "region", region: list[float] | None = None,
            mask_path: str = "", ellipse: bool = False, strength: float = 1.0, grow: int = 16,
            quality: str = "", seed: int = -1, megapixels: float = 1.0, save_to: str = "", feather: int = 16,
            edit_model: str = "") -> list:
    """Regenerate part of an image; outside the blend area the prepared source stays pixel-identical.

    mask_mode:
      "region": region = [x0, y0, x1, y1] as fractions of width/height (0-1), e.g. [0.3,0.1,0.7,0.5].
                 ellipse=True for a round mask (faces, heads).
      "file": mask_path = black/white image, white = change.
      "subject": change the main subject (auto-detected).  "background": replace the background.
    prompt describes desired replacement content concisely for dedicated inpainting models.
    For z_image compatibility mode, describe the whole desired image.
    strength 1.0 = fully new content in the mask; 0.5-0.8 = keep some of the original there.
    grow expands the edit area in working pixels. feather (0-31, default 16) softens its boundary.
    Sampling extends beyond the blend area; feather=0 gives a hard paste.
    edit_model: auto (default) prefers an installed dedicated content-aware inpainting checkpoint;
    sd15_inpaint selects the supplied SD 1.5 inpainting backend; z_image uses the old compatibility
    workflow. quality only selects a Z-Image tier when that backend is used.
    save_to: optional absolute path (a file like C:/proj/assets/hero.png, or a folder) to also save
    the result into, e.g. the user's project. A .jpg/.webp extension converts the format.
    """
    if not 0 <= feather <= 31:
        raise ValueError("feather must be between 0 and 31 pixels")
    if mask_mode in ("subject", "background"):
        _need("bg_removal")
    qname, q, editing_model, note = _pick_edit_model(edit_model, quality, "inpaint")
    seed = _seed(seed)
    ensure_comfy()
    prepared, w, h = imaging.prepare(image_path, WORK, megapixels)
    name = comfy.upload_image(prepared)
    mask_name = None
    if mask_mode == "region":
        if not region or len(region) != 4:
            raise ValueError("mask_mode 'region' needs region=[x0, y0, x1, y1] (fractions 0-1)")
        mask_name = comfy.upload_image(imaging.region_mask(w, h, region, WORK, ellipse=ellipse))
    elif mask_mode == "file":
        mask_name = comfy.upload_image(imaging.resize_mask(mask_path, w, h, WORK))
    wf = workflows.inpaint(q, CFG, prompt, name, mask_mode, mask_name, strength, grow, seed,
                           q.get("steps") if q else None, feather=feather, edit_model=editing_model)
    meta = {"tool": "inpaint", "prompt": prompt, "source": image_path, "mask_mode": mask_mode,
            "region": region, "strength": strength, "grow": grow, "feather": feather, "seed": seed, "quality": qname,
            "edit_model": qname if editing_model else "z_image"}
    return _run(wf, meta, note, save_to)


@mcp.tool(structured_output=False)
def outpaint(image_path: str, prompt: str, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0,
             quality: str = "", seed: int = -1, megapixels: float = 0.8, save_to: str = "", feather: int = 40,
             edit_model: str = "") -> list:
    """Extend an image beyond its borders (e.g. make a square photo wide).
    left/top/right/bottom: pixels to add on each side, measured at the working size
    (the source is first scaled to ~megapixels; e.g. 1024x768 -> add right=256 left=256).
    prompt describes the whole extended scene.
    feather: overlap into the original in working pixels (default 40). The original outside
    this overlap is preserved; 0 disables blending. Must be less than half each source dimension.
    edit_model: auto prefers an installed dedicated content-aware inpainting checkpoint;
    sd15_inpaint requires that backend; z_image forces compatibility mode. quality only applies to Z-Image.
    save_to: optional absolute path (a file like C:/proj/assets/hero.png, or a folder) to also save
    the result into, e.g. the user's project. A .jpg/.webp extension converts the format.
    """
    if not 0 <= feather <= 16368:
        raise ValueError("feather must be between 0 and 16368 pixels")
    qname, q, editing_model, note = _pick_edit_model(edit_model, quality, "outpaint")
    seed = _seed(seed)
    r16 = lambda v: max(0, round(v / 16) * 16)
    left, top, right, bottom = r16(left), r16(top), r16(right), r16(bottom)
    if not any((left, top, right, bottom)):
        raise ValueError("Give at least one of left/top/right/bottom > 0")
    name, w, h = _upload(image_path, megapixels)
    if feather and feather * 2 >= min(w, h):
        raise ValueError("feather must be less than half each working source dimension")
    wf = workflows.outpaint(q, CFG, prompt, name, left, top, right, bottom, seed,
                            q.get("steps") if q else None, feather=feather, edit_model=editing_model)
    meta = {"tool": "outpaint", "prompt": prompt, "source": image_path, "pad": [left, top, right, bottom],
            "seed": seed, "quality": qname, "feather": feather, "width": w + left + right, "height": h + top + bottom,
            "edit_model": qname if editing_model else "z_image"}
    return _run(wf, meta, note, save_to)


@mcp.tool(structured_output=False)
def recolor_image(image_path: str, color: str, mask_path: str = "", region: list[float] | None = None,
                  strength: float = 1.0, saturation: float = 1.0, feather: float = 0,
                  save_to: str = "") -> list:
    """Change colour inside a selection while preserving shading, linework and geometry.

    Use for colour-only edits such as an orange shirt becoming blue. Runs locally on CPU;
    ComfyUI and model files are not needed. Keeps original displayed dimensions and transparency.
    color: opaque hex or colour name, e.g. '#3267C8' or 'royalblue'. The target supplies hue/chroma;
    its brightness is ignored to retain source shading. Neutral targets desaturate.
    mask_path: black/white image, white=change. A precise garment/prop mask is best.
    region: optional [x0,y0,x1,y1] in fractions (0-1) or original pixels. If both are supplied,
    the region intersects the file mask. A rectangle alone changes everything inside it.
    strength 0-1 blends the colour change; saturation 0-2 controls target colour intensity.
    feather 0-64 softens only INSIDE the mask; pixels outside its support stay identical.
    save_to: optional absolute PNG/JPG/WebP file or folder, as with the other image tools.
    """
    started = time.time()
    image = recoloring.load_image(image_path)
    mask = recoloring.selection(image.size, mask_path, region)
    result, applied_mask, stats = recoloring.recolor(image, mask, color, strength, saturation, feather)
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"recolor_{time.time_ns()}.png"
    mask_output = output.with_name(output.stem + "_mask.png")
    result.save(output, icc_profile=image.info.get("icc_profile"))
    applied_mask.save(mask_output)
    meta = {"tool": "recolor_image", "source": image_path, "color": color, "mask_path": mask_path,
            "region": region, "strength": strength, "saturation": saturation, "feather": feather,
            "width": image.width, "height": image.height, "seconds": round(time.time()-started, 3),
            "applied_mask": str(mask_output), **stats}
    saved, save_error = [], ""
    if save_to:
        meta["save_to"] = save_to
        try:
            saved = _copy_to([str(output)], save_to)
        except Exception as error:
            save_error = f"\nCould not save to {save_to}: {error}"
    if saved:
        meta["saved_to"] = saved
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    text = (f"Recoloured {stats['changed_pixels']} pixels in {meta['seconds']}s. "
            f"Original resolution {image.width}x{image.height}; shading and geometry preserved.\n"
            f"Images:\n{output}\nApplied mask: {mask_output}")
    if saved:
        text += "\nSaved to: " + ", ".join(saved)
    text += save_error
    return _result(text, [str(output)])


@mcp.tool(structured_output=False)
def upscale(image_path: str, scale: float = 2.0, refine: bool = False, prompt: str = "",
            refine_strength: float = 0.3, quality: str = "", seed: int = -1, save_to: str = "") -> list:
    """Upscale an image with 4x-UltraSharp (fast, a few seconds). scale 1-4.
    refine=True then runs a low-denoise Z-Image pass at the new size to add real detail
    (much slower; keep the result <= ~2.5 megapixels on this 6 GB GPU). Give a prompt
    describing the image when refining; refine_strength 0.2-0.4.
    save_to: optional absolute path (a file like C:/proj/assets/hero.png, or a folder) to also save
    the result into, e.g. the user's project. A .jpg/.webp extension converts the format.
    """
    _need("upscaler")
    name, w, h = _upload(image_path)
    meta = {"tool": "upscale", "source": image_path, "scale": scale, "refine": refine,
            "width": round(w * scale), "height": round(h * scale)}
    if not refine:
        return _run(workflows.upscale(CFG, name, scale), meta, save_to=save_to)
    qname, q, note = _pick_quality(quality, "upscale")
    seed = _seed(seed)
    wf = workflows.upscale_refine(q, CFG, prompt or "high quality, detailed, sharp", name, scale, refine_strength, seed, q.get("steps"))
    meta.update({"prompt": prompt, "refine_strength": refine_strength, "seed": seed, "quality": qname})
    return _run(wf, meta, note, save_to)


@mcp.tool(structured_output=False)
def remove_background(image_path: str, save_to: str = "") -> list:
    """Cut out the main subject into a transparent PNG (BiRefNet). Fast.
    save_to: optional absolute path (a file like C:/proj/assets/hero.png, or a folder) to also save
    the result into, e.g. the user's project. A .jpg/.webp extension converts the format.
    """
    _need("bg_removal")
    name, w, h = _upload(image_path)
    return _run(workflows.remove_background(CFG, name), {"tool": "remove_background", "source": image_path},
                save_to=save_to)


@mcp.tool(structured_output=False)
def run_custom_workflow(name: str, params: dict | None = None, save_to: str = "") -> list:
    """Run a custom workflow (ComfyUI "Export (API)" format) from ~/.comfyclaude/workflows/<name>.json
    or the bundled workflows_custom/ folder. Values written as "{{key}}" in that JSON are replaced
    from params. Params whose key starts with "image" and point to an existing local file are uploaded first.
    save_to: optional absolute path (a file like C:/proj/assets/hero.png, or a folder) to also save
    the result into, e.g. the user's project. A .jpg/.webp extension converts the format.
    """
    found = _custom_workflows()
    if name not in found:
        raise FileNotFoundError(f"No custom workflow '{name}'. Available: {sorted(found)}")
    path = found[name]
    params = dict(params or {})
    ensure_comfy()
    for k, v in params.items():
        if k.startswith("image") and isinstance(v, str) and Path(v).exists():
            params[k] = comfy.upload_image(v)
    params.setdefault("seed", _seed(-1))
    wf = workflows.fill_template(json.loads(path.read_text()), params)
    return _run(wf, {"tool": "run_custom_workflow", "workflow": name, "params": params}, save_to=save_to)


def _custom_workflows():
    found = {}
    for d in CUSTOM_DIRS:  # user folder last, so it overrides bundled examples
        found.update({p.stem: p for p in d.glob("*.json")} if d.exists() else {})
    return found


# ---- utility tools ---------------------------------------------------------------
@mcp.tool(structured_output=False)
def job_status(prompt_id: str, wait_seconds: int = 50) -> list:
    """Wait (up to wait_seconds) for a job that was still running and return its images."""
    ensure_comfy()
    status, paths = _collect(prompt_id, wait_seconds)
    return _result(status + ("\nImages:\n" + "\n".join(paths) if paths else ""), paths)


@mcp.tool(structured_output=False)
def list_capabilities() -> str:
    """Show every tool with what it's good for, whether its models are installed, typical
    run times measured on this PC per quality tier, style presets, and custom workflows.
    Use this to choose the right tool and quality for a task."""
    try:
        ensure_comfy()
        comfy.object_info(refresh=True)  # pick up newly added model files
    except ConnectionError as e:
        return f"ComfyUI unavailable: {e}\nLOCAL TOOLS\n  recolor_image [ready]: masked colour change; no ComfyUI/models required"
    timings = {}
    for m in _load_jobs().values():
        if m.get("seconds"):
            timings.setdefault((m.get("tool"), m.get("quality")), []).append(m["seconds"])
    lines = ["QUALITY TIERS"]
    for name, q in CFG["qualities"].items():
        problem = _quality_problem(q)
        lines.append(f"  {name}: {q['description']} [{'ready' if not problem else problem}]")
    lines.append("\nGENERATION MODELS (generate_with_model; optional SDXL checkpoints)")
    for name, profile in CFG.get("generation_models", {}).items():
        ready = _has("CheckpointLoaderSimple", "ckpt_name", profile["checkpoint"])
        lines.append(f"  {name}: {profile.get('description', name)} [{'ready' if ready else 'missing checkpoint: ' + profile['checkpoint']}]")
    lines.append("  LoRAs: " + (", ".join(comfy.options("LoraLoader", "lora_name") or []) or "(none installed)"))
    lines.append("\nTOOLS")
    for tool, info in CAPS["tools"].items():
        missing = [CFG["models"][k] for k in info["needs"] if not _has(*CAPS["model_nodes"][k], CFG["models"][k])]
        state = "ready" if not missing else f"missing models: {missing}"
        times = ", ".join(f"{q or '-'} ~{sum(v) / len(v):.0f}s (n={len(v)})"
                          for (t, q), v in timings.items() if t == tool)
        lines.append(f"  {tool} [{state}] default quality={info['quality']}: {info['good_for']}"
                     + (f"\n      measured: {times}" if times else ""))
    lines.append("\nSTYLE PRESETS: " + ", ".join(PRESETS))
    lines.append("\nEDITING MODELS (inpaint/outpaint)")
    for name, model in CFG.get("editing", {}).get("models", {}).items():
        ready = _has("CheckpointLoaderSimple", "ckpt_name", model["checkpoint"])
        lines.append(f"  {name}: {model.get('description', name)} [{'ready' if ready else 'missing checkpoint: ' + model['checkpoint']}]")
    lines.append("  auto: prefer an installed editing model; z_image: legacy compatibility workflow")
    customs = sorted(_custom_workflows())
    lines.append("CUSTOM WORKFLOWS: " + (", ".join(customs) if customs else "(none)"))
    lines.append(f"\nOutputs: {OUT}   User config: {settings.user_config_path()}")
    return "\n".join(lines)


@mcp.tool(structured_output=False)
def comfy_status() -> str:
    """Check ComfyUI is running (starts it if not) and show GPU/VRAM/RAM info."""
    ensure_comfy()
    s = comfy.system_stats()
    sysinfo = s.get("system", {})
    devs = [f"{d['name']}: {d['vram_free'] / 2**30:.1f}/{d['vram_total'] / 2**30:.1f} GB VRAM free"
            for d in s.get("devices", [])]
    ram = f"RAM free {sysinfo.get('ram_free', 0) / 2**30:.1f}/{sysinfo.get('ram_total', 0) / 2**30:.1f} GB"
    return f"ComfyUI {sysinfo.get('comfyui_version')} at {comfy.base}\n" + "\n".join(devs) + f"\n{ram}"


@mcp.tool(structured_output=False)
def free_memory() -> str:
    """Unload models from VRAM/RAM in ComfyUI (use when switching model families or RAM is tight)."""
    ensure_comfy()
    comfy.free_memory()
    return "Models unloaded."


if __name__ == "__main__":
    mcp.run()
