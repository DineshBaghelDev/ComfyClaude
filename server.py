"""ComfyClaude MCP server: gives Claude image generation via local ComfyUI + Z-Image Turbo."""
import json
import random
import subprocess
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import imaging
import workflows
from comfy_client import ComfyClient

ROOT = Path(__file__).parent
CFG = json.loads((ROOT / "config.json").read_text())
CAPS = json.loads((ROOT / "capabilities.json").read_text())
PRESETS = json.loads((ROOT / "presets.json").read_text())
CFG["models"] = CAPS["models"]
OUT = Path(CFG["output_dir"])
WORK = OUT / "_work"
JOBS_FILE = OUT / "jobs.json"
CUSTOM_DIR = ROOT / "workflows_custom"

INSTRUCTIONS = """Local image generation (ComfyUI + Z-Image Turbo on a 6 GB GPU; jobs take ~30-150 s).
- Call list_capabilities once per session to see ready tools, default qualities and measured times.
- Quality: balanced (Q8, default) for normal use, fast (6 steps) for drafts/throwaway, max (bf16) only
  when the user asks for the best quality. Avoid switching tiers back and forth (each switch reloads models).
- Use generate_drafts when the direction is unclear, then generate_image with the chosen seed.
- Prompts: detailed natural language; text to render goes in quotes; no negative prompts; ~1 MP sizes.
- ALWAYS open each result with Read and check it against the request (subject, counts, text spelling,
  hands/faces). Fix problems (new seed, better prompt, or inpaint the broken area) before presenting.
- If a tool returns a job id, call job_status until done. Report file paths and seeds to the user."""

mcp = FastMCP("comfyclaude", instructions=INSTRUCTIONS)
comfy = ComfyClient(CFG["comfy_urls"])


# ---- ComfyUI process --------------------------------------------------------
def _popen_comfy(root, shared, log, flags):
    # Same flags as start_comfy.ps1. stdin/stdout must not inherit the MCP stdio channel.
    subprocess.Popen(
        [str(root / ".venv" / "Scripts" / "python.exe"), "main.py", "--port", "8188", "--enable-manager",
         "--extra-model-paths-config", str(ROOT / "comfy_paths.yaml"),
         "--input-directory", str(shared / "input"), "--output-directory", str(shared / "output"),
         *CFG.get("comfy_args", [])],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
    )


def ensure_comfy(timeout=150):
    """Connect to ComfyUI, starting it headless (start_comfy.ps1) if it isn't running."""
    try:
        return comfy.connect()
    except ConnectionError:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "comfy_server.log", "ab")
    root = Path(CFG["comfy_root"])
    shared = Path(CFG["models_dir"]).parent
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    # Break away from the parent's job object so ComfyUI survives MCP server restarts.
    for extra in (subprocess.CREATE_BREAKAWAY_FROM_JOB, 0):
        try:
            _popen_comfy(root, shared, log, flags | extra)
            break
        except OSError:
            continue
    end = time.time() + timeout
    while time.time() < end:
        time.sleep(3)
        try:
            return comfy.connect()
        except ConnectionError:
            continue
    raise ConnectionError(f"Started ComfyUI but it did not come up in {timeout}s. See {OUT / 'comfy_server.log'}")


# ---- models / quality --------------------------------------------------------
def _exists(folder, name):
    md = Path(CFG["models_dir"])
    folders = {"unet": ("unet", "diffusion_models"), "diffusion_models": ("diffusion_models", "unet"),
               "text_encoders": ("text_encoders", "clip")}.get(folder, (folder,))
    return any((md / f / name).exists() for f in folders)


def _quality_present(q):
    return _exists("unet" if q["loader"] == "gguf" else "diffusion_models", q["unet"]) and _exists("text_encoders", q["clip"])


def _need(*keys):
    missing = [CAPS["models"][k] for k in keys if not _exists(CAPS["model_folders"][k], CAPS["models"][k])]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}. Run: python setup_models.py --extras-only")


def _pick_quality(name, tool):
    """Resolve a quality tier; falls back to another installed tier if files are missing."""
    name = name or CAPS["tools"].get(tool, {}).get("quality") or CFG["default_quality"]
    if name not in CFG["qualities"]:
        raise ValueError(f"Unknown quality '{name}'. Options: {list(CFG['qualities'])}")
    if _quality_present(CFG["qualities"][name]):
        return name, CFG["qualities"][name], ""
    for alt in ("balanced", "fast", "max"):
        if alt != name and _quality_present(CFG["qualities"][alt]):
            return alt, CFG["qualities"][alt], f"(note: '{name}' models not installed, used '{alt}')\n"
    raise FileNotFoundError(f"No Z-Image model files found in {CFG['models_dir']}")


def _seed(seed):
    return seed if seed >= 0 else random.randint(0, 2**31 - 1)


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
        return f"Done ({meta.get('quality')}).", meta["files"]
    paths = comfy.download_outputs(hist, OUT)
    meta["seconds"] = round(time.time() - meta.get("submitted", time.time()), 1)
    meta["files"] = paths
    for p in paths:
        Path(p).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    _save_job(pid, meta)
    return f"Done in {meta['seconds']}s ({meta.get('quality') or meta.get('tool')}).", paths


def _run(wf, meta, note=""):
    pid = _submit(wf, meta)
    status, paths = _collect(pid, CFG["wait_seconds"])
    return note + status + ("\nImages:\n" + "\n".join(paths) if paths else "")


# ---- generation tools ----------------------------------------------------------
@mcp.tool()
def generate_image(prompt: str, width: int = 1024, height: int = 1024, quality: str = "",
                   style: str = "", seed: int = -1, steps: int = 0) -> str:
    """Generate an image from text with Z-Image Turbo.

    quality: "balanced" (GGUF Q8, default, near-identical to original), "fast" (same model, 6 steps,
    for drafts), "max" (bf16 original, slowest), "lowmem" (Q5, only if RAM is tight). style: optional preset name (see list_capabilities).
    Width/height: multiples of 16, ~1 megapixel is the sweet spot (1024x1024, 1216x832, 832x1216).
    Write detailed natural-language prompts; put any text to render in quotes.
    Returns saved image paths (open them with Read to check the result) or a job id if still running.
    """
    qname, q, note = _pick_quality(quality, "generate_image")
    seed = _seed(seed)
    full = _styled(prompt, style)
    steps = steps or q.get("steps") or CFG["sampler"]["steps"]
    wf = workflows.text_to_image(q, CFG, full, width, height, seed, steps)
    meta = {"tool": "generate_image", "prompt": full, "width": width, "height": height,
            "seed": seed, "quality": qname, "steps": steps}
    return _run(wf, meta, note)


@mcp.tool()
def generate_drafts(prompt: str, count: int = 4, width: int = 768, height: int = 768,
                    quality: str = "", style: str = "", seed: int = -1) -> str:
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
    out = note
    if results:
        sheet = imaging.contact_sheet([p for _, p in results], OUT / f"drafts_{base}.png",
                                      [f"seed {s}" for s, _ in results])
        out += f"Contact sheet: {sheet}\n" + "\n".join(f"seed {s}: {p}" for s, p in results)
    if pending:
        out += "\nStill running:\n" + "\n".join(pending)
    return out


@mcp.tool()
def generate_from_reference(prompt: str, image_path: str, strength: float = 0.6, quality: str = "",
                            style: str = "", seed: int = -1, megapixels: float = 1.0) -> str:
    """Generate a new image based on a reference image (img2img). Keeps composition and colors.

    strength 0.2-0.4: small changes, stays close to the reference.
    strength 0.5-0.7: restyle/rework while keeping composition and colors.
    strength 0.8-0.95: only loose layout survives.
    Tip: look at the reference first and describe what should be kept in the prompt.
    For exact outlines or a person's pose use generate_with_control instead.
    """
    qname, q, note = _pick_quality(quality, "generate_from_reference")
    seed = _seed(seed)
    full = _styled(prompt, style)
    name, w, h = _upload(image_path, megapixels)
    wf = workflows.image_to_image(q, CFG, full, name, strength, seed, q.get("steps"))
    meta = {"tool": "generate_from_reference", "prompt": full, "reference": image_path,
            "strength": strength, "seed": seed, "quality": qname, "width": w, "height": h}
    return _run(wf, meta, note)


@mcp.tool()
def generate_with_control(prompt: str, image_path: str, mode: str = "canny", strength: float = 0,
                          quality: str = "", style: str = "", seed: int = -1, megapixels: float = 1.0) -> str:
    """Generate an image that follows the structure of a reference using Z-Image ControlNet Union.

    mode "canny": follow the outlines/edges of the reference (architecture, objects, layouts).
    mode "pose": copy the body pose of people in the reference (extracted with SDPose).
    mode "direct": image_path already IS a control map (canny/depth/pose image).
    strength 0.6-1.0 (lower = more freedom); 0 = auto (canny 0.8, pose/direct 1.0).
    Pose quality depends on the extracted skeleton: check the *_map image (limbs hidden by clothing
    can be missed). The prompt fully decides colors, style and content.
    Also saves the control map (*_map) so you can check what was extracted.
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
    return _run(wf, meta, note)


@mcp.tool()
def inpaint(image_path: str, prompt: str, mask_mode: str = "region", region: list[float] | None = None,
            mask_path: str = "", ellipse: bool = False, strength: float = 1.0, grow: int = 16,
            quality: str = "", seed: int = -1, megapixels: float = 1.0) -> str:
    """Regenerate only part of an image; the rest stays pixel-identical.

    mask_mode:
      "region": region = [x0, y0, x1, y1] as fractions of width/height (0-1), e.g. [0.3,0.1,0.7,0.5].
                 ellipse=True for a round mask (faces, heads).
      "file": mask_path = black/white image, white = change.
      "subject": change the main subject (auto-detected).  "background": replace the background.
    prompt describes the WHOLE desired image (not just the patch).
    strength 1.0 = fully new content in the mask; 0.5-0.8 = keep some of the original there.
    grow expands the mask in pixels for smoother blending.
    """
    if mask_mode in ("subject", "background"):
        _need("bg_removal")
    qname, q, note = _pick_quality(quality, "inpaint")
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
    wf = workflows.inpaint(q, CFG, prompt, name, mask_mode, mask_name, strength, grow, seed, q.get("steps"))
    meta = {"tool": "inpaint", "prompt": prompt, "source": image_path, "mask_mode": mask_mode,
            "region": region, "strength": strength, "seed": seed, "quality": qname}
    return _run(wf, meta, note)


@mcp.tool()
def outpaint(image_path: str, prompt: str, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0,
             quality: str = "", seed: int = -1, megapixels: float = 0.8) -> str:
    """Extend an image beyond its borders (e.g. make a square photo wide).
    left/top/right/bottom: pixels to add on each side, measured at the working size
    (the source is first scaled to ~megapixels; e.g. 1024x768 -> add right=256 left=256).
    prompt describes the whole extended scene.
    """
    qname, q, note = _pick_quality(quality, "outpaint")
    seed = _seed(seed)
    r16 = lambda v: max(0, round(v / 16) * 16)
    left, top, right, bottom = r16(left), r16(top), r16(right), r16(bottom)
    if not any((left, top, right, bottom)):
        raise ValueError("Give at least one of left/top/right/bottom > 0")
    name, w, h = _upload(image_path, megapixels)
    wf = workflows.outpaint(q, CFG, prompt, name, left, top, right, bottom, seed, q.get("steps"))
    meta = {"tool": "outpaint", "prompt": prompt, "source": image_path, "pad": [left, top, right, bottom],
            "seed": seed, "quality": qname, "width": w + left + right, "height": h + top + bottom}
    return _run(wf, meta, note)


@mcp.tool()
def upscale(image_path: str, scale: float = 2.0, refine: bool = False, prompt: str = "",
            refine_strength: float = 0.3, quality: str = "", seed: int = -1) -> str:
    """Upscale an image with 4x-UltraSharp (fast, a few seconds). scale 1-4.
    refine=True then runs a low-denoise Z-Image pass at the new size to add real detail
    (much slower; keep the result <= ~2.5 megapixels on this 6 GB GPU). Give a prompt
    describing the image when refining; refine_strength 0.2-0.4.
    """
    _need("upscaler")
    name, w, h = _upload(image_path)
    meta = {"tool": "upscale", "source": image_path, "scale": scale, "refine": refine,
            "width": round(w * scale), "height": round(h * scale)}
    if not refine:
        return _run(workflows.upscale(CFG, name, scale), meta)
    qname, q, note = _pick_quality(quality, "upscale")
    seed = _seed(seed)
    wf = workflows.upscale_refine(q, CFG, prompt or "high quality, detailed, sharp", name, scale, refine_strength, seed, q.get("steps"))
    meta.update({"prompt": prompt, "refine_strength": refine_strength, "seed": seed, "quality": qname})
    return _run(wf, meta, note)


@mcp.tool()
def remove_background(image_path: str) -> str:
    """Cut out the main subject into a transparent PNG (BiRefNet). Fast."""
    _need("bg_removal")
    name, w, h = _upload(image_path)
    return _run(workflows.remove_background(CFG, name), {"tool": "remove_background", "source": image_path})


@mcp.tool()
def run_custom_workflow(name: str, params: dict | None = None) -> str:
    """Run a user workflow from workflows_custom/<name>.json (ComfyUI "Export (API)" format).
    Values written as "{{key}}" in that JSON are replaced from params. Params whose key starts
    with "image" and point to an existing local file are uploaded first.
    """
    path = CUSTOM_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No custom workflow '{name}'. Available: {[p.stem for p in CUSTOM_DIR.glob('*.json')]}")
    params = dict(params or {})
    ensure_comfy()
    for k, v in params.items():
        if k.startswith("image") and isinstance(v, str) and Path(v).exists():
            params[k] = comfy.upload_image(v)
    params.setdefault("seed", _seed(-1))
    wf = workflows.fill_template(json.loads(path.read_text()), params)
    return _run(wf, {"tool": "run_custom_workflow", "workflow": name, "params": params})


# ---- utility tools ---------------------------------------------------------------
@mcp.tool()
def job_status(prompt_id: str, wait_seconds: int = 50) -> str:
    """Wait (up to wait_seconds) for a job that was still running and return its images."""
    ensure_comfy()
    status, paths = _collect(prompt_id, wait_seconds)
    return status + ("\nImages:\n" + "\n".join(paths) if paths else "")


@mcp.tool()
def list_capabilities() -> str:
    """Show every tool with what it's good for, whether its models are installed, typical
    run times measured on this PC per quality tier, style presets, and custom workflows.
    Use this to choose the right tool and quality for a task."""
    timings = {}
    for m in _load_jobs().values():
        if m.get("seconds"):
            timings.setdefault((m.get("tool"), m.get("quality")), []).append(m["seconds"])
    lines = ["QUALITY TIERS"]
    for name, q in CFG["qualities"].items():
        ok = "installed" if _quality_present(q) else "NOT INSTALLED"
        lines.append(f"  {name}: {q['description']} [{ok}]")
    lines.append("\nTOOLS")
    for tool, info in CAPS["tools"].items():
        missing = [k for k in info["needs"] if not _exists(CAPS["model_folders"][k], CAPS["models"][k])]
        state = "ready" if not missing else f"missing models: {missing}"
        times = ", ".join(f"{q or '-'} ~{sum(v) / len(v):.0f}s (n={len(v)})"
                          for (t, q), v in timings.items() if t == tool)
        lines.append(f"  {tool} [{state}] default quality={info['quality']}: {info['good_for']}"
                     + (f"\n      measured: {times}" if times else ""))
    lines.append("\nSTYLE PRESETS: " + ", ".join(PRESETS))
    customs = [p.stem for p in CUSTOM_DIR.glob("*.json")]
    lines.append("CUSTOM WORKFLOWS: " + (", ".join(customs) if customs else "(none)"))
    return "\n".join(lines)


@mcp.tool()
def comfy_status() -> str:
    """Check ComfyUI is running (starts it if not) and show GPU/VRAM/RAM info."""
    ensure_comfy()
    s = comfy.system_stats()
    sysinfo = s.get("system", {})
    devs = [f"{d['name']}: {d['vram_free'] / 2**30:.1f}/{d['vram_total'] / 2**30:.1f} GB VRAM free"
            for d in s.get("devices", [])]
    ram = f"RAM free {sysinfo.get('ram_free', 0) / 2**30:.1f}/{sysinfo.get('ram_total', 0) / 2**30:.1f} GB"
    return f"ComfyUI {sysinfo.get('comfyui_version')} at {comfy.base}\n" + "\n".join(devs) + f"\n{ram}"


@mcp.tool()
def free_memory() -> str:
    """Unload models from VRAM/RAM in ComfyUI (use when switching model families or RAM is tight)."""
    ensure_comfy()
    comfy.free_memory()
    return "Models unloaded."


if __name__ == "__main__":
    mcp.run()
