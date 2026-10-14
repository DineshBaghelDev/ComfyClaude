"""ComfyClaude MCP server: gives Claude image generation via local ComfyUI + Z-Image Turbo."""
import json
import random
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import workflows
from comfy_client import ComfyClient

ROOT = Path(__file__).parent
CFG = json.loads((ROOT / "config.json").read_text())
OUT = Path(CFG["output_dir"])
JOBS_FILE = OUT / "jobs.json"

mcp = FastMCP("comfyclaude")
comfy = ComfyClient(CFG["comfy_urls"])


# ---- helpers ---------------------------------------------------------------
def _model_present(q):
    md = Path(CFG["models_dir"])
    sub_unet = "unet" if q["loader"] == "gguf" else "diffusion_models"
    unet_ok = any((md / d / q["unet"]).exists() for d in (sub_unet, "unet", "diffusion_models"))
    clip_ok = any((md / d / q["clip"]).exists() for d in ("text_encoders", "clip"))
    return unet_ok and clip_ok


def _pick_quality(name):
    name = name or CFG["default_quality"]
    if name not in CFG["qualities"]:
        raise ValueError(f"Unknown quality '{name}'. Options: {list(CFG['qualities'])}")
    q = CFG["qualities"][name]
    if not _model_present(q):
        # Fall back to bf16 if the GGUF files aren't downloaded yet.
        if name != "max" and _model_present(CFG["qualities"]["max"]):
            return "max", CFG["qualities"]["max"], f"(note: '{name}' models not found, used 'max')"
        raise FileNotFoundError(f"Model files for quality '{name}' not found in {CFG['models_dir']}")
    return name, q, ""


def _load_jobs():
    return json.loads(JOBS_FILE.read_text()) if JOBS_FILE.exists() else {}


def _save_job(pid, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = _load_jobs()
    jobs[pid] = meta
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))


def _submit_and_wait(wf, meta):
    pid = comfy.queue_prompt(wf)
    meta.update({"prompt_id": pid, "submitted": time.time()})
    _save_job(pid, meta)
    return _finish(pid, CFG["wait_seconds"])


def _finish(pid, timeout):
    hist = comfy.wait(pid, timeout)
    meta = _load_jobs().get(pid, {})
    if hist is None:
        pos = comfy.queue_position(pid) or "running"
        return f"Job {pid} still {pos}. Call job_status('{pid}') to keep waiting."
    if hist.get("status", {}).get("status_str") == "error":
        msgs = [m for m in hist["status"].get("messages", []) if m[0] == "execution_error"]
        return f"Job {pid} failed: {json.dumps(msgs)[:1500]}"
    paths = comfy.download_outputs(hist, OUT)
    meta["seconds"] = round(time.time() - meta.get("submitted", time.time()), 1)
    meta["files"] = paths
    for p in paths:
        Path(p).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    _save_job(pid, meta)
    return f"Done in {meta['seconds']}s ({meta.get('quality')}). Images:\n" + "\n".join(paths)


# ---- tools -----------------------------------------------------------------
@mcp.tool()
def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    quality: str = "",
    seed: int = -1,
    steps: int = 0,
) -> str:
    """Generate an image from text with Z-Image Turbo.

    quality: "fast" (GGUF Q5, drafts/throwaway), "balanced" (GGUF Q8, default,
    near-identical to original), "max" (bf16 original, slowest).
    Width/height should be multiples of 16. Use detailed natural-language prompts;
    Z-Image renders text in images well. Returns saved image paths (view them with Read),
    or a job id if still running.
    """
    qname, q, note = _pick_quality(quality)
    seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
    wf = workflows.text_to_image(q, CFG, prompt, width, height, seed, steps or None)
    meta = {"tool": "generate_image", "prompt": prompt, "width": width, "height": height,
            "seed": seed, "quality": qname, "steps": steps or CFG["sampler"]["steps"]}
    return (note + "\n" if note else "") + _submit_and_wait(wf, meta)


@mcp.tool()
def generate_from_reference(
    prompt: str,
    image_path: str,
    strength: float = 0.6,
    quality: str = "",
    seed: int = -1,
    megapixels: float = 1.0,
) -> str:
    """Generate a new image based on a reference image (img2img).

    strength 0.2-0.4: small changes, keeps the reference closely.
    strength 0.5-0.7: restyle/rework while keeping composition and colors.
    strength 0.8-0.95: only loose layout survives.
    image_path: absolute path to a local image file. Prompt should describe the desired result.
    """
    qname, q, note = _pick_quality(quality)
    seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
    name = comfy.upload_image(image_path)
    wf = workflows.image_to_image(q, CFG, prompt, name, strength, seed, megapixels)
    meta = {"tool": "generate_from_reference", "prompt": prompt, "reference": image_path,
            "strength": strength, "seed": seed, "quality": qname}
    return (note + "\n" if note else "") + _submit_and_wait(wf, meta)


@mcp.tool()
def job_status(prompt_id: str, wait_seconds: int = 50) -> str:
    """Wait for (up to wait_seconds) and fetch results of a job that was still running."""
    return _finish(prompt_id, wait_seconds)


@mcp.tool()
def list_qualities() -> str:
    """List quality tiers and whether their model files are installed."""
    lines = []
    for name, q in CFG["qualities"].items():
        ok = "installed" if _model_present(q) else "MISSING (run setup_models.py)"
        lines.append(f"{name}: {q['description']} [{ok}]")
    return f"default: {CFG['default_quality']}\n" + "\n".join(lines)


@mcp.tool()
def comfy_status() -> str:
    """Check ComfyUI is running and show GPU/VRAM info."""
    s = comfy.system_stats()
    devs = [f"{d['name']}: {d['vram_free'] / 2**30:.1f}/{d['vram_total'] / 2**30:.1f} GB free"
            for d in s.get("devices", [])]
    return f"ComfyUI at {comfy.base}, version {s.get('system', {}).get('comfyui_version')}\n" + "\n".join(devs)


@mcp.tool()
def free_memory() -> str:
    """Unload models from VRAM/RAM in ComfyUI (use when switching models or RAM is tight)."""
    comfy.free_memory()
    return "Models unloaded."


if __name__ == "__main__":
    mcp.run()
