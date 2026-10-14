"""Configuration: shipped defaults <- user config file <- environment variables.

User config file: $COMFYCLAUDE_CONFIG, or ~/.comfyclaude/config.json. Any key from
config/defaults.json can be overridden there (dicts are merged, e.g. one quality tier).

Environment variables (set by the Claude Code plugin from its user settings):
  COMFYUI_URL            ComfyUI server address (default http://127.0.0.1:8188)
  COMFYUI_PATH           ComfyUI folder containing main.py; enables auto-start
  COMFYCLAUDE_OUTPUT_DIR where images are saved (default ~/ComfyClaude/outputs)
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
CONFIG_DIR = ROOT / "config"
USER_DIR = Path.home() / ".comfyclaude"


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _env(name):
    v = os.environ.get(name, "").strip()
    # Unset plugin options can arrive as empty or as an unsubstituted "${...}" template.
    return "" if v.startswith("${") else v


def user_config_path():
    return Path(_env("COMFYCLAUDE_CONFIG") or USER_DIR / "config.json")


def load():
    cfg = json.loads((CONFIG_DIR / "defaults.json").read_text(encoding="utf-8"))
    path = user_config_path()
    if path.exists():
        _merge(cfg, json.loads(path.read_text(encoding="utf-8")))
    for key, env in (("comfy_url", "COMFYUI_URL"), ("comfy_path", "COMFYUI_PATH"),
                     ("output_dir", "COMFYCLAUDE_OUTPUT_DIR")):
        if _env(env):
            cfg[key] = _env(env)
    cfg["comfy_url"] = cfg["comfy_url"].rstrip("/")
    cfg["output_dir"] = str(Path(cfg["output_dir"]).expanduser()) if cfg["output_dir"] \
        else str(Path.home() / "ComfyClaude" / "outputs")
    caps = json.loads((CONFIG_DIR / "capabilities.json").read_text(encoding="utf-8"))
    cfg["models"] = _merge(caps["models"], cfg.get("models", {}))
    presets = json.loads((CONFIG_DIR / "presets.json").read_text(encoding="utf-8"))
    user_presets = USER_DIR / "presets.json"
    if user_presets.exists():
        presets.update(json.loads(user_presets.read_text(encoding="utf-8")))
    return cfg, caps, presets


def find_comfy_python(comfy_path, override=""):
    """Locate the Python that runs ComfyUI (venv, portable build, or Desktop install)."""
    if override:
        return override
    root = Path(comfy_path)
    candidates = [
        root / ".venv" / "Scripts" / "python.exe", root / "venv" / "Scripts" / "python.exe",
        root.parent / "python_embeded" / "python.exe",           # Windows portable build
        root / ".venv" / "bin" / "python", root / "venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "python"
