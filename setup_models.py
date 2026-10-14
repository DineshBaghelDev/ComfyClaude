"""One-time setup: installs the ComfyUI-GGUF node and downloads Q8 + Q5 GGUF models.

Run (with ComfyUI closed):   python setup_models.py
Downloads ~19 GB total. Safe to re-run; finished files are skipped.
"""
import json
import subprocess
import sys
from pathlib import Path

CFG = json.loads((Path(__file__).parent / "config.json").read_text())
COMFY = Path(CFG["comfy_root"])
MODELS = Path(CFG["models_dir"])
COMFY_PY = COMFY / ".venv" / "Scripts" / "python.exe"

# Candidate repos, tried in order. File names are matched by quant tag.
UNET_REPOS = ["jayn7/Z-Image-Turbo-GGUF", "QuantStack/Z-Image-Turbo-GGUF", "city96/Z-Image-Turbo-gguf"]
CLIP_REPOS = ["unsloth/Qwen3-4B-GGUF", "Qwen/Qwen3-4B-GGUF"]
QUANTS = {"Q8_0": "balanced", "Q5_K_M": "fast"}


def install_gguf_node():
    node = COMFY / "custom_nodes" / "ComfyUI-GGUF"
    if not node.exists():
        print("Cloning ComfyUI-GGUF ...")
        subprocess.check_call(["git", "clone", "https://github.com/city96/ComfyUI-GGUF", str(node)])
    py = str(COMFY_PY) if COMFY_PY.exists() else sys.executable
    print(f"Installing gguf package into {py} ...")
    subprocess.check_call([py, "-m", "pip", "install", "-r", str(node / "requirements.txt")])


def find_file(api, repos, quant):
    for repo in repos:
        try:
            files = api.list_repo_files(repo)
        except Exception:
            continue
        hits = [f for f in files if f.endswith(".gguf") and quant.lower() in f.lower()]
        if hits:
            return repo, sorted(hits, key=len)[0]
    return None, None


def download_models():
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    for quant, tier in QUANTS.items():
        for kind, repos, sub in (("unet", UNET_REPOS, "unet"), ("clip", CLIP_REPOS, "text_encoders")):
            repo, fname = find_file(api, repos, quant)
            if not repo:
                print(f"!! No {kind} {quant} file found in {repos}. Download manually.")
                continue
            dest_dir = MODELS / sub
            dest_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {repo}/{fname} -> {dest_dir}")
            path = Path(hf_hub_download(repo, fname, local_dir=dest_dir))
            # Point config at the real file name.
            CFG["qualities"][tier][kind] = path.name
    (Path(__file__).parent / "config.json").write_text(json.dumps(CFG, indent=2))
    print("config.json updated with downloaded file names.")


if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    install_gguf_node()
    download_models()
    print("Done. Restart ComfyUI Desktop.")
