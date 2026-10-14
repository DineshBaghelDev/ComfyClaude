"""ComfyClaude setup: installs the ComfyUI-GGUF node and downloads models into your ComfyUI.

Examples:
  python setup_models.py --comfy-path C:/ComfyUI                 # balanced tier (Q8) + VAE
  python setup_models.py --comfy-path C:/ComfyUI --tiers balanced,lowmem --extras
  python setup_models.py --models-dir D:/shared/models --tiers max  # custom models folder
  python setup_models.py --check                                   # show what's installed

Tiers: balanced/fast (Q8, ~11 GB), lowmem (Q5, ~8.5 GB), max (bf16, ~19 GB).
Extras (~4.5 GB): ControlNet Union 2.1 lite, SDPose, BiRefNet, 4x-UltraSharp.
Re-running is safe: finished files are skipped, interrupted downloads resume.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import settings

# name -> (repo, file in repo, models subfolder)
FILES = {
    "vae":        ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors", "vae"),
    "unet_bf16":  ("Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors", "diffusion_models"),
    "clip_bf16":  ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders"),
    "unet_q8":    ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q8_0.gguf", "unet"),
    "clip_q8":    ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf", "text_encoders"),
    "unet_q5":    ("jayn7/Z-Image-Turbo-GGUF", "z_image_turbo-Q5_K_M.gguf", "unet"),
    "clip_q5":    ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q5_K_M.gguf", "text_encoders"),
    "controlnet": ("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
                   "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors", "model_patches"),
    "pose":       ("Comfy-Org/SDPose", "checkpoints/sdpose_wholebody_fp16.safetensors", "checkpoints"),
    "bg_removal": ("Comfy-Org/BiRefNet", "background_removal/birefnet.safetensors", "background_removal"),
    "upscaler":   ("lokCX/4x-Ultrasharp", "4x-UltraSharp.pth", "upscale_models"),
}
TIERS = {
    "balanced": ["unet_q8", "clip_q8"], "fast": ["unet_q8", "clip_q8"],
    "lowmem": ["unet_q5", "clip_q5"], "max": ["unet_bf16", "clip_bf16"],
}
EXTRAS = ["controlnet", "pose", "bg_removal", "upscaler"]
GGUF_NODE = "https://github.com/city96/ComfyUI-GGUF"


def install_gguf_node(comfy_path, python):
    node = Path(comfy_path) / "custom_nodes" / "ComfyUI-GGUF"
    if not node.exists():
        print(f"Installing ComfyUI-GGUF node into {node.parent}")
        subprocess.check_call(["git", "clone", "--depth", "1", GGUF_NODE, str(node)])
    print(f"Installing its Python requirements with {python}")
    subprocess.check_call([python, "-m", "pip", "install", "-q", "-r", str(node / "requirements.txt")])


def download(key, models_dir, retries=5):
    from huggingface_hub import hf_hub_download

    repo, fname, sub = FILES[key]
    dest = models_dir / sub / Path(fname).name
    if dest.exists():
        print(f"  have  {sub}/{dest.name}")
        return
    print(f"  get   {sub}/{dest.name}  (from {repo})")
    for attempt in range(1, retries + 1):
        try:
            tmp = Path(hf_hub_download(repo, fname, local_dir=models_dir / sub / ".download"))
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp.replace(dest)
            return
        except Exception as e:  # network drops: resume
            print(f"    attempt {attempt} failed: {e}")
    raise SystemExit(f"Could not download {repo}/{fname}")


def main():
    cfg, _, _ = settings.load()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy-path", default=cfg["comfy_path"], help="ComfyUI folder (contains main.py)")
    ap.add_argument("--models-dir", help="models folder (default: <comfy-path>/models)")
    ap.add_argument("--tiers", default="balanced", help="comma list: balanced,lowmem,max (fast = balanced files)")
    ap.add_argument("--extras", action="store_true", help="also download ControlNet, pose, background removal, upscaler")
    ap.add_argument("--no-node", action="store_true", help="skip installing the ComfyUI-GGUF custom node")
    ap.add_argument("--check", action="store_true", help="only report which files exist")
    a = ap.parse_args()

    if not a.models_dir and not a.comfy_path:
        ap.error("give --comfy-path (your ComfyUI folder) or --models-dir")
    models_dir = Path(a.models_dir or Path(a.comfy_path) / "models")
    tiers = [t.strip() for t in a.tiers.split(",") if t.strip()]
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        ap.error(f"unknown tiers {unknown}; choose from {list(TIERS)}")
    keys = ["vae"] + [k for t in tiers for k in TIERS[t]] + (EXTRAS if a.extras else [])
    keys = list(dict.fromkeys(keys))

    if a.check:
        for k, (_, fname, sub) in FILES.items():
            print(f"  {'OK ' if (models_dir / sub / Path(fname).name).exists() else '-- '} {sub}/{Path(fname).name}")
        return

    needs_gguf = any(t != "max" for t in tiers)
    if needs_gguf and not a.no_node:
        if not a.comfy_path:
            print("!! No --comfy-path: install https://github.com/city96/ComfyUI-GGUF yourself (ComfyUI Manager works).")
        else:
            install_gguf_node(a.comfy_path, settings.find_comfy_python(a.comfy_path, cfg.get("comfy_python", "")))

    print(f"Downloading into {models_dir}")
    for k in keys:
        download(k, models_dir)
    print("Done. Restart ComfyUI so it sees the new node/models.")


if __name__ == "__main__":
    main()
