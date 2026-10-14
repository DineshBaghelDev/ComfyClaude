"""Builds ComfyUI API-format workflows for Z-Image Turbo and helper models.

Workflows are built in code so every graph works for each quality tier
(bf16 / GGUF Q8 / GGUF Q5) by swapping only the loader nodes. Images arrive
already resized (see imaging.py), so graphs don't need resize logic.
"""


def _loaders(q, vae):
    if q["loader"] == "gguf":
        unet = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": q["unet"]}}
        clip = {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": q["clip"], "type": "lumina2"}}
    else:
        unet = {"class_type": "UNETLoader", "inputs": {"unet_name": q["unet"], "weight_dtype": "default"}}
        clip = {"class_type": "CLIPLoader", "inputs": {"clip_name": q["clip"], "type": "lumina2", "device": "default"}}
    return {
        "unet": unet,
        "clip": clip,
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }


def _base(q, cfg, prompt, seed, steps, prefix, denoise=1.0):
    """Shared graph: loaders -> sampler -> decode -> save. Caller adds a "latent" node."""
    s = cfg["sampler"]
    wf = _loaders(q, cfg["vae"])
    wf.update({
        "model": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": s["shift"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["model", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "latent_image": ["latent", 0],
                "seed": seed,
                "steps": steps or s["steps"],
                "cfg": s["cfg"],
                "sampler_name": s["sampler_name"],
                "scheduler": s["scheduler"],
                "denoise": denoise,
            },
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    })
    return wf


def _load(name):
    return {"class_type": "LoadImage", "inputs": {"image": name}}


def text_to_image(q, cfg, prompt, width, height, seed, steps=None, prefix="claude/t2i"):
    wf = _base(q, cfg, prompt, seed, steps, prefix)
    wf["latent"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
    return wf


def image_to_image(q, cfg, prompt, image_name, strength, seed, steps=None, prefix="claude/i2i"):
    """strength = denoise: 0.3 keeps the reference closely, 0.8 only keeps rough layout/colors."""
    wf = _base(q, cfg, prompt, seed, steps, prefix, denoise=strength)
    wf["load"] = _load(image_name)
    wf["latent"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["load", 0], "vae": ["vae", 0]}}
    return wf


def controlnet(q, cfg, prompt, control_name, mode, width, height, strength, seed, steps=None, prefix="claude/control"):
    """Z-Image Fun ControlNet Union. mode: canny | pose | direct (image is already a control map)."""
    wf = _base(q, cfg, prompt, seed, steps, prefix)
    wf["load"] = _load(control_name)
    if mode == "canny":
        wf["ctrl"] = {"class_type": "Canny", "inputs": {"image": ["load", 0], "low_threshold": 0.1, "high_threshold": 0.32}}
    elif mode == "pose":
        wf["pose_ckpt"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": cfg["models"]["pose"]}}
        wf["pose_kp"] = {"class_type": "SDPoseKeypointExtractor", "inputs": {
            "model": ["pose_ckpt", 0], "vae": ["pose_ckpt", 2], "image": ["load", 0], "batch_size": 1}}
        wf["ctrl"] = {"class_type": "SDPoseDrawKeypoints", "inputs": {
            "keypoints": ["pose_kp", 0], "draw_body": True, "draw_hands": True, "draw_face": True,
            "draw_feet": True, "stick_width": 4, "face_point_size": 2, "score_threshold": 0.3, "draw_head": True}}
    elif mode == "direct":
        wf["ctrl"] = {"class_type": "ImageScale", "inputs": {
            "image": ["load", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}}
    else:
        raise ValueError(f"Unknown control mode '{mode}'. Use canny, pose or direct.")
    wf["patch"] = {"class_type": "ModelPatchLoader", "inputs": {"name": cfg["models"]["controlnet"]}}
    wf["cn"] = {"class_type": "ZImageFunControlnet", "inputs": {
        "model": ["model", 0], "model_patch": ["patch", 0], "vae": ["vae", 0],
        "strength": strength, "image": ["ctrl", 0]}}
    wf["sampler"]["inputs"]["model"] = ["cn", 0]
    wf["latent"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
    # Also save the control map so the result can be checked against it.
    wf["save_ctrl"] = {"class_type": "SaveImage", "inputs": {"images": ["ctrl", 0], "filename_prefix": prefix + "_map"}}
    return wf


def _mask_nodes(wf, cfg, mask_mode, mask_name, grow):
    """Adds a "mask" node. White/1 = area to regenerate."""
    if mask_mode in ("region", "file"):
        wf["mask_img"] = _load(mask_name)
        wf["mask_raw"] = {"class_type": "ImageToMask", "inputs": {"image": ["mask_img", 0], "channel": "red"}}
    elif mask_mode in ("subject", "background"):
        wf["bg_model"] = {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": cfg["models"]["bg_removal"]}}
        wf["mask_raw"] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["bg_model", 0], "image": ["load", 0]}}
        if mask_mode == "background":
            wf["mask_fg"] = wf.pop("mask_raw")
            wf["mask_raw"] = {"class_type": "InvertMask", "inputs": {"mask": ["mask_fg", 0]}}
    else:
        raise ValueError(f"Unknown mask_mode '{mask_mode}'. Use region, file, subject or background.")
    wf["mask"] = {"class_type": "GrowMask", "inputs": {"mask": ["mask_raw", 0], "expand": grow, "tapered_corners": True}}


def inpaint(q, cfg, prompt, image_name, mask_mode, mask_name, strength, grow, seed, steps=None, prefix="claude/inpaint"):
    """Regenerate only the masked area; everything else is pasted back pixel-exact."""
    wf = _base(q, cfg, prompt, seed, steps, prefix, denoise=strength)
    wf["load"] = _load(image_name)
    _mask_nodes(wf, cfg, mask_mode, mask_name, grow)
    wf["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["load", 0], "vae": ["vae", 0]}}
    wf["latent"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["enc", 0], "mask": ["mask", 0]}}
    wf["composite"] = {"class_type": "ImageCompositeMasked", "inputs": {
        "destination": ["load", 0], "source": ["decode", 0], "x": 0, "y": 0, "resize_source": False, "mask": ["mask", 0]}}
    wf["save"]["inputs"]["images"] = ["composite", 0]
    return wf


def outpaint(q, cfg, prompt, image_name, left, top, right, bottom, seed, steps=None, prefix="claude/outpaint"):
    """Extend the canvas; only the new border area is generated."""
    wf = _base(q, cfg, prompt, seed, steps, prefix)
    wf["load"] = _load(image_name)
    wf["pad"] = {"class_type": "ImagePadForOutpaint", "inputs": {
        "image": ["load", 0], "left": left, "top": top, "right": right, "bottom": bottom, "feathering": 40}}
    wf["enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["pad", 0], "vae": ["vae", 0]}}
    wf["latent"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["enc", 0], "mask": ["pad", 1]}}
    return wf


def upscale(cfg, image_name, scale, prefix="claude/upscale"):
    """Model upscale (4x-UltraSharp) then resize down to the requested scale."""
    return {
        "load": _load(image_name),
        "up_model": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": cfg["models"]["upscaler"]}},
        "up": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["up_model", 0], "image": ["load", 0]}},
        "down": {"class_type": "ImageScaleBy", "inputs": {"image": ["up", 0], "upscale_method": "lanczos", "scale_by": scale / 4}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["down", 0], "filename_prefix": prefix}},
    }


def upscale_refine(q, cfg, prompt, image_name, scale, strength, seed, steps=None, prefix="claude/upscale_refine"):
    """Model upscale, then a low-denoise Z-Image pass to add real detail ("hires fix")."""
    wf = _base(q, cfg, prompt, seed, steps, prefix, denoise=strength)
    wf.update({k: v for k, v in upscale(cfg, image_name, scale).items() if k != "save"})
    wf["latent"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["down", 0], "vae": ["vae", 0]}}
    return wf


def remove_background(cfg, image_name, prefix="claude/nobg"):
    """Transparent PNG of the subject."""
    return {
        "load": _load(image_name),
        "bg_model": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": cfg["models"]["bg_removal"]}},
        "fg": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["bg_model", 0], "image": ["load", 0]}},
        # JoinImageWithAlpha treats the mask as "transparent where 1", so invert the foreground mask.
        "inv": {"class_type": "InvertMask", "inputs": {"mask": ["fg", 0]}},
        "rgba": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["load", 0], "alpha": ["inv", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["rgba", 0], "filename_prefix": prefix}},
    }


def fill_template(template, params):
    """Custom workflows: replace "{{name}}" placeholders in an API-format JSON."""
    def sub(v):
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sub(x) for x in v]
        if isinstance(v, str) and "{{" in v:
            key = v.strip()[2:-2].strip()
            if v.strip().startswith("{{") and v.strip().endswith("}}") and key in params:
                return params[key]  # whole-value placeholder keeps the param's type
            for k, x in params.items():
                v = v.replace("{{" + k + "}}", str(x))
        return v
    return sub(template)
