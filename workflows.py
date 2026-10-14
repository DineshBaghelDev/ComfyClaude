"""Builds ComfyUI API-format workflows for Z-Image Turbo.

Workflows are built in code so the same graph works for every quality tier
(bf16 / GGUF Q8 / GGUF Q5) by swapping only the loader nodes.
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


def _base(q, cfg, prompt, seed, steps, prefix):
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
                "denoise": 1.0,
            },
        },
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    })
    return wf


def text_to_image(q, cfg, prompt, width, height, seed, steps=None, prefix="claude/t2i"):
    wf = _base(q, cfg, prompt, seed, steps, prefix)
    wf["latent"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
    return wf


def image_to_image(q, cfg, prompt, image_name, strength, seed, megapixels=1.0, steps=None, prefix="claude/i2i"):
    """strength = denoise: 0.3 keeps the reference closely, 0.8 only keeps rough layout/colors."""
    wf = _base(q, cfg, prompt, seed, steps, prefix)
    wf["load"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    wf["scale"] = {
        "class_type": "ImageScaleToTotalPixels",
        "inputs": {"image": ["load", 0], "upscale_method": "lanczos", "megapixels": megapixels},
    }
    wf["latent"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["scale", 0], "vae": ["vae", 0]}}
    wf["sampler"]["inputs"]["denoise"] = strength
    return wf
