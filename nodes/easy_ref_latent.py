"""
EasyConditionUtils - Easy Reference Latent Nodes
Compatible with FLUX.2 Klein 9B (flux2klein9b) and other Flux Kontext edit models.

Core feature: Automatically apply reference latent conditioning for any number
of input images in a single node, replacing the need to manually chain
multiple ReferenceLatent nodes.
"""

from __future__ import annotations
import torch
import comfy.utils
import node_helpers


def _scale_image_to_megapixels(
    img: torch.Tensor,
    megapixels: float,
    upscale_method: str,
) -> torch.Tensor:
    """
    Scale a [B, H, W, C] image tensor so that H*W ≈ megapixels * 1_000_000.
    Dimensions are rounded to the nearest multiple of 16 (safe for Flux VAE).
    Returns the image unchanged if it's already within 1% of the target size.
    """
    h, w = img.shape[1], img.shape[2]
    current_mp = (h * w) / 1_000_000.0
    if abs(current_mp - megapixels) / megapixels < 0.01:
        return img

    scale = (megapixels / current_mp) ** 0.5
    new_h = max(16, round(h * scale / 16) * 16)
    new_w = max(16, round(w * scale / 16) * 16)

    # comfy.utils.common_upscale expects [B, C, H, W]
    img_bchw = img.movedim(-1, 1)
    scaled = comfy.utils.common_upscale(img_bchw, new_w, new_h, upscale_method, "disabled")
    return scaled.movedim(1, -1)


def _ensure_divisible(img: torch.Tensor, divisor: int = 16) -> torch.Tensor:
    """Pad / crop image height and width to be divisible by `divisor`."""
    h, w = img.shape[1], img.shape[2]
    new_h = (h // divisor) * divisor
    new_w = (w // divisor) * divisor
    if new_h == h and new_w == w:
        return img
    return img[:, :new_h, :new_w, :]


# ---------------------------------------------------------------------------
# Node 1 — Main node: batch images → reference latent conditioning
# ---------------------------------------------------------------------------

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


class EasyReferenceLatentApply:
    """
    Automatically encode every image in an IMAGE batch and apply them as
    reference latents to the provided conditioning.

    This replaces manually chaining N separate ReferenceLatent nodes when you
    have N reference images. Works with FLUX.2 Klein 9B and any other
    ComfyUI edit model that supports the `reference_latents` conditioning key.

    Inputs
    ------
    conditioning : CONDITIONING
        Base conditioning (e.g. from CLIP Text Encode).
    vae : VAE
        VAE used to encode images to latent space.
    images : IMAGE
        One or more reference images as a batched tensor [B, H, W, C].
        Connect an Image Batch node or multiple Load Image nodes merged
        into a batch to pass several images at once.
    upscale_method : str
        Interpolation used when scaling images before VAE encoding.
    scale_to_megapixels : float
        Resize each reference image to approximately this many megapixels
        before encoding (e.g. 1.0 = 1 MP). Set to 0.0 to skip rescaling.

    Outputs
    -------
    conditioning : CONDITIONING
        Conditioning with all reference latents appended, equivalent to
        chaining N ReferenceLatent nodes.
    image_count : INT
        Number of reference images that were processed (handy for debugging).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Base conditioning to append reference latents to.",
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE model used to encode the reference images.",
                }),
                "images": ("IMAGE", {
                    "tooltip": (
                        "Batch of reference images [B, H, W, C]. "
                        "Each image in the batch becomes one reference latent."
                    ),
                }),
            },
            "optional": {
                "upscale_method": (UPSCALE_METHODS, {
                    "default": "bilinear",
                    "tooltip": "Interpolation algorithm used when rescaling images.",
                }),
                "scale_to_megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 8.0,
                    "step": 0.1,
                    "tooltip": (
                        "Resize each reference image to this many megapixels "
                        "before encoding. Set to 0 to keep original resolution."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "image_count")
    FUNCTION = "apply"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = (
        "Apply reference latent conditioning for every image in an IMAGE batch. "
        "Replaces chaining multiple ReferenceLatent nodes manually."
    )

    def apply(
        self,
        conditioning,
        vae,
        images: torch.Tensor,
        upscale_method: str = "bilinear",
        scale_to_megapixels: float = 1.0,
    ):
        batch_size = images.shape[0]
        result = conditioning

        for i in range(batch_size):
            img = images[i : i + 1]  # [1, H, W, C]

            # Optional rescale to target megapixels
            if scale_to_megapixels > 0.0:
                img = _scale_image_to_megapixels(img, scale_to_megapixels, upscale_method)

            # Ensure dimensions are divisible by 16 (Flux VAE requirement)
            img = _ensure_divisible(img, divisor=16)

            # Encode RGB channels only (drop alpha if present)
            latent = vae.encode(img[:, :, :, :3])

            # Append this reference latent to the conditioning
            result = node_helpers.conditioning_set_values(
                result,
                {"reference_latents": [latent]},
                append=True,
            )

        return (result, batch_size)


# ---------------------------------------------------------------------------
# Node 2 — Lightweight version: skip VAE encoding, accept pre-encoded LATENT
# ---------------------------------------------------------------------------

class EasyReferenceLatentFromLatent:
    """
    Apply a pre-encoded LATENT (or a list of latents concatenated along dim 0)
    as individual reference latents to the conditioning.

    Use this if you have already encoded your images through a VAE Encode node.
    Each slice along batch dimension 0 becomes one reference latent entry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Base conditioning to append reference latents to.",
                }),
                "latent": ("LATENT", {
                    "tooltip": (
                        "Pre-encoded latent. Each item in the batch dimension "
                        "becomes a separate reference latent entry."
                    ),
                }),
            },
            "optional": {
                "split_batch": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "If True, split the latent batch into individual reference "
                        "latents (one per batch item). If False, pass the whole "
                        "batch as a single reference."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "latent_count")
    FUNCTION = "apply"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = (
        "Apply a batched LATENT as multiple reference latents. "
        "Each batch item becomes one reference entry in the conditioning."
    )

    def apply(self, conditioning, latent, split_batch: bool = True):
        samples = latent["samples"]  # [B, C, H, W]
        result = conditioning

        if split_batch:
            count = samples.shape[0]
            for i in range(count):
                single = samples[i : i + 1]  # [1, C, H, W]
                result = node_helpers.conditioning_set_values(
                    result,
                    {"reference_latents": [single]},
                    append=True,
                )
        else:
            count = 1
            result = node_helpers.conditioning_set_values(
                result,
                {"reference_latents": [samples]},
                append=True,
            )

        return (result, count)


# ---------------------------------------------------------------------------
# Node 3 — Utility: clear all reference latents from conditioning
# ---------------------------------------------------------------------------

class EasyClearReferenceLatents:
    """
    Remove all reference latents from a conditioning.
    Useful when you want to reuse a conditioning but strip out previously
    appended reference images.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Conditioning from which reference latents will be removed.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "clear"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = "Strip all reference latents from the conditioning."

    def clear(self, conditioning):
        out = []
        for cond_tensor, cond_dict in conditioning:
            new_dict = {k: v for k, v in cond_dict.items() if k != "reference_latents"}
            out.append((cond_tensor, new_dict))
        return (out,)


# ---------------------------------------------------------------------------
# Node 4 — Utility: inspect / count reference latents in conditioning
# ---------------------------------------------------------------------------

class EasyCountReferenceLatents:
    """
    Read-only utility that counts how many reference latents are currently
    stored in a conditioning. Outputs the count as an INT and prints a summary
    to the console.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Conditioning to inspect.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("conditioning", "count", "summary")
    FUNCTION = "count"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = "Count and summarise reference latents stored in a conditioning."

    def count(self, conditioning):
        total = 0
        shapes = []
        for _cond_tensor, cond_dict in conditioning:
            refs = cond_dict.get("reference_latents", [])
            for ref in refs:
                total += 1
                if isinstance(ref, torch.Tensor):
                    shapes.append(str(list(ref.shape)))
                else:
                    shapes.append("?")

        summary = f"Reference latents: {total}"
        if shapes:
            summary += " | shapes: " + ", ".join(shapes)

        print(f"[EasyConditionUtils] {summary}")
        return (conditioning, total, summary)
