from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F
import node_helpers

from .easy_ref_latent import _ensure_divisible, _scale_image_to_megapixels, UPSCALE_METHODS


class _DynamicImageInputs(dict):
    """Allow ComfyUI to pass arbitrary dynamically-created img_nn inputs."""

    _image_spec = ("IMAGE", {"forceInput": True})

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        return isinstance(key, str) and bool(re.fullmatch(r"img_\d{2}", key))

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str) and re.fullmatch(r"img_\d{2}", key):
            return self._image_spec
        raise KeyError(key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


def _add_reference_latent(conditioning: list, latent_tensor: torch.Tensor) -> list:
    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [latent_tensor]},
        append=True,
    )


def _make_empty_flux_latent(width: int, height: int, batch_size: int) -> dict:
    width = max(16, (width // 8) * 8)
    height = max(16, (height // 8) * 8)
    return {"samples": torch.zeros([batch_size, 16, height // 8, width // 8])}


def _coerce_batch(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if mask.shape[0] == batch_size:
        return mask
    if mask.shape[0] == 1:
        return mask.repeat(batch_size, *([1] * (mask.ndim - 1)))
    return mask[:batch_size]


def _resize_mask(mask: torch.Tensor, height: int, width: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    mask_standard = F.interpolate(
        mask.unsqueeze(1).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    mask_standard = _coerce_batch(mask_standard, batch_size)
    noise_mask = mask_standard.unsqueeze(1)
    return noise_mask, mask_standard


def _resolve_ratio_value(ratio: str, img_01: torch.Tensor | None) -> tuple[int, int]:
    if ratio == "default":
        if img_01 is not None:
            return int(img_01.shape[2]), int(img_01.shape[1])
        return (1, 1)

    try:
        width_text, height_text = ratio.split(":", 1)
        width_ratio = int(width_text)
        height_ratio = int(height_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Unsupported ratio value: {ratio}") from exc

    if width_ratio <= 0 or height_ratio <= 0:
        raise ValueError(f"Ratio must be positive: {ratio}")

    return width_ratio, height_ratio


def _resolve_size_from_ratio(ratio_width: int, ratio_height: int, megapixels: float) -> tuple[int, int]:
    target_pixels = max(megapixels, 0.01) * 1_000_000.0
    aspect_ratio = ratio_width / ratio_height

    width = int(round(math.sqrt(target_pixels * aspect_ratio)))
    height = int(round(math.sqrt(target_pixels / aspect_ratio)))

    width = max(16, round(width / 16) * 16)
    height = max(16, round(height / 16) * 16)
    return width, height


class EasyFlux2KleinCondition:
    @classmethod
    def INPUT_TYPES(cls):
        optional = _DynamicImageInputs(
            {
                "mask": ("MASK", {"forceInput": True}),
                "upscale_method": (
                    UPSCALE_METHODS,
                    {
                        "default": "bilinear",
                        "tooltip": "Interpolation used for scaling reference images and masks.",
                    },
                ),
                "img_01": ("IMAGE", {"forceInput": True}),
            }
        )

        return {
            "required": {
                "conditioning": ("CONDITIONING", {"forceInput": True}),
                "vae": ("VAE", {"forceInput": True}),
                "ratio": (
                    [
                        "default",
                        "1:1",
                        "16:9",
                        "9:16",
                        "4:3",
                        "3:4",
                    ],
                    {
                        "default": "default",
                        "tooltip": "Aspect ratio selector. 'default' follows img_01 if connected, otherwise 1:1.",
                    },
                ),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 8.0,
                        "step": 0.05,
                        "tooltip": "Total pixel budget in megapixels. Default is 1.0 MP.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Latent batch size. Default is 1.",
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "NOISE_MASK", "MASK", "INT", "INT")
    RETURN_NAMES = ("conditioning", "latent", "noise_mask", "mask", "width", "height")
    FUNCTION = "process"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = (
        "Unified Flux2 Klein conditioning helper.\n"
        "Automatically injects reference latents from img_01/img_02/... and builds either:\n"
        "1) empty latent with img_01/default ratio,\n"
        "2) empty latent with a fixed ratio, or\n"
        "3) img_01-based latent with noise mask when mask is connected."
    )

    def process(
        self,
        conditioning,
        vae,
        ratio: str,
        megapixels: float = 1.0,
        batch_size: int = 1,
        upscale_method: str = "bilinear",
        mask: torch.Tensor | None = None,
        img_01: torch.Tensor | None = None,
        **kwargs,
    ):
        image_inputs = self._collect_images(img_01=img_01, **kwargs)
        primary_image = img_01

        positive = conditioning
        scaled_primary = None
        primary_latent = None

        for image_name, image_tensor in image_inputs:
            scaled_image, encoded_latent = self._encode_reference_image(
                image_tensor=image_tensor,
                vae=vae,
                megapixels=megapixels,
                upscale_method=upscale_method,
            )
            positive = _add_reference_latent(positive, encoded_latent)

            if image_name == "img_01":
                scaled_primary = scaled_image
                primary_latent = encoded_latent

        if mask is not None and primary_image is None:
            raise ValueError("mask requires img_01 to be connected.")

        width, height = self._resolve_output_size(
            ratio=ratio,
            megapixels=megapixels,
            primary_image=primary_image,
            scaled_primary=scaled_primary,
        )

        noise_mask = None
        standard_mask = None

        if mask is not None:
            if scaled_primary is None or primary_latent is None:
                raise ValueError("img_01 must be encoded before building a masked latent.")

            width = int(scaled_primary.shape[2])
            height = int(scaled_primary.shape[1])

            samples = primary_latent
            if batch_size > 1:
                samples = samples.repeat(batch_size, 1, 1, 1)

            noise_mask, standard_mask = _resize_mask(
                mask=mask,
                height=height,
                width=width,
                batch_size=batch_size,
            )
            latent = {
                "samples": samples,
                "noise_mask": noise_mask,
            }
        else:
            latent = _make_empty_flux_latent(width=width, height=height, batch_size=batch_size)

        return (positive, latent, noise_mask, standard_mask, width, height)

    def _collect_images(self, img_01: torch.Tensor | None = None, **kwargs) -> list[tuple[str, torch.Tensor]]:
        images = []
        if img_01 is not None:
            images.append(("img_01", img_01))

        for key, value in sorted(kwargs.items(), key=self._sort_image_key):
            if not isinstance(key, str) or not re.fullmatch(r"img_\d{2}", key):
                continue
            if value is None or key == "img_01":
                continue
            images.append((key, value))

        return images

    def _encode_reference_image(self, image_tensor, vae, megapixels: float, upscale_method: str):
        scaled_image = _scale_image_to_megapixels(
            image_tensor,
            megapixels,
            upscale_method,
        )
        scaled_image = _ensure_divisible(scaled_image, divisor=16)
        encoded_latent = vae.encode(scaled_image[:, :, :, :3])
        return scaled_image, encoded_latent

    def _resolve_output_size(
        self,
        ratio: str,
        megapixels: float,
        primary_image: torch.Tensor | None,
        scaled_primary: torch.Tensor | None,
    ) -> tuple[int, int]:
        if ratio == "default" and scaled_primary is not None:
            return int(scaled_primary.shape[2]), int(scaled_primary.shape[1])

        ratio_width, ratio_height = _resolve_ratio_value(ratio, primary_image)
        return _resolve_size_from_ratio(ratio_width, ratio_height, megapixels)

    @staticmethod
    def _sort_image_key(item):
        key = item[0]
        if isinstance(key, str):
            match = re.fullmatch(r"img_(\d{2})", key)
            if match:
                return int(match.group(1))
        return key
