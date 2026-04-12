from __future__ import annotations

import math
import re

import comfy.utils
import node_helpers
import torch
import torch.nn.functional as F

from .easy_ref_latent import _ensure_divisible, UPSCALE_METHODS


RATIO_OPTIONS = [
    "default",
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "4:1",
]

MEGAPIXEL_OPTIONS = [
    "default",
    "1.00",
    "1.50",
    "2.00",
    "3.00",
    "4.00",
    "6.00",
    "8.00",
]

RATIO_BUCKETS_1MP = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "4:1": (2048, 512),
}

MAX_DEFAULT_IMAGE_MP = 4.2
DEFAULT_IMAGE_CAP_MP = 4.0
FALLBACK_DEFAULT_MP = 1.0


class _DynamicImageInputs(dict):
    """Allow ComfyUI to accept dynamically-created img_nn inputs."""

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


def _align_to_16(value: float) -> int:
    return max(16, int(round(value / 16.0) * 16))


def _image_megapixels(image: torch.Tensor) -> float:
    return (float(image.shape[1]) * float(image.shape[2])) / 1_000_000.0


def _parse_megapixels(value) -> tuple[bool, float]:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "default":
            return True, 0.0
        try:
            parsed = float(text)
        except ValueError as exc:
            raise ValueError(f"Unsupported megapixels value: {value}") from exc
        return False, parsed

    return False, float(value)


def _resolve_bucket_size(ratio: str, megapixels: float) -> tuple[int, int]:
    if ratio not in RATIO_BUCKETS_1MP:
        raise ValueError(f"Unsupported fixed ratio: {ratio}")

    base_width, base_height = RATIO_BUCKETS_1MP[ratio]
    scale = math.sqrt(max(megapixels, 0.01))
    return (_align_to_16(base_width * scale), _align_to_16(base_height * scale))


def _resolve_size_from_ratio_value(width_ratio: int, height_ratio: int, megapixels: float) -> tuple[int, int]:
    target_pixels = max(megapixels, 0.01) * 1_000_000.0
    aspect_ratio = width_ratio / height_ratio

    width = math.sqrt(target_pixels * aspect_ratio)
    height = math.sqrt(target_pixels / aspect_ratio)
    return (_align_to_16(width), _align_to_16(height))


def _resolve_size_from_image_ratio(image: torch.Tensor, megapixels: float) -> tuple[int, int]:
    image_width = int(image.shape[2])
    image_height = int(image.shape[1])
    gcd = math.gcd(image_width, image_height)
    return _resolve_size_from_ratio_value(
        width_ratio=max(1, image_width // gcd),
        height_ratio=max(1, image_height // gcd),
        megapixels=megapixels,
    )


def _make_empty_flux_latent(width: int, height: int, batch_size: int, device=None) -> dict:
    width = max(16, (width // 16) * 16)
    height = max(16, (height // 16) * 16)
    return {
        "samples": torch.zeros([batch_size, 16, height // 16, width // 16], device=device),
    }


def _scale_image_to_size(image: torch.Tensor, width: int, height: int, upscale_method: str) -> torch.Tensor:
    image_bchw = image.movedim(-1, 1)
    scaled = comfy.utils.common_upscale(image_bchw, width, height, upscale_method, "disabled")
    return scaled.movedim(1, -1)


def _coerce_mask_batch(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if mask.shape[0] == batch_size:
        return mask
    if mask.shape[0] == 1:
        return mask.repeat(batch_size, 1, 1)
    return mask[:batch_size]


def _resize_mask(mask: torch.Tensor, height: int, width: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    resized = F.interpolate(
        mask.unsqueeze(1).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    resized = _coerce_mask_batch(resized, batch_size)
    return resized.unsqueeze(1), resized


class EasyFlux2KleinCondition:
    @classmethod
    def INPUT_TYPES(cls):
        optional = _DynamicImageInputs(
            {
                "mask": ("MASK", {"forceInput": True}),
                "upscale_method": (
                    UPSCALE_METHODS,
                    {
                        "default": "lanczos",
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
                    RATIO_OPTIONS,
                    {
                        "default": "default",
                        "tooltip": "default follows img_01 if connected, otherwise falls back to 1:1.",
                    },
                ),
                "megapixels": (
                    MEGAPIXEL_OPTIONS,
                    {
                        "default": "default",
                        "tooltip": "default follows img_01 rules for ratio=default, otherwise fixed ratios use 1.00 MP.",
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

    RETURN_TYPES = ("CONDITIONING", "LATENT", "NOISE_MASK", "MASK", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("conditioning", "latent", "noise_mask", "mask", "img_01_processed", "width", "height")
    FUNCTION = "process"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = (
        "Unified Flux2 Klein conditioning helper.\n"
        "Resolves task routing from inputs, uses standardized ratio buckets, and builds\n"
        "either an empty latent or an img_01+mask latent for sampling."
    )

    def process(
        self,
        conditioning,
        vae,
        ratio: str,
        megapixels="default",
        batch_size: int = 1,
        upscale_method: str = "lanczos",
        mask: torch.Tensor | None = None,
        img_01: torch.Tensor | None = None,
        **kwargs,
    ):
        if mask is not None and img_01 is None:
            raise ValueError("mask requires img_01 to be connected.")

        image_inputs = self._collect_images(img_01=img_01, **kwargs)
        routing = self._resolve_routing(
            ratio=ratio,
            megapixels_value=megapixels,
            primary_image=img_01,
            has_mask=mask is not None,
        )

        positive = conditioning
        primary_latent = None
        primary_scaled = None

        for image_name, image_tensor in image_inputs:
            reference_width, reference_height = self._resolve_reference_size(
                image_name=image_name,
                image_tensor=image_tensor,
                routing=routing,
                has_mask=mask is not None,
            )
            scaled_image = _scale_image_to_size(
                image_tensor,
                width=reference_width,
                height=reference_height,
                upscale_method=upscale_method,
            )
            scaled_image = _ensure_divisible(scaled_image, divisor=16)
            encoded_latent = vae.encode(scaled_image[:, :, :, :3])

            positive = _add_reference_latent(positive, encoded_latent)

            if image_name == "img_01":
                primary_scaled = scaled_image
                primary_latent = encoded_latent

        noise_mask = None
        standard_mask = None
        processed_primary_image = primary_scaled if primary_scaled is not None else None

        if routing["use_mask_latent"]:
            if primary_scaled is None or primary_latent is None:
                raise ValueError("img_01 must be available before building a masked latent.")

            latent_width = int(primary_scaled.shape[2])
            latent_height = int(primary_scaled.shape[1])
            samples = primary_latent
            if batch_size > 1:
                samples = samples.repeat(batch_size, 1, 1, 1)

            noise_mask, standard_mask = _resize_mask(
                mask=mask,
                height=latent_height,
                width=latent_width,
                batch_size=batch_size,
            )
            latent = {
                "samples": samples,
                "noise_mask": noise_mask,
            }
            output_width = latent_width
            output_height = latent_height
        else:
            output_width = routing["output_width"]
            output_height = routing["output_height"]
            latent = _make_empty_flux_latent(
                width=output_width,
                height=output_height,
                batch_size=batch_size,
                device=img_01.device if img_01 is not None else None,
            )

        return (positive, latent, noise_mask, standard_mask, processed_primary_image, output_width, output_height)

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

    def _resolve_routing(
        self,
        ratio: str,
        megapixels_value,
        primary_image: torch.Tensor | None,
        has_mask: bool,
    ) -> dict:
        is_default_mp, parsed_megapixels = _parse_megapixels(megapixels_value)

        # Root-cause rule: mask route always uses img_01 ratio, independent of fixed-ratio bucket choice.
        if has_mask:
            effective_mp = self._resolve_default_ratio_megapixels(
                is_default_mp=is_default_mp,
                parsed_megapixels=parsed_megapixels,
                primary_image=primary_image,
            )
            width, height = self._resolve_default_ratio_size(
                primary_image=primary_image,
                effective_megapixels=effective_mp,
                is_default_megapixels=is_default_mp,
            )
            return {
                "use_mask_latent": True,
                "output_width": width,
                "output_height": height,
                "effective_megapixels": effective_mp,
                "reference_megapixels": effective_mp,
            }

        if ratio != "default":
            effective_mp = FALLBACK_DEFAULT_MP if is_default_mp else parsed_megapixels
            width, height = _resolve_bucket_size(ratio, effective_mp)
            return {
                "use_mask_latent": False,
                "output_width": width,
                "output_height": height,
                "effective_megapixels": effective_mp,
                "reference_megapixels": effective_mp,
            }

        effective_mp = self._resolve_default_ratio_megapixels(
            is_default_mp=is_default_mp,
            parsed_megapixels=parsed_megapixels,
            primary_image=primary_image,
        )
        width, height = self._resolve_default_ratio_size(
            primary_image=primary_image,
            effective_megapixels=effective_mp,
            is_default_megapixels=is_default_mp,
        )
        return {
            "use_mask_latent": False,
            "output_width": width,
            "output_height": height,
            "effective_megapixels": effective_mp,
            "reference_megapixels": effective_mp,
        }

    def _resolve_default_ratio_megapixels(
        self,
        is_default_mp: bool,
        parsed_megapixels: float,
        primary_image: torch.Tensor | None,
    ) -> float:
        if not is_default_mp:
            return parsed_megapixels
        if primary_image is None:
            return FALLBACK_DEFAULT_MP
        if _image_megapixels(primary_image) <= MAX_DEFAULT_IMAGE_MP:
            return _image_megapixels(primary_image)
        return DEFAULT_IMAGE_CAP_MP

    def _resolve_default_ratio_size(
        self,
        primary_image: torch.Tensor | None,
        effective_megapixels: float,
        is_default_megapixels: bool,
    ) -> tuple[int, int]:
        if primary_image is None:
            return _resolve_bucket_size("1:1", effective_megapixels)

        if is_default_megapixels and _image_megapixels(primary_image) <= MAX_DEFAULT_IMAGE_MP:
            return (int(primary_image.shape[2]), int(primary_image.shape[1]))

        return _resolve_size_from_image_ratio(primary_image, effective_megapixels)

    def _resolve_reference_size(
        self,
        image_name: str,
        image_tensor: torch.Tensor,
        routing: dict,
        has_mask: bool,
    ) -> tuple[int, int]:
        if image_name == "img_01" and has_mask:
            return (routing["output_width"], routing["output_height"])

        if image_name == "img_01" and routing["output_width"] == int(image_tensor.shape[2]) and routing["output_height"] == int(image_tensor.shape[1]):
            return (routing["output_width"], routing["output_height"])

        reference_mp = routing["reference_megapixels"]
        return _resolve_size_from_image_ratio(image_tensor, reference_mp)

    @staticmethod
    def _sort_image_key(item):
        key = item[0]
        if isinstance(key, str):
            match = re.fullmatch(r"img_(\d{2})", key)
            if match:
                return int(match.group(1))
        return key
