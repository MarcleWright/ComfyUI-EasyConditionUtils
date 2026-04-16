from __future__ import annotations

import re

from .easy_flux2_klein_condition import (
    EasyFlux2KleinCondition,
    _add_reference_latent,
    _ensure_divisible,
    _make_empty_flux_latent,
    _resize_mask,
    _scale_image_to_size,
)


REFERENCE_CONTROL_TYPE = "REFERENCE_CONTROL"
REFERENCE_IMAGE_PATTERN = re.compile(r"img_\d{2}")
REFERENCE_WEIGHT_PATTERN = re.compile(r"img_\d{2}_weight")


class _DynamicImageWeightInputs(dict):
    """Allow ComfyUI to accept dynamic img_nn and img_nn_weight fields."""

    _image_spec = ("IMAGE", {"forceInput": True})
    _weight_spec = (
        "FLOAT",
        {
            "default": 1.0,
            "min": 0.0,
            "max": 8.0,
            "step": 0.05,
            "tooltip": "Per-reference base weight used by EasyFlux2KleinReferenceWeightControl.",
        },
    )

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        if not isinstance(key, str):
            return False
        return bool(REFERENCE_IMAGE_PATTERN.fullmatch(key) or REFERENCE_WEIGHT_PATTERN.fullmatch(key))

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str):
            if REFERENCE_IMAGE_PATTERN.fullmatch(key):
                return self._image_spec
            if REFERENCE_WEIGHT_PATTERN.fullmatch(key):
                return self._weight_spec
        raise KeyError(key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


def _build_reference_control(reference_entries: list[dict]) -> dict:
    names = []
    base_weights = []
    token_counts = []
    token_ranges = []

    offset = 0
    for entry in reference_entries:
        token_count = int(entry["token_count"])
        start = offset
        end = start + token_count
        offset = end

        names.append(entry["name"])
        base_weights.append(float(entry["base_weight"]))
        token_counts.append(token_count)
        token_ranges.append((start, end))

    return {
        "reference_names": names,
        "reference_base_weights": base_weights,
        "reference_token_counts": token_counts,
        "reference_token_ranges": token_ranges,
        "total_reference_tokens": offset,
    }


def _extract_reference_weights(kwargs: dict) -> dict[str, float]:
    weight_map: dict[str, float] = {}
    for key, value in kwargs.items():
        if not isinstance(key, str) or not REFERENCE_WEIGHT_PATTERN.fullmatch(key):
            continue
        image_name = key[:-7]
        weight_map[image_name] = float(value)
    return weight_map


class EasyFlux2KleinConditionAdvanced(EasyFlux2KleinCondition):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        optional = _DynamicImageWeightInputs(dict(base["optional"]))

        return {
            "required": dict(base["required"]),
            "optional": optional,
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "LATENT",
        "NOISE_MASK",
        "MASK",
        "IMAGE",
        "INT",
        "INT",
        REFERENCE_CONTROL_TYPE,
    )
    RETURN_NAMES = (
        "conditioning",
        "latent",
        "noise_mask",
        "mask",
        "img_01_processed",
        "width",
        "height",
        "reference_control",
    )
    DESCRIPTION = (
        "Flux2 Klein conditioning helper with per-reference weight capture.\n"
        "Matches EasyFlux2KleinCondition routing while also emitting a reference_control\n"
        "protocol for downstream reference weight control."
    )

    def process(
        self,
        conditioning,
        vae,
        ratio: str,
        megapixels="default",
        batch_size: int = 1,
        upscale_method: str = "lanczos",
        mask=None,
        img_01=None,
        **kwargs,
    ):
        if mask is not None and img_01 is None:
            raise ValueError("mask requires img_01 to be connected.")

        image_inputs = self._collect_images(img_01=img_01, **kwargs)
        weight_map = _extract_reference_weights(kwargs)

        routing = self._resolve_routing(
            ratio=ratio,
            megapixels_value=megapixels,
            primary_image=img_01,
            has_mask=mask is not None,
        )

        positive = conditioning
        primary_latent = None
        primary_scaled = None
        reference_entries = []

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

            latent_h = int(encoded_latent.shape[2])
            latent_w = int(encoded_latent.shape[3])
            reference_entries.append(
                {
                    "name": image_name,
                    "base_weight": float(weight_map.get(image_name, 1.0)),
                    "token_count": latent_h * latent_w,
                }
            )

            if image_name == "img_01":
                primary_scaled = scaled_image
                primary_latent = encoded_latent

        reference_control = _build_reference_control(reference_entries)

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

        return (
            positive,
            latent,
            noise_mask,
            standard_mask,
            processed_primary_image,
            output_width,
            output_height,
            reference_control,
        )

    def _collect_images(self, img_01=None, **kwargs):
        images = []
        if img_01 is not None:
            images.append(("img_01", img_01))

        image_items = [
            (key, value)
            for key, value in kwargs.items()
            if isinstance(key, str) and REFERENCE_IMAGE_PATTERN.fullmatch(key)
        ]
        for key, value in sorted(image_items, key=self._sort_image_key):
            if value is None or key == "img_01":
                continue
            images.append((key, value))

        return images
