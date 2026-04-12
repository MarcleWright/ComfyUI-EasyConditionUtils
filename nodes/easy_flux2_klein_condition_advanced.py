from __future__ import annotations

import re

import node_helpers

from .easy_flux2_klein_condition import (
    EasyFlux2KleinCondition,
    _add_reference_latent,
    _ensure_divisible,
    _parse_megapixels,
    _resize_mask,
    _scale_image_to_size,
    _make_empty_flux_latent,
    MEGAPIXEL_OPTIONS,
    RATIO_OPTIONS,
)


def _parse_reference_weights(weights_text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not weights_text:
        return weights

    for raw_part in weights_text.split(","):
        part = raw_part.strip()
        if not part:
            continue

        if "=" not in part:
            raise ValueError(
                "reference_weights must use 'img_01=1.0,img_02=0.8' format."
            )

        name, raw_value = part.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"img_\d{2}", name):
            raise ValueError(f"Unsupported reference weight key: {name}")

        try:
            weights[name] = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid weight for {name}: {raw_value}") from exc

    return weights


def _build_reference_control(reference_entries: list[dict]) -> dict:
    local_ranges = []
    names = []
    weights = []
    token_counts = []
    latent_shapes = []

    offset = 0
    for entry in reference_entries:
        token_count = int(entry["token_count"])
        start = offset
        end = start + token_count
        offset = end

        local_ranges.append((start, end))
        names.append(entry["name"])
        weights.append(float(entry["weight"]))
        token_counts.append(token_count)
        latent_shapes.append((int(entry["latent_h"]), int(entry["latent_w"])))

    return {
        "mode": "local_reference_spans",
        "names": names,
        "weights": weights,
        "token_counts": token_counts,
        "latent_shapes": latent_shapes,
        "local_token_ranges": local_ranges,
        "total_reference_tokens": offset,
    }


def _add_reference_control(conditioning: list, reference_control: dict) -> list:
    return node_helpers.conditioning_set_values(
        conditioning,
        {
            "reference_control": reference_control,
            "reference_control_version": 1,
        },
        append=False,
    )


def _make_reference_summary(reference_control: dict) -> str:
    if not reference_control["names"]:
        return "No reference images connected."

    parts = []
    for name, weight, token_count, token_range in zip(
        reference_control["names"],
        reference_control["weights"],
        reference_control["token_counts"],
        reference_control["local_token_ranges"],
    ):
        parts.append(
            f"{name}: weight={weight:.3f}, tokens={token_count}, local_span={token_range[0]}:{token_range[1]}"
        )
    return " | ".join(parts)


class EasyFlux2KleinConditionAdvanced(EasyFlux2KleinCondition):
    @classmethod
    def INPUT_TYPES(cls):
        base = super().INPUT_TYPES()
        required = dict(base["required"])
        optional = dict(base["optional"])

        required["reference_weights"] = (
            "STRING",
            {
                "default": "",
                "multiline": False,
                "tooltip": (
                    "Optional per-reference weights, e.g. "
                    "'img_01=1.0,img_02=0.8,img_03=1.2'. "
                    "This node stores metadata for a later attention patch."
                ),
            },
        )
        optional["default_reference_weight"] = (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 8.0,
                "step": 0.05,
                "tooltip": "Fallback weight when an image is not explicitly listed.",
            },
        )

        return {
            "required": required,
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
        "STRING",
    )
    RETURN_NAMES = (
        "conditioning",
        "latent",
        "noise_mask",
        "mask",
        "img_01_processed",
        "width",
        "height",
        "reference_summary",
    )
    DESCRIPTION = (
        "Advanced Flux2 Klein conditioning helper.\n"
        "Builds the same routing outputs as EasyFlux2KleinCondition, and also stores\n"
        "per-reference weight and local token-span metadata for future attention hooks."
    )

    def process(
        self,
        conditioning,
        vae,
        ratio: str,
        megapixels="default",
        batch_size: int = 1,
        reference_weights: str = "",
        upscale_method: str = "lanczos",
        default_reference_weight: float = 1.0,
        mask=None,
        img_01=None,
        **kwargs,
    ):
        if mask is not None and img_01 is None:
            raise ValueError("mask requires img_01 to be connected.")

        _parse_megapixels(megapixels)
        weight_map = _parse_reference_weights(reference_weights)
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
                    "weight": float(weight_map.get(image_name, default_reference_weight)),
                    "latent_h": latent_h,
                    "latent_w": latent_w,
                    "token_count": latent_h * latent_w,
                }
            )

            if image_name == "img_01":
                primary_scaled = scaled_image
                primary_latent = encoded_latent

        reference_control = _build_reference_control(reference_entries)
        positive = _add_reference_control(positive, reference_control)

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

        summary = _make_reference_summary(reference_control)
        return (
            positive,
            latent,
            noise_mask,
            standard_mask,
            processed_primary_image,
            output_width,
            output_height,
            summary,
        )
