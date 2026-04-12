from __future__ import annotations

from typing import Any


def _extract_reference_control(conditioning) -> dict | None:
    for _cond_tensor, cond_dict in conditioning:
        if not isinstance(cond_dict, dict):
            continue
        reference_control = cond_dict.get("reference_control")
        if isinstance(reference_control, dict):
            return reference_control
    return None


def _scale_reference_tail_tokens(
    context,
    value,
    reference_control: dict,
    global_weight_scale: float,
    include_img_01: bool,
):
    if context is None:
        return context, value

    names = list(reference_control.get("names", []))
    weights = list(reference_control.get("weights", []))
    local_ranges = list(reference_control.get("local_token_ranges", []))
    total_reference_tokens = int(reference_control.get("total_reference_tokens", 0))

    if not names or total_reference_tokens <= 0:
        return context, value

    token_axis = 1
    sequence_length = int(context.shape[token_axis])
    if total_reference_tokens > sequence_length:
        return context, value

    reference_start = sequence_length - total_reference_tokens
    scaled_context = context.clone()
    scaled_value = value.clone() if value is not None else scaled_context

    for name, weight, local_range in zip(names, weights, local_ranges):
        if name == "img_01" and not include_img_01:
            continue

        local_start, local_end = int(local_range[0]), int(local_range[1])
        start = max(0, reference_start + local_start)
        end = min(sequence_length, reference_start + local_end)
        if end <= start:
            continue

        final_weight = float(weight) * float(global_weight_scale)
        if final_weight == 1.0:
            continue

        scaled_context[:, start:end, :] *= final_weight
        scaled_value[:, start:end, :] *= final_weight

    return scaled_context, scaled_value


class EasyFlux2KleinReferenceWeightPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"forceInput": True}),
                "conditioning": ("CONDITIONING", {"forceInput": True}),
            },
            "optional": {
                "global_weight_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 8.0,
                        "step": 0.05,
                        "tooltip": "Global multiplier applied on top of per-reference weights.",
                    },
                ),
                "include_img_01": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If false, img_01 keeps weight 1.0 even if metadata contains another value.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "conditioning", "patch_summary")
    FUNCTION = "patch"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = (
        "Apply per-reference cross-attention weighting using metadata emitted by "
        "EasyFlux2KleinConditionAdvanced. Assumes reference tokens are appended "
        "to the tail of the context sequence."
    )

    def patch(
        self,
        model,
        conditioning,
        global_weight_scale: float = 1.0,
        include_img_01: bool = True,
    ):
        reference_control = _extract_reference_control(conditioning)
        if reference_control is None:
            return (model, conditioning, "No reference_control metadata found.")

        names = list(reference_control.get("names", []))
        if not names:
            return (model, conditioning, "No reference images recorded in reference_control.")

        patched_model = model.clone()

        def attn2_patch(q, context, value, extra_options: dict[str, Any]):
            del extra_options
            return (
                q,
                *_scale_reference_tail_tokens(
                    context=context,
                    value=value,
                    reference_control=reference_control,
                    global_weight_scale=global_weight_scale,
                    include_img_01=include_img_01,
                ),
            )

        patched_model.set_model_attn2_patch(attn2_patch)

        summary_parts = []
        for name, weight in zip(
            reference_control.get("names", []),
            reference_control.get("weights", []),
        ):
            effective = 1.0 if (name == "img_01" and not include_img_01) else float(weight) * float(global_weight_scale)
            summary_parts.append(f"{name}={effective:.3f}")

        summary = "Reference weight patch active (tail-span assumption): " + ", ".join(summary_parts)
        return (patched_model, conditioning, summary)
