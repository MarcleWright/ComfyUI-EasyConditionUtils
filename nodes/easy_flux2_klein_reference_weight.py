from __future__ import annotations

from typing import Any

from .easy_flux2_klein_condition_advanced import REFERENCE_CONTROL_TYPE


def _validate_reference_control(reference_control: dict) -> tuple[list[str], list[float], list[int], list[tuple[int, int]], int]:
    names = list(reference_control.get("reference_names", []))
    base_weights = [float(value) for value in reference_control.get("reference_base_weights", [])]
    token_counts = [int(value) for value in reference_control.get("reference_token_counts", [])]
    token_ranges = [tuple(item) for item in reference_control.get("reference_token_ranges", [])]
    total_reference_tokens = int(reference_control.get("total_reference_tokens", 0))

    if not names:
        raise ValueError("reference_control does not contain any reference_names.")
    if len(names) != len(base_weights) or len(names) != len(token_counts) or len(names) != len(token_ranges):
        raise ValueError("reference_control field lengths do not match.")

    expected_start = 0
    for token_count, (start, end) in zip(token_counts, token_ranges):
        start = int(start)
        end = int(end)
        if start != expected_start or end < start or end - start != token_count:
            raise ValueError("reference_control contains non-contiguous token ranges.")
        expected_start = end

    if total_reference_tokens != expected_start:
        raise ValueError("reference_control total_reference_tokens does not match token ranges.")

    normalized_ranges = [(int(start), int(end)) for start, end in token_ranges]
    return names, base_weights, token_counts, normalized_ranges, total_reference_tokens


def _apply_reference_weight_patch(
    q,
    k,
    v,
    reference_control: dict,
    extra_options: dict[str, Any],
):
    ref_token_counts_runtime = extra_options.get("reference_image_num_tokens", [])
    if not ref_token_counts_runtime:
        return {}

    names, base_weights, token_counts, _token_ranges, _total_reference_tokens = _validate_reference_control(reference_control)
    if len(ref_token_counts_runtime) < len(names):
        return {}

    normalized_runtime_counts = [int(value) for value in ref_token_counts_runtime[: len(names)]]
    if normalized_runtime_counts != token_counts:
        return {}

    total_ref = sum(normalized_runtime_counts)
    if total_ref <= 0:
        return {}

    scaled_k = k.clone()
    scaled_v = v.clone()

    token_offset = 0
    for weight, token_count in zip(base_weights, normalized_runtime_counts):
        if token_count <= 0 or weight == 1.0:
            token_offset += token_count
            continue

        seq_start = -total_ref + token_offset
        seq_end = seq_start + token_count
        seq_end_idx = None if seq_end == 0 else seq_end
        scaled_k[:, :, seq_start:seq_end_idx, :] *= float(weight)
        scaled_v[:, :, seq_start:seq_end_idx, :] *= float(weight)
        token_offset += token_count

    return {
        "q": q,
        "k": scaled_k,
        "v": scaled_v,
    }


class EasyFlux2KleinReferenceWeightControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"forceInput": True}),
                "conditioning": ("CONDITIONING", {"forceInput": True}),
                "reference_control": (REFERENCE_CONTROL_TYPE, {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "patch"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = (
        "Apply per-reference weight control from reference_control.\n"
        "Consumes the local reference span protocol emitted by EasyFlux2KleinConditionAdvanced\n"
        "and patches the model used by the downstream sampler."
    )

    def patch(
        self,
        model,
        conditioning,
        reference_control,
    ):
        _validate_reference_control(reference_control)

        patched_model = model.clone()

        def attn1_patch(q, k, v, extra_options: dict[str, Any] | None = None, **kwargs):
            del kwargs
            return _apply_reference_weight_patch(
                q=q,
                k=k,
                v=v,
                reference_control=reference_control,
                extra_options=extra_options or {},
            )

        patched_model.set_model_attn1_patch(attn1_patch)
        return (patched_model, conditioning)
