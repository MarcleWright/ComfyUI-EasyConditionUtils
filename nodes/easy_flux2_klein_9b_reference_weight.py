from __future__ import annotations

from typing import Any

from .easy_flux2_klein_condition_advanced import REFERENCE_CONTROL_TYPE
from .easy_flux2_klein_reference_weight import _apply_reference_weight_patch, _validate_reference_control


class EasyFlux2Klein9BReferenceWeightControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"forceInput": True}),
                "conditioning": ("CONDITIONING", {"forceInput": True}),
                "reference_control": (REFERENCE_CONTROL_TYPE, {"forceInput": True}),
                "patch_mode": (
                    ["debug_only", "attn1_kv", "attn1_output"],
                    {
                        "default": "debug_only",
                        "tooltip": "9B test mode. debug_only prints diagnostics; attn1_kv uses the 4B path; attn1_output scales attention output spans.",
                    },
                ),
                "debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Print FLUX.2 Klein 9B attention patch diagnostics.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "patch"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = (
        "9B development copy of Easy Flux2 Klein Reference Weight Control.\n"
        "This node is intentionally separated from the stable reference weight node\n"
        "so FLUX.2 Klein 9B-specific patch behavior can be tested safely."
    )

    def patch(
        self,
        model,
        conditioning,
        reference_control,
        patch_mode,
        debug,
    ):
        names, base_weights, token_counts, token_ranges, total_reference_tokens = _validate_reference_control(reference_control)

        patched_model = model.clone()
        debug_state = {
            "attn1_printed": False,
            "attn1_output_printed": False,
        }

        if debug:
            print("[EasyFlux2Klein9BReferenceWeightControl] patch installed")
            print(f"  patch_mode={patch_mode}")
            print(f"  model_type={type(model).__name__}")
            print(f"  inner_model_type={type(getattr(model, 'model', None)).__name__}")
            print(f"  reference_names={names}")
            print(f"  reference_base_weights={base_weights}")
            print(f"  reference_token_counts={token_counts}")
            print(f"  reference_token_ranges={token_ranges}")
            print(f"  total_reference_tokens={total_reference_tokens}")

        def attn1_patch(q, k, v, extra_options: dict[str, Any] | None = None, **kwargs):
            del kwargs
            options = extra_options or {}
            if debug and not debug_state["attn1_printed"]:
                runtime_counts = options.get("reference_image_num_tokens", [])
                total_runtime_ref = sum(int(value) for value in runtime_counts) if runtime_counts else 0
                print("[EasyFlux2Klein9BReferenceWeightControl] attn1_patch entered")
                print(f"  q.shape={tuple(q.shape)}")
                print(f"  k.shape={tuple(k.shape)}")
                print(f"  v.shape={tuple(v.shape)}")
                print(f"  extra_options.keys={sorted(options.keys())}")
                print(f"  block_index={options.get('block_index', None)}")
                print(f"  reference_image_num_tokens={runtime_counts}")
                print(f"  reference_control_token_counts={token_counts}")
                print(f"  total_runtime_reference_tokens={total_runtime_ref}")
                if runtime_counts:
                    token_offset = 0
                    for index, runtime_count in enumerate(runtime_counts[: len(names)]):
                        runtime_count = int(runtime_count)
                        seq_start = -total_runtime_ref + token_offset
                        seq_end = seq_start + runtime_count
                        print(
                            "  ref_span"
                            f"[{index}] name={names[index]} weight={base_weights[index]} "
                            f"runtime_tokens={runtime_count} seq_start={seq_start} seq_end={seq_end}"
                        )
                        token_offset += runtime_count
                debug_state["attn1_printed"] = True
            if patch_mode == "attn1_kv":
                return _apply_reference_weight_patch(
                    q=q,
                    k=k,
                    v=v,
                    reference_control=reference_control,
                    extra_options=options,
                )
            return {}

        def attn1_output_patch(attn, extra_options: dict[str, Any] | None = None):
            options = extra_options or {}
            runtime_counts = options.get("reference_image_num_tokens", [])
            total_runtime_ref = sum(int(value) for value in runtime_counts) if runtime_counts else total_reference_tokens
            if debug and not debug_state["attn1_output_printed"]:
                print("[EasyFlux2Klein9BReferenceWeightControl] attn1_output_patch entered")
                print(f"  attn.shape={tuple(attn.shape)}")
                print(f"  extra_options.keys={sorted(options.keys())}")
                patches = options.get("patches", {})
                print(f"  transformer_patch_keys={sorted(patches.keys()) if isinstance(patches, dict) else type(patches).__name__}")
                print(f"  block_index={options.get('block_index', None)}")
                print(f"  block_type={options.get('block_type', None)}")
                print(f"  img_slice={options.get('img_slice', None)}")
                print(f"  reference_image_num_tokens={runtime_counts}")
                print(f"  output_total_reference_tokens={total_runtime_ref}")
                token_offset = 0
                counts_for_debug = [int(value) for value in runtime_counts[: len(names)]] if runtime_counts else token_counts
                for index, token_count in enumerate(counts_for_debug):
                    seq_start = -total_runtime_ref + token_offset
                    seq_end = seq_start + token_count
                    print(
                        "  output_ref_span"
                        f"[{index}] name={names[index]} weight={base_weights[index]} "
                        f"tokens={token_count} seq_start={seq_start} seq_end={seq_end}"
                    )
                    token_offset += token_count
                debug_state["attn1_output_printed"] = True
            if patch_mode != "attn1_output":
                return attn

            if total_runtime_ref <= 0:
                return attn

            counts = [int(value) for value in runtime_counts[: len(names)]] if runtime_counts else token_counts
            if len(counts) != len(names):
                return attn

            scaled_attn = attn.clone()
            token_offset = 0
            for weight, token_count in zip(base_weights, counts):
                if token_count <= 0 or weight == 1.0:
                    token_offset += token_count
                    continue

                seq_start = -total_runtime_ref + token_offset
                seq_end = seq_start + token_count
                seq_end_idx = None if seq_end == 0 else seq_end
                scaled_attn[:, seq_start:seq_end_idx, :] *= float(weight)
                token_offset += token_count
            return scaled_attn

        patched_model.set_model_attn1_patch(attn1_patch)
        patched_model.set_model_attn1_output_patch(attn1_output_patch)
        return (patched_model, conditioning)
