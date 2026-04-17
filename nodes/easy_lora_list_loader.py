from __future__ import annotations

import folder_paths
import comfy.sd
import comfy.utils


MAX_LORA_SLOTS = 50
LORA_NONE = "None"
_LOADED_LORA_CACHE: dict[str, tuple[str, object]] = {}


def _lora_choices():
    return [LORA_NONE] + folder_paths.get_filename_list("loras")


def _slot_name(index: int) -> str:
    return f"lora_{index:02d}"


def _strength_name(index: int) -> str:
    return f"strength_model_{index:02d}"


def _load_lora_data(lora_name: str):
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    cached = _LOADED_LORA_CACHE.get(lora_name)
    if cached is not None and cached[0] == lora_path:
        return cached[1]

    lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
    _LOADED_LORA_CACHE[lora_name] = (lora_path, lora_data)
    return lora_data


class EasyLoraListLoader:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model": ("MODEL", {"forceInput": True}),
            "count": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": MAX_LORA_SLOTS,
                    "tooltip": "Number of visible LoRA slots. Hidden slots are cleared.",
                },
            ),
            "index": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": MAX_LORA_SLOTS - 1,
                    "tooltip": "0-based slot index to load.",
                },
            ),
        }

        optional = {}
        for slot_index in range(1, MAX_LORA_SLOTS + 1):
            optional[_slot_name(slot_index)] = (
                _lora_choices(),
                {
                    "default": LORA_NONE,
                    "tooltip": f"LoRA selection for slot {slot_index}.",
                },
            )
            optional[_strength_name(slot_index)] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.05,
                    "tooltip": f"Model strength for slot {slot_index}. Can be overridden by a float input.",
                },
            )

        return {
            "required": required,
            "optional": optional,
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "lora_name")
    FUNCTION = "load"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = (
        "Select one LoRA from a visible list and apply it to the model.\n"
        "The node exposes up to 50 LoRA slots; count only controls widget visibility."
    )

    def load(self, model, count: int, index: int, **kwargs):
        visible_count = int(count)
        selected_index = int(index)

        if selected_index < 0 or selected_index >= visible_count:
            raise ValueError(f"index {selected_index} is out of range for count {visible_count}.")

        slot_index = selected_index + 1
        lora_name = kwargs.get(_slot_name(slot_index), LORA_NONE)
        strength_model = float(kwargs.get(_strength_name(slot_index), 1.0))

        if not lora_name or lora_name == LORA_NONE:
            raise ValueError(f"No LoRA selected in slot {slot_index} (index {selected_index}).")

        if strength_model == 0:
            return (model, lora_name)

        lora_data = _load_lora_data(lora_name)
        loaded_model, _ = comfy.sd.load_lora_for_models(model, None, lora_data, strength_model, 0)
        return (loaded_model, lora_name)
