from __future__ import annotations


MAX_TEXT_SLOTS = 50


def _slot_name(index: int) -> str:
    return f"text_{index:02d}"


class EasyTextListSelector:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "count": (
                "INT",
                {
                    "default": 1,
                    "min": 1,
                    "max": MAX_TEXT_SLOTS,
                    "tooltip": "Number of visible text slots.",
                },
            ),
            "index": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": MAX_TEXT_SLOTS - 1,
                    "tooltip": "0-based slot index to output.",
                },
            ),
        }

        optional = {}
        for slot_index in range(1, MAX_TEXT_SLOTS + 1):
            optional[_slot_name(slot_index)] = (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "tooltip": f"Text value for slot {slot_index}.",
                },
            )

        return {
            "required": required,
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "select"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = (
        "Select one string from a visible text list.\n"
        "The node exposes up to 50 text slots; count only controls widget visibility."
    )

    def select(self, count: int, index: int, **kwargs):
        visible_count = int(count)
        selected_index = int(index)

        if selected_index < 0 or selected_index >= visible_count:
            raise ValueError(f"index {selected_index} is out of range for count {visible_count}.")

        slot_index = selected_index + 1
        text_value = kwargs.get(_slot_name(slot_index), "")
        if text_value is None or str(text_value) == "":
            raise ValueError(f"No text set in slot {slot_index} (index {selected_index}).")

        return (str(text_value),)
