from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from pathlib import Path


STATE_FILE = Path(__file__).resolve().parent.parent / ".easy_text_batch_state.json"
ENCODING_AUTO = "auto"
ENCODING_CHOICES = (ENCODING_AUTO, "utf-8-sig", "utf-8", "gb18030", "gbk")


def _read_state() -> dict[str, dict[str, object]]:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: dict[str, dict[str, object]]) -> None:
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _decode_text_file(file_path: str, encoding_name: str) -> str:
    if encoding_name == ENCODING_AUTO:
        for candidate in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return Path(file_path).read_text(encoding=candidate)
            except UnicodeDecodeError:
                continue
        return Path(file_path).read_text(encoding="utf-8", errors="replace")

    return Path(file_path).read_text(encoding=encoding_name)


class EasyLoadTextBatch:
    class BatchTextLoader:
        def __init__(self, directory_path: str, label: str, pattern: str):
            self.directory_path = os.path.abspath(directory_path)
            self.label = label
            self.pattern = pattern
            self.text_paths: list[str] = []
            self._load_texts()
            self.text_paths.sort()

            state = _read_state()
            label_state = state.get(label, {})
            stored_directory_path = label_state.get("path")
            stored_pattern = label_state.get("pattern")

            if stored_directory_path != self.directory_path or stored_pattern != pattern:
                self.index = 0
                self._save_index(0)
            else:
                stored_index = label_state.get("index", 0)
                self.index = int(stored_index) if isinstance(stored_index, (int, float)) else 0

        def _load_texts(self) -> None:
            for file_name in glob.glob(
                os.path.join(glob.escape(self.directory_path), self.pattern),
                recursive=True,
            ):
                abs_file_path = os.path.abspath(file_name)
                if os.path.isfile(abs_file_path):
                    self.text_paths.append(abs_file_path)

        def _save_index(self, index: int) -> None:
            state = _read_state()
            state[self.label] = {
                "path": self.directory_path,
                "pattern": self.pattern,
                "index": int(index),
            }
            _write_state(state)

        def _normalize_index(self) -> None:
            if not self.text_paths:
                self.index = 0
                return
            if self.index >= len(self.text_paths):
                self.index = 0
            if self.index < 0:
                self.index = 0

        def _read_by_path(self, text_path: str, encoding_name: str) -> tuple[str, str]:
            return (_decode_text_file(text_path, encoding_name), os.path.basename(text_path))

        def get_text_by_id(self, text_id: int, encoding_name: str) -> tuple[str, str]:
            if text_id < 0 or text_id >= len(self.text_paths):
                raise ValueError(f"Invalid text index {text_id}.")
            return self._read_by_path(self.text_paths[text_id], encoding_name)

        def get_next_text(self, encoding_name: str) -> tuple[str, str]:
            self._normalize_index()
            text_path = self.text_paths[self.index]
            current_index = self.index
            self.index += 1
            if self.index == len(self.text_paths):
                self.index = 0
            self._save_index(self.index)
            return self._read_by_path(self.text_paths[current_index], encoding_name)

        def get_current_text_filename(self) -> str:
            self._normalize_index()
            if not self.text_paths:
                return ""
            return os.path.basename(self.text_paths[self.index])

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["single_text", "incremental_text", "random"],
                    {
                        "default": "single_text",
                        "tooltip": "How to choose the current text file.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Used when mode is random.",
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0x7FFFFFFF,
                        "tooltip": "0-based file index used when mode is single_text.",
                    },
                ),
                "label": (
                    "STRING",
                    {
                        "default": "default",
                        "multiline": False,
                        "tooltip": "State key for incremental_text mode.",
                    },
                ),
                "path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Directory that contains prompt text files.",
                    },
                ),
                "pattern": (
                    "STRING",
                    {
                        "default": "*.txt",
                        "multiline": False,
                        "tooltip": "Glob pattern used to collect text files.",
                    },
                ),
                "encoding": (
                    list(ENCODING_CHOICES),
                    {
                        "default": ENCODING_AUTO,
                        "tooltip": "Text file decoding. auto tries utf-8-sig, utf-8, then gb18030.",
                    },
                ),
                "filename_text_extension": (
                    ["true", "false"],
                    {
                        "default": "true",
                        "tooltip": "Whether filename_text keeps the file extension.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "filename_text")
    FUNCTION = "load_texts"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = (
        "Load one prompt text from a directory batch.\n"
        "Supports single, incremental, and random selection modes."
    )

    def load_texts(
        self,
        mode: str,
        seed: int,
        index: int,
        label: str,
        path: str,
        pattern: str,
        encoding: str,
        filename_text_extension: str,
    ):
        if not path or not os.path.exists(path):
            return ("", "")

        loader = self.BatchTextLoader(path, label, pattern)
        if not loader.text_paths:
            return ("", "")

        if mode == "single_text":
            text, filename = loader.get_text_by_id(int(index), encoding)
        elif mode == "incremental_text":
            text, filename = loader.get_next_text(encoding)
        else:
            random.seed(seed)
            random_index = int(random.random() * len(loader.text_paths))
            text, filename = loader.get_text_by_id(random_index, encoding)

        if filename_text_extension == "false":
            filename = os.path.splitext(filename)[0]

        return (text, filename)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs["mode"] != "single_text":
            return float("NaN")

        path = kwargs["path"]
        if not path or not os.path.exists(path):
            return float("NaN")
        loader = cls.BatchTextLoader(path, kwargs["label"], kwargs["pattern"])
        if not loader.text_paths:
            return float("NaN")
        index = int(kwargs["index"])
        if index < 0 or index >= len(loader.text_paths):
            return float("NaN")
        file_path = loader.text_paths[index]

        digest = hashlib.sha256()
        digest.update(Path(file_path).read_bytes())
        return digest.hexdigest()
