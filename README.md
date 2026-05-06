# ComfyUI_EasyConditionUtils

Custom nodes for ComfyUI focused on FLUX.2 Klein / Flux Kontext conditioning workflows.

The main node in the current version is `EasyFlux2KleinCondition`, a unified entrypoint that replaces separate edit / masked edit / reference generation style front-end workflows with one normalized conditioning node.

## Main Node

### Easy Flux2 Klein Condition

Category: `EasyConditionUtils`

What it does:

- accepts dynamic image inputs: `img_01`, `img_02`, `img_03`, ...
- appends all connected images as reference latents
- uses `img_01` as the default size reference image
- supports mask-based latent construction when `mask` is connected
- supports fixed ratio buckets for common generation aspect ratios
- outputs:
  - `conditioning`
  - `latent`
  - `noise_mask`
  - `mask`
  - `img_01_processed`
  - `width`
  - `height`

### Routing Rules

- If `mask` is connected, the node builds a masked latent from `img_01`.
- If `ratio != default`, the node uses the fixed bucket table.
- If `ratio = default`, the node follows `img_01` ratio when present, otherwise falls back to `1:1`.

### Megapixels Rules

- Fixed ratio + `megapixels=default` -> treated as `1.00`
- `ratio=default` + no `img_01` -> treated as `1.00`
- `ratio=default` + `img_01 <= 4.2 MP` -> uses original `img_01` size
- `ratio=default` + `img_01 > 4.2 MP` -> capped to `4.00 MP` while preserving image ratio

### Fixed 1MP Buckets

| Ratio | Width | Height |
|---|---:|---:|
| `1:1` | 1024 | 1024 |
| `4:3` | 1152 | 864 |
| `3:4` | 864 | 1152 |
| `3:2` | 1248 | 832 |
| `2:3` | 832 | 1248 |
| `5:4` | 1120 | 896 |
| `4:5` | 896 | 1120 |
| `16:9` | 1344 | 768 |
| `9:16` | 768 | 1344 |
| `2:1` | 1440 | 720 |
| `1:2` | 720 | 1440 |
| `21:9` | 1568 | 672 |
| `9:21` | 672 | 1568 |
| `4:1` | 2048 | 512 |
| `1:4` | 512 | 2048 |

`default` is resolved dynamically and is not part of the fixed table.

## Other Nodes

### Easy LoRA List Loader

Keeps a prepared list of LoRA candidates inside one node and applies exactly
one selected LoRA to the incoming `MODEL`.

What it does:

- exposes up to 50 LoRA slots
- uses `count` to control how many slots are visible
- uses `index` to select exactly one active LoRA
- applies the selected LoRA to `model`
- outputs the patched `model` and the selected `lora_name`

### Easy Load Text Batch

Loads one prompt text file from a directory batch and outputs:

- `text`
- `filename_text`

What it does:

- scans a folder with `path + pattern`
- supports `single_text`, `incremental_text`, and `random`
- loops in `incremental_text` mode
- preserves raw file content, including empty files and line breaks

### Easy Flux2 Klein Condition Advanced

An advanced variant of `Easy Flux2 Klein Condition` that keeps the same
conditioning / latent routing behavior, while also:

- adding per-reference weight widgets alongside dynamic `img_01`, `img_02`, ...
- outputting `reference_control` for downstream reference weight control

Its standard outputs remain aligned with the base node, with one extra output:

- `reference_control`

### Easy Flux2 Klein Reference Weight Control

Consumes:

- `model`
- `conditioning`
- `reference_control`

and outputs:

- patched `model`

This node applies reference-specific attention weighting from
`reference_control` and is intended to be connected in parallel with the
conditioning / latent path, then merged at the sampler.

Typical wiring:

```text
EasyFlux2KleinConditionAdvanced
    -> conditioning -----> sampler
    -> latent -----------> sampler
    -> reference_control -> EasyFlux2KleinReferenceWeightControl

model -------------------> EasyFlux2KleinReferenceWeightControl
EasyFlux2KleinReferenceWeightControl
    -> model -----------> sampler
```

### Easy Reference Latent Apply (Batch)

Encodes an image batch and appends each image as a reference latent.

### Easy Reference Latent (from Latent)

Appends pre-encoded latent items as reference latents.

### Easy Clear Reference Latents

Removes `reference_latents` from conditioning.

### Easy Count Reference Latents

Reports how many reference latents are currently attached to conditioning.

## Installation

Copy this folder into:

```text
ComfyUI/custom_nodes/ComfyUI_EasyConditionUtils
```

Then restart ComfyUI.

## Docs

- Node spec: [doc/EasyFlux2KleinCondition.md](doc/EasyFlux2KleinCondition.md)
- Advanced + reference weight control: [doc/EasyFlux2KleinCondition_advanced_reference_weight_ZH.md](doc/EasyFlux2KleinCondition_advanced_reference_weight_ZH.md)
- 9B reference weight control diagnostics: [doc/EasyFlux2Klein9BReferenceWeightControl.md](doc/EasyFlux2Klein9BReferenceWeightControl.md)
- Easy LoRA List Loader: [doc/EasyLoraListLoader.md](doc/EasyLoraListLoader.md)
- Easy Load Text Batch: [doc/EasyLoadTextBatch.md](doc/EasyLoadTextBatch.md)
- Bucket list: [doc/EasyFlux2KleinCondition_bucket_list.md](doc/EasyFlux2KleinCondition_bucket_list.md)
- Workflow example: [doc/flux2klein_switch_route_v2.json](doc/flux2klein_switch_route_v2.json)

## Acknowledgements

- `WASasquatchm` / `WAS Node Suite`
  - `Easy Load Text Batch` is based on the batch-loading structure used by
    `Load Image Batch`, adapted here from image files to text prompt files.
- `capitan01R` / `ComfyUI-Flux2Klein-Enhancer`
  - the current `Easy Flux2 Klein Reference Weight Control` direction draws
    from the runtime patching approach explored in that project.

## Notes For External Frontends

The current node can already be driven by other frontends through workflow JSON editing, as long as they write legal widget values for fields such as `ratio` and `megapixels`.

## License

MIT
