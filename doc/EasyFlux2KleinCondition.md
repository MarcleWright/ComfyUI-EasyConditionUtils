# EasyFlux2KleinCondition

## Goal

`EasyFlux2KleinCondition` is a productized ComfyUI custom node for Flux Kontext / FLUX.2 Klein workflows.

Its purpose is to replace the current "4 independent input workflows + 1 shared backend" setup with one unified node entrypoint.

The node should:

- accept text conditioning, VAE, mask, ratio, megapixels, batch size, and a dynamic list of image inputs
- automatically determine how the task should be routed based on actual input state
- build the shared sampling-stage outputs in a standard format
- hide the current workflow-level switching logic from end users

This node does not expose a manual `mode` selector.
All behavior is inferred from connected inputs and parameter values.

## Node Name

- Internal class name: `EasyFlux2KleinCondition`
- Display name: `Easy Flux2 Klein Condition`
- Category: `EasyConditionUtils`

## Design Principles

- `img_01` is the primary image input
- `img_02+` are additional reference images only
- `mask` always applies to `img_01`
- `ratio` controls aspect ratio
- `megapixels` controls total pixel count
- no `width` / `height` input is exposed
- final outputs should always be normalized into one standard downstream contract

## Inputs

### Required Inputs

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `conditioning` | `CONDITIONING` | none | Positive conditioning input, usually from text encoding / guidance nodes |
| `vae` | `VAE` | none | Used for image encoding and mask-aligned latent preparation |
| `ratio` | dropdown | `default` | Aspect ratio selector |
| `megapixels` | `FLOAT` | `1.0` | Controls final latent/image area |
| `batch_size` | `INT` | `1` | Sampling batch size |

### Optional Inputs

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `mask` | `MASK` | none | If connected, it always targets `img_01` |
| `upscale_method` | dropdown | `bilinear` or implementation default | Used when scaling images and masks |
| `img_01` | `IMAGE` | none | Primary image input |

### Dynamic Optional Inputs

The node supports dynamic image inputs:

- `img_01`
- `img_02`
- `img_03`
- `img_04`
- ...

Rules:

- the node initially shows `img_01`
- when one image input is connected, a new next image input should appear automatically
- `img_01` has special semantic meaning
- `img_02+` are reference-only images

## Input Semantics

### `img_01`

`img_01` is the only image that can affect latent construction behavior.

It is used for:

- deriving aspect ratio when `ratio=default`
- generating base latent when `mask` is connected
- being injected as a reference latent

### `img_02+`

`img_02`, `img_03`, and later inputs are additional reference images only.

They:

- are encoded and appended into reference latent conditioning
- do not control latent type selection
- do not control default aspect ratio
- are never associated with `mask`

### `mask`

`mask` is always interpreted as the mask for `img_01`.

If `mask` is connected:

- latent generation always uses the `SetLatentNoiseMask` route
- `ratio` no longer determines latent type
- the mask must be resized to match the scaled image size used for `img_01` VAE encoding

## Ratio Options

The `ratio` parameter defines aspect ratio only, not final absolute size.

Supported values are expected to include:

- `default`
- `1:1`
- `16:9`
- `9:16`
- `4:3`
- `3:4`
- other implementation-defined fixed ratios

### `ratio=default`

`default` means:

- if `img_01` exists, use the aspect ratio of `img_01`
- if `img_01` does not exist, fall back to `1:1`

## Size Rules

The node does not expose `width` or `height` as inputs.

Final output size is derived from:

- selected aspect ratio
- `megapixels`

Defaults:

- if `megapixels` is not explicitly set, use `1.0`
- if `batch_size` is not explicitly set, use `1`

Expected behavior:

- compute final `width` and `height` from `ratio + megapixels`
- align dimensions to Flux-compatible divisibility requirements
- return the resolved `width` and `height` as outputs

## Automatic Routing Rules

The node must infer behavior from input state instead of a manual mode selector.

### Rule 1: Mask Present

If `mask` is connected:

- always use `img_01` as the base image
- scale `img_01`
- VAE encode the scaled `img_01`
- resize `mask` to the same scaled image size
- create latent through the `SetLatentNoiseMask` route
- ignore `ratio` when deciding latent type

This is the highest-priority route.

### Rule 2: No Mask, `ratio=default`

If `mask` is not connected and `ratio=default`:

- if `img_01` exists, create an empty Flux latent using `img_01`'s aspect ratio
- if `img_01` does not exist, create an empty Flux latent using `1:1`

### Rule 3: No Mask, Fixed Ratio Selected

If `mask` is not connected and `ratio` is a concrete value such as `16:9`, `1:1`, `3:4`, etc.:

- create an empty Flux latent using the selected ratio
- ignore `img_01` aspect ratio for latent sizing

### Rule 4: No Image Inputs

If no image inputs are connected:

- build empty latent from `ratio + megapixels`
- if `ratio=default`, treat it as `1:1`

## Reference Latent Injection Rules

Every connected image input should be appended into the positive conditioning as a reference latent.

That includes:

- `img_01`
- `img_02`
- `img_03`
- ...

However:

- only `img_01` participates in latent route selection
- `img_02+` are reference-only

Recommended processing order:

1. collect connected images in numeric order
2. scale each image as needed
3. VAE encode each image
4. append each encoded latent to `conditioning` as a reference latent

## Outputs

The node should output the following six values:

| Name | Type | Description |
|------|------|-------------|
| `conditioning` | `CONDITIONING` | Positive conditioning with all reference latents appended |
| `latent` | `LATENT` | Final latent passed to the sampler |
| `noise_mask` | mask-like tensor output | Noise-mask representation intended for latent/inpaint sampling flow |
| `mask` | `MASK` | Standard mask output for normal ComfyUI mask-compatible nodes |
| `width` | `INT` | Final resolved width |
| `height` | `INT` | Final resolved height |

## Mask Output Semantics

Two mask-related outputs are required because they serve different downstream uses.

### `noise_mask`

`noise_mask` is intended for latent/inpaint sampling logic.

It should follow the same representation used by `SetLatentNoiseMask`, typically:

- `[B, 1, H, W]`

This is not assumed to be universally compatible with every ordinary `MASK` input in ComfyUI.

### `mask`

`mask` is the standard mask-form output for normal mask-processing nodes.

It should remain in standard ComfyUI mask semantics, such as:

- `[H, W]`
- or `[B, H, W]`

depending on implementation conventions

### Behavior Without Mask Input

If no mask is connected:

- `noise_mask` should be empty / `None` / absent-equivalent according to implementation needs
- `mask` should be empty / `None` / absent-equivalent according to implementation needs

## Latent Construction Summary

There are only three latent construction outcomes in this design.

### A. Empty latent using `img_01` aspect ratio

Used when:

- no `mask`
- `ratio=default`
- `img_01` exists

### B. Empty latent using selected ratio

Used when:

- no `mask`
- `ratio` is a specific non-default value

Also used when:

- no images are connected
- `ratio=default`, which falls back to `1:1`

### C. Latent with noise mask from `img_01`

Used when:

- `mask` is connected

This route always has priority over ratio-based empty latent logic.

## Priority Rules

Priority should be evaluated in this order:

1. if `mask` exists, use `img_01 + resized mask -> SetLatentNoiseMask`
2. else if `ratio=default`, prefer `img_01` aspect ratio if available
3. else use selected fixed ratio
4. if no `img_01` exists and `ratio=default`, fall back to `1:1`

## Suggested Internal Processing Flow

1. read all connected `img_nn` inputs in order
2. identify whether `mask` exists
3. determine effective aspect ratio
4. compute final `width` and `height` from `ratio + megapixels`
5. encode all connected images as reference latents
6. append them into positive conditioning
7. construct final latent according to routing rules
8. build `noise_mask` and `mask` outputs if mask route is active
9. return standardized outputs

## Non-Goals

The node is not intended to:

- expose a manual task mode selector
- expose direct `width` / `height` inputs
- let `img_02+` affect base latent selection
- allow `mask` to target any image other than `img_01`

## Purpose Relative to Existing Workflow

This node is intended to replace the current workflow pattern where four separate front-end paths are manually built for:

- plain image edit
- masked image edit
- reference-image-driven generation
- txt2img

Instead, users should interact with one unified node and let it infer the correct behavior from connected inputs and parameters.
