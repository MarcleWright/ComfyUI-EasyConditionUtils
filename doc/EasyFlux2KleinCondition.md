# EasyFlux2KleinCondition

## Goal

`EasyFlux2KleinCondition` is a unified ComfyUI custom node for Flux Kontext / FLUX.2 Klein workflows.

It replaces the old pattern of multiple separate workflow front-ends with one node that:

- accepts text conditioning, VAE, ratio, megapixels, mask, batch size, and dynamic image inputs
- decides routing automatically from connected inputs
- standardizes latent preparation for both empty-latent and masked-latent cases
- appends all connected images as reference latents
- exposes the resolved downstream outputs in one place

This node does not use a manual `mode` selector.

## Node Identity

| Field | Value |
|------|------|
| Internal class | `EasyFlux2KleinCondition` |
| Display name | `Easy Flux2 Klein Condition` |
| Category | `EasyConditionUtils` |

## Core Principles

- `img_01` is the only primary image
- `img_02+` are reference-only images
- `mask` always belongs to `img_01`
- fixed ratio always uses standardized buckets
- `ratio=default` is the only mode that can follow `img_01`
- `megapixels=default` has special behavior only for `ratio=default`
- the node outputs the processed `img_01` that actually determined latent sizing

## Inputs

### Required Inputs

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `conditioning` | `CONDITIONING` | none | Positive conditioning input |
| `vae` | `VAE` | none | Used to encode reference images and masked base image |
| `ratio` | dropdown | `default` | Aspect ratio selector |
| `megapixels` | dropdown | `default` | Target MP mode |
| `batch_size` | `INT` | `1` | Latent batch size |

### Optional Inputs

| Name | Type | Default | Notes |
|------|------|---------|-------|
| `mask` | `MASK` | none | Always interpreted as the mask for `img_01` |
| `upscale_method` | dropdown | `lanczos` | Used for image and mask resizing |
| `img_01` | `IMAGE` | none | Primary image input |

### Dynamic Optional Image Inputs

The node supports:

- `img_01`
- `img_02`
- `img_03`
- `img_04`
- ...

Behavior:

- the node starts with `img_01`
- connecting the last visible image input creates the next image input
- disconnecting image inputs trims extra trailing empty image inputs
- `img_01` is primary
- `img_02+` are additional reference images only

## Ratio Options

The current ratio list is:

- `default`
- `1:1`
- `16:9`
- `9:16`
- `4:3`
- `3:4`
- `3:2`
- `2:3`
- `4:1`

## 1MP Bucket Table

All fixed ratios are based on this standardized `1MP` bucket table.
All values are divisible by `16`.

| Ratio | Width | Height |
|------|------:|-------:|
| `1:1` | 1024 | 1024 |
| `16:9` | 1344 | 768 |
| `9:16` | 768 | 1344 |
| `4:3` | 1152 | 864 |
| `3:4` | 864 | 1152 |
| `3:2` | 1248 | 832 |
| `2:3` | 832 | 1248 |
| `4:1` | 2048 | 512 |

## Megapixels Options

The current megapixels list is:

- `default`
- `1.00`
- `1.50`
- `2.00`
- `3.00`
- `4.00`
- `6.00`
- `8.00`

## Megapixels Rules

### Fixed Ratio

If `ratio != default`:

- fixed ratio always wins
- `img_01` does not change the final output ratio
- `megapixels=default` is treated as `1.00`
- all fixed-ratio sizes are derived from the `1MP` bucket table

Computation:

1. select the `1MP` bucket for that ratio
2. compute `scale = sqrt(target_mp)`
3. scale width and height by that factor
4. align width and height to multiples of `16`

### Default Ratio

If `ratio=default`:

- if `img_01` exists, use the aspect ratio of `img_01`
- if `img_01` does not exist, fall back to `1:1`

### `megapixels=default` with `ratio=default`

This follows the special rule set below:

1. if there is no `img_01`, it behaves as `1.00 MP`
2. if `img_01` exists and its size is `<= 4.2 MP`, the node uses the original `img_01` width and height directly
3. if `img_01` exists and its size is `> 4.2 MP`, the node caps to `4.00 MP` while keeping the aspect ratio of `img_01`

This behavior applies to both empty-latent and masked-latent routes.

## Input Semantics

### `img_01`

`img_01` is used for:

- `ratio=default` aspect-ratio inference
- base latent creation when `mask` exists
- reference latent injection
- processed `img_01` output

### `img_02+`

`img_02+` are used only for:

- reference latent injection

They do not:

- affect final latent route selection
- affect fixed ratio bucket sizing
- receive `mask`

### `mask`

If `mask` is connected:

- the node always uses the masked-latent route
- `img_01` must be connected
- `img_01` is resized to the final latent size
- the mask is resized to match the same pixel size
- the latent is built from encoded `img_01` plus `noise_mask`

## Routing Rules

### Rule 1: Mask Route

If `mask` is connected:

- use `img_01` as base image
- determine final size from the `ratio=default` family of rules
- resize `img_01` to that final size
- VAE encode the resized `img_01`
- resize `mask` to the same size
- output latent with `noise_mask`

This route has the highest priority.

### Rule 2: Empty Latent with Fixed Ratio

If `mask` is not connected and `ratio != default`:

- ignore `img_01` for final latent sizing
- use the fixed ratio bucket system
- use `1.00 MP` if `megapixels=default`

### Rule 3: Empty Latent with Default Ratio

If `mask` is not connected and `ratio=default`:

- if `img_01` exists, follow `img_01` ratio
- if `img_01` does not exist, fall back to `1:1`
- if `megapixels=default`, use the special default-MP rules above

## Reference Latent Rules

Every connected image input is appended to positive conditioning as a reference latent:

- `img_01`
- `img_02`
- `img_03`
- ...

Processing order is numeric:

1. `img_01`
2. `img_02`
3. `img_03`
4. ...

Reference image sizing rules:

- if `img_01` is being used for the masked route, it is resized to the final latent size
- otherwise each image is resized by aspect ratio using the resolved reference megapixels policy

## Outputs

The node outputs the following seven values:

| Name | Type | Description |
|------|------|-------------|
| `conditioning` | `CONDITIONING` | Positive conditioning with all reference latents appended |
| `latent` | `LATENT` | Final latent used for sampling |
| `noise_mask` | mask-like tensor | Mask representation for latent/inpaint flow |
| `mask` | `MASK` | Standard ComfyUI mask output |
| `img_01_processed` | `IMAGE` | The processed version of `img_01` actually used for latent sizing / encoding |
| `width` | `INT` | Final resolved output width |
| `height` | `INT` | Final resolved output height |

## Output Semantics

### `noise_mask`

Used for latent / inpaint flow, usually shaped like:

- `[B, 1, H, W]`

### `mask`

Used for standard ComfyUI mask-compatible nodes, usually shaped like:

- `[H, W]`
- or `[B, H, W]`

### `img_01_processed`

This is not recomputed separately.
It should be the exact internal `img_01` tensor that was already resized and used for latent preparation.

That means:

- no second resize pass
- no second crop pass
- no duplicate image transformation logic

## Latent Construction Summary

There are two actual latent families now:

### A. Empty Latent

Used when:

- no `mask`

Final size comes from either:

- fixed ratio bucket rules
- or `ratio=default` rules

### B. Masked Latent

Used when:

- `mask` exists

Final size always follows the `ratio=default` family of rules and is built from:

- resized `img_01`
- encoded latent from `img_01`
- resized `mask`

## Root-Cause Design Intent

The node should not be implemented by patching separate mode branches independently.

Instead, it should follow one main sequence:

1. resolve routing
2. resolve final sizing policy
3. resize images from that policy
4. encode reference latents
5. build either empty latent or masked latent
6. return standardized outputs

This keeps all downstream behavior aligned with one source of truth for sizing and routing.
