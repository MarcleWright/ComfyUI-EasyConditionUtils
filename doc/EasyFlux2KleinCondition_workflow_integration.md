# EasyFlux2KleinCondition Workflow Integration

## Purpose

This document is for developers who want to drive a ComfyUI workflow from another frontend by editing workflow JSON.

The goal is to explain how to locate `EasyFlux2KleinCondition` inside a workflow and how to safely modify its inputs.

## Recommended Integration Strategy

If your frontend edits workflow JSON directly, the safest approach is:

1. load a prepared workflow template
2. locate the `EasyFlux2KleinCondition` node
3. update its widget values and connected inputs
4. submit the updated workflow to ComfyUI

## How To Locate The Node

Recommended ways to locate the node:

1. by node `type`
   `EasyFlux2KleinCondition`

2. by a stable workflow-specific node `id`
   this is useful when your template is fixed

3. by both `type` and expected title/name in your own workflow convention

If possible, prefer:

- fixed workflow template
- fixed node id
- fixed input wiring convention

## What Can Be Modified From Workflow JSON

This node currently supports frontend-driven workflow editing for:

- `ratio`
- `megapixels`
- `batch_size`
- `upscale_method`
- connected images
- connected mask

## Current Widget Fields

The node currently exposes these widget-backed parameters:

1. `ratio`
2. `megapixels`
3. `batch_size`
4. `upscale_method`

When editing workflow JSON, these are typically stored in `widgets_values`.

Important:

- treat the order as part of the current node contract
- if the node UI changes in a future version, the widget order may also change
- for production integration, use a pinned workflow template version

## Legal `ratio` Values

Current legal values are:

- `default`
- `1:1`
- `4:3`
- `3:4`
- `3:2`
- `2:3`
- `5:4`
- `4:5`
- `16:9`
- `9:16`
- `2:1`
- `1:2`
- `21:9`
- `9:21`
- `4:1`
- `1:4`

## Legal `megapixels` Values

Current legal values are:

- `default`
- `1.00`
- `1.50`
- `2.00`
- `3.00`
- `4.00`
- `6.00`
- `8.00`

## Input Semantics

### Images

This node uses ordered image inputs:

- `img_01`
- `img_02`
- `img_03`
- ...

All connected images are appended as reference latents in order.

`img_01` has additional meaning:

- it is the default size-reference image
- it is the base image when `mask` is connected

### Mask

`mask` always applies to `img_01`.

If `mask` is connected:

- masked latent mode is used
- `img_01` must also be connected

## Size Rules

### Fixed Ratio

If `ratio != default`:

- fixed ratio always uses the bucket system
- `img_01` does not control final output ratio
- `megapixels=default` is treated as `1.00`

### Default Ratio

If `ratio = default`:

- with `img_01`, use `img_01` aspect ratio rules
- without `img_01`, fall back to `1:1`

### `megapixels = default`

If `ratio = default`:

- no `img_01` -> behaves like `1.00`
- `img_01 <= 4.2 MP` -> use original `img_01` width and height
- `img_01 > 4.2 MP` -> cap to `4.00 MP` while preserving image ratio

## Outputs

The node outputs:

- `conditioning`
- `latent`
- `noise_mask`
- `mask`
- `img_01_processed`
- `width`
- `height`

`img_01_processed` is the actual processed `img_01` used internally for latent sizing / encoding.

## Integration Notes

For external frontend integration:

- legal widget values are enough for current workflow-driven usage
- you do not need native ComfyUI UI interaction
- you only need to update workflow JSON correctly

If your frontend needs values outside the current legal widget list, the backend node definition must be updated first.

## Suggested Practice

- keep one stable workflow template
- document the target node id in that template
- update only approved widget values and input links
- avoid depending on random node ids from user-generated workflows
