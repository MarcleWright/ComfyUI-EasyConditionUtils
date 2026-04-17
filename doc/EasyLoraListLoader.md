# EasyLoraListLoader

## Purpose

`EasyLoraListLoader` is a LoRA selector node for ComfyUI.

Its purpose is not to apply a whole chain of LoRAs at once, but to:

- keep a prepared list of LoRA candidates inside one node
- let the workflow select exactly one LoRA from that list by index
- apply the selected LoRA to the incoming `MODEL`
- output both the patched `MODEL` and the selected `lora_name`

This is useful when:

- you want to switch between multiple LoRAs without rewiring the graph
- you want to drive LoRA selection from logic or integer control
- you want one compact node instead of many separate model-only LoRA loaders

## Product Definition

`EasyLoraListLoader` is a model-only LoRA selector.

It:

- accepts one `MODEL` input
- exposes a list of LoRA slots
- exposes a `strength_model` value for each slot
- uses `index` to select exactly one active LoRA
- applies only the selected LoRA to the incoming model

It does not:

- apply multiple LoRAs in one pass
- modify CLIP
- output multiple models

## Inputs

### Required Inputs

- `model`
  - the incoming model to which the selected LoRA will be applied

- `count`
  - controls how many LoRA slots are visible in the UI
  - the node internally supports up to 50 slots
  - hidden slots are cleared when `count` is reduced

- `index`
  - 0-based selector
  - `0` means the first visible LoRA slot
  - must point to a visible and valid slot

### Slot Inputs

For each slot, the node provides:

- `lora_01`, `lora_02`, `lora_03`, ...
  - LoRA selection widgets

- `strength_model_01`, `strength_model_02`, `strength_model_03`, ...
  - model strength for the corresponding slot
  - each value can be edited directly
  - each value can also be overridden by a float input wire

## Outputs

- `model`
  - the model after applying the selected LoRA

- `lora_name`
  - the filename string of the selected LoRA

## Selection Rule

The node always selects exactly one LoRA:

1. read `count`
2. read `index`
3. locate the corresponding visible slot
4. load that LoRA
5. apply it to `model` using its matching `strength_model`

If `index` is out of range, the node raises an error.

If the selected slot does not contain a valid LoRA, the node raises an error.

## UI Behavior

The node is designed around a fixed internal list of 50 LoRA slots.

`count` only controls how many of those slots are shown.

This design is used so that:

- the UI can behave like a dynamic list
- the workflow still has stable slot identities
- each visible LoRA slot can be paired with its own `strength_model`

## Typical Use Case

Typical pattern:

```text
model
  -> EasyLoraListLoader
  -> model
  -> downstream sampler / guider / model consumer
```

Control pattern:

- use `count` to define the visible range
- use `index` to choose which LoRA is active
- optionally drive `strength_model_xx` from float nodes

## Summary

`EasyLoraListLoader` is a compact model-only LoRA selector node.

It is best understood as:

- one prepared LoRA list
- one active selection at a time
- one selected LoRA applied to one incoming model
