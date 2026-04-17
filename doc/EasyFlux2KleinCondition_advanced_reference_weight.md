# EasyFlux2KleinConditionAdvanced and EasyFlux2KleinReferenceWeightControl

## Attribution

The current `EasyFlux2KleinReferenceWeightControl` direction explicitly draws
from the runtime patching approach explored by `capitan01R` in
`ComfyUI-Flux2Klein-Enhancer`.

This project adapts that idea into the current reference-weight-control node
and integrates it with this project's `reference_control` workflow.

## Product Definition

These two nodes should be treated as a minimal-increment reference weight control design, not as two parallel conditioning systems.

The product role of `EasyFlux2KleinConditionAdvanced` is:

- to preserve the exact behavior of `EasyFlux2KleinCondition`
- to additionally output a span protocol that can be consumed by a reference weight control node
- to add per-reference weight inputs that match the dynamic image inputs one by one
- not to change size resolution
- not to change latent routing
- not to modify model attention
- to only organize token metadata for each reference after reference latents have been constructed

Its purpose is to provide standardized input for reference weight control without destabilizing the base node.

Once this protocol is confirmed, it should remain fixed as the current product definition and be carried forward if merged back into `EasyFlux2KleinCondition`.

The product role of `EasyFlux2KleinReferenceWeightControl` is:

- to consume the span protocol emitted by `EasyFlux2KleinConditionAdvanced`
- to apply independent weight control to the `K/V` segment that corresponds to each reference inside model attention
- to pass through the `conditioning` that belongs to the current sampling chain
- not to generate reference latents
- not to decide image size, mask behavior, empty latent behavior, or masked latent behavior
- not to redefine the meaning of `img_01` and `img_02+`

Its only responsibility is to turn the reference span protocol into runtime per-reference `K/V` weight control.

## One-line Definition

### `EasyFlux2KleinConditionAdvanced`

A stable incremental variant of `EasyFlux2KleinCondition` that only adds per-reference weight inputs and one extra `reference_control` output.

### `EasyFlux2KleinReferenceWeightControl`

Consumes that protocol, applies independent reference weighting at the attention `K/V` level, and forwards `conditioning` unchanged.

## Why It Is Defined This Way

The reasoning is straightforward:

- `EasyFlux2KleinCondition` already owns size resolution, mask routing, empty latent / masked latent construction, and reference latent appending
- those base behaviors are already relatively stable and should not be rewritten just to support experimental reference weight control
- therefore `Advanced` should be treated as a protocol-enhanced version of the base node, not as a separate conditioning system

This means:

- `Advanced` should not grow into a long-term parallel replacement
- the value of `Advanced` is to carry the reference span protocol with minimal disturbance to the base node
- it should later be merged back into `EasyFlux2KleinCondition`

## Definition of EasyFlux2KleinConditionAdvanced

### Product Goal

`EasyFlux2KleinConditionAdvanced` should satisfy the following definition:

- its inputs should remain aligned with `EasyFlux2KleinCondition`
- its base semantics should remain aligned with `EasyFlux2KleinCondition`
- its routing behavior should remain aligned with `EasyFlux2KleinCondition`
- its standard outputs should remain aligned with `EasyFlux2KleinCondition`
- it should additionally provide per-reference weight inputs that track the dynamic image inputs
- it should add exactly one extra `reference_control` output

### Base Behavior That Must Stay Identical

`Advanced` must not redefine:

- `img_01` as the primary image
- `img_02+` as reference-only images
- `mask` as always belonging to `img_01`
- the same ratio / megapixels resolution rules
- the same empty latent / masked latent routing rules
- the same reference latent append order
- the same `img_01_processed / width / height` semantics

In other words:

The upstream behavior of `Advanced` should remain fully equivalent to `EasyFlux2KleinCondition`.

### Added Capabilities

`Advanced` adds two things:

- a weight input for each dynamic reference image
- a `reference_control` output that can be consumed by reference weight control

That protocol needs to describe:

- reference order
- reference names
- token count for each reference
- the local token span of each reference inside the reference region
- the base weight of each reference

### Protocol Boundary

`Advanced` is only responsible for outputting local spans inside the reference region. It is not responsible for emitting absolute spans inside the full transformer sequence.

At the same time, the additional weight inputs in `Advanced` are only used to define the base weight for each reference and store it in `reference_control`.

Therefore `Advanced` should not:

- infer text token length
- infer target image token length
- infer absolute spans in the full transformer sequence
- modify the model
- execute any patch logic

## Definition of EasyFlux2KleinReferenceWeightControl

### Product Goal

`EasyFlux2KleinReferenceWeightControl` should satisfy the following definition:

- it receives a reference span protocol from `Advanced`
- it receives the `conditioning` that belongs to the current sampling chain
- it receives a `MODEL`
- at runtime, it maps the local reference spans to the actual reference `K/V` segments through an attention patch
- it applies an independent weight scale to the `K/V` segment for each reference
- it outputs a patched model
- it outputs the same `conditioning` unchanged

### What It Should Not Do

`ReferenceWeightControl` should not:

- generate reference latents
- decide image sizes
- handle mask routing
- handle empty latent / masked latent construction
- change the image encoding logic of the references
- accept separate manual per-reference weight inputs
- modify the content of `conditioning` itself

Its boundary should remain:

- consume the protocol
- interpret spans
- rewrite `K/V`
- pass `conditioning` through

### Core Behavior

The core behavior of `ReferenceWeightControl` should be understood as:

```python
k[:, :, start:end, :] *= weight_i
v[:, :, start:end, :] *= weight_i
```

Where:

- `start:end` is the target span that belongs to one reference
- `weight_i` is the final weight factor for that reference

The current implementation assumes:

- it uses `attn1_patch`
- the patch callback directly handles runtime `q / k / v`
- it depends on runtime `reference_image_num_tokens`
- it then combines that runtime information with `reference_control.reference_token_counts` and `reference_token_ranges` to determine the `K/V` segment for each reference

Therefore, the key point of the current attention patch is not to guess a fixed tail-span. Instead it is:

- first read runtime reference token counts
- then map the local spans recorded by `Advanced` onto the actual runtime reference token segments
- finally apply direct scaling to `k / v` on that segment

If runtime reference token information is not available reliably, the node cannot claim to implement strict per-reference weight control.

## Responsibility Split Between the Two Nodes

The responsibilities of the two nodes should be fixed as follows:

### `Advanced` is responsible for

- generating reference latents
- preserving the base conditioning / latent routing behavior
- recording reference token metadata
- outputting the reference span protocol

### `ReferenceWeightControl` is responsible for

- reading the span protocol
- reading the `conditioning` that belongs to the current sampling chain
- reading runtime reference token information inside `attn1_patch`
- interpreting local spans at runtime
- mapping them to the actual reference `K/V` segments used by attention
- rewriting the target `K/V` region
- returning `conditioning` unchanged to keep the workflow chain complete

The result of this design is:

- `Advanced` remains only a stable incremental extension of `EasyFlux2KleinCondition`
- `ReferenceWeightControl` remains only an independent model behavior modifier
- `Advanced` can naturally merge back into the base node later

## Suggested Protocol Definition

The extra output from `Advanced` should be standardized as `reference_control`.

Suggested semantics:

```python
{
    "reference_names": ["img_01", "img_02", "img_03"],
    "reference_base_weights": [1.0, 0.8, 1.2],
    "reference_token_counts": [n1, n2, n3],
    "reference_token_ranges": [
        (0, n1),
        (n1, n1 + n2),
        (n1 + n2, n1 + n2 + n3),
    ],
    "total_reference_tokens": n1 + n2 + n3,
}
```

The fields should mean:

- `reference_names`: the reference input order
- `reference_base_weights`: the base weight of each reference
- `reference_token_counts`: the token count for each reference
- `reference_token_ranges`: the local span of each reference inside the reference region
- `total_reference_tokens`: the total token count across all references

`reference_token_ranges` must be interpreted as:

- local spans inside the reference region
- not absolute spans in the full transformer sequence

## Weight Source

Each reference weight used by `ReferenceWeightControl` should come directly from `reference_control.reference_base_weights` emitted by `Advanced`.

That means:

- each reference base weight is defined in `Advanced`
- those weights are passed together with `reference_control`
- `ReferenceWeightControl` no longer receives separate manual per-reference weight inputs

## Recommended Minimal Implementation Rules

If this is implemented formally, the following rules should hold:

1. `Advanced` must preserve the full base behavior of `EasyFlux2KleinCondition`
2. `Advanced` should only add per-reference weight inputs that match dynamic image inputs, plus the `reference_control` output
3. `Advanced` should only emit local spans and must not infer absolute spans
4. `ReferenceWeightControl` should no longer accept manual weight inputs and should only consume `reference_control`
5. `ReferenceWeightControl` should interpret spans at runtime
6. The final action of `ReferenceWeightControl` should be direct rewriting of the target `K/V`

## Summary

This design is not meant to create two parallel conditioning systems. It is meant to create one stable mainline plus one protocol enhancement path that can later be merged:

- `EasyFlux2KleinCondition` remains the stable conditioning / latent node
- `EasyFlux2KleinConditionAdvanced` provides the reference span protocol with minimal added complexity
- `EasyFlux2KleinReferenceWeightControl` consumes that protocol and performs strict per-reference `K/V` weight control

The most reasonable end state is:

- merge the protocol output capability of `Advanced` back into `EasyFlux2KleinCondition`
- keep `ReferenceWeightControl` as a separate node

This preserves the stability of the base node while making reference weight control a clear, explainable, and maintainable design.
