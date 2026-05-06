# Easy Flux2 Klein 9B Reference Weight Control

## Purpose

`Easy Flux2 Klein 9B Reference Weight Control` is a development-only copy of
`Easy Flux2 Klein Reference Weight Control`.

It exists so FLUX.2 Klein 9B-specific reference-weight behavior can be tested
without changing the existing node that is already confirmed to work on
`flux2klein4b`.

## Current Status

The current stable reference-weight path is:

1. `EasyFlux2KleinConditionAdvanced` outputs `reference_control`
2. `ReferenceWeightControl` clones `model` and installs an attention patch
3. sampler runs the patched model
4. `attn1_patch` receives runtime `q / k / v`
5. the patch reads `extra_options["reference_image_num_tokens"]`
6. the patch locates each reference span
7. the patch applies:

```python
k[:, :, start:end, :] *= weight
v[:, :, start:end, :] *= weight
```

This path has been observed to work on `flux2klein4b`.

## Observed 4B Behavior

Using the 9B test node on `flux2klein4b`, the debug log confirms that the
attention patch enters correctly:

```text
[EasyFlux2Klein9BReferenceWeightControl] patch installed
  reference_names=['img_01', 'img_02']
  reference_base_weights=[1.0, 1.0]
  reference_token_counts=[3927, 3916]
  reference_token_ranges=[(0, 3927), (3927, 7843)]
  total_reference_tokens=7843

[EasyFlux2Klein9BReferenceWeightControl] attn1_patch entered
  q.shape=(1, 24, 12282, 128)
  k.shape=(1, 24, 12282, 128)
  v.shape=(1, 24, 12282, 128)
  extra_options.keys=[
    'block_index',
    'block_type',
    'callbacks',
    'cond_or_uncond',
    'img_slice',
    'patches',
    'reference_image_num_tokens',
    'sample_sigmas',
    'sigmas',
    'total_blocks',
    'uuids',
    'wrappers'
  ]
  block_index=0
  reference_image_num_tokens=[3927, 3916]
  reference_control_token_counts=[3927, 3916]
  total_runtime_reference_tokens=7843
  ref_span[0] name=img_01 weight=1.0 runtime_tokens=3927 seq_start=-7843 seq_end=-3916
  ref_span[1] name=img_02 weight=1.0 runtime_tokens=3916 seq_start=-3916 seq_end=0
```

This confirms that, on 4B:

- the patched model is used by the sampler
- `attn1_patch` is called
- runtime `q / k / v` are available
- `reference_image_num_tokens` is present
- runtime token counts match `reference_control`
- the current span mapping is technically reachable

## Observed 9B Behavior

Using the same test node on `flux2klein9b`, the debug log currently shows only:

```text
[EasyFlux2Klein9BReferenceWeightControl] patch installed
  reference_names=['img_01', 'img_02']
  reference_base_weights=[1.0, 1.0]
  reference_token_counts=[7788, 7839]
  reference_token_ranges=[(0, 7788), (7788, 15627)]
  total_reference_tokens=15627

Requested to load Flux2
loaded completely; 8996.02 MB loaded, full load: True
Prompt executed in 41.15 seconds
```

After adding `attn1_output_patch` diagnostics, the current observed 9B behavior
is more specific:

- `attn1_patch` is not entered
- `attn1_output_patch` is entered
- `reference_image_num_tokens` is missing from `extra_options`

Example:

```text
[EasyFlux2Klein9BReferenceWeightControl] attn1_output_patch entered
attn.shape=(1, 23927, 4096)
extra_options.keys=[
  'block_index',
  'block_type',
  'callbacks',
  'cond_or_uncond',
  'img_slice',
  'patches',
  'sample_sigmas',
  'sigmas',
  'total_blocks',
  'uuids',
  'wrappers'
]
block_index=0
block_type=double
img_slice=[512, 23927]
reference_image_num_tokens=[]
```

This means the node installs the patch and 9B does enter the transformer patch
system, but not through the pre-attention `q / k / v` patch path used by 4B.

Therefore, the current 9B problem is not a reference span calculation problem.
The strict `K/V` callback is not reached.

## Current Diagnosis

The most likely current causes are:

- the 9B workflow uses a model wrapper or sampler path that does not trigger
  ComfyUI's pre-attention `attn1_patch`
- the 9B full-load path differs from the 4B dynamic-VRAM path
- the 9B model path does not carry `transformer_options["patches"]` into the
  Flux attention layers
- the 9B implementation may use a different attention path than the tested 4B
  path

## Output-Patch Experiment

An experimental `patch_mode=attn1_output` path was added to test whether 9B can
be affected at all through the available output patch.

In this mode the node applies:

```python
attn[:, start:end, :] *= weight
```

Observed result on 9B:

- `img_02_weight=0.0`
- `attn1_output_patch` enters
- output image changes strongly
- the image collapses or loses detail instead of simply ignoring `img_02`

Interpretation:

- the output patch does affect the 9B model path
- the reference tail span is probably touching important image-stream tokens
- however, attention-output scaling is too late and too destructive
- it should be kept as a diagnostic mode, not treated as final reference weight
  control

The correct target is still a pre-attention `K/V` control point or an equivalent
9B-specific hook.

The current diagnostic direction is:

- keep the existing stable node unchanged
- use this 9B-specific node for instrumentation
- test which patch hooks are actually entered by 9B

## Current Debug Hooks

The 9B test node currently installs:

- `attn1_patch`
- `attn1_output_patch`

Reason:

- current ComfyUI Flux layers use `attn1_patch` before attention
- current ComfyUI Flux layers use `attn1_output_patch` after attention
- Flux layers do not appear to use `attn2_patch` in the same path

The node also provides `patch_mode`:

- `debug_only`
  - only print diagnostics
- `attn1_kv`
  - use the 4B path and scale runtime `k / v`
- `attn1_output`
  - experimental 9B path that scales the attention output span:

```python
attn[:, start:end, :] *= weight
```

`attn1_output` is not equivalent to strict `K/V` control. It is an
output-level approximation used to test whether the reference tokens are still
located in the tail span on 9B.

## Next Test

Run `flux2klein9b` with:

- `Easy Flux2 Klein 9B Reference Weight Control`
- `debug=True`

Then check the terminal for either:

```text
[EasyFlux2Klein9BReferenceWeightControl] attn1_patch entered
```

or:

```text
[EasyFlux2Klein9BReferenceWeightControl] attn1_output_patch entered
```

If `attn1_patch` enters, the next step is to inspect runtime token counts and
span mapping.

If only `attn1_output_patch` enters, test:

- `patch_mode=attn1_output`
- `img_02_weight=0.0`
- fixed seed

If the output changes, the reference tail span is probably reachable through
the attention output stream. If the output still does not change, either the
tail-span assumption is wrong for 9B or output-level scaling is not an effective
control point.

If neither enters, the next step is to inspect the 9B sampler/model wrapper and
find where transformer patches are dropped or bypassed.

After observing that `attn1_output` affects the image destructively, the next
diagnostic step is to inspect `extra_options["patches"].keys()` during
`attn1_output_patch`.

This determines whether:

- `attn1_patch` is missing from runtime transformer patches
- or `attn1_patch` is present but ignored by the 9B attention implementation
