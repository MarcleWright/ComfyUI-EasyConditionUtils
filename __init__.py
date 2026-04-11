"""
EasyConditionUtils — ComfyUI Custom Nodes
==========================================
Utility nodes for streamlining conditioning workflows, especially for
FLUX.2 Klein 9B (flux2klein9b) and other Flux Kontext edit models that
use reference latent conditioning.

Nodes provided
--------------
EasyReferenceLatentApply
    Main node. Feed it a batch of images + a VAE + any conditioning, and it
    automatically encodes every image and appends the resulting reference
    latent to the conditioning — equivalent to manually chaining N separate
    ReferenceLatent nodes.

EasyReferenceLatentFromLatent
    Same as above but accepts pre-encoded LATENT instead of raw images.
    Useful when your images are already encoded.

EasyClearReferenceLatents
    Strip all reference latents from a conditioning (reset).

EasyCountReferenceLatents
    Read-only utility: count and summarise reference latents in a conditioning.
"""

from .nodes import (
    EasyReferenceLatentApply,
    EasyReferenceLatentFromLatent,
    EasyClearReferenceLatents,
    EasyCountReferenceLatents,
)

# ------------------------------------------------------------------
# ComfyUI registration
# ------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "EasyReferenceLatentApply":       EasyReferenceLatentApply,
    "EasyReferenceLatentFromLatent":  EasyReferenceLatentFromLatent,
    "EasyClearReferenceLatents":      EasyClearReferenceLatents,
    "EasyCountReferenceLatents":      EasyCountReferenceLatents,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyReferenceLatentApply":       "Easy Reference Latent Apply (Batch)",
    "EasyReferenceLatentFromLatent":  "Easy Reference Latent (from Latent)",
    "EasyClearReferenceLatents":      "Easy Clear Reference Latents",
    "EasyCountReferenceLatents":      "Easy Count Reference Latents",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
