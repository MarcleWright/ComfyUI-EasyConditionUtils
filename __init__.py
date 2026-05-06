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
    EasyFluxKontextHelper,
    EasyFlux2KleinCondition,
    EasyFlux2KleinConditionAdvanced,
    EasyFlux2Klein9BReferenceWeightControl,
    EasyFlux2KleinReferenceWeightControl,
    EasyLoraListLoader,
    EasyLoadTextBatch,
    EasyTextListSelector,
)

# ------------------------------------------------------------------
# ComfyUI registration
# ------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "EasyReferenceLatentApply":       EasyReferenceLatentApply,
    "EasyReferenceLatentFromLatent":  EasyReferenceLatentFromLatent,
    "EasyClearReferenceLatents":      EasyClearReferenceLatents,
    "EasyCountReferenceLatents":      EasyCountReferenceLatents,
    "EasyFluxKontextHelper":          EasyFluxKontextHelper,
    "EasyFlux2KleinCondition":        EasyFlux2KleinCondition,
    "EasyFlux2KleinConditionAdvanced": EasyFlux2KleinConditionAdvanced,
    "EasyFlux2Klein9BReferenceWeightControl": EasyFlux2Klein9BReferenceWeightControl,
    "EasyFlux2KleinReferenceWeightControl": EasyFlux2KleinReferenceWeightControl,
    "EasyLoraListLoader":             EasyLoraListLoader,
    "EasyLoadTextBatch":             EasyLoadTextBatch,
    "EasyTextListSelector":           EasyTextListSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyReferenceLatentApply":       "Easy Reference Latent Apply (Batch)",
    "EasyReferenceLatentFromLatent":  "Easy Reference Latent (from Latent)",
    "EasyClearReferenceLatents":      "Easy Clear Reference Latents",
    "EasyCountReferenceLatents":      "Easy Count Reference Latents",
    "EasyFluxKontextHelper":          "Easy Flux Kontext Helper",
    "EasyFlux2KleinCondition":        "Easy Flux2 Klein Condition",
    "EasyFlux2KleinConditionAdvanced": "Easy Flux2 Klein Condition Advanced",
    "EasyFlux2Klein9BReferenceWeightControl": "Easy Flux2 Klein 9B Reference Weight Control",
    "EasyFlux2KleinReferenceWeightControl": "Easy Flux2 Klein Reference Weight Control",
    "EasyLoraListLoader":             "Easy LoRA List Loader",
    "EasyLoadTextBatch":             "Easy Load Text Batch",
    "EasyTextListSelector":           "Easy Text List Selector",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
