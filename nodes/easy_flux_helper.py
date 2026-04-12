"""
EasyFluxKontextHelper — 一体化 Flux Kontext 工作流辅助节点

将工作流中的「state group」封装为单个节点，支持 4 种工作模式：
  - img_edit          : 图片编辑（无蒙版），以参考图原始尺寸生成空 Latent
  - img_edit_mask     : 带蒙版局部重绘，以 VAE 编码的 img_01 + NoiseMask 作为 Latent
  - ref_generation    : 参考图引导生成（自定义宽高），生成空 Latent
  - txt_to_img        : 文生图（自定义宽高），生成空 Latent，参考图可选接入

兼容 FLUX.2 Klein 9B (flux2klein9b) 及其他支持 reference_latents 的 Flux Kontext 模型。
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import comfy.utils
import node_helpers

from .easy_ref_latent import (
    _scale_image_to_megapixels,
    _ensure_divisible,
    UPSCALE_METHODS,
)


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _zero_out_conditioning(conditioning: list) -> list:
    """
    复现 ComfyUI ConditioningZeroOut 节点的行为：
    将 conditioning 张量全部清零，同时清零 pooled_output（如存在）。
    用于生成 Flux 所需的"空"负向 conditioning。
    """
    result = []
    for cond_tensor, cond_dict in conditioning:
        d = cond_dict.copy()
        if "pooled_output" in d:
            d["pooled_output"] = torch.zeros_like(d["pooled_output"])
        result.append([torch.zeros_like(cond_tensor), d])
    return result


def _make_empty_flux_latent(width: int, height: int, batch_size: int) -> dict:
    """
    复现 EmptyFlux2LatentImage 节点的行为：
    Flux VAE 使用 8× 空间压缩、16 个潜在通道。
    宽高均向下对齐到 8 的倍数。
    """
    w = max(8, (width // 8) * 8)
    h = max(8, (height // 8) * 8)
    return {"samples": torch.zeros([batch_size, 16, h // 8, w // 8])}


def _add_ref_latent(conditioning: list, latent_tensor: torch.Tensor) -> list:
    """向 conditioning 追加一个 reference latent（复现 ReferenceLatent 节点行为）。"""
    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [latent_tensor]},
        append=True,
    )


def _encode_image_as_ref(
    img: torch.Tensor,
    vae,
    resolution: float,
    upscale_method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将图片缩放至目标 MP 并 VAE 编码。
    返回 (scaled_img [1,H,W,C], latent_tensor [1,16,H//8,W//8])。
    """
    scaled = _scale_image_to_megapixels(img, resolution, upscale_method)
    scaled = _ensure_divisible(scaled, 16)
    latent = vae.encode(scaled[:, :, :, :3])
    return scaled, latent


def _resize_mask_to(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    将蒙版双线性插值缩放到目标像素尺寸 (height × width)。
    输入支持 [H, W] 或 [B, H, W] 格式，输出为 [B, 1, H, W]（兼容 SetLatentNoiseMask 存储格式）。
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # → [1, H, W]
    # → [B, 1, H, W] for F.interpolate
    mask_4d = mask.unsqueeze(1).float()
    resized = F.interpolate(
        mask_4d,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized  # [B, 1, H, W]


# ---------------------------------------------------------------------------
# 主节点
# ---------------------------------------------------------------------------

_MODE_TOOLTIPS = {
    "img_edit": (
        "图片编辑（无蒙版）：img_01 作为参考，输出 Latent 为 img_01 原始尺寸的空 Latent。"
    ),
    "img_edit_mask": (
        "带蒙版局部重绘：img_01 编码后配合 mask 生成 NoiseMask Latent，"
        "仅蒙版区域被重绘，其余区域保留。"
    ),
    "ref_generation": (
        "参考图引导生成：img_01 / img_02 作为风格/内容参考，"
        "在自定义宽高下全新生成图片。"
    ),
    "txt_to_img": (
        "文生图：纯文字驱动，在自定义宽高下生成图片。"
        "img_01 / img_02 若接入则同样作为参考。"
    ),
}

MODES = list(_MODE_TOOLTIPS.keys())


class EasyFluxKontextHelper:
    """
    Flux Kontext / FLUX.2 Klein 一体化工作流辅助节点。

    将工作流中需要手动搭建的「state group」封装为单个节点，
    根据所选 mode 自动完成参考图编码、ReferenceLatent 注入、
    负向 Conditioning 生成和输出 Latent 构建。

    Inputs
    ------
    conditioning  : 来自文本编码器（CLIPTextEncode / FluxGuidance）的正向 conditioning。
    vae           : VAE 模型，用于编码参考图和（img_edit_mask 模式下）编码基础 latent。
    mode          : 工作模式，决定输出 latent 的生成方式。
    resolution    : 参考图缩放目标像素量（百万像素）。推荐 1.0 MP（Flux 训练分辨率）。
    width         : 输出宽度（ref_generation / txt_to_img / img_edit 无 img_01 时生效）。
    height        : 输出高度（同上）。
    batch_size    : 生成批次数量。
    img_01        : 主参考图。img_edit 时也决定输出尺寸；img_edit_mask 时作为重绘基础。
    img_02        : 次参考图（可选），仅作为 ReferenceLatent 注入 conditioning。
    mask          : 蒙版（img_edit_mask 模式必须）。白色区域将被重绘。
    upscale_method: 图片缩放插值算法。

    Outputs
    -------
    positive      : 含 reference latent 的正向 conditioning。
    negative      : 清零后的负向 conditioning（兼容 Flux 所需格式）。
    latent        : 根据 mode 生成的 LATENT，直接接入 KSampler。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "来自 CLIPTextEncode 或 FluxGuidance 的正向 conditioning。",
                }),
                "vae": ("VAE", {
                    "tooltip": "用于编码参考图的 VAE 模型。",
                }),
                "mode": (MODES, {
                    "default": "img_edit",
                    "tooltip": (
                        "工作模式：\n"
                        "• img_edit         — 图片编辑（无蒙版）\n"
                        "• img_edit_mask    — 带蒙版局部重绘\n"
                        "• ref_generation   — 参考图引导生成（自定义尺寸）\n"
                        "• txt_to_img       — 文生图（自定义尺寸）"
                    ),
                }),
                "resolution": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.05,
                    "tooltip": (
                        "将参考图缩放到此像素量（MP）后再 VAE 编码。\n"
                        "1.0 = Flux 标准训练分辨率，可避免像素偏移。"
                    ),
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": (
                        "输出图像宽度（像素）。\n"
                        "img_edit 且接入 img_01 时自动使用 img_01 宽度，此参数作为后备值。"
                    ),
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 8,
                    "tooltip": (
                        "输出图像高度（像素）。\n"
                        "img_edit 且接入 img_01 时自动使用 img_01 高度，此参数作为后备值。"
                    ),
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "tooltip": "一次生成的图片数量（batch）。",
                }),
            },
            "optional": {
                "img_01": ("IMAGE", {
                    "tooltip": (
                        "主参考图（可选）。\n"
                        "• img_edit / img_edit_mask：同时作为重绘基础和参考 latent\n"
                        "• ref_generation / txt_to_img：仅作为风格/内容参考"
                    ),
                }),
                "img_02": ("IMAGE", {
                    "tooltip": "次参考图（可选），仅作为 ReferenceLatent 注入 conditioning。",
                }),
                "mask": ("MASK", {
                    "tooltip": (
                        "蒙版（img_edit_mask 模式必须接入）。\n"
                        "白色（值=1）区域被重绘，黑色（值=0）区域保留 img_01 内容。"
                    ),
                }),
                "upscale_method": (UPSCALE_METHODS, {
                    "default": "bilinear",
                    "tooltip": "缩放参考图时使用的插值算法。",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "process"
    CATEGORY = "EasyConditionUtils"
    DESCRIPTION = (
        "Flux Kontext / FLUX.2 Klein 一体化工作流辅助节点。\n"
        "自动完成参考图编码（ReferenceLatent）、负向 Conditioning 生成和输出 Latent 构建。\n"
        "支持 4 种模式：img_edit / img_edit_mask / ref_generation / txt_to_img。"
    )

    # ------------------------------------------------------------------
    def process(
        self,
        conditioning: list,
        vae,
        mode: str,
        resolution: float,
        width: int,
        height: int,
        batch_size: int,
        img_01: torch.Tensor | None = None,
        img_02: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        upscale_method: str = "bilinear",
    ):
        positive = conditioning

        # ── 1. 将 img_01 / img_02 编码并注入 ReferenceLatent ──────────────
        #   img_01 在 img_edit_mask 模式下还需要作为基础 latent，先缓存缩放后的图片和编码。
        img_01_scaled: torch.Tensor | None = None
        img_01_latent: torch.Tensor | None = None

        if img_01 is not None:
            img_01_scaled, img_01_latent = _encode_image_as_ref(
                img_01, vae, resolution, upscale_method
            )
            positive = _add_ref_latent(positive, img_01_latent)

        if img_02 is not None:
            _, img_02_latent = _encode_image_as_ref(img_02, vae, resolution, upscale_method)
            positive = _add_ref_latent(positive, img_02_latent)

        # ── 2. 负向 Conditioning（ConditioningZeroOut）────────────────────
        negative = _zero_out_conditioning(conditioning)

        # ── 3. 生成输出 Latent ────────────────────────────────────────────
        latent_out = self._build_latent(
            mode=mode,
            width=width,
            height=height,
            batch_size=batch_size,
            img_01_scaled=img_01_scaled,
            img_01_latent=img_01_latent,
            mask=mask,
            vae=vae,
        )

        return (positive, negative, latent_out)

    # ------------------------------------------------------------------
    def _build_latent(
        self,
        mode: str,
        width: int,
        height: int,
        batch_size: int,
        img_01_scaled: torch.Tensor | None,
        img_01_latent: torch.Tensor | None,
        mask: torch.Tensor | None,
        vae,
    ) -> dict:
        """根据 mode 构建 KSampler 所需的 LATENT 字典。"""

        if mode == "img_edit":
            return self._latent_img_edit(
                width, height, batch_size, img_01_scaled
            )

        if mode == "img_edit_mask":
            return self._latent_img_edit_mask(
                width, height, batch_size, img_01_scaled, img_01_latent, mask
            )

        # ref_generation 和 txt_to_img 均使用自定义尺寸的空 Latent
        return _make_empty_flux_latent(width, height, batch_size)

    # ------------------------------------------------------------------
    def _latent_img_edit(
        self,
        width: int,
        height: int,
        batch_size: int,
        img_01_scaled: torch.Tensor | None,
    ) -> dict:
        """
        img_edit 模式：
        以 img_01 缩放后的像素尺寸为准创建空 Latent。
        若未接入 img_01，则退回到 width × height。
        """
        if img_01_scaled is not None:
            h, w = img_01_scaled.shape[1], img_01_scaled.shape[2]
        else:
            h, w = height, width
        return _make_empty_flux_latent(w, h, batch_size)

    # ------------------------------------------------------------------
    def _latent_img_edit_mask(
        self,
        width: int,
        height: int,
        batch_size: int,
        img_01_scaled: torch.Tensor | None,
        img_01_latent: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> dict:
        """
        img_edit_mask 模式：
        以 img_01 缩放后编码的 latent 为基础，叠加 NoiseMask。
        蒙版（白色区域）对应的像素将被重新生成，其余保留 img_01 内容。

        流程对应工作流中：
            img_01 → ImageScaleToTotalPixels → VAEEncode → SetLatentNoiseMask → KSampler
        """
        if img_01_latent is None:
            # 未接入 img_01：退回到空 Latent（无法做 inpainting）
            print(
                "[EasyConditionUtils] 警告：img_edit_mask 模式未接入 img_01，"
                "将退回到空 Latent，无法执行真正的蒙版重绘。"
            )
            return _make_empty_flux_latent(width, height, batch_size)

        # 将 latent 重复到 batch_size（VAE encode 单图输出 batch=1）
        base = img_01_latent  # [1, 16, H//8, W//8]
        if batch_size > 1:
            base = base.repeat(batch_size, 1, 1, 1)

        latent_out = {"samples": base}

        if mask is not None and img_01_scaled is not None:
            scaled_h, scaled_w = img_01_scaled.shape[1], img_01_scaled.shape[2]
            # 将蒙版插值到与缩放后 img_01 相同的像素尺寸
            mask_resized = _resize_mask_to(mask, scaled_h, scaled_w)
            # ComfyUI SetLatentNoiseMask 将蒙版存储为 [B, 1, H, W]（像素空间）
            latent_out["noise_mask"] = mask_resized
        elif mask is None:
            print(
                "[EasyConditionUtils] 提示：img_edit_mask 模式未接入 mask，"
                "将对整张图片重绘（等同于 img_edit）。"
            )

        return latent_out
