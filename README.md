# ComfyUI_EasyConditionUtils

ComfyUI 自定义节点包，专为简化 **FLUX.2 Klein 9B**（flux2klein9b）及其他 Flux Kontext 编辑模型的 reference latent conditioning 工作流而设计。

> **新增**：`Easy Flux Kontext Helper` — 一体化工作流辅助节点，将 img_edit / img_edit_mask / ref_generation / txt_to_img 四种 state 封装为单个节点，替代手动搭建的 state group。

## 解决什么问题？

使用 flux2klein9b 时，若要传入多张参考图片，原本需要手动串联多个 `ReferenceLatent` 节点，图片数量不固定时非常繁琐：

```
[Image 1] → ReferenceLatent ─┐
[Image 2] → ReferenceLatent ─┤→ conditioning → KSampler
[Image 3] → ReferenceLatent ─┘
```

**ComfyUI_EasyConditionUtils** 将上述流程简化为一个节点：

```
[Image Batch (1~N张)] ─┐
[VAE]                  ├→ Easy Reference Latent Apply → conditioning → KSampler
[conditioning]         ─┘
```

无需关心图片数量，节点自动处理。

---

## 安装

将本文件夹整体复制到 ComfyUI 的 `custom_nodes` 目录下，然后重启 ComfyUI：

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI_EasyConditionUtils/     ← 放这里
        ├── __init__.py
        ├── nodes/
        │   ├── __init__.py
        │   ├── easy_ref_latent.py
        │   └── easy_flux_helper.py
        └── README.md
```

---

## 节点说明

### ⭐ Easy Flux Kontext Helper（一体化工作流辅助节点）

**分类**：`EasyConditionUtils`

将工作流中的「state group」完整封装为单个节点，根据 `mode` 参数自动完成：

- 将 img_01 / img_02 缩放并 VAE 编码，注入为 ReferenceLatent
- 生成符合所选模式的输出 Latent（直接接入 KSampler）
- 生成清零后的负向 Conditioning

**四种模式对照表**：

| mode | 对应 state | 输出 Latent | img_01 角色 | 适用场景 |
|------|-----------|------------|------------|---------|
| `img_edit` | workflow 1 | EmptyLatent（img_01 尺寸） | 参考 + 决定尺寸 | 图片整体编辑 |
| `img_edit_mask` | workflow 2 | VAEEncode(img_01) + NoiseMask | 参考 + 重绘基底 | 局部蒙版重绘 |
| `ref_generation` | workflow 3 | EmptyLatent（自定义宽高） | 风格/内容参考 | 参考图引导生成新图 |
| `txt_to_img` | workflow 4 | EmptyLatent（自定义宽高） | 可选风格参考 | 纯文字驱动生成 |

| 输入 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `conditioning` | CONDITIONING | ✅ | 来自 CLIPTextEncode / FluxGuidance 的正向 conditioning |
| `vae` | VAE | ✅ | VAE 模型，用于编码参考图 |
| `mode` | 下拉 | ✅ | 四选一工作模式 |
| `resolution` | FLOAT | ✅ | 参考图缩放目标（MP），推荐 1.0 |
| `width` | INT | ✅ | 输出宽度（ref_generation / txt_to_img 时生效） |
| `height` | INT | ✅ | 输出高度（同上） |
| `batch_size` | INT | ✅ | 批次数量 |
| `img_01` | IMAGE | ❌ | 主参考图（img_edit / img_edit_mask 时建议接入） |
| `img_02` | IMAGE | ❌ | 次参考图（可选，仅作为 ReferenceLatent） |
| `mask` | MASK | ❌ | 蒙版（img_edit_mask 模式必须接入） |
| `upscale_method` | 下拉 | ❌ | 缩放算法（默认 bilinear） |

| 输出 | 类型 | 说明 |
|------|------|------|
| `positive` | CONDITIONING | 含 reference latent 的正向 conditioning |
| `negative` | CONDITIONING | 清零后的负向 conditioning |
| `latent` | LATENT | 根据 mode 生成的 latent，直接接入 KSampler |

**典型连接方式**：

```
CLIPTextEncode → FluxGuidance ──────────────────────────────────────┐
Load Image (img_01) ────────────────────────────────────────────────┤
Load Image (img_02) ─────────────────────── Easy Flux Kontext Helper ─→ KSampler (positive, latent)
Load VAE ────────────────────────────────────────────────────────────┤      └─→ KSampler (negative)
Load Image (mask, 可选) ─────────────────────────────────────────────┘
```

---

### 1. Easy Reference Latent Apply (Batch) — 批量参考图节点

**分类**：`EasyConditionUtils`

自动将一批图片编码并逐一追加为 reference latent conditioning，等价于串联 N 个原生 `ReferenceLatent` 节点。

| 输入 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `conditioning` | CONDITIONING | ✅ | 基础 conditioning（来自 CLIP Text Encode 等） |
| `vae` | VAE | ✅ | 用于将图片编码到 latent 空间的 VAE 模型 |
| `images` | IMAGE | ✅ | 参考图片批次 `[B, H, W, C]`，每张图片成为一个 reference latent |
| `upscale_method` | 下拉 | ❌ | 缩放算法（默认 bilinear） |
| `scale_to_megapixels` | FLOAT | ❌ | 编码前将每张图缩放到此像素数（MP），0 = 不缩放，默认 1.0 |

| 输出 | 类型 | 说明 |
|------|------|------|
| `conditioning` | CONDITIONING | 追加了所有 reference latent 的 conditioning |
| `image_count` | INT | 实际处理的图片数量（方便调试） |

**典型用法**：

```
Load Image → Image Batch ──────────────────────────────┐
Load VAE ──────────────────────────────────────────────┤
CLIPTextEncode (positive) → Easy Reference Latent Apply → KSampler
```

---

### 2. Easy Reference Latent (from Latent)

**分类**：`EasyConditionUtils`

接收已编码的 LATENT（而非原始图片），将其拆分为多个 reference latent 追加到 conditioning。  
适合图片已经过 VAE Encode 的场景。

| 输入 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `conditioning` | CONDITIONING | ✅ | 基础 conditioning |
| `latent` | LATENT | ✅ | 已编码的 latent，批次中每个样本成为一个 reference |
| `split_batch` | BOOLEAN | ❌ | True = 按样本拆分（默认）；False = 整个批次作为单个 reference |

| 输出 | 类型 | 说明 |
|------|------|------|
| `conditioning` | CONDITIONING | 追加了 reference latent 的 conditioning |
| `latent_count` | INT | 追加的 reference latent 数量 |

---

### 3. Easy Clear Reference Latents

**分类**：`EasyConditionUtils`

从 conditioning 中移除所有 reference latents，用于复用 conditioning 时重置参考图片。

| 输入 | 类型 | 说明 |
|------|------|------|
| `conditioning` | CONDITIONING | 需要清理的 conditioning |

| 输出 | 类型 | 说明 |
|------|------|------|
| `conditioning` | CONDITIONING | 移除了所有 reference latent 的 conditioning |

---

### 4. Easy Count Reference Latents

**分类**：`EasyConditionUtils`

只读工具节点，统计当前 conditioning 中已有多少个 reference latent，并在控制台打印详细信息。

| 输入 | 类型 | 说明 |
|------|------|------|
| `conditioning` | CONDITIONING | 要检查的 conditioning（原样透传） |

| 输出 | 类型 | 说明 |
|------|------|------|
| `conditioning` | CONDITIONING | 原样透传，不做修改 |
| `count` | INT | reference latent 总数 |
| `summary` | STRING | 文字摘要，包含每个 latent 的 shape |

---

## 推荐工作流（flux2klein9b 多图参考）

```
┌─ Load Image (ref1) ─┐
├─ Load Image (ref2) ─┤→ Make Image Batch ─→ Easy Reference Latent Apply ─→ KSampler (positive)
└─ Load Image (ref3) ─┘        ↑                       ↑
                           Load VAE            CLIPTextEncode (positive)

同时配合 FluxKontextMultiReferenceLatentMethod 节点设置多图策略（offset / index / uxo）
```

---

## 兼容性

- ComfyUI（需要 `node_helpers` 模块，2024 年底之后的版本均支持）
- FLUX.2 Klein 9B（flux2klein9b）
- Flux Kontext Dev / Pro / Max
- 任何支持 `reference_latents` conditioning key 的模型

---

## License

MIT
