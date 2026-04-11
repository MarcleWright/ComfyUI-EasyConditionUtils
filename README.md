# EasyConditionUtils

ComfyUI 自定义节点包，专为简化 **FLUX.2 Klein 9B**（flux2klein9b）及其他 Flux Kontext 编辑模型的 reference latent conditioning 工作流而设计。

## 解决什么问题？

使用 flux2klein9b 时，若要传入多张参考图片，原本需要手动串联多个 `ReferenceLatent` 节点，图片数量不固定时非常繁琐：

```
[Image 1] → ReferenceLatent ─┐
[Image 2] → ReferenceLatent ─┤→ conditioning → KSampler
[Image 3] → ReferenceLatent ─┘
```

**EasyConditionUtils** 将上述流程简化为一个节点：

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
    └── EasyConditionUtils/     ← 放这里
        ├── __init__.py
        ├── nodes/
        │   ├── __init__.py
        │   └── easy_ref_latent.py
        └── README.md
```

---

## 节点说明

### 1. Easy Reference Latent Apply (Batch) ⭐ 主节点

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
