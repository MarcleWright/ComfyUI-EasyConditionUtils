# EasyFlux2KleinCondition Advanced 与 Reference Weight Patch

## 目标

本文档记录两个新增节点的设计目标、当前实现方案、核心原理，以及它们在 `FLUX.2 Klein` 工作流中的职责边界。

这两个节点分别是：

- `EasyFlux2KleinConditionAdvanced`
- `EasyFlux2KleinReferenceWeightPatch`

设计出发点是：

- 保留 `EasyFlux2KleinCondition` 现有的尺寸决策和 latent 路由能力
- 在不破坏原节点职责的前提下，为多参考图权重控制准备元数据
- 将“生成 reference 元数据”和“执行 attention 权重调节”拆成两个独立节点

## 节点分工

### `EasyFlux2KleinConditionAdvanced`

这个节点的职责是：

- 继承原始 `EasyFlux2KleinCondition` 的 routing 行为
- 继续输出标准的 `conditioning / latent / noise_mask / mask / img_01_processed / width / height`
- 接受可选的 `reference_weights` 字符串输入
- 为每一张参考图记录权重与 token 元数据
- 将这些元数据写入 `conditioning["reference_control"]`

它本身不直接修改模型 attention。

### `EasyFlux2KleinReferenceWeightPatch`

这个节点的职责是：

- 从 `conditioning` 中读取 `reference_control`
- clone 一份 `MODEL`
- 对模型挂接 `attn2 patch`
- 在 cross-attention 路径中，按 reference span 对上下文区段做缩放

它不重新生成 reference latents，也不负责尺寸决策。

## 为什么要拆成两个节点

原因有三个：

- `conditioning` 准备阶段天然适合保存 reference 顺序、token 数、局部 span 等元数据
- attention patch 阶段天然适合消费这些元数据并对运行时张量生效
- 这样可以把 WIP 的 patch 逻辑与稳定的 latent/route 逻辑解耦

拆分之后的好处是：

- 原来的 `EasyFlux2KleinCondition` 可以继续保持稳定
- `Advanced` 节点可以先稳定输出 metadata
- `Patch` 节点可以单独迭代，不影响前面的 conditioning 逻辑

## 背后的原理

## 1. Reference latent 不是普通的 img2img 起始 latent

在这套工作流里，参考图首先会被编码成 latent，再作为 reference conditioning 提供给模型。

它更接近“视觉上下文”而不是“从原图出发的采样起点”。

这也是为什么多参考图权重问题，最合理的落点通常不是“缩图”或“改基础 latent”，而是：

- 要先知道每张 reference 对应哪一段 token
- 再在 attention 层面对对应 span 做单独控制

## 2. Token span 是单独调权重的前提

如果一张参考图进入模型后会对应一段连续 token，那么单独调整它的影响力，本质上就变成：

- 先确定这张图对应的 token 数
- 再确定这段 token 在 reference 区中的起止位置

只有知道这一点，才可能做到：

- 只调 `img_02`
- 不误伤 `img_01`
- 不误伤 target image tokens
- 不误伤 text tokens

## 3. 为什么现在先记录 local span

当前 `Advanced` 节点记录的是 reference 区内部的局部区间，而不是整条 attention 序列里的绝对区间。

例如：

```text
img_01 -> local span [0, n1)
img_02 -> local span [n1, n1+n2)
img_03 -> local span [n1+n2, n1+n2+n3)
```

这样做的原因是：

- 在 conditioning 阶段，可以可靠地知道每张 reference 自己占多少 token
- 但不一定能在这个阶段准确知道 text tokens 与 target image tokens 在总序列中的真实偏移

因此当前设计把问题分成两步：

1. `Advanced` 记录 local span
2. `Patch` 在运行时根据实际 attention 序列去解释这些 span

## 当前实现方案

## `EasyFlux2KleinConditionAdvanced`

### 输入

它保留了原节点的主要输入，并新增：

- `reference_weights`
- `default_reference_weight`

其中 `reference_weights` 使用字符串格式：

```text
img_01=1.0,img_02=0.8,img_03=1.2
```

未显式指定的图片会使用 `default_reference_weight`。

### 每张参考图记录了什么

当前实现里，每张图都会记录：

- `name`
- `weight`
- `latent_h`
- `latent_w`
- `token_count`

这里的 `token_count` 采用当前工程里更稳定的近似方式：

```text
token_count = latent_h * latent_w
```

这与当前节点尺寸逻辑保持一致，因为本项目里 FLUX.2 latent 路径以 `/16` 的 latent 空间尺寸来理解。

### 输出到 conditioning 的 metadata

当前写入的是：

```python
conditioning["reference_control"] = {
    "mode": "local_reference_spans",
    "names": [...],
    "weights": [...],
    "token_counts": [...],
    "latent_shapes": [...],
    "local_token_ranges": [...],
    "total_reference_tokens": ...,
}
```

另外还写入：

```python
conditioning["reference_control_version"] = 1
```

这相当于给后续 patch 留了一层协议版本。

### 为什么还输出 `reference_summary`

为了便于在 workflow 中快速确认 metadata 是否符合预期，节点额外输出一个字符串摘要。

这个摘要可以帮助快速观察：

- 参考图顺序
- 每张图的权重
- 每张图估算出的 token 数
- 每张图的 local span

## `EasyFlux2KleinReferenceWeightPatch`

### 输入

这个节点输入：

- `model`
- `conditioning`

以及两个附加控制项：

- `global_weight_scale`
- `include_img_01`

`global_weight_scale` 会乘到每张 reference 的独立权重上。

`include_img_01 = false` 时，即使 `reference_control` 里给了 `img_01` 权重，也会把它按 `1.0` 对待。

### 当前 patch 的实现方式

当前版本使用的是：

- `model.clone()`
- `set_model_attn2_patch(...)`

也就是说，它当前挂的是 cross-attention 路径的 patch。

回调会拿到：

- `q`
- `context`
- `value`
- `extra_options`

然后对 `context / value` 中属于 reference 的 span 做缩放。

### 当前的关键假设

当前 patch 使用了一个非常明确的工程假设：

- reference tokens 位于 `context` 序列尾部

于是它会用：

```text
reference_start = sequence_length - total_reference_tokens
```

然后把 `local_token_ranges` 映射到 tail 区间中去。

这就是当前版本里 summary 写的 `tail-span assumption`。

### 为什么这是 WIP 但仍然值得先落地

因为这个版本虽然还不是最终形态，但已经满足两个重要目的：

- `Advanced` 产出的 metadata 已经真的被消费起来了
- 可以验证“按 reference 分别调权重”这条工程链路是不是通的

它适合作为：

- 原型验证
- 工作流试跑
- 后续升级到更精确 patch 的中间台阶

## 当前限制

当前实现有几个重要限制，需要明确记录：

## 1. 还不是严格意义上的投影后 K/V patch

当前逻辑是在 `attn2_patch` 中对 `context / value` 区段进行缩放。

它在工程效果上接近 reference weight 控制，但不是最严格的“投影后 key/value 张量 span 级调节”。

更激进的后续版本可以考虑：

- 使用 replace 类 patch
- 直接在投影后的 `k / v` 上改写

## 2. 依赖 tail-span 假设

当前 patch 假定 reference tokens 落在 context 序列尾部。

如果某个具体模型实现、某个 ComfyUI 版本，或者某个后续优化节点改变了 token 排列方式，那么：

- 权重可能落不到预期 reference 上
- 或者只部分生效

## 3. local span 不是绝对 span

当前 metadata 中记录的是 reference 区内的局部 span，不是整个 transformer 输入序列中的绝对 span。

这意味着当前版本对“reference 在总序列中的定位”仍然使用推断，而不是精确回填。

## 4. 没有做 layer-wise 或 block-wise 区分

当前 patch 没有区分：

- 不同 transformer block
- 不同 attention layer
- 不同时刻的不同 patch 策略

它是一个全局、统一的 reference weight 缩放方案。

## 推荐工作流

当前建议的接法是：

1. 文本 conditioning 正常生成
2. 输入 `EasyFlux2KleinConditionAdvanced`
3. 让它输出 `conditioning + latent`
4. 把同一个 `conditioning` 接给 `EasyFlux2KleinReferenceWeightPatch`
5. 用 patch 节点输出的 `model` 去进入后续采样器

概念上可以写成：

```text
text conditioning
    -> EasyFlux2KleinConditionAdvanced
    -> conditioning, latent

model
conditioning
    -> EasyFlux2KleinReferenceWeightPatch
    -> patched model

patched model + latent + conditioning
    -> sampler
```

## 为什么不把 patch 直接塞回 Advanced 节点

原因是职责不同：

- `Advanced` 本质上还是 conditioning/latent 准备节点
- `Patch` 本质上是 model 行为修改节点

把二者拆开更符合 ComfyUI 里的常见使用方式，也更方便调试：

- 你可以只用 `Advanced` 看 metadata
- 你可以把不同 patch 节点接在同一个 `conditioning` 上做实验
- 后续如果 patch 方案变动，不需要重写前面的 latent 准备逻辑

## 后续升级方向

后面如果要继续做强，可以沿着这几个方向升级：

## 1. 更精确地确定 reference 在总序列中的绝对 span

理想状态下，patch 能知道：

- text token 区间
- target token 区间
- reference token 区间

这样就能从 local span 升级到 absolute span。

## 2. 从 `attn2_patch` 升级到更底层的 replace/hook

如果后续能稳定拿到投影后的 `k / v`，可以把当前的“上下文缩放”升级成更直接的：

```python
k[..., start:end, :] *= weight
v[..., start:end, :] *= weight
```

这会更接近理论上的 per-reference K/V weight control。

## 3. 支持更丰富的权重输入方式

目前 `reference_weights` 还是字符串输入，后续可以考虑：

- 每张图单独 widget
- 支持更友好的批量编辑格式
- 支持从别的节点输入权重表

## 4. 引入更多调节维度

例如：

- 只调 `K`
- 只调 `V`
- 按 layer 范围调权重
- 按 reference 类型调不同策略

## 总结

这两个节点当前形成的是一个两段式方案：

1. `EasyFlux2KleinConditionAdvanced` 负责生成 reference latents，同时记录 per-reference weight 与 token span metadata
2. `EasyFlux2KleinReferenceWeightPatch` 负责读取 metadata，并在 cross-attention 中执行 reference 权重调节

它的价值不在于“已经是最终方案”，而在于：

- 路径清晰
- 职责分离
- 可以在 public repo 中公开保留一套可解释、可迭代的工程方案

如果后面要继续演进，更合理的方向不是推翻现有结构，而是在现有 metadata 协议基础上，把 patch 的定位精度和执行层级继续往下做深。
