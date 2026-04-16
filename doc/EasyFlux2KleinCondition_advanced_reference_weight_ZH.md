# EasyFlux2KleinConditionAdvanced 与 EasyFlux2KleinReferenceWeightControl

## 产品定义

这两个节点应被定义成一套“最小增量”的 reference weight 控制方案，而不是两套并行的 conditioning 系统。

`EasyFlux2KleinConditionAdvanced` 的产品定位是：

- 在 `EasyFlux2KleinCondition` 完全相同行为的基础上，额外输出一份可供 reference weight 控制节点使用的 span 协议
- 增加与动态 image 输入一一对应的 per-reference weight 输入
- 它不改变尺寸决策
- 它不改变 latent 路由
- 它不修改模型 attention
- 它只负责在 reference latent 构建完成时，把每张 reference 对应的 token 元数据整理出来

它存在的目的，是在不破坏基础节点稳定性的前提下，为 reference 权重控制提供标准化输入。

这份协议确认后，应直接作为当前产品定义固定下来，并在后续合并回 `EasyFlux2KleinCondition` 本体时沿用。

`EasyFlux2KleinReferenceWeightControl` 的产品定位是：

- 消费 `EasyFlux2KleinConditionAdvanced` 输出的 span 协议
- 在模型 attention 内对每张 reference 对应的 `K/V` 区段施加独立权重控制
- 透传与当前采样链绑定的 `conditioning`
- 它不负责 reference latent 生成
- 它不负责尺寸、mask、empty latent 或 masked latent 决策
- 它也不负责重新定义 `img_01` / `img_02+` 的语义

它的唯一职责，是把“reference span 协议”转化为运行时的 per-reference `K/V weight control`。

## 一句话定义

### `EasyFlux2KleinConditionAdvanced`

`EasyFlux2KleinCondition` 的稳定增量版，只增加 per-reference weight 输入，并额外输出 `reference_control`。

### `EasyFlux2KleinReferenceWeightControl`

基于该协议，在 attention 的 `K/V` 层面对不同 reference 执行独立权重控制，并原样透传 `conditioning`。

## 为什么这样定义

原因很直接：

- `EasyFlux2KleinCondition` 本身已经承担了尺寸决策、mask 路由、empty latent / masked latent 构造，以及 reference latent 追加逻辑
- 这些基础行为已经相对稳定，不应为了实验性的 reference weight 方案而被改写
- 因此 `Advanced` 最合理的定位不是“另一种 conditioning 节点”，而是“基础节点的协议增强版”

这意味着：

- `Advanced` 不应发展成一套长期并行的替代逻辑
- `Advanced` 的存在价值是以最小增量方式承载 reference span 协议
- 后续应合并回 `EasyFlux2KleinCondition`

## EasyFlux2KleinConditionAdvanced 的定义

### 产品目标

`EasyFlux2KleinConditionAdvanced` 应满足以下定义：

- 输入与 `EasyFlux2KleinCondition` 保持一致
- 基础语义与 `EasyFlux2KleinCondition` 保持一致
- 基础 routing 行为与 `EasyFlux2KleinCondition` 保持一致
- 标准输出与 `EasyFlux2KleinCondition` 保持一致
- 额外增加与动态 image 输入一一对应的 per-reference weight 输入
- 仅额外增加一份 `reference_control` 输出

### 必须保持一致的基础行为

`Advanced` 不应重新定义以下内容：

- `img_01` 为主图
- `img_02+` 为 reference-only 图
- `mask` 始终属于 `img_01`
- 相同的 ratio / megapixels 决策规则
- 相同的 empty latent / masked latent 路由规则
- 相同的 reference latent 追加顺序
- 相同的 `img_01_processed / width / height` 语义

换句话说：

`Advanced` 的前置行为应与 `EasyFlux2KleinCondition` 完全等价。

### 新增能力

`Advanced` 新增两项能力：

- 为每张动态 reference image 提供一个对应的 weight 输入
- 在 reference latent 构建过程中，输出一份供 reference weight 控制使用的 `reference_control`

这份协议需要描述：

- reference 的顺序
- reference 的名称
- 每张 reference 对应的 token 数
- 每张 reference 在 reference 区内部的 local token span
- 每张 reference 的基础权重

### 协议边界

`Advanced` 只负责输出 reference 区内部的 local span 协议，不负责输出 transformer 总序列中的 absolute span。

同时，`Advanced` 中新增的 weight 输入只用于定义每张 reference 的基础权重，并写入 `reference_control`。

因此 `Advanced` 不应承担以下职责：

- 不推测 text token 长度
- 不推测 target image token 长度
- 不推测 transformer 总序列 absolute span
- 不修改模型
- 不执行 patch

## EasyFlux2KleinReferenceWeightControl 的定义

### 产品目标

`EasyFlux2KleinReferenceWeightControl` 应满足以下定义：

- 输入一份来自 `Advanced` 的 reference span 协议
- 输入与当前采样链保持一致的 `conditioning`
- 输入一个 `MODEL`
- 在运行时通过 attention patch 将 local reference spans 映射为实际的 reference `K/V` 区段
- 对每张 reference 对应的 `K/V` 执行独立权重缩放
- 输出 patched model
- 原样输出 `conditioning`

### 它不负责什么

`ReferenceWeightControl` 不应承担以下职责：

- 不生成 reference latents
- 不决定图像尺寸
- 不处理 mask routing
- 不处理 empty latent / masked latent 构造
- 不改变 reference 图的编码逻辑
- 不再接收 manual per-reference weight 输入
- 不修改 `conditioning` 内容本身

它的职责边界应保持为：

- 消费协议
- 解释 span
- 改写 `K/V`
- 透传 `conditioning`

### 核心行为

`ReferenceWeightControl` 的核心行为应以如下语义为准：

```python
k[:, :, start:end, :] *= weight_i
v[:, :, start:end, :] *= weight_i
```

其中：

- `start:end` 指向单张 reference 对应的目标 span
- `weight_i` 是该 reference 的最终权重系数

当前实现约定为：

- 使用 `attn1_patch`
- 在 patch 回调中直接处理运行时的 `q / k / v`
- 依赖运行时提供的 `reference_image_num_tokens`
- 再结合 `reference_control.reference_token_counts` 与 `reference_token_ranges` 确定每张 reference 对应的 `K/V` 区段

因此，当前 attention patch 的关键点不是去猜测一个固定的 tail-span，而是：

- 先从运行时拿到 reference token 数量
- 再把 `Advanced` 记录的 local span 映射到实际的 reference token 区段
- 最后直接对该区段的 `k / v` 做缩放

如果运行时不能可靠提供 reference token 信息，就不能宣称节点实现了严格的 per-reference weight control。

## 两节点之间的职责边界

这两个节点之间的职责边界应固定为：

### `Advanced` 负责

- 生成 reference latents
- 保持基础 conditioning / latent 路由行为
- 记录 reference token metadata
- 输出 reference span 协议

### `ReferenceWeightControl` 负责

- 读取 span 协议
- 读取与当前采样链一致的 `conditioning`
- 在 `attn1_patch` 中读取运行时 reference token 信息
- 在运行时解释 local spans
- 映射到实际 attention 中的 reference `K/V` 区段
- 对目标 `K/V` 区间做权重改写
- 原样输出 `conditioning`，用于保持 workflow 链路完整

这种设计的结果是：

- `Advanced` 始终只是 `EasyFlux2KleinCondition` 的稳定增量版
- `ReferenceWeightControl` 始终只是一个独立的 model behavior modifier
- `Advanced` 可以自然合并回基础节点

## 协议定义建议

建议把 `Advanced` 的新增输出统一定义为 `reference_control`。

推荐语义如下：

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

这里的字段含义应明确为：

- `reference_names`：reference 输入顺序
- `reference_base_weights`：每张 reference 的基础权重
- `reference_token_counts`：每张 reference 对应的 token 数
- `reference_token_ranges`：每张 reference 在 reference 区内部的 local spans
- `total_reference_tokens`：所有 reference token 总数

这里的 `reference_token_ranges` 必须明确解释为：

- 它是 reference 区内部的 local spans
- 它不是 transformer 总序列中的 absolute spans

## 权重来源建议

`ReferenceWeightControl` 使用的每张 reference 权重，直接来自 `Advanced` 输出的 `reference_control.reference_base_weights`。

也就是说：

- 每张 reference 的基础权重在 `Advanced` 中定义
- 这些权重随 `reference_control` 一起传递给 `ReferenceWeightControl`
- `ReferenceWeightControl` 不再单独接收 manual per-reference weight 输入

## 推荐的最小实现原则

如果后续要正式实现，应坚持以下最小实现原则：

1. `Advanced` 与 `EasyFlux2KleinCondition` 的基础行为完全一致
2. `Advanced` 只新增与动态 image 输入一一对应的 per-reference weight 输入，以及 `reference_control` 输出
3. `Advanced` 只输出 local spans，不承担 absolute span 推导
4. `ReferenceWeightControl` 不再接 manual 权重输入，只消费 `reference_control`
5. `ReferenceWeightControl` 负责运行时 span 解释
6. `ReferenceWeightControl` 的最终动作是直接改写目标 `K/V`

## 总结

这套方案不是要建立两条并行的 conditioning 系统，而是要建立一条稳定主线加一条可合并的协议增强线：

- `EasyFlux2KleinCondition` 负责稳定的 conditioning / latent 行为
- `EasyFlux2KleinConditionAdvanced` 负责以最小增量方式提供 reference span 协议
- `EasyFlux2KleinReferenceWeightControl` 负责消费协议，并执行严格的 per-reference `K/V` 权重控制

最合理的终局是：

- 将 `Advanced` 的协议输出能力合并回 `EasyFlux2KleinCondition`
- 让 `ReferenceWeightControl` 继续作为独立节点存在

这样既能维持基础节点稳定，也能把 reference weight 控制设计成一套结构清晰、职责明确、可长期维护的方案。
