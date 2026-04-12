# EasyFlux2KleinCondition Workflow Integration

## 目的

本文档面向需要通过修改 workflow JSON 来驱动 ComfyUI 工作流的开发者。

目标是说明如何在 workflow 中定位 `EasyFlux2KleinCondition` 节点，以及如何安全地修改它的输入值。

## 推荐接入方式

如果你的前端是直接编辑 workflow JSON，最稳妥的方式是：

1. 读取一份准备好的 workflow 模板
2. 定位 `EasyFlux2KleinCondition` 节点
3. 修改它的 widget 值和连线输入
4. 将修改后的 workflow 提交给 ComfyUI

## 如何定位节点

推荐的定位方式：

1. 通过节点 `type`
   `EasyFlux2KleinCondition`

2. 通过固定的 workflow 节点 `id`
   这适用于模板固定不变的场景

3. 结合 `type` 和你们自己约定的标题或命名方式

如果可以，建议优先采用：

- 固定 workflow 模板
- 固定节点 id
- 固定输入连线约定

## 可以通过 Workflow JSON 修改的内容

当前这个节点适合通过前端修改 workflow JSON 的字段包括：

- `ratio`
- `megapixels`
- `batch_size`
- `upscale_method`
- 图片输入连线
- `mask` 输入连线

## 当前 Widget 字段

该节点当前暴露的 widget 参数为：

1. `ratio`
2. `megapixels`
3. `batch_size`
4. `upscale_method`

在 workflow JSON 中，这些值通常存放在 `widgets_values` 中。

注意：

- 当前顺序应视为现版本节点的一部分约定
- 如果将来节点 UI 调整，widget 顺序也可能变化
- 用于正式集成时，建议固定 workflow 模板版本

## 当前合法 `ratio` 值

当前合法值为：

- `default`
- `1:1`
- `4:3`
- `3:4`
- `3:2`
- `2:3`
- `5:4`
- `4:5`
- `16:9`
- `9:16`
- `2:1`
- `1:2`
- `21:9`
- `9:21`
- `4:1`
- `1:4`

## 当前合法 `megapixels` 值

当前合法值为：

- `default`
- `1.00`
- `1.50`
- `2.00`
- `3.00`
- `4.00`
- `6.00`
- `8.00`

## 输入语义

### 图片输入

该节点使用有序图片输入：

- `img_01`
- `img_02`
- `img_03`
- ...

所有接入的图片都会按顺序追加为 reference latents。

其中 `img_01` 还有额外语义：

- 它是默认的尺寸参考图
- 当接入 `mask` 时，它是 base image

### Mask

`mask` 永远作用于 `img_01`。

如果接入了 `mask`：

- 节点会进入 masked latent 模式
- 同时必须接入 `img_01`

## 尺寸规则

### 固定 Ratio

如果 `ratio != default`：

- 固定 ratio 一律走 bucket 规则
- `img_01` 不决定最终输出比例
- `megapixels=default` 等效于 `1.00`

### 默认 Ratio

如果 `ratio = default`：

- 有 `img_01` 时，按 `img_01` 的宽高比规则处理
- 没有 `img_01` 时，回退到 `1:1`

### `megapixels = default`

当 `ratio = default` 时：

- 没有 `img_01` -> 等效于 `1.00`
- `img_01 <= 4.2 MP` -> 直接使用 `img_01` 原始宽高
- `img_01 > 4.2 MP` -> 按 `4.00 MP` 封顶，同时保持原图比例

## 输出

该节点输出：

- `conditioning`
- `latent`
- `noise_mask`
- `mask`
- `img_01_processed`
- `width`
- `height`

其中 `img_01_processed` 是节点内部真实用于 latent 尺寸决策和编码的处理后 `img_01`。

## 集成说明

对外部前端来说：

- 当前版本只要写入合法 widget 值，就已经可以通过 workflow 驱动
- 不需要真的操作 ComfyUI 前端界面
- 只需要正确修改 workflow JSON

如果前端需要超出当前合法 widget 列表的值，则需要先更新后端节点定义。

## 建议实践

- 保持一份稳定的 workflow 模板
- 在模板中记录目标节点 id
- 只修改约定好的 widget 值和输入连线
- 尽量不要依赖用户随意生成 workflow 后出现的不稳定节点 id
