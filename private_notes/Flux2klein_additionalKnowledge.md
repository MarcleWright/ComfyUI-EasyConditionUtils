# Flux2Klein Additional Knowledge

这份文档用于整理和校正与 `FLUX.1 Kontext / FLUX.2 Klein` 相关的补充理解，重点关注 `reference latent`、尺寸策略、mask 路线，以及多参考图控制。

## 1. Reference latent 的本质

### 结论

`FLUX.1 Kontext / FLUX.2 Klein` 中的参考图更接近“视觉条件”而不是传统 img2img 里的“起始 latent”。

更准确地说：

- 参考图会先经过 VAE 编码，变成 latent token
- 这些 token 作为上下文条件进入模型
- 目标图像的生成仍然是一个独立的生成过程
- 模型通过 attention 在目标 token 和参考 token 之间建立关系

### 依据

`FLUX.1 Kontext` 论文明确说明：

- context image 被编码成 latent tokens
- 这些 context tokens 会追加到图像 token 序列中
- 通过 3D RoPE 的时间偏移区分 target token 和 context token

因此，高层理解上可以认为：

- 传统 img2img：参考图更像“生成起点”
- Kontext/Klein：参考图更像“视觉 prompt”

但需要注意，不能把这句话绝对化成“永远不是 latent 路线”，因为在具体工作流里，是否还存在 base latent、mask latent、noise mask，是 workflow 设计层面的事情，不完全等于模型论文里的 conditioning 机制。

## 2. Reference 分辨率应该怎么处理

### 结论

参考图通常不应该直接使用超大原图，而应该优先缩放到接近模型训练时使用的标准尺寸。

这条结论是成立的，而且有官方工程依据。

### 依据

ComfyUI 官方 `FluxKontextImageScale` 文档说明：

- 该节点会把输入图缩放到 Flux Kontext 训练时使用的最佳尺寸
- 过大的输入可能导致输出质量下降
- 甚至可能出现多个主体

这说明：

- 更高 reference 分辨率不等于模型一定能更好利用细节
- 超过训练习惯范围后，质量和稳定性可能反而下降

### 实际建议

- reference 图优先缩放到 `FluxKontextImageScale` 的训练预设尺寸
- 不要默认把 reference 保持为 4MP 甚至更高
- 输出图可以比 1MP 大，但 reference 图未必需要同步变大

## 3. 关于 token 数和分辨率的关系

### 可以保留的理解

分辨率越高，参考图对应的 latent token 数通常越多，因此 attention 的计算量、显存占用和运行时间都会上升。

这是合理的工程判断。

### 需要修正的地方

原文把 VAE 下采样写成了 `8x`，这在当前 `FLUX.2 Klein` 语境里并不稳妥。

更适合采用 ComfyUI 当前官方节点语义来描述：

- `EmptyFlux2LatentImage` 的 latent 宽高是 `width / 16`、`height / 16`
- 输出 shape 为 `[batch_size, 128, height // 16, width // 16]`

因此，如果我们在当前节点设计和 ComfyUI 实现语境下讨论 token 数，应该按 `/16` 的 latent 尺寸逻辑去理解，而不是写死 `/8`。

### 更稳的表述

- 分辨率越高，latent 空间越大
- latent token 越多，reference 进入 attention 的成本越高
- 但更多 token 不必然等于模型能更准确提取参考细节

## 4. FLUX1 / 传统 img2img 与 Flux Kontext / Klein 的差异

### 总结

这两套体系对“参考图”的使用哲学确实不同，这个判断基本正确。

传统 img2img 更像：

- 先把图编码成 latent
- 再加噪
- 从“被扰动的原图 latent”开始采样

Kontext / Klein 更像：

- 把参考图作为视觉上下文条件
- 目标图像通过 target token 生成
- 模型在 attention 中利用参考图信息

### 但要避免过度绝对化

原文里“`KSampler` 始终从纯噪声开始、`denoise=1.0` 固定”这个说法不适合作为普遍真理。

原因：

- 这是某些工作流的习惯，不是所有 Klein 工作流的唯一方式
- 在你当前节点设计里，只要接入 `mask`，就会走“编码后的 `img_01` + noise_mask”的 masked latent 路线
- 也就是说，workflow 层仍然可能存在“以图像编码 latent 为基础”的生成入口

所以更准确的说法应该是：

- `reference latent` 本身不是传统 img2img 那种“偏离原图程度由 denoise 控制的初始化图像 latent”
- 但整个工作流是否使用空 latent、masked latent、noise mask，仍然要看节点和采样链路设计

## 5. Mask 与 inpaint 路线

### 你当前项目语境下的准确理解

在 `EasyFlux2KleinCondition` 的设计里：

- 如果连接了 `mask`
- 就必须同时连接 `img_01`
- 节点会优先走 masked latent 路线
- `img_01` 会先按最终尺寸规则处理
- 然后编码为 latent
- `mask` 也会缩放到同尺寸
- 最终输出 latent + `noise_mask`

所以这里的重点不是“是不是传统 inpaint conditioning 节点”，而是：

- mask 路线是存在的
- 而且它优先级最高
- 它和 reference latent 追加是两件并行但不同层面的事情

### 更稳的表述

- 没有 mask：通常走 empty latent 路线
- 有 mask：走基于 `img_01` 编码结果的 masked latent 路线
- `reference latent` 是 conditioning 侧的参考信息
- `noise_mask` / masked latent 是采样起点侧的结构约束

## 6. FLUX.2 Klein 的尺寸上限该怎么理解

### 可以确认的部分

ComfyUI 官方 `EmptyFlux2LatentImage` 文档写明：

- `width` 范围是 `16` 到 `8192`
- `height` 范围是 `16` 到 `8192`
- 必须是 `16` 的倍数
- 输出 latent 的 shape 是 `[batch, 128, h/16, w/16]`

这说明：

- 节点参数层面的理论上限确实可以到 `8192 x 8192`

### 需要修正的部分

原文把 `FluxKontextImageScale` 的训练预设理解成“最大约 `1568 x 1568`，约 `2.5MP`”，这是不准确的。

实际情况是：

- `FluxKontextImageScale` 给的是一组按宽高比分桶的训练预设
- 例如 `1568 x 672`、`1456 x 720`、`1024 x 1024`
- 这些尺寸整体都大致在 1MP 级别附近

因此更准确的结论是：

- 训练参考尺寸大致是 1MP 级别的 bucket 集合
- 不是“最大训练预设约 2.5MP”

### 实用建议

- 理论可设很大，不代表模型在极大尺寸下仍稳定
- 工程上更应该把“节点允许范围”和“模型稳定工作范围”分开理解
- 如果追求稳健，优先采用官方 bucket 体系和保守的输出尺寸

## 7. 通过缩小不同 reference 图分辨率来控制权重，是否合理

### 部分成立

从工程直觉上说，降低某张参考图分辨率，会降低它携带的信息量，也可能减少它对应的 token 数，因此确实可能间接削弱它的影响。

所以“缩小图像可能会削弱某个 reference 的影响”这个方向不是完全错的。

### 但它不是理想控制方式

原因主要有三点：

- 它会同时损失这张图的细节
- 控制粒度粗糙，本质上是用信息损失换权重变化
- 很容易把输入拉出模型更熟悉的训练分辨率区间

### 还需要修正的一句

原文里说“不能分别控制 B 和 C”，这个不准确。

因为你完全可以：

- 把 B 缩成一个尺寸
- 把 C 缩成另一个尺寸

所以“不能分别控制”不成立。

真正的问题是：

- 这种控制方式不精确
- 副作用大
- 不利于稳定复现

## 8. 更理想的多参考图权重控制方式

### 结论

如果已经有能直接作用于 reference token / attention 的控制节点，那么通常比“改分辨率”更合理。

你提到的第三方思路大方向是可信的：

- 先把 reference 图都缩放到合理训练 bucket
- 再在模型层面对不同 reference 的影响力做单独调节

### 需要加上的限定

这类做法目前更适合表述为：

- 第三方增强节点的经验性工程方案
- 不是 BFL 官方论文直接定义的标准 API

因此写法上最好避免说成“正确做法只有这一种”，更适合写成：

- 如果工作流里已经接入这类 reference-weight 控制节点，这通常是更精确、且副作用更小的方案

## 9. 什么是 Attention 里的 K / V

### 基本概念

在 attention 里，通常会同时出现 `Q`、`K`、`V`：

- `Q` = Query，当前 token 想找什么
- `K` = Key，每个 token 以什么方式被匹配到
- `V` = Value，每个 token 真正提供什么内容

可以把它类比成检索系统：

- `Q` 像搜索词
- `K` 像每条资料的标签
- `V` 像资料正文

attention 的高层过程可以理解为：

1. 当前生成中的 token 先产生一个 `Q`
2. 用这个 `Q` 去和所有 token 的 `K` 计算相似度
3. 相似度更高的 token，会把自己的 `V` 更多地贡献出来
4. 所有 `V` 按权重聚合，形成当前 token 更新时使用的信息

公式可以粗略写成：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

其中：

- `QK^T` 决定“当前更关注谁”
- `V` 决定“被关注后输出什么信息”

### 在 reference latent 权重控制里的意义

如果某一张 reference 图对应一段独立 token 区间，那么对那一段 token 的 `K / V` 做缩放，就相当于直接控制它在 attention 中的影响力。

例如：

```python
k[:, :, start:end, :] *= weight
v[:, :, start:end, :] *= weight
```

这段操作的效果可以理解为：

- `weight < 1.0`：这张 reference 仍然存在，但“说话声音变小了”
- `weight > 1.0`：这张 reference 更容易被注意到，而且输出的信息也更强

更直观地记：

- `K` 更像“我有多容易被注意到”
- `V` 更像“我被注意到后能输出多少内容”

### 为什么通常同时改 K 和 V

- 只改 `K`：主要影响“被不被注意到”
- 只改 `V`：主要影响“注意到后输出多少”
- 同时改 `K / V`：更接近一个完整的“reference weight”旋钮

这也是为什么，从机制上说，attention 层的 `K / V` 控制比“直接缩小图片”或“直接乘 latent”更像真正的 reference 权重控制。

### 为什么输出是 patched model，而不是“修改后的 conditioning”

这里有一个很关键的分层理解：

- `conditioning` 决定“有哪些 token / latent / reference 被送进模型”
- `k[:, :, start:end, :] *= weight_i` 决定“模型在运行 attention 时，如何使用这些 token”

这两者不是同一层面的东西。

`conditioning` 更像“输入材料”：

- 它告诉模型这次有哪些文本条件
- 有哪些 reference latents
- 这些 reference 的顺序和 span 信息是什么

但它本身不执行 attention 计算，也不直接持有真正参与 attention 的 `k`、`v` 张量。

真正的 `k`、`v` 是在模型前向传播时，由当前 token / context 经过 projection 临时计算出来的。

可以把过程粗略理解为：

1. `conditioning` 和其它输入先进入模型
2. 模型在某一层 attention 中把输入投影成 `q / k / v`
3. patch 在这个时刻拦截到运行时的 `k / v`
4. 对属于某张 reference 的 `start:end` 区间做缩放
5. attention 结果因此发生变化

所以：

- 如果改的是 `conditioning`，本质上是在改“给模型什么”
- 如果改的是 `k / v`，本质上是在改“模型如何处理已经给到它的东西”

reference weight control 属于后者。

这也是为什么这类节点的输出应该是 `patched model`，而不是“修改后的 conditioning”。

更准确地说，这里的 `patched model` 也不是“离线改写了参数权重的模型”，而是：

- 一个被挂接了额外行为的模型对象
- 它会在运行到 attention 对应位置时，对运行时生成的 `k / v` 做额外处理

因此：

- `conditioning` 负责提供 reference 和其它条件
- `reference_control` 负责描述 reference 对应的 span 协议
- `patched model` 负责在真正执行 attention 时，按这份协议去改写 `k / v`

### 当前项目里的 attention patch 实现要点

在当前项目语境里，reference weight control 的实现重点应理解为：

- 使用 `attn1_patch`
- 在 patch 回调中直接拿到运行时的 `q / k / v`
- 优先依赖运行时提供的 `reference_image_num_tokens`
- 再结合 `reference_control.reference_token_counts` 与 `reference_token_ranges`
- 最终对 reference 对应区段的 `k / v` 直接做缩放

更准确地说，当前有效路径应理解为：

1. `Advanced` 输出 `reference_control`
2. `ReferenceWeightControl` 把 `model` 变成 `patched model`
3. sampler 执行模型时，`attn1_patch` 拿到运行时 `q / k / v`
4. patch 根据运行时 reference token 数量定位每张 reference 的区段
5. 对对应的 `k / v` 执行 `*= weight`

这比旧的“`attn2_patch` + `context/value` 缩放”更接近真正的 per-reference `K / V` weight control。

### 当前实验现象

在 `put the woman in image 2 in image 1` 这类案例中，仅调整 `img_02_weight`，已经观察到如下连续变化：

- `0 ~ 0.7`：基本表现为同一个 another woman
- `0.8`：another woman，但衣着开始接近 `img_02`
- `0.85`：another woman，但衣着已经和 `img_02` 相同
- `0.9`：开始稳定表现为 `img_02` 的同一女人
- `1.0`：表现为 `img_02` 的同一女人，并会额外带入一点 `img_02` 的环境颜色

这说明至少从工程效果上看，当前权重控制已经不是“无效开关”，而是会连续影响 reference 信息注入强度。

### LoRA 与 ReferenceWeightControl 的关系

如果工作流中还接入了 LoRA，需要把两者理解成不同层级的叠加，而不是互相替代：

- LoRA 修改的是模型本身的行为分布
- `ReferenceWeightControl` 修改的是运行时 reference 对应 attention `K / V` 区段的权重

因此更合适的理解是：

- LoRA 决定模型更倾向于“怎么画”
- reference weight control 决定模型有多强地参考某一张 reference

如果先对模型应用 LoRA，再把这个模型送入 `ReferenceWeightControl`，那么最终 sampler 使用的是：

- 已经带有 LoRA 的模型
- 再叠加 reference attention patch 后得到的 patched model

这意味着：

- `ReferenceWeightControl` 仍然可能有效
- 但同一个 weight 数值的“手感”可能会被 LoRA 改变

例如：

- 没有 LoRA 时，也许 `0.9` 才能稳定进入同一人物
- 有 LoRA 时，可能 `0.8` 就足够
- 也可能反过来，需要更高 weight 才能压过 LoRA 的风格偏置

因此，在理解实验结果时应区分两类情况：

- 无 LoRA 的基准测试：用于判断 reference weight control 本身是否有效
- 有 LoRA 的叠加测试：用于判断真实生产工作流中的表现

更稳的工程结论是：

- LoRA 不会从原理上否定 `ReferenceWeightControl`
- 但它会改变 weight 的响应阈值、影响强度以及最终视觉表现
- 所以即使同属 `FLUX.2 Klein` 工作流，也应分别记录“无 LoRA”和“有 LoRA”的权重手感

## 10. Reference token 在序列中的排列方式

### 一张 reference 图为什么会变成一串 token

图像不会以“整张图”的形式直接进入 transformer，而是会先经历类似这样的过程：

1. 图片经过 VAE 编码
2. 得到 latent 表示
3. latent 被展开或切分成一串图像 token
4. 再投影到 transformer 使用的内部表示空间

因此，一张 reference 图进入模型后，通常不是一个单点，而是一整段连续 token。

可以概念化写成：

```text
ref_1 -> [r1_0, r1_1, r1_2, ..., r1_n]
ref_2 -> [r2_0, r2_1, r2_2, ..., r2_m]
ref_3 -> [r3_0, r3_1, r3_2, ..., r3_k]
```

### 在 Flux / Klein 里的高层理解

根据 `FLUX.1 Kontext` 论文的高层描述，context image tokens 会与 target image tokens 一起进入 visual stream，并通过位置 / 时间区分机制让模型知道哪些是 target、哪些是 context。

在工程实现上，图像流的排列通常可以近似理解为：

```text
[ target image tokens | ref_1 tokens | ref_2 tokens | ref_3 tokens ]
```

如果再把文本一起考虑，总序列在某些阶段可能近似为：

```text
[ text tokens | target image tokens | ref_1 tokens | ref_2 tokens | ref_3 tokens ]
```

具体顺序会受实现细节影响，但有一条稳定认识：

- 每张 reference 通常都会对应一段连续、可定位的 token span

### 为什么 token span 很重要

如果想对某一张 reference 单独调权重，关键不在于“乘不乘一个数”，而在于：

- 能不能准确知道这张 reference 对应的是哪一段 token 区间

只有知道了：

- `ref_1` 的起止位置
- `ref_2` 的起止位置
- `ref_3` 的起止位置

才能做到：

- 只调 `img_02`
- 不误伤 `img_01`
- 不影响 target image tokens
- 不影响 text tokens

实现上通常会表现为：

```python
k[:, :, start:end, :] *= weight_i
v[:, :, start:end, :] *= weight_i
```

### 为什么不同 reference 的 token 数可能不同

如果不同 reference 图的最终 latent 尺寸不同，那么它们对应的 token 数通常也会不同。

不过在当前项目里，reference 图会先统一走标准化尺寸策略，这带来两个好处：

- token 数不会乱得太离谱
- 更容易在构建 `reference_latents` 时记录每张图各自的 token 数

### 对实现最有价值的元数据

如果后续要做真正的 per-reference weight 控制，建议在 reference latent 构建阶段就保留这类信息：

```python
reference_latents = [ref1, ref2, ref3]
reference_token_counts = [n1, n2, n3]
reference_names = ["img_01", "img_02", "img_03"]
```

或者在可行时，直接记录：

```python
reference_token_ranges = [
    (start1, end1),
    (start2, end2),
    (start3, end3),
]
```

因为一旦能准确映射每张 reference 的 token span，attention 层的单独权重控制才真正可做。

## 11. 对当前项目最有价值的稳定结论

如果把这些信息收束到 `EasyFlux2KleinCondition` 节点设计里，最重要的结论是：

1. `img_01` 的角色必须和 `img_02+` 分开理解
2. `img_01` 既可能参与尺寸决策，也可能在 mask 路线中作为 base latent 来源
3. 所有接入图像都可以作为 reference latent 追加到 conditioning
4. reference 图不应默认保留超高分辨率，最好对齐训练 bucket 思路
5. “reference conditioning” 和 “latent 起点构造” 是两条不同但会同时存在的逻辑线
6. 控制不同 reference 影响力时，优先考虑显式控制机制，而不是单纯靠压缩分辨率

## 12. 参考依据

- FLUX.1 Kontext 论文：说明 context image token 的拼接方式与 3D RoPE 区分机制  
  <https://arxiv.org/abs/2506.15742>

- FLUX.1 Kontext 论文 HTML 版本  
  <https://ar5iv.labs.arxiv.org/html/2506.15742v2>

- ComfyUI `FluxKontextImageScale` 官方文档：说明训练预设尺寸与 oversized input 风险  
  <https://docs.comfy.org/built-in-nodes/FluxKontextImageScale>

- ComfyUI `EmptyFlux2LatentImage` 官方文档：说明 `FLUX.2` latent 的尺寸规则与 shape  
  <https://docs.comfy.org/built-in-nodes/EmptyFlux2LatentImage>

- 第三方参考：`ComfyUI-Flux2Klein-Enhancer`，用于说明“按 reference 单独调权”这类工程实践确实存在，但应视为第三方方案  
  <https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer>
