---
title: >-
  [论文解读] TransPixeler: Advancing Text-to-Video Generation with Transparency
description: >-
  [CVPR 2025][视频生成][RGBA视频生成] TransPixeler 提出在预训练的 DiT 视频生成模型中引入 alpha 通道 token，通过位置编码共享、域嵌入、部分 LoRA 微调和注意力掩码设计，在极少 RGBA 训练数据下实现高质量的 RGB 与 alpha 通道联合生成。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "RGBA视频生成"
  - "透明通道"
  - "Transformer"
  - "LoRA微调"
  - "注意力机制"
---

# TransPixeler: Advancing Text-to-Video Generation with Transparency

**会议**: CVPR 2025  
**arXiv**: [2501.03006](https://arxiv.org/abs/2501.03006)  
**代码**: [https://wileewang.github.io/TransPixeler/](https://wileewang.github.io/TransPixeler/)  
**领域**: 扩散模型 / 图像生成  
**关键词**: RGBA视频生成, 透明通道, 扩散Transformer, LoRA微调, 注意力机制

## 一句话总结
TransPixeler 提出在预训练的 DiT 视频生成模型中引入 alpha 通道 token，通过位置编码共享、域嵌入、部分 LoRA 微调和注意力掩码设计，在极少 RGBA 训练数据下实现高质量的 RGB 与 alpha 通道联合生成。

## 研究背景与动机

**领域现状**：文本到视频（T2V）生成模型已取得显著进展，基于 DiT 架构的模型（如 CogVideoX、Wan）能生成高质量视频。然而，这些模型只能输出 RGB 视频，不支持包含 alpha 透明通道的 RGBA 视频。

**现有痛点**：RGBA 视频在视觉特效（VFX）中至关重要——烟雾、反射、玻璃等透明元素需要 alpha 通道才能无缝合成到场景中。目前的解决方案有两类：(1) 先生成 RGB 再用 video matting 方法提取 alpha，但 matting 方法本身受限于 RGBA 数据稀缺，泛化能力差；(2) 类似 LayerDiffusion 的方法修改 VAE 来解码 alpha 通道，但 VAE 缺乏语义理解能力，在复杂场景下效果有限。这两类方法的共同问题是信息从 RGB 到 alpha 的单向流动，alpha 无法反过来影响 RGB 的生成。

**核心矛盾**：RGBA 视频训练数据极度稀缺（仅约 484 个视频），如果直接在如此少的数据上训练，生成的内容多样性将严重受限。因此需要最大化利用预训练 RGB 模型的能力，同时扩展支持 alpha 通道。

**本文目标**：在保持预训练 RGB 视频模型原有能力的前提下，扩展其同时生成 RGB 和 alpha 通道的能力，使生成内容超越有限 RGBA 训练集的范围。

**切入角度**：将 alpha 通道视为与 RGB 平行的 token 序列，通过在 DiT 的注意力机制中精心设计 token 间的交互方式，实现联合生成而非分步预测。

**核心 idea**：将 RGB token 序列加倍为 RGB+Alpha 双域序列，alpha token 复用 RGB 的位置编码并加入可学习域嵌入区分两个域，仅对 alpha token 的 QKV 投影应用 LoRA 适配，同时用注意力掩码阻止 text-to-alpha 的直接注意力，最大化保留预训练模型的 RGB 生成能力。

## 方法详解

### 整体框架
TransPixeler 基于 DiT 架构的视频生成模型（如 CogVideoX）。输入为文本提示，输出为 RGB 视频和对应的 alpha 视频。在模型内部，将原本长度为 $L$ 的视频 token 序列扩展为 $2L$，前 $L$ 个解码为 RGB 视频，后 $L$ 个解码为 alpha 视频。文本 token 仍然前置拼接在视频 token 前。整个序列 $[\text{text}; \text{RGB}; \text{alpha}]$ 通过带注意力掩码的全自注意力进行处理。

### 关键设计

1. **位置编码共享与域嵌入（Shared Positional Encoding + Domain Embedding）**:

    - 功能：让 alpha token 与对应的 RGB token 在空间-时间上对齐，同时让模型能区分两个域。
    - 核心思路：alpha token 不使用连续递增的位置索引（即不从 $L+1$ 到 $2L$），而是复用 RGB token 的位置编码（从 $1$ 到 $L$），使每帧的 RGB 和 alpha token 共享相同的时空位置信息。额外引入一个零初始化的可学习域嵌入 $d$ 加到 alpha token 上，用于区分两个域。公式为：$\mathbf{f}^*(\mathbf{x}^m_{\text{video}}) = \mathbf{W}^*(\mathbf{x}^m + \mathbf{p}^{m-L} + d)$，其中 $m > L$ 表示 alpha token。
    - 设计动机：实验发现如果使用连续位置编码，模型会将 alpha 序列视为 RGB 视频的"后续帧"而非独立域，导致两者生成相似内容。共享位置编码消除了空间-时间对齐的学习难度，加速收敛（1000 次迭代即可初步收敛）。

2. **部分 LoRA 微调（Partial LoRA Fine-tuning）**:

    - 功能：在保持 RGB 生成质量的前提下适配 alpha 通道的生成。
    - 核心思路：仅对 alpha token（$m > L$）的 QKV 投影层应用 LoRA 适配：$\mathbf{W}^*(\cdot) = \mathbf{W}(\cdot) + \gamma \cdot \text{LoRA}(\cdot)$，其中 $\gamma$ 控制残差强度。RGB token 和 text token 的 QKV 投影保持冻结，完全沿用预训练权重。这意味着在注意力矩阵的 $3 \times 3$ 分组中，text-to-RGB 和 RGB-to-text 的计算与原始模型完全一致。
    - 设计动机：全参数微调在仅 484 个视频上很容易过拟合并破坏原始 RGB 能力。LoRA 仅适用于 alpha 域意味着 RGB 输出不受影响，模型可以自由生成超出训练集分布的 RGB 内容，alpha 通道随之适配。

3. **自适应注意力掩码（Attention Mask Design）**:

    - 功能：控制 text、RGB、alpha 三组 token 之间的注意力交互，去除有害交互、保留有益交互。
    - 核心思路：构建注意力掩码 $\mathbf{M}^*_{mn}$，当 $m \leq L_{\text{text}}$ 且 $n > L_{\text{text}} + L$ 时设为 $-\infty$（即阻断 text-attend-to-alpha），其余为 $0$（允许）。这意味着：text↔RGB 保持不变（保留原模型能力）；RGB-attend-to-Alpha 被允许（RGB 可以根据 alpha 信息调整自身，增强对齐）；alpha-attend-to-RGB 被允许（alpha 可以从 RGB 获取语义信息）；text-attend-to-alpha 被阻断（防止有限训练数据对文本表示的污染）。
    - 设计动机：作者系统分析了 $3 \times 3$ 注意力矩阵中每一项的作用。关键发现是 RGB-attend-to-Alpha 是必需的——它使 RGB token 能根据 alpha 信息调整自身，改善两者的对齐。而 text-attend-to-alpha 则有害，因为少量 RGBA 数据不足以让文本 token 学会如何解读 alpha 信息，反而可能污染文本表示。

### 损失函数 / 训练策略
使用 flow matching 或标准扩散过程训练。训练数据仅约 484 个 RGBA 视频。通过上述设计，可训练参数极少（仅 alpha LoRA + 域嵌入），使得在极小数据集上也能有效训练而不过拟合。

## 实验关键数据

### 主实验
在 RGBA 视频生成任务上与多种基线方法对比（Video Matting、Marigold-style 预测等方法）：

| 方法 | Alpha MAE↓ | RGBA SSIM↑ | Gen Diversity | RGB 保持 |
|------|----------|----------|--------------|---------|
| RVM (matting) | 较高 | 较低 | 受限于matting能力 | 不影响 |
| Marigold-style | 中等 | 中等 | RGB-alpha 不对齐 | 不影响 |
| LayerDiffusion | 中等 | 中等 | 受 VAE 限制 | 部分降级 |
| **TransPixeler** | **最低** | **最高** | **最丰富** | **完全保持** |

### 消融实验

| 配置 | Alpha 质量 | RGB-Alpha 对齐 | RGB 保持 |
|------|----------|--------------|---------|
| 连续位置编码 | 差（收敛慢） | 差 | 差（alpha影响RGB） |
| 共享位置编码 | 好（快速收敛） | 好 | 好 |
| w/o 域嵌入 | 中等（两域混淆） | 中等 | 中等 |
| w/ 域嵌入 | 好 | 好 | 好 |
| 允许 text-to-alpha | 差（性能退化） | 差 | 差 |
| 阻断 text-to-alpha | 好 | 好 | 好 |
| w/o RGB-to-alpha | 中等 | 差（不对齐） | 好 |
| w/ RGB-to-alpha | 好 | 好 | 好 |

### 关键发现
- 位置编码设计是收敛速度的关键：共享位置编码在 1000 步即可初步收敛，连续编码需要更多步数且效果更差
- RGB-attend-to-Alpha 注意力是保证 RGB-Alpha 对齐的关键——这是 TransPixeler 超越"先生成后预测"方案的核心原因
- 阻断 text-attend-to-alpha 对保持原始 RGB 生成质量至关重要，否则有限训练数据会污染文本表示
- 仅使用约 484 个 RGBA 视频训练，TransPixeler 就能生成训练集未曾出现过的多样化 RGBA 内容

## 亮点与洞察
- **token 域扩展范式**：通过序列加倍+域嵌入+部分 LoRA 的方式扩展预训练模型到新模态，这一范式可以迁移到其他"RGB+X"的联合生成任务（深度、法线、光流等），非常通用且优雅
- **注意力交互的系统分析**：$3 \times 3$ 分组注意力矩阵的逐项分析提供了深刻的洞察——哪些信息流是有益的、哪些是有害的，这种分析方法论可以用于其他多域联合生成场景
- **极少数据下的有效微调**：仅 484 个视频就能训练出多样化的 RGBA 视频生成，得益于精心设计的架构使得预训练知识最大化保留

## 局限与展望
- 依赖 RGB VAE 解码 alpha 通道（将 alpha 视为灰度图），可能在边缘精度上存在局限
- 目前仅支持 text-to-RGBA，未扩展到 image-to-RGBA 或 video editing
- RGBA 训练集太小（484个），在某些特定类型的透明物体（如复杂玻璃折射）上可能效果不佳
- 序列长度翻倍带来的计算开销需要考虑

## 相关工作与启发
- **vs LayerDiffusion**: LayerDiffusion 修改 VAE 来解码 alpha，但 VAE 缺乏语义理解。TransPixeler 在 DiT 层面联合生成，语义理解更强
- **vs Marigold/Lotus**: Marigold 等方法是先生成 RGB 再预测深度/alpha，信息单向流动。TransPixeler 的双向注意力（RGB-to-alpha 和 alpha-to-RGB 都开启）实现更好的对齐
- **vs Video Matting**: matting 方法受限于自身训练数据的覆盖范围，TransPixeler 利用预训练 RGB 模型的知识实现更广泛的泛化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个 DiT 架构的 RGBA 视频联合生成方法，注意力分析提供了深刻洞察
- 实验充分度: ⭐⭐⭐⭐ 消融实验充分验证了每个设计选择，但定量指标的基线方法有限
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，注意力机制的分析由浅入深，图示直观
- 价值: ⭐⭐⭐⭐ 解决了 VFX 领域的实际需求，token 域扩展范式有广泛迁移价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models](shotadapter_text-to-multi-shot_video_generation_with_diffusion_models.md)
- [\[CVPR 2025\] Identity-Preserving Text-to-Video Generation by Frequency Decomposition](identity-preserving_text-to-video_generation_by_frequency_decomposition.md)
- [\[CVPR 2025\] Can Text-to-Video Generation Help Video-Language Alignment?](can_text-to-video_generation_help_video-language_alignment.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] The Devil is in the Prompts: Retrieval-Augmented Prompt Optimization for Text-to-Video Generation](the_devil_is_in_the_prompts_retrieval-augmented_prompt_optimization_for_text-to-.md)

</div>

<!-- RELATED:END -->
