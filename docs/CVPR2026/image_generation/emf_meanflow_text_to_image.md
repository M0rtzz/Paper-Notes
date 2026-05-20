---
title: >-
  [论文解读] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation
description: >-
  [CVPR 2026][图像生成][MeanFlow] 首次将 MeanFlow 框架从类别标签条件扩展到文本条件图像生成，发现限制步数下文本表示的语义区分性和解耦性是关键瓶颈，基于 BLIP3o-NEXT 文本编码器实现了高质量的少步/单步 T2I 生成。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "MeanFlow"
  - "单步生成"
  - "文本到图像"
  - "文本编码器"
  - "语义区分性"
---

# Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation

**会议**: CVPR 2026  
**arXiv**: [2604.18168](https://arxiv.org/abs/2604.18168)  
**代码**: [https://github.com/AMAP-ML/EMF](https://github.com/AMAP-ML/EMF)  
**领域**: 图像生成  
**关键词**: MeanFlow, 单步生成, 文本到图像, 文本编码器, 语义区分性

## 一句话总结

首次将 MeanFlow 框架从类别标签条件扩展到文本条件图像生成，发现限制步数下文本表示的语义区分性和解耦性是关键瓶颈，基于 BLIP3o-NEXT 文本编码器实现了高质量的少步/单步 T2I 生成。

## 研究背景与动机

**领域现状**：MeanFlow 是一种有理论基础的 flow matching 加速方法，通过学习两个时间点之间的平均速度场实现单步生成，在 ImageNet 类别条件生成上取得了与标准多步模型媲美的效果。后续工作（如改进训练策略和架构）也主要集中在类别条件设定下。

**现有痛点**：将 MeanFlow 从固定类别标签扩展到灵活文本输入看似直接，实际上困难重重。直接将 LLM 文本编码器接入 MeanFlow 框架并使用常规训练策略，效果令人失望。JVP 项的稳定性问题被反复认定为将 consistency 类方法扩展到大规模 T2I 的主要瓶颈。

**核心矛盾**：类别标签是离散且易于区分的条件信号，而文本条件是连续且语义复杂的。在极少步（如单步）推理中，模型几乎没有机会通过多次去噪来修正语义偏差，因此对条件信号的质量要求极高。

**本文目标**：(1) 理解为什么某些文本编码器在少步设定下失败；(2) 识别高质量文本表示应具备的关键属性；(3) 基于这些发现实现首个有效的文本条件 MeanFlow 生成模型。

**切入角度**：作者对比了不同文本编码器在限制推理步数时的表现差异，发现 BLIP3o-NEXT 的文本编码器即使在单步时也能保持基本语义完整性，而 SANA-1.5 的编码器在少步时语义严重退化。

**核心 idea**：高质量文本表示需要两个核心属性——区分性（discriminability，区分细微语义差异）和解耦性（disentanglement，保持文本的语言结构），具备这两个属性的编码器才能构建可靠的速度场方向，使少步甚至单步生成成为可能。

## 方法详解

### 整体框架

基于预训练的 BLIP3o-NEXT 扩散模型，将其适配为 MeanFlow 框架。关键修改是引入双时间嵌入层：$\phi_{interval}(t-r)$ 编码时间区间长度，$\phi_{end}(t)$ 编码当前时间点，组合条件嵌入 $\phi_{cond}(t,r) = \phi_{interval}(t-r) + \phi_{end}(t)$ 与文本特征共同控制速度网络。

### 关键设计

1. **文本表示区分性分析与验证**:

    - 功能：验证文本编码器的跨模态对齐质量
    - 核心思路：在 COCO 2017 的 118K 训练集上，用待评估的文本编码器编码查询 prompt，检索最相似的图像-文本对，再用 DINOv3 评估检索图像与查询图像的视觉特征相似度。BLIP3o-NEXT 得分 0.734，CLIP 得分 0.730，Gemma 0.713，T5 仅 0.634
    - 设计动机：区分性意味着文本编码器的输出与对应图像表示对齐良好，能准确区分语义相似但不同的文本。在少步生成中，每步的速度场方向必须足够准确，而区分性差的编码器会导致模糊的速度场

2. **文本表示解耦性分析与验证**:

    - 功能：评估文本编码器保持语言结构的能力
    - 核心思路：在 DPG-Bench 的完整 prompt 上，随机删除部分文本生成缩减版本，分别编码后计算余弦距离。好的编码器应使缩减版本与完整版本距离尽可能小（结构保持）。BLIP3o-NEXT 得分 0.999，Gemma 0.987，CLIP 0.967，T5 0.893
    - 设计动机：解耦性确保编码后的文本特征保留原始语言结构，不会因为文本变化而产生不成比例的表示漂移

3. **MeanFlow T2I 适配**:

    - 功能：将预训练 flow matching 模型适配为 MeanFlow 单步/少步生成
    - 核心思路：复制时间嵌入层为区间层和终点层；自适应采样时间步对 $(t, r)$，从均匀或 logit-normal 分布中采样，训练过程中逐渐增加 $t \neq r$ 的比例；使用标准 MeanFlow 目标 $\mathcal{L}_{MF}(\theta) = \mathbb{E}[\|u_\theta - \text{sg}(u_{tgt})\|^2]$，其中目标通过 JVP 计算
    - 设计动机：MeanFlow 在预训练模型基础上微调比从头训练容易得多（模型已编码速度场），但关键是文本编码器必须具备足够的区分性和解耦性

### 损失函数 / 训练策略

使用约 170K 样本（BLIP3o-60k + shareGPT-4o + Echo-4o），学习率 1e-5，batch size 128，训练 150 epochs。基于 BLIP3o-NEXT 模型微调。

## 实验关键数据

### 主实验

| 模型 | 步数 | GenEval↑ | DPG-Bench↑ | HPSv2↑ |
|------|------|---------|-----------|--------|
| BLIP3o-NEXT | 30 | 0.91 | 82.05 | 29.42 |
| BLIP3o-NEXT | 4 | 0.86 | 78.15 | 26.96 |
| BLIP3o-NEXT | 1 | 0.46 | 57.05 | 18.54 |
| **EMF (本文)** | 4 | **0.90** | **81.20** | **29.25** |
| **EMF (本文)** | 2 | 0.85 | 79.44 | 27.21 |
| **EMF (本文)** | 1 | 0.74 | 77.36 | 25.77 |
| SANA-Sprint | 4 | 0.77 | - | - |
| rCM | 4 | 0.83 | - | - |

### 消融实验

| 配置 | GenEval (1步) | 说明 |
|------|-------------|------|
| BLIP3o-NEXT 编码器 + MeanFlow | 0.74 | 高区分性+高解耦性 |
| SANA-1.5 编码器 + MeanFlow | 失败 | 区分性不足 |
| SANA-1.5 编码器 + SFT微调 + MeanFlow | 仍失败 | 微调无法弥补编码器缺陷 |

### 关键发现

- EMF 4 步生成几乎匹配 BLIP3o-NEXT 30 步（GenEval 0.90 vs 0.91），实现了约 7.5× 的加速
- EMF 超越所有蒸馏模型（SDXL-Turbo/Lightning/DMD2 等），且不需要教师模型
- SANA-1.5 编码器即使经过 SFT 微调也无法在 MeanFlow 中有效工作，证明编码器本身的属性而非训练数据是瓶颈
- EMF 的性能随步数增加持续提升（1步→2步→4步→8步），不像传统 consistency 模型那样出现步数增加后性能饱和甚至下降

## 亮点与洞察

- 对文本编码器"区分性"和"解耦性"的系统分析非常有价值。之前的工作往往只看最终生成质量，本文深入到文本表示空间的属性分析，为选择/设计少步生成的文本编码器提供了明确指标
- "为什么类别标签在 MeanFlow 中有效但文本不行"的分析很有洞察：类别标签天然离散且易区分，等于天然具备高区分性
- 与 consistency 方法的对比分析也很精彩：consistency 方法步数增加后可能退化，而 MeanFlow 作为连续流的稳定离散化可以持续受益于更多步

## 局限与展望

- 目前仅在 BLIP3o-NEXT 上验证，该编码器恰好同时具备高区分性和解耦性，能否推广到其他符合条件的编码器尚不确定
- 1 步生成的 GenEval 0.74 与多步基线仍有差距，距离真正的单步高质量 T2I 还有空间
- JVP 计算的数值稳定性问题虽然通过选择好的编码器缓解，但没有根本解决
- 未来方向：可探索专门为少步生成设计/训练的文本编码器

## 相关工作与启发

- **vs 原始 MeanFlow**: 仅支持类别条件，本文首次扩展到文本条件
- **vs SANA-Sprint**: 蒸馏方法，4 步 GenEval 0.77，本文 0.90 显著更优
- **vs Consistency Models**: 步数增加可能退化，本文的 MeanFlow 方法可持续提升

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将 MeanFlow 扩展到 T2I，文本表示分析有深度
- 实验充分度: ⭐⭐⭐⭐ 多基准对比充分，编码器分析系统
- 写作质量: ⭐⭐⭐⭐ 从观察到分析到方法的推导逻辑清晰
- 价值: ⭐⭐⭐⭐ 为少步 T2I 生成提供了编码器选择的指导性见解

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](solace_self_confidence_rewards_t2i.md)
- [\[CVPR 2026\] ChordEdit: One-Step Low-Energy Transport for Image Editing](chordedit_one-step_low-energy_transport_for_image_editing.md)
- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)
- [\[CVPR 2026\] Agentic Retoucher for Text-To-Image Generation](agentic_retoucher_for_texttoimage_generation.md)

</div>

<!-- RELATED:END -->
