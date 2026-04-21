---
title: >-
  [论文解读] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models
description: >-
  [CVPR 2025][3D视觉][感知token] 本文提出 Perception Tokens，一种将中间视觉表示（如深度图、目标框）编码为辅助推理 token 的方法，使多模态语言模型能像语言 chain-of-thought 一样，通过生成感知 token 作为中间步骤来增强视觉推理能力。
tags:
  - CVPR 2025
  - 3D视觉
  - 感知token
  - 多模态语言模型
  - 链式推理
  - 深度估计
  - 目标计数
---

# Perception Tokens Enhance Visual Reasoning in Multimodal Language Models

**会议**: CVPR 2025  
**arXiv**: [2412.03548](https://arxiv.org/abs/2412.03548)  
**代码**: 无（将在项目页面发布）  
**领域**: 3D视觉/多模态  
**关键词**: 感知token, 多模态语言模型, 链式推理, 深度估计, 目标计数

## 一句话总结

本文提出 Perception Tokens，一种将中间视觉表示（如深度图、目标框）编码为辅助推理 token 的方法，使多模态语言模型能像语言 chain-of-thought 一样，通过生成感知 token 作为中间步骤来增强视觉推理能力。

## 研究背景与动机

**领域现状**：多模态语言模型（MLM）如 LLaVA 在高层视觉语言任务上表现出色，但在需要基础视觉感知能力的任务（如深度推理、目标计数）上仍然困难。专用视觉模型在这些任务上表现更好，但 MLM 无法原生地生成深度图或检测框来辅助推理。

**现有痛点**：（1）直接 fine-tune MLM 在感知任务上效果有限，且泛化能力差；（2）调用外部视觉工具（如深度估计器、检测器）需要额外的计算和内存开销，且是多模型级联容易累积错误；（3）MLM 的词汇表只有文本和 CLIP 图像 token，无法表示深度、分割等中低层视觉特征。

**核心矛盾**：视觉推理需要中间视觉表示（如深度图）来支撑推理过程，但 MLM 的 token 空间只有语言 token——自然语言无法精确描述像素级的深度关系或精确的物体位置。

**本文目标**：扩展 MLM 的 token 空间，引入辅助感知 token，使模型能在推理过程中生成并利用视觉感知表示。

**切入角度**：类比语言模型的 chain-of-thought 推理——语言 CoT 通过生成中间文本步骤来辅助推理，那么视觉任务也可以通过生成中间视觉表示（编码为 token）来辅助推理。

**核心 idea**：用 VQVAE 将深度图等视觉表示 tokenize 为离散 token 加入 MLM 词汇表，训练 MLM 在回答视觉问题时先生成这些感知 token 作为中间推理步骤（如"深度图是<<<perception tokens>>>，因此 D 点最近"），然后基于这些 token 得出最终答案。

## 方法详解

### 整体框架

Aurora 框架扩展了 LLaVA 的词汇表 $V' = V \cup V_{\text{aux}}$。首先用 VQVAE 将深度图 tokenize（像素级表示）或直接编码边界框（结构化表示）为辅助 token。然后用课程学习策略训练 MLM：从简单的 token 预测任务开始，逐步过渡到使用 token 进行 chain-of-thought 视觉推理。

### 关键设计

1. **感知 Token 的 Tokenization**:

    - 功能：将中间视觉表示统一编码为 MLM 可以生成和处理的离散 token
    - 核心思路：对于像素级表示（深度图、分割 mask），使用 VQVAE/VQGAN 将其编码为离散 codebook 索引作为 token。对于结构化表示（边界框、坐标），根据域范围直接定义 token（如坐标范围 0 到图像最大像素数）。所有 token 统一加入 $V_{\text{aux}}$，构成扩展词汇表。
    - 设计动机：统一的 tokenization 空间使不同类型的视觉表示可以在同一个自回归框架中无缝处理，无需修改模型架构。

2. **专家到通才蒸馏 + 重建损失**:

    - 功能：训练 MLM 生成准确的辅助 token
    - 核心思路：使用预训练的专用模型（如深度估计器）提供目标分布 $q_i$，通过交叉熵蒸馏损失 $\ell_{dist} = \min_M (-\sum_i q_i \log p_{M(i)})$ 训练 MLM 对齐辅助 token 预测。同时引入轻量级解码器 $g$ 将 token 映射回特征空间，加入重建损失 $\ell_{rec} = \|g(t) - f\|_2^2$ 增强 token 的可解释性和预测准确性。
    - 设计动机：蒸馏确保生成的 token 语义上与专用模型一致，重建确保 token 在解码回原始表示时保持高保真。两者结合避免了 token 预测的退化。

3. **课程学习 + 渐进 CoT**:

    - 功能：避免灾难性遗忘，逐步建立多步推理能力
    - 核心思路：定义任务难度 $d_1 < d_2 < \cdots < d_T$，用温度退火的 Softmax 采样概率 $p(d_t, s) = \exp(-d_t/\tau(s)) / \sum_i \exp(-d_i/\tau(s))$ 控制训练进度，$\tau(s) = \tau_0 / (1 + \lambda \cdot s/S)$ 随训练步数逐渐降低温度。三类数据子集：（a）原子任务：学习生成辅助 token；（b）CoT 数据：先生成 token 再回答问题；（c）直接标注：不生成 token 直接回答。同一图像上顺序展示两种推理风格。
    - 设计动机：直接用固定数据混合训练会导致 token 预测准确性和推理能力之间的 trade-off。课程学习先掌握基础（生成 token），再逐步学习复杂推理，有效避免灾难性遗忘。

### 损失函数 / 训练策略

结合蒸馏损失和重建损失训练辅助 token 预测。使用约束解码（限制只采样辅助 token）和信息瓶颈（截断 CoT 链路中只保留辅助 token）来强制模型依赖感知 token 推理。基于 LLaVA 1.5 13B 实现。

## 实验关键数据

### 主实验

相对深度估计（准确率 %）：

| 方法 | BLINK 2点 | HardBLINK 3点 | HardBLINK 4点 | HardBLINK 5点 | 平均 |
|---|---|---|---|---|---|
| LLaVA 1.5 13B | 54.0 | 35.5 | 37.9 | 29.0 | 39.1 |
| Fine-tuned LLaVA | 68.5 | 58.9 | 52.4 | 41.1 | 55.2 |
| GPT-4o | 53.2 | 58.9 | 50.0 | 36.3 | 49.6 |
| **LLaVA-Aurora** | **64.5** | **66.9** | **60.5** | **54.8** | **61.6** |

目标计数（准确率 %）：

| 方法 | BLINK | CVBench | SEED-Bench |
|---|---|---|---|
| LLaVA 1.5 13B | 34.7 | 43.3 | 54.2 |
| Fine-tuned LLaVA | 35.2 | 48.5 | 57.5 |
| **LLaVA-Aurora** | **45.5** | **54.6** | **62.5** |

### 消融实验

| 配置 | BLINK 深度 | BLINK 计数 | 说明 |
|---|---|---|---|
| 基线（无感知 token） | 39.1 | 34.7 | 原始 LLaVA |
| 仅 fine-tune | 55.2 | 35.2 | 有限提升 |
| 仅 token 预测（无 CoT） | 低于完整 | 低于完整 | token 未被推理利用 |
| **完整 Aurora** | **61.6** | **45.5** | 感知 token + CoT |

### 关键发现

- LLaVA-Aurora 在深度推理上平均提升 +6.4%（vs fine-tune），在更难的 5 点配置上提升 +13.7%
- 计数任务上跨三个 benchmark 一致提升（+10.8/+11.3/+8.3 个百分点）
- 感知 token 在难度越高的任务上优势越明显——简单任务可能不需要中间推理
- 课程学习策略对避免灾难性遗忘至关重要，去掉后性能显著下降
- 即使不用外部工具，端到端的感知 token 推理也能超越 GPT-4 Turbo + Tool

## 亮点与洞察

1. **"视觉 chain-of-thought"范式**：将 CoT 从语言扩展到视觉模态，用生成的深度图 token 辅助深度推理、用框 token 辅助计数——这是一个全新的推理范式
2. **VQVAE tokenization 的统一性**：将各种视觉表示统一为离散 token，与语言 token 共享同一个自回归空间，优雅地解决了多模态表示的兼容性问题
3. **课程学习策略**：温度退火的采样策略巧妙平衡了 token 学习和推理学习，是处理多任务异质数据的有效方案

## 局限与展望

- 目前仅验证了深度和计数两类任务，未扩展到分割、姿态等更多感知任务
- VQVAE 的 tokenization 会引入信息损失，影响精细推理
- 推理时需要额外生成感知 token，增加了生成长度和推理时间
- 未来可以扩展到视频理解、具身智能等需要更丰富中间表示的场景

## 相关工作与启发

- **vs Unified-IO**：Unified-IO 可以生成视觉输出但不能对自己的生成进行推理；Aurora 的核心是让模型在自己的生成上做推理
- **vs LISA**：LISA 生成分割 token 用于 grounding；Aurora 的感知 token 是通用的、用于推理的中间步骤
- **vs Tool-using MLMs**：如 Visual ChatGPT 调用外部工具；Aurora 将工具能力内化为 token，无需额外模型
- 启发：MLM 的推理能力受限于其 token 空间，扩展 token 空间是提升推理能力的有效途径

## 评分

- 新颖性: 8/10 — "视觉 CoT"概念新颖，将感知能力编码为推理 token 的想法具有开创性
- 实验充分度: 7/10 — 多个 benchmark 验证但仅覆盖深度和计数两类任务
- 写作质量: 8/10 — 框架描述清晰，类比语言 CoT 直观易懂
- 价值: 8/10 — 开辟了 MLM 视觉推理的新方向，感知 token 框架具有很强的扩展性

<!-- RELATED:START -->

## 相关论文

- [InteractVLM: 3D Interaction Reasoning from 2D Foundational Models](interactvlm_3d_interaction_reasoning_from_2d_foundational_models.md)
- [GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](../../ICCV2025/3d_vision/geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)
- [DepthCues: Evaluating Monocular Depth Perception in Large Vision Models](depthcues_evaluating_monocular_depth_perception_in_large_vision_models.md)
- [SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception](sphereuformer_a_u-shaped_transformer_for_spherical_360_perception.md)
- [Empowering Large Language Models with 3D Situation Awareness](empowering_large_language_models_with_3d_situation_awareness.md)

<!-- RELATED:END -->
