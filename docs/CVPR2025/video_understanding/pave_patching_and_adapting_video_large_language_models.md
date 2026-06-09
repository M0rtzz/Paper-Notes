---
title: >-
  [论文解读] PAVE: Patching and Adapting Video Large Language Models
description: >-
  [CVPR 2025][视频理解][Video LLM适配] PAVE 提出了一种通过轻量级"补丁"（patch）适配预训练 Video LLM 的框架，能将音频、3D 线索、多视角视频等侧信道信号以约 0.1% 的额外参数和计算量融入基础模型，在音视频 QA、3D QA 等任务上超越专用模型。
tags:
  - "CVPR 2025"
  - "视频理解"
  - "Video LLM适配"
  - "侧信道信号"
  - "轻量适配器"
  - "音视频问答"
  - "3D场景理解"
---

# PAVE: Patching and Adapting Video Large Language Models

**会议**: CVPR 2025  
**arXiv**: [2503.19794](https://arxiv.org/abs/2503.19794)  
**代码**: [https://github.com/dragonlzm/PAVE](https://github.com/dragonlzm/PAVE)  
**领域**: 多模态VLM / 视频理解  
**关键词**: Video LLM适配, 侧信道信号, 轻量适配器, 音视频问答, 3D场景理解

## 一句话总结

PAVE 提出了一种通过轻量级"补丁"（patch）适配预训练 Video LLM 的框架，能将音频、3D 线索、多视角视频等侧信道信号以约 0.1% 的额外参数和计算量融入基础模型，在音视频 QA、3D QA 等任务上超越专用模型。

## 研究背景与动机

**领域现状**：视频大语言模型（Video LLM）如 LLaVA-OneVision、Qwen2-VL 等在视频理解上展示了强大的推理能力，但它们通常只接受视频-文本输入。许多下游任务需要额外模态或数据类型（如音频、3D 深度、多视角视频）。

**现有痛点**：针对这些任务通常需要从头训练专用模型（如 CAT 处理音视频、LLaVA-3D 处理 3D QA），代价高昂且无法复用 Video LLM 的预训练知识。有趣的是，作者发现即使不使用音频/3D 数据，仅在目标数据集上微调 Video LLM 也能接近甚至超越这些专用模型——说明 Video LLM 本身已经具备了很强的推理能力。

**核心矛盾**：如何在不改变预训练 Video LLM 架构和权重的前提下，高效地注入侧信道信号的信息？

**本文目标**：设计一个灵活的、参数高效的框架，通过添加轻量"补丁"适配预训练 Video LLM 到多种下游任务。

**切入角度**：类比 LoRA 在 LLM 和扩散模型上的成功——小的补丁可以定制化大模型且方便分享。通过交叉注意力将侧信道信号融入视觉 tokens，不增加 LLM 的输入 token 数量。

**核心 idea**：学习一个融合函数 $g(\cdot)$ 将侧信道 tokens 与视觉 tokens 通过时序对齐的交叉注意力融合，以残差方式更新视觉 tokens，配合 LoRA 适配 LLM。

## 方法详解

### 整体框架

给定视频和侧信道信号，分别用各自的编码器提取 tokens。PAVE 的"补丁"由两部分组成：（1）融合函数 $g(\cdot)$ —— 一个基于时序对齐交叉注意力的轻量模块，将侧信道 tokens 融入视觉 tokens；（2）LoRA 模块 —— 对 LLM 的低秩适配。融合结果以残差方式加到原始视觉 tokens 上：$\hat{z}^v(k) = z^v(k) + z^{v|s}(k)$，然后送入 LLM。

### 关键设计

1. **Temporal-Aligned Cross-Attention（时序对齐交叉注意力）**:

    - 功能：将侧信道信号的信息融入视频的视觉特征
    - 核心思路：将视觉 tokens $z^v(k)$ 作为 query，仅关注其时间邻域内的侧信道 tokens $z^s(k_s)$（$k_s \in N(k)$），实现局部时序对齐。加入 RoPE 位置编码，后接 MLP 和 LayerNorm，形成完整的 Transformer block
    - 设计动机：侧信道信号（如音频、3D 点云）与视频帧在时间上有自然的对应关系。局部注意力比全局注意力计算量小得多，同时保持了时序对齐的精度

2. **残差融合 + 不改变 token 数量**:

    - 功能：确保适配过程对基础模型零侵入
    - 核心思路：融合函数的输出形状与视觉 tokens 完全相同，通过简单加法更新视觉 tokens。LLM 接收的 token 数量不变，因此计算开销几乎不增加
    - 设计动机：LLM 是 Video LLM 中计算量最大的部分，保持其输入不变意味着额外开销可忽略（~0.1% FLOPs）

3. **统一框架适配多种侧信道**:

    - 功能：用同一架构处理音频、3D、多视角、高帧率等多种信号
    - 核心思路：所有侧信道信号统一表示为时序排列的 token 序列 $\{z^s(k_s)\}$，不同的只是编码器。音频用 ImageBind，3D 用 LLaVA-3D 编码器，多视角/高帧率视频用 Video LLM 自身的视觉编码器
    - 设计动机：统一的接口使得 PAVE 可以用同一个框架处理各种下游任务，甚至支持多任务联合训练

### 损失函数 / 训练策略

使用标准的自回归负对数似然损失。只训练融合函数 $g(\cdot)$（~9M 参数）和 LoRA 模块（~0.5M 参数），冻结基础模型的所有参数。训练 1-2 epochs。

## 实验关键数据

### 主实验

| 任务 | 方法 | 关键指标 | 额外参数 | 额外 FLOPs |
|------|------|---------|---------|-----------|
| 音视频 QA (AVSD) | COST (SOTA) | CIDEr=108.5 | - | - |
| | **PAVE** | **CIDEr=152.9** | 9M | 0.1 TB |
| 音视频 QA (AVQA) | CAT-7B-FT | Acc=92.0 | - | - |
| | **PAVE** | **Acc=93.8** | 9M | 0.1 TB |
| 3D QA (SQA3D) | LLaVA-3D | EM@1=55.6 | - | - |
| | **PAVE** | **EM@1=59.0** | 9M | 0.15 TB |
| 视频 QA (VideoMME) | LLaVA-OV-7B | Avg=58.2 | - | - |
| | **PAVE** | **Avg=59.9** | 9M | 0.1 TB |

### 消融实验

| 配置 | AVQA Acc | 说明 |
|------|---------|------|
| LLaVA-OV-7B (zero-shot) | 85.6 | 无微调、无音频 |
| LLaVA-OV-7B-FT (LoRA only) | 90.8 | 微调但无音频 |
| PAVE (w/ audio) | 93.8 | 完整模型 |

### 关键发现

- **仅用视频微调 Video LLM 已经很强**：LLaVA-OV-7B-FT 无需音频就接近专用音视频模型，说明 Video LLM 的推理能力被严重低估
- **加入侧信道信号带来稳定提升**：PAVE 比仅视频微调一致性地提升 2-3%，证明侧信道信息的价值
- **极低的开销**：仅 0.1% 的额外参数和 FLOPs，实现 SOTA 性能
- 在 Music-AVQA 的"音频-视觉"分割上 PAVE 弱于 CAT-7B-FT，因为 Video LLM 倾向于优先使用视觉线索，面对音视频冲突时表现不佳

## 亮点与洞察

- **"Video LLM 比你想象的更强"**是很有价值的发现——仅用视频微调就能接近专用多模态模型，这意味着 Video LLM 已经隐式学到了跨模态的推理能力
- **patch as plugin**的设计哲学非常实用：小补丁可以方便地分享和组合，类似 LoRA 在扩散模型社区的生态。这种模式可以催生一个"Video LLM 补丁生态"
- **时序对齐交叉注意力**的设计简洁有效——利用时间轴上的自然对齐关系，避免了全局注意力的高计算量

## 局限与展望

- 在需要强音频推理的场景下（如音视频冲突），基于视频预训练的 PAVE 仍有劣势
- LoRA + patch 的参数预算固定，对于特别复杂的新模态（如点云、触觉）可能不够
- 目前各任务的 patch 需要独立训练，统一的多任务 patch 仍是开放问题
- 可以探索将 patch 与更多 Video LLM 骨干（如 Qwen2-VL）组合

## 相关工作与启发

- **vs CAT**: CAT 把音视频理解内嵌到模型训练中，需要大量多模态数据；PAVE 仅需一个小补丁就能适配现有 Video LLM，方法更轻量灵活
- **vs LLaVA-3D**: LLaVA-3D 专门设计3D编码器和训练流程；PAVE 将其3D编码器作为侧信道输入，用统一框架达到更好效果
- **vs SeViLA**: SeViLA 通过 Localizer + Answerer 适配视频任务，但架构改动较大；PAVE 的残差式补丁设计更加简洁

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一的侧信道适配框架设计优雅，但核心技术（交叉注意力 + LoRA）不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 四个任务全面验证，对比充分，消融细致
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，motivation 论证有力
- 价值: ⭐⭐⭐⭐⭐ 框架通用性强，可直接促进 Video LLM 在多种下游任务的落地

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)
- [\[CVPR 2025\] VoCo-LLaMA: Towards Vision Compression with Large Language Models](voco-llama_towards_vision_compression_with_large_language_models.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[CVPR 2025\] Efficient Transfer Learning for Video-language Foundation Models](efficient_transfer_learning_for_video-language_foundation_models.md)

</div>

<!-- RELATED:END -->
