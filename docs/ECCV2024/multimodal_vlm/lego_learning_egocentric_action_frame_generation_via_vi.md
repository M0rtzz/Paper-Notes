---
title: >-
  [论文解读] LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning
description: >-
  [ECCV 2024][多模态VLM][第一人称视角] 提出 LEGO 模型，通过视觉指令微调增强 VLLM 的动作描述能力，并将 VLLM 的图像/文本嵌入作为额外条件注入扩散模型，实现从第一人称视角生成动作执行帧。
tags:
  - ECCV 2024
  - 多模态VLM
  - 第一人称视角
  - 动作帧生成
  - 视觉指令微调
  - 扩散模型
  - VLLM
---

# LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning

**会议**: ECCV 2024  
**arXiv**: [2312.03849](https://arxiv.org/abs/2312.03849)  
**代码**: [https://bolinlai.github.io/Lego_EgoActGen/](https://bolinlai.github.io/Lego_EgoActGen/)  
**领域**: 多模态VLM  
**关键词**: 第一人称视角, 动作帧生成, 视觉指令微调, 扩散模型, VLLM

## 一句话总结

提出 LEGO 模型，通过视觉指令微调增强 VLLM 的动作描述能力，并将 VLLM 的图像/文本嵌入作为额外条件注入扩散模型，实现从第一人称视角生成动作执行帧。

## 研究背景与动机

### 现有痛点

**现有痛点**：核心问题**：如何从第一人称视角（egocentric）合成展示动作执行过程的图像（action frame），以高效地传递技能

### 领域现状

**领域现状**：现有困境**：

### 核心矛盾

**核心矛盾**：现有第一人称动作数据集（Ego4D、Epic-Kitchens）的标注仅为"动词+名词"，缺乏动作执行的详细描述

### 解决思路

**解决思路**：现有扩散模型主要在第三人称图像上预训练，存在与第一人称图像之间的 domain gap

### 补充说明

**补充说明**：纯文本指令不够直观，人脑处理图像远快于文本

### 补充说明

**补充说明**：动机**：用户佩戴相机后，输入当前场景图像和动作查询，直接生成展示动作执行的目标图像，提供可视化指导

## 方法详解

### 整体框架

LEGO 由两个核心阶段组成：
1. **Prompt Enhancement 阶段**：通过视觉指令微调训练 VLLM，生成丰富的动作描述
2. **Action Frame Generation 阶段**：将 VLLM 嵌入注入潜在扩散模型（LDM），生成第一人称动作帧

### 关键设计

**1. 数据整理与视觉指令微调**
- 利用 GPT-3.5 基于动作标签和物体边界框进行 in-context learning，生成详细动作描述
- 在 VLLM（基于 LLaVA）上进行视觉指令微调：冻结 CLIP 图像编码器，微调投影层和 LLM
- 微调后的 VLLM 无需边界框输入，可大规模生成丰富动作描述

**2. VLLM 嵌入注入 LDM**
- CLIP 文本编码器提取动作描述的文本表示 $\psi(\mathcal{R})$
- VLLM 图像嵌入 $\mathcal{H}_i$ 经线性层投影到 LDM 特征空间
- VLLM 文本嵌入 $\mathcal{H}_t$ 经投影层 + 自注意力层处理
- 三者拼接形成完整条件 $\mathcal{C} = [\psi(\mathcal{R}), \sigma(\mathcal{H}_i), \pi(\mu(\mathcal{H}_t))]$
- 通过交叉注意力机制注入 UNet 的多个层

### 损失函数 / 训练策略

- VLLM 阶段：交叉熵损失，训练 3 个 epoch
- LDM 阶段：L2 回归损失（预测噪声 vs 真实噪声），训练 20K 迭代
- 采用 classifier-free guidance
- 数据预处理：基于美学评分筛选帧、基于相似度过滤过于相似或过于不同的帧对

## 实验关键数据

### 主实验

| 方法 | EgoVLP | EgoVLP+ | CLIP | FID↓ | PSNR | LPIPS↓ |
|------|--------|---------|------|------|------|--------|
| ProxEdit | 44.51 | 72.68 | 68.17 | 33.01 | 11.88 | 40.90 |
| SDEdit | 50.07 | 72.90 | 73.35 | 33.35 | 11.81 | 41.60 |
| IP2P | 62.19 | 78.84 | 78.75 | 24.73 | 12.16 | 37.16 |
| **LEGO** | **65.65** | **80.44** | **80.61** | **23.83** | **12.29** | **36.43** |

（Ego4D 数据集结果）

### 消融实验

| 条件设定 | User Study | EgoVLP | EgoVLP+ | CLIP |
|---------|-----------|--------|---------|------|
| Action Labels | 5.33 | 62.19 | 78.84 | 78.75 |
| Descriptions | 13.00 | 62.91 | 79.09 | 79.18 |
| Desc.+Img Embed. | 26.00 | 65.35 | 80.13 | 80.57 |
| Desc.+Txt Embed. | 21.33 | 63.29 | 79.40 | 79.21 |
| Desc.+Joint Embed. | **34.34** | **65.65** | **80.44** | **80.61** |

### 关键发现

- VLLM 图像嵌入比文本嵌入带来更大性能提升，蕴含未被自编码器或文本捕获的高层语义信息
- 微调后的 VLLM 嵌入优于未微调版本（EgoVLP 提升 1.08%），视觉指令微调对缩小 domain gap 至关重要
- 用户研究中 LEGO 的 win rate 超过最强基线 44%（Ego4D）和 38.34%（E-Kitchens）

## 亮点与洞察

1. **首次提出第一人称动作帧生成问题**，将技能传递从文本指导升级为视觉指导
2. **创新的 VLLM-LDM 耦合架构**：不仅用 VLLM 生成更好的文本描述，还将其内部嵌入作为扩散条件
3. 视觉指令微调使 prompt 对齐度从 27%→87%，显著减少幻觉
4. 模型具备泛化能力：同一输入帧+不同动作查询，可生成不同且合理的动作帧

## 局限与展望 / 可改进方向

- 生成分辨率仅 256×256，限制了实际应用
- 依赖已有数据集的动作标注结构，难以推广到开放域
- 未考虑视频时序一致性，仅生成单帧
- BLIP-based 文本评估指标仍受 domain gap 影响

## 相关工作与启发

- InstructPix2Pix 启发了基于指令的图像编辑范式，但缺乏第一人称域适应
- GILL 探索了从 VLLM 学习图像嵌入用于生成，但未进行视觉指令微调
- 可启发：将 VLLM+扩散模型的耦合范式拓展到机器人操作指令可视化、AR 辅助教学等场景

## 评分

- 新颖性：⭐⭐⭐⭐（新问题定义 + 创新的 VLLM 嵌入注入设计）
- 技术深度：⭐⭐⭐⭐
- 实验充分度：⭐⭐⭐⭐⭐（双数据集、用户研究、全面消融）
- 写作质量：⭐⭐⭐⭐
- 综合推荐：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ECCV 2024\] MotionChain: Conversational Motion Controllers via Multimodal Prompts](motionchain_conversational_motion_controllers_via_multimodal_prompts.md)
- [\[ECCV 2024\] IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)
- [\[ECCV 2024\] Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild](nymeria_a_massive_collection_of_multimodal_egocentric_daily_motion_in_the_wild.md)
- [\[ECCV 2024\] QUAR-VLA: Vision-Language-Action Model for Quadruped Robots](quar-vla_vision-language-action_model_for_quadruped_robots.md)

</div>

<!-- RELATED:END -->
