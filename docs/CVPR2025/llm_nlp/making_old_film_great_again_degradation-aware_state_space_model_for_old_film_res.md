---
title: >-
  [论文解读] Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration
description: >-
  [CVPR 2025][LLM/NLP][老电影修复] 本文提出MambaOFR框架，针对老电影特有的复合退化问题，设计退化感知prompt引导Mamba模型动态调整修复模式，配合光流引导的掩码变形对齐模块防止结构缺陷传播，并引入首个包含合成与真实数据的老电影修复benchmark数据集。
tags:
  - "CVPR 2025"
  - "LLM/NLP"
  - "老电影修复"
  - "Mamba"
  - "退化感知"
  - "光流对齐"
  - "状态空间模型"
---

# Making Old Film Great Again: Degradation-aware State Space Model for Old Film Restoration

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 无  
**领域**: 视频修复 / 老电影修复  
**关键词**: 老电影修复, Mamba, 退化感知, 光流对齐, 状态空间模型

## 一句话总结

本文提出MambaOFR框架，针对老电影特有的复合退化问题，设计退化感知prompt引导Mamba模型动态调整修复模式，配合光流引导的掩码变形对齐模块防止结构缺陷传播，并引入首个包含合成与真实数据的老电影修复benchmark数据集。

## 研究背景与动机

### 领域现状

**领域现状**：老电影修复旨在恢复模拟胶片时代的影片质量，需要处理划痕、闪烁、噪点、颜色衰减、胶片颗粒等特有退化。近年来视频修复方法（如BasicVSR++、RVRT等）在现代视频去噪、超分等任务上表现出色。

**现有痛点**：(1) 退化类型特殊且复合——老电影的退化不同于数码视频（JPEG压缩、运动模糊等），包括胶片物理老化（划痕、断裂）、化学变化（颜色偏移、褪色）和数字化伪影，且往往多种退化同时出现。(2) 专用方法反而不如通用方法——现有针对老电影的专用修复方法（如针对划痕的检测-修复流水线）在处理复合退化时表现不如通用视频修复方法。(3) 时间维度特征传播困难——划痕等结构性缺陷在视频帧间具有空间一致性，简单的时间注意力/对齐会将缺陷特征传播到相邻帧。

**核心矛盾**：老电影修复需要长程时序建模（利用多帧信息修复）和退化自适应处理（针对不同退化类型调整策略），但长程建模可能传播结构性缺陷，而固定的修复策略无法适应复合多变的退化。

**本文目标** 如何设计一个能自适应处理多种复合退化、同时避免缺陷时间传播的老电影修复框架？

**切入角度**：利用Mamba（状态空间模型）的高效长序列建模能力处理视频时间维度，通过退化感知prompt动态调整修复行为，并设计专门的掩码对齐防止缺陷传播。

**核心 idea**：退化感知prompt驱动Mamba自适应修复 + 掩码变形对齐阻断缺陷传播 + 首个综合benchmark数据集。

## 方法详解

### 整体框架

MambaOFR采用多帧输入的编码器-解码器架构，核心是Mamba-based时空处理骨干。给定连续视频帧：(1) 退化估计模块分析每帧的退化类型和程度，生成退化感知prompt。(2) Mamba骨干网络在prompt引导下提取时空特征，其中光流引导的掩码变形对齐模块处理帧间对齐。(3) 解码器重建修复后的帧。

### 关键设计

1. **退化感知Prompt生成**：
    - 功能：让网络根据不同退化类型动态调整修复策略
    - 核心思路：设计一个轻量级退化估计网络，将输入帧的特征提取后预测退化表示向量（编码退化类型和程度）。该表示通过仿射变换生成scale和shift参数，作为prompt注入Mamba骨干网络的各层，动态调制特征处理过程。$\gamma, \beta = \text{MLP}(z_{deg})$，$\hat{f} = \gamma \cdot f + \beta$
    - 设计动机：划痕修复需要强烈的空间修补能力，而色彩衰减修复需要全局色彩映射，固定参数网络无法同时优化两种策略

2. **光流引导掩码变形对齐（Flow-Guided Mask Deformable Alignment）**：
    - 功能：在利用多帧信息时阻断结构缺陷的传播
    - 核心思路：首先估计帧间光流，同时训练一个缺陷检测分支预测帧上划痕/断裂等结构性缺陷的二值掩码。在变形对齐（deformable alignment）时，利用缺陷掩码降低缺陷区域特征的权重——被划痕覆盖的区域在相邻帧的对应位置不应被传播过来。具体实现上，将掩码与变形注意力的权重相乘，实现软性屏蔽
    - 设计动机：传统光流对齐会将参考帧上划痕位置的错误特征传播到当前帧，掩码机制确保只传播"干净"区域的信息

3. **综合Benchmark数据集**：
    - 功能：首个同时包含合成和真实退化的老电影修复标准评测集
    - 核心思路：合成部分使用复合退化模拟流水线（随机叠加划痕、噪点、闪烁、色彩衰减等），从高质量数字视频生成配对训练数据。真实部分收集实际老电影片段用于定性评估。涵盖不同年代、不同胶片类型的退化模式
    - 设计动机：现有方法缺乏统一评测平台，不同论文使用不同测试集导致结果不可比

### 损失函数 / 训练策略
模型采用端到端训练，优化目标综合考虑任务损失和正则化项。


## 实验关键数据

### 关键发现

- MambaOFR在合成测试集上的PSNR/SSIM显著超越现有老电影专用方法和通用视频修复方法
- 退化感知prompt使模型对不同退化类型的自适应能力提升约1.5-2dB PSNR
- 掩码变形对齐在包含划痕的帧上修复质量提升最为显著
- 消融实验表明三个组件（prompt、掩码对齐、Mamba骨干）均有独立贡献
- 真实老电影的定性修复结果在视觉质量上明显优于竞争方法

## 亮点与洞察

- **问题驱动设计**：每个模块都直接对应老电影修复的具体痛点
- **Mamba的合适应用**：利用SSM的长序列建模优势处理视频时间维度，计算效率优于Transformer
- **benchmark贡献**：填补了老电影修复领域缺乏标准评测的空白

## 局限与展望

- 极严重的物理损坏（大面积画面缺失）仍难以修复
- 合成退化与真实退化之间仍有domain gap
- 颜色恢复依赖训练数据的色彩分布，对特殊色彩风格的胶片可能效果有限
- 未来可结合生成模型（如扩散模型）进行修复区域的语义重建


## 相关工作与启发
- **vs 同领域代表性方法**：本文在方法设计上有独特贡献，与现有方法形成互补
- **vs 传统方法**：相比传统方案，本文方法在关键指标上取得了显著提升
- **启发**：本文的技术路线对后续相关工作有重要参考价值


## 评分
- 新颖性: ⭐⭐⭐⭐ 方法设计有独特贡献
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证
- 写作质量: ⭐⭐⭐⭐ 条理清晰
- 价值: ⭐⭐⭐⭐ 对领域有推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](../../AAAI2026/llm_nlp/an_invariant_latent_space_perspective_on_language_model_inve.md)
- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](../../ACL2025/llm_nlp/biased_llms_can_influence_political_decision-making.md)
- [\[ICML 2026\] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State](../../ICML2026/llm_nlp/a_geometric_relation_of_the_error_introduced_by_sampling_a_language_models_outpu.md)
- [\[NeurIPS 2025\] Unifying Attention Heads and Task Vectors via Hidden State Geometry in In-Context Learning](../../NeurIPS2025/llm_nlp/unifying_attention_heads_and_task_vectors_via_hidden_state_geometry_in_in-contex.md)
- [\[ACL 2025\] Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](../../ACL2025/llm_nlp/why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)

</div>

<!-- RELATED:END -->
