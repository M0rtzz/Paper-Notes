---
title: >-
  [论文解读] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning
description: >-
  [ICCV 2025][图像差异描述] 提出包含324个多样场景（真实+3D合成）的细粒度图像差异描述数据集 OmniDiff，并设计即插即用的多尺度差异感知（MDP）模块嵌入 MLLM 构建 M3Diff 模型，在 OmniDiff 及多个公开基准上取得 SOTA。
tags:
  - ICCV 2025
  - 图像差异描述
  - 多模态大语言模型
  - 多尺度差异感知
  - 变化检测
  - benchmark
---

# OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning

**会议**: ICCV 2025  
**arXiv**: [2503.11093](https://arxiv.org/abs/2503.11093)  
**代码**: [yuan-liu-omnidiff.github.io](https://yuan-liu-omnidiff.github.io)  
**领域**: 视觉-语言理解  
**关键词**: 图像差异描述, 多模态大语言模型, 多尺度差异感知, 变化检测, benchmark

## 一句话总结

提出包含324个多样场景（真实+3D合成）的细粒度图像差异描述数据集 OmniDiff，并设计即插即用的多尺度差异感知（MDP）模块嵌入 MLLM 构建 M3Diff 模型，在 OmniDiff 及多个公开基准上取得 SOTA。

## 研究背景与动机

### 问题定义

图像差异描述（Image Difference Captioning, IDC）旨在生成自然语言描述来精确表达两张相似图像之间的细微差异，需要同时具备精确的视觉变化定位能力和连贯的语义表达能力。

### 已有方法的不足

**数据集层面**存在广度和深度两方面的不足：

**广度不足**：现有数据集局限于特定场景中有限的对象变化。Spot-the-Diff 仅覆盖固定视角的街道监控，Birds-to-Words 专注于鸟类细粒度差异，CLEVR-Change 仅渲染简单桌面场景

**深度不足**：先前基准的描述过于简短。IEdit 的平均描述长度仅 8 个词，无法反映真实场景的复杂变化

**核心矛盾**：缺乏兼顾场景多样性和描述精细度的统一基准，限制了 IDC 模型在复杂动态环境中的适用性。

### 本文切入角度

同时从数据和模型两个角度解决问题：构建覆盖12种变化类型、平均描述60词的高质量数据集，并设计增强 MLLM 细粒度差异感知能力的专用模块。

## 方法详解

### 整体框架

M3Diff 基于 LLaVA-OneVision-7B 架构，在标准 MLLM 框架中引入多尺度差异感知（MDP）模块，对图像对进行特征级差分，通过自适应跨层融合生成连贯的差异表示，再送入语言解码器生成描述。

### 关键设计

#### 1. **OmniDiff 数据集构建**
- **功能**：构建包含324个场景的高质量 IDC 数据集，覆盖真实世界和3D合成环境
- **核心思路**：
    - 真实数据：实地拍摄 + 网络爬取，覆盖224个真实场景
    - 合成数据：使用 Blender 渲染复杂3D场景（来自 ArtStation 的50个室内+50个室外场景）
    - 人工标注：平均60词的细粒度差异描述，涵盖12种变化类型（视角、光照、添加、消失、替换、大小、颜色、朝向、姿态、OCR、计数等）
- **设计动机**：区别于 CLEVR 系列仅限简单桌面环境，本文注重构建接近真实的复杂场景，同时对模型的3D空间感知能力提出更高要求

#### 2. **差异感知模块（Differential Perception）**
- **功能**：通过通道级减法提取图像对的差异特征，并通过交叉注意力融合回原始特征
- **核心思路**：
  1. 对图像对特征 $\mathbf{F}_1^i, \mathbf{F}_2^i$ 进行通道级门控调制：$\boldsymbol{\lambda}_k = \sigma(\mathbf{W}_m[\mathbf{F}_1^i \| \mathbf{F}_2^i])$
  2. 计算差异特征：$\Delta\mathbf{F}^i = \mathbf{W}_p[\hat{\mathbf{F}}_1^i \| \hat{\mathbf{F}}_2^i \| (\hat{\mathbf{F}}_1^i - \hat{\mathbf{F}}_2^i)]$
  3. 通过自注意力和交叉注意力将差异信号融合回原始特征
- **设计动机**：直接减法对未对齐图像对鲁棒性差，通过门控机制和交叉注意力增强对干扰因素（如视角/光照变化）的抗干扰能力

#### 3. **多尺度集成（Multi-Scale Integration）**
- **功能**：融合视觉编码器不同层（第17/20/23/26层）的特征
- **核心思路**：
    - 通过均值池化和 MLP 计算每层的融合权重 $\text{Score}^i$
    - 加权求和得到最终特征：$\mathbf{F}_k' = \sum_i \text{Score}^i \odot \tilde{\mathbf{F}}_k^i$
- **设计动机**：低层特征缺乏语义一致性，高层特征丢失细节感知，多层融合可兼顾语义和细节

### 损失函数 / 训练策略

- 采用简单高效的**单阶段微调**策略，不同于先前的多阶段方法
- 对 LLM 使用 LoRA（rank=128, alpha=256）进行参数高效微调
- 视觉编码器、投影器和 MDP 模块进行全参数微调
- 微调数据集包含896K问答对，来源于 OmniDiff + Spot-the-Diff + IEdit + Birds-to-Words + CLEVR-Change + CLEVR-DC
- 8×A100(40G) GPU，全局 batch size 256，训练26小时

## 实验关键数据

### 主实验（OmniDiff 基准）

| 方法 | Real BLEU-4 | Real CIDEr | Render BLEU-4 | Render CIDEr |
|------|------------|------------|---------------|-------------|
| CARD (ACL'24) | 9.1 | 9.2 | 11.3 | 7.3 |
| GPT-4o（zero-shot）| 3.1 | 5.2 | 4.6 | 5.6 |
| Qwen-2.5-VL-7B（zero-shot）| 3.8 | 6.2 | 2.1 | 3.3 |
| FINER-MLLM (MM'24) | 8.9 | 11.7 | 13.6 | 14.0 |
| **M3Diff (ours)** | **14.3** | **31.3** | **15.7** | **28.3** |

### 跨基准性能（Spot-the-Diff / IEdit / CLEVR-DC）

| 基准 | 指标 | M3Diff | 之前SOTA | 提升 |
|------|------|--------|----------|------|
| Spot-the-Diff | CIDEr | **71.1** | 61.8 (FINER-MLLM) | +15% |
| IEdit | CIDEr | **136.6** | 109.6 (OneDiff) | +25% |
| CLEVR-DC | CIDEr | **109.4** | 84.1 (DIRL) | +30% |

### 消融实验

| 配置 | OmniDiff-Real CIDEr | IEdit CIDEr | 说明 |
|------|---------------------|-------------|------|
| w/o OmniDiff & MDP | 1.1 | 133.5 | 基线 |
| w/o OmniDiff | 1.9 | 132.8 | 仅加 MDP |
| w/o MDP | 35.3 | 135.2 | 仅加 OmniDiff 数据 |
| M3Diff (full) | **31.3** | **136.6** | 数据+模块均有贡献 |

### 关键发现

1. 零样本 MLLM（GPT-4o、Qwen-2.5-VL）在复杂场景差异描述任务上表现不佳，说明 IDC 需要专门的差异感知能力
2. OmniDiff 数据集对复杂场景的描述能力至关重要——没有 OmniDiff 训练数据时模型在该基准上几乎失效
3. MDP 模块和 OmniDiff 数据互补：数据提供场景多样性，模块增强差异感知能力
4. M3Diff 的单阶段微调策略在避免复杂训练流程的同时实现了 SOTA

## 亮点与洞察

1. **数据集设计精良**：12种变化类型 × 真实+合成场景 × 60词平均描述长度，弥补了现有 IDC 基准的广度和深度不足
2. **即插即用设计**：MDP 模块通过一阶段微调即可嵌入已有 MLLM，工程实现简洁
3. **多尺度差异感知**：将视觉编码器多层的特征差异显式建模，比单依赖最后一层更有效
4. 3D合成场景使用 ArtStation 资源构建近真实复杂环境，而非 CLEVR 式的简单桌面

## 局限与展望

1. 数据集规模（15,598对）虽然在描述质量上有优势，但总量上不如 CLEVR-Change（79,606对），可进一步扩展
2. 当前仅支持图像对的差异描述，未扩展到视频序列的连续变化描述
3. 3D合成数据与真实数据的比例约为1:1.2，可能存在域偏移
4. MDP 模块在部分指标上（如 OmniDiff-Real CIDEr）出现轻微波动，需更鲁棒的融合策略

## 相关工作与启发

- FINER-MLLM 和 OneDiff 是最近将 MLLM 应用于 IDC 的代表工作，但缺乏专门的差异感知模块
- CLEVR 系列提供了可控的合成评估环境，本文将合成场景从简单桌面扩展到复杂真实环境
- 多尺度特征融合的思路可借鉴到其他需要细粒度视觉对比的任务（如 VQA、图像编辑评估）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 数据集构建思路完整，MDP 模块设计清晰，但技术方案相对直接
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖6个基准，消融充分，包含零样本 MLLM 对比
- **写作质量**: ⭐⭐⭐⭐ — 数据集和方法的组织逻辑清晰
- **价值**: ⭐⭐⭐⭐ — OmniDiff 填补了 IDC 基准的空白，M3Diff 提供了强基线

<!-- RELATED:START -->

## 相关论文

- [3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)
- [On the Robustness Tradeoff in Fine-Tuning](on_the_robustness_tradeoff_in_fine-tuning.md)
- [Enabling Fine-Grained Operating Points for Black-Box LLMs](../../ICLR2026/llm_evaluation/enabling_fine-grained_operating_points_for_black-box_llms.md)
- [PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](../../ACL2025/llm_evaluation/physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)
- [ForCenNet: Foreground-Centric Network for Document Image Rectification](forcennet_foreground-centric_network_for_document_image_rectification.md)

<!-- RELATED:END -->
