---
title: >-
  [论文解读] CompSlider: Compositional Slider for Disentangled Multiple-Attribute Image Generation
description: >-
  [ICCV 2025][图像生成][属性解耦] 提出 CompSlider，一个组合式滑块模型，通过生成条件先验来实现对 T2I 基础模型中多个属性的同时、独立、细粒度控制，利用解耦损失和结构损失来解决多属性之间的纠缠问题。
tags:
  - ICCV 2025
  - 图像生成
  - 属性解耦
  - 滑块控制
  - 文本到图像生成
  - 条件先验
  - 多属性操控
---

# CompSlider: Compositional Slider for Disentangled Multiple-Attribute Image Generation

**会议**: ICCV 2025  
**arXiv**: [2509.01028](https://arxiv.org/abs/2509.01028)  
**代码**: 无  
**领域**: image_generation  
**关键词**: 属性解耦, 滑块控制, 文本到图像生成, 条件先验, 多属性操控

## 一句话总结

提出 CompSlider，一个组合式滑块模型，通过生成条件先验来实现对 T2I 基础模型中多个属性的同时、独立、细粒度控制，利用解耦损失和结构损失来解决多属性之间的纠缠问题。

## 研究背景与动机

在文本到图像（T2I）生成中，仅通过文本 prompt 难以精确控制图像属性的强度（如年龄、微笑程度），因此出现了基于滑块的生成方法（如 ConceptSliders、PromptSlider），允许用户通过滑块连续调节属性。但现有方法为每个属性单独训练一个 adapter，忽视了多属性之间的纠缠问题：

**属性纠缠**：按不同顺序叠加滑块会导致不同结果，例如先加 smile 再加 age 与反序结果不同

**结构不一致**：调节一个属性时会改变背景、发型等无关因素

**可扩展性差**：N 个属性需要 N 次前向传播，计算负担大

## 方法详解

### 整体框架

CompSlider 替代了 T2I 基础模型中 CLIP 图像编码器的角色。输入为用户定义的滑块值和文本 prompt，输出为图像条件 $\bm{c}^{\mathcal{I}}$，作为多属性先验送入基础扩散模型生成图像。公式为：

$$\bm{c}^{\mathcal{I}} = \text{CompSlider}(\bm{c}^{\mathcal{S}}, \bm{c}^{\mathcal{T}})$$

其中 $\bm{c}^{\mathcal{S}}$ 为滑块嵌入，$\bm{c}^{\mathcal{T}}$ 为 T5 文本 token。整个过程不需要微调基础模型。

### 关键设计

1. **DiT 扩散模型作为 CompSlider 骨干**：使用 Diffusion Transformer (DiT) 模型，采用重参数化技巧直接预测纯净图像条件 $\bm{c}_0^{\mathcal{I}}$ 而非噪声。由于图像条件是 1024 维向量，不需要 U-Net 中的下采样操作，DiT 更合适。模型包含 10 个 DiT block，输入 128 个文本 token 和 16 个滑块 token，总参数量 277M。

2. **滑块值嵌入机制**：通过预训练属性分类器获取属性分数并归一化到 [0,1]，使用位置编码（正弦余弦编码）将连续滑块值映射为向量 $\bm{p}^{\mathcal{S}} \in \mathbb{R}^{N \times \frac{dim}{2}}$，并引入可学习类别嵌入 $\bm{w} \in \mathbb{R}^{N \times \frac{dim}{2}}$ 让模型区分不同属性。最终滑块嵌入为两者拼接：$\bm{c}^{\mathcal{S}} = [\bm{p}^{\mathcal{S}}, \bm{w}]$。

3. **随机属性组合训练策略**：关键创新在于不依赖配对数据（即同一个人不同属性强度的图像），而是在训练中引入随机采样的属性值组合 $\bm{v}^{\mathcal{S}*}$，确保模型不仅学到训练数据中常见的属性共现模式，还能泛化到任意组合。

### 损失函数 / 训练策略

总损失由三部分组成：$\mathcal{L} = \mathcal{L}_{\text{diff}} + \mathcal{L}_{\text{st}} + \mathcal{L}_{\text{clss}}$

- **扩散损失** $\mathcal{L}_{\text{diff}}$：确保生成的条件与 CLIP 图像编码器的输出域匹配，$\mathcal{L}_{\text{diff}} = \mathbb{E}[\|\bm{c}_0^{\mathcal{I}} - \text{DiT}(\bm{c}_t^{\mathcal{I}}, \bm{c}^{\mathcal{S}}, \bm{c}^{\mathcal{T}}, t)\|^2]$
- **解耦损失** $\mathcal{L}_{\text{clss}}$：训练 MLP 分类器从原始和随机属性组合的条件差异中恢复属性差值。将差值离散化为 B=20 个桶，用交叉熵损失约束
- **结构损失** $\mathcal{L}_{\text{st}}$：当属性差值 $|\Delta v_i| \leq 0.1$ 时，约束两组条件输出的 L2 距离，保持局部结构一致性

训练数据约 300 万张图像，扩散损失和结构损失训练 DiT，解耦损失同时训练 DiT 和 MLP 分类器。

## 实验关键数据

### 主实验 (表格)

人类相关滑块定量对比（300 prompts × 5 属性 × 5 值 = 7500 图像）：

| 方法 | Cont.%↑ | Cons.%↑ | Scope%↑ | Entang.%↓ | LPIPS↓ | CLIP↑ |
|------|---------|---------|---------|-----------|--------|-------|
| Prompt2Prompt | - | 88.47 | 49.46 | 28.99 | 0.19 | 4.15 |
| PromptSlider | 61.17 | 80.23 | 46.25 | 24.31 | 0.10 | 4.79 |
| ConceptSlider | 73.41 | 83.17 | 54.43 | 27.22 | 0.16 | 5.76 |
| **CompSlider** | **81.07** | **90.95** | **59.02** | **14.04** | 0.12 | **6.20** |

非人类滑块 A/B 用户测试（Vector Style + Scene Complexity）：CompSlider 用户偏好 54.66% vs ConceptSlider 34.16%。

### 消融实验 (表格)

解耦损失和结构损失的消融：

| $\mathcal{L}_{\text{diff}}$ | $\mathcal{L}_{\text{clss}}$ | $\mathcal{L}_{\text{st}}$ | Cont.%↑ | Cons.%↑ | Scope%↑ | Entang.%↓ |
|:---:|:---:|:---:|---------|---------|---------|-----------|
| ✓ | | | 68.96 | 63.21 | 42.06 | 36.68 |
| ✓ | ✓ | | 76.49 | 49.29 | 63.27 | 19.87 |
| ✓ | ✓ | ✓ | **81.07** | **90.95** | **59.02** | **14.04** |

### 关键发现

- 仅用扩散损失时纠缠率高达 36.68%，加入解耦损失后降至 19.87%，但结构一致性崩溃（49.29%）
- 结构损失将一致性从 49.29% 大幅提升至 90.95%（+41.66%），同时进一步降低纠缠至 14.04%
- CompSlider 支持单次前向传播控制所有滑块，推理效率远优于逐属性的方法

## 亮点与洞察

- **核心创新**：在条件先验的潜空间中操作，不需要微调基础模型，大幅降低训练和推理成本
- **无需配对数据**：通过随机采样属性组合训练解耦，巧妙绕过了获取同一主体不同属性强度配对数据的困难
- **提出 4 个新评估指标**：Continuity、Scope、Consistency、Entanglement，比 LPIPS/CLIP 更全面衡量滑块生成质量
- 可扩展到视频生成

## 局限与展望

- 依赖预训练属性分类器获取滑块值的 ground truth，分类器质量影响训练
- 滑块属性集合是封闭集（16 个预定义），不支持开放域属性
- 未与更新的扩散模型（如 SDXL、SD3）结合验证
- 非人类属性缺乏自动化评估指标，只能依赖用户研究

## 相关工作与启发

- ConceptSliders 和 PromptSlider 是直接前驱，分别用 LoRA adapter 和 textual inversion 做单属性滑块
- eDiff-I 提供了条件图像先验的基础框架
- 解耦思路可启发其他多条件可控生成任务（如同时控制构图、风格、内容等）

## 评分

- 新颖性: ⭐⭐⭐⭐ 在条件先验空间做组合滑块是新颖视角，解耦损失设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 人类和非人类属性都做了评估，提出新指标，有消融和扩展
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述详尽，图表丰富
- 价值: ⭐⭐⭐⭐ 解决了多属性同时控制的实际问题，有明确应用场景

<!-- RELATED:START -->

## 相关论文

- [All-in-One Slider for Attribute Manipulation in Diffusion Models](../../CVPR2026/image_generation/all_in_one_slider_attribute_manipulation.md)
- [Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention](fair_generation_without_unfair_distortions_debiasing_text-to-image_generation_wi.md)
- [Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](../../NeurIPS2025/image_generation/evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)
- [Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Models and Small Edge Models](adaptive_routing_of_text-to-image_generation_requests_between_large_cloud_model_.md)
- [coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](../../CVPR2025/image_generation/codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)

<!-- RELATED:END -->
