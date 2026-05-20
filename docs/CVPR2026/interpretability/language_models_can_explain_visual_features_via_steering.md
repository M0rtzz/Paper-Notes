---
title: >-
  [论文解读] Language Models Can Explain Visual Features via Steering
description: >-
  [CVPR 2026][可解释性][稀疏自编码器] 提出通过对VLM视觉编码器进行SAE特征因果干预（steering），在输入空白图像后让语言模型描述其"看到"的视觉概念，从而实现无需评估图像集的可扩展视觉特征自动解释，并提出混合方法Steering-informed Top-k达到SOTA。
tags:
  - "CVPR 2026"
  - "可解释性"
  - "稀疏自编码器"
  - "视觉特征解释"
  - "因果干预"
  - "VLM"
  - "自动可解释性"
---

# Language Models Can Explain Visual Features via Steering

**会议**: CVPR 2026  
**arXiv**: [2603.22593](https://arxiv.org/abs/2603.22593)  
**代码**: [GitHub](https://github.com/HPAI-BSC/vision-interp)  
**领域**: 可解释性  
**关键词**: 稀疏自编码器, 视觉特征解释, 因果干预, VLM, 自动可解释性

## 一句话总结

提出通过对VLM视觉编码器进行SAE特征因果干预（steering），在输入空白图像后让语言模型描述其"看到"的视觉概念，从而实现无需评估图像集的可扩展视觉特征自动解释，并提出混合方法Steering-informed Top-k达到SOTA。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：稀疏自编码器（SAE）已成为发现视觉模型中可解释特征的有力工具，但当SAE扩展到发现数千个特征时，自动解释这些特征仍是开放问题。

现有方法（Top-k方法）的局限：
1. **基于相关性而非因果性**：选择最高激活图像让解释器找共同模式，本质是相关性分析
2. **依赖评估图像集**：需要大规模图像集来找到top激活图像，引入数据集偏差
3. **计算成本高**：需要对整个评估集前向传播来排序激活值

本文核心insight：VLM将视觉编码器与预训练语言模型连接，如果我们对视觉编码器进行**因果干预**——在空白图像上注入特定SAE特征向量——语言模型应该能够表达它"看到"了什么视觉概念。

## 方法详解

### 整体框架

1. 训练TopK SAE（d_SAE=8192）在ImageNet上解码视觉编码器特征
2. 对每个SAE特征，在空白图像的前向传播中注入该特征向量
3. 提示语言模型解释其"看到"的内容
4. 可选：混合方法同时结合Top-k图像和因果干预

### 关键设计

1. **Steering-based解释（纯因果干预）**:
    - 功能：无需图像集即可生成SAE特征的自然语言解释
    - 核心思路：输入空白（全白）图像 $\tilde{I}$ 到VLM，在视觉编码器的第l层残差流中，对所有位置添加SAE解码器权重向量 $W_{dec}[i,:] \times \alpha$。语言模型基于被干预的视觉表示生成解释。形式化为：$e_i \sim m_{exp}(e | P, \tilde{I}, \text{do}(m_{sub}^l(\tilde{I}) \leftarrow m_{sub}^l(\tilde{I}) + \alpha W_{dec}[i,:]))$
    - 设计动机：空白图像不提供有意义的视觉信号，因此语言模型的输出完全由因果干预驱动，是纯粹的因果解释。仅需单次前向传播，极其高效

2. **Steering-informed Top-k（混合方法）**:
    - 功能：结合因果干预和输入图像的优势
    - 核心思路：在条件化Top-k激活图像的同时，对视觉编码器进行相同的SAE特征因果干预。将相关性证据（Top-k图像）和因果证据（特征注入）结合，引导解释器生成更精准的解释
    - 设计动机：纯Steering在低级特征上表现更好，Top-k在高级语义特征上更强。混合方法在四个互补指标上都达到最优，且不增加额外计算成本

3. **评估指标体系**:
    - 功能：从多角度量化解释质量
    - 支持四个互补指标：
        - 激活IoU：解释文本的高激活图像与SAE特征的高激活图像的重叠度
        - Detection Score：解释生成的文本能否被VLM在图像中检测到
        - CLIP相似度：解释与top激活图像的CLIP嵌入距离
        - 单义性（Monosemanticity）：特征是否对应单一概念

### 损失函数 / 训练策略

SAE训练使用标准TopK目标，在ImageNet上训练。干预强度α在500个特征的验证集上选择。解释生成方法本身不涉及训练——纯推理时干预。

## 实验关键数据

### 主实验 — 解释质量对比（Gemma 3视觉编码器）

| 方法 | 激活IoU↑ | Detection↑ | CLIP↑ | 单义性↑ |
|------|---------|-----------|-------|--------|
| Top-k (原始图像) | 基线 | 基线 | 基线 | 基线 |
| Top-k (Mask) | 略好 | 略好 | 微降 | 相当 |
| Top-k (Heatmap) | 相当 | 相当 | 相当 | 相当 |
| Steering (纯干预) | 低于Top-k | 低于Top-k | 低于Top-k | 相当 |
| **Steering-informed Top-k** | **最优** | **最优** | **最优** | **最优** |

### 消融实验 — 语言模型规模效应

| LM规模 | 解释质量趋势 |
|--------|-------------|
| 小模型 | 基准水平 |
| 中等模型 | 显著提升 |
| 大模型 | 持续提升 |

解释质量随语言模型规模持续改善，无饱和迹象

### 关键发现

- 纯Steering方法在低级特征（纹理、颜色、边缘）上优于Top-k，因果干预更能捕获这些原始视觉概念
- Top-k方法在高级语义特征（对象类别）上更强，因为有具体图像作为参考
- 混合方法（Steering-informed Top-k）在所有指标上达到SOTA，无额外计算开销
- 语言模型规模是解释质量的关键因素——更大的LM能更好地"表达"视觉概念
- 在Gemma 3和Intern VL3两个不同VLM上结论一致

## 亮点与洞察

- 从"相关性"到"因果性"的范式转变：Steering直接干预模型内部表示，比Top-k的相关性分析更有因果基础
- 极其高效：仅需单次前向传播即可解释一个特征，不需要遍历整个评估集
- 语言模型规模效应暗示未来更强的LM将进一步提升自动可解释性
- 混合方法的设计思路优雅：在Top-k的图像上下文中同时注入因果信号，两种信息互补

## 局限与展望

- 纯Steering在高级语义特征上弱于Top-k，因为空白图像缺乏上下文
- 干预强度α对结果敏感，需要在验证集上调优
- 仅在VLM架构上验证，纯视觉模型（无语言模型组件）无法直接应用
- SAE维度固定为8192，更大字典的扩展效果未知
- 评估指标主要是自动化指标，缺少人工评估

## 相关工作与启发

- **vs 标准Top-k方法**: Top-k基于相关性、需要评估集、计算密集；Steering基于因果性、无需图像集、单次前向传播
- **vs PatchScopes/SELFIE**: 这些方法在语言模型中做自解释，本文首次将范式扩展到视觉编码器
- **vs CB-SAE (同会议)**: CB-SAE关注SAE的可控性和可解释性度量，本文关注SAE特征的自然语言解释生成

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 因果干预解释视觉特征是全新范式，方法简洁优雅
- 实验充分度: ⭐⭐⭐⭐ 多指标、多VLM、规模效应分析，但缺少人工评估
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法直观，但LaTeX公式渲染有问题
- 价值: ⭐⭐⭐⭐ 对视觉模型自动可解释性研究有重要推动，方法可扩展性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](../../ICML2026/interpretability/corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [\[ICML 2026\] Interpretability Can Be Actionable](../../ICML2026/interpretability/interpretability_can_be_actionable.md)
- [\[CVPR 2026\] Draft and Refine with Visual Experts](draft_and_refine_with_visual_experts.md)
- [\[AAAI 2026\] Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs](../../AAAI2026/interpretability/finding_the_translation_switch_discovering_and_exploiting_the_task-initiation_fe.md)

</div>

<!-- RELATED:END -->
