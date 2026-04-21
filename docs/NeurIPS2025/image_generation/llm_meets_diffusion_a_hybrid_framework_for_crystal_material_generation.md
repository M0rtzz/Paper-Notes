---
title: >-
  [论文解读] LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation
description: >-
  [NeurIPS 2025][图像生成][晶体材料生成] 提出CrysLLMGen混合框架，结合LLM擅长离散原子类型预测和扩散模型擅长连续坐标/晶格参数建模的互补优势，在晶体材料生成任务中同时实现高结构有效性和组成有效性。
tags:
  - NeurIPS 2025
  - 图像生成
  - 晶体材料生成
  - LLM
  - 扩散模型
  - 混合框架
  - 条件生成
---

# LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation

**会议**: NeurIPS 2025  
**arXiv**: [2510.23040](https://arxiv.org/abs/2510.23040)  
**代码**: [GitHub](https://github.com/kdmsit/crysllmgen)  
**领域**: 扩散模型 / 晶体材料生成  
**关键词**: 晶体材料生成, LLM, 扩散模型, 混合框架, 条件生成

## 一句话总结

提出CrysLLMGen混合框架，结合LLM擅长离散原子类型预测和扩散模型擅长连续坐标/晶格参数建模的互补优势，在晶体材料生成任务中同时实现高结构有效性和组成有效性。

## 研究背景与动机

晶体材料生成是材料科学中的关键挑战，它需要同时预测离散变量（原子类型）和连续变量（原子坐标和晶格参数）。现有生成方法主要分为两大类：

**LLM方法**（如LLaMA-2微调）：擅长处理离散的原子类型信息，组成有效性高（~93%），但由于有限精度编码的限制，在连续数据（坐标和晶格）上表现较弱，结构有效性偏低（~96%）。

**去噪模型**（如CDVAE、DiffCSP）：天然适合处理连续变量并保持几何等变性，结构有效性高（~100%），但在离散原子类型预测上不够准确，组成有效性偏低（~83-87%）。

核心矛盾在于：没有任何单一模型能同时在离散和连续两类变量上都表现出色。本文的核心insight是：既然两类模型各有所长，不如将它们组合起来，让LLM负责原子类型预测，扩散模型负责坐标和晶格的精准调整。

## 方法详解

### 整体框架

CrysLLMGen包含两个独立训练的模块：微调的LLM（LLaMA-2-7B）和预训练的扩散模型（基于DiffCSP的等变扩散网络）。采样过程分两步进行：
1. LLM生成原子类型$\hat{A}$、坐标$\hat{X}$和晶格$\hat{L}$的中间表示
2. 保留$\hat{A}$，将$\hat{X}$和$\hat{L}$送入扩散模型从中间时间步$\tau$开始去噪精炼

### 关键设计

1. **LLM组件（$f_\phi^{LLM}$）**：基于LLaMA-2-7B，将晶体结构转换为CIF文本格式的序列表示。分数坐标保留两位小数，晶格长度一位小数，角度取整数。使用LoRA进行微调，并采用数据增强策略处理平移和旋转对称性。LLM的核心优势在于自回归预测时能有效捕捉原子类型的离散分布，且支持自然语言条件输入。

2. **等变扩散模型（$f_\theta^{Diff}$）**：训练为结构预测任务——给定原子类型$A$，学习坐标$X$和晶格$L$的联合分布。坐标扩散使用Wrapped Normal分布处理周期性边界条件：
$$X_t = f_w(X_0 + \sigma_t \epsilon^X)$$
其中$f_w$为截断函数确保坐标在$[0,1)$内。晶格扩散使用标准DDPM：
$$L_t = \sqrt{\bar{\alpha}_t} L_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon^L$$
去噪网络基于CSPNet（等变图神经网络），通过Fourier变换处理分数坐标的相对位置差，保证周期性平移不变性。

3. **中间时间步注入策略**：与FlowLLM不同，CrysLLMGen不是直接将LLM输出送入扩散模型，而是将其视为中间表示，在时间步$\tau$（$0 \le \tau \le T$）注入扩散模型开始去噪。这基于一个关键观察：LLM输出不是纯噪声，而是有意义的近似结构，因此不需要从$T$步开始全程去噪。$\tau$作为超参数在验证集上选择。

### 损失函数 / 训练策略

LLM和扩散模型独立并行训练（区别于FlowLLM的串行训练），扩散模型使用联合损失：

$$\mathcal{L} = \mathcal{L}_{lattice} + \mathcal{L}_{coord}$$

坐标损失$\mathcal{L}_{coord}$为Score Matching目标，晶格损失$\mathcal{L}_{lattice}$为标准DDPM的$\ell_2$去噪损失。

推理时约2-5%的LLM生成样本因无效化学元素被过滤。

## 实验关键数据

### 主实验：De Novo材料生成

| 数据集 | 指标 | CrysLLMGen (7B) | LLaMA-2 (7B) | DiffCSP | FlowMM |
|--------|------|-----------------|--------------|---------|--------|
| MP-20 | 结构有效性↑ | **99.94** | 97.70 | 100 | 96.85 |
| MP-20 | 组成有效性↑ | **93.55** | 93.55 | 83.25 | 83.19 |
| MP-20 | COV-Precision↑ | **99.84** | 99.32 | 99.76 | 99.58 |
| MP-20 | COV-Recall↑ | **98.52** | 96.95 | 99.71 | 99.49 |
| Perov-5 | 结构有效性↑ | **100** | 99.09 | 100 | 100 |
| Perov-5 | 组成有效性↑ | **98.92** | 98.92 | 98.85 | 97.91 |

### 稳定性-唯一性-新颖性（S.U.N.）

| 模型 | % Meta-Stable↑ | % M.S.U.N.↑ | % Stable↑ | % S.U.N.↑ |
|------|----------------|-------------|-----------|----------|
| CrysLLMGen (7B) | **62.02** | **35.94** | **16.79** | **9.21** |
| LLaMA-2 (7B) | 56.60 | 26.66 | 12.67 | 4.84 |
| SymmCD | 40.01 | 31.69 | 9.99 | 6.76 |
| DiffCSP++ | 42.39 | 30.56 | 8.58 | 6.55 |

### 关键发现

- CrysLLMGen在MP-20上组成有效性相比最佳去噪模型提升4.64%，结构有效性相比LLM提升2.29%
- 生成的稳定材料比LLM多32%，比最佳去噪模型多68%
- 条件生成中空间群匹配率比LLM基线提升42%，得益于扩散模型的结构精炼
- LLM负责原子组成降低了$E_{hull}$中的组成不稳定性贡献，扩散模型降低了多晶型能量差异

## 亮点与洞察

- 框架设计简洁有效：并行训练+串行推理，架构无关（可替换更先进的LLM和扩散模型）
- 中间时间步注入$\tau$的设计巧妙，避免了全程去噪的浪费
- LLM的自然语言接口天然支持条件生成，比去噪模型更灵活
- 首次系统地验证了LLM+扩散的混合策略在材料生成中的优越性

## 局限与展望

- LLM和扩散模型之间没有交互，两模块训练完全独立，未来可探索双向反馈机制
- 仅使用了LLaMA-2-7B，更强的LLM（如LLaMA-3）可能带来进一步提升
- 扩散模型基于vanilla DiffCSP，更先进的晶格生成模型可能改善结果
- $\tau$需要在验证集上调参，自适应选择策略更有价值

## 相关工作与启发

- FlowLLM同为LLM+流模型混合，但采用串行训练和直接精炼策略，CrysLLMGen的中间时间步注入更加灵活
- 这种"离散部分用LLM、连续部分用扩散"的设计思想可推广到其他多模态生成任务
- 对称性感知生成（DiffCSP++、SymmCD）是互补方向，可集成到框架中

## 评分

- **新颖性**: ⭐⭐⭐⭐ 混合框架思路直观但首次在材料生成中系统验证，中间时间步注入有新意
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖无条件生成、S.U.N.、条件生成三大任务，baselines全面
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，与FlowLLM的区别讨论充分
- **价值**: ⭐⭐⭐⭐ 为材料科学提供了实用的混合框架，架构无关设计便于未来扩展

<!-- RELATED:START -->

## 相关论文

- [ObCLIP: Oblivious Cloud-Device Hybrid Image Generation with Privacy Preservation](obclip_oblivious_cloud-device_hybrid_image_generation_with_privacy_preservation.md)
- [MARBLE: Material Recomposition and Blending in CLIP-Space](../../CVPR2025/image_generation/marble_material_recomposition_and_blending_in_clip-space.md)
- [HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation](../../ECCV2024/image_generation/hybridbooth_hybrid_prompt_inversion_for_efficient_subje.md)
- [Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction](toward_a_unified_geometry_understanding_riemannian_diffusion_framework_for_graph.md)
- [DDIS: When Model Knowledge Meets Diffusion Model](../../ICML2025/image_generation/when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)

<!-- RELATED:END -->
