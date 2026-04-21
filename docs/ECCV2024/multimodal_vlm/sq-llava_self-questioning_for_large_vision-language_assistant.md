---
title: >-
  [论文解读] SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant
description: >-
  [ECCV 2024][多模态][Visual Instruction Tuning] 提出视觉自提问（Visual Self-Questioning）训练范式，让 LLM 不仅学习回答问题，还学习根据图像主动提问，通过充分利用指令数据中问题本身的丰富语义信息来增强视觉-语言对齐。
tags:
  - ECCV 2024
  - 多模态
  - Visual Instruction Tuning
  - Self-Questioning
  - Prototype Extractor
  - LoRA
  - 视觉语言
---

# SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant

**会议**: ECCV 2024  
**arXiv**: [2403.11299](https://arxiv.org/abs/2403.11299)  
**代码**: https://github.com/heliossun/SQ-LLaVA (有)  
**领域**: 多模态视觉语言模型  
**关键词**: Visual Instruction Tuning, Self-Questioning, Prototype Extractor, LoRA, Vision-Language Alignment

## 一句话总结
提出视觉自提问（Visual Self-Questioning）训练范式，让 LLM 不仅学习回答问题，还学习根据图像主动提问，通过充分利用指令数据中问题本身的丰富语义信息来增强视觉-语言对齐。

## 研究背景与动机

当前大规模视觉语言模型（如 LLaVA 系列）通过视觉指令微调在多种视觉任务上取得良好泛化。然而，预训练视觉编码器和大语言模型之间的**模态鸿沟**依然是整个网络的瓶颈。现有方法通常通过收集更多、更高质量的视觉指令数据来改善跨模态对齐，但这种方式成本高昂，且并未充分挖掘图像中蕴含的丰富上下文信息。

作者发现了一个被忽视的关键洞察：在现有视觉指令数据集中，**问题往往比答案包含更多与图像相关的信息**。通过计算 CLIPScore 验证，在 LLaVA-instruct 数据集上，问题的平均 CLIPScore（μ_q=0.184）高于答案（μ_a=0.183）；在 ShareGPT4V-instruct 上同样如此（μ_q=0.186 > μ_a=0.184）。这说明问题中编码了更丰富的视觉线索。

**核心矛盾**在于：传统视觉指令微调只训练模型预测答案，完全忽略了问题中包含的语义信息，这是一种资源浪费。

**本文的切入角度**：类比人类学习——主动提问比回答问题需要更深层的理解和背景知识。因此，训练模型学习"如何提问"可以促进更深层的视觉-语言对齐。

**核心 idea**：将"提问"作为额外的训练目标，通过自监督方式利用指令数据中的问题文本，无需收集额外数据即可增强模型的视觉理解能力。

## 方法详解

### 整体框架
SQ-LLaVA 由四个核心组件构成：
1. **预训练视觉编码器**（CLIP-ViT）：提取图像 token 序列嵌入 Z_v
2. **原型提取器** φ(·)：通过聚类学习增强原始图像 token 表征
3. **可训练投影模块** W(·)：两层线性层，将增强后的图像 token 映射到语言域 H_v
4. **LLM 骨干** f(·)：基于预训练 Vicuna，进行自回归 next-token 预测

模型遵循标准自回归预测范式：$p_\theta(H_a^{(i+1)} | H_v, H_q, H_a^{(1:i)}) = \sigma(f(H_v, H_q, H_a^{(1:i)}))$

### 关键设计

1. **视觉自提问指令（Visual Self-Questioning Instruction）**:

    - **功能**：定义新的特殊 token `[vusr]`，作为"提问"指令，让模型在看到图像后主动生成相关问题
    - **核心思路**：对于多轮对话数据中第 j 轮，以 50% 概率（δ=0.5）将 `[usr]` 替换为 `[vusr]`，此时模型需预测问题序列而非执行用户指令。训练序列格式为：`System-message → [vusr] → X_q → [aswr] → X_a → <o^d>`
    - **设计动机**：在现实中，主动提出好问题比回答问题需要更多理解。通过学习提问，模型被迫建立图像与问题之间更深层的对齐关系。训练后，SQ-LLaVA 能零样本生成多样化问题，包括多选题、推理题等，多样性甚至超过 GPT-4V

2. **原型提取器（Prototype Extractor）**:

    - **功能**：通过 EM 聚类算法从图像 token 潜在空间中提取语义原型，增强视觉表征
    - **核心思路**：随机初始化 K=256 个聚类中心 C，通过 T=2 次迭代 EM 步骤优化。E-step 计算软分配矩阵 $\mathcal{M}^{(t)} = \sigma(q(C^{(t)}) \cdot k(Z_v)^\top)$，M-step 更新中心 $C^{(t+1)} = \mathcal{M}^{(t)} \cdot v(Z_v)$。然后将原型信息通过余弦相似度加权聚合回原始 token：$Z_v^{(i)} = Z_v^{(i)} + z(\frac{1}{K}\sum_{j=1}^K S_c(C_j, Z_v^{(i)}) \times C_j)$
    - **设计动机**：聚类将语义相似的 token 归并，使原型能描述内在语义（如"草地"、"狗"），增强上下文理解，弥补投影层的表征不足

3. **ViT-LoRA + LLM-LoRA 联合微调**:

    - **功能**：在微调阶段同时对视觉编码器和 LLM 添加 LoRA 适配器
    - **核心思路**：ViT-LoRA (rank=32, α=64) 和 LLM-LoRA (rank=128, α=256)，保持预训练权重冻结，仅训练 LoRA、原型提取器和投影层
    - **设计动机**：以极少可训练参数实现视觉和语言域的联合优化，避免全量微调的高计算开销

### 损失函数 / 训练策略

**两阶段训练**：
- **Stage1 预训练**：冻结 ViT 和 LLM，只训练原型提取器 φ 和投影层 W。目标为最大化图像描述预测概率：$\sum_{v,a} -\log p_\theta(H_a | H_v)$
- **Stage2 微调**：添加 LoRA，优化两个目标——自提问损失 $-\log p_\theta(H_q^{(j+1)} | H_v, H_c^{(1:j)})$ 和回答损失 $-\log p_\theta(H_a^{(j+1)} | H_v, H_c^{(1:j)}, H_q^{(j+1)})$

## 实验关键数据

### 主实验

| 基准 | SQ-LLaVA-7B | LLaVA-v1.5-7B | ShareGPT4V-7B | 要点 |
|------|-------------|---------------|---------------|------|
| VQAv2 | 79.2 | 78.5 | 80.6 | +0.7 vs LLaVA |
| GQA | 62.8 | 62.0 | 63.3 | +0.8 vs LLaVA |
| VizWiz | 54.0 | 50.0 | 57.2 | +4.0 vs LLaVA |
| ScienceQA-IMG | 68.9 | 66.8 | 68.4 | +2.1 vs LLaVA |
| POPE | 87.7 | 85.9 | 86.8 | +1.8/+0.9 |
| MM-Vet | 32.5 | 30.5 | 37.6 | +2.0 vs LLaVA |
| LLaVA-Wild | 66.3 | 63.4 | 72.6 | +2.9 vs LLaVA |
| MMBench | 66.2 | 64.3 | 68.8 | +1.9 vs LLaVA |

SQ-LLaVA-7B 在 10 个基准中的 9 个超过 LLaVA-v1.5-7B，13B 规模同样在 8/10 基准上取得提升。

### 消融实验

| 配置 (558K PT + 665K IT) | VizWiz | SQAI | VQAT | POPE | LLaVAW | Avg. |
|--------------------------|--------|------|------|------|--------|------|
| Baseline（无 LoRA/SQ/Proto） | 49.4 | 68.4 | 58.2 | 86.5 | 67.1 | 65.9 |
| + V-LoRA + Proto | 52.4 | 67.9 | 58.6 | 87.7 | 65.6 | 66.4 |
| + V-LoRA + SQ | 52.6 | 68.4 | 57.8 | 88.2 | 67.3 | 66.9 |
| + SQ + Proto | 53.4 | 69.3 | 58.1 | 87.9 | 67.9 | 67.3 |
| Full (V-LoRA + SQ + Proto) | 54.0 | 68.9 | 58.6 | 87.7 | 68.1 | **67.5** |

- 完整模型比 baseline 平均提升 **2.4%**（小数据集）/ **3.0%**（大数据集）
- SQ 模块贡献最为一致，单独使用即可在所有基准上获益

### 关键发现
- 视觉自提问（SQ）在几乎所有基准上带来一致提升，验证了"学会提问改善理解"的假说
- 在 POPE 基准上的提升表明 SQ-LLaVA 能有效减少物体幻觉，提高可信度
- 在零样本图像描述任务上，SQ-LLaVA 平均比 LLaVA-v1.5 提升 2%，并能生成包含具体概念（如品牌名 "Hyundai"）的描述

## 亮点与洞察
- **问题比答案更具视觉相关性**——用 CLIPScore 量化验证了这一直觉，开辟了视觉指令调优的新维度
- **零数据增益**——不需要任何额外数据收集，仅通过重新利用已有数据中的问题文本即可提升性能
- **可训练参数显著减少**——LoRA 策略使微调开销远低于全量微调方案

## 局限与展望
- 原型提取器是无监督的，缺少像素级引导，在某些数据集上可能不稳定；可利用预训练分割模型（如 SAM）提供伪目标掩码
- 在 VQAv2 和 MMBench 等部分基准上仍落后于 ShareGPT4V，说明数据质量和规模仍然重要
- 自提问阈值 δ=0.5 为固定值，未探索动态调整策略

## 相关工作与启发
- **vs LLaVA-v1.5**：SQ-LLaVA 在相同数据上通过自提问获得额外增益，说明数据利用效率可以进一步提高
- **vs ShareGPT4V**：SQ-LLaVA* 在部分基准上追平或超越 ShareGPT4V，但使用更少可训练参数
- **vs GPT-4V 提问能力**：GPT-4V 需要显式语言指令才能提问，且依赖 prompt 工程；SQ-LLaVA 零样本提问，多样性更高

## 评分
- 新颖性: ⭐⭐⭐⭐ 将"提问"作为训练目标是视觉指令微调领域的首次尝试，角度新颖
- 实验充分度: ⭐⭐⭐⭐ 10 个 VQA 基准 + 4 个 captioning 数据集 + 详细消融，但缺少更多分析
- 写作质量: ⭐⭐⭐⭐ 动机论证清晰，CLIPScore 可视化直观，整体逻辑流畅
- 价值: ⭐⭐⭐⭐ 提出了一种零成本提升方案，对社区有启发意义，但绝对提升幅度有限

<!-- RELATED:START -->

## 相关论文

- [NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models](navgpt2_unleashing_navigational_reasoning_capability.md)
- [Omniview-Tuning: Boosting Viewpoint Invariance of Vision-Language Pre-training Models](omniviewtuning_boosting_viewpoint_invariance_of_visionlangua.md)
- [Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)
- [DeCUR: Decoupling Common and Unique Representations for Multimodal Self-supervised Learning](decoupling_common_and_unique_representations_for_multimodal_.md)
- [Robust Calibration of Large Vision-Language Adapters](robust_calibration_of_large_visionlanguage_adapters.md)

<!-- RELATED:END -->
