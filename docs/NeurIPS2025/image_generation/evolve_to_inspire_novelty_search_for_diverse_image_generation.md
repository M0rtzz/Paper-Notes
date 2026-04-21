---
title: >-
  [论文解读] Evolve to Inspire: Novelty Search for Diverse Image Generation
description: >-
  [NeurIPS 2025][图像生成][新颖性搜索] 提出 Wander 框架，基于新颖性搜索（novelty search）和 LLM 驱动的 prompt 进化，从单个文本提示出发生成高度多样化的图像集合，在 Vendi Score 上超越现有进化式 prompt 优化基线。
tags:
  - NeurIPS 2025
  - 图像生成
  - 新颖性搜索
  - 图像多样性
  - 进化策略
  - 提示优化
  - CLIP
---

# Evolve to Inspire: Novelty Search for Diverse Image Generation

**会议**: NeurIPS 2025  
**arXiv**: [2511.00686](https://arxiv.org/abs/2511.00686)  
**代码**: 未公开  
**领域**: image_generation  
**关键词**: 新颖性搜索, 图像多样性, 进化策略, 提示优化, CLIP

## 一句话总结

提出 Wander 框架，基于新颖性搜索（novelty search）和 LLM 驱动的 prompt 进化，从单个文本提示出发生成高度多样化的图像集合，在 Vendi Score 上超越现有进化式 prompt 优化基线。

## 背景与动机

文本到图像扩散模型（如 FLUX、Stable Diffusion）在生成高保真图像方面表现出色，但在多样性上存在显著不足——重复同一 prompt 往往生成相似结果，手动调整 prompt 效果不可预测。这对创意探索、头脑风暴等需要快速生成多样想法的应用场景形成瓶颈。现有 prompt 优化技术（如 QDAIF、PromptBreeder 等）主要针对质量或 NLP 任务性能进行优化，而非系统性地提升视觉多样性。

## 核心问题

1. 如何从单一文本提示自动生成一组高度多样化的图像？
2. 基于 VLM 评分的 QDAIF 方法在图像领域失败——VLM 无法可靠识别视觉新颖性
3. 如何设计高效的 mutation 策略引导进化探索不同的提示空间区域？

## 方法详解

### Wander 框架概览

框架由三个核心组件迭代组成：**Emitter 选择**、**Prompt 进化**、**Pool 更新**。

### 问题形式化

给定包含最多 $N$ 个 prompt-image 对 $x_i = (p_i, I_i)$ 的池 $\mathbf{P}$，定义新颖性分数 $f(x_i, \mathbf{P}) \in [0, 1]$，目标是最大化池中最低的新颖性分数：

$$\mathbf{P}^* = \max_{\mathbf{P}} \left( \min_i (f(x_i, \mathbf{P})) \right)$$

### 新颖性度量

采用 k-近邻平均 CLIP 嵌入距离作为新颖性分数：

$$f(x_i, \mathbf{P}) = \frac{1}{k} \sum_{j=1}^{k} d(I_i, I_j)$$

其中 $d(I_i, I_j)$ 为 CLIP 图像嵌入之间的余弦距离。

### Emitter 机制

Emitter 是预定义的 mutation 策略（如"改变构图"、"调整光照"、"添加元素"等），嵌入 mutation prompt 中引导 LLM 在特定方向上变异。使用 bandit 策略或随机采样选择 emitter。

### Prompt 进化

每次 mutation 有两种操作（各 50% 概率）：
- **变异（Mutation）**：LLM 依据 emitter 指令修改单个 prompt
- **交叉（Crossover）**：LLM 将两个现有 prompt 的元素组合创造新变体

### Pool 更新

新生成的候选 prompt 经扩散模型生成图像后计算 CLIP 嵌入，若新颖性分数高于池中最低者则替换之。

### Vendi Score 评估

$$\text{VS}(K) = \exp\left(-\sum_{i=1}^{n} \lambda_i \log \lambda_i\right)$$

其中 $\lambda_i$ 是归一化多样性矩阵 $K/n$ 的特征值，反映池中有效多样样本数。

## 实验关键数据

### 主实验（10 个 prompt × 10 次运行）

| 方法 | Vendi↑ | LPIPS↑ | Relevance↑ | Token 用量↓ |
|------|--------|--------|------------|------------|
| EvoPrompt-DE | 1.42±0.04 | 0.51±0.01 | 0.292±0.001 | 38,243 |
| QDAIF | 1.80±0.02 | 0.51±0.02 | 0.297±0.001 | 43,464 |
| Lluminate | 3.29±0.02 | 0.75±0.01 | 0.210±0.070 | 175,902 |
| Wander-NE (无 emitter) | 2.61±0.10 | 0.79±0.01 | 0.279±0.004 | 23,884 |
| **Wander** | **3.60±0.09** | **0.80±0.01** | 0.272±0.003 | **24,347** |

- Wander 的 Vendi Score 比 Lluminate 高 9.4%，Token 用量仅为其 **1/7**
- 引入多 emitter 使 Vendi Score 从 2.61 提升至 3.60（+38%）

### LLM 模型对比（20 代进化）

| 模型 | Vendi Score↑ | Token 用量↓ |
|------|-------------|------------|
| GPT-4o-mini | 4.2±0.1 | 61,402 |
| GPT-4o | 4.8±0.1 | 78,067 |
| o3 | **5.2±0.1** | 236,081 |

更强的 LLM 产生更多样的结果，但推理 token 消耗也更大。

## 亮点

- ⭐ 新颖性搜索 + CLIP 嵌入距离的组合简洁有效，无需微调任何模型
- ⭐ Emitter 机制设计巧妙，多 emitter随机采样即可显著提升多样性
- ⭐ 固定大小池设计比 Lluminate 的无界池节省 7 倍 token
- 方法完全模型无关（model-agnostic），可迁移到任意扩散模型
- UMAP 可视化清晰展示了进化过程中嵌入空间的多样性扩展

## 局限与展望

- 新颖性目标偶尔导致语义漂移（约每 5 次运行出现 1 次），可加入相关性惩罚
- Emitter 需手动设计，可能限制或偏置多样性上限
- 未评估生成图像的美学质量（仅在 FLUX-DEV 上未观察到问题）
- 仅在单一扩散模型（FLUX-DEV）上验证，跨模型泛化有待确认
- Relevance 分数略低于 QDAIF（0.272 vs 0.297），存在一定语义保持 trade-off

## 与相关工作的对比

| 方法 | 初始 Prompt | 目标 | 进化策略 |
|------|-----------|------|---------|
| APE | 多个 | 适应度 | 交叉 |
| QDAIF | 单个 | 质量-多样性 | 定向变异 |
| Lluminate | 单个 | 新颖性 | 创意策略变异 |
| **Wander** | **单个** | **新颖性** | **定向变异+交叉** |

## 启发与关联

- 框架可扩展到文本、音频等其他有嵌入距离度量的领域
- Emitter 概念可与自动 prompt 工程结合，用 LLM 自生成 emitter
- 可用于数据增强——生成多样训练图像提升下游视觉任务泛化性
- 新颖性搜索方法可应用于 jailbreak 检测或红队测试

## 评分

- 新颖性: ⭐⭐⭐⭐ (将新颖性搜索引入图像生成是新颖的应用方向)
- 实验充分度: ⭐⭐⭐⭐ (多基线对比、消融实验、LLM 对比充分)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，问题定义明确)
- 价值: ⭐⭐⭐ (Workshop 论文，实际影响力有限但方向有趣)

<!-- RELATED:START -->

## 相关论文

- [Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](../../CVPR2025/image_generation/learning_to_sample_effective_and_diverse_prompts_for_text-to-image_generation.md)
- [Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals](continuous_uniqueness_and_novelty_metrics_for_generative_modeling_of_inorganic_c.md)
- [LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](../../ICCV2025/image_generation/loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [Watermarking Autoregressive Image Generation](watermarking_autoregressive_image_generation.md)
- [Exposing Hidden Biases in Text-to-Image Models via Automated Prompt Search](../../ICLR2026/image_generation/exposing_hidden_biases_in_text-to-image_models_via_automated_prompt_search.md)

<!-- RELATED:END -->
