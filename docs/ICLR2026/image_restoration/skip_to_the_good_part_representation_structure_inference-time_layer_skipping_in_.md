---
title: >-
  [论文解读] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs
description: >-
  [ICLR 2026][图像恢复][扩散语言模型] 首次系统比较扩散语言模型（dLLM）和自回归模型（AR LLM）的层间表征结构，发现原生 dLLM 具有更强的层级抽象和早期层冗余性，据此提出静态、任务无关的推理时层跳过策略，在 LLaDA 上跳过 6 层（18.75% FLOPs 削减）仍保持 90%+ 性能。
tags:
  - ICLR 2026
  - 图像恢复
  - 扩散语言模型
  - 层跳过
  - 表征冗余
  - 推理加速
  - LLaDA
---

# Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs

**会议**: ICLR 2026  
**arXiv**: [2603.07475](https://arxiv.org/abs/2603.07475)  
**代码**: 无  
**领域**: 模型压缩 / LLM效率  
**关键词**: 扩散语言模型, 层跳过, 表征冗余, 推理加速, LLaDA

## 一句话总结
首次系统比较扩散语言模型（dLLM）和自回归模型（AR LLM）的层间表征结构，发现原生 dLLM 具有更强的层级抽象和早期层冗余性，据此提出静态、任务无关的推理时层跳过策略，在 LLaDA 上跳过 6 层（18.75% FLOPs 削减）仍保持 90%+ 性能。

## 研究背景与动机

**领域现状**：自回归（AR）语言模型通过从左到右的 next-token prediction 逐步构建表征，而扩散语言模型（dLLM）如 LLaDA、Dream 等通过全序列去噪训练。近期 dLLM 在推理和代码生成上已接近 AR 模型性能。

**现有痛点**：尽管 dLLM 与 AR 模型在性能上趋近，但两者的内部表征结构差异尚未被系统研究。现有的效率优化工作（如 YOCO）聚焦于 KV-cache 共享等架构级改造，需要修改模型结构。

**核心矛盾**：训练目标（扩散 vs 自回归）是否从根本上重塑了模型的内部表征？如果存在系统性差异，能否直接利用这种差异来加速推理？

**本文目标**：(a) 量化 dLLM 和 AR LLM 在层间和 token 间表征相似性上的差异；(b) 探究 AR 初始化对 dLLM 表征的持久影响；(c) 利用发现的表征冗余实现无需架构修改的推理加速。

**切入角度**：通过逐层 cosine 相似度分析，发现原生 dLLM 早期层表征高度相似（>0.95），意味着这些层的计算是冗余的，可以安全跳过。

**核心 idea**：扩散训练目标诱导出层级化的"粗到细"抽象，早期层建立粗糙表征（高冗余），后期层进行精细化。这种冗余可以直接用于静态层跳过，无需 KV-cache 共享或架构修改。

## 方法详解

### 整体框架

比较三个模型家族：LLaDA-8B（原生 dLLM）、Qwen2.5-7B（原生 AR）、Dream-7B（AR 初始化的 dLLM）。先进行层间/token 间表征相似度分析，然后基于分析结果设计层跳过策略，在推理时直接跳过高相似度层。

### 关键设计

1. **层间 Cosine 相似度分析**:

    - **功能**：追踪连续层表征 $\mathbf{h}_\ell$ 和 $\mathbf{h}_{\ell+1}$ 之间的 cosine 相似度
    - **核心发现**：LLaDA 早期层（前 60-70%）相似度 >0.95（高冗余平台），后期层相似度下降（精细化区域）。Qwen2.5 全深度相似度较低且均匀分布（无明显冗余）。Dream-7B 尽管经过扩散训练，但表征模式与 Qwen2.5 高度一致——强初始化偏置
    - **设计动机**：量化训练目标对表征结构的影响，为层跳过提供理论依据

2. **Token 间相似度与 Recency Bias 分析**:

    - **功能**：分析不同层中 token 间的表征变化模式
    - **核心发现**：LLaDA 展现最小的 recency bias（全局平滑高相似度），Qwen2.5 和 Dream-7B 都有显著 recency bias（每个新 token 都引起跨层的表征变化）
    - **设计动机**：验证 dLLM 的全序列反馈训练确实导致了更全局化的表征抽象

3. **静态层跳过策略（Algorithm 1）**:

    - **功能**：在推理时跳过高相似度的冗余层
    - **核心思路**：基于训练时相似度分析，选择 cosine 相似度 > θ（默认 0.95）的层。按相似度降序排列，选择 top-k 层跳过，但不允许跳过相邻层（保持表征连续性）。跳过时直接传递 $\mathbf{h}_{\ell-1}$ 到 $\ell+1$ 层
    - **设计动机**：纯推理时操作，无需重训练、无需 KV-cache 共享、无需架构修改，与 KV-cache 优化正交可叠加

### 损失函数 / 训练策略

本文不修改训练过程，仅在推理时应用层跳过。分析基于已训练好的 LLaDA-8B、Qwen2.5-7B、Dream-7B 模型。

## 实验关键数据

### 主实验

| 模型 | 跳过层数 | FLOPs 削减 | GSM8K 保持率 | HumanEval 保持率 | MATH500 保持率 |
|------|---------|-----------|------------|----------------|--------------|
| LLaDA-8B | 0 | 0% | 100% (0.83) | 100% (0.51) | 100% (0.37) |
| LLaDA-8B | 2 | 6.25% | 101.3% | 100% | 108.5% |
| LLaDA-8B | 4 | 12.5% | 102.5% | 92.2% | 89.4% |
| LLaDA-8B | 6 | 18.75% | 91.8% | 88.2% | 102.1% |
| LLaDA-8B | 8 | 25% | 91.8% | 62.7% | 70.2% |
| Qwen2.5-7B | 2 | 7.14% | 34.9% | 64.7% | - |
| Dream-7B | 2 | 7.14% | 76.8% | 66.2% | 81.4% |

### 消融实验

| 配置 | GSM8K 保持率 | HumanEval 保持率 | 说明 |
|------|------------|----------------|------|
| LLaDA 6层 非连续跳过 | 91.8% | 88.2% | 完整方法 |
| LLaDA 6层 允许连续跳过 | 75.3% | 64.7% | 连续跳过导致严重退化 |
| Dream-7B 2层 非连续跳过 | 76.8% | 66.2% | AR初始化限制了可跳过性 |
| Dream-7B 2层 连续跳过 | 14.1% | - | 崩溃 |

### 关键发现
- **原生 dLLM 支持激进跳过**：LLaDA 跳 6 层仍保持 88-102% 性能，而 Qwen2.5 跳 2 层就崩溃（34.9%）
- **AR 初始化偏置持久存在**：Dream-7B 经过扩散训练后，表征模式仍与 Qwen2.5 一致，跳层鲁棒性也类似 AR 模型
- **不允许连续跳过是关键**：连续跳过导致表征连续性断裂，性能严重下降
- **跳过的层集中在网络前 40-60%**：与早期层建立粗糙表征的分析一致
- dLLM 可实现 2.6× 更大 FLOPs 削减 + 1.4× 更高质量保持

## 亮点与洞察
- **训练目标决定表征结构**：这是一个重要的发现——不是架构而是训练目标决定了表征的冗余模式。这意味着可以通过改变训练目标来"设计"所需的表征属性
- **初始化偏置的深度**：Dream-7B 的案例表明，扩散微调不足以覆盖 AR 预训练的表征结构，initialization matters more than fine-tuning objective
- **与 KV-cache 正交**：层跳过减少深度计算，KV-cache 减少 token 冗余计算，两者可乘性组合

## 局限与展望
- 仅测试了 7-8B 规模的模型，更大规模模型的冗余模式可能不同
- 跳过策略是静态的，动态/自适应跳过可能更优
- 未探索 dLLM 专门为层跳过优化训练的可能性（如引入层间正则化）
- benchmarks 有限（GSM8K, MATH500, HumanEval, MBPP）

## 相关工作与启发
- **vs YOCO**：YOCO 需要 cache-once 架构设计，本文方法不需要任何架构修改
- **vs Early Exit**：Early Exit 在 prefill 阶段退出，本文在全序列去噪的每个步骤都跳层
- **vs DiffuCoder (Gong et al., 2025)**：DiffuCoder 从行为层面分析 dLLM 的 AR-ness，本文从表征层面提供了互补的视角

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统分析 dLLM vs AR 表征差异，发现有价值但技术贡献偏分析性
- 实验充分度: ⭐⭐⭐⭐ 三个模型家族+四个 benchmark+消融，但规模有限
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，可视化出色，故事线流畅
- 价值: ⭐⭐⭐⭐ 对理解 dLLM 内部机制有重要贡献，层跳过方法实用但简单

<!-- RELATED:START -->

## 相关论文

- [AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)
- [Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes](beyond_scattered_acceptance_fast_and_coherent_inference_for_dlms_via_longest_sta.md)
- [TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation](../../ICML2025/image_restoration/timedart_a_diffusion_autoregressive_transformer_for_self-supervised_time_series_.md)
- [Activation Steering for Masked Diffusion Language Models](activation_steering_for_masked_diffusion_language_models.md)
- [wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)

<!-- RELATED:END -->
