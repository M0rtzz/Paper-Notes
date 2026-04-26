---
title: >-
  [论文解读] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation
description: >-
  [ACL 2026][图像恢复][检索增强生成] 提出 EviOmni，通过"先推理再提取"的范式学习从检索文档中提取理性证据：将证据推理和证据提取整合为统一轨迹，用知识 token 掩码避免信息泄露，通过 GRPO 以可验证奖励优化，在 5 个基准上以极高压缩比（~38x）取得优于全文检索的准确率。
tags:
  - ACL 2026
  - 图像恢复
  - 检索增强生成
  - 证据提取
  - 强化学习
  - 推理引导提取
  - GRPO
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation

**会议**: ACL 2026  
**arXiv**: [2507.15586](https://arxiv.org/abs/2507.15586)  
**代码**: [GitHub](https://github.com/HITsz-TMG/EviOmni)  
**领域**: 检索增强生成  
**关键词**: 检索增强生成, 证据提取, 强化学习, 推理引导提取, GRPO

## 一句话总结

提出 EviOmni，通过"先推理再提取"的范式学习从检索文档中提取理性证据：将证据推理和证据提取整合为统一轨迹，用知识 token 掩码避免信息泄露，通过 GRPO 以可验证奖励优化，在 5 个基准上以极高压缩比（~38x）取得优于全文检索的准确率。

## 研究背景与动机

**领域现状**：RAG 通过检索外部段落增强 LLM 的准确性。然而检索到的段落常含大量噪声和无关内容，需要证据提取/降噪。现有方法包括重排序（将相关段落置顶）和摘要/提取（用 SFT 训练过滤模型）。

**现有痛点**：现有方法直接提取证据而不进行深度思考，可能遗漏关键线索。例如，直接提取可能因上下文理解不足而丢掉分散在多个段落中的关键信息。训练数据通常通过启发式构建（如字符串包含、词汇重叠），不直接对齐 RAG 的最终目标。

**核心矛盾**：普通证据提取是"看到什么提什么"，缺乏对检索内容的深度推理——当关键线索需要跨段落推理才能识别时，直接提取容易遗漏。

**本文目标**：让证据提取器先推理（识别检索内容中的线索及其相关性），再基于推理结果提取，并通过 RL 优化使提取结果直接对齐下游任务准确率。

**切入角度**：实证研究发现，在 SFT 数据中加入推理步骤（reason→extract）后，证据的答案召回率从 70.8% 提升至 75.2%（NQ 数据集），证明推理引导的价值。

**核心 idea**：将证据推理 <reason> 和证据提取 <extract> 统一到一个生成轨迹中，用知识 token 掩码分离评估，用 GRPO + 三类可验证奖励（答案、长度、格式）端到端优化。

## 方法详解

### 整体框架

输入查询 q 和 top-k 检索段落 P，EviOmni 生成包含三部分的响应：<reason>推理</reason><extract>证据</extract><answer>答案</answer>。训练时通过知识 token 掩码分别评估推理和证据的质量，用三类奖励驱动 GRPO 优化。

### 关键设计

1. **理性证据提取范式**:

    - 功能：通过推理引导证据提取，减少关键线索遗漏
    - 核心思路：模型先生成 <reason> 部分（分析每个段落的相关性和包含的线索），再基于推理生成 <extract> 部分（简洁的证据摘要）。公式化为 $e \sim \mathcal{M}_\mathcal{E}(\cdot|q,P,r) \cdot \mathcal{M}_\mathcal{E}(r|q,P)$
    - 设计动机：推理过程可以识别分散的线索、排除误导信息、关联跨段落信息，比直接提取更可靠

2. **知识 Token 掩码**:

    - 功能：在训练中分离评估推理和证据的质量
    - 核心思路：(1) 掩码证据 e → 仅基于推理 r 生成答案 $o_r$，评估推理质量；(2) 掩码段落 P 和推理 r → 仅基于证据 e 生成答案 $o_e$，评估证据质量。使用硬掩码（替换输入 token）而非软掩码（调整注意力），避免因 causal attention 已聚合的信息导致泄露
    - 设计动机：如果不分离，证据生成后答案可以从完整上下文（包括原始段落）中获取信息，无法反映证据本身的质量

3. **三类可验证奖励**:

    - 功能：引导模型优化三个期望属性
    - 核心思路：答案奖励 $R^{ans}$ = unigram F1（统一跨任务的评估）；长度奖励 $R^{len}$ 鼓励推理全面（长于证据）且证据简洁（远短于段落）；格式奖励 $R^{fmt}$ 确保标签格式正确。最终奖励 $R^{final} = \lambda_1 R^{ans} + \lambda_2 R^{len} + \lambda_3 R^{fmt}$
    - 设计动机：直接用下游答案准确率作为奖励，避免启发式指标与最终目标的错位

### 损失函数 / 训练策略

GRPO 在线策略优化，使用 Qwen2.5-1.5B/7B-Instruct 作为基础模型，同一模型同时作为提取器和生成器。

## 实验关键数据

### 主实验

1.5B 模型在 NQ/TQA/HotpotQA 上（EM/F1/压缩比）：

| 方法 | NQ EM | NQ CR | TQA EM | HotpotQA EM |
|------|-------|-------|--------|-------------|
| Full（无压缩） | 41.97 | 1.0x | 57.02 | 19.20 |
| FilCo | 36.62 | 16.3x | 54.06 | 18.18 |
| SEER | 36.93 | 13.2x | 54.57 | 18.60 |
| **EviOmni** | **41.14** | **38.1x** | **56.84** | **20.46** |

EviOmni 在 ~38x 压缩下接近甚至超过全文性能。

### 消融实验

| 配置 | NQ AR | HotpotQA AR |
|------|-------|------------|
| Vanilla Evidence (无推理) | 70.79% | 60.55% |
| Rational Evidence (有推理) | **75.24%** | **67.74%** |
| Rationale 本身 | 77.30% | 71.48% |

### 关键发现

- 理性证据的答案召回率比普通证据高 4-7 个百分点，证实推理引导的价值
- 在 38x 压缩比下性能接近全文输入，说明提取的证据高度精炼
- 在 OOD 数据集（HotpotQA）上也有提升，表明泛化性好
- 同时支持传统 RAG 和 Agentic RAG（提前终止、噪声鲁棒等特性）

## 亮点与洞察

- **"先推理再提取"的范式转换**具有广泛意义——不仅适用于 RAG，任何需要从噪声信息中提取关键内容的任务都可以借鉴
- **知识 Token 掩码**解决了训练中信息泄露的技术难题，设计精巧
- **38x 压缩比下性能不降**的结果令人印象深刻，对推理效率有重大实践意义

## 局限与展望

- 推理过程增加了生成长度，虽然证据更短但总输出更长
- 仅在 QA 任务上训练和评估，对话/摘要等任务的适用性需验证
- 推理质量受基础模型能力限制，1.5B 模型的推理深度有限

## 相关工作与启发

- **vs Recomp/FilCo/SEER**: 这些方法直接提取/摘要，无推理引导，压缩比和准确率均不如 EviOmni
- **vs SFT 方法（Wang et al., 2023）**: SFT 依赖启发式构建训练数据，RL 直接对齐下游任务目标

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "推理→提取"范式 + 知识掩码 + RL优化的组合非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个基准、两种模型规模、传统+Agentic RAG 均验证
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，实证研究有说服力
- 价值: ⭐⭐⭐⭐⭐ 对 RAG 管线效率和质量有直接实践意义

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](../../NeurIPS2025/image_restoration/real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)
- [\[CVPR 2026\] PNG: Diffusion-Based sRGB Real Noise Generation via Prompt-Driven Noise Representation Learning](../../CVPR2026/image_restoration/diffusion-based_srgb_real_noise_generation_via_prompt-driven_noise_representatio.md)
- [\[CVPR 2026\] Learning to Translate Noise for Robust Image Denoising](../../CVPR2026/image_restoration/learning_to_translate_noise_for_robust_image_denoising.md)
- [\[ICLR 2026\] Mechanism of Task-oriented Information Removal in In-context Learning](../../ICLR2026/image_restoration/mechanism_of_task-oriented_information_removal_in_in-context_learning.md)
- [\[ICLR 2026\] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](../../ICLR2026/image_restoration/protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)

<!-- RELATED:END -->
