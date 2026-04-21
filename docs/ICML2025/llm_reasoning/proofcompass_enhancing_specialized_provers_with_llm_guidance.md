---
title: >-
  [论文解读] ProofCompass: Enhancing Specialized Provers with LLM Guidance
description: >-
  [ICML 2025 (AI for MATH Workshop)][LLM推理][Formal Theorem Proving] ProofCompass 提出一种无需额外训练的混合方法，用通用 LLM 为专业定理证明器（如 DeepSeek-Prover-v1.5-RL）提供自然语言证明策略和中间引理选择，在 miniF2F 上用 25 倍少的尝试次数超越了基线性能（54.9% → 55.3%）。
tags:
  - ICML 2025 (AI for MATH Workshop)
  - LLM推理
  - Formal Theorem Proving
  - Hybrid Methodology
  - DeepSeek-Prover
  - LLM Guidance
  - Lemma Decomposition
  - miniF2F
---

# ProofCompass: Enhancing Specialized Provers with LLM Guidance

**会议**: ICML 2025 (AI for MATH Workshop)  
**arXiv**: [2507.14335](https://arxiv.org/abs/2507.14335)  
**代码**: —  
**领域**: LLM推理 / 形式化定理证明 / 数学推理  
**关键词**: Formal Theorem Proving, Hybrid Methodology, DeepSeek-Prover, LLM Guidance, Lemma Decomposition, miniF2F  

## 一句话总结

ProofCompass 提出一种无需额外训练的混合方法，用通用 LLM 为专业定理证明器（如 DeepSeek-Prover-v1.5-RL）提供自然语言证明策略和中间引理选择，在 miniF2F 上用 25 倍少的尝试次数超越了基线性能（54.9% → 55.3%）。

## 研究背景与动机

形式化定理证明领域存在两种主流方法，各有局限：

| 方法类型 | 代表 | 优势 | 劣势 |
|----------|------|------|------|
| 大型通用 LLM | GPT-4, Claude | 高层推理能力强 | 形式化语法不精确，尝试次数多 |
| 小型专业证明器 | DSP-v1.5-RL | 形式化语法精确 | 缺乏高层策略规划 |

训练一个兼具两者优势的大型专业模型需要巨大计算资源。**核心想法**：能否通过工程化的方式组合两者——让通用 LLM 提供"宏观策略"，专业证明器执行"微观战术"？

## 方法详解

### 整体架构

ProofCompass 的核心是一个**两层循环系统**：

- **外层循环（LLM 策略师）**：通用 LLM 分析问题，提供自然语言证明策略和分解引理
- **内层循环（专业执行者）**：DSP-v1.5-RL 根据策略执行 Lean 代码生成和证明搜索

### 1. 策略生成阶段

LLM 接收自然语言数学问题描述，生成证明策略：

$$\text{Strategy} = \text{LLM}(\text{Problem Description}, \text{History of Failed Attempts})$$

策略包括：
- **证明方法建议**：如"尝试数学归纳法"、"用反证法"
- **关键中间步骤**：将复杂问题分解为若干子问题
- **失败分析**：基于之前的失败尝试，指出哪些方向不可行

### 2. 引理分解

LLM 将复杂定理分解为多个中间引理（lemma）：

$$\text{Theorem} \xrightarrow{\text{LLM}} \{\text{Lemma}_1, \text{Lemma}_2, \ldots, \text{Lemma}_m\}$$

每个引理比原定理更简单，专业证明器更容易找到证明。LLM 同时生成引理的自然语言描述和初步的 Lean 形式化。

### 3. 失败反馈循环

当专业证明器未能证明某个引理时，将失败信息（包括尝试的战术和错误信息）返回给 LLM：

$$\text{New Strategy} = \text{LLM}(\text{Problem}, \text{Previous Strategy}, \text{Failed Tactics}, \text{Error Messages})$$

LLM 据此调整策略：可能修改引理分解、建议不同的证明方法、或提供更具体的战术提示。

### 4. 资源分配策略

ProofCompass 智能分配有限的尝试次数：

$$\text{Budget} = 128 \text{ attempts total}$$

- 对 LLM 认为"容易"的子问题分配较少尝试
- 对"困难"的子问题分配更多尝试
- 通过 LLM 的置信度评估动态调整分配

## 实验

### 主实验：miniF2F 基准

| 方法 | 通过率 | 尝试次数 | 效率（通过率/尝试） |
|------|--------|---------|-------------------|
| DSP-v1.5-RL | 54.9% | 3,200 | 0.017%/次 |
| ProofCompass | **55.3%** | **128** | **0.432%/次** |
| GPT-4 (direct) | ~35% | 128 | 0.273%/次 |
| LLaMA-3-70B (direct) | ~28% | 128 | 0.219%/次 |

ProofCompass 用 25 倍少的尝试就超越了 DSP-v1.5-RL 的性能。

### 消融实验

| 组件 | 移除后通过率 | 影响 |
|------|------------|------|
| 完整 ProofCompass | 55.3% | 基线 |
| 去除策略生成 | 50.1% | -5.2% |
| 去除引理分解 | 51.7% | -3.6% |
| 去除失败反馈 | 52.8% | -2.5% |
| 去除资源分配 | 53.4% | -1.9% |

策略生成和引理分解是最关键的两个组件。

### 资源效率分析

| 尝试次数 | DSP-v1.5-RL | ProofCompass |
|----------|-------------|-------------|
| 32 | 38.2% | 49.1% |
| 64 | 45.6% | 53.2% |
| 128 | 49.8% | 55.3% |
| 256 | 52.1% | 56.1% |
| 3,200 | 54.9% | — |

ProofCompass 在各个计算预算下均显著优于 DSP-v1.5-RL。

## 亮点与洞察

- **极致的效率提升**：25x 少的计算换来同等或更好的性能，实用价值极高
- **无需训练**：完全利用现有模型的能力组合，不需要额外训练任何模型
- **角色分工的巧妙设计**：LLM 负责"想"（策略和分解），证明器负责"做"（形式化证明）
- 揭示了当前专业证明器的瓶颈并非形式化能力不足，而是缺乏高层策略规划

## 局限性

- 仅在 miniF2F 基准上测试，未扩展到更大规模的形式化基准（如 FMC、ProofNet）
- 绝对通过率提升较小（0.4%），统计显著性需更多实验验证
- LLM 的 API 调用引入额外成本（虽然 GPU 计算成本大幅降低）
- 方法的有效性可能依赖于 LLM 对数学的理解深度，较弱的 LLM 可能效果下降
- 引理分解的质量难以保证，错误的分解可能浪费计算预算

## 相关工作与启发

- **DeepSeek-Prover-v1.5 (Xin et al., 2024)**：基于 RL 训练的专业 Lean 证明器
- **Subgoal-based Provers (Zhao et al., 2024)**：将证明目标分解为子目标
- **LLM for Math (Trinh et al., 2024)**：AlphaGeometry 等 LLM 辅助的数学推理
- 本文的创新点在于"组合已有能力"的范式——不训练新模型，而是通过工程化组合最大化已有模型的互补优势

## 评分

⭐⭐⭐⭐ — 工程化思路简洁高效，25x 计算效率提升具有很强的实用价值，为形式化证明的混合范式开辟方向

<!-- RELATED:START -->

## 相关论文

- [Local Look-Ahead Guidance via Verifier-in-the-Loop for Automated Theorem Proving](../../ACL2025/llm_reasoning/local_look-ahead_guidance_via_verifier-in-the-loop_for_automated_theorem_proving.md)
- [Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](../../AAAI2026/llm_reasoning/jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)
- [MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](../../ACL2025/llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [Enhancing Mathematical Reasoning in LLMs by Stepwise Correction](../../ACL2025/llm_reasoning/enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)
- [AdaDecode: Accelerating LLM Decoding with Adaptive Layer Parallelism](adadecode_accelerating_llm_decoding_with_adaptive_layer_parallelism.md)

<!-- RELATED:END -->
