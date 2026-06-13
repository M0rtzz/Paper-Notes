---
title: >-
  [论文解读] Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points
description: >-
  [ACL 2025][LLM对齐][DPO] 发现代码生成模型的错误高度集中在特定"错误易发点"（error-prone points），前缀/后缀几乎不变而中间段决定正确性，提出 Focused-DPO：通过 PageRank 在代码-测试二部图上排序定位关键中间段，并在 DPO 损失中对该段加权放大（$w_{focused}=2$），仅用 5000 样本即可在 HumanEval+ 上提升 4.41%、LiveCodeBench-Hard 上相对提升 42.86%。
tags:
  - "ACL 2025"
  - "LLM对齐"
  - "DPO"
  - "代码生成"
  - "错误易发点"
  - "PageRank"
  - "偏好优化加权"
  - "前缀后缀匹配"
---

# Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points

**会议**: ACL 2025  
**arXiv**: [2502.11475](https://arxiv.org/abs/2502.11475)  
**代码**: 无  
**领域**: LLM对齐 / 代码生成  
**关键词**: DPO, 代码生成, 错误易发点, PageRank, 偏好优化加权, 前缀后缀匹配

## 一句话总结

发现代码生成模型的错误高度集中在特定"错误易发点"（error-prone points），前缀/后缀几乎不变而中间段决定正确性，提出 Focused-DPO：通过 PageRank 在代码-测试二部图上排序定位关键中间段，并在 DPO 损失中对该段加权放大（$w_{focused}=2$），仅用 5000 样本即可在 HumanEval+ 上提升 4.41%、LiveCodeBench-Hard 上相对提升 42.86%。

## 研究背景与动机

**领域现状**: DPO、RLHF 等偏好优化方法已广泛用于代码生成模型的后训练对齐。Qwen2.5-Coder、DeepSeekCoder 等模型通过百万级数据的 SFT + DPO 训练，在 HumanEval 等基准上取得了 >90% 的通过率。

**现有痛点**: 标准 DPO 将所有 token 等权对待，但代码中不同部分对正确性的贡献极不均匀——函数签名、import 语句、return 语句等几乎总是正确的，真正的错误集中在算法逻辑的核心区域（中间代码段）。大量梯度信号浪费在已经正确的 token 上，无法有效提升关键位置的生成质量。

**关键观察**: 作者对 Qwen2.5-Coder-7B 采样 20 次发现：代码的公共前缀/后缀与正确性的 Phi 相关系数仅 0.07–0.08（几乎不相关），而错误易发中间段的 Phi 系数达 0.57–0.61（强相关）。从正确中间段继续生成的 pass@1 为 90.02%，从错误中间段继续则仅 3.17%。

**核心矛盾**: 错误易发点对最终正确性至关重要，但标准 DPO 和 Step-DPO 等方法无法区分代码不同位置的重要性差异，导致优化效率低下。

**解决思路**: 自动定位代码中的错误易发点，在偏好优化损失中对该区域加权放大，使模型集中学习最关键的代码决策。

## 方法详解

### 整体框架

三阶段流水线：(1) 从真实开源代码仓库提取编程概念并合成问题 prompt；(2) 策略模型同时生成代码和测试用例，PageRank 排序 + 公共前缀/后缀匹配定位错误易发点，构建细粒度偏好数据集；(3) 使用修改后的加权 DPO 损失进行聚焦偏好优化训练。

### 关键设计

1. **Error-Point Identification（错误易发点数据集构建）**: 对每个问题 prompt，用策略模型以 temperature=1.5 采样 $k=10$ 个代码候选和测试用例。构建代码-测试二部图，代码 $c_i$ 通过测试 $t_j$ 则连边。用 PageRank 迭代更新代码和测试的分数直至排名稳定，最高分代码通过的测试作为 ground truth。将代码分为 correct/incorrect 两类后，对每对 (chosen, rejected) 匹配公共前缀和后缀，提取中间差异段 (mid_chosen, mid_rej) 作为错误易发点。通过最大化 Diff 函数（排名差 + $\lambda$ × 公共部分长度）选择最具区分力的配对，最终过滤得到 5000 训练 + 1000 验证样本。

2. **Focused-DPO 加权损失设计**: 将代码分为 prefix / mid (focus) / suffix 三段。对 chosen 样本的奖励函数中，mid 段乘以权重 $w_{focused}=2$，prefix 和 suffix 权重为 1；对 rejected 样本，直接去掉 suffix 部分的贡献（因实证发现 suffix 与正确性几乎不相关）。最终损失函数简化为 $\mathcal{L} = -\mathbb{E}[\log\sigma(\Delta_{mid} + \Delta_{suffix})]$，其中 $\Delta_{mid}$ 被 $w_{focused}$ 放大，梯度信号集中到错误易发点。

3. **自生成自验证的 PageRank 排序机制**: 区别于 Magicoder 等方法直接用所有生成的测试用例作为 ground truth，本文用 PageRank 迭代过滤低质量测试用例（低分测试用例被自动降权）。这使得即使策略模型本身生成质量不如 GPT-4，也能产出高质量的偏好数据集。实验发现不同模型在相同问题上的错误易发点有 32% 的重叠，说明错误易发点具有跨模型的通用性。

## 实验关键数据

### 主实验：HumanEval(+) / MBPP(+)

| 模型 | 方法 | HumanEval | HumanEval+ | MBPP | MBPP+ |
|------|------|-----------|------------|------|-------|
| Qwen2.5-Coder-Instruct-7B | Baseline | 91.5% | 84.1% | 82.8% | 71.4% |
| | +Focused-DPO | 92.7% | **87.8%** | 84.7% | **76.2%** |
| | +DPO/Step-DPO | 92.1% | 85.4% | 84.1% | 74.3% |
| | +Token-DPO | 92.7% | 87.2% | 83.3% | 75.1% |
| MagiCoder-S-DS-6.7B | Baseline | 73.2% | 68.3% | 76.7% | 66.7% |
| | +Focused-DPO | **82.3%** | **74.4%** | **79.4%** | **69.8%** |
| DeepSeekCoder-Instruct-6.7B | Baseline | 77.4% | 70.1% | 75.1% | 65.9% |
| | +Focused-DPO | **82.3%** | **73.2%** | 76.5% | 66.9% |

### LiveCodeBench 分难度结果

| 模型 | 方法 | Easy | Medium | Hard | Avg |
|------|------|------|--------|------|-----|
| Qwen2.5-Coder-Instruct-7B | Baseline | 69.2% | 22.0% | 3.4% | 31.2% |
| | +Focused-DPO | 73.5% | 24.2% | **4.8%** | 33.9% |
| | 相对提升 | +6.2% | +10.0% | **+42.9%** | +8.4% |
| MagiCoder-S-DS-6.7B | Baseline | 48.1% | 10.7% | 0.1% | 19.3% |
| | +Focused-DPO | 51.3% | 11.8% | **1.9%** | 21.3% |
| | 相对提升 | +6.6% | +10.1% | **+1752%** | +10.1% |

### 错误易发点与正确性的相关性分析

| 代码段 | 在正确代码中频率 | 在错误代码中频率 | Phi 系数 |
|--------|----------------|----------------|----------|
| 公共前缀 | 0.791 | 0.733 | 0.068 |
| 公共后缀 | 0.848 | 0.786 | 0.080 |
| 前缀 + Chosen Mid | 0.637 | 0.091 | **0.565** |
| 前缀 + Reject Mid | 0.012 | 0.558 | **-0.609** |

### 多阶段训练叠加效果

| 训练阶段 | HumanEval | HumanEval+ | MBPP | MBPP+ |
|----------|-----------|------------|------|-------|
| DeepSeekCoder-base-6.7B | 47.6% | 39.6% | 70.2% | 56.6% |
| + SFT (MagiCoder) | 73.2% | 68.3% | 76.7% | 66.7% |
| + First DPO (CodeDPO) | 83.5% | 76.2% | 80.7% | 70.9% |
| + Focused-DPO | **87.2%** | **79.3%** | **82.3%** | **72.8%** |

### 关键发现

- **更严格的 benchmark 提升更大**：HumanEval+ (+4.41%) > HumanEval (+1.29%)，MBPP+ (+6.71%) > MBPP (+2.24%)，说明方法真正改善了代码正确性而非表面匹配
- **Hard 题目提升最显著**：LiveCodeBench-Hard 相对提升 42.86%（Qwen2.5）到 1752%（MagiCoder），越难的题目错误易发点效应越强
- **对已经过大规模 SFT+DPO 的强模型仍有效**：Qwen2.5-Coder 经百万级数据对齐后，用 5000 样本 Focused-DPO 仍可进一步提升
- **多阶段叠加有效**：在 base→SFT→DPO 后再加 Focused-DPO，每个阶段都有正向增益，无收益递减
- **消融实验**：$w_{focused}=2$ 最优，过大（5）或过小（1）均导致性能下降；去掉 rejected 中的 suffix 项有正向效果

## 亮点与洞察

- **"错误集中在中间"的实证发现**简单但有力：90% vs 3% 的 pass@1 差异量化了错误易发点的决定性作用，为代码对齐研究提供了全新视角
- **PageRank 自验证机制**巧妙解决了无人工标注下的数据质量问题，使得策略模型可以自举（self-bootstrap）生成高质量偏好数据
- **数据效率极高**：仅 5000 个样本即可在已经过百万级对齐训练的 Qwen2.5 上取得显著提升，实用价值高

## 局限与展望

- 仅验证了 Python 代码生成，其他编程语言（C++/Java/Rust 等）的错误分布模式可能不同
- $w_{focused}=2$ 是手动设定的固定值，可探索自适应权重学习或按问题难度动态调整
- 前缀/后缀匹配方式较为简单，git-diff 等更精细的差异定位方法虽当前效果略低但有改进空间
- 需要多次采样（$k=10$）和执行测试用例，数据构建的计算开销较大

## 评分

- **新颖性**: ⭐⭐⭐⭐ 错误集中现象的量化观察 + PageRank 自验证 + 位置加权 DPO 的组合新颖且合理
- **实验充分度**: ⭐⭐⭐⭐ 5 个模型 × 5 个 benchmark + 相关性分析 + 数据集/损失函数双消融 + 多阶段训练实验
- **写作质量**: ⭐⭐⭐⭐ 三个 RQ 层层递进，从动机验证到主实验到消融，逻辑清晰
- **实用价值**: ⭐⭐⭐⭐ 5000 样本即可提升已对齐强模型，方法简单可复现，对代码对齐有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)
- [\[ACL 2025\] RISE: Subtle Errors in Reasoning: Preference Learning via Error-injected Self-editing](rise_error_inject_preference.md)
- [\[ACL 2025\] FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings](focalpo_enhancing_preference_optimizing_by_focusing_on_correct_preference_rankin.md)
- [\[ACL 2025\] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)
- [\[ICML 2025\] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](../../ICML2025/llm_alignment/tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)

</div>

<!-- RELATED:END -->
