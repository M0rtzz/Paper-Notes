---
title: >-
  [论文解读] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training
description: >-
  [可解释性] 提出 ZeroTuning，仅需对初始 token（如 `<BOS>`）的注意力分数进行头部特异性缩放，即可在无训练情况下提升 LLM 在 15 个数据集上的表现，仅需修改 4 行代码。
tags:
  - "可解释性"
---

# ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training

## 论文信息
- **会议**: ICLR 2026
- **arXiv**: [2505.11739](https://arxiv.org/abs/2505.11739)
- **代码**: [https://anonymous.4open.science/r/ZeroTuning](https://anonymous.4open.science/r/ZeroTuning)
- **领域**: 可解释性
- **关键词**: 注意力调优, 初始 token, attention sink, 零训练增强, 头部特异性

## 一句话总结
提出 ZeroTuning，仅需对初始 token（如 `<BOS>`）的注意力分数进行头部特异性缩放，即可在无训练情况下提升 LLM 在 15 个数据集上的表现，仅需修改 4 行代码。

## 研究背景与动机

### 核心问题
Token 级注意力调优（如 PASTA、ACT）虽有效，但依赖**外部启发式**识别任务特定的"重要" token，带来偏差和适用性限制。能否找到一个**通用、任务无关的控制点**？

### Attention Sink 现象
初始 token 倾向于成为 attention sink（注意力汇聚点），但其增强性能的潜力尚未被开发。

### 关键发现
- 调整初始 token 的注意力一致产生最大且最稳定的增益
- 增益方向取决于任务：分类任务需上调（$\gamma > 1$），QA 任务需下调（$\gamma < 1$）
- 不同注意力头对初始 token 缩放的响应异质

## 方法详解

### 注意力缩放形式化

引入缩放因子 $\gamma > 0$ 调整初始 token 的注意力权重：

$$a_0' = \frac{\gamma a_0}{D}, \quad a_i' = \frac{a_i}{D}, \quad D = (\gamma-1)a_0 + 1$$

**关键性质**：
- 保持非初始 token 间的相对比例不变
- $\gamma > 1$：放大初始 token，平坦化其余分布
- $\gamma < 1$：压缩初始 token，锐化其余分布

### 调控效应分析

注意力差异变化量：
$$E_{\text{diff},i,j} = |a_i - a_j| \cdot \frac{|\gamma-1| a_0}{(\gamma-1) a_0 + 1}$$

对 $a_0$ 求导：
$$\frac{\partial E_{\text{diff},i,j}}{\partial a_0} = |a_i - a_j| |\gamma-1| \cdot \frac{1}{((\gamma-1)a_0+1)^2} \geq 0$$

**核心洞察**：初始 token 的注意力越大（attention sink），其作为杠杆的调控效力越强。

### 层级分析

浅层（1-10）和中间层（11-21）的调整通常比深层（22-31）更有效，因为：
- 早期/中间层主要支持表示学习和知识整合
- 深层聚焦于任务特定推理

### 头部特异性

不同头对初始 token 缩放的响应异质：
- **Up-effective 头**：放大注意力提升性能
- **Down-effective 头**：缩小注意力提升性能
- 功能差异源于预训练中的头部功能特化

### ZeroTuning 方法

三步流程：
1. **头部行为分析**：评估每个头对初始 token 缩放的敏感性
2. **选择性缩放**：仅对主导头类型施加缩放因子 $\gamma$
3. **重归一化**：softmax 重归一化保持有效分布

**两种校准模式**：
- **监督模式**：在标注验证集上最大化准确率搜索 $\gamma$
- **无监督模式**：最小化输出熵——$\gamma$ 最小化熵与最大化准确率强相关

**兼容性**：支持 SDPA 和 FlashAttention（通过缩放 query/key states）。

## 实验

### 分类任务

| 模型 | Vanilla | ACT | Auto-PASTA | **ZeroTuning** |
|------|---------|-----|------------|---------------|
| Llama-3.1-8B Avg | 59.59 | 60.11 | 63.73 | **71.44** |
| Qwen-2-7B Avg | 55.10 | - | 65.57 | **68.19** |
| Deepseek-R1-14B Avg | 67.67 | - | 69.04 | **71.87** |

最大单数据集提升：SST-2 上 73.20 → 91.60（+18.4%），SUBJ 上 44.60 → 66.60（+22.0%）。

### 多选 QA 任务

| 模型 | Vanilla | Auto-PASTA | **ZeroTuning** |
|------|---------|------------|---------------|
| Llama-3.1-8B Avg | 58.84 | 60.18 | **61.48** |
| Qwen-2-7B Avg | 63.10 | 64.01 | **64.84** |
| Deepseek-R1-14B Avg | 60.05 | 60.31 | **62.20** |

LogiQA 上 Deepseek-R1-14B: 27.80 → 35.60（+7.80%）。

### MT-Bench 对话

| 模型 | Vanilla | ZeroTuning |
|------|---------|-----------|
| Llama-3.1-8B | 7.804 | **7.966** |
| Llama-2-13B | 6.650 | **6.916** |

### 关键发现

1. **仅调整单个 token 即一致超越多 token 调优方法**
2. **准确率与输出熵的强逆相关性**：验证无监督模式的可行性
3. **头部特异性调优远优于均匀调优**
4. **对量化推理、长上下文、few-shot 设置均保持稳健**
5. **仅需 4 行代码修改**

## 亮点

1. **极简主义设计**：一个 token、一个缩放因子、4 行代码
2. **理论分析深入**：从注意力重塑到偏差校正的完整推导
3. **无监督模式**：基于熵最小化，无需标注数据
4. **内核无关**：兼容 SDPA 和 FlashAttention
5. **跨模型跨任务一致有效**

## 局限性

1. 最优缩放方向依赖于任务（分类 vs QA），需要初步实验或启发式判断
2. 头部行为分析仍需一定计算成本
3. 提升幅度在已经很强的大模型上（如 Deepseek-R1-14B）相对有限
4. Up+Down 混合策略未超越单一策略，联合优化尚需探索
5. 对生成任务（开放式对话）的提升有限

## 相关工作
- **注意力调优**: PASTA, Auto-PASTA, ACT — 需要识别重要 token
- **Attention Sink 研究**: StreamingLLM, Barbero et al. — 解释现象但未利用
- **推理时优化**: 自一致性, CoT — 提示工程方向

## 评分
- **创新性**: ⭐⭐⭐⭐ — 极简但有效的创意，将 attention sink 从被动观察转为主动利用
- **实验充分性**: ⭐⭐⭐⭐⭐ — 15 个数据集、4 个模型、多维度分析
- **写作质量**: ⭐⭐⭐⭐⭐ — 逻辑递进，由理论到方法到实验的完整故事
- **实用性**: ⭐⭐⭐⭐⭐ — 4 行代码即可部署，无训练无额外内存

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[ACL 2025\] Enhancing Automated Interpretability with Output-Centric Feature Descriptions](../../ACL2025/interpretability/output_centric_interpretability.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](../../ACL2025/interpretability/a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](../../ICML2026/interpretability/towards_atoms_of_large_language_models.md)
- [\[ICLR 2026\] Hidden Breakthroughs in Language Model Training](hidden_breakthroughs_in_language_model_training.md)

</div>

<!-- RELATED:END -->
