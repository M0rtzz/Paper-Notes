---
title: >-
  [论文解读] Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models
description: >-
  [AAAI 2026][LLM安全][幻觉] 从计算复杂度理论出发证明 Transformer LLM 每步推理复杂度为 $O(N^2 \cdot d)$，基于时间层次定理（Hartmanis-Stearns）…
tags:
  - "AAAI 2026"
  - "LLM安全"
  - "幻觉"
  - "计算复杂度"
  - "时间层次定理"
  - "Transformer"
  - "智能体 AI"
  - "验证不可能性"
---

# Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models

**会议**: AAAI 2026  
**arXiv**: [2507.07505](https://arxiv.org/abs/2507.07505)  
**领域**: LLM安全  
**关键词**: 幻觉, 计算复杂度, 时间层次定理, Transformer, 智能体 AI, 验证不可能性

## 一句话总结

从计算复杂度理论出发证明 Transformer LLM 每步推理复杂度为 $O(N^2 \cdot d)$，基于时间层次定理（Hartmanis-Stearns），任何需要超过此复杂度的计算任务——如 $O(n^3)$ 矩阵乘法、$O(n^k)$ token 组合、TSP 验证等——LLM 必然无法正确完成（即产生幻觉），且 LLM Agent 也无法验证此类任务的正确性。

## 研究背景与动机

**领域现状**：LLM 已广泛应用于各领域，但"幻觉"问题——模型生成虚假、不准确或无意义信息——是部署的核心障碍。同时，Agentic AI 的兴起使 LLM 从信息提供扩展到自主执行真实世界任务（金融交易、预订服务、法律文件等）。

**现有痛点**：(1) 对幻觉的理解多停留在经验层面（数据不足、分布偏移），缺乏从计算理论角度的根本性解释；(2) 人们寄希望于让一个 Agent 验证另一个 Agent 的输出来解决幻觉，但这一策略的可行性缺乏理论分析；(3) 推理模型（如 o3、R1）被认为能克服幻觉，但尚无严格论证。

**核心矛盾**：LLM 的推理能力受限于其架构的计算复杂度上界，但实际任务的复杂度可以任意高——这一固有鸿沟意味着部分幻觉是不可避免的。

**本文目标** 从计算复杂度的角度严格论证 LLM 的能力上界，说明幻觉是架构层面的根本限制。

**切入角度**：将 LLM 推理视为确定性计算过程，利用经典的时间层次定理证明存在无穷多任务超出此计算能力。

**核心 idea**：LLM 的计算能力有 $O(N^2 \cdot d)$ 的硬上界，超出此复杂度的任务必定幻觉——这不是缺陷而是定理。

## 方法详解

### 整体框架

论文采用理论分析+实例论证结构：首先确立 LLM 推理的计算复杂度上界，然后通过三个递进例子说明超复杂度任务必然失败，最后基于时间层次定理给出形式化证明。

### 关键设计

1. **LLM 计算复杂度分析**
    - **核心论点**：Transformer 自注意力对 $N$ 个 token（$d$ 维向量）的计算复杂度为 $O(N^2 \cdot d)$，其他操作为线性或更低
    - **实验验证**：Llama-3.2-3B-Instruct 对任何 17 token 输入，总是执行恰好 109,243,372,873 次浮点运算——与输入内容无关
    - **核心直觉**：如果 prompt 表达的计算任务复杂度高于 $O(N^2 \cdot d)$，LLM 不可能正确完成

2. **三个递进的不可能性示例**
    - **示例 1（Token 组合）**：列举 $n$ 个 token 组成的所有长度 $k$ 字符串需 $O(n^k)$ 时间，当 $k$ 稍大远超 $O(N^2 \cdot d)$
    - **示例 2（矩阵乘法）**：朴素矩阵乘法需 $O(n^3)$，当维度超过词表大小时 LLM 无法正确计算
    - **示例 3（Agentic AI 验证）**：TSP 解的验证需检查 $(n-1)!/2$ 条路线，一个 Agent 也无法验证另一个 Agent 的 TSP 解

3. **形式化定理与证明**
    - **定理 1**：给定长度 $N$ 的 prompt 包含复杂度 $O(n^3)$ 或更高的计算任务，LLM 或 LLM Agent 必然幻觉
    - **证明**：Hartmanis-Stearns 时间层次定理保证存在 $O(t_2(n))$ 可解但 $O(t_1(n))$ 不可解的决策问题
    - **推论**：存在 LLM Agent 可执行、但其正确性无法被任何 LLM Agent 验证的任务

### 关于推理模型的讨论

作者认为推理模型（o3、R1）不能克服此限制：(1) 基础操作复杂度未变；(2) 推理 token 预算远小于复杂任务所需。Apple 研究表明推理模型在河内塔等指数复杂度问题上出现"推理崩溃"。

## 实验关键数据

### FLOPs 实测（Llama-3.2-3B-Instruct）

| 输入 | Token 数 | 总 FLOPs | 变化 |
|------|---------|---------|------|
| "You are a helpful assistant..." | 17 | 109,243,372,873 | - |
| "Can you find a number that..." | 17 | 109,243,372,873 | **完全相同** |

### 不同任务复杂度与 LLM 能力对比

| 任务 | 复杂度 | 超出 $O(N^2 \cdot d)$？ | LLM 能否正确 |
|------|--------|------------------------|-------------|
| 简单问答 | $O(N)$ | 否 | 概率上可以 |
| Token 组合枚举 | $O(n^k)$ | 是 | ✗ |
| 大矩阵乘法 | $O(n^3)$ | 是 | ✗ |
| TSP 最优解验证 | $O((n-1)!/2)$ | 是 | ✗ |

### 关键发现

- LLM 的 FLOPs 固定于输入长度和模型维度，与任务内容完全无关
- 时间层次定理保证了无穷多不可正确完成的任务类存在
- Agent 互验策略在超复杂度任务上同样失败

## 亮点与洞察

1. **将幻觉从经验现象提升为理论必然性**：幻觉是计算复杂度上界的直接推论
2. **Agent 验证 Agent 的不可能性**：对多 Agent 互检策略的根本性质疑
3. **Minsky 式洞察**：承认单个 LLM 能力有限，但多 LLM 协作可能涌现更强能力
4. **实测验证直观有力**：不同内容相同长度输入得到完全相同 FLOPs

## 局限与展望

1. 论证粗粒度：未区分确定性正确性和概率近似正确性
2. 多步推理（自回归生成多 token）可累积计算能力，Chain-of-Thought 可能突破单步限制
3. 未严格分析工具调用场景——LLM 调用计算器/代码解释器时实际计算能力显著扩展
4. 论文仅 6 页，未与电路复杂度 TC0 的相关工作深入联系
5. "hallucination"定义过宽——将计算错误与生成虚假事实混为一谈

## 相关工作与启发

- 与 Xu et al. (2024) "Hallucination is Inevitable" 互为补充（后者从信息论角度）
- 对 Agentic AI 的启示：超复杂度任务必须外包给专用求解器
- 推理模型的"思考 token"本质上是增加计算步数而非改变单步复杂度

## 评分

⭐⭐⭐

- **新颖性** ⭐⭐⭐：核心观点不新鲜，但系统阐述有价值
- **实验充分度** ⭐⭐：仅有 FLOPs 实测，缺少系统实验验证
- **写作质量** ⭐⭐⭐⭐：6 页内论证清晰，例子直观
- **价值** ⭐⭐⭐：对 LLM 能力边界的理论认识有贡献，但论证深度有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [\[ICML 2025\] Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Model](../../ICML2025/llm_safety/sorbet_a_neuromorphic_hardware-compatible_transformer-based_spiking_language_mod.md)
- [\[ACL 2025\] Beyond Facts: Evaluating Intent Hallucination in Large Language Models](../../ACL2025/llm_safety/intent_hallucination_eval.md)
- [\[NeurIPS 2025\] TRUST -- Transformer-Driven U-Net for Sparse Target Recovery](../../NeurIPS2025/llm_safety/trust_--_transformer-driven_u-net_for_sparse_target_recovery.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)

</div>

<!-- RELATED:END -->
