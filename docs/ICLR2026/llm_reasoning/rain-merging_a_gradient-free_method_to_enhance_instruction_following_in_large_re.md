---
title: >-
  [论文解读] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format
description: >-
  [ICLR 2026][LLM推理][模型合并] 针对大推理模型（LRM）推理能力强但指令遵循能力弱的矛盾，提出 RAIN-Merging 方法，通过零空间投影保持 thinking 格式不变、注意力引导系数增强指令相关性，无需梯度训练即可将指令微调模型（ITM）的能力合并进 LRM，在 4 个指令遵循和 9 个推理基准上均取得稳定提升。
tags:
  - ICLR 2026
  - LLM推理
  - 模型合并
  - 指令遵循
  - 大推理模型
  - 零空间投影
  - 注意力引导
---

# RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format

**会议**: ICLR 2026  
**arXiv**: [2602.22538](https://arxiv.org/abs/2602.22538)  
**代码**: https://github.com/K1nght/RAIN-Merging (有)  
**领域**: LLM推理  
**关键词**: 模型合并, 指令遵循, 大推理模型, 零空间投影, 注意力引导

## 一句话总结
针对大推理模型（LRM）推理能力强但指令遵循能力弱的矛盾，提出 RAIN-Merging 方法，通过零空间投影保持 thinking 格式不变、注意力引导系数增强指令相关性，无需梯度训练即可将指令微调模型（ITM）的能力合并进 LRM，在 4 个指令遵循和 9 个推理基准上均取得稳定提升。

## 研究背景与动机
大推理模型（如 DeepSeek-R1、OpenAI-o1）在数学推导和代码生成等多步推理任务上表现出色，但在指令遵循方面存在悖论性缺陷：模型能够生成冗长的逻辑推导，却常常忽略用户指定的格式、约束或特定操作要求。这一问题严重影响了 LRM 在 agent 场景和实际工具部署中的实用性。

直接的解决思路是用 SFT 继续训练 LRM，但构建高质量的长 CoT 监督数据成本极高，且容易导致能力退化。模型合并（Model Merging）作为一种无训练的轻量替代方案，通过线性组合任务向量来融合多种能力。然而，LRM 和 ITM 存在**输出结构不匹配**的根本问题：LRM 使用 `<think>...</think>` 显式分隔推理和回答段，而 ITM 只输出最终答案。直接合并会破坏 LRM 的结构化推理格式。

核心 idea：先通过参数空间分析发现 LRM 和 ITM 的任务向量主子空间近乎正交（相似度 < 0.1），说明两种能力耦合度低、合并可行；再分两阶段解决输出格式问题和指令增强问题——Stage 1 用零空间投影保护 thinking token 的分布不变，Stage 2 用注意力统计引导模块级缩放系数来强化指令相关组件。

## 方法详解

### 整体框架
RAIN-Merging（Reasoning-Aware Instruction-attention guided Null-space projection Merging）是一个两阶段的无梯度合并管线。以 LRM 参数 θ_R 为锚点，将 ITM 任务向量 Δ_I = θ_I − θ_B 经过变换后加到 LRM 上，最终模型为 θ* = θ_R + λ ⊕_k α*_k Δ_I^{⊥,k}。

### 关键设计

1. **Stage 1: 推理感知零空间投影（Reasoning-aware Null-space Projection）**

    - **功能**：将 ITM 任务向量投影到 thinking 特殊 token 前向特征的零空间中
    - **为什么**：确保合并后模型在 thinking token 位置的中间表示和最终 logits 与原始 LRM 保持一致，从而保护 `<think>...</think>` 结构化格式
    - **怎么做**：对每个子模块 k，用少量推理校准数据（150 条）构建 thinking token 位置的前向特征算子 Φ，计算正交投影矩阵 P^⊥(Φ) = I − Φ^T(ΦΦ^T)^+Φ，然后将 ITM 任务向量投影：vec(Δ_I^{⊥,k}) = P^⊥(Φ) vec(Δ_I^k)
    - **理论保证**：通过 softmax-KL 散度的二阶展开证明，投影后的任务向量满足 L_think ≈ 0（Proposition 1），即合并后在 thinking token 上的分布偏移可忽略
    - **区别**：传统合并方法（如 Task Arithmetic）忽略输出分布不匹配，导致 6.4% 的生成缺失 `</think>` 标记；本方法将缺失率降到 0%

2. **Stage 2: 指令注意力引导的合并系数（Instruction-attention Guided Merging Coefficients）**

    - **功能**：为每个子模块计算自适应缩放系数 α，放大指令相关组件、抑制泄漏
    - **为什么**：指令遵循失败常源于解码时对指令 span 注意力不足，不同层和头对指令的响应具有异质性
    - **怎么做**：用 365 条指令校准数据，计算每个注意力头的对齐度（alignment）和泄漏度（leakage）；定义指令注意力得分 J = alignment − ρ·leakage，通过二阶 Taylor 展开得到闭式解：α*_k = clip(g^k / H^k)
    - **区别**：现有 activation-based 合并方法（如 ACM、LEWIS）缺乏对输出结构不匹配的显式处理，而本方法通过 alignment/leakage 分解提供了可解释的指令增强机制

### 损失函数 / 训练策略
本方法完全无梯度（gradient-free），无需训练。仅需两个小规模校准集：
- 推理校准集：150 条 Mixture-of-Thoughts 数据，用于 Stage 1 的零空间计算
- 指令校准集：365 条 IFEval 数据经 R1 蒸馏 + LLM 筛选 + 人工审核，用于 Stage 2 的注意力统计

全局缩放系数 λ 控制合并强度，仅合并 Q、K、V、O 和 FFN 参数。

## 实验关键数据

### 主实验

| 方法 | IFEval | CELLO | InfoBench | ComplexBench | IF Avg. | Math | GPQA | Aider | Arena-Hard | RG Avg. |
|------|--------|-------|-----------|-------------|---------|------|------|-------|------------|---------|
| ITM (Qwen2.5-7B-Inst) | 70.43 | 19.15 | 78.49 | 43.63 | 52.92 | 47.27 | 29.80 | 33.33 | 62.86 | 43.32 |
| LRM (R1-Distill-Qwen-7B) | 55.45 | 16.59 | 71.73 | 32.72 | 44.12 | 64.75 | 44.44 | 29.63 | 65.29 | 51.03 |
| SFT | 62.48 | 17.11 | 68.58 | 32.15 | 45.08 | 62.57 | 41.92 | 28.89 | 64.67 | 49.51 |
| Task Arithmetic | 60.44 | 16.97 | 73.07 | 33.34 | 45.96 | 64.22 | 42.93 | 26.67 | 64.53 | 49.59 |
| AIM-TIES | 62.78 | 17.93 | 73.11 | 34.28 | 47.02 | 65.92 | 49.49 | 33.33 | 63.64 | 53.10 |
| **RAIN-Merging** | **63.22** | **19.03** | **74.53** | **35.66** | **48.11** | **68.75** | **54.55** | **33.33** | **65.73** | **55.59** |

RAIN-Merging 在 IF Avg.（48.11）和 RG Avg.（55.59）上均显著领先所有合并基线和 SFT，运行时间仅约 21 分钟（SFT 需 120 分钟）。

### 消融实验

| 方法 | 指令遵循 Avg. | 推理/通用 Avg. |
|------|-------------|--------------|
| RAIN-Merging w/o Stage 2 | 46.58 | 54.92 |
| RAIN-Merging w/o Stage 1 | 47.62 | 52.44 |
| **RAIN-Merging (完整)** | **48.11** | **55.59** |

去掉 Stage 1 后推理能力明显下降（52.44 vs 55.59），去掉 Stage 2 后指令遵循提升有限。两阶段互补不可缺。

### 关键发现
- **跨规模一致性**：在 1.5B/7B/8B/14B/32B 五种规模和 Qwen/Llama 两种架构上均有稳定提升，IF Avg. 相对增益 1.57%–9.18%，RG Avg. 相对增益 2.89%–14.47%
- **Agent 场景有效**：在 ALFWorld 和 WebShop 上，合并模型（25.0/29.42）超越 LRM（22.0/26.63）和 ITM（17.5/10.45）
- **零空间投影效果**：Task Arithmetic 的 L_think = 0.1224，缺失 `</think>` 率 6.4%；RAIN-Merging 的 L_think = 0.0065，缺失率 0%
- **MathIF 上尤为突出**：在需要同时满足数学正确和格式约束的 MathIF 上，Both Acc. 从 12.62% 提升到 20.48%（+62.26%）

## 亮点与洞察
- 参数空间正交性分析为合并可行性提供了理论支撑，是一个优雅的观察
- 两阶段设计巧妙分离了"保护推理格式"和"增强指令遵循"两个目标
- 零空间投影具有严格的理论保证（Proposition 1），不是纯经验方法
- 整个流程完全无梯度，仅需 ~500 条校准数据和约 20 分钟计算，极其实用
- alignment/leakage 分解提供了可解释的注意力分析视角

## 局限与展望
- 合并后的 IF 能力仍低于 ITM（48.11 vs 52.92），无训练方法有天花板
- 对 thinking token 位置的零空间投影依赖于校准数据的代表性
- 未探索对 thinking 内容本身质量的优化，仅保护格式
- 在极大规模模型（>70B）上的效果尚未验证
- 校准集的构建仍需 LLM 蒸馏和人工筛选，未完全自动化

## 相关工作与启发
- 与 TIES、DARE 等数据无关合并方法相比，RAIN-Merging 引入了输出结构约束
- 与 ACM、LEWIS、AIM 等 activation-based 方法相比，显式处理了 LRM/ITM 输出格式不匹配
- 零空间投影思路可推广到其他需要"保护某种特定行为不被合并破坏"的场景
- 指令注意力得分的 alignment/leakage 框架可用于分析任何模型的指令遵循机制

## 评分
- 新颖性: ⭐⭐⭐⭐ 两阶段设计和零空间投影用于合并场景是新颖的，但模型合并这条路线已有大量工作
- 实验充分度: ⭐⭐⭐⭐⭐ 4个IF + 9个推理基准 + 5种规模 + 2种架构 + agent场景 + 完整消融
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，可视化丰富，但符号较多读起来稍显繁复
- 价值: ⭐⭐⭐⭐ 解决了LRM的实际痛点，方法实用且轻量，对工业部署有直接参考价值

<!-- RELATED:START -->

## 相关论文

- [RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging](rain-merging_a_gradient-free_method_to_enhance_instruction_following_through_mod.md)
- [Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)
- [RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)

<!-- RELATED:END -->
