---
title: >-
  [论文解读] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging
description: >-
  [ICLR 2026][LLM推理][模型合并] 提出 RAIN-Merging，一种无梯度的两阶段模型合并方法：先通过零空间投影保护大推理模型 (LRM) 的思维格式，再用指令注意力引导的合并系数增强指令遵循能力，在保持推理质量的同时大幅提升 LRM 的指令遵循性能。
tags:
  - ICLR 2026
  - LLM推理
  - 模型合并
  - 指令遵循
  - 推理模型
  - 零空间投影
  - 无梯度方法
---

# RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging

**会议**: ICLR 2026  
**arXiv**: [2602.22538](https://arxiv.org/abs/2602.22538)  
**代码**: [GitHub](https://github.com/K1nght/RAIN-Merging)  
**领域**: LLM推理  
**关键词**: 模型合并, 指令遵循, 推理模型, 零空间投影, 无梯度方法

## 一句话总结

提出 RAIN-Merging，一种无梯度的两阶段模型合并方法：先通过零空间投影保护大推理模型 (LRM) 的思维格式，再用指令注意力引导的合并系数增强指令遵循能力，在保持推理质量的同时大幅提升 LRM 的指令遵循性能。

## 研究背景与动机

### 大推理模型的指令遵循矛盾

DeepSeek-R1、OpenAI-o1 等大推理模型 (LRM) 在多步推理上表现优秀，但在指令遵循上存在显著缺陷：

- 生成长链逻辑推导但忽略用户指定的格式、约束或特定需求
- 这种不一致严重影响 LRM 在 agent、专业工具部署中的实用性

### 现有解决方案的局限

- **SFT 继续训练**：需要大量标注数据（含长思维链），成本高且易导致能力退化
- **朴素模型合并**：LRM 和 ITM (Instruction-tuned Model) 的输出格式根本不同——LRM 有 `<think>...</think>` 分隔思维和回答，ITM 直接输出答案。朴素合并会破坏思维格式

### 参数空间分析的关键发现

对 LRM 和 ITM 的 task vector 做 SVD 分析后发现：**两者的主子空间在所有关键模块上几乎正交**（余弦相似度 < 0.1）。这意味着推理能力和指令遵循能力在参数空间中低耦合，合并有望不产生干扰。

但正交性不足以保证输出行为不变——特殊 token（`<think>`, `</think>`）的生成概率在前向传播中可能被改变。

## 方法详解

### 整体框架

RAIN-Merging 分两阶段，最终合并模型为：

$$\theta_\star = \theta_R + \lambda \bigoplus_{k=1}^{K} \alpha_\star^k \Delta_I^{\perp, k}$$

其中 $\theta_R$ 是 LRM 参数，$\Delta_I^{\perp, k}$ 是经零空间投影后的 ITM task vector，$\alpha_\star^k$ 是指令注意力引导的合并系数，$\lambda$ 是全局缩放因子。

### Stage 1：推理感知的零空间投影

**目标**：保护 LRM 的思维格式（`<think>...</think>` 结构）不被合并破坏。

**核心思想**：将 ITM task vector 投影到思维特殊 token 的前向特征算子 $\Phi$ 的零空间中——在这个子空间内的扰动不会改变思维位置的中间表示。

对每个子模块 $k$，构造零空间投影：

$$P^\perp(\Phi^k_{\Omega_{\text{think}}}) = \text{diag}(1) - {\Phi^k_{\Omega_{\text{think}}}}^\top (\Phi^k_{\Omega_{\text{think}}} {\Phi^k_{\Omega_{\text{think}}}}^\top)^+ \Phi^k_{\Omega_{\text{think}}}$$

投影后的 task vector 满足 $\Phi_{\Omega_{\text{think}}} \text{vec}(\Delta_I^\perp) = 0$，即在思维 token 位置的前向特征完全不变。

**理论保证 (Proposition 1)**：通过对 softmax-KL 散度的二阶近似证明：

$$\mathcal{L}_{\text{think}}(\theta_R + \Delta_I^\perp) = O(\|\Delta_I^\perp\|_2^2) \approx 0$$

只需 150 个推理校准样本即可构建投影。

### Stage 2：指令注意力引导的合并系数

**目标**：在保持思维格式的前提下最大化指令遵循性能。

对每个注意力头 $\tilde{k}$，定义指令对齐度和泄漏度：

$$a^{\tilde{k}}(x, \tilde{\alpha}) = \sum_{t \in \mathcal{R}(x)} \sum_{\tau \in \mathcal{I}(x)} \frac{\text{Att}^{\tilde{k}}(x, \tilde{\alpha})[t, \tau]}{|\mathcal{I}(x)| |\mathcal{R}(x)|}$$

$$u^{\tilde{k}}(x, \tilde{\alpha}) = \sum_{t \in \mathcal{U}(x)} \sum_{\tau \in \mathcal{I}(x)} \frac{\text{Att}^{\tilde{k}}(x, \tilde{\alpha})[t, \tau]}{|\mathcal{I}(x)| |\mathcal{U}(x)|}$$

其中 $\mathcal{I}(x)$ 为指令 token 集合，$\mathcal{R}(x)$ 为受指令约束的输出 token，$\mathcal{U}(x)$ 为无关输出 token。

**优化目标**：最大化指令注意力得分（高对齐 - 低泄漏）：

$$\max_{\tilde{\alpha}} \mathcal{J}_I^{\text{Proxy}}(\tilde{\alpha}) := \bar{a}(\tilde{\alpha}) - \rho \bar{u}(\tilde{\alpha})$$

通过二阶 Taylor 展开和工程近似得到闭式解：

$$\tilde{\alpha}_\star^{\tilde{k}} = \text{clip}_{[\tilde{\alpha}_l, \tilde{\alpha}_u]} \left(\frac{g^{\tilde{k}}}{\tilde{H}^{\tilde{k}}}\right)$$

整个过程仅需前向传播的注意力统计，完全无梯度。使用 365 个指令校准样本。

### 损失函数

RAIN-Merging 不涉及训练损失——是一种无梯度的模型合并方法。其优化目标是约束优化问题：

$$\max_\Delta \mathcal{J}_I(\theta_R + \Delta) \quad \text{s.t.} \quad \mathcal{L}_{\text{think}}(\theta_R + \Delta) \leq \delta$$

## 实验关键数据

### 主实验：7B 模型合并（DeepSeek-R1-Distill-Qwen-7B + Qwen2.5-7B-Instruct）

| 方法 | IF 平均 ↑ | 推理平均 ↑ | 运行时间 |
|------|-----------|-----------|---------|
| LRM (原始) | 44.12 | 51.03 | - |
| ITM (原始) | 52.92 | 43.32 | - |
| SFT | 45.08 | 49.51 | 120.32 min |
| Task Arithmetic | 45.96 | 49.59 | 0.93 min |
| SLERP | 45.95 | 50.97 | 1.12 min |
| TIES | 46.35 | 51.99 | 1.18 min |
| AIM-TIES | 47.02 | 53.10 | 18.51 min |
| **RAIN-Merging** | **48.11** | **55.59** | 20.96 min |

RAIN-Merging 在指令遵循（+4.0 vs LRM）和推理能力（+4.6 vs LRM）上同时提升。

### 多尺度和多架构验证

| 模型配置 | IF 相对提升 | 推理相对提升 |
|---------|-----------|------------|
| Qwen2.5-1.5B | +6.09% | +8.20% |
| Qwen2.5-7B | +9.06% | +8.93% |
| Llama-3.1-8B | +5.86% | +7.78% |
| Qwen2.5-14B | +6.11% | +6.17% |
| Qwen2.5-32B | +1.57% | +3.83% |

跨 1.5B-32B 规模和 Qwen/Llama 两种架构一致有效。

### 消融实验

| 方法 | IF 平均 | 推理平均 |
|------|---------|---------|
| w/o Stage 2 (仅零空间投影) | 46.58 | **54.92** |
| w/o Stage 1 (仅注意力引导) | **47.62** | 52.44 |
| **RAIN-Merging 完整** | **48.11** | **55.59** |

- 去掉 Stage 2：指令遵循提升有限，但推理保持良好
- 去掉 Stage 1：指令遵循更强，但推理能力明显下降
- 两阶段互补：同时获得最佳指令遵循和推理性能

### 思维格式保护验证

| 方法 | $\mathcal{L}_{\text{think}}$ | `</think>` 缺失率 |
|------|------------------------------|-------------------|
| Task Arithmetic | 0.1224 | 6.4% |
| **RAIN-Merging** | **0.0065** | **0.0%** |

零空间投影将 KL 散度从 0.12 降至 0.006，完全消除了 `</think>` 缺失问题。

### Agent 场景

| 模型 | ALFWorld | WebShop |
|------|----------|---------|
| ITM | 17.50 | 10.45 |
| LRM | 22.00 | 26.63 |
| **RAIN-Merging** | **25.00** | **29.42** |

### 关键发现

1. **推理和指令遵循是正交能力**：task vector 的主子空间余弦相似度 < 0.1
2. **思维格式保护至关重要**：朴素合并破坏 `<think>` 结构导致推理退化
3. **增强指令遵循可反向提升推理**：更好的指令理解改善了思维链质量
4. **无梯度方法 vs SFT**：RAIN-Merging 运行时间仅 21 分钟 vs SFT 120 分钟，性能更优

## 亮点与洞察

1. **精确定位问题**：从参数空间正交性分析出发，同时发现输出格式不匹配的风险
2. **零空间投影的优雅**：用线性代数工具在保证思维格式不变的同时最大化指令能力
3. **指令注意力的可解释性**：通过对齐度和泄漏度量化每个注意力头对指令的响应
4. **实用性极强**：无需训练、仅需 ~500 校准样本、20 分钟完成，远优于 SFT
5. **理论保证完整**：Proposition 1 证明零空间投影满足 KL 约束

## 局限性

1. 额外的零空间计算和注意力统计虽比 SFT 快，但比最简单的 Task Arithmetic 慢约 20 倍
2. 依赖校准数据集质量——推理校准集和指令校准集的选择可能影响结果
3. 仅合并了 Q、K、V、O 和 FFN 参数，其他模块（如 embedding）未考虑
4. 在 32B 规模时提升幅度减小，超大模型的适用性有待验证
5. 方法假设 LRM 和 ITM 共享同一 base model，不适用于完全不同架构的模型

## 相关工作与启发

- **Task Arithmetic (Ilharco et al. 2023)**：RAIN-Merging 的基础，但朴素线性加法会破坏推理格式
- **TIES / DARE**：数据无关的 task vector 修剪方法，RAIN-Merging 通过数据驱动约束超越
- **AIM (Nobari et al. 2025)**：激活感知合并但不考虑输出格式不匹配
- **Guardieiro et al. 2025**：指令注意力分析的灵感来源，但 RAIN-Merging 将其系统化为合并系数
- **启发**：模型合并不仅需要参数空间分析，还需考虑输出分布的格式兼容性

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次系统解决 LRM+ITM 合并中的思维格式保护问题
- **理论深度**: ⭐⭐⭐⭐⭐ — 零空间投影的理论推导和 KL 约束的证明严谨完整
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 个 IF 基准 + 9 个推理基准 + 5 种模型规模 + Agent 场景
- **实用价值**: ⭐⭐⭐⭐⭐ — 无梯度、20 分钟、500 样本即可显著提升 LRM 的指令遵循
- **总评**: ⭐⭐⭐⭐⭐ — 问题重要、方法优雅、理论严谨、实验全面，是模型合并领域的优秀工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following in Large Reasoning Models with Preserved Thinking Format](rain-merging_a_gradient-free_method_to_enhance_instruction_following_through_mod.md)
- [\[ICLR 2026\] On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](on_the_design_of_kl-regularized_policy_gradient_algorithms_for_llm_reasoning.md)
- [\[ICLR 2026\] On The Fragility of Benchmark Contamination Detection in Reasoning Models](on_the_fragility_of_benchmark_contamination_detection_in_reasoning_models.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[ICLR 2026\] Estimating the Empowerment of Language Model Agents](estimating_the_empowerment_of_language_model_agents.md)

</div>

<!-- RELATED:END -->
