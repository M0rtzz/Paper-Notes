---
title: >-
  [论文解读] Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards
description: >-
  [ICLR 2026][LLM/NLP][复合AI系统] 提出 Optimas 框架，为复合 AI 系统中每个组件维护一个与全局奖励对齐的局部奖励函数（LRF），使异构组件（prompt、模型参数、超参数、模型选择）可独立优化，在五个真实系统上平均提升 11.92%。
tags:
  - ICLR 2026
  - LLM/NLP
  - 复合AI系统
  - 局部奖励函数
  - 全局对齐
  - 异构参数优化
  - 收敛保证
---

# Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards

**会议**: ICLR 2026  
**arXiv**: [2507.03041](https://arxiv.org/abs/2507.03041)  
**代码**: https://optimas.stanford.edu/  
**领域**: LLM NLP / 系统优化  
**关键词**: 复合AI系统, 局部奖励函数, 全局对齐, 异构参数优化, 收敛保证

## 一句话总结
提出 Optimas 框架，为复合 AI 系统中每个组件维护一个与全局奖励对齐的局部奖励函数（LRF），使异构组件（prompt、模型参数、超参数、模型选择）可独立优化，在五个真实系统上平均提升 11.92%。

## 研究背景与动机
**领域现状**：现代 AI 系统越来越多地集成 LLM、检索器、工具调用、传统 ML 模型等多个组件，形成复合 AI 系统来处理复杂任务。这些系统对组件故障高度敏感——一个组件的错误会沿 pipeline 级联放大。

**现有痛点**：(a) 组件间不可微分，无法端到端梯度优化；(b) 配置空间高度异构——文本 prompt、连续超参数、模型权重、离散模型选择等需要完全不同的优化策略；(c) 每次评估全局性能都需运行完整系统，成本高昂，数据效率低下。

**核心矛盾**：现有方法（DSPy 优化 prompt、TextGrad 用文本反馈优化、OPRO 单步优化）只能处理**单一类型**的参数。即使各组件独立优化到最佳，上游组件也无法感知下游偏好，组件间协作可能是次优的。缺乏统一框架来同时优化异构配置。

**核心idea**：为每个组件学习一个局部奖励函数（LRF），只要 LRF 与全局奖励保持对齐（即局部最优方向与全局一致），就可以用各组件最适合的方法独立优化，无需频繁运行全系统。这本质上将联合优化分解为多个独立的坐标优化问题。

## 方法详解

### 整体框架
复合系统建模为 DAG $\mathcal{G}=(\mathcal{C},\mathcal{E})$，包含 $K$ 个组件 $\{C_k\}_{k=1}^K$。每个组件 $C_k$ 有配置策略 $\mathbf{v}_k$（可以是 prompt、超参或模型权重）。系统支持动态规划——对不同输入 $x$，组件间连接 $\mathcal{E}(x)$ 可以自适应变化。输入按拓扑序经过各组件产生输出 $y=f(x;\mathbf{v})$，目标是最大化 $\mathbf{v}^{\star}=\arg\max_{\mathbf{v}} \mathbb{E}_{x\sim\mathcal{D}}[R(x,f(x;\mathbf{v}))]$。

### 关键设计
1. **局部奖励函数（LRF）**：

    - 功能：对每个组件 $C_k$ 学习评分函数 $r_k(x_k,y_k)$，评估其输出对全局性能的贡献
    - 核心思路：所有 LRF 共享 LLM backbone $\phi$，加组件特定线性投影头 $h_k$：$r_k(x_k,y_k) = h_k \circ \phi([x_k, y_k])$。共享 backbone 保证扩展性，独立头捕获组件特异性
    - 对齐性质（关键）：如果 $r_k(x_k,y_k^+) \geq r_k(x_k,y_k^-)$，则用 $y_k^+$ 替换后的下游系统全局奖励也应更高。训练时用 pairwise log-sigmoid ranking loss：$\mathcal{L}_k = -\mathbb{E}[\log\sigma(r_k(x_k,y_k^+)-r_k(x_k,y_k^-))]$，偏好数据通过 Monte Carlo 采样下游输出构造
    - 设计动机：将全局优化分解为独立局部优化的理论基石——Theorem 4.1 证明最小化此 loss 的 LRF 必然满足对齐性质
2. **自适应 LRF 更新**：

    - 功能：配置变化时轻量级更新 LRF 保持对齐
    - 核心思路：Stage 1 初始离线训练 LRF 至收敛；Stage 2 每次配置更新后仅采样小批量偏好数据在线 adaptation，维护历史 buffer 提升稳定性
    - 设计动机：系统配置变化后 LRF 会过时（上游更新改变了同一输出的全局价值，下游更新使 LRF 面临分布外输入）。避免从头重训 LRF 的昂贵开销
3. **异构组件优化**：

    - 文本 prompt：用 OPRO 按 LRF 平均分数排序候选 prompt 选最优
    - 可训练模型（如 LLM）：用 PPO 等 RL 算法，以 LRF 作为 critic
    - 离散/低维连续配置（模型选择、超参）：构建基于 LRF 分数的概率分布采样更新
    - 验证门控：仅当小验证集上全局奖励提升时才接受新配置，防止级联错误

### 理论保证
- **Theorem 4.1**：LRF 的 ranking loss 最小化器满足局部-全局对齐性质，且最大化 LRF 与最大化条件全局奖励具有相同解
- **Theorem 4.2**：在紧致性和唯一分量最优条件下，Optimas 收敛到 component-wise maximum（坐标最大化经典结果的直接推论）

## 实验关键数据

### 主实验（五个真实复合系统）

| 系统 | 任务 | Unoptimized | DSPy | TextGrad | **Optimas** | 相对提升 |
|------|------|-------------|------|----------|-------------|----------|
| Amazon 产品推荐 | Acc | 21.21 | 18.18 | 20.88 | **24.24** | +14.3% |
| PubMedQA 医疗 | Acc | 57.46 | 60.26 | 56.96 | **69.13** | +1.8% |
| STaRK-Prime 检索 | MRR | 40.73 | 41.40 | 41.31 | **50.54** | +22.1% |
| HotpotQA RAG | F1 | 33.80 | 44.90 | 24.86 | **50.48** | +12.4% |
| BigCodeBench 代码 | Pass | 36.67 | 33.81 | 35.71 | **38.92** | +9.0% |

### 消融与关键分析

| 配置 | 说明 |
|------|------|
| Optimas (完整) | 全部组件使用对齐 LRF 独立优化，5 个系统全部提升 |
| w/o LRF adaptation | 下降 2-5%，LRF 不更新导致对齐退化 |
| Global reward only | 下降 3-8%，缺乏局部信号数据效率低 |
| DSPy (仅prompt) | 在 Amazon 推荐上反而下降 14.3%，优化单一配置类型不可靠 |

- **Optimas 是唯一在全部 5 个任务上都提升性能的方法**；DSPy 和 TextGrad 在部分系统上反而降低性能
- LRF 排序准确率平均 77.96%，远超 LLM Judge (49.52%)，说明学习的 LRF 比直接用 LLM 打分更可靠
- 系统运行次数平均 0.71k vs DSPy 0.79k，数据效率更高
- LRF 的 adaptive update 是长期效果的关键——不更新时后期性能退化明显

### 关键发现
- 异构配置联合优化是决定性因素：仅优化 prompt 在行为驱动推荐（需要超参调整）上失效
- LRF 的对齐性质在实践中确实成立——局部改进一致地带来全局提升
- 复合系统中的瓶颈组件各不相同：Amazon 推荐的瓶颈在超参，HotpotQA 的瓶颈在 prompt

## 亮点
- 统一框架处理异构配置优化，DSPy/TextGrad 只能单类型
- LRF 对齐有严格理论保证（收敛到分量最优）
- 共享 backbone + 独立头的 LRF 架构可扩展且内存高效
- 5 个真实系统上一致提升，DSPy 在 Amazon 上反而下降 14.3%

## 局限与展望
- 坐标最大化在非凸问题中只保证分量最优，非全局最优
- LRF 在线适配仍需少量系统运行和 Monte Carlo 采样，成本并非为零
- 实验中组件数量有限（2-5个），更大规模系统的可扩展性未验证
- LRF 共享 backbone 在组件输入分布差异极大时可能学习冲突表征

## 与相关工作的对比
- **DSPy/TextGrad**: 仅优化 prompt，不支持异构配置；DSPy 在部分任务上性能不稳定
- **OPRO**: 单步生成优化，无法处理多组件多步骤
- **LLMSelector**: 仅做模型路由，系统运行成本 3x 于 Optimas
- **过程奖励模型**: 依赖人工标注或 MCTS，Optimas 通过偏好自动构造对齐数据

## 评分
- 新颖性: ⭐⭐⭐⭐ (LRF 对齐思路新颖，统一异构优化)
- 实验充分度: ⭐⭐⭐⭐⭐ (5 个真实系统 + 丰富消融 + 理论分析)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图表丰富)
- 价值: ⭐⭐⭐⭐ (复合 AI 系统优化是重要方向)

<!-- RELATED:START -->

## 相关论文

- [ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](ellmob_event-driven_human_mobility_generation_with_self-aligned_llm_framework.md)
- [PhysVid: Physics Aware Local Conditioning for Generative Video](../../CVPR2026/llm_nlp/physvid_physics_aware_local_conditioning_for_generative_video_models.md)
- [LLM-AT: Automatic Transmission for LLM Tiers Optimizing Cost and Accuracy](../../ACL2025/llm_nlp/automatic_transmission_for_llm_tiers_optimizing_cost_and_accuracy_in_large_langu.md)
- [Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](../../ACL2025/llm_nlp/unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models](../../NeurIPS2025/llm_nlp/geocad_local_geometry-controllable_cad_generation_with_large_language_models.md)

<!-- RELATED:END -->
