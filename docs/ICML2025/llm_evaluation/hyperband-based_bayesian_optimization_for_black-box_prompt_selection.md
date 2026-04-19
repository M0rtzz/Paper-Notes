---
title: >-
  [论文解读] Hyperband-based Bayesian Optimization for Black-box Prompt Selection
description: >-
  [ICML 2025][提示学习] 提出 HbBoPs 方法，结合结构感知深度核高斯过程（对 instruction 和 few-shot exemplar 分别编码）与 Hyperband 多保真度调度器，在黑盒 LLM 的 prompt 选择问题上同时实现样本高效和查询高效，在十个基准和三个 LLM 上超越所有 SOTA 方法。
tags:
  - ICML 2025
  - 提示学习
  - 贝叶斯优化
  - Hyperband
  - 多保真度优化
  - 高斯过程
  - 深度核
  - 黑盒LLM
---

# Hyperband-based Bayesian Optimization for Black-box Prompt Selection

**会议**: ICML 2025  
**arXiv**: [2412.07820](https://arxiv.org/abs/2412.07820)  
**代码**: 未公开  
**领域**: llm_nlp  
**关键词**: prompt选择, 贝叶斯优化, Hyperband, 多保真度优化, 高斯过程, 深度核, 黑盒LLM

## 一句话总结

提出 HbBoPs 方法，结合结构感知深度核高斯过程（对 instruction 和 few-shot exemplar 分别编码）与 Hyperband 多保真度调度器，在黑盒 LLM 的 prompt 选择问题上同时实现样本高效和查询高效，在十个基准和三个 LLM 上超越所有 SOTA 方法。

## 研究背景与动机

LLM 对输入 prompt 高度敏感，最优 prompt 选择对下游任务性能至关重要。在黑盒设定下（仅通过 API 访问 LLM，无法获取参数、梯度或 token 概率），prompt 选择面临三大挑战：

**搜索空间巨大**：instruction 和 few-shot exemplar 的组合爆炸（如 5 条 instruction × 50 个 exemplar = 250 个候选）

**无梯度信息**：无法通过反向传播优化

**评估成本高昂**：每次评估需在整个验证集上查询 LLM

现有方法的关键局限：
- **EASE、TRIPLE-SH/GSE** 未设计为联合选择 instruction 和 exemplar
- **没有方法同时具备样本高效性**（借助代理模型减少评估的 prompt 数量）**和查询高效性**（通过多保真度减少每次评估的 LLM 调用次数）

## 方法详解

### 整体框架

HbBoPs 由三个核心组件构成：

1. **结构感知深度核高斯过程**：学习 prompt 的低维表示以预测性能
2. **Hyperband 多保真度调度器**：控制每个 prompt 在多少验证实例上评估
3. **贝叶斯优化提议**：在 Hyperband 框架内使用 GP 指导候选 prompt 选择

### 结构感知深度核 GP

传统方法将整个 prompt 作为一个文本块编码，忽略了 prompt 的组合结构。本文提出分别编码 instruction 和 exemplar，利用结构信息提升代理模型预测能力。

特征提取器 $\phi$ 的架构：

- **Instruction 编码器** $\phi_{\text{enc}(i)}$：Lin(d, 64) → ReLU → Lin(64, 32) → ReLU
- **Exemplar 编码器** $\phi_{\text{enc}(e)}$：相同架构但独立参数
- **融合网络** $\phi_{(\phi_{\text{enc}(i)}, \phi_{\text{enc}(e)})}$：Lin(64, 32) → ReLU → Lin(32, 10)

使用 ARD Matérn 5/2 核，通过最大化对数边际似然联合优化核参数 $\theta$ 和特征提取器参数 $\mathbf{w}$：

$$\hat{\theta}, \hat{\mathbf{w}} = \arg\max_{\theta, \mathbf{w}} -\mathbf{v}^\intercal \mathbf{K}_t(\theta, \mathbf{w})^{-1}\mathbf{v} - \log|\mathbf{K}_t(\theta, \mathbf{w})|$$

### Hyperband 多保真度调度

将验证实例数量作为保真度参数。与 Successive Halving (SH) 相比，Hyperband 通过运行多个 bracket（不同起始 prompt 数和起始预算组合），对冲"预算 vs 配置数"困境。

关键设计决策：
- 高阶段的验证实例包含低阶段的实例（扩展而非重新采样）
- 返回在全验证集上评估的最佳 prompt
- 下限 $b_{\min} = 10$ 个验证实例，减半参数 $\eta = 2.0$

### 贝叶斯优化提议

在每个 bracket 的首轮中，用 GP 的 Expected Improvement (EI) 采集函数替代随机采样：

$$\alpha_{\text{EI}}(p|\mathcal{D}_{t|b}) = \mathbb{E}[\max\{v_{\min,b} - f(\mathbf{z}_p), 0\}]$$

GP 在线训练——在选择过程中逐步积累训练数据，不依赖预训练代理模型。使用至少有 4 个观测的最高保真度层级训练 GP。

## 实验关键数据

### 实验设置

- **基准**：10 个任务（ARC, GSM8K, 8 个 BIG-bench/Instruction Induction 任务）
- **LLM**：Claude 3 Haiku, Llama3 8B Instruct, Mistral 7B Instruct
- **搜索空间**：5 条 instruction × 50 个 exemplar = 250 个候选 prompt
- **预算**：25 次全保真度评估等价的 LLM 调用

### 总体性能

在完整预算下的平均归一化测试误差：

| 方法 | 测试误差 | 类型 |
|------|----------|------|
| RS (随机搜索) | 0.214 | baseline |
| HDBO | 0.185 | 全保真度 SOTA |
| TRIPLE-GSE | 0.158 | 多保真度 SOTA |
| TRIPLE-SH | 0.159 | 多保真度 |
| **HbBoPs** | **0.150** | **本文** |

### 分 LLM 分析（HbBoPs vs TRIPLE-SH 中位相对改进）

| 预算比例 | Claude 3 Haiku | Llama3 8B | Mistral 7B |
|----------|----------------|-----------|------------|
| 0.25 | 6.6% (test) | 3.6% (test) | 3.9% (test) |
| 0.50 | 2.7% (test) | 1.0% (test) | 1.6% (test) |
| 1.00 | -0.6% (test) | 0.0% (test) | 0.5% (test) |

关键发现：
- **低预算下优势更显著**：0.25 预算时，HbBoPs 比 HDBO 好 ~35%，比 TRIPLE-SH 好 ~24%
- **Anytime 性能最优**：在整个优化过程中持续领先
- 结构感知编码（分别编码 instruction 和 exemplar）显著优于整体编码

### 消融实验

- 移除深度核 → 性能下降（vanilla GP 无法处理高维嵌入）
- 移除 Hyperband（改为全保真度）→ 查询效率大幅下降
- 更换编码器（BERT→其他模型）→ 方法对编码器选择具有鲁棒性

## 亮点与洞察

1. **同时解决两个效率瓶颈**：首次在 prompt 选择中实现样本高效（代理模型少评估 prompt）和查询高效（多保真度少调用 LLM）的统一。

2. **结构感知表示**的关键性：将 prompt 视为 instruction + exemplar 的组合结构，而非无差异文本块，显著提升了代理模型的预测能力。

3. **实用性强**：方法完全在线训练，不需要预训练代理模型；适用于任何黑盒 LLM。

4. **Hyperband 优于 SH**：多 bracket 策略有效对冲了"探索 vs 利用"的不确定性。

## 局限性

- 搜索空间固定为 250 个候选 prompt，未测试更大规模搜索空间
- 仅考虑 5-shot 设定，未探索不同 shot 数的影响
- 代码未公开，可复现性有待验证
- 未与基于 LLM 的 prompt 生成方法直接比较

## 相关工作

- **Prompt 优化/选择**：APE (Zhou 2023), DSPy/MIPROv2 (Opsahl-Ong 2024), EASE (Wu 2024), TRIPLE (Shi 2024)
- **贝叶斯优化**：深度核 GP (Wilson 2016), 高维 BO (Hvarfner 2024)
- **多保真度优化**：Hyperband (Li 2018), BOHB (Falkner 2018)

## 评分

⭐⭐⭐⭐ — 方法设计优雅，将 HPO 领域的成熟技术（Hyperband + BO + 深度核 GP）有机融合到 prompt 选择问题中，实验充分。核心贡献在于框架整合而非单点突破。

<!-- RELATED:START -->

## 相关论文

- [Enabling Fine-Grained Operating Points for Black-Box LLMs](../../ICLR2026/llm_evaluation/enabling_fine-grained_operating_points_for_black-box_llms.md)
- [Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning](../../NeurIPS2025/llm_evaluation/cost-sensitive_freeze-thaw_bayesian_optimization_for_efficient_hyperparameter_tu.md)
- [Prompt and Parameter Co-Optimization for Large Language Models](../../ICLR2026/llm_evaluation/prompt_and_parameter_co-optimization_for_large_language_models.md)
- [IBDR: Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness](promoting_ensemble_diversity_with_interactive_bayesian_distributional_robustness.md)
- [Improving Generalization with Flat Hilbert Bayesian Inference](improving_generalization_with_flat_hilbert_bayesian_inference.md)

<!-- RELATED:END -->
