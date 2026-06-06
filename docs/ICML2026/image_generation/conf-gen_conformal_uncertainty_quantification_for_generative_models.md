---
title: >-
  [论文解读] Conf-Gen: Conformal Uncertainty Quantification for Generative Models
description: >-
  [ICML 2026][图像生成][保形预测] 提出 Conf-Gen 框架，将保形风险控制（CRC）扩展到生成任务，通过参数化选择函数和可容许性函数为 LLM 问答、图像生成、对话系统、AI Agent 等任务提供形式化的不确定性保证，同时放松了 CRC 的单调性等理论假设。
tags:
  - "ICML 2026"
  - "图像生成"
  - "保形预测"
  - "不确定性量化"
  - "生成模型"
  - "保形风险控制"
  - "选择函数"
---

# Conf-Gen: Conformal Uncertainty Quantification for Generative Models

**会议**: ICML 2026  
**arXiv**: [2605.28920](https://arxiv.org/abs/2605.28920)  
**代码**: https://github.com/layer6ai-labs/conf-gen (有)  
**领域**: 不确定性量化 / 生成模型  
**关键词**: 保形预测, 不确定性量化, 生成模型, 保形风险控制, 选择函数  

## 一句话总结
提出 Conf-Gen 框架，将保形风险控制（CRC）扩展到生成任务，通过参数化选择函数和可容许性函数为 LLM 问答、图像生成、对话系统、AI Agent 等任务提供形式化的不确定性保证，同时放松了 CRC 的单调性等理论假设。

## 研究背景与动机

**领域现状**：保形预测（Conformal Prediction, CP）及其扩展保形风险控制（CRC）是监督学习中不确定性量化的主流框架，能为预测集合提供分布无关的覆盖率保证。然而，当前 AI 的重大突破主要由无监督生成模型（LLM、扩散模型、Agent 系统等）驱动，这些模型并不直接适用于传统的 CP/CRC 框架。

**现有痛点**：已有将保形方法应用于生成模型的工作大多是高度任务特定的。例如 Quach et al. (2024) 和 Kladny et al. (2025) 将 CRC 用于 LLM 问答，但它们对生成的答案集合进行多轮独立筛选（如似然过滤、去重过滤），每个筛选步骤需要在不同数据子集上独立校准，导致最终产出的保形集合过大、实用性差。此外，CRC 要求效用函数几乎处处单调递增，这在许多生成任务中并不成立。

**核心矛盾**：CRC 的理论框架仅支持集合输出、要求可调用的效用函数、需要严格的单调性假设，而生成任务的输出可以是序列等更复杂结构，可容许性可能由人工评估定义（不可调用），且单调性可能仅在条件期望意义下成立。

**本文目标**：设计一个统一框架，将保形方法推广到各类生成任务，同时放松 CRC 的理论假设使其更广泛适用。

**切入角度**：作者观察到 CRC 中的几个限制（输出必须是集合、效用函数必须可调用、单调性必须几乎处处成立）并非理论证明的本质要求，而是可以被系统性放松的。

**核心 idea**：引入参数化选择函数 $\mathbf{C}_\lambda$ 处理输入与生成序列，配合可容许性函数 $A$，将 CRC 的校准机制推广为"找到最小 $\lambda$ 使校准集上平均可容许性达标"，从而为任意生成任务提供形式化保证。

## 方法详解

### 整体框架
Conf-Gen 的输入是一个三元组 $\mathbf{G} = (X, \mathbf{Y}, Y_{\text{GT}})$，其中 $X$ 是条件输入（如问题），$\mathbf{Y} = (Y_1, \dots, Y_T)$ 是生成模型产出的一系列输出（如多个回答），$Y_{\text{GT}}$ 是可选的 ground truth。框架包含三个核心组件：选择函数族 $\mathbf{C}_\lambda$、可容许性函数 $A$ 和校准数据集 $\mathbf{D}_{:n}$。校准阶段在标注数据上找到最小的 $\hat{\lambda}$ 使平均可容许性超过阈值 $\frac{n+1}{n}\gamma$；推理阶段用 $\mathbf{C}_{\hat{\lambda}}$ 处理新输入，输出的结构（集合或序列）可以获得形式化的可容许性下界保证 $\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \geq \gamma$。

### 关键设计

1. **参数化选择函数族 $\mathbf{C}_\lambda$**:

    - 功能：将输入 $X$ 和生成序列 $\mathbf{Y}$ 映射到输出空间 $\mathcal{S}$（可以是集合、序列或单个元素），$\lambda$ 越大输出越保守
    - 核心思路：定义基于分数的选择策略，例如 $\mathbf{C}_\lambda(x, \mathbf{y}) = \mathbf{y}_{:\tau(x,\mathbf{y},\lambda)}$，其中 $\tau(x,\mathbf{y},\lambda) = \inf\{t : \texttt{accum}(S_1^\uparrow, \dots, S_t^\uparrow) > \lambda\} \wedge |\mathbf{y}|$ 是基于累积分数的停止时刻。由于选择函数作为 $\lambda$ 的函数仅有有限个像，即使 $\Lambda$ 是无穷集合也可以高效校准
    - 设计动机：与 CRC 中 $\mathcal{C}_\lambda$ 只能输出集合不同，$\mathbf{C}_\lambda$ 允许输出序列或单元素，且以生成序列 $\mathbf{Y}$ 作为额外输入，天然适配生成任务。有限像性质使得即使可容许性由人工评估（不可调用）定义，校准仍然可行

2. **可容许性函数与实例级分解**:

    - 功能：度量选择函数输出的质量，$A(x, \mathbf{C}_\lambda(x, \mathbf{y}), y_{\text{GT}}) \in [0, \infty]$，值越大越好
    - 核心思路：将全局可容许性分解为 $A(x, \mathbf{y}, y_{\text{GT}}) = \texttt{agg}(A_1', \dots, A_T')$，其中 $A_t' = A'(x, y_t, y_{\text{GT}})$ 是单个生成元素的实例级可容许性，$\texttt{agg}$ 可以是 max（至少一个好答案）或 min（所有答案都好）。这样只需评估 $\sum_{i=1}^n T_i$ 次 $A'$，而非为每种 $\mathbf{C}_\lambda$ 重新评估
    - 设计动机：将可容许性评估与选择函数解耦，更换选择函数不需要重新收集标注，大幅降低校准成本

3. **$\gamma$-sensible 条件与放松的单调性假设**:

    - 功能：为保形保证 $\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \geq \gamma$ 提供充分条件
    - 核心思路：CRC 要求效用函数 $U^{(i)}(\lambda)$ 几乎处处关于 $\lambda$ 单调递增，Conf-Gen 将其放松为条件期望单调，即 $\lambda \mapsto \mathbb{E}[A^{(n+1)}(\lambda) \mid \lambda', \lambda'']$ 单调递增即可。此外还放松了右连续性假设，并给出了更一般的上界 $\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \leq \gamma + \frac{a_{\max}}{n+1} + \mathbb{E}[H]$
    - 设计动机：在图像去记忆化等任务中，更多修改提示词的图像偶尔反而更"记忆化"，导致几乎处处单调性不成立，但条件期望单调性仍然合理

## 实验关键数据

### 主实验

论文在 5 个任务上验证了 Conf-Gen 的有效性：

| 任务 | 数据集 | 模型 | 评估方式 | 保证内容 |
|------|--------|------|----------|----------|
| 开放域问答 | TriviaQA | LLaMA-13B | 自动 (LLM judge) | 输出序列含正确答案 |
| 非记忆化图像生成 | Webster (2023) | Stable Diffusion v1.5 | 人工评估 (10人) | 生成图像未记忆训练数据 |
| 对话式 AI | ClariQ | LLM | 二值标签 | 已问足够多澄清问题 |
| Agent 网页任务 | WebVoyager | LMM Agent | LLM judge | 轨迹序列含成功完成方案 |
| 随机森林 | Click Prediction | RF (100棵树) | 准确率 | 子集含 ≥k 棵正确树 |

| 任务 | Conf-Gen 优势 | 具体表现 |
|------|--------------|----------|
| 问答 (vs Quach et al.) | 更短的输出序列 | 在多数 $\gamma$ 值下序列长度更短，且 LLM 调用次数更少（得益于 partial generation） |
| Agent 任务 | 显著提升成功率 | 单次尝试 ~60% → 平均不到 2 次尝试即保证 >65%，允许更多尝试可达 ~80% |
| 随机森林 | 有效剪枝 | 选中树数量在大范围 $\gamma$ 下低于 59 棵（$2k-1$），保证多数投票正确 |

### 关键发现
- 与 Quach et al. (2024) 相比，Conf-Gen 避免了多轮独立校准带来的数据浪费和集合膨胀，因此在 TriviaQA 上产出更紧凑的保形集合
- 在图像去记忆化任务中，$\lambda \mapsto A(\lambda)$ 并非几乎处处单调，但 $\gamma$-sensible 条件在实践中仍然成立，保形保证依然得到满足
- 对话任务中，$\gamma$ 增大时所需澄清轮次增加但不会退化为总是最大轮数，说明 Conf-Gen 输出非平凡（不是总选最保守方案）

## 亮点与洞察
- **有限像技巧使人工评估校准可行**：选择函数作为 $\lambda$ 的函数仅有有限个像，因此即使可容许性由人工标注定义（不可调用），也只需对有限个输出做评估即可完成校准。这一设计优雅地绕过了"无法对连续 $\lambda$ 做人工标注"的障碍
- **框架的统一性极强**：保形事实性（conformal factuality）、保形摘要（conformal summarization）、保形 Agent 错误归因等此前独立的方法都可以作为 Conf-Gen 的特例恢复，Table 1 清晰总结了各种选择函数/聚合函数的兼容组合
- **partial generation 节省推理成本**：基于累积分数的停止策略允许在推理时只生成 $\tau$ 个输出就停止，避免浪费生成剩余 $T - \tau$ 个元素的计算

## 局限与展望
- 校准数据集仍需覆盖足够多样的输入场景，对分布偏移的鲁棒性未深入探讨
- $\gamma$-sensible 的条件期望单调性虽然比 CRC 的假设弱，但在某些场景下仍难以验证
- 保形保证是边际期望意义上的（$\mathbb{E}[A^{(n+1)}(\hat{\lambda})] \geq \gamma$），对单个测试样本不提供逐例保证
- 未来方向包括优化各任务中的分数函数设计、探索更多 Conf-Gen 的新应用场景

## 相关工作与启发
- **vs Quach et al. (2024) / Kladny et al. (2025)**：它们对 LLM 答案集合做多轮独立筛选，每轮需独立校准，导致保形集合过大。Conf-Gen 用单一 $\lambda$ 统一校准，产出更紧凑的集合
- **vs Mohri & Hashimoto (2024) 保形事实性**：仅处理声明级别的事实性筛选（$\texttt{agg} = \min$），是 Conf-Gen 的一个特例（Table 1 第三行）
- **vs Feng et al. (2026) 保形 Agent 错误归因**：仅处理连续子序列包含首次错误的场景，同样可归入 Conf-Gen 框架

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 CRC 系统推广到生成任务的统一框架，理论假设放松有实质创新
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个不同任务覆盖 LLM/图像/对话/Agent/传统 ML，含人工评估
- 写作质量: ⭐⭐⭐⭐⭐ 论述清晰，从 CP→CRC→Conf-Gen 的推导链条严密，running example 贯穿全文
- 价值: ⭐⭐⭐⭐ 为生成模型不确定性量化提供了统一的保形框架，应用前景广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Conformal Reliability: A New Evaluation Metric for Conditional Generation](conformal_reliability_a_new_evaluation_metric_for_conditional_generation.md)
- [\[ICLR 2026\] Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](../../ICLR2026/image_generation/unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)
- [\[ACL 2025\] D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Models](../../ACL2025/image_generation/d-gen_automatic_distractor_generation_and_evaluation_for_reliable_assessment_of_.md)
- [\[ICML 2026\] Generative Visual Code Mobile World Models](generative_visual_code_mobile_world_models.md)
- [\[ICML 2026\] Threshold-Guided Optimization for Visual Generative Models](threshold-guided_optimization_for_visual_generative_models.md)

</div>

<!-- RELATED:END -->
