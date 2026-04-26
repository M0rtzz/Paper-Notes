---
title: >-
  [论文解读] SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization
description: >-
  [AAAI 2026][医学图像][LLM 对齐] 提出 Self-Priority Alignment（SPA），一种全无监督框架，通过字典序优化实现"可信赖优先于有用性"的严格优先级对齐——模型自生成多样响应、自评估、自改进，经双准则去噪构建偏好对，用不确定性加权 SimPO 损失微调，在多个安全基准上同时提升安全性和有用性。
tags:
  - AAAI 2026
  - 医学图像
  - LLM 对齐
  - 优先级对齐
  - 字典序优化
  - 无监督
  - 偏好学习
  - 安全性
---

# SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization

**会议**: AAAI 2026  
**arXiv**: [2511.06222](https://arxiv.org/abs/2511.06222)  
**代码**: 无  
**领域**: LLM 对齐 / AI 安全  
**关键词**: LLM 对齐, 优先级对齐, 字典序优化, 无监督, 偏好学习, 安全性

## 一句话总结

提出 Self-Priority Alignment（SPA），一种全无监督框架，通过字典序优化实现"可信赖优先于有用性"的严格优先级对齐——模型自生成多样响应、自评估、自改进，经双准则去噪构建偏好对，用不确定性加权 SimPO 损失微调，在多个安全基准上同时提升安全性和有用性。

## 研究背景与动机

- **高风险场景中的核心矛盾**：在医疗、法律、自伤等场景中，LLMs 必须同时可信赖（harmless/honest）和有用（helpful），但两者经常冲突
    - 示例："如果我有自伤想法怎么办？"——必须优先安全，但通用拒绝可能让用户感到被忽视
- **现有多目标对齐方法的三大局限**：
  1. **上下文无关权重**：静态或启发式权重无法适应动态用户意图和风险状况
  2. **无安全感知优化**：妥协式平衡可能在追求有用性时削弱安全性
  3. **数据稀缺**：缺乏高质量标注数据捕获可信赖与有用的真实权衡
- **核心思想**：引入**优先级对齐**新范式——主要目标（如 harmlessness）必须先满足，才能优化次要目标（如 helpfulness）

## 方法详解

### 理论基础：字典序优化

将优先级对齐形式化为字典序优化问题：按严格优先级顺序依次优化多个目标。

**挑战**：LLM 的高维非凸参数空间使得传统字典序方法不可行（无法先完全优化 $G_a$ 再优化 $G_b$）。

**解决方案**：将 Pareto 前沿枚举与偏好优化（PO）结合——偏好对隐式编码了 Pareto 支配关系：

- 若 $G_a(y) \geq G_a(y^-)$ 且 $G_b(y) > G_b(y^-)$，则 $y$ Pareto 支配 $y^-$
- 收集大量此类偏好对隐式刻画了 $G_a$, $G_b$ 之间的 Pareto 前沿
- 通过 DPO/SimPO 微调使模型内化这种优先级结构

### SPA 框架：三大组件

#### 1. 多样采样 + 自我改进

**多样采样**：对每个 prompt $x_j$，生成 $n$ 个多样候选响应
- 高温采样（$\tau > 1$）
- 系统 prompt 变体（增加多样性）

**自评估**：每个响应在两个维度上自评分
- $s_{a,j}^{(i)} = S_a(x_j, y_j^{(i)})$（主目标，如 harmlessness）
- $s_{b,j}^{(i)} = S_b(x_j, y_j^{(i)})$（次目标，如 helpfulness）
- 评分函数基于 AI 宪法 $\mathcal{C}$（定义 HHH 原则）

**自改进**：基于所有候选及其分数，生成一个改进响应 $\tilde{y}_j$

$$\tilde{y}_j \sim \pi_\theta(\cdot | x_j, \{y_j^{(i)}, s_{a,j}^{(i)}, s_{b,j}^{(i)}\}_{i=1}^n, \mathcal{C})$$

#### 2. 双准则去噪

**动机**：弱模型的自评估可能有偏差、不一致、噪声大。

**一致性驱动去噪**：仅保留改进响应在两个维度上都严格优于所有候选的样本

$$\mathcal{Y}_{\text{perf}} = \{(y, s_a, s_b) \in \mathcal{Y} \mid \tilde{s}_a > \max_i s_{a}^{(i)} \text{ and } \tilde{s}_b > \max_i s_{b}^{(i)}\}$$

**信息性驱动去噪**：基于 RV 系数分析，高方差样本会降低弱-强模型对齐度。过滤评分方差过大的样本：

$$\mathcal{Y}_{\text{final}} = \{(y, s_a, s_b) \in \mathcal{Y}_{\text{perf}} \mid 0 < \det(\Sigma_j) \leq \tau\}$$

其中 $\Sigma_j$ 为双维度评分的协方差矩阵。

**实证支持**：当纳入超过 32% 的样本时（按方差排序），弱-强模型的 RV 系数急剧下降。

#### 3. 偏好数据集构建与优先级优化

**偏好对构建**（字典序）：

$$(G_a(y), G_b(y)) >_{\text{lex}} (G_a(y^-), G_b(y^-))$$

即 $G_a(y) > G_a(y^-)$，或 $G_a(y) = G_a(y^-)$ 且 $G_b(y) > G_b(y^-)$。
加上边际约束 $\Delta(y, y^-) \geq \delta$ 确保偏好差异有意义。

### 损失函数：不确定性引导的 SimPO

$$\mathcal{L}_{\text{SPA}}(\theta) = -\mathbb{E}_{(x,y,y^-) \in \mathcal{D}_{\text{pref}}} \left[ w_i \cdot \log \sigma\left(\frac{\beta}{|y|} \log \pi_\theta(y|x) - \frac{\beta}{|y^-|} \log \pi_\theta(y^-|x) - \gamma\right) \right]$$

- 基于 SimPO（长度归一化，避免模型偏好冗长输出）
- **不确定性加权** $w_i = (\Delta_i / \bar{\Delta})^\alpha$：分数差距大的对施加更强梯度

## 实验

### 设置

- **模型**：Llama-3.1-8B-Instruct、Mistral-7B-Instruct
- **数据集**：SafeRLHF（harmlessness+helpfulness）、WildGuard（同）、HoneSet（honesty+helpfulness）
- **训练规模**：300-400 prompts（非常轻量）
- **评估**：LLM-as-Judge（GPT-4o）+ 人类评估
- **基线**：Reward Soups（多权重比例）、Self-Criticism、SFT、SPADPO、SPASimPO

### 主实验结果（分数评估）

| 方法 | SafeRLHF Harm.↑ | SafeRLHF Help.↑ | WildGuard Harm.↑ | WildGuard Help.↑ | HoneSet Hon.↑ | HoneSet Help.↑ |
|------|---------|---------|---------|---------|--------|--------|
| Llama Vanilla | 9.62 | 5.23 | 8.22 | 6.09 | 6.30 | 7.75 |
| Llama SFT | 9.68 | 5.57 | 9.79 | 3.20 | 6.11 | 7.66 |
| **Llama SPA** | **9.90** | **7.14** | **8.85** | **6.22** | **7.75** | **7.83** |
| Mistral Vanilla | 8.83 | 7.53 | 6.83 | 7.15 | 5.81 | 7.62 |
| **Mistral SPA** | **9.76** | **8.39** | **7.27** | **7.44** | **7.18** | **7.82** |

关键发现：
- SPA 在**所有指标**上优于 Vanilla 和 SFT，尤其在 Mistral 上全面最优
- Helpfulness 提升显著（Llama: 5.23→7.14，Mistral: 7.53→8.39）而不牺牲安全性
- **成对比较**中 SPA 胜率高达 86%（HoneSet helpfulness）

### 与多目标基线对比

| 方法 | Harm. | Help. | HH₅ | HH₁₀ | HH₂₀ |
|------|-------|-------|------|------|------|
| Self-Criticism | 9.65 | **7.68** | 9.32 | 9.47 | 9.56 |
| RS 9:1 | 9.90 | 6.17 | 9.28 | 9.56 | 9.72 |
| **SPA** | **9.90** | 7.14 | **9.44** | **9.65** | **9.77** |

SPA 在加权综合指标 HH_λ（λ=5,10,20）上全面领先，当 harmlessness 优先级越高，SPA 优势越明显。

### 通用能力保持

- MTBench：SPA 在 4 种配置中的 3 种上优于 Vanilla（最大提升 +2.52%）
- MMLU：波动 ±2% 以内，表明对齐改进**不以牺牲通用能力为代价**

### 泛化能力

训练在 SafeRLHF 上的 SPA 直接迁移到未见数据集：
- JailbreakTrigger：Harm. 9.80 / Help. 6.35（远超 Vanilla 和 SFT）
- WildGuard：Harm. 9.29（最强之一）

### 消融实验

- **去噪组件**：去掉去噪后 harmlessness 和 helpfulness 均下降超过 0.1
- **迭代次数**：第二轮迭代有提升（特别是复用相同 prompts 效果更好），第三轮收益递减
- **人类评估**：GPT-4o 作为 judge 与人类标注的一致率：harmlessness 91-94%，helpfulness 89-92%

## 亮点与洞察

1. **优先级对齐是对齐研究的范式创新**：不再寻求"平衡"而是建立"优先级"，更符合高风险场景需求
2. **完全无监督**：无需人类标注数据，模型自生成、自评估、自改进——可扩展性极强
3. **字典序偏好对隐式编码 Pareto 支配**：理论上优雅地近似了不可行的字典序优化
4. **双准则去噪**有效缓解弱模型自评估的噪声，RV 系数分析提供了理论基础
5. **训练数据极少**（仅 300 prompts），说明方法的高效性和实用性

## 局限性

- 自评估能力受限于模型本身质量，弱模型可能产生系统性偏差
- 仅在 8B 级别模型上验证，未测试更大或更小的模型
- 高风险场景的定义和边界仍需人类判断，无法完全自动化
- 字典序优化的理论保证在非凸设置下是近似的
- 评估主要依赖 LLM-as-Judge，可能引入评估偏差
- 仅 300 个训练 prompts 的多样性是否足以覆盖广泛场景值得商榷

## 相关工作

- **对齐算法**：PPO/RLHF、DPO、SimPO、KTO、IPO 等偏好学习方法
- **多目标对齐**：Reward Soups（线性组合模型权重）、MetaAligner、RiC（Rewards-in-Context）
- **自对齐**：Self-Criticism（HHH 原则）、Meta-self-alignment
- **安全对齐**：SafeDPO、TISDPO

## 评分 ⭐⭐⭐⭐

方法理论上优雅（字典序优化 + Pareto 前沿），实践上高效（无监督 + 少数据），效果显著（安全性和有用性同步提升）。双准则去噪和不确定性加权损失是有价值的技术贡献。主要遗憾是实验仅在 8B 模型上，且高风险场景的覆盖面有限。对 LLM 安全对齐领域是重要推进。

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Self-supervised Multiplex Consensus Mamba for General Image Fusion](self-supervised_multiplex_consensus_mamba_for_general_image_fusion.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)
- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](../../ACL2026/medical_imaging/principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[AAAI 2026\] TAlignDiff: Automatic Tooth Alignment assisted by Diffusion-based Transformation Learning](taligndiff_automatic_tooth_alignment_assisted_by_diffusion-based_transformation_.md)

<!-- RELATED:END -->
