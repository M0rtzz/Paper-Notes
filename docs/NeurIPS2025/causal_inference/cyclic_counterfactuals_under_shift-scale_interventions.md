---
title: >-
  [论文解读] Cyclic Counterfactuals under Shift–Scale Interventions
description: >-
  [NeurIPS 2025][因果推断] 本文在循环（非DAG）结构因果模型中建立了shift-scale软干预下反事实推理的理论框架，证明了全局收缩条件保证循环SCM的唯一可解性，并推导出反事实分布的sub-Gaussian集中不等式。
tags:
  - NeurIPS 2025
  - 因果推断
  - 循环因果模型
  - 反事实推理
  - 软干预
  - 收缩映射
---

# Cyclic Counterfactuals under Shift–Scale Interventions

**会议**: NeurIPS 2025  
**arXiv**: [2510.25005](https://arxiv.org/abs/2510.25005)  
**代码**: 无  
**领域**: 因果推断  
**关键词**: 因果推断, 循环因果模型, 反事实推理, 软干预, 收缩映射

## 一句话总结

本文在循环（非DAG）结构因果模型中建立了shift-scale软干预下反事实推理的理论框架，证明了全局收缩条件保证循环SCM的唯一可解性，并推导出反事实分布的sub-Gaussian集中不等式。

## 研究背景与动机

**领域现状**：反事实推理是因果推断的核心问题之一。绝大多数反事实推理框架（如Pearl的do-calculus、twin network）假设因果结构是有向无环图（DAG），即变量之间没有反馈循环。

**现有痛点**：然而现实世界大量系统存在反馈循环——基因调控网络中的正/负反馈回路、宏观经济模型中消费与收入的相互影响、生态系统中捕食者与被捕食者的关系。这些系统的因果结构包含环路，违反了DAG假设。在循环SCM中，结构方程可能没有唯一解（甚至没有解），使得反事实推理的定义本身就成问题。此外，现有理论主要考虑硬干预（do-intervention），即将变量强制设为固定值，但实际政策往往是软干预——如"给每个人增加20%药物剂量"或"降低每个学生5人的班级规模"——这些取决于个体原始值的政策无法用$do(X=x)$表达。

**核心矛盾**：（1）循环系统中反事实推理缺乏唯一性保证；（2）soft intervention（shift-scale）比hard intervention更具表达力，但其理论基础不完善。

**本文目标** 在循环因果模型中，shift-scale干预下的反事实分布何时存在且唯一？这类干预是否具有代数稳定性（可组合性）？反事实结果的分布有多集中？

**切入角度**：借用动力系统理论中的收缩映射原理（Banach不动点定理），为循环SCM的唯一可解性提供充分条件，然后证明shift-scale干预保持收缩性。

**核心 idea**：用全局收缩条件统一处理循环SCM中soft intervention的反事实推理，将唯一可解性从DAG推广到满足收缩条件的循环图。

## 方法详解

### 整体框架

论文的理论架构分为四层：（1）证明收缩SCM是simple SCM（对所有变量子集唯一可解）；（2）证明有界的shift-scale干预保持收缩性；（3）证明此类干预的组合封闭性；（4）在Gaussian噪声+Lipschitz正则条件下推导反事实分布的集中不等式。

### 关键设计

1. **全局收缩条件 → 唯一可解性（Theorem 1）**:

    - 功能：为循环SCM的唯一可解性提供可验证的充分条件
    - 核心思路：若SCM的结构方程$f: \mathcal{X} \times \mathcal{E} \to \mathcal{X}$满足全局$\kappa$-收缩（$\kappa < 1$），即$\|f(x,e) - f(y,e)\|_p \leq \kappa \|x - y\|_p$，则对任意变量子集$\mathcal{O}$，存在唯一的不动点解。证明利用Banach不动点定理：对固定的外生变量$e$和非$\mathcal{O}$变量，$f_\mathcal{O}$在完备度量空间$\mathcal{X}_\mathcal{O}$上是$\kappa$-收缩，因此存在唯一不动点。Picard迭代的逐点收敛保证了解映射的可测性
    - 设计动机：Bongers et al. (2021)的simple SCM封闭性结果假设simplicity，但不给出simplicity的充分条件。本文补上了这个关键环节

2. **Shift-scale干预保持收缩性（Theorem 2）**:

    - 功能：证明干预后的twin SCM仍然唯一可解，从而反事实分布well-defined
    - 核心思路：shift-scale干预将$X_j \leftarrow a_j f_j(x,e) + b_j$，当$|a_j| \leq 1$时，干预后的映射$\tilde{g}$仍然是$\kappa$-收缩。证明关键在于對角缩放矩阵$D = \text{diag}(a_j)$满足$\|Du\|_p \leq \|u\|_p$（因为$|a_j| \leq 1$），所以$\|\tilde{g}(u,e) - \tilde{g}(v,e)\|_p \leq \|f(u,e) - f(v,e)\|_p \leq \kappa\|u-v\|_p$
    - 设计动机：确保干预后的模型仍然well-posed，是建立反事实推理理论的必要前提

3. **组合封闭性与集中不等式（Proposition 1 & 2）**:

    - 功能：证明多次shift-scale干预可等价为一次，且反事实分布集中于均值附近
    - 核心思路：（Prop 1）多次shift-scale组合等价于$a_j^{\text{comp}} = \prod_r a_j^{(r)}$和对应的仿射漂移，由于$|a_j^{(r)}| \leq 1$所以$|a_j^{\text{comp}}| \leq 1$，收缩性保持。（Prop 2）当外生噪声为Gaussian时，解映射$\Phi$是$L = \frac{\sqrt{2}}{1-\kappa}$-Lipschitz的，利用Gaussian Lipschitz集中不等式得到$\mathbb{P}(h(\mathbf{X},\mathbf{X}') - \mathbb{E}[h] \geq t) \leq \exp(-\frac{t^2}{4(1-\kappa)^{-2}\sigma^2})$
    - 设计动机：组合封闭性使得序贯干预分析在代数上稳定；集中不等式给出反事实结果的不确定性定量界

### 损失函数 / 训练策略

本文是纯理论工作，无训练过程。

## 实验关键数据

### 主实验

本文通过一个消费-收入循环经济学模型展示理论的应用：

| 量 | 观测分布 | 干预后分布 | 变化 |
|---------|-----------|-----------|------|
| $\mathbb{E}[C]$ | 1.5625 | 2.024 | +29% |
| $\mathbb{E}[I]$ | 1.125 | 2.048 | +82% |
| $\text{Corr}(C,I)$ | 0.75 | 0.69 | -8% |
| 收缩常数$\kappa$ | 0.6403 | 0.5936 | 保持<1 |

干预为：对收入$I$进行$\alpha=0.8$的缩放 + $\beta=1.0$的平移（模拟财政改革：抑制消费对收入的反馈效应，同时提供固定收入补贴）。

### 消融实验

| 条件 | 结果 | 说明 |
|------|------|------|
| $\|a\| \leq 1$ | 收缩性保持 | 定理保证 |
| $\|a\| > 1$但$\kappa_{\max} < 1$ | 仍可解 | Remark 1的扩展条件 |
| $\kappa_{\max} \geq 1$ | 不保证唯一 | 需要额外分析 |
| Gaussian噪声 | sub-Gaussian集中 | Proposition 2 |
| 重尾噪声 | 仅多项式集中 | 论文未覆盖 |

### 关键发现

- 系统矩阵$A$的谱范数直接决定收缩性：$\|A\|_2 < 1$是充分条件
- Shift-scale干预通过缩放对角矩阵保持收缩性，且收缩常数不增大
- 反事实分布的集中度随$\kappa \to 1$急剧恶化（集中参数$\propto (1-\kappa)^{-2}$）
- 在线性循环模型中，反事实响应映射是仿射的，可得到闭式解

## 亮点与洞察

- **收缩映射=万能钥匙**：Banach不动点定理在循环因果模型中的应用非常优雅——将环路的"可解性"问题转化为"收缩性"问题，一把钥匙打开了唯一性、可测性、twin SCM封闭性三道锁。这个框架可以推广到任何保持收缩性的干预类型
- **soft intervention的理论基础**：shift-scale干预严格推广了hard intervention（$a=0, b=\xi$是特例），为更灵活的因果询问（如"所有人剂量增加20%"）提供了严格的数学基础
- **集中不等式的实用性**：sub-Gaussian tail bound给出了反事实预测的置信区间，在医疗决策等高风险场景下尤为有用

## 局限与展望

- **全局收缩条件过强**：实际系统可能只在局部满足收缩性，但本文要求全局成立。需要探索局部收缩或分段收缩的理论
- **scale因子限制$|a_j| \leq 1$**：放大干预（$|a_j| > 1$）需要验证$\kappa_{\max} < 1$的额外条件。随机策略或非线性干预未覆盖
- **Gaussian噪声假设**：集中不等式依赖Gaussian噪声，对重尾分布只能得到多项式集中
- **缺乏实际数据验证**：仅有一个2变量线性经济学玩具模型的演示，未在真实生物系统或基因调控网络上验证
- **未涉及因果发现**：假设因果图已知，如何从数据中学习循环因果结构是独立的open问题

## 相关工作与启发

- **vs Bongers et al. (2021)**：Bongers给出了simple SCM的封闭性结果（对do、边缘化、twin封闭），但不给简单性的充分条件。本文用收缩条件补上了这个gap，并扩展到soft intervention
- **vs Rothenhäusler et al. (2015)**：他们用shift干预来学习因果循环图，但未处理反事实推理和唯一可解性
- **vs Lorch et al. (2024)**：他们用平稳扩散过程建模因果系统中的shift-scale干预，但侧重连续时间，不涉及反事实

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地处理循环SCM中soft intervention的反事实推理
- 实验充分度: ⭐⭐ 仅有一个2变量玩具模型，缺乏实际应用验证
- 写作质量: ⭐⭐⭐⭐ 定理证明清晰严谨，但符号较重
- 价值: ⭐⭐⭐ 填补了循环因果模型中软干预反事实推理的理论空白

<!-- RELATED:START -->

## 相关论文

- [Characterization and Learning of Causal Graphs from Hard Interventions](characterization_and_learning_of_causal_graphs_from_hard_interventions.md)
- [Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)
- [Causal Abstraction Inference under Lossy Representations](../../ICML2025/causal_inference/causal_abstraction_inference_under_lossy_representations.md)
- [RATE: Causal Explainability of Reward Models with Imperfect Counterfactuals](../../ICML2025/causal_inference/rate_causal_explainability_of_reward_models_with_imperfect_counterfactuals.md)
- [Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](../../ICLR2026/causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)

<!-- RELATED:END -->
