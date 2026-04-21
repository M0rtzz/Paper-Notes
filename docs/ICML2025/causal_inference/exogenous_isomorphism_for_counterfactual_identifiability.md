---
title: >-
  [论文解读] Exogenous Isomorphism for Counterfactual Identifiability
description: >-
  [ICML2025][反事实可辨识性] 提出**外生同构（Exogenous Isomorphism, EI）**概念，证明 $\sim_{\mathrm{EI}}$-identifiability 蕴含 $\sim_{\mathcal{L}_3}$-identifiability（完整反事实层可辨识性），并在双射SCM和三角单调SCM两类特殊模型上给出实现EI的充分条件，统一并推广了已有反事实可辨识性理论。
tags:
  - ICML2025
  - 反事实可辨识性
  - Pearl因果层级
  - 外生同构
  - 结构因果模型
  - 三角单调SCM
---

# Exogenous Isomorphism for Counterfactual Identifiability

**会议**: ICML2025  
**arXiv**: [2505.02212](https://arxiv.org/abs/2505.02212)  
**代码**: [cyisk/tmscm](https://github.com/cyisk/tmscm)  
**领域**: 因果推断 (Causal Inference)  
**关键词**: 反事实可辨识性, Pearl因果层级, 外生同构, 结构因果模型, 三角单调SCM

## 一句话总结

提出**外生同构（Exogenous Isomorphism, EI）**概念，证明 $\sim_{\mathrm{EI}}$-identifiability 蕴含 $\sim_{\mathcal{L}_3}$-identifiability（完整反事实层可辨识性），并在双射SCM和三角单调SCM两类特殊模型上给出实现EI的充分条件，统一并推广了已有反事实可辨识性理论。

## 研究背景与动机

- **反事实推理**是Pearl因果层级（PCH）的最高层 $\mathcal{L}_3$，编码了SCM的全部因果信息
- 已有工作要么只识别特定反事实效应（结构约束），要么只识别反事实结果（机制约束），缺乏对**整个反事实层**的统一识别框架
- $\mathcal{L}_3$-identifiability 要求满足假设的所有SCM对**任意**反事实陈述给出一致答案，是PCH框架内因果可辨识性的终极目标
- 直接从PCH定义出发处理 $\sim_{\mathcal{L}_3}$ 非常困难，而完全恢复外生变量（$=$-identifiability）的假设又过强
- **核心动机**：寻找比"完全恢复模型"弱、但仍蕴含完整反事实一致性的中间概念

## 方法详解

### 框架总览

1. 以模型可辨识性视角统一因果可辨识性问题
2. 提出外生同构（EI）作为连接模型等价与反事实一致性的桥梁
3. 在双射SCM（BSCM）和三角单调SCM（TM-SCM）两类模型上分别给出EI的充分条件
4. 用神经TM-SCM实现实际反事实推断

### 核心概念：外生同构

**定义**：两个递归SCM $\mathcal{M}^{(1)}$ 和 $\mathcal{M}^{(2)}$ 外生同构（$\sim_{\mathrm{EI}}$），如果存在共享因果序和分量双射 $\mathbf{h} = (h_i)_{i \in \mathcal{I}}$，满足：

- **分量双射**：每个 $h_i: \Omega_{U_i}^{(1)} \to \Omega_{U_i}^{(2)}$ 是双射
- **外生分布同构**：$P_{\mathbf{U}}^{(2)} = \mathbf{h}_{\sharp} P_{\mathbf{U}}^{(1)}$
- **因果机制同构**：$f_i^{(2)}(\mathbf{v}, h_i(u_i^{(1)})) = f_i^{(1)}(\mathbf{v}, u_i^{(1)})$

**核心定理 (Theorem 3.2)**：$\sim_{\mathrm{EI}}$ 蕴含 $\sim_{\mathcal{L}_3}$，即外生同构的两个SCM对所有反事实陈述给出一致结果。

### 双射SCM (BSCM) 上的EI

- BSCM定义：解映射 $\Gamma$ 是双射，等价于每个 $f_i(\mathbf{v}, \cdot)$ 对固定 $\mathbf{v}$ 是双射
- **反事实传输（Counterfactual Transport）**：定义 $K_{\mathcal{M},i}(\cdot, \mathbf{v}, \mathbf{v}') = (f_i(\mathbf{v}', \cdot)) \circ (f_i(\mathbf{v}, \cdot))^{-1}$
- Markov BSCM下反事实传输即条件分布间的传输映射
- **Theorem 4.6**：已知BSCM + 因果序 + 观测分布 + 反事实传输 → $\sim_{\mathrm{EI}}$-identifiable
- **Theorem 4.8**：若反事实传输恰好是KR传输，则只需BSCM + 因果序 + Markov + 观测分布

### 三角单调SCM (TM-SCM) 上的EI

- TM映射：三角映射且每个分量对最后一个变量严格单调
- TM-SCM定义：解映射经过向量化重排后是TM映射
- **关键性质**：TM映射的复合、求逆仍保持TM性质；两个相同单调签名的TM映射复合后为TMI映射
- **Corollary 5.4（核心推论）**：TM-SCM + 因果序 + Markov + 观测分布 → $\sim_{\mathrm{EI}}$-identifiable

这一推论统一了 Lu et al. 2020, Nasr-Esfahany et al. 2023, Scetbon et al. 2024 的结果。

### 损失函数与训练

神经TM-SCM采用最大似然估计，损失为负对数似然（NLL）：

$$\arg\min_\theta -\sum_{i=1}^N \log p_{\mathbf{V}_\theta}(\mathbf{v}^{(i)})$$

外生分布用无约束 normalizing flow (MAF) 建模，满足 Markov 独立性。

### 四种神经TM-SCM原型

| 原型 | 机制形式 | 代表工作 |
|------|---------|---------|
| DNME | 对角噪声：$f_{i,\theta} = \mathbf{b} + \mathbf{a} \odot \mathbf{u}_i$ | LSNM |
| TNME | 三角噪声：$f_{i,\theta} = \mathbf{b} + \mathbf{A} \mathbf{u}_i^\intercal$ | FiP |
| CMSM | 解映射=多个TM映射复合 | CausalNF |
| TVSM | 解映射由三角速度场ODE定义 | CFM |

## 实验关键数据

### 合成数据集

| 数据集 | 描述 |
|--------|------|
| TM-SCM-Sym | 4个小数据集（Barbell, Stair, Fork, Backdoor），≤4因果变量 |

实验使用合成数据集验证神经TM-SCM在反事实一致性问题上的有效性：
- 仅用观测分布样本训练，在反事实测试集上验证一致性
- 四种原型模型均能有效学习并给出与真实SCM一致的反事实结果

### 主要发现

- 验证了 Corollary 5.4 的理论正确性：TM-SCM类模型在满足假设时确实实现 $\mathcal{L}_3$-一致性
- 不同原型（DNME, TNME, CMSM, TVSM）在表达力和效率上各有权衡
- 外生分布的具体实现不影响可辨识性（与理论预测一致）

## 亮点与洞察

1. **EI概念精准定位**：在"完全恢复模型"和"反事实一致性"之间找到恰当的等价关系，精确刻画了实现完整反事实可辨识性所需的模型识别强度
2. **统一已有理论**：Corollary 5.4 将 Lu, Nasr-Esfahany, Scetbon 三组工作统一为一个推论的特例
3. **从标量推广到向量**：将内生变量空间从 $\mathbb{R}$ 推广到 $\mathbb{R}^{d_i}$，支持更广泛的SCM类
4. **反事实传输的新视角**：将反事实可辨识性与最优传输理论建立联系，提供全新解读
5. **理论到实践的完整路径**：从抽象等价类到具体神经网络实现，完成理论保障下的落地

## 局限与展望

1. **TM-SCM假设较强**：要求因果机制严格单调，排除了许多现实中常见的非单调关系
2. **因果序需已知**：所有理论结果都依赖于已知因果序，但因果发现本身就是困难问题
3. **仅考虑递归SCM**：未讨论非递归（含因果环）SCM的情况
4. **合成数据验证**：实验仅在合成数据上进行，尚未验证在真实数据（如医学、公平性场景）的效果
5. **$\sim_{\mathrm{EI}}$ 与 $\sim_{\mathcal{L}_3}$ 的间隙**：EI是 $\mathcal{L}_3$-一致的充分非必要条件，是否存在更弱但仍充分的条件尚不清楚

## 相关工作与启发

- **反事实等价**：Peters et al. 2017 定义的counterfactual equivalence要求外生分布完全相同，比EI更强
- **BGM等价**：Nasr-Esfahany et al. 2023 在BSCM框架内定义的等价关系，是EI的特例
- **CausalNF**：Javaloy et al. 2023 建立TMI映射的表示可辨识性，本文将其推广到完整反事实层
- **最优传输**：KR传输提供了一种规范的反事实传输构造，且与TMI映射间存在深刻联系（Lemma 5.1）
- **可启发方向**：将EI推广到半Markov SCM、带隐变量的因果表示学习

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 外生同构概念新颖，精准定位了可辨识性的"最佳中间层"
- 实验充分度: ⭐⭐⭐ — 仅合成数据，规模较小
- 写作质量: ⭐⭐⭐⭐ — 数学严谨，但符号密度极高，门槛陡峭
- 价值: ⭐⭐⭐⭐⭐ — 统一并推广了因果推断中反事实可辨识性的核心理论

<!-- RELATED:START -->

## 相关论文

- [Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation](transformer-based_spatial-temporal_counterfactual_outcomes_estimation.md)
- [Classifier Reconstruction Through Counterfactual-Aware Wasserstein Prototypes](classifier_reconstruction_through_counterfactual-aware_wasserstein_prototypes.md)
- [Towards Fine-Grained Interpretability: Counterfactual Explanations for Misclassification](../../CVPR2025/causal_inference/towards_fine-grained_interpretability_counterfactual_explanations_for_misclassif.md)
- [Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](../../NeurIPS2025/causal_inference/few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)

<!-- RELATED:END -->
