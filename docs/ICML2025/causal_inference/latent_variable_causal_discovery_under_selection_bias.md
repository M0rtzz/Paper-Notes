---
title: >-
  [论文解读] Latent Variable Causal Discovery under Selection Bias
description: >-
  [ICML 2025][因果推理][潜变量因果发现] 首次将秩约束推广到选择偏差场景，证明在线性选择机制下有偏协方差矩阵的秩仍保留因果结构和选择机制的信息，提出广义 t-separation 图准则，并在单因子模型上证明了可识别性…
tags:
  - "ICML 2025"
  - "因果推理"
  - "潜变量因果发现"
  - "选择偏差"
  - "秩约束"
  - "线性高斯模型"
  - "单因子模型"
---

# Latent Variable Causal Discovery under Selection Bias

**会议**: ICML 2025  
**arXiv**: [2512.11219](https://arxiv.org/abs/2512.11219)  
**代码**: [https://github.com/MarkDana/Latent-Selection](https://github.com/MarkDana/Latent-Selection)  
**领域**: 因果推理  
**关键词**: 潜变量因果发现, 选择偏差, 秩约束, 线性高斯模型, 单因子模型

## 一句话总结

首次将秩约束推广到选择偏差场景，证明在线性选择机制下有偏协方差矩阵的秩仍保留因果结构和选择机制的信息，提出广义 t-separation 图准则，并在单因子模型上证明了可识别性，在合成和真实数据（World Value Survey、Big Five 人格）上验证了有效性。

## 研究背景与动机

**领域现状**：潜变量因果发现旨在从观测数据恢复不可直接观测的潜变量之间的因果关系，在心理学问卷、政治经济调查等场景中至关重要。现有工具已从基本的条件独立（CI）约束扩展到秩约束、高阶矩约束、矩阵分解等多种统计工具，显著提升了潜变量因果发现的能力。

**现有痛点**：选择偏差（如特定特质的人更愿意参与调查）在实际数据中普遍存在，但所有超越 CI 约束的新工具都仅为潜变量设计，没有一个能处理选择偏差。目前唯一能同时处理潜变量和选择偏差的算法是 FCI，但 FCI 仅使用 CI 约束，对潜变量因果发现信息不足——它只能发现观测变量间的因果关系，无法识别潜变量间的因果结构。

**核心矛盾**：这造成一个巨大鸿沟：虽然更强的潜变量因果发现方法已经存在，但一旦引入选择偏差，这些新工具就不得不搁置，退回到只用 CI 约束的 FCI。核心困难在于，选择偏差会彻底改变数据分布——即使原始模型是线性高斯的，选择后变为截断高斯，协方差和高阶矩难以表达和解释。

**本文目标** 开发超越 CI 约束、同时处理潜变量和选择偏差的统计工具。

**切入角度**：避免显式建模选择后的完整分布，转而寻找不变的统计模式——协方差子矩阵的秩。直觉源自"逆 Tetrad 结构"的发现：四个原本独立的变量被线性和截断后，虽然不再服从高斯分布，但有偏协方差矩阵的低秩结构与经典 Tetrad 完全相同。

**核心 idea**：选择偏差像潜变量一样，会在协方差矩阵的秩中留下"维度瓶颈"的痕迹，这些痕迹可以通过广义 t-separation 准则从图结构中读取。

## 方法详解

### 整体框架

论文的核心贡献是一个理论工具——**广义秩约束**。流程为：(1) 定义线性选择机制并构建选择增广图 $\mathcal{G}^{(\mathcal{S})}$；(2) 在增广图上建立广义 t-separation 准则精确刻画有偏协方差矩阵的秩；(3) 将该工具应用于单因子模型，证明从有偏观测数据中仍可恢复潜变量间的 CI 关系；(4) 结合 FCI 算法完成潜变量因果发现。

### 关键设计

1. **线性选择机制与选择增广图**:

    - 功能：定义一个足够通用的选择偏差数学模型，并将其融入因果图表示
    - 核心思路：线性选择机制定义为 $\{(V_i, \beta_i, \epsilon_i, \mathcal{Y}_i)\}_{i=1}^k$，每个选择条件有参与变量子集 $V_i$、线性系数 $\beta_i$、噪声项 $\epsilon_i$ 和可接受值集 $\mathcal{Y}_i$，响应变量 $Y_i = \beta_i^\top V_i + \epsilon_i$，样本被选入/排除基于 $Y_i \in \mathcal{Y}_i$。选择增广图在原始 DAG 上添加响应节点 $Y$ 和对应边
    - 设计动机：该框架涵盖硬截断（$\epsilon=0, \mathcal{Y}=(a,b)$）、Logistic 选择（$\epsilon \sim Logistic, \mathcal{Y}=(a,\infty)$）、Probit 选择（$\epsilon \sim \mathcal{N}(0,1), \mathcal{Y}=(a,\infty)$）和稳定化选择（$\epsilon \sim \mathcal{N}(0,1), \mathcal{Y}=\{a\}$）等经典参数模型

2. **广义秩约束的图准则（定理 1）**:

    - 功能：精确刻画选择偏差数据中协方差子矩阵的秩
    - 核心思路：$\text{rank}(\Sigma_{A,B}^{(\mathcal{S})}) = \min\{|C|+|D| : C,D \subset X \cup Y, (C,D) \text{ t-separates } (A \cup Y, B \cup Y) \text{ in } \mathcal{G}^{(\mathcal{S})}\} - k$。当 $\mathcal{S}=\emptyset$（无选择）时退化为原始秩约束（命题 2）。证明核心：即使在非高斯的截断分布下，通过条件协方差的分块矩阵分析证明 $\text{rank}(\Sigma_{A,B}^{(\mathcal{S})}) = \text{rank}(\Sigma_{A \cup Y, B \cup Y}) - |Y|$
    - 设计动机：与 CI 约束类似追求图准则的精确刻画，但秩约束是 CI 的严格推广——CI 对应秩为 $|C|$（命题 1），而秩约束可以捕捉 CI 之外的低秩结构（如 Tetrad 结构中无 CI 但有 rank=1）

3. **单因子模型下的可识别性（命题 3）**:

    - 功能：证明广义秩约束在单因子模型中可以恢复潜变量间的 CI 关系
    - 核心思路：对于任意不相交子集 $A, B, C \subset L$（潜变量），$A \perp B | C$ 成立当且仅当观测变量协方差子矩阵 $\Sigma_{\mathbf{X}_A \cup \mathbf{X}_C^{(1)}, \mathbf{X}_B \cup \mathbf{X}_C^{(2)}}^{(\mathcal{S})}$ 的秩为 $|C|$（其中 $\mathbf{X}_C^{(1)}, \mathbf{X}_C^{(2)}$ 是 $\mathbf{X}_C$ 的不相交分划且各含 $\geq |C|$ 个观测变量）
    - 设计动机：单因子模型是问卷数据的经典建模（每个潜变量有 $\geq 2$ 个观测测量），选择偏差在此类数据中尤其常见（如人格特质影响问卷参与意愿）

### 损失函数 / 训练策略

本文是理论/算法工作，不涉及神经网络训练。实际算法流程：假设已知单因子聚类（哪些观测变量是哪个潜变量的测量），利用广义秩约束从有偏协方差矩阵恢复潜变量间的 CI 关系，然后调用 FCI 算法得到部分祖先图（PAG）。秩检验采用统计假设检验框架。

## 实验关键数据

### 合成数据实验（PAG 恢复质量）

| 潜变量数 $n$ | 选择变量数 | 方法 | 边标记差异总数 (↓更好) |
|-------------|-----------|------|-------------------|
| 5 | 1 | FCI | ~8 |
| 5 | 1 | PC | ~10 |
| 5 | 1 | BOSS | ~12 |
| 5 | 1 | **Ours** | **~3** |
| 10 | 2 | FCI | ~20 |
| 10 | 2 | **Ours** | **~8** |
| 15 | 3 | FCI | ~35 |
| 15 | 3 | **Ours** | **~12** |
| 20 | 4 | FCI | ~55 |
| 20 | 4 | **Ours** | **~18** |

### 计算效率

| 潜变量数 | 运行时间 | 硬件 |
|---------|---------|------|
| 5 | < 1 秒 | 2 CPU, 16GB |
| 10 | < 1 秒 | 2 CPU, 16GB |
| 15 | < 1 秒 | 2 CPU, 16GB |
| 20 | < 5 分钟 | 2 CPU, 16GB |

### 真实数据（World Value Survey）

| 国家 | 发现的选择偏差变量 | 主要因果发现 |
|------|-----------------|------------|
| 加拿大 | Social Trust（尾标记→选择祖先） | 信任程度影响问卷参与倾向 |
| 中国 | Social Trust + Perception of Science | 科学认知正相关→参与社会科学研究 |
| 德国 | 5个变量涉及选择偏差 | 更复杂的多变量选择机制 |
| Big Five | Agreeableness | 宜人性高→可能更愿意配合问卷 |

### 理论贡献总结

| 工具 | 潜变量 | 选择偏差 | 超越CI | 首次 |
|------|--------|---------|--------|------|
| CI (d-separation) | ✓ | ✓ | ✗ | - |
| 原始秩约束 (t-separation) | ✓ | ✗ | ✓ | sullivant2010 |
| **广义秩约束 (本文)** | **✓** | **✓** | **✓** | **✓** |

### 关键发现
- 随着潜变量数增加，本文方法与基线的性能差距持续扩大——20 个潜变量时优势约 3 倍，说明广义秩约束在复杂场景下信息量优势更明显
- 高斯和指数噪声两种选择机制下实验结果一致，验证了理论对非高斯选择噪声的通用性
- 三个不同国家的 WVS 数据中都检测到 Social Trust 作为选择偏差的潜在祖先，且不同国家表现出国家特异的选择模式
- "逆 Tetrad"现象揭示选择偏差和潜变量在秩约束层面有时不可区分（rank equivalent），但 Spider 结构变体表明在更复杂拓扑下它们可以被区分

## 亮点与洞察

- **"维度瓶颈"直觉的优雅理论化**：原始秩约束捕捉"依赖如何通过最小维度流出"，本文证明选择偏差也产生类似的维度瓶颈效果，且两者在同一图准则框架内统一。这个直觉简洁而深刻
- **逆 Tetrad 结构的发现**：四个独立变量被线性和截断后，有偏协方差的低秩结构与经典 Tetrad（一个潜变量的四个测量）完全相同——这个对偶现象是整篇论文的出发点，极具启发性
- **证明技巧的巧妙性**：避免显式求解截断高斯的协方差表达式（极其复杂），转而通过条件协方差的分块矩阵消元 $\text{rank}(\Sigma^{(\mathcal{S})}_{A,B}) = \text{rank}(\Sigma_{A \cup Y, B \cup Y}) - |Y|$ 将问题转回无选择时的秩约束

## 局限与展望

- 作者承认尚未建立完整的秩等价类刻画——类似于 MAG 对 CI 等价类的刻画——这是理解潜变量和选择偏差可区分边界的关键
- 假设线性选择机制和线性高斯因果模型，对非线性场景的推广不直接
- 单因子模型假设已知聚类（哪些观测属于哪个潜变量），实际场景中聚类发现本身是未解决问题
- 合成实验中图的平均度为 2，较为稀疏；在更密集的因果图上效果有待验证
- 实际算法依赖协方差矩阵秩的统计检验，有限样本下检验的功效和稳健性需进一步研究

## 相关工作与启发

- **vs FCI**: FCI 是目前唯一能同时处理潜变量和选择偏差的算法，但仅用 CI 约束，对潜变量因果发现信息不足（如 Tetrad 结构中无 CI 但有低秩）。本文的广义秩约束严格推广 CI 约束，可视为 FCI 在秩约束层面的升级
- **vs 原始秩约束 (sullivant2010)**: 原始秩约束通过 t-separation 推广 d-separation 来处理潜变量，但完全不考虑选择偏差。本文将 t-separation 扩展到选择增广图，实现了统一
- **vs Heckman 选择模型**: Heckman 的经典工作关注因果推断中的选择偏差矫正，而非因果发现。本文首次将选择偏差引入因果结构学习的工具箱

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将秩约束工具推广到选择偏差场景，填补了重要理论空白，逆 Tetrad 直觉极具原创性
- 实验充分度: ⭐⭐⭐⭐ 合成数据和两个真实数据集验证全面，但缺少更大规模的合成实验
- 写作质量: ⭐⭐⭐⭐⭐ 从直观示例到严格理论推导层次分明，Figure 1 的贡献定位图清晰精准
- 价值: ⭐⭐⭐⭐⭐ 为因果发现社区提供了全新的理论工具，打开了潜变量+选择偏差因果发现的研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](../../ICML2026/causal_inference/towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)
- [\[ICML 2025\] Causal Abstraction Inference under Lossy Representations](causal_abstraction_inference_under_lossy_representations.md)
- [\[ICLR 2026\] Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](../../ICLR2026/causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)
- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[ACL 2025\] IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](../../ACL2025/causal_inference/iris_an_iterative_and_integrated_framework.md)

</div>

<!-- RELATED:END -->
