---
title: >-
  [论文解读] Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study
description: >-
  [ICLR 2026][KAN] 首次对样条KAN的初始化策略进行系统性研究，提出LeCun/Glorot启发的方差保持方案和可调幂律初始化族，在126K+模型实例的大规模实验中证明幂律初始化在函数拟合和PDE求解上全面超越基线，Glorot方案在大参数量模型上增益显著，NTK特征谱分析揭示了其背后的优化动力学机制。
tags:
  - ICLR 2026
  - KAN
  - 初始化策略
  - 方差保持
  - 幂律初始化
  - 神经切线核
---

# Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study

**会议**: ICLR 2026  
**arXiv**: [2509.03417](https://arxiv.org/abs/2509.03417)  
**代码**: [GitHub](https://github.com/srigas/KAN_Initialization_Schemes)  
**领域**: 深度学习理论/KAN  
**关键词**: KAN, 初始化策略, 方差保持, 幂律初始化, 神经切线核

## 一句话总结

首次对样条KAN的初始化策略进行系统性研究，提出LeCun/Glorot启发的方差保持方案和可调幂律初始化族，在126K+模型实例的大规模实验中证明幂律初始化在函数拟合和PDE求解上全面超越基线，Glorot方案在大参数量模型上增益显著，NTK特征谱分析揭示了其背后的优化动力学机制。

## 研究背景与动机

**领域现状**：Kolmogorov-Arnold Networks（KAN）用可训练的B样条基函数替代MLP中固定的激活函数，在函数拟合、PDE求解、算子学习等科学计算任务上展现了独特优势。KAN的每一层输出为 $y_j = \sum_{i=1}^{n_{\text{in}}} (r_{ji} R(x_i) + c_{ji} \sum_{m=1}^{G+k} b_{jim} B_m(x_i))$，其中包含三类可训练参数：残差权重 $r_{ji}$、缩放权重 $c_{ji}$ 和样条基系数 $b_{jim}$。目前KAN社区普遍沿用原始论文的初始化策略——缩放权重设为1，残差权重用Glorot初始化，基系数从 $\mathcal{N}(0, 0.1)$ 采样——从未被系统地挑战过。

**现有痛点**：MLP领域积累了丰富的初始化理论（LeCun 1998、Glorot 2010、He 2015），这些成果的核心是"方差保持"——保证信号在层间传播时不放大也不衰减。但这套理论无法直接迁移到KAN：第一，KAN每层有三类参数而非一类，方差分解更复杂；第二，样条基函数 $B_m(x)$ 依赖于网格划分，其统计矩没有通用的解析形式；第三，残差分支使用SiLU而非线性函数，进一步增加了分析难度。

**核心矛盾**：初始化对深度网络训练至关重要——不好的初始化会导致信号爆炸/消失、早期隐层饱和、收敛缓慢。但KAN社区长期使用的 $\sigma=0.1$ 方案只是一个人为选择，缺乏理论依据，尤其在参数量增大时问题愈加严重。

**本文目标** 为样条KAN建立系统化的初始化理论，回答三个问题：(1) MLP的方差保持原则能否适配KAN？(2) 什么样的初始化策略能跨任务、跨架构一致地优于基线？(3) 初始化如何影响KAN的优化动力学？

**切入角度**：作者从MLP初始化的两大经典出发点——前向方差保持（LeCun）和前向+反向联合保持（Glorot）——推导KAN的方差公式，同时引入一族可调的幂律初始化作为经验对照。通过大规模网格搜索 + NTK特征谱分析 + Feynman物理公式数据集三级验证来建立结论。

**核心 idea**：在KAN的三类参数上推导方差保持公式并提出幂律初始化族，通过126K+模型的系统实验证明幂律初始化全面最优。

## 方法详解

### 整体框架

本文的方法框架分为三个层次：(1) 理论推导——从方差保持原则出发推导KAN适配的LeCun和Glorot初始化公式；(2) 经验探索——设计幂律初始化族，通过大规模网格搜索寻找最优配置；(3) 动力学分析——用NTK特征谱解释不同初始化的性能差异。输入是KAN的架构参数（层数、宽度、网格大小），输出是每类权重的初始化标准差 $\sigma_r$ 和 $\sigma_b$。

### 关键设计

1. **LeCun启发的前向方差保持初始化**:

    - 功能：保证每层输出方差等于输入方差，防止信号在前向传播中逐层放大或衰减
    - 核心思路：固定 $c_{ji}=1$，对 $r_{ji} \sim \mathcal{N}(0, \sigma_r^2)$ 和 $b_{jim} \sim \mathcal{N}(0, \sigma_b^2)$，利用独立性假设推导 $\text{Var}(y_j) = \text{Var}(x_i)$ 的约束，得到 $\sigma_r = \sqrt{\text{Var}(x_i) / (n_{\text{in}}(G+k+1)\mu_R^{(0)})}$，其中 $\mu_R^{(0)} = \mathbb{E}[R(x_i)^2]$ 是SiLU的二阶矩。对于 $\sigma_b$，公式类似但包含样条基的二阶矩 $\mu_B^{(0)} = \mathbb{E}[B_m(x_i)^2]$
    - 设计动机：直接仿照LeCun对MLP的分析，但需要处理KAN特有的残差+样条双分支结构。由于 $\mu_B^{(0)}$ 依赖网格，无法解析计算，因此衍生出两种变体：LeCun-numerical（在初始化时数值采样估计矩）和LeCun-normalized（对基函数做标准化使矩恒为1，即 $\tilde{B}_m(x_i) = (B_m(x_i) - \mathbb{E}[B_m]) / \sqrt{\mu_B^{(0)} - \mathbb{E}^2[B_m]}$）

2. **Glorot启发的前向+反向联合方差保持初始化**:

    - 功能：同时稳定前向激活和反向梯度的方差，避免两个方向上的信号退化
    - 核心思路：在LeCun的基础上额外约束反向传播的方差保持，推导得到 $\sigma_r = \sqrt{(G+k+1)^{-1} \cdot 2 / (n_{\text{in}}\mu_R^{(0)} + n_{\text{out}}\mu_R^{(1)})}$，其中 $\mu_R^{(1)} = \mathbb{E}[R'(x_i)^2]$ 是SiLU导数的二阶矩。$\sigma_b$ 的公式类似，涉及基函数导数的矩 $\mu_B^{(1)}$，可通过自动微分数值估计
    - 设计动机：MLP中Glorot初始化比LeCun更鲁棒，这里将同样的"双向平衡"思想迁移到KAN。额外引入 $n_{\text{out}}$ 和导数矩意味着标准差会随层的输入输出维度自适应调整

3. **幂律初始化族（Power-Law Family）**:

    - 功能：提供一个简洁的经验公式族，通过调节两个指数参数覆盖广阔的初始化空间
    - 核心思路：$\sigma_r = (n_{\text{in}}(G+k+1))^{-\alpha}$，$\sigma_b = (n_{\text{in}}(G+k+1))^{-\beta}$，其中 $\alpha, \beta \in \{0.0, 0.25, 0.5, \ldots, 2.0\}$。不做任何理论推导，纯粹通过网格搜索 $(\alpha, \beta)$ 的81种组合来找最优配置
    - 设计动机：理论公式虽然优雅但需要估计基函数矩（尤其在PDE场景下存在归一化传导问题），幂律族完全绕过这一难点。一旦在某类任务上找到好的指数区间（如函数拟合推荐 $\alpha \approx 0.25, \beta \approx 1.0\text{-}1.75$），可以直接复用于同类新问题

### 实验设计

实验覆盖三大基准：(1) 函数拟合——5个二维目标函数，训练2000轮；(2) PDE求解——Allen-Cahn、Burgers、Helmholtz三个前向PDE，训练5000轮；(3) Feynman物理公式数据集的子集。架构搜索范围：1-4个隐层，宽度 $2^1$ 到 $2^6$，网格大小 $G \in \{5,10,20,40\}$。每组实验跑5个随机种子取中位数（网格搜索用3个种子以降低计算量）。所有实验在JAX/jaxKAN上实现，单卡RTX 4090。

## 实验关键数据

### 函数拟合网格搜索（126,240个模型实例）

| 初始化方案 | $f_1$优于基线(Loss/L2/Both) | $f_3$优于基线(Loss/L2/Both) | $f_5$优于基线(Loss/L2/Both) |
|:---|:---|:---|:---|
| LeCun-numerical | 18.75% / 6.25% / 1.04% | 12.50% / 5.21% / 0.00% | 26.04% / 2.08% / 0.00% |
| LeCun-normalized | 19.79% / 11.46% / 2.08% | 19.79% / 11.46% / 5.21% | 31.25% / 6.25% / 1.04% |
| Glorot | 78.13% / 78.13% / **78.13%** | 78.13% / 78.13% / **78.13%** | 72.92% / 72.92% / 64.59% |
| **Power-Law** | **100%** / **100%** / **100%** | **100%** / **100%** / **100%** | 98.96% / 96.88% / **95.83%** |

### PDE基准网格搜索（56,882个模型实例）

| 初始化方案 | Allen-Cahn(Loss/L2/Both) | Burgers(Loss/L2/Both) | Helmholtz(Loss/L2/Both) |
|:---|:---|:---|:---|
| LeCun-numerical | 11.11% / 16.67% / 8.33% | 11.11% / 22.22% / 6.94% | 8.33% / 15.28% / 2.78% |
| LeCun-normalized | 2.78% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| Glorot | 55.56% / 51.39% / 41.67% | 50.00% / 54.17% / 36.11% | **76.39%** / **72.22%** / **62.50%** |
| **Power-Law** | **98.61%** / **94.44%** / **94.44%** | **100%** / 73.61% / 73.61% | 98.61% / 87.50% / 87.50% |

### Feynman数据集代表性结果（大架构 G=20, 3×32）

| 物理公式 | 基线 L2 | Glorot L2 | Power-Law L2 | Power-Law提升倍数 |
|:---|:---|:---|:---|:---|
| I.12.11 | 3.77×10⁻¹ | 1.47×10⁻³ | **1.66×10⁻⁴** | 2271× |
| I.16.6 | 6.31×10⁻¹ | 1.63×10⁻² | **1.48×10⁻²** | 43× |
| I.26.2 | 1.10×10⁰ | 8.98×10⁻³ | **1.25×10⁻³** | 880× |
| I.30.3 | 7.72×10⁻¹ | 2.92×10⁻³ | **4.17×10⁻⁴** | 1851× |
| II.6.15a | 7.60×10⁰ | 5.47×10⁻² | **4.40×10⁻³** | 1727× |
| II.35.18 | 1.19×10⁰ | 1.18×10⁻² | **7.77×10⁻⁴** | 1531× |
| III.10.19 | 2.74×10⁻¹ | 9.89×10⁻⁴ | **8.70×10⁻⁵** | 3149× |

### 关键发现

- **幂律初始化全面碾压**：在函数拟合中，最优配置 $(\alpha, \beta) = (0.25, 1.0)$ 在5个目标函数上同时改善Loss和L2误差的比例为87.5%~97.9%。最优区间集中在 $\alpha \in \{0.25, 0.5\}$, $\beta \geq 1.0$
- **Glorot在大模型上崛起**：当参数量增大（更深/更宽/更细网格），Glorot的胜率从几乎与基线持平跃升到60-78%。这说明大KAN更需要双向方差保持
- **LeCun-normalized在PDE上完全失败**：胜率接近0%。原因是归一化基函数会将标准差传导到所有导数中，改变PDE残差的刚度，破坏物理信息损失的平衡
- **NTK特征谱揭示机制**：基线和LeCun的NTK特征值分布极度偏斜（少数大特征值主导），训练中进一步坍塌→有效秩低→优化困难。幂律初始化的特征谱近似幂律衰减且训练全程稳定，Glorot次之但也远优于基线
- **基线在大模型上"反向退化"**：许多Feynman公式上，从小架构切到大架构时，基线的L2误差反而变差（如I.12.11从3.67×10⁻³恶化到3.77×10⁻¹），而Glorot和Power-Law在大模型上误差普遍下降2-3个数量级

## 亮点与洞察

- **首次完整适配**：将MLP经典初始化理论（前向方差保持、前向+反向联合保持）系统迁移到KAN的三类参数框架，推导过程清晰严谨，为KAN变体的初始化研究树立了方法论模板
- **幂律族的极简之美**：两个超参 $(\alpha, \beta)$ 的幂律公式，无需计算任何基函数矩、无需数值采样、无需修改网络架构，却在超过12万个模型实例上全面击败需要精确矩估计的理论方案。"简单但足够好"的经验方法有时胜过精巧的理论方法
- **NTK作为初始化诊断工具**：不只是说"某方案更好"，而是通过NTK特征值谱给出了可视化的机制解释——好的初始化应该产生分散的、稳定的特征谱。这个分析框架可以直接迁移到评估任何新的KAN初始化方案
- **PDE领域的特殊教训**：LeCun-normalized在函数拟合上不错但在PDE上完全崩溃，揭示了物理信息损失（涉及高阶导数）对初始化的敏感性远超普通回归任务

## 局限与展望

- **仅限样条KAN**：Chebyshev KAN、Fourier KAN、Wavelet KAN等变体有不同的基函数结构，最优初始化策略可能完全不同
- **幂律最优指数缺乏理论解释**：为什么 $\alpha \approx 0.25, \beta \approx 1.0\text{-}1.75$ 效果好？论文承认这是纯经验发现，缺乏深层理论
- **任务域偏窄**：只验证了函数拟合和PDE，未涉及分类、强化学习、生成模型等场景。幂律指数的最优区间是否跨域可迁移尚未确认
- **规模天花板**：实验的最大架构是3层×32宽度×G=20，远小于实际应用中可能出现的大规模KAN。万参+量级KAN的初始化行为值得探索
- **未与自适应优化交互**：所有实验使用固定学习率（附录中有调度器实验但非主线），初始化与Adam/学习率warmup等现代训练技巧的交互效应未充分研究

## 相关工作与启发

- **vs 原始KAN (Liu et al., 2025)**：原始KAN使用 $\sigma=0.1$ 的固定初始化，本文证明这在大模型上严重不足，尤其在Feynman数据集上大架构用基线初始化性能反而下降
- **vs Chebyshev KAN初始化 (Rigas et al., 2026)**：Chebyshev KAN已成功应用Glorot初始化但去掉了残差分支，与本文的样条+残差设定有本质差异，不能直接迁移
- **vs He初始化**：He初始化专为ReLU设计（考虑了ReLU零点处的非对称性），KAN中SiLU的SiLU光滑且对称，LeCun/Glorot比He更合适作为起点

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统探索KAN初始化，填补了明确的空白
- 实验充分度: ⭐⭐⭐⭐⭐ 12.6万+模型实例的超大规模网格搜索，NTK分析+Feynman验证三层递进
- 写作质量: ⭐⭐⭐⭐ 理论推导与实验分析衔接紧密，图表设计清晰
- 价值: ⭐⭐⭐⭐ 为KAN社区提供了即插即用的实用初始化策略

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](../../AAAI2026/interpretability/flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)
- [\[ACL 2025\] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](../../ACL2025/interpretability/an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)
- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](modal_logical_neural_networks_for_financial_ai.md)
- [\[ICLR 2026\] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks](salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)

<!-- RELATED:END -->
