---
title: >-
  [论文解读] Center-Outward q-Dominance: A Sample-Computable Proxy for Strong Stochastic Dominance in Multi-Objective Optimisation
description: >-
  [AAAI 2026][随机多目标优化] 基于最优传输理论中的中心向外分布函数，提出 q-dominance 关系作为强一阶随机支配（strong FSD）的可计算近似，证明全分位数范围的 q-dominance 可推导出强 FSD，并给出显式样本量阈值控制 Type I 错误，在超参数调优排名和噪声多目标优化中验证了其实用性。
tags:
  - AAAI 2026
  - 随机多目标优化
  - 随机支配
  - 最优传输
  - 超参数调优
  - 中心向外分位数
---

# Center-Outward q-Dominance: A Sample-Computable Proxy for Strong Stochastic Dominance in Multi-Objective Optimisation

**会议**: AAAI 2026  
**arXiv**: [2511.12545](https://arxiv.org/abs/2511.12545)  
**代码**: 无  
**领域**: 其他  
**关键词**: 随机多目标优化, 随机支配, 最优传输, 超参数调优, 中心向外分位数

## 一句话总结

基于最优传输理论中的中心向外分布函数，提出 q-dominance 关系作为强一阶随机支配（strong FSD）的可计算近似，证明全分位数范围的 q-dominance 可推导出强 FSD，并给出显式样本量阈值控制 Type I 错误，在超参数调优排名和噪声多目标优化中验证了其实用性。

## 研究背景与动机

### 问题定义

**随机多目标优化问题（SMOOP）**：当目标函数含有随机性时（如超参数优化中算法的多次运行结果），每个候选解对应一个多维随机向量的分布。比较不同解需要对这些**分布进行排序**，而非比较确定性点。

### 现有方法的不足

**标量化**：将多目标转为单目标，丢失分布结构信息，依赖预定义的标量化函数

**矩代理**（均值/方差/CVaR）：用统计摘要替代随机目标，忽略依赖结构

**弱随机支配（Weak FSD）**：通过 CDF 比较实现，但在多维（$d>1$）下过于宽松——可能声明 $\mathbf{X} \succeq_w \mathbf{Y}$，但存在某些单调效用函数下 $\mathbf{X}$ 实际不如 $\mathbf{Y}$

**Rioux et al. (2025)**的 OT 方法：为每对分布计算一个最优耦合，多分布比较时缺乏全局一致的参考框架，且计算量随分布对数平方增长

### 弱 FSD 与强 FSD 的差距

论文通过反例清晰展示：在二维情况下，$\mathbf{X}=\{(0,1),(1,0)\}$ 弱支配 $\mathbf{Y}=\{(0,0),(1,1)\}$（CDF 意义下），但不强支配（考虑效用 $u(x_1,x_2)=\exp(x_1+x_2)$ 时 $\mathbb{E}[u(\mathbf{X})] < \mathbb{E}[u(\mathbf{Y})]$）。

### 核心动机

需要一种**可计算的、全局一致的**强 FSD 近似方法，同时具有：无超参数、样本量保证、自然松弛层次。

## 方法详解

### 整体框架

核心思路：利用中心向外分布函数将每个分布映射到单位球上的均匀分布 $U_d$，形成**统一参考框架**。在此框架下，具有相同中心向外秩和符号的样本点自然配对，从而进行逐分位数比较。

### 关键设计

#### 1. **中心向外分布/分位数函数**

基于 McCann 定理和 Hallin (2017, 2021) 的理论：

- **中心向外分位数函数** $\mathbf{Q}^\pm$：从均匀分布 $U_d$ 到目标分布 $P$ 的最优二次传输映射，$\mathbf{Q}^\pm \# U_d = P$
- **中心向外分布函数** $\mathbf{F}^\pm$：其反函数，将 $P$ 映射到 $U_d$，$\mathbf{F}^\pm \# P = U_d$

关键性质：$\|\mathbf{F}^\pm(\mathbf{Z})\|$ 服从 $[0,1]$ 上的均匀分布（秩），$\mathbf{F}^\pm(\mathbf{Z})/\|\mathbf{F}^\pm(\mathbf{Z})\|$ 在单位球面上均匀分布（符号），且两者独立。

**经验估计**：通过在单位球上构建增广网格（$n_R$ 个同心超球面 × $n_S$ 个单位向量），使用匈牙利算法（$O(n^3)$）求解最优分配问题。

#### 2. **q-dominance 定义**

对 $q \in [0,1)$，$P_1 \succeq_q P_2$ 当且仅当对所有 $\mathbf{y} \in \mathbb{C}_{P_2}(q)$（$P_2$ 的 $q$ 阶分位数区域）：

$$\mathbf{x} := \mathbf{Q}_1^\pm(\mathbf{F}_2^\pm(\mathbf{y})) \geq \mathbf{y}$$

直观含义：将 $\mathbf{y}$ 通过 $P_2$ 的分布函数映射到单位球，再通过 $P_1$ 的分位数函数映射回数据空间，得到的点 $\mathbf{x}$ 逐分量不小于 $\mathbf{y}$。

**与强 FSD 的关系**（定理 5）：若 $P_1 \succeq_q P_2$ 对所有 $q \in [0,1)$ 成立，则 $P_1$ 强一阶随机支配 $P_2$。

#### 3. **有限样本检验与样本量阈值**

定理 6 给出：在双 Lipschitz 条件下，对任意置信水平 $0<\delta<1$，存在显式阈值 $n^*(\delta)$，使得当样本量 $n \geq n^*(\delta)$ 时，经验 q-dominance 以至少 $1-\delta$ 的概率正确反映理论 q-dominance。

$n^*(\delta)$ 的表达式虽然复杂（涉及维度 $d$、Lipschitz 常数、分位数函数的最小间距 $\Delta$ 等），但**完全显式**，无需通过 Monte Carlo 或 bootstrap 估计。

### 损失函数 / 训练策略

本文不涉及神经网络训练。核心算法流程：

1. 对每个分布，计算经验中心向外分位数映射（一次性计算）
2. 基于共享增广网格进行逐对 q-dominance 检验
3. 使用排序算法（Algorithm 1）根据 q-dominance 构建非支配前沿

## 实验关键数据

### 主实验：多目标超参数优化排名

在 YAHPO-MO 基准上比较 7 种 HPO 方法（Random Search、ParEGO、SMS-EGO、EHVI、MEGO、MIES 等）：

| 方法 | q-dominance 平均排名 | HVI 平均排名 | 关键差异 |
|------|---------------------|-------------|---------|
| MIES | **最优（~2.0）** | 中等 | q-dominance 认为最好 |
| ParEGO | 中等（~3.1） | **最优** | HVI 认为最好 |
| Random Search | 最差 | 最差 | 两种方法一致 |

| 分析维度 | q-dominance 排名 | HVI 排名 |
|---------|-----------------|---------|
| 方法区分度 | 方法排名更紧凑 | 方法排名分散更大 |
| 显著不优于 RS 的方法数 | 仅 EHVI (1个) | EHVI + SMS-EGO + MIES (3个) |
| 信息保留 | 保留分布完整信息 | 仅使用期望值 |

### 消融实验：噪声增强 ZDT 基准

| ZDT 问题 | NSGA-II + q-dom | NSGA-II mean | NSGA-II single | 说明 |
|----------|----------------|-------------|---------------|------|
| ZDT1 | **最快收敛** | 慢 | 最慢 | q-dominance 显著优势 |
| ZDT2 | **最快收敛** | 慢 | 最慢 | 同上 |
| ZDT3 | **最快收敛** | 慢 | 最慢 | 同上 |
| ZDT4 | 三者相近 | 相近 | 相近 | ZDT4 对噪声敏感 |
| ZDT6 | **最快收敛** | 慢 | 最慢 | q-dominance 显著优势 |

实验设置：$\sigma=0.1$，种群大小 20，200 代，每解 64 个样本（$n_R=n_S=8$）。

### 关键发现

1. **MIES vs ParEGO 案例分析**：在 lcbench 167152 上，MIES 高概率获得更好结果但有略高风险，HVI（期望值）因 ParEGO 更低方差而偏好它。q-dominance 正确识别了 MIES 的分布优势
2. q-dominance 排名下，当 HVI 无法区分方法时（期望值接近），仍能给出有意义的排名
3. 在 NSGA-II 中替换均值选择为 q-dominance 选择，收敛速度在 5/6 个 ZDT 问题上显著提升

## 亮点与洞察

1. **全局一致参考框架**：每个分布只需计算一次到均匀分布的映射，所有比较共享同一参考——避免了 Rioux et al. 方法中不同耦合之间不一致的问题
2. **自然的松弛层次**：不同 $q$ 值对应不同严格程度的支配关系，且这些松弛在计算一次映射后**零额外代价**获得
3. **无超参数**：不同于 Rioux et al. 的方法需要调节正则化参数 $\lambda$ 和函数 $h$
4. **显式样本量阈值**：提供了可操作的有限样本保证

## 局限与展望

1. **匈牙利算法的 $O(n^3)$ 复杂度**：限制了大样本量场景的可扩展性
2. $n^*(\delta)$ 的表达式依赖 Lipschitz 常数和 $\Delta$ 等实际难以准确估计的量
3. 在维度 $d \geq 5$ 时可用的 $\theta$ 范围变窄，可能影响实用性
4. 实验规模有限（仅与少数 baseline 比较），作者明确指出大规模 benchmarking 留待未来
5. 与 Rioux et al. (2025) 的理论关系尚未完全澄清

## 相关工作与启发

- **Hallin (2021)** 的中心向外分位数理论为本文提供了核心数学工具
- **Rioux et al. (2025)** 的 entropic OT 方法用于近似随机支配检验，但缺乏全局一致性
- **NSGA-II** 中的选择机制可直接替换为 q-dominance，推广性强
- q-dominance 的思想可能推广到其他需要分布比较的场景（如鲁棒优化、贝叶斯优化）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （中心向外分位数与随机支配的结合是全新的理论贡献）
- 实验充分度: ⭐⭐⭐ （实验规模有限，但足以验证理论）
- 写作质量: ⭐⭐⭐⭐⭐ （数学严谨，结构清晰，定理-证明-应用逻辑顺畅）
- 价值: ⭐⭐⭐⭐ （为随机多目标优化提供了理论上更坚实的基础）

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Approximation Algorithm for Constrained k-Center Clustering: A Local Search Approach](approximation_algorithm_for_constrained_k-center_clustering_.md)
- [\[NeurIPS 2025\] Computable Universal Online Learning](../../NeurIPS2025/others/computable_universal_online_learning.md)
- [\[AAAI 2026\] Online Linear Regression with Paid Stochastic Features](online_linear_regression_with_paid_stochastic_features.md)
- [\[CVPR 2026\] Bounds on Agreement between Subjective and Objective Measurements](../../CVPR2026/others/bounds_on_agreement_between_subjective_and_objective_measurements.md)
- [\[ACL 2025\] Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation](../../ACL2025/others/bone_soups_multi_objective_gen.md)

<!-- RELATED:END -->
