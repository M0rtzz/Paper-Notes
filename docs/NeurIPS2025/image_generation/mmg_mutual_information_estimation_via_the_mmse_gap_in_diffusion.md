---
title: >-
  [论文解读] MMG: Mutual Information Estimation via the MMSE Gap in Diffusion
description: >-
  [NeurIPS 2025][图像生成][mutual information estimation] 利用扩散模型的信息论公式，证明互信息等于条件与无条件去噪 MMSE 之间的差值在所有信噪比上的积分的一半，提出 MMG 估计器，结合自适应重要性采样和正交原理显著提升估计精度和稳定性。
tags:
  - NeurIPS 2025
  - 图像生成
  - mutual information estimation
  - 扩散模型
  - MMSE
  - 去噪
  - importance sampling
---

# MMG: Mutual Information Estimation via the MMSE Gap in Diffusion

**会议**: NeurIPS 2025  
**arXiv**: [2509.20609](https://arxiv.org/abs/2509.20609)  
**代码**: [GitHub](https://github.com/fengsxy/Diffusion_MI)  
**领域**: 信息论 / 扩散模型  
**关键词**: mutual information estimation, diffusion models, MMSE, denoising, importance sampling

## 一句话总结

利用扩散模型的信息论公式，证明互信息等于条件与无条件去噪 MMSE 之间的差值在所有信噪比上的积分的一半，提出 MMG 估计器，结合自适应重要性采样和正交原理显著提升估计精度和稳定性。

## 研究背景与动机

互信息（Mutual Information, MI）是衡量随机变量间关系最通用的度量，但从样本估计 MI 是一个核心挑战：

**传统方法局限**：基于 KDE 或 k-NN 的方法在高维数据上受维度诅咒影响严重

**变分方法的样本复杂度问题**：MINE、InfoNCE 等基于变分下界的方法，其样本复杂度或方差随真实 MI 指数增长，且受批大小限制

**分数匹配的中间步骤困难**：MINDE 方法虽然利用了扩散模型，但依赖准确逼近对数密度梯度（score function），这是一个具有挑战性的中间步骤

核心动机：扩散模型近年来在密度估计上取得了巨大进展，能否绕过 score matching，直接利用去噪目标本身来估计 MI？

## 方法详解

### 整体框架

从扩散模型的信息论公式出发，推导出 MI 与 MMSE 差值的精确关系，然后用神经网络去噪器近似 MMSE，通过数值积分估计 MI。

### 关键设计

1. **MMSE Gap 公式推导**：

   给定高斯噪声通道 $z_\gamma = \sqrt{\gamma/(1+\gamma)} x + \sqrt{1/(1+\gamma)} \epsilon$，信噪比为 $\gamma$。从 ITD（Information-Theoretic Diffusion）的结果出发：

    $-\log p(x) = \frac{d}{2}\log(2\pi e) - \frac{1}{2}\int_0^\infty d\gamma \left(\frac{d}{1+\gamma} - \text{mmse}(x|\gamma)\right)$

   对条件分布 $p(x|y)$ 写同样的等式，相减得到逐点互信息：

    $\log p(x|y) - \log p(x) = \frac{1}{2}\int_0^\infty d\gamma \left(\text{mmse}(x|\gamma) - \text{mmse}(x|\gamma, y)\right)$

   取期望得到 MI 的精确表达：

    $I(x;y) = \frac{1}{2}\int_0^\infty d\gamma \left(\text{mmse}_x(\gamma) - \text{mmse}_{x|y}(\gamma)\right)$

   即 **MI = 条件与无条件 MMSE 曲线之间面积的一半**。

2. **模型训练**：训练单个去噪网络 $\hat{x}_\theta(z_\gamma, \gamma, y)$，训练时以50%概率用空值替换 $y$（类似 classifier-free guidance），使同一网络同时学习条件和无条件去噪。损失为标准 MSE 去噪损失。

3. **自适应重要性采样**：

    - SNR 空间的积分用蒙特卡洛估计，采样分布 $q(\gamma)$ 为 logistic 分布
    - 先训练初步模型，分析条件 MMSE 曲线的过渡区域
    - 定位参数 $\mu$ 设为 MMSE 曲线穿过 $d/2$ 误差阈值的 log-SNR
    - 尺度参数 $\sigma$ 由 $d/4$ 阈值位置推导
    - 这使采样集中在去噪器从无效到有效的关键过渡区域

4. **正交原理**：
   基于 MMSE 估计的正交性质，MMSE gap 可以等价表示为条件和无条件去噪器输出之差的二范数：

    $\text{MMSE Gap} = \mathbb{E}[\|\hat{x}(z_\gamma, y) - \hat{x}(z_\gamma)\|^2]$

   优点：(a) 永远非负（单个平方项），避免两个大数相减的数值不稳定；(b) 积分被积函数更平滑，方差更低

### 四个估计器变体

| 变体 | 自适应采样 | 正交原理 |
|---|---|---|
| MMG | ✗ | ✗ |
| MMG-adaptive | ✓ | ✗ |
| MMG-orthogonal | ✗ | ✓ |
| MMG-orthogonal-adaptive | ✓ | ✓ |

## 实验关键数据

### 主实验：40任务基准测试成功率

| 方法 | 成功任务数 (/40) |
|---|---|
| MINE | 30 |
| InfoNCE | 33 |
| NWJ | 30 |
| DoE (Gaussian) | 27 |
| MINDE-j | 35 |
| MINDE-c | 35 |
| MMG | 33 |
| MMG-adaptive | 35 |
| MMG-orthogonal | 37 |
| **MMG-orthogonal-adaptive** | **39** |

### 高 MI 场景对比

在 MI $\in [10, 15]$ 的高互信息区间（3×3 稀疏多正态分布）：

| 场景 | MMG-adaptive | MINDE | MMG-orthogonal |
|---|---|---|---|
| 原始分布 | 最准确 | 严重低估 | 保守偏差 |
| Half-cube 变换 | 最准确 | 明显低估 | 保守偏差 |
| Spiral 变换 | 所有方法受挑战 | 低估 | 保守偏差 |

MINDE 在高 MI 场景下低估严重，原因是 score matching 需要逼近尖锐的高频 score function，而神经网络的频谱偏差导致失败。

### 自一致性测试（MNIST, 28×28）

| 测试 | 理想结果 | MMG 表现 |
|---|---|---|
| Baseline: $I(A;B_r)/I(A;B)$ | 随 $r$ 单调趋近 1 | ✓ 通过 |
| Data Processing: $I(A;[B_{r+k},B_r])/I(A;B_{r+k})$ | 恒等于 1 | ✓ 通过 |
| Additivity: $I([A^1,A^2];[B_r^1,B_r^2])/I(A^1;B_r^1)$ | 恒等于 2 | ✓ 通过 |

### 关键发现

- **偏差-方差权衡**：正交原理提供低方差但引入保守偏差（去噪器近似的距离小于真实距离），而直接估计方差高但偏差低。低 MI 场景主要挑战是方差 → 用正交变体；高 MI 场景主要挑战是偏差 → 用 adaptive 变体
- **对离散/连续/混合变量通用**：理论公式对任意变量类型成立
- **不保证上下界**：因为 MMSE 在公式中正负号都有出现，无法保证估计是上界或下界
- 自适应采样显著提升精度，聚焦于去噪器过渡区域避免浪费计算

## 亮点与洞察

- **公式推导优雅**：从 ITD 的密度-MMSE 关系出发，通过简单的相减操作自然得到 MI 的 MMSE gap 表示，理论链条简洁漂亮
- **"MI = 两条 MMSE 曲线间面积"的几何直觉**极其清晰，远优于变分界的抽象表达
- 正交原理的引入解决了"两个大数做差"的数值稳定性问题，是工程上的关键贡献
- 发现偏差-方差权衡并提供两系列变体供用户按场景选择，体现了扎实的实验洞察
- 统一 MI 估计库的发布对社区贡献很大

## 局限与展望

- 无法保证估计的方向性（不是上界也不是下界），依赖于神经网络逼近全局最优的假设
- 正交变体在高 MI 场景的保守偏差是系统性的，目前没有自动检测和切换的机制
- 训练两个去噪器（或一个条件去噪器 + 50% dropout）的计算成本高于简单的变分方法
- 自适应采样的两阶段训练增加了方法复杂度
- 在超高维数据上的可扩展性仍需验证

## 相关工作与启发

- 与 MINDE 的核心区别：MINDE 依赖 score function（密度梯度），MMG 直接使用去噪目标，避免了梯度逼近的困难
- 可以扩展到其他信息论量（如条件熵、传输信息、互信息的其他分解）
- 对扩散模型理论社区的启发：MMSE 视角为理解扩散模型提供了信息论工具箱

## 评分

- **新颖性**: ⭐⭐⭐⭐ — MMSE gap 公式虽然在 Guo (2011) 中有先行版本，但结合现代扩散模型实现和自适应采样是重要贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ — 40任务基准、高 MI 专项、自一致性测试、消融分析，极其全面
- **写作质量**: ⭐⭐⭐⭐⭐ — 理论推导清晰，几何直觉图（Figure 1）出色
- **实用价值**: ⭐⭐⭐⭐ — MI 估计是基础工具，开源库降低了使用门槛

<!-- RELATED:START -->

## 相关论文

- [Information-Theoretic Discrete Diffusion](information-theoretic_discrete_diffusion.md)
- [Information Theoretic Learning for Diffusion Models with Warm Start](information_theoretic_learning_for_diffusion_models_with_warm_start.md)
- [ItDPDM: Information-Theoretic Discrete Poisson Diffusion Model](itdpdm_information-theoretic_discrete_poisson_diffusion_model.md)
- [Learning a Distance Measure from the Information-Estimation Geometry of Data](../../ICLR2026/image_generation/learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [A Connection Between Score Matching and Local Intrinsic Dimension](a_connection_between_score_matching_and_local_intrinsic_dimension.md)

<!-- RELATED:END -->
