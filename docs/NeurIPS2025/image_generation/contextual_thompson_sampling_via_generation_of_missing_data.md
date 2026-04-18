---
title: >-
  [论文解读] Contextual Thompson Sampling via Generation of Missing Data
description: >-
  [NeurIPS 2025][图像生成][Thompson sampling] 提出 Generative Thompson Sampling (TS-Gen)，将上下文老虎机中的不确定性建模为缺失数据而非未知参数，通过生成模型对缺失结果做自回归填充来实现 Thompson 采样，建立了与离线预测损失直接挂钩的遗憾界。
tags:
  - NeurIPS 2025
  - 图像生成
  - Thompson sampling
  - contextual bandit
  - 生成模型
  - missing data imputation
  - 元学习
  - 遗憾界
---

# Contextual Thompson Sampling via Generation of Missing Data

**会议**: NeurIPS 2025  
**arXiv**: [2502.07064](https://arxiv.org/abs/2502.07064)  
**代码**: 待确认  
**领域**: 生成模型 / 决策系统 / 上下文老虎机  
**关键词**: Thompson sampling, contextual bandit, 生成模型, missing data imputation, 元学习, 遗憾界

## 一句话总结
提出 Generative Thompson Sampling (TS-Gen)，将上下文老虎机中的不确定性建模为缺失数据而非未知参数，通过生成模型对缺失结果做自回归填充来实现 Thompson 采样，建立了与离线预测损失直接挂钩的遗憾界。

## 研究背景与动机

**领域现状**：经典 Thompson sampling (TS) 将环境不确定性归因于未知潜在参数 $\theta$，依赖三个操作：设定先验、从后验采样、更新后验。然而这三个操作在神经网络场景下都非常困难（后验不可追踪、先验设定不直观、在线更新成本高）。

**现有痛点**：(a) 神经网络的贝叶斯不确定性量化方法（变分推断、MC dropout、集成等）效果有限且计算昂贵；(b) 已有的 TS 理论大多限于参数化模型（线性/逻辑回归），无法自然扩展到使用基础模型的复杂工业场景；(c) 元学习设置下如何利用历史任务的先验信息缺乏系统化框架。

**核心矛盾**：如何让 Thompson sampling 兼容现代深度学习（序列模型、基础模型），同时保持可证明的理论保证？

**切入角度**：不把不确定性看作未知参数，而看作缺失的可观测数据（反事实结果）。如果所有 action 的结果都已知，就没有不确定性——直接用完整数据拟合最优策略即可。因此在每个决策点，用生成模型填充缺失结果、在补全数据集上拟合策略、按该策略选择行动。

## 方法详解

### 核心框架：缺失数据视角

将任务 $\tau$ 定义为潜在结果表 (potential outcomes table)，包含任务信息 $Z_\tau$、上下文序列 $X_{1:T}$、所有 action 的潜在结果 $\{Y_t^{(a)}\}$。观察到的只是被选中 action 的结果，其余全是"缺失数据"。

**Algorithm 1: Generative Thompson Sampling**

每个决策时刻 $t$：
1. 观察上下文 $X_t$
2. 用生成模型 $p$ 从条件分布采样补全任务：$\hat{\tau}_t \sim p(\tau \in \cdot \mid \mathcal{H}_t)$
3. 在补全数据集 $\hat{\tau}_t$ 上拟合最优策略 $\pi^*(\cdot; \hat{\tau}_t)$
4. 选择动作 $A_t \leftarrow \pi^*(X_t; \hat{\tau}_t)$
5. 观察结果 $Y_t = Y_t^{(A_t)}$，更新历史

### 自回归后验采样 (Algorithm 3)

对每个 action $a$，将观察到的结果排在前面、缺失结果排在后面，然后用序列模型 $p_\theta$ 自回归生成缺失结果：

$$\hat{Y}_i^{(a)} \sim p_\theta(\cdot \mid Z, \{\hat{X}_j, \hat{Y}_j^{(a)}\}_{j \prec_a i}, \hat{X}_i)$$

这种排序保证模型始终先条件于所有已观察结果再生成缺失值，实现了正确的后验采样。

### 离线训练 (Algorithm 2)

在历史任务数据 $\mathcal{D}^{\text{offline}}$ 上用标准自回归损失训练序列模型：

$$\ell(p_\theta) = -\mathbb{E}\left[\log p_\theta(X_{1:T}, \{Y_{1:t-1}^{(a)}\}_{a \in \mathcal{A}_\tau} \mid Z_\tau)\right]$$

实际中可以做简化假设（上下文独立于历史结果、各 action 结果独立），用 mini-batch SGD 优化。

### 理论保证

**Theorem 1（完美模型遗憾界）**：使用真实分布 $p^*$ 时，

$$\Delta(\mathbb{A}_{\text{TS-Gen}}(p^*)) \leq \sqrt{\frac{|\mathcal{A}_\tau|}{2T} \cdot H(\boldsymbol{\pi}^*(X_{1:T}) \mid Z_\tau)}$$

**Theorem 2（近似模型遗憾界）**：使用近似模型 $p_\theta$ 时，

$$\Delta(\mathbb{A}_{\text{TS-Gen}}(p_\theta)) \leq \underbrace{\sqrt{\frac{|\mathcal{A}_\tau|}{2T} \cdot H(\boldsymbol{\pi}^*(X_{1:T}) \mid Z_\tau)}}_{\text{完美 TS 遗憾}} + \underbrace{\sqrt{2\{\ell(p_\theta) - \ell(p^*)\}}}_{\text{模型质量罚项}}$$

生成模型质量仅通过离线预测损失影响遗憾界——与模型架构无关，可用任何序列模型。

**Proposition 2（复杂度界）**：通过 VC 维上界熵：$H(\boldsymbol{\pi}^*(X_{1:T}) \mid Z_\tau) = O(d \cdot \log(T \cdot |\mathcal{A}_\tau|))$，直接适用于无穷策略类。

## 实验关键数据

### 设置
$T=500$，$|\mathcal{A}|=10$，二值结果 $Y \in \{0,1\}$，先验信息 $Z^{(a)} \in \mathbb{R}^2$。

| 方法 | 累计遗憾（合成） | 累计遗憾（半合成/新闻推荐） |
|------|----------------|------------------------|
| **TS-Gen** | **最低** | **最低** |
| Greedy | 高 | 高 |
| ε-Greedy | 中高 | 中高 |
| TS-Neural-Linear (uninformative) | 中 | 中 |
| TS-Neural-Linear (fitted prior) | 中 | 中 |
| LinUCB | 中高 | 中高 |
| TS-Linear | 中高 | 高 |

- TS-Gen 在两种设置下均优于所有基线
- 相同 $p_\theta$ 模型但不同决策策略（Greedy、ε-Greedy、TS-Neural-Linear）表现更差，验证了自回归生成方法在不确定性量化上的优势
- 离线预测损失越低，TS-Gen 遗憾越低（与 Theorem 2 一致）
- 计算成本：半合成实验中每次决策生成 4.2s + 策略拟合 2.2s（CPU）

## 亮点与洞察

- **缺失数据视角重新定义了 TS 的原语操作**：从"设先验-采后验-更新后验"变为"最小化预测损失-自回归生成-拟合策略"，天然兼容现代深度学习
- **模块化设计**：生成模型 $p_\theta$ 和策略类 $\Pi$ 完全解耦——可以独立换生成模型架构或策略拟合方法（甚至加公平性约束）
- **理论贡献**：首次用信息论方法对无穷策略类的上下文 TS 推导遗憾界，无需离散化参数空间
- **在上下文学习 (ICL) 意义下的元学习**：$p_\theta$ 训练后不需要在线梯度更新，通过条件化历史观测来"在上下文中学习"——与 LLM 的 ICL 能力天然契合

## 局限与展望

- **仅在两个实验设置上验证**：合成 + 半合成，缺少大规模真实场景
- **训练数据要求**：需要完整任务数据集训练 $p_\theta$，实际可能只有部分数据（用 bootstrap 近似，缺乏理论保证）
- **计算成本较高**：每次决策都需自回归生成完整任务数据 + 拟合策略，实时性受限
- **策略类选择没有指导**：如何选择合适的 $\Pi$ 是开放问题
- **未探索 MDP 扩展**：当前框架限于 bandit 设置

## 相关工作对比

- **vs 经典 TS**：经典 TS 需要参数化模型和后验推断，TS-Gen 用生成模型填充缺失数据替代
- **vs Deep Ensemble TS**：集成方法需要在线梯度更新且不能利用任务元信息 $Z$，TS-Gen 通过离线预训练利用 $Z$
- **vs 模仿学习 (Decision Transformer 等)**：模仿学习直接预测最优动作，TS-Gen 预测未来结果——后者能进行不确定性量化
- **vs Wen et al. 2021**：非上下文、需要建模潜在参数，TS-Gen 扩展到上下文设置且无需潜在参数

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 缺失数据视角重构 TS 非常新颖，理论贡献扎实
- 实验充分度: ⭐⭐⭐ 作为概念验证可以接受，但只有两个实验场景，缺少真实大规模验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、理论严谨、问题建模优雅
- 价值: ⭐⭐⭐⭐ 打通了生成式 AI 与在线决策的理论桥梁，具有重要的方向性意义
