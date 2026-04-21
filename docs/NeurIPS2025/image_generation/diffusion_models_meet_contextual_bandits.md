---
title: >-
  [论文解读] Diffusion Models Meet Contextual Bandits
description: >-
  [NeurIPS 2025][图像生成][扩散模型] 将预训练扩散模型作为上下文赌博机 (contextual bandits) 问题中动作参数的表达性先验，提出 diffusion Thompson Sampling (dTS) 算法，通过高效的层次化后验近似实现快速更新与采样，在大动作空间下显著优于传统方法。
tags:
  - NeurIPS 2025
  - 图像生成
  - 扩散模型
  - 上下文赌博机
  - Thompson 采样
  - 贝叶斯先验
  - 后验近似
---

# Diffusion Models Meet Contextual Bandits

**会议**: NeurIPS 2025  
**arXiv**: [2402.10028](https://arxiv.org/abs/2402.10028)  
**代码**: 有 (GitHub)  
**领域**: 图像生成 / 扩散模型 / 在线学习  
**关键词**: 扩散模型, 上下文赌博机, Thompson 采样, 贝叶斯先验, 后验近似

## 一句话总结

将预训练扩散模型作为上下文赌博机 (contextual bandits) 问题中动作参数的表达性先验，提出 diffusion Thompson Sampling (dTS) 算法，通过高效的层次化后验近似实现快速更新与采样，在大动作空间下显著优于传统方法。

## 研究背景与动机

- **上下文赌博机中的挑战**：在大规模动作空间（$K$ 很大）中，标准探索策略（LinUCB、LinTS）面临计算或统计效率瓶颈
- **动作相关性未被利用**：现实中动作之间往往存在相关性（如推荐系统中相似电影），观察一个动作可提供对其他动作的信息
- **扩散模型的潜力**：扩散模型擅长逼近复杂高维分布，可用于编码动作参数间的结构化先验
- **核心思路**：用离线数据预训练扩散模型捕获动作参数的分布结构，作为 Thompson 采样的信息先验，在线交互中高效更新后验

## 方法详解

### 整体框架

#### 问题建模

上下文赌博机 + 扩散模型先验的层次贝叶斯模型：

$$\psi_L \sim \mathcal{N}(0, \Sigma_{L+1})$$
$$\psi_{\ell-1} | \psi_\ell \sim \mathcal{N}(f_\ell(\psi_\ell), \Sigma_\ell), \quad \forall \ell \in [L] \setminus \{1\}$$
$$\theta_a | \psi_1 \sim \mathcal{N}(f_1(\psi_1), \Sigma_1), \quad \forall a \in [K]$$
$$Y_t | X_t, \theta_{A_t} \sim P(\cdot | X_t; \theta_{A_t}), \quad \forall t \in [n]$$

- $\psi_1, \ldots, \psi_L$：$L$ 层隐变量，对应扩散模型的去噪层
- $f_\ell$：链接函数（线性或非线性神经网络）
- $\theta_a$：每个动作的参数向量
- 奖励分布为广义线性模型 (GLM)，均值为 $g(x^\top \theta_a)$

#### dTS 算法（Algorithm 1）

每轮 $t$ 的层次采样：
1. 从后验 $p(\psi_L | H_t)$ 采样顶层隐变量
2. 逐层向下采样 $\psi_{\ell-1} | \psi_\ell, H_t$
3. 给定 $\psi_1$，独立采样每个动作参数 $\theta_a | \psi_1, H_{t,a}$
4. 选择 $A_t = \arg\max_a r(X_t, a; \theta_t)$

### 关键设计

#### 1. 两层后验近似

**问题**：非线性奖发与非线性扩散链接使后验不可解。

**(i) 似然近似**：用 Laplace 式方法将 GLM 似然近似为高斯：

$$\prod_{i \in S_{t,a}} P(Y_i | X_i; \theta_a) \approx \mathcal{N}(\theta_a; \hat{B}_{t,a}, \hat{G}_{t,a}^{-1})$$

其中 $\hat{B}_{t,a}$ 为 MLE，$\hat{G}_{t,a}$ 为负对数似然的 Hessian。

**(ii) 扩散近似**：基于线性情况的闭式解，将线性项 $W_\ell \psi_\ell$ 替换为非线性 $f_\ell(\psi_\ell)$。

#### 2. 闭式后验表达

**动作后验**（精度加权形式）：

$$\hat{\Sigma}_{t,a}^{-1} = \underbrace{\Sigma_1^{-1}}_{\text{先验精度}} + \underbrace{\hat{G}_{t,a}}_{\text{数据精度}}$$

$$\hat{\mu}_{t,a} = \hat{\Sigma}_{t,a}\left(\underbrace{\Sigma_1^{-1} f_1(\psi_1)}_{\text{先验贡献}} + \underbrace{\hat{G}_{t,a} \hat{B}_{t,a}}_{\text{数据贡献}}\right)$$

**隐变量后验**（递归计算）：

$$\bar{\Sigma}_{t,\ell-1}^{-1} = \Sigma_\ell^{-1} + \bar{G}_{t,\ell-1}, \quad \bar{\mu}_{t,\ell-1} = \bar{\Sigma}_{t,\ell-1}\left(\Sigma_\ell^{-1} f_\ell(\psi_\ell) + \bar{B}_{t,\ell-1}\right)$$

信息从动作层通过递推公式 $\bar{G}_{t,\ell}, \bar{B}_{t,\ell}$ 向上传播到所有隐变量层。

#### 3. 计算复杂度优势

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 联合后验 | $O(K^3 d^3)$ | $O(K^2 d^2)$ |
| **dTS** | $O((L+K)d^3)$ | $O((L+K)d^2)$ |
| LinTS (独立) | $O(Kd^3)$ | $O(Kd^2)$ |

当 $K \gg L$ 时，dTS 接近 LinTS 的计算成本，但利用了动作相关性。

### 损失函数 / 训练策略

- **离线预训练**：在历史数据上训练扩散模型，学习 $f_\ell$ 和 $\Sigma_\ell$
- **在线更新**：每轮仅需更新高斯统计量（MLE $\hat{B}_{t,a}$、Hessian $\hat{G}_{t,a}$），无需重训扩散模型
- 后验更新为闭式矩阵运算，计算高效

## 实验关键数据

### 主实验：合成数据

实验设置：$d \in \{5, 20\}$, $L \in \{2, 4\}$, $K \in \{10^2, 10^4\}$, $n = 5000$

| 设置 | dTS vs LinTS | dTS vs HierTS | dTS vs UCB |
|------|-------------|---------------|------------|
| 线性扩散 + 线性奖励 | 显著更低 regret | 显著更低 | 显著更低 |
| 线性扩散 + 非线性奖励 | 显著更低 | - | 显著更低 |
| 非线性扩散 + 线性奖励 | 显著更低 | 显著更低 | 显著更低 |
| 非线性扩散 + 非线性奖励 | 显著更低 | - | 显著更低 |

### 消融实验与扩展

#### K 增大时的优势放大

LinTS/dTS 累积 regret 比值随 $K$ 从 10 增到 $5 \times 10^4$ 而单调增大——动作空间越大，利用相关性的收益越大。

#### 参数敏感性

| 参数 | 效果 |
|------|------|
| $K$ 增大 | regret 增大（需学习更多参数） |
| $d$ 增大 | regret 增大 |
| $L$ 增大 | regret 增大（更多隐变量需学习） |

#### 先验鲁棒性

- 先验参数加噪声 $v \in \{0.5, 1.0, 1.5\}$，dTS 仍优于基线（仅 $v = 1.5$ 时接近持平）
- Swiss Roll 数据：非扩散先验，预训练 $L \approx 40$ 最优
- MovieLens：真实推荐场景，dTS 显著优于 LinTS

### 关键发现

1. **隐变量结构比奖励分布更重要**：用错误奖励分布但正确扩散先验的 dTS，仍优于用正确奖励但忽略结构的 GLM-TS
2. **优势随动作空间增大而放大**：$K = 10^4$ 时 dTS 的优势远大于 $K = 10^2$
3. **对先验误指定鲁棒**：即使在 Swiss Roll（非扩散生成）和 MovieLens（真实数据）上也有效
4. **低预训练数据即可**：仅 50 个样本预训练就能使 dTS 性能翻倍超过 LinTS

## 亮点与洞察

- **概念创新**：将扩散模型从"生成器"转变为"贝叶斯先验编码器"，这是一个全新的角色定位
- **理论-实践统一**：线性情况有精确后验和 regret 上界，非线性情况有基于精确解的自然近似
- **两层近似的优雅性**：似然近似保留了扩散层次结构的表达力（vs 全局 Laplace 近似），扩散近似利用了闭式解的形式
- **Bayes regret 分析**：dTS 的 regret 中动作项仅含 $K\sigma_1^2$（条件方差），LinTS 含 $K\Sigma$（边际方差，远大于条件方差）
- **稀疏性加速**：当混合矩阵有列稀疏时，隐变量学习的维度从 $d$ 降到 $d_\ell$

## 局限与展望

1. **理论仅限线性情况**：Bayes regret 上界仅对线性高斯设定成立，非线性情况缺乏理论保证
2. **近似误差无界**：似然近似和扩散近似的误差未被量化
3. **依赖预训练质量**：离线数据不足或有偏时，先验可能欠正则化
4. **未考虑时变环境**：假设动作参数固定（static bandit），非平稳设定有待扩展
5. **缺乏 matching lower bound**：贝叶斯下界仍是开放问题

## 相关工作与启发

- **HierTS** (Hong et al., 2022)：层次贝叶斯赌博机，但仅支持线性先验；dTS 推广到非线性扩散先验
- **Hsieh et al. (2023)**：多臂赌博机 + 扩散先验，本文是首个上下文赌博机版本
- **Kveton et al. (2024)**：类似方向的并行工作
- **启发**：预训练生成模型作为结构化先验的思路可推广到 RL、贝叶斯优化等更广泛的在线决策问题

## 评分

| 维度 | 分数 | 评价 |
|------|------|------|
| 新颖性 | ★★★★★ | 首次将扩散模型作为上下文赌博机的先验，概念新颖 |
| 技术深度 | ★★★★☆ | 后验推导严谨，理论分析（线性情况）完整 |
| 实验充分性 | ★★★★☆ | 合成+真实数据，多维度消融，但大规模场景有限 |
| 实用价值 | ★★★★☆ | 推荐系统、广告等大动作空间场景有直接应用 |
| 写作质量 | ★★★★☆ | 结构清晰，公式推导详尽，但篇幅较长 |

<!-- RELATED:START -->

## 相关论文

- [Contextual Thompson Sampling via Generation of Missing Data](contextual_thompson_sampling_via_generation_of_missing_data.md)
- [Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [When Are Concepts Erased From Diffusion Models?](when_are_concepts_erased_from_diffusion_models.md)
- [Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)

<!-- RELATED:END -->
