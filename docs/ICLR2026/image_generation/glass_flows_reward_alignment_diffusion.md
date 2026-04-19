---
title: >-
  [论文解读] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models
description: >-
  [ICLR 2026 Oral][图像生成][flow matching] 提出 GLASS (Gaussian Latent Sufficient Statistic) Flows——一种"流模型中的流模型"新采样范式，通过高斯充分统计量重参数化将随机马尔可夫转移 $p_{t'|t}(x_{t'} | x_t)$ 重铸为内部 ODE 求解问题（复用预训练去噪器，无需重训），在无需权衡 ODE 效率和 SDE 随机性的条件下实现 Feynman-Kac Steering，在 FLUX 文生图模型上一致超越 Best-of-N ODE 基线，刷新推理时奖励对齐 SOTA。
tags:
  - ICLR 2026 Oral
  - 图像生成
  - flow matching
  - 扩散模型
  - reward alignment
  - Feynman-Kac steering
  - GLASS
  - stochastic transitions
  - inference-time scaling
---

# GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models

**会议**: ICLR 2026 Oral  
**OpenReview**: [vH7OAPZ2dR](https://openreview.net/forum?id=vH7OAPZ2dR)  
**代码**: 有  
**领域**: 图像生成 / 扩散模型  
**关键词**: flow matching, diffusion models, reward alignment, Feynman-Kac steering, GLASS, stochastic transitions, inference-time scaling

## 一句话总结
提出 GLASS (Gaussian Latent Sufficient Statistic) Flows——一种"流模型中的流模型"新采样范式，通过高斯充分统计量重参数化将随机马尔可夫转移 $p_{t'|t}(x_{t'} | x_t)$ 重铸为内部 ODE 求解问题（复用预训练去噪器，无需重训），在无需权衡 ODE 效率和 SDE 随机性的条件下实现 Feynman-Kac Steering，在 FLUX 文生图模型上一致超越 Best-of-N ODE 基线，刷新推理时奖励对齐 SOTA。

## 研究背景与动机
**领域现状**：流匹配/扩散模型在推理时可通过奖励适配算法增强性能（推理时缩放 inference-time scaling）。现有方法如 Sequential Monte Carlo (SMC)、Feynman-Kac Steering (FKS) 需要在去噪轨迹中引入随机性来探索高奖励区域。

**现有痛点**：随机转移（SDE 采样）效率远低于确定性 ODE 采样，且在少步数情况下质量严重下降。实验表明标准 FKS 使用 SDE 转移甚至无法超越简单的 Best-of-N ODE 基线——效率与随机性之间存在根本矛盾。

**核心矛盾**：FKS/SMC 等方法理论上需要 SDE 提供的随机分支来有效探索后验分布，但 SDE 的计算和质量代价使其在实际 SOTA 模型上不可行。Best-of-N ODE 足够高效但不利用中间步骤的奖励信号。

**本文目标** 消除效率与随机性之间的权衡——让 ODE 采样也能产生丰富的随机转移，使 FKS 真正有效。

**切入角度**：观察到高斯转移核 $p_{t'|t}$ 可以通过充分统计量和时间重参数化，转化为用预训练去噪器驱动的内部条件流匹配 ODE。

**核心 idea**：将随机转移重铸为"内部流匹配"ODE，通过充分统计量复用预训练模型，实现"ODE 速度 + SDE 多样性"。

## 方法详解

### 整体框架
给定预训练流匹配模型的速度场 $u_t(x)$ 和去噪器 $D_t(x)$。GLASS Flows 将两步转移 $x_t \to x_{t'}$ 视为一个条件生成问题：引入辅助变量 $\bar{X}_s$（$s \in [0,1]$），构建内部流 ODE $\frac{d\bar{x}_s}{ds} = \bar{u}_s(\bar{x}_s | x_t, t)$，其中 $\bar{x}_0 \sim \mathcal{N}(\bar{\gamma} x_t, \bar{\sigma}_0^2 I)$（随机初始条件提供随机性），$\bar{x}_1 \sim p_{t'|t}(\cdot | x_t)$（终态服从目标转移分布）。

### 关键设计
1. **GLASS 转移核构造**:
    - 功能：定义一族由相关参数 $\rho$ 参数化的高斯马尔可夫转移 $p_{t'|t}^{\text{GLASS}}$
    - 核心思路：将 $(X_t, X_{t'})$ 建模为潜变量 $Z$ 的两个"含噪观测"：$X_t = \alpha_t Z + \sigma_t \epsilon_1$，$X_{t'} = \alpha_{t'} Z + \sigma_{t'} \epsilon_2$，其中 $\text{Corr}(\epsilon_1, \epsilon_2) = \rho$。联合分布为
    $$\begin{pmatrix} X_t \\ X_{t'} \end{pmatrix} = \begin{pmatrix} \alpha_t \\ \alpha_{t'} \end{pmatrix} Z + \begin{pmatrix} \sigma_t \epsilon_1 \\ \sigma_{t'} \epsilon_2 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} \sigma_t^2 & \rho \sigma_t \sigma_{t'} \\ \rho \sigma_t \sigma_{t'} & \sigma_{t'}^2 \end{pmatrix}$$
    - 设计动机：$\rho$ 控制随机性强度。当 $\rho = \alpha_t \sigma_{t'} / (\sigma_t \alpha_{t'})$ 时退化为 DDPM 转移；$\rho = 1$ 时退化为确定性 ODE。默认 $\rho = 0.4$ 实验最优

2. **充分统计量重参数化（核心贡献）**:
    - 功能：证明 GLASS 去噪器可直接由预训练去噪器 $D_t$ 表示，无需重训
    - 核心思路：定义充分统计量 $S(\mathbf{x}) = \frac{\mu^\top \Sigma^{-1}}{\mu^\top \Sigma^{-1} \mu} \begin{pmatrix} x_t \\ \bar{x}_s \end{pmatrix}$，其中 $\mu = (\alpha_t, \bar{\alpha}_s + \bar{\gamma}\alpha_t)^\top$。则 GLASS 去噪器为
    $$D_{\mu, \Sigma}(x_t, \bar{x}_s) = D_{t^\star}(\alpha_{t^\star} S(\mathbf{x}))$$
    这里 $t^\star = g^{-1}((\mu^\top \Sigma^{-1} \mu)^{-1})$，$g(t) = \sigma_t^2 / \alpha_t^2$ 是信噪比函数。即：将两个含噪观测压缩为充分统计量 → 用预训练去噪器在等效时间 $t^\star$ 上去噪
    - 设计动机：高斯共轭/充分统计量的数学结构保证了精确的训练无关重参数化，无需任何额外训练

3. **内部条件流 ODE**:
    - 功能：将 GLASS 去噪器代入条件流匹配框架，得到内部 ODE 速度场 $\bar{u}_s(\bar{x}_s | x_t, t)$
    - 核心思路：选择 CondOT 调度 $\bar{\alpha}_s = s \bar{\alpha}_1$，$\bar{\sigma}_s = (1-s) \bar{\sigma}_0 + s \bar{\sigma}_1$，速度场为
    $$\bar{u}_s = w_1(s) \bar{x}_s + w_2(s) D_{\mu(s), \Sigma(s)}(x_t, \bar{x}_s)$$
    其中 $w_1(s) = \frac{\dot{\bar{\sigma}}_s}{\bar{\sigma}_s}$，$w_2(s) = \dot{\bar{\alpha}}_s - \bar{\alpha}_s \frac{\dot{\bar{\sigma}}_s}{\bar{\sigma}_s}$。用 Euler 方法积分 $M$ 步即可采样
    - 设计动机：每步只需 1 次神经网络调用（与 SDE 相同），但利用 ODE 积分器的稳定性获得更好质量

### 即插即用应用
- **FKS-GLASS**：用 GLASS 转移替换 Feynman-Kac Steering 中的 SDE 转移，粒子 reweight + resample
- **GLASS + 梯度引导**：在内部 ODE 中添加奖励梯度 $\nabla_y r(D_{t^\star}(y))\big|_{y=\alpha_{t^\star}S(\mathbf{x})}$
- **总 NFE = K × M**：$K$ 个外部步 × 每步 $M$ 个内部 ODE 步。与 SDE 公平对比（相同总 NFE）

## 实验关键数据

### 主实验：FLUX 文生图奖励对齐（GenEval + PartiPrompts）
| 方法 | CLIP ↑ | PickScore ↑ | HPSv2 ↑ | ImageReward ↑ | GenEval ↑ |
|------|--------|------------|---------|--------------|-----------|
| ODE (Best-of-8) | 基线 | 基线 | 基线 | 基线 | 基线 |
| FKS + SDE | < Best-of-8 | < Best-of-8 | < Best-of-8 | < Best-of-8 | ≈ |
| FKS + GLASS | **> Best-of-8** | **> Best-of-8** | **> Best-of-8** | **> Best-of-8** | **提升** |
| FKS + GLASS + Guidance | **最高** | **最高** | **最高** | **最高** | **最高** |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Best-of-N (ODE) | 基线 | 简单但无法利用中间奖励信号 |
| Best-of-N (GLASS) | ≈ Best-of-N (ODE) | 同分布采样，终点质量一致 |
| FKS + SDE | < Best-of-N (ODE) | SDE 质量太差，拖累 FKS |
| $\rho = 0.2, 0.4, 0.6, 0.8, 1.0$ | $\rho = 0.4$ 最优 | 所有 $\rho$ 值均达到 ODE 级质量 |
| DreamSim Diversity | ODE ≈ SDE ≈ GLASS | 三种方法采样自同一边际分布 |
| SiT-XL ImageNet-256 | GLASS FID ≈ ODE FID | 在非 FLUX 模型上同样有效 |

### 关键发现
- **FKS + SDE 在 FLUX 上不工作**：标准 SDE 转移在 FLUX 的 50 步 ODE 配置下严重降质（产生残余噪声），甚至不如 Best-of-N ODE
- **GLASS 消除了效率-随机性权衡**：FKS-GLASS 在所有 4 个奖励模型 × 2 个 benchmark 上一致超越 Best-of-N ODE，而 FKS-SDE 做不到
- **GLASS 的随机性来源是初始条件**：$\bar{X}_0 \sim \mathcal{N}(\bar{\gamma} x_t, \bar{\sigma}_0^2 I)$ 提供随机分支，后续演化是确定性 ODE——不同于 SDE 的逐步注入噪声
- **GLASS 精确保持边际分布**：理论证明组合 GLASS 转移后 $X_{t_k} \sim p_{t_k}$，对任意 $\rho$ 都成立

## 亮点与洞察
- **"流中流"的概念原创且优雅**：将一步随机转移视为一个完整的条件流匹配问题，概念简洁、实现简单
- **充分统计量构造的数学美感**：利用高斯共轭将两个含噪观测压缩为一个，精确复用预训练去噪器，零额外训练
- **解决了一个实际痛点**：FKS/SMC 在 SOTA 模型上因 SDE 质量差而不可用，GLASS 直接解决了此瓶颈
- **drop-in replacement 的实用性**：不改变模型、不重训、只改采样器——任何使用 SDE 转移的现有方法都可受益
- **与 RL fine-tuning 互补**：GLASS 可加速 DDPO/Flow-GRPO 等 RL 训练中的 SDE 采样，也可应用于已 fine-tune 模型的推理

## 局限与展望
- 依赖高斯转移核假设，对非高斯架构的适用性未验证
- $\rho$ 目前为常数，理论上可做时间依赖的自适应 $\rho(t, t')$
- 仅在 FLUX 和 SiT-XL 上验证，其他架构（SD3、Stable Cascade）未测试
- 内部 ODE 步数 $M$ 增大会增加计算量，最优 $M$ 的选择依赖任务
- 未与离散时间扩散模型（如 DDPM 的标准离散调度）深入对比数值稳定性

## 相关工作与启发
- **vs DDPM/SDE 采样**: GLASS 在连续极限下精确采样相同转移分布，但用 ODE 积分器代替 SDE，质量和效率同时提升
- **vs TADA (arXiv:2506.21757)**: TADA 使用相同的高斯共轭/充分统计量数学工具，但用于不同目的（augmented dynamics for training-free improvement）。GLASS 用于构建随机转移替代 SDE
- **vs Transition Matching (DTM)**: DTM 是 $\rho = 1$ 的特例，但 DTM 是不同的预训练范式（per-patch 近似），不直接可比
- **vs Best-of-N**: Best-of-N 只用终点奖励且各采样独立，GLASS+FKS 利用中间步骤奖励和粒子重采样

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "流中流"概念和充分统计量构造极具原创性，数学优雅
- 实验充分度: ⭐⭐⭐⭐ 在 FLUX 768×1360 上的 4 奖励模型 × 2 benchmark 验证有说服力，但缺少更多架构对比
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨清晰，从直觉到形式化层层递进，reviewer dnmw 评 "excellent soundness"
- 价值: ⭐⭐⭐⭐⭐ 直接解决了推理时奖励对齐的实际瓶颈，作为 drop-in replacement 的实用性极强

<!-- RELATED:START -->

## 相关论文

- [DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)
- [Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)
- [Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models](../../NeurIPS2025/image_generation/psi-sampler_initial_particle_sampling_for_smc-based_inference-time_reward_alignm.md)
- [Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](../../NeurIPS2025/image_generation/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)

<!-- RELATED:END -->
