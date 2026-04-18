---
title: >-
  [论文解读] Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is also a GAN Discriminator
description: >-
  [ICML 2025][图像生成][生成模型微调] DDO 提出将似然模型本身参数化为 GAN 判别器（通过似然比），无需额外判别器网络即可用 GAN 目标微调预训练的扩散/自回归模型，在 CIFAR-10 和 ImageNet 上大幅刷新 FID 记录（EDM: 1.97→1.38, EDM2-S: 1.58→0.97）。
tags:
  - ICML 2025
  - 图像生成
  - 生成模型微调
  - GAN
  - 似然比参数化
  - 扩散模型
  - 自回归模型
---

# Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is also a GAN Discriminator

**会议**: ICML 2025  
**arXiv**: [2503.01103](https://arxiv.org/abs/2503.01103)  
**代码**: https://research.nvidia.com/labs/dir/ddo/ (有)  
**领域**: Image Generation  
**关键词**: 生成模型微调, GAN判别器, 似然比参数化, 扩散模型, 自回归模型

## 一句话总结
DDO 提出将似然模型本身参数化为 GAN 判别器（通过似然比），无需额外判别器网络即可用 GAN 目标微调预训练的扩散/自回归模型，在 CIFAR-10 和 ImageNet 上大幅刷新 FID 记录（EDM: 1.97→1.38, EDM2-S: 1.58→0.97）。

## 研究背景与动机

**领域现状**：扩散模型和自回归模型是当前视觉生成的主流范式，以稳定性和可扩展性著称，已在图像和视频合成中取得卓越成就。

**现有痛点**：这些似然模型优化的是前向 KL 散度 $\min_\theta D_{\text{KL}}(p_{\text{data}} \| p_\theta)$，这天然具有"模式覆盖"(mode-covering)倾向——在有限模型容量下，学习到的密度会过度扩散，导致生成样本可能模糊。因此它们严重依赖 CFG 等引导方法来提升生成质量。

**核心矛盾**：GAN 优化 JS 散度，倾向于生成更锐利、更真实的样本，但训练不稳定且容易模式崩塌。如何将 GAN 的锐化优势引入已训练好的似然模型，同时避免 GAN 的工程复杂性？

**本文目标**：在不改变网络结构、不增加推理成本的前提下，利用 GAN 目标对预训练的似然生成模型进行微调，突破 MLE 的质量上限。

**切入角度**：关键洞察——似然模型本身就能充当 GAN 判别器！通过两个似然模型的似然比来隐式参数化判别器，类似 DPO 中用策略对数比来参数化奖励模型。

**核心 idea**：用 $d_\theta(\mathbf{x}) = \sigma\left(\log \frac{p_\theta(\mathbf{x})}{p_{\theta_{\text{ref}}}(\mathbf{x})}\right)$ 隐式参数化判别器，将 GAN 训练简化为直接微调生成模型本身。

## 方法详解

### 整体框架

DDO 的 pipeline 非常简洁：
- **输入**：预训练的似然生成模型 $p_{\theta_{\text{ref}}}$（作为固定参考模型）、训练数据集（真实样本）
- **过程**：初始化 $\theta = \theta_{\text{ref}}$，用参考模型生成假样本，用 GAN 判别器损失训练 $\theta$
- **输出**：微调后的模型 $p_\theta$，直接替换原模型用于推理，无任何额外开销

### 关键设计

1. **隐式判别器参数化**:

    - 在标准 GAN 中，最优判别器为 $d^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_{\theta_{\text{ref}}}(\mathbf{x})} = \sigma\left(\log \frac{p_{\text{data}}(\mathbf{x})}{p_{\theta_{\text{ref}}}(\mathbf{x})}\right)$
    - DDO 的核心思路：用可学习的生成模型 $p_\theta$ 替代未知的 $p_{\text{data}}$，定义判别器为 $d_\theta(\mathbf{x}) = \sigma\left(\log \frac{p_\theta(\mathbf{x})}{p_{\theta_{\text{ref}}}(\mathbf{x})}\right)$
    - **定理保证**：当 $p_\theta^* = p_{\text{data}}$ 时损失最小，即最优解仍然是匹配数据分布
    - **设计动机**：这种参数化消除了对独立判别器网络的需要，也不需要对生成过程做反向传播（对扩散模型而言这非常昂贵）

2. **广义目标函数与超参数控制**:

    - 由于似然模型的 $\log p_\theta(\mathbf{x})$ 可达 $10^3$ 量级，直接用 Sigmoid 会梯度消失
    - 引入超参数 $\alpha, \beta$：$\mathcal{L}_{\alpha,\beta}(\theta) = -\mathbb{E}_{p_{\text{data}}}[\log \sigma(\beta \log \frac{p_\theta}{p_{\theta_{\text{ref}}}})] - \alpha \mathbb{E}_{p_{\theta_{\text{ref}}}}[\log(1 - \sigma(\beta \log \frac{p_\theta}{p_{\theta_{\text{ref}}}})]$
    - $\beta$ 控制概率比的缩放，$\alpha$ 控制两项损失的相对权重
    - 当 $\beta < 1$ 时最优解会"过冲"数据分布（$p_\theta^* \propto p_{\theta_{\text{ref}}}^{1-1/\beta} p_{\text{data}}^{1/\beta}$），与引导方法在理论上相通

3. **扩散模型的单步近似**:

    - 扩散模型的似然比需要多时间步的 ELBO 近似：$\log \frac{p_\theta}{p_{\theta_{\text{ref}}}} \approx \mathbb{E}_{t,\epsilon}[\Delta_{\mathbf{x}_t, t, \epsilon}]$
    - 其中 $\Delta = -w(t)(||\epsilon_\theta(\mathbf{x}_t,t) - \epsilon||^2 - ||\epsilon_{\theta_{\text{ref}}}(\mathbf{x}_t,t) - \epsilon||^2)$
    - 利用 Jensen 不等式得到上界，使得每个样本只需一次前向传播
    - **设计动机**：避免多时间步计算的高昂开销，使扩散 DDO 的计算量与标准训练相当

4. **多轮自博弈精炼**:

    - 每轮微调后，将最优模型作为下一轮的参考模型：$p_{\theta_{n-1}^*} \to p_{\theta_n}$
    - 类似 Iterative DPO 和 SPIN，但不直接更新参考模型
    - 每轮仅需不到预训练 1% 的迭代量
    - **设计动机**：单轮 DDO 提供有用梯度但不会收敛到数据分布，多轮迭代逐步逼近

### 损失函数 / 训练策略

- **扩散模型 (EDM-DDO)**：利用 F-parameterization，损失为
  $\mathcal{L}_{\alpha,\beta}^{\text{EDM-DDO}} = -\mathbb{E}_{t,\epsilon}[\mathbb{E}_{p_{\text{data}}} \log \sigma(-\beta(\|F_\theta - \hat{F}\|^2 - \|F_{\theta_{\text{ref}}} - \hat{F}\|^2)) + \alpha \mathbb{E}_{p_{\theta_{\text{ref}}}} \log \sigma(\beta(\|F_\theta - \hat{F}\|^2 - \|F_{\theta_{\text{ref}}} - \hat{F}\|^2))]$
- **自回归模型 (VAR)**：直接使用 next-token 对数似然比，在线生成参考样本，保留 label dropout 以兼容 CFG
- 禁用混合精度（扩散模型）以保持数值稳定；禁用所有 dropout 层
- 每轮在 $\alpha \in [0.5, 6.0], \beta \in [0.01, 0.1]$（扩散）或 $\alpha \in [10, 100], \beta = 0.02$（VAR）范围内网格搜索

## 实验关键数据

### 主实验

**CIFAR-10 (FID↓)**

| 方法 | NFE | 无条件 FID | 类条件 FID |
|------|-----|-----------|-----------|
| EDM (基线) | 35 | 1.97 | 1.85 |
| EDM + DG | 53 | 1.77 | 1.64 |
| **EDM + DDO** | **35** | **1.38** | **1.30** |
| StyleGAN-XL | 1 | - | 1.85 |

**ImageNet-64 (FID↓)**

| 方法 | NFE | FID |
|------|-----|-----|
| EDM2-S (基线) | 63 | 1.58 |
| EDM2-S + AG | 126 | 1.01 |
| **EDM2-S + DDO** | **63** | **0.97** |
| EDM2-XL | 63 | 1.33 |

**ImageNet 256×256 (VAR, FID↓)**

| 方法 | w/o CFG | w/ CFG |
|------|---------|--------|
| VAR-d30 (有 trick) | 2.17 | 1.90 |
| VAR-d30 (无 trick) | 4.74 | 1.92 |
| **VAR-d30 + DDO** | **1.79** | **1.73** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| $\alpha \in [0.5, 6.0]$, $\beta = 0.05$ | FID 均改善 | 大范围 $\alpha$ 一致有效 |
| $\beta \in [0.01, 0.1]$, $\alpha = 4.0$ | FID 均改善 | 大范围 $\beta$ 一致有效 |
| 单轮 DDO (CIFAR-10) | FID 1.72/1.58 | 单轮即超 DG |
| EDM2-S 3轮 (ImageNet-64) | FID 1.31 | 280M 模型超越 1119M 的 EDM2-XL |
| VAR-d16 + DDO 无 CFG | FID 3.12 → 超越 CFG 基线 3.30 | 推理成本减半 |

### 关键发现

- **记录性 FID**：CIFAR-10 1.30, ImageNet-64 0.97，均为新 SOTA
- **效率惊人**：每轮微调仅需预训练 <1% 的迭代量，EDM 每轮约 3 小时
- **消除采样技巧**：DDO 微调后的 VAR 无需 top-k/top-p 即可获得更好的 FID
- **消除 CFG 依赖**：VAR-d30 + DDO 无引导 FID=1.79，优于原始 CFG 增强的 1.90
- **参数效率**：EDM2-S (280M) + DDO 超越 4 倍大的 EDM2-XL (1119M)

## 亮点与洞察

1. **优雅的理论框架**：用似然比参数化判别器，在 DPO 和 GAN 之间建立深刻联系
2. **与引导方法的统一视角**：DDO ($\beta < 1$) 等价于 $p_\theta^* \propto p_{\text{ref}}^{1-1/\beta} p_{\text{data}}^{1/\beta}$，与 CFG/AG 的"分布锐化"本质相同
3. **零推理开销**：不同于 DG/AG/CFG 需要额外模型或多次前向传播，DDO 直接替换原模型
4. **通用性**：同一框架同时适用于连续（扩散）和离散（自回归）生成模型

## 局限与展望

1. 超参数 $\alpha, \beta$ 需要网格搜索，缺乏自动调参策略
2. 多轮精炼带来额外训练成本（尽管每轮很短），需 ~20 节点并行搜索
3. 当前仅验证在类条件图像生成上，未扩展到文本到图像等更复杂任务
4. 理论分析依赖有界似然比假设，在强分布偏移下可能不成立

## 相关工作与启发

- **DPO → DDO 的迁移**：DPO 用对数策略比参数化奖励模型→DDO 用对数似然比参数化判别器，但 DDO 面向分布对齐而非偏好学习
- **与蒸馏互补**：DDO 提升模型质量，蒸馏提升推理速度，两者可级联使用
- **启发**：这种"模型即判别器"的思想可能推广到音频、3D、视频等其他模态

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将似然模型隐式参数化为 GAN 判别器的 insight 非常优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 扩散+自回归双验证，多数据集多轮消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论严谨，动机清晰，论述流畅
- 价值: ⭐⭐⭐⭐⭐ 提供了后训练提升生成质量的简洁统一方案，实用性极高
