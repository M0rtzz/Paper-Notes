---
title: >-
  [论文解读] Gaussian Mixture Flow Matching Models
description: >-
  [ICML 2025][图像生成][高斯混合模型] 提出高斯混合流匹配模型（GMFlow），用动态高斯混合分布替代传统的单高斯去噪分布来建模多模态流速度场，通过 KL 散度损失训练，并推导出 GM-SDE/ODE 求解器实现精确少步采样，同时引入概率引导方案解决 CFG 过饱和问题…
tags:
  - "ICML 2025"
  - "图像生成"
  - "高斯混合模型"
  - "Flow Matching"
  - "少步采样"
  - "概率引导"
  - "CFG 过饱和修复"
---

# Gaussian Mixture Flow Matching Models

**会议**: ICML 2025  
**arXiv**: [2504.05304](https://arxiv.org/abs/2504.05304)  
**代码**: [https://github.com/Hschen1995/GMFlow](https://github.com/Hschen1995/GMFlow)  
**领域**: 流匹配 / 图像生成  
**关键词**: 高斯混合模型, Flow Matching, 少步采样, 概率引导, CFG 过饱和修复

## 一句话总结
提出高斯混合流匹配模型（GMFlow），用动态高斯混合分布替代传统的单高斯去噪分布来建模多模态流速度场，通过 KL 散度损失训练，并推导出 GM-SDE/ODE 求解器实现精确少步采样，同时引入概率引导方案解决 CFG 过饱和问题，在 ImageNet 256×256 上仅 6 步采样即达到 Precision 0.942。

## 研究背景与动机

**领域现状**：Flow matching（如 Rectified Flow、Conditional Flow Matching）已成为扩散模型的重要替代方案，通过学习从噪声到数据的确定性流来实现生成。SD3、FLUX 等最新模型均基于 flow matching。

**现有痛点**：
   - **少步采样质量差**：标准 flow matching 在少步（4-8 步）采样时离散化误差大，导致图像质量明显下降。因为单高斯假设在 $t$ 较远的步长时近似误差急剧增大
   - **CFG 过饱和**：Classifier-free guidance 在提升文本对齐的同时导致颜色过饱和、图像不自然。这是因为 CFG 在概率层面不合理——它线性外插条件和无条件 score，但外插后的 score 不再对应有效分布

**核心矛盾**：去噪分布的单高斯假设过于简化——真实的条件分布 $p(x_0|x_t)$ 通常是多模态的（一个噪声图可能对应多种合理去噪结果），单高斯只能捕获均值，丢失多模态信息。

**本文目标**：(1) 提升少步采样精度，(2) 解决 CFG 过饱和。

**切入角度**：将去噪分布从单高斯推广到 **高斯混合分布**，用更强的参数化捕获多模态性，并基于此推导更精确的采样器和更合理的引导机制。

**核心 idea**：预测高斯混合参数替代单一均值预测，用 KL 散度替代 $L_2$ 损失训练，从解析的 GM 分布推导精确求解器。

## 方法详解

### 整体框架

- **前向过程**：标准 flow matching 或扩散的高斯插值 $x_t = \alpha_t x_0 + \sigma_t \epsilon$
- **模型输出**：不再预测单一均值 $\mu_\theta(x_t, t)$，而是预测高斯混合参数 $\{(\pi_k, \mu_k, \Sigma_k)\}_{k=1}^K$
- **采样**：使用推导的 GM-ODE 或 GM-SDE 求解器

### 关键设计

1. **高斯混合参数化**:

    - 传统方法：$p_\theta(x_0|x_t) = \mathcal{N}(\mu_\theta, \sigma^2 I)$，用 $L_2$ 损失训练
    - GMFlow：$p_\theta(x_0|x_t) = \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \Sigma_k)$
    - 网络输出 $K$ 个混合分量的权重 $\pi_k$、均值 $\mu_k$、协方差 $\Sigma_k$
    - 使用 **KL 散度损失** 训练：$\mathcal{L} = D_{KL}(q(x_0|x_t, x_0) \| p_\theta(x_0|x_t))$
    - **为什么用 GM**：单高斯只能表示单峰分布，而真实的 $p(x_0|x_t)$ 在高噪声（大 $t$）时是多模态的——例如一个模糊的噪声图可能对应"猫"或"狗"两种去噪结果。GM 能捕获这种多模态性
    - **广义化关系**：当 $K=1$ 且 $\Sigma$ 固定时，KL 损失退化为标准 $L_2$ 去噪损失，因此 GMFlow 严格推广了传统方法

2. **GM-SDE/ODE 求解器**:

    - 传统 ODE 求解器（如 Euler、DDIM）：速度场 $v_\theta(x_t, t) = \frac{\mu_\theta - x_t}{\Delta t}$（单高斯简化）
    - GM-ODE 求解器：利用 GM 分布的解析形式
    $v_{GM}(x_t, t) = \sum_k \pi_k(x_t) \frac{\mu_k(x_t) - x_t}{\Delta t}$
    - GM-SDE 求解器：额外利用 GM 方差信息注入噪声
    - **为什么更精确**：标准 ODE 在大步长时将曲线轨迹近似为直线，误差大；GM 求解器利用多个高斯分量的解析信息，能更准确地估计下一步状态，等效于在概率层面做更精确的积分

3. **概率引导（Probabilistic Guidance）**:

    - 标准 CFG：$\tilde{v} = v_{uncond} + w(v_{cond} - v_{uncond})$（速度场的线性外推）
    - 问题：外推后的速度场不对应有效概率分布，导致过饱和
    - GMFlow 的概率引导：在 GM 分布层面做引导，而非在速度场层面
    - 具体做法：调整高斯混合的权重 $\pi_k$ 而非直接外推速度场
    $\tilde{\pi}_k \propto \pi_k^{cond} \cdot (\pi_k^{cond} / \pi_k^{uncond})^{w-1}$
    - **为什么更好**：在概率分布层面操作天然保证引导后的结果仍是有效分布，不会产生超出分布支撑的采样点（即不会过饱和）

### 损失函数 / 训练策略

$$\mathcal{L}_{GMFlow} = \mathbb{E}_{t, x_0, \epsilon} \left[ D_{KL}\left(\mathcal{N}(x_0; \frac{x_t - \sigma_t \epsilon}{\alpha_t}, \eta^2 I) \Big\| \sum_k \pi_k \mathcal{N}(\mu_k, \Sigma_k)\right) \right]$$

- $\eta$ 是目标分布的方差超参数（类似 DDPM 的 $\beta$）
- 实际实现中采用各向同性协方差 $\Sigma_k = \sigma_k^2 I$ 简化计算
- 混合分量数 $K$ 通常取 4-8

## 实验关键数据

### 主实验：ImageNet 256×256 少步采样

| 方法 | 采样步数 | FID ↓ | Precision ↑ | Recall ↑ |
|------|---------|-------|-------------|----------|
| Flow Matching (Euler) | 6 | ~15 | ~0.85 | ~0.55 |
| Flow Matching (Euler) | 50 | ~2.5 | ~0.90 | ~0.60 |
| DDIM | 6 | ~18 | ~0.82 | ~0.50 |
| Heun 2nd-order | 6 | ~10 | ~0.88 | ~0.55 |
| **GMFlow (GM-ODE)** | **6** | **更低** | **0.942** | **竞争力** |
| **GMFlow (GM-SDE)** | **6** | **更低** | **最佳** | **最佳** |

### 消融实验

| 配置 | FID (6步) | Precision (6步) | 说明 |
|------|----------|----------------|------|
| GMFlow K=1 (退化为 FM) | 基线水平 | ~0.85 | 等价于标准 flow matching |
| GMFlow K=2 | 改善 | ~0.91 | 2 分量已有显著提升 |
| GMFlow K=4 | 进一步改善 | ~0.93 | 最佳性价比 |
| GMFlow K=8 | 微小改善 | ~0.942 | 收益递减 |
| 标准 CFG | 过饱和 | 高 Precision / 低 Recall | 多样性损失 |
| 概率引导 | 无过饱和 | 高 Precision / 合理 Recall | 质量和多样性兼顾 |

### 关键发现
- **少步采样优势显著**：6 步 GMFlow 的 Precision 达到 0.942，远超标准 flow matching 50 步的水平
- **GM 分量数的效果**：$K=4$ 是最佳性价比点，再增加分量收益递减
- **概率引导彻底解决过饱和**：在维持高文本对齐的同时消除不自然的颜色问题
- **GM 方差信息的价值**：GM-SDE 利用方差注入适量噪声，比 GM-ODE 在多样性上更好

## 亮点与洞察
- **理论严谨**：严格证明了 GMFlow 是标准扩散/flow matching 的推广，$K=1$ 时精确退化
- **概率引导的优雅性**：在分布层面而非 score 层面做引导，天然避免分布外采样
- **实用性强**：额外计算成本有限（预测 K 个分量 vs 1 个均值），但少步采样质量大幅提升
- **洞察**：去噪分布的多模态性在高噪声水平尤为重要，这也解释了为什么传统方法在第一步（最高噪声）采样时误差最大

## 局限与展望
- 目前在 ImageNet 256×256 上验证，尚未在大规模文生图模型（如 SDXL 级别）上测试
- K 个高斯分量增加了模型输出维度，对大规模模型的 memory/compute 开销需要评估
- 概率引导方案在大 guidance scale 下的表现需要更多验证
- 尚未探索与蒸馏方法（如 consistency distillation）的结合

## 相关工作与启发
- **Rectified Flow / CFM**：GMFlow 建立在 flow matching 框架之上，是其自然推广
- **DDPM analytic models**：之前有工作预测高斯方差（如 improved DDPM），GMFlow 更进一步使用 GM
- **Guidance 方法**：CFG、Autoguidance 等——GMFlow 的概率引导提供了新视角
- **启发**：GM 参数化的思路可推广到视频生成、3D 生成等需要少步采样的场景；概率引导的框架也适用于任何条件生成任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 高斯混合参数化是对扩散/FM 模型的自然且深刻的推广
- 实验充分度: ⭐⭐⭐⭐ ImageNet 上有系统的消融和对比，但缺少大规模 T2I 实验
- 写作质量: ⭐⭐⭐⭐ 理论推导完整，概念清晰
- 价值: ⭐⭐⭐⭐⭐ 少步采样和过饱和修复都是实际部署的关键痛点，价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] ContinualFlow: Learning and Unlearning with Neural Flow Matching](continualflow_learning_and_unlearning_with_neural_flow_matching.md)
- [\[ICML 2025\] BinauralFlow: A Causal and Streamable Approach for High-Quality Binaural Speech Synthesis with Flow Matching Models](binauralflow_a_causal_and_streamable_approach_for_high-quality_binaural_speech_s.md)
- [\[ICML 2025\] Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers](elucidating_flow_matching_ode_dynamics_with_respect_to_data_geometries_and_denoi.md)
- [\[NeurIPS 2025\] Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch](../../NeurIPS2025/image_generation/shortcutting_pre-trained_flow_matching_diffusion_models_is_almost_free_lunch.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)

</div>

<!-- RELATED:END -->
