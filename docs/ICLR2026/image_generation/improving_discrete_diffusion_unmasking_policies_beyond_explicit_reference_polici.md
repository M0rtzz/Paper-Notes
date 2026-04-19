---
title: >-
  [论文解读] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)
description: >-
  [ICLR 2026][图像生成][扩散模型] 提出 Unmasking Policy Optimization（UPO），将 Masked Diffusion Model 的去噪过程建模为 KL 正则化 MDP，通过强化学习训练轻量级的 unmasking 策略模型来替代 max-confidence 等启发式调度器，在理论和实验上均证明学习到的策略能生成更接近真实数据分布的样本。
tags:
  - ICLR 2026
  - 图像生成
  - 扩散模型
  - Unmasking Policy
  - reinforcement-learning
  - KL-正则化MDP
  - GRPO
---

# Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)

**会议**: ICLR 2026  
**arXiv**: [2510.05725](https://arxiv.org/abs/2510.05725)  
**代码**: [GitHub](https://github.com/chunsanHong/UPO)  
**领域**: 离散扩散模型 / 语言建模  
**关键词**: Masked Diffusion Models, Unmasking Policy, reinforcement-learning, KL-正则化MDP, GRPO

## 一句话总结

提出 Unmasking Policy Optimization（UPO），将 Masked Diffusion Model 的去噪过程建模为 KL 正则化 MDP，通过强化学习训练轻量级的 unmasking 策略模型来替代 max-confidence 等启发式调度器，在理论和实验上均证明学习到的策略能生成更接近真实数据分布的样本。

## 研究背景与动机

Masked Diffusion Models（MDMs）通过迭代去掩码实现离散空间的生成，已在语言建模中展现与自回归模型竞争的能力（如 LLaDA、Dream-7B）。在推理阶段，选择**哪个位置先去掩码**对生成质量至关重要：

**理论动机**：Kim et al. (2025) 证明了多项式时间算法无法在所有掩码句子上精确恢复数据分布，存在"困难子问题"。但经验表明，max-confidence 策略可以绕过这些困难实例。

**现有方法的局限**：当前大规模 MDM（LLaDA、Dream-7B）依赖规则式调度器（max-confidence、max-margin、entropy），这些只是启发式的，没有理论最优性保证。通过 Pass@N 实验发现，当采样 N 条轨迹时，随机和 Top-5 策略都能超过 max-confidence 的单路径准确率，说明存在比 max-confidence 更好的 unmasking 路径。

**核心问题**：如何学习一个比启发式更优的 unmasking 策略，同时保持训练稳定性和理论保证？

## 方法详解

### 整体框架

UPO 将 MDM 的去噪过程形式化为有限时间步的马尔可夫决策过程（MDP），冻结基础 MDM $\pi_\theta$ 不做修改，仅训练一个轻量级的 unmasking 策略模型 $g_\phi$ 来决定去掩码的顺序。策略模型基于 MDM 的中间特征运行，参数量仅约 134M（基础 MDM 为 8B）。

### 关键设计

1. **MDP 建模**：状态为当前部分去掩码的序列 $\mathbf{x}_n$，动作空间为所有掩码位置的索引集 $\mathcal{A}_{\mathbf{x}_n}$。策略 $g_\phi(a^i | \mathbf{x}_n)$ 通过 softmax 参数化来选择去掩码位置，环境动态由冻结的 MDM 给出：

$$p_{g_\phi}(\mathbf{x}_{n-1} | \mathbf{x}_n) = g_\phi(a_n | \mathbf{x}_n) \cdot \pi_\theta(\mathbf{x}_{n-1} | \mathbf{x}_n, a_n)$$

策略模型架构包含一个 Transformer 层 + 3 层 MLP，复用 MDM 的特征提取，计算高效。

2. **KL 正则化 GRPO 目标**：引入强参考策略 $g_{\mathrm{ref}}$（如 max-confidence），优化终端输出级的 GRPO 损失，带 KL 散度正则化：

$$\max_\phi \mathbb{E}\left[\frac{p_{g_\phi}(\mathbf{x}_0|\mathbf{q})}{p_{g_{\phi_{\mathrm{old}}}}(\mathbf{x}_0|\mathbf{q})} A(\mathbf{q}, \mathbf{x}_0) - \beta D_{\mathrm{KL}}(p_{g_\phi} \| p_{g_{\mathrm{ref}}})\right]$$

正则化项保持 $g_\phi$ 接近 $g_{\mathrm{ref}}$，充当信任域防止不稳定。

3. **理论保证（Theorem 1 & 2）**：

    - **收敛性**：在迭代优化下，策略的期望奖励收敛到高于参考策略的不动点 $r_{g^*} > r_{g_{\mathrm{ref}}}$
    - **分布接近性**：优化后的策略生成的终端分布与真实数据分布的 KL 散度严格小于参考策略，即 $D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{g_{\phi^*}}) < D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{g_{\mathrm{ref}}})$

### 损失函数 / 训练策略

由于 $p_{g_\phi}(\mathbf{x}_0|\mathbf{q})$ 难以计算（需边缘化所有轨迹），通过 Proposition 1 证明 token 级代理损失与输出级损失的梯度近似相等。最终可操作的 UPO 损失为带 clipping 的 token 级 GRPO：

$$\mathcal{L}_{\mathrm{UPO}} = \frac{1}{G}\sum_g \left(\frac{1}{L}\sum_n \min\left(\frac{g_\phi(a_n^{(g)}|\mathbf{x}_n^{(g)})}{g_{\phi_{\mathrm{old}}}(a_n^{(g)}|\mathbf{x}_n^{(g)})} A_g, \mathrm{clip}(\cdot, 1-\epsilon, 1+\epsilon) A_g\right) - \beta D(p_{g_\phi} \| p_{g_{\mathrm{ref}}})\right)$$

提供三种参考策略实现：max-confidence（CE 正则）、softmax-confidence（KL 正则）、Top-K（KL 正则）。Top-K 变体支持随机初始化无需预训练。

## 实验关键数据

### 主实验

| 数据集 | 指标 | UPO | Max-Confidence | Random | 提升(vs conf) |
|--------|------|-----|----------------|--------|------|
| Sudoku | Accuracy | **0.817** | 0.705 | 0.616 | +11.2% |
| Zebra | Accuracy | **0.362** | 0.337 | 0.339 | +2.5% |
| GSM8K | Accuracy | **0.703** | 0.684 | 0.612 | +1.9% |
| Math500 | Accuracy | **0.284** | 0.272 | 0.196 | +1.2% |

### 消融实验

| 配置 | GSM8K Acc | 说明 |
|------|-----------|------|
| diffu-GRPO + Random | 0.638 | 基线 MDM 后训练 |
| diffu-GRPO + Max-Confidence | 0.751 | 启发式调度 |
| diffu-GRPO + **UPO** | **0.764** | 在后训练 MDM 上再加 UPO，+1.3% |
| KL 正则（Top-K, GSM8K） | 0.703 | 有 KL 散度项 |
| 无 KL 正则（GSM8K） | ~0.68 | 性能下降，早期路径坍塌 |
| 随机初始化无正则（≈DColT） | ~0.67 | 比有参考策略差 2-3% |

### 关键发现

- Sudoku 等结构化任务中 unmasking 顺序极为关键，UPO 增益最大（+20.1% vs random）
- 正则化项防止早期路径坍塌，保持更大的 group reward 方差
- UPO 与 diffu-GRPO 是互补的——前者优化调度策略，后者优化 MDM 本身
- 最优参考策略因任务而异：Sudoku 用 max-confidence，GSM8K 用 Top-K

## 亮点与洞察

- 将 unmasking 策略学习与 MDM 训练解耦，策略模型仅 134M 参数（MDM 8B 的 1.7%），训练成本极低
- 提供了从理论到实践的完整链路：理论保证（Theorem 1&2）→ 代理损失（Proposition 1）→ 可操作训练目标
- Pass@N 实验直观展示了启发式策略的次优性，为学习策略提供了强动机
- KL 正则化的信任域设计既保证了训练稳定性又允许超越参考策略

## 局限与展望

- 在 GSM8K/Math500 等数学推理任务上增益相对有限，可能因为长文本中的顺序信号不如 Sudoku 明显
- 策略模型需要复用 MDM 的中间特征，对 MDM 架构有一定耦合
- 目前每步仅去掩码一个位置，扩展到多位置并行去掩码是重要方向
- 泛化性实验不足——训练集与测试集来自同一任务分布

## 相关工作与启发

- 与 DColT（Huang et al., 2025）对比：UPO 引入显式参考策略和 KL 正则化，更稳定且效果更好
- diffu-GRPO 通过 RL 后训练 MDM 本身，UPO 训练调度策略，两者互补
- 这一思路可推广到连续扩散模型的采样调度策略学习（如 ODE 求解器步长选择）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将 MDM 的 unmasking 建模为 KL 正则化 MDP 并提供完整理论保证
- 实验充分度: ⭐⭐⭐⭐ 4 个 benchmark 覆盖逻辑与数学推理，消融充分，但缺少开放文本生成评测
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，实验设计清晰，动机阐述有力
- 价值: ⭐⭐⭐⭐ 为离散扩散模型推理提供了新范式，实际改进在结构化任务上显著

<!-- RELATED:START -->

## 相关论文

- [Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [PADiff: Predictive and Adaptive Diffusion Policies for Ad Hoc Teamwork](../../AAAI2026/image_generation/padiff_predictive_and_adaptive_diffusion_policies_for_ad_hoc_teamwork.md)
- [Contractive Diffusion Policies: Robust Action Diffusion via Contractive Score-Based Sampling with Differential Equations](contractive_diffusion_policies_robust_action_diffusion_via_contractive_score-bas.md)
- [Failure Prediction at Runtime for Generative Robot Policies](../../NeurIPS2025/image_generation/failure_prediction_at_runtime_for_generative_robot_policies.md)

<!-- RELATED:END -->
