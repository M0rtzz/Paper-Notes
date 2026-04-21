---
title: >-
  [论文解读] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics
description: >-
  [CVPR 2026][机器人][线性化注意力] 通过 NTK 框架证明线性化注意力不会收敛到无限宽度核极限（需要宽度 $m = \Omega(\kappa^6)$），并提出"影响可塑性"指标量化其双面效应：注意力比 ReLU 网络高 6–9× 的数据依赖灵活性，既能降低近似误差也增加对抗脆弱性。
tags:
  - CVPR 2026
  - 机器人
  - 线性化注意力
  - 神经正切核
  - 影响可塑性
  - 特征学习
  - 对抗鲁棒性
---

# Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics

**会议**: CVPR 2026  
**arXiv**: [2603.13085](https://arxiv.org/abs/2603.13085)  
**代码**: 无  
**领域**: 深度学习理论 / 注意力机制  
**关键词**: 线性化注意力, 神经正切核, 影响可塑性, 特征学习, 对抗鲁棒性  

## 一句话总结

通过 NTK 框架证明线性化注意力不会收敛到无限宽度核极限（需要宽度 $m = \Omega(\kappa^6)$），并提出"影响可塑性"指标量化其双面效应：注意力比 ReLU 网络高 6–9× 的数据依赖灵活性，既能降低近似误差也增加对抗脆弱性。

## 研究背景与动机

**领域现状**：NTK（神经正切核）理论建立了无限宽度网络与核方法的等价关系，预测足够宽的网络在训练中核保持近似不变（"懒训练"体制）。这一框架已扩展到深度网络、任意架构等设置，但注意力机制一直未被严格纳入 NTK 分析。

**现有痛点**：(1) 传统研究要么关注注意力的架构属性，要么关注最终性能，忽视了注意力学习过程的动力学特征；(2) NTK 理论对注意力是否适用完全未知——Wenger et al. 指出 NTK 理论仅对"比深度宽数个量级"的网络有效；(3) 缺乏量化注意力对训练数据敏感性的理论工具。

**核心矛盾**：注意力机制的表达力（灵活适应数据结构）与其对训练数据的脆弱敏感性可能共享同一来源——偏离核体制。

**本文目标** 线性化注意力是否收敛到无限宽度 NTK 极限？如果不收敛，这种非收敛行为对模型的训练数据依赖性意味着什么？

**切入角度**：设计无参数线性化注意力 $f^{att}(X) = XX^TX$，建立其与数据依赖 Gram 诱导核的精确对应，利用谱分析解释非收敛现象，并通过"影响可塑性"量化双面效应。

**核心 idea**：注意力的 power 与 vulnerability 同源于它对核体制的超越——数据依赖核带来灵活性的同时也带来脆弱性。

## 方法详解

### 整体框架

输入原始数据 $X \in \mathbb{R}^{n \times d}$ → 线性化注意力变换 $f^{att}(X) = XX^TX$ → $\ell_2$ 归一化 → 送入两层 ReLU MLP → 输出预测。对比基线为直接在原始输入上运行的 2L-ReLU 网络。在不同宽度 $m$ 上计算有限宽 NTK 与无限宽 NTK 的距离 $\|f_m - f_{NTK}\|$，并通过影响函数量化数据依赖性。

### 关键设计

1. **线性化注意力与 Gram 诱导核（Theorem 4.1）**
    - 无参线性化注意力 $f^{att}(X) = XX^TX$ 对应 scaled dot-product attention（$W_Q = W_K = W_V = I$，linearized softmax）
    - 精确对应一个数据依赖的 Gram 诱导核 $K_{LinAttn} = G^3$（$G = XX^T$）
    - 每个核元素为四阶交互项 $\sum_{k,\ell}(x_i^Tx_k)(x_k^Tx_\ell)(x_\ell^Tx_j)$，实现传递性相似度传播 $i \to k \to \ell \to j$
    - 与标准多项式核 $(x^Ty)^p$ 的本质区别：$K_{LinAttn}$ 的敏感度依赖于整个数据集的相关结构（通过 $G$），而非仅输入对之间的关系
2. **谱放大与 NTK 非收敛（Theorem 4.7）**
    - 注意力变换将 Gram 矩阵条件数立方化：$\kappa(\tilde{G}) = \kappa(G)^3$
    - NTK 收敛要求宽度 $m = \Omega(\kappa(G)^6 / \epsilon^2)$：对 MNIST（$\kappa \approx 1.2 \times 10^3$）需 $m \gg 10^{18}$，对 CIFAR-10（$\kappa \approx 8.7 \times 10^3$）需 $m \gg 10^{24}$，远超实验范围 $m \leq 4096$
    - 而 2L-ReLU 仅需 $m = \Omega(1/\epsilon^2)$，无谱放大因子
    - 物理解释：$k$ 层线性化自注意力堆叠产生 $G^{2k+1}$，条件数以 $\kappa^{2k+1}$ 增长，层数越多离核体制越远
3. **影响可塑性度量（Definition 3.4）**
    - Influence Flip Rate：在 top 10% 高影响训练样本上，经 PGD 扰动（$\epsilon = 0.3$）后影响函数符号翻转的比例
    - 互补指标：影响排名的 Spearman 相关系数 $\rho$（越低说明可塑性越高）
    - 三种数据干预策略：Curated（去除 top-$\tau$ 影响样本）、Transformed（替换为对抗版本）、Adversarial（对所有训练数据做 PGD 扰动）
    - 影响函数通过经验有限宽 NTK 矩阵 $(K_m + \lambda I)^{-1}$ 高效计算，无需重训练

### 损失函数 / 训练策略

交叉熵（多分类）或 MSE（二分类）+ L2 正则化（$\lambda = 10^{-3}$）。Adam 优化器，$lr = 10^{-3}$，batch=128，训练 500 epochs。FGSM/PGD/MIM 三种对抗扰动方法用于影响可塑性测量。

## 实验关键数据

### 主实验

| 数据集 | 指标 | MLP-Attn | 2L-ReLU | 倍率 |
|--------|------|----------|---------|------|
| MNIST (10类) | Flip Rate (PGD) | 28.9% | 3.3% | 8.8× |
| MNIST (10类) | Flip Rate (FGSM) | 34.6% | 4.1% | 8.4× |
| CIFAR-10 (10类) | Flip Rate (PGD) | 19.1% | 3.1% | 6.2× |
| CIFAR-10 (10类) | Flip Rate (FGSM) | 26.4% | 3.3% | 8.0× |
| MNIST (二分类) | Flip Rate (PGD) | 41.0% | 8.4% | 4.9× |

| 数据集 | NTK距离 (m=16) | NTK距离 (m=4096) | 趋势 |
|--------|---------------|-----------------|------|
| MNIST 2L-ReLU | 45.1 | 39.2 | 单调↓（收敛） |
| MNIST MLP-Attn | 10.3 | 43.4 | 非单调↑（不收敛） |
| CIFAR-10 2L-ReLU | 246.2 | 56.9 | 单调↓（收敛） |
| CIFAR-10 MLP-Attn | 3.7 | 12.6 | 单调↑（不收敛） |

### 消融实验

| 实验 | 结果 |
|------|------|
| 对抗训练对 2L-ReLU | Flip Rate 3.3% → 43.4%（MNIST），说明 AT 可诱导可塑性 |
| 对抗训练对 MLP-Attn | Flip Rate 28.9% → 42.2%（MNIST），提升幅度小——架构已内禀高可塑性 |
| 二分类 CIFAR-10 | MLP-Attn 优势消失（≈1×），因二分类 Gram 条件数较低，立方放大效应减弱 |
| 扰动强度 $\epsilon$: 0.1→0.5 | MLP-Attn 始终高于 2L-ReLU，排序不变 |

### 关键发现

- 注意力在所有测试条件下均展现远高于 ReLU 的影响可塑性（6–9×），且无需对抗训练即可获得
- 经验 Gram 条件数与理论宽度需求高度一致：MNIST $\kappa \approx 10^3$，CIFAR-10 $\kappa \approx 10^{3.9}$
- 对抗训练和注意力架构产生可塑性的机制不同：前者是训练诱导，后者是架构内禀

## 亮点与洞察

- 首次从 NTK 角度严格证明注意力不进入核体制：谱放大 $\kappa^3$ 导致宽度需求呈六次方增长，这是一个干净优雅的理论结果
- "影响可塑性"概念精准刻画了注意力的 power 与 vulnerability 同源问题——这为理解 Transformer 为何强大又脆弱提供了新视角
- 对抗训练实验揭示两种产生可塑性的机制（架构内禀 vs 训练诱导），理清了混淆因素
- 理论与实验高度自洽：经验 Gram 条件数准确预测了非收敛行为

## 局限与展望

- 仅分析线性化注意力（identity QKV），未扩展至完整 softmax 注意力——softmax 的行归一化可能进一步放大非收敛
- 实验规模受限于精确 NTK 计算（MNIST/CIFAR-10，两层网络，$m \leq 4096$），与实际 Transformer 有 gap
- Theorem 4.7 仅约束初始化时的 NTK 偏差，不直接预测训练后轨迹
- 未探索低秩正则化（截断注意力保留 top-$r$ 奇异值）是否能恢复收敛并降低对抗脆弱性

## 相关工作与启发

- **vs Jacot et al. (NTK, 2018)**：经典 NTK 理论预测宽网络收敛；本文证明注意力架构违反此预测
- **vs Chizat et al. (Lazy vs Feature Learning, 2019)**：本文提供了注意力作为"特征学习体制"的具体架构实例和量化证据
- **vs Zhang et al. (NTK-based Influence, 2022)**：延续其 NTK 影响函数方法，但首次用于比较架构间的影响可塑性差异
- **实践启发**：线性化注意力的谱放大效应暗示，对注意力模块进行低秩近似可能在保持表达力的同时恢复 NTK 收敛，为注意力头剪枝/低秩分解提供新理论指导
- 影响可塑性可作为衡量模型鲁棒性的新指标，比传统对抗精度更细粒度

## 评分

- 新颖性: ⭐⭐⭐⭐ 从 NTK 角度建立注意力理论是新视角，影响可塑性概念有洞察力
- 实验充分度: ⭐⭐⭐⭐ 覆盖多数据集、多扰动类型、多分类设置，理论与实验对应好
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨完整，行文结构清晰，证明放附录不影响可读性
- 价值: ⭐⭐⭐ 理论工作，对实际 Transformer 的直接指导有待后续验证

<!-- RELATED:START -->

## 相关论文

- [Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)
- [Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior](boosting_vision-language-action_finetuning_with_feasible_action_neighborhood_pri.md)
- [MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)

<!-- RELATED:END -->
