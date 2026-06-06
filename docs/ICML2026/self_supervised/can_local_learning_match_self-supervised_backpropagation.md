---
title: >-
  [论文解读] Can Local Learning Match Self-Supervised Backpropagation?
description: >-
  [ICML 2026][自监督学习][局部学习规则] 本文从理论上证明了在深度线性网络中局部自监督学习（local-SSL）可以精确实现全局反向传播自监督学习（BP-SSL）的梯度更新，并据此提出 CLAPP++ 算法（引入 2D 空间依赖和直接反馈）…
tags:
  - "ICML 2026"
  - "自监督学习"
  - "局部学习规则"
  - "反向传播近似"
  - "CLAPP++"
  - "生物合理性"
---

# Can Local Learning Match Self-Supervised Backpropagation?

**会议**: ICML 2026  
**arXiv**: [2601.21683](https://arxiv.org/abs/2601.21683)  
**代码**: 待确认  
**领域**: 自监督/表示学习  
**关键词**: 局部学习规则, 自监督学习, 反向传播近似, CLAPP++, 生物合理性  

## 一句话总结
本文从理论上证明了在深度线性网络中局部自监督学习（local-SSL）可以精确实现全局反向传播自监督学习（BP-SSL）的梯度更新，并据此提出 CLAPP++ 算法（引入 2D 空间依赖和直接反馈），在 CIFAR-10/STL-10/Tiny ImageNet 上达到了与全局 BP-SSL 相当的性能，刷新了 local-SSL 的 SOTA。

## 研究背景与动机

**领域现状**：端到端自监督学习（global BP-SSL）已成为训练现代 AI 系统的核心方法，但其反向传播（BP）的反馈网络结构在生物学上缺乏对应物。局部自监督学习（local-SSL）如 CLAPP、Forward-forward、LPL 等方法试图用逐层的 Hebbian-like 更新规则替代全局 BP，更贴合生物神经可塑性机制。

**现有痛点**：现有 local-SSL 方法在深度网络中构建有效表征的能力远不如 global BP-SSL。在有监督设置下，局部学习规则已经能够较好地近似 BP，但在自监督设置下，性能差距更大，且缺乏将 local-SSL 与全局表征学习原则联系起来的理论基础。

**核心矛盾**：local-SSL 在每一层独立优化局部损失，梯度不跨层传播，这意味着浅层的权重更新并没有直接优化深层的 SSL 目标——但理论上两者之间的关系尚不清楚。

**本文目标**：(1) 建立 local-SSL 与 global BP-SSL 之间的理论联系；(2) 基于理论发现，设计改进的 local-SSL 算法以缩小性能差距。

**切入角度**：作者首先在深度线性网络中分析，发现当权重矩阵正交归一化时，local-SSL 的逐层更新恰好等价于 global BP-SSL 的梯度更新。这一理论洞察指导了非线性卷积网络中的算法改进。

**核心 idea**：通过引入 2D 空间依赖投影和顶层直接反馈，使 local-SSL 的梯度更好地近似 global BP-SSL，从而在无需全局反向传播的情况下匹配其性能。

## 方法详解

### 整体框架
本文首先建立了一个统一的形式化框架，将 CLAPP、Forward-forward、PhyLL、SCFF 等多种 local-SSL 算法纳入同一公式体系。在此框架下，每层的局部对比损失定义为 $\mathcal{L}^l = f(z_{\text{pos}}^{l\top} B^l c_{\text{pos}}^l) + f(-z_{\text{neg}}^{l\top} B^l c_{\text{neg}}^l)$，其中 $f$ 是递减函数，$B^l$ 是可训练或固定的投影矩阵，$c^l$ 是参考向量。不同算法的区别仅在于 $f$、$B^l$、$c^l$ 的具体选择。在此基础上，作者在深度线性网络中证明了精确等价定理（Theorem 3.1），然后将理论洞察迁移到卷积网络中，提出了 CLAPP++ 及其变体。

### 关键设计

1. **深度线性网络中的精确等价理论**:

    - 功能：证明 local-SSL 逐层梯度更新与 global BP-SSL 全局梯度更新完全一致
    - 核心思路：在 $L$ 层线性网络中，假设所有权重矩阵 $W^l$ 正交归一，$B^l$ 可训练且达到最优 $B_*^l$，则有 $\frac{\partial \mathcal{L}_*^l}{\partial W_{ij}^l} = \frac{\partial \mathbf{L}_*}{\partial W_{ij}^l}$。直觉上，正交归一的权重矩阵使得从顶层反向传播到第 $l$ 层的梯度 $(W^L \cdots W^{l+1})^\top B_*^L c^L$ 恰好等于局部最优投影 $B_*^l c^l$，因为正交矩阵的乘积自行抵消
    - 设计动机：为 local-SSL 提供理论基础——在满足条件时局部优化等价于全局优化，打破了"局部学习必然损失信息"的直觉

2. **直接反馈机制（DFB）**:

    - 功能：当网络层宽度递减时，改善 local-SSL 对 BP-SSL 梯度的近似质量
    - 核心思路：将参考向量从同层活动 $c^l = z'^l$ 替换为顶层活动 $c^l = z'^L$（Theorem 3.3）。在半正交权重矩阵（行正交但维度递减）的线性网络中，可证明 $\|\frac{\partial \mathcal{L}_*^l}{\partial W^l} - \frac{\partial \mathbf{L}_*}{\partial W^l}\|_F^2 \geq \|\frac{\partial \mathcal{L}_{*,\text{fb}}^l}{\partial W^l} - \frac{\partial \mathbf{L}_*}{\partial W^l}\|_F^2$，即 DFB 版本与 BP-SSL 的差距更小
    - 设计动机：实际网络层宽度通常递减，等宽正交条件被打破。DFB 通过直接使用顶层信号补偿了维度缩减导致的信息丢失，同时在生物学上对应大脑皮层中顶树突整合远距离高级脑区输入来调节突触可塑性的机制

3. **2D 空间依赖投影**:

    - 功能：在卷积网络中让投影矩阵 $B^l$ 具有空间位置相关性，大幅提升 local-SSL 的 BP 近似质量
    - 核心思路：原始 CLAPP 对特征图全局平均池化后计算损失 $\mathcal{L}^l = f(\text{pool}(z^l)^\top B^l \text{pool}(c^l))$，导致梯度在空间维度上共享。改进后使用分块池化 $\mathcal{L}^l = f(\text{flatten}(\text{pool}_{k_1}(z^l))^\top B^l \text{flatten}(\text{pool}_{k_2}(c^l)))$，使 $B^l$ 能学习不同空间位置之间的交叉依赖，梯度仅在 $k_1$ 大小的局部 patch 内共享
    - 设计动机：BP-SSL 的梯度 $\partial \mathbf{L}/\partial z^l$ 在空间维度上是不共享的，全局平均池化丢失了这一空间结构信息。理论上（Proposition 3.5），当 $k_1 = k_2 = 1$ 时，2D 空间依赖的 local-SSL 可以精确计算 BP 梯度

## 实验关键数据

### 主实验

| 方法 | 局部更新 | 2D空间依赖 | CIFAR-10 | STL-10 | Tiny-ImageNet | ImageNet |
|------|---------|-----------|----------|--------|---------------|----------|
| BP-CLAPP++ | 否 | - | 80.49 | 80.36 | 37.55 | 48.52 |
| BP-InfoNCE | 否 | - | 80.69 | 81.97 | 36.78 | 55.19 |
| CLAPP | 是 | 否 | - | 73.6 | - | - |
| LPL | 是 | 否 | 59.4 | 63.2 | - | - |
| SoftHebb | 是 | 否 | 80.31 | 76.23 | - | 27.3 |
| SCFF | 是 | 是 | 80.60 | 77.14 | 35.67 | - |
| CLAPP++ | 是 | 是 | **80.51** | 78.66 | 36.63 | 42.55 |
| CLAPP++DFB | 是 | 是 | **80.65** | 79.38 | 36.70 | 44.16 |
| CLAPP++both | 是 | 是 | **81.18** | **79.62** | **37.78** | 42.49 |

### 消融实验

| 配置 | STL-10 准确率 | 说明 |
|------|-------------|------|
| CLAPP++ (无 2D 空间依赖) | 75.10 | 去掉空间依赖后大幅下降 |
| CLAPP++ | 78.66 | 加入 2D 空间依赖，+3.56% |
| CLAPP++DFB | 79.38 | 再加直接反馈，+0.72% |
| CLAPP++both | 79.62 | 双损失组合，+0.24% |
| BP-CLAPP++ (全局 BP) | 80.36 | 全局反向传播上界 |

### 关键发现
- **2D 空间依赖是最关键的改进**：从无空间依赖到有空间依赖，STL-10 上提升 3.56%（75.10→78.66），是所有改进中贡献最大的
- **local-SSL 首次匹配 BP-SSL**：在 CIFAR-10/STL-10/Tiny-ImageNet 上，CLAPP++ 变体与 BP-CLAPP++ 和 BP-InfoNCE 的差距消失，这在 local-SSL 历史上是首次
- **VRAM 节省显著**：local-SSL 不需要存储全部层的激活值，CLAPP++ 在 STL-10 上节省 38%、ImageNet 上节省 59% 的峰值 VRAM
- **ImageNet 上仍有差距**：在更高分辨率的 ImageNet 上，local-SSL（42-44%）与 BP-SSL（48-55%）仍有明显差距，说明高分辨率场景下局部近似的质量还有待提升

## 亮点与洞察
- **统一形式化框架**：将 CLAPP、Forward-forward、PhyLL、SCFF 等看似不同的 local-SSL 算法统一到一个参数化公式中，仅通过 $f$、$B^l$、$c^l$ 的选择来区分。这种抽象使得跨算法的理论分析成为可能，是一个优雅的贡献
- **"理论→实践"闭环**：从线性网络等价定理出发，识别出空间依赖和直接反馈两个关键改进方向，最终在实际卷积网络上验证了理论预测与性能提升的一致性。这种理论驱动的算法设计范式值得借鉴
- **生物学意义**：CLAPP++ 的权重更新 $\Delta W_{ji}^l = \gamma \cdot (B^l c^l)_j \cdot \rho'(a_j^l) z_i^{l-1}$ 可以分解为"神经调制因子 × 树突预测 × Hebbian 项"，DFB 中的顶层反馈对应大脑皮层中顶树突整合远距离输入调节突触可塑性的生物学证据，为连接 AI 与神经科学提供了桥梁

## 局限与展望
- **架构局限**：实验仅使用 VGG 卷积网络，未扩展到 ResNet 等带残差连接的现代架构，如何在残差连接中定义局部损失是开放问题
- **ImageNet 差距**：在大规模高分辨率数据集上 local-SSL 仍落后于 BP-SSL 约 6-13%，说明高分辨率场景下局部梯度近似的误差会累积
- **正交假设**：核心定理依赖权重矩阵正交归一，实际训练中权重远非正交，虽然实验表明结论可定性推广到非正交情况，但理论保障有限
- **改进方向**：(1) 设计适用于残差连接的 local-SSL 规则；(2) 探索非对比式 local-SSL（如 VICReg 风格）的类似理论；(3) 结合投影头等现代 SSL 技术进一步缩小 ImageNet 差距

## 相关工作与启发
- **CLAPP** (Illing et al., 2021)：本文的直接基础，使用逐层对比预测可塑性规则
- **Forward-forward** (Hinton, 2022)：另一种 local-SSL，通过正负样本的激活范数对比学习
- **SCFF** (Chen et al., 2025)：Forward-forward 的自对比变体，本文之前的 local-SSL SOTA
- **LPL** (Halvagal & Zenke, 2023)：非对比式局部学习，使用 VICReg-like 逐层损失
- 本文的"局部近似全局"思路可迁移到其他需要减少通信/内存开销的分布式训练场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](understanding_self-supervised_learning_via_latent_distribution_matching.md)
- [\[NeurIPS 2025\] The Complexity of Finding Local Optima in Contrastive Learning](../../NeurIPS2025/self_supervised/the_complexity_of_finding_local_optima_in_contrastive_learning.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](../../CVPR2026/self_supervised/mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[AAAI 2026\] Self-Supervised Inductive Logic Programming](../../AAAI2026/self_supervised/self-supervised_inductive_logic_programming.md)

</div>

<!-- RELATED:END -->
