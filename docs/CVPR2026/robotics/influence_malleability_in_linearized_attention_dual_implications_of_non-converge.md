---
title: >-
  [论文解读] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics
description: >-
  [CVPR2026][机器人][Neural Tangent Kernel] 本文揭示线性化注意力机制在 NTK 框架下不收敛至无穷宽极限，并提出"影响力可塑性"(influence malleability) 度量，证明注意力的强大能力与对抗脆弱性共享同一来源——偏离核regime的数据依赖核结构。
tags:
  - "CVPR2026"
  - "机器人"
  - "Neural Tangent Kernel"
  - "线性化注意力"
  - "影响力可塑性"
  - "核方法"
  - "特征学习"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics

**会议**: CVPR2026  
**arXiv**: [2603.13085](https://arxiv.org/abs/2603.13085)  
**代码**: 待确认  
**领域**: 深度学习理论 / 注意力机制  
**关键词**: Neural Tangent Kernel, 线性化注意力, 影响力可塑性, 核方法, 特征学习

## 一句话总结

本文揭示线性化注意力机制在 NTK 框架下不收敛至无穷宽极限，并提出"影响力可塑性"(influence malleability) 度量，证明注意力的强大能力与对抗脆弱性共享同一来源——偏离核regime的数据依赖核结构。

## 背景与动机

1. **注意力机制理论空白**: 注意力机制在深度学习中取得巨大成功，但其学习动态缺乏严格理论刻画，大多数工作关注初始化或最终性能，忽略中间训练过程
2. **NTK 理论局限**: NTK 框架预测足够宽的网络处于"lazy training"regime（核不变），但实际注意力架构是否满足这一条件未被系统研究
3. **特征学习 vs 懒训练**: Chizat et al. (2019) 区分了lazy training和feature learning两种regime，但注意力机制落在哪个regime缺乏实证和理论支撑
4. **数据依赖性缺乏量化**: 注意力模型对训练数据的敏感性缺少可度量的指标，无法系统评估其对数据质量的依赖程度
5. **对抗鲁棒性理解不足**: 注意力架构的对抗脆弱性与其表达能力之间的联系尚不清楚，需要统一的理论解释
6. **核方法与注意力的连接**: softmax 注意力与 Nadaraya-Watson 核回归的联系已知，但线性化注意力的精确核刻画及其对学习动态的影响尚未建立

## 方法详解

### 整体框架

这篇论文想搞清楚一件事：注意力机制到底处在 NTK 理论说的「懒训练」regime，还是「特征学习」regime？为此它构造了一个能被**精确核刻画**的线性化注意力架构 MLP-Attn，沿 NTK 框架推导其训练动态，证明它根本不收敛到无穷宽极限，最后用「影响力可塑性」这个新指标把「注意力很强」和「注意力很脆」统一归因到同一个源头。整条论证是一个分叉-汇合结构：从可精确刻画的核 $\mathbf{K}=\mathbf{G}^3$ 出发，一支走「谱放大→不收敛→特征学习」解释了表达能力的来源，另一支走「数据依赖敏感性→影响力可塑性」量化了对抗脆弱性，两支最终汇合到「能力与脆弱性同源」。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["可精确核刻画的线性化注意力 MLP-Attn<br/>QKV 取恒等、softmax 线性化 → 诱导核 K = G³"] --> B["谱放大定理<br/>κ(G̃) = κ(G)³ → NTK 不收敛 → 特征学习 regime"]
    A --> C["数据依赖敏感性<br/>扰动经 Gram 矩阵全局传播 |ΔK| ≤ ‖Gxⱼ‖₁·ε"]
    C --> D["影响力可塑性度量<br/>翻转率 + 排名相关性 + 三种干预"]
    B --> E["双重含义：能力与脆弱性同源<br/>偏离核 regime"]
    D --> E
```

### 关键设计

**1. 可精确核刻画的线性化注意力 MLP-Attn：给注意力找一个能算清楚核的化身**

要判断注意力落在哪个 regime，先得有一个能写出闭式核的注意力。本文把 softmax 线性化、QKV 投影取恒等，得到注意力算子 $f^{\text{att}}(\mathbf{X}) = \mathbf{X}\mathbf{X}^T\mathbf{X}$，完整架构为

$$f_{\text{MLP-Attn}}(\mathbf{X}) = \frac{1}{\sqrt{m}} \sum_{r=1}^{m} a_r \sigma(\mathbf{w}_r^T \cdot f^{\text{att}}(\mathbf{X}))$$

该变换在整个训练集上计算（transductive），编码全局成对关系，输出做 $\ell_2$ 归一化后送入 MLP。在此架构下可以证明（Thm 4.1）它诱导的核为 $\mathbf{K}_{\text{LinAttn}} = \mathbf{G}^3$（$\mathbf{G}=\mathbf{X}\mathbf{X}^T$），带有 $i \to k \to \ell \to j$ 的传递相似性结构——这正是后面所有结论的出发点。

**2. 谱放大定理：注意力把条件数三次方放大，NTK 因此不收敛**

经典 NTK 理论假设足够宽的网络处于 lazy regime、核不变，但注意力是否满足从没被验证。本文证明（Thm 4.7）注意力把 Gram 矩阵条件数立方放大 $\kappa(\tilde{\mathbf{G}}) = \kappa(\mathbf{G})^3$，于是 NTK 收敛所需宽度高达 $m = \Omega(\kappa(\mathbf{G})^6 / \epsilon^2)$。对自然图像这个数字大得离谱——MNIST 需 $m \gg 10^{18}$、CIFAR-10 需 $m \gg 10^{24}$，远超任何实际宽度。换句话说，真实注意力永远到不了 lazy regime，天然处在特征学习 regime。

**3. 数据依赖敏感性：扰动通过 Gram 矩阵全局传播**

特征学习 regime 的代价是核随数据而变、对输入扰动敏感。本文给出界（Prop 4.5）

$$|K_{\text{LinAttn}}(\mathbf{x}_i+\delta, \mathbf{x}_j) - K_{\text{LinAttn}}(\mathbf{x}_i, \mathbf{x}_j)| \leq \|\mathbf{G}\mathbf{x}_j\|_1 \cdot \epsilon$$

关键在于扰动不是局部影响，而是经 Gram 矩阵 $\mathbf{G}$ 传播到所有样本对——这就是注意力既能「看全局关系」又「一碰就变」的数学根源，表达力和脆弱性同源于此。

**4. 影响力可塑性度量：把「一碰就变」做成可量化指标**

为了实测这种敏感性，本文定义影响力可塑性。**影响力翻转率（Flip Rate）**：对 top-10% 高影响力训练样本施加对抗扰动（PGD, $\epsilon=0.3$），统计影响力符号翻转的比例；**影响力排名相关性**：用 Spearman 秩相关 $\rho$ 衡量扰动前后影响力排名的稳定性，$\rho$ 越低可塑性越高。配套三种干预——Curated（移除高影响样本）、Transformed（替换为对抗版本）、Adversarial（全局 PGD 扰动），用来从不同角度暴露注意力对训练数据的依赖程度。

## 实验关键数据

### NTK 距离不收敛

| 模型 | 数据集 | m=16 | m=1024 | m=4096 | 趋势 |
|------|--------|------|--------|--------|------|
| 2L-ReLU | MNIST | 45.1 | 39.9 | 39.2 | ↓ 收敛 |
| MLP-Attn | MNIST | 10.3 | 33.3 | 43.4 | ↑ 非单调 |
| 2L-ReLU | CIFAR-10 | 246.2 | 101.7 | 56.9 | ↓ 收敛 |
| MLP-Attn | CIFAR-10 | 3.7 | 10.4 | 12.6 | ↑ 单调递增 |

- 2L-ReLU 随宽度增加 NTK 距离单调下降（符合经典 NTK 理论）；MLP-Attn 距离反而增大，确认其处于特征学习 regime

### 影响力翻转率（10类，$\epsilon=0.3$）

| 数据集 | 模型 | FGSM | PGD | MIM |
|--------|------|------|-----|-----|
| MNIST | 2L-ReLU | 4.1% | 3.3% | 3.4% |
| MNIST | MLP-Attn | **34.6%** | **28.9%** | **21.9%** |
| CIFAR-10 | 2L-ReLU | 3.3% | 3.1% | 3.2% |
| CIFAR-10 | MLP-Attn | **26.4%** | **19.1%** | **20.5%** |

- MLP-Attn 的翻转率是 2L-ReLU 的 **6–9 倍**，验证注意力架构对训练数据高度敏感

### 消融：对抗训练的影响

| 数据集 | 模型 | 标准训练 | 对抗训练 |
|--------|------|----------|----------|
| MNIST | 2L-ReLU | 3.3% | 43.4% |
| MNIST | MLP-Attn | 28.9% | 42.2% |
| CIFAR-10 | 2L-ReLU | 3.1% | 36.5% |
| CIFAR-10 | MLP-Attn | 19.1% | 38.6% |

- 对抗训练大幅提升 ReLU 的可塑性（3.3%→43.4%），但 MLP-Attn 在标准训练下就天然具备高可塑性，说明这是架构层面的固有特性而非训练诱导
- 对抗训练后两种架构的可塑性趋于接近（42–43%），暗示对抗训练可能将 ReLU 推入类似的 feature learning regime
- 二分类设置下（MNIST 3 vs 8），MLP-Attn PGD flip rate 达 41.0%（ReLU 仅 8.4%），但 CIFAR-10 二分类差异消失，与 $\kappa(\mathbf{G})$ 较低一致

## 亮点

- **理论贡献扎实**: 精确建立了线性化注意力→Gram-induced kernel→谱放大→NTK不收敛的完整理论链条，每一步都有严格证明
- **双重含义的统一解释**: 首次将注意力的"表达能力强"和"对抗脆弱"归因于同一源头（偏离核regime），这一洞察具有启发意义
- **新度量指标**: 影响力可塑性（flip rate + rank correlation）提供了可量化的训练数据敏感性指标，可推广到其他架构分析
- **实验与理论一致**: Gram 矩阵条件数的实测值（$\kappa \approx 10^3$）与理论预测的收敛宽度要求（$m \gg 10^{18}$）完美解释了实验中 $m \leq 4096$ 时的不收敛
- **对抗训练消融实验设计巧妙**: 通过对比标准训练与对抗训练下两种架构的可塑性变化，清晰区分了"架构固有"vs"训练诱导"两种敏感性机制
- **多层推广**: 理论自然推广到多层线性化注意力（$k$ 层 → $\kappa(\mathbf{G})^{2k+1}$ 放大），且提出截断注意力作为可能的正则化方案

## 局限与展望

- **线性化简化**: 仅分析 $f^{\text{att}}=\mathbf{X}\mathbf{X}^T\mathbf{X}$，未扩展到完整 softmax 注意力，softmax 的行归一化可能进一步放大效应
- **数据集和模型规模小**: 实验仅在 MNIST/CIFAR-10 + 两层网络（$m \leq 4096$）上验证，未在大规模 Transformer（如 ViT）上实证
- **参数无关注意力**: QKV 为恒等矩阵的假设虽理论上可推广（Proposition B.4），但实际 attention 有可学习投影，gap 有多大未验证
- **二分类场景效果消失**: 在 CIFAR-10 二分类中，注意力优势几乎消失（flip rate $\approx 1\times$），说明结论对数据维度/复杂度敏感
- **缺乏防御方案**: 发现对抗脆弱性但未提出具体缓解方法，低秩正则化（截断注意力）仅在理论备注中提及
- **Transductive 设计限制**: 注意力变换在整个训练集上计算一次，这与实际中按 mini-batch 处理的 Transformer 有本质区别，实用性受限

## 与相关工作的对比

- **vs NTK 理论 (Jacot et al., 2018)**: 经典 NTK 假设宽网络处于 lazy regime，本文证明注意力架构打破该假设
- **vs Wenger et al. (2023)**: 后者指出 NTK 理论仅适用于"远宽于深"的网络，本文给出了具体的注意力架构反例并量化了所需宽度
- **vs Nichani et al. (2025)**: 后者提供了特征学习的可证明保证，本文提供了注意力作为天然满足特征学习条件的具体架构证据
- **vs Hron et al. (2020)**: 后者将 NTK 理论扩展到多头注意力（heads→∞时为GP），本文关注有限宽度下的不收敛现象
- **vs Performers (Choromanski et al., 2021)**: Performers 提供线性化注意力的高效实现，本文则从 NTK 理论角度分析线性化注意力的学习动态本质
- **vs Zhang et al. (2022)**: 后者建立 NTK 影响力函数框架，本文在此基础上引入影响力可塑性来比较架构差异
- **vs Chizat et al. (2019)**: lazy/feature learning 的理论区分，本文给出注意力天然处于 feature learning regime 的具体证据

## 评分

- 新颖性: ⭐⭐⭐⭐ — 谱放大→NTK不收敛的理论链条新颖，影响力可塑性指标首次提出
- 实验充分度: ⭐⭐⭐ — 理论验证充分但数据集规模小，缺乏大模型实验
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，双重含义的叙事有说服力
- 价值: ⭐⭐⭐⭐ — 为理解注意力机制的本质提供了新视角，对对抗鲁棒性研究有启发
- 总评: ⭐⭐⭐⭐ — 一篇扎实的理论工作，核心洞察"注意力的能力与脆弱性同源"既优雅又实用，
  若能扩展到 softmax attention 和大规模模型将更具影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Nonparametric Teaching of Attention Learners](../../ICLR2026/robotics/nonparametric_teaching_of_attention_learners.md)
- [\[ICML 2026\] Dual Advantage Fields](../../ICML2026/robotics/dual_advantage_fields.md)
- [\[CVPR 2026\] AVA-VLA: Improving Vision-Language-Action models with Active Visual Attention](ava_vla_improving_vision_language_action_models_with_active_visual_attention.md)
- [\[ICCV 2025\] TransiT: Transient Transformer for Non-line-of-sight Videography](../../ICCV2025/robotics/transit_transient_transformer_for_non-line-of-sight_videography.md)
- [\[ICML 2025\] Geometric Contact Flows: Contactomorphisms for Dynamics and Control](../../ICML2025/robotics/geometric_contact_flows_contactomorphisms_for_dynamics_and_control.md)

</div>

<!-- RELATED:END -->
