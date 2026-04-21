---
title: >-
  [论文解读] Challenging Forgets: Unveiling the Worst-Case Forget Sets in Machine Unlearning
description: >-
  [ECCV 2024][图像生成][machine unlearning] 提出从对抗视角识别"最坏情况遗忘集"的方法，通过双层优化框架找到最难被遗忘的数据子集，利用 SignSGD 将二阶 BLO 简化为一阶问题，从而更可靠地评估机器遗忘方法的真实效能。
tags:
  - ECCV 2024
  - 图像生成
  - machine unlearning
  - Bi-Level Optimization
  - Worst-Case Evaluation
  - Forget Set Selection
  - SignSGD
---

# Challenging Forgets: Unveiling the Worst-Case Forget Sets in Machine Unlearning

**会议**: ECCV 2024  
**arXiv**: [2403.07362](https://arxiv.org/abs/2403.07362)  
**代码**: [有](https://github.com/OPTML-Group/Unlearn-WorstCase)  
**领域**: 可信机器学习 / 机器遗忘  
**关键词**: machine unlearning, Bi-Level Optimization, Worst-Case Evaluation, Forget Set Selection, SignSGD

## 一句话总结

提出从对抗视角识别"最坏情况遗忘集"的方法，通过双层优化框架找到最难被遗忘的数据子集，利用 SignSGD 将二阶 BLO 简化为一阶问题，从而更可靠地评估机器遗忘方法的真实效能。

## 研究背景与动机

机器遗忘（Machine Unlearning, MU）旨在从已训练模型中消除特定数据的影响，同时保持模型对其余数据的效用。该领域源于"被遗忘权"等数据保护法规，已扩展到模型安全、公平性、版权保护等多个可信 ML 方向。

然而，现有 MU 评估存在根本性的可靠性缺陷：

**高方差问题**：现有评估几乎全部基于随机选择的遗忘集，同一遗忘方法在不同随机遗忘集上表现差异巨大（UA 方差可达 ±0.69），难以进行公平的方法间比较

**无法揭示最坏情况**：随机遗忘场景无法暴露 MU 方法在极端条件下的真实脆弱性，无法回答"这个遗忘方法到底有多鲁棒"这一关键问题

**缺少系统性压力测试**：与安全领域中对抗评估的标准做法相比，MU 评估缺乏完善的对抗性基准

作者实验发现即使是"完美"的精确遗忘方法 Retrain，在不同遗忘集选择下 UA 和 MIA 的变化幅度也很大。这促使他们提出核心研究问题：**如何系统性地找到对遗忘最具挑战性的数据子集？**

## 方法详解

### 整体框架

方法核心是一个**双层优化（Bi-Level Optimization, BLO）**框架，通过上下两层优化的交替迭代来识别最坏遗忘集：

- **上层优化**（数据选择层）：优化数据选择变量 $\mathbf{w}$，使遗忘后的模型在被选数据上仍保持低损失（即遗忘失败）
- **下层优化**（遗忘执行层）：给定遗忘集，执行标准遗忘训练获得遗忘后模型 $\theta_u(\mathbf{w})$

引入二值选择变量 $\mathbf{w} \in \{0,1\}^N$，其中 $w_i=1$ 表示第 $i$ 个训练样本被选入遗忘集 $\mathcal{D}_f$。完整 BLO 问题形式化为：

$$\min_{\mathbf{w} \in \mathcal{S}} \sum_{\mathbf{z}_i \in \mathcal{D}} [w_i \ell(\theta_u(\mathbf{w}); \mathbf{z}_i)] + \gamma \|\mathbf{w}\|_2^2 \quad \text{s.t.} \quad \theta_u(\mathbf{w}) = \arg\min_\theta \ell_{\text{MU}}(\theta; \mathbf{w})$$

上层目标的物理含义是：找到一组数据，使得遗忘后的模型仍然能很好地"记住"它们——即遗忘最难成功的那些数据点。

### 关键技术设计

**1. SignSGD 简化隐式梯度（核心创新）**

BLO 的关键难点在于上层梯度中的隐式梯度项 $\frac{d\theta_u(\mathbf{w})}{d\mathbf{w}}$。传统方法需要计算二阶导数：影响函数（IF）方法需要逆 Hessian-梯度乘积，梯度展开（GU）方法需沿优化路径反向传播，两者在深度网络上都面临严重的扩展性问题。

作者的核心洞察是：在下层优化中使用 SignSGD 替代标准 SGD 作为优化器：

$$\theta_j = \theta_{j-1} - \beta \cdot \text{sign}(\nabla_\theta \ell_{\text{MU}}(\theta_{j-1}; \mathbf{w}))$$

由于 sign 函数的导数几乎处处为零（$\frac{d\,\text{sign}(\mathbf{x})}{d\mathbf{x}} = \mathbf{0}$），梯度展开链条中的每一步隐式梯度都被截断，最终整个隐式梯度项消失：$\text{IG} = \mathbf{0}$。这将原本需要高代价二阶信息的 BLO 问题简化为仅需一阶偏导数 $\nabla_{\mathbf{w}} f$ 的交替优化。

**2. 交替优化求解**

- 上层：投影梯度下降（PGD）更新 $\mathbf{w}$，共 20 次迭代，学习率 $\alpha = 10^{-3}$
- 下层：SignSGD 执行遗忘训练，$K = 10$ 个 epoch
- 约束松弛：将二值约束 $\{0,1\}^N$ 松弛为连续约束 $[0,1]^N$ 且 $\mathbf{1}^\top \mathbf{w} = m$，投影步骤有闭式解
- 松弛后的连续 $\mathbf{w}$ 值可作为"遗忘难度分数"：最大值对应最难遗忘的数据，最小值对应最易遗忘的数据

**3. 扩展到类级和提示级遗忘**

框架具有良好的通用性：将 $\mathbf{w}$ 重新解释为类选择变量（class-wise forgetting，用于图像分类中的整类遗忘）或文本提示选择变量（prompt-wise forgetting，用于文生图模型的概念遗忘），即可无缝扩展到其他 MU 场景。

### 损失函数与训练策略

- 下层遗忘目标基于 FT 策略：$\ell_f = -\ell_r$（对遗忘集做梯度上升、保留集做标准训练），$\lambda = 1$
- 上层 $\ell_2$ 正则化参数 $\gamma = 10^{-4}$，同时起到鼓励 $\mathbf{w}$ 稀疏性和增强 BLO 优化稳定性的双重作用

## 实验关键数据

### 精确遗忘验证

在 CIFAR-10 / ResNet-18 上，用 Retrain（精确遗忘）验证最坏遗忘集的有效性：

| 指标 | 遗忘率 | 随机遗忘集 | 最坏情况遗忘集 | 差异 |
|------|--------|-----------|-------------|------|
| UA   | 10%    | 5.28 ± 0.33 | 0.00 ± 0.00 | ▼5.28 |
| MIA  | 10%    | 12.86 ± 0.61 | 0.00 ± 0.00 | ▼12.86 |
| RA   | 10%    | 100.00 | 100.00 | — |
| TA   | 10%    | 94.38 ± 0.15 | 94.66 ± 0.09 | ▲0.28 |

在 1%/5%/10%/20% 四个遗忘比例下，最坏遗忘集始终将 UA 和 MIA 压低至 ≈0%，且方差为 0，同时模型效用（RA/TA）完全不受损。

### 近似遗忘方法重新评估

对 9 种近似遗忘方法在 CIFAR-10 上以 10% 遗忘率测评，对比随机 vs 最坏遗忘集下与 Retrain 的平均性能差距：

| 方法类别 | 代表方法 | 随机遗忘集 Avg. Gap | 最坏遗忘集 Avg. Gap | 趋势 |
|---------|---------|-------------------|-------------------|------|
| 无重标签 | FT | 2.00 | 1.37 | 差距缩小 |
| 无重标签 | CF-k | 4.36 | 0.08 | 差距大幅缩小 |
| 无重标签 | SCRUB | 9.91 | 0.82 | 差距大幅缩小 |
| 重标签 | RL | 4.38 | **24.88** | 差距剧烈扩大 |
| 重标签 | BS | 6.84 | **37.22** | 差距剧烈扩大 |
| 重标签 | BE | 4.28 | **34.70** | 差距剧烈扩大 |
| 重标签 | SalUn | 1.89 | **24.51** | 差距剧烈扩大 |

核心发现：重标签方法（RL, BE, BS, SalUn）在最坏遗忘集下暴露严重缺陷——MIA 飙升至 80-96%，表明它们提供了"虚假的遗忘安全感"。而无重标签方法（FT, CF-k, SCRUB）反而跟随 Retrain 的趋势表现稳定。

### Coreset 分析

最坏遗忘集的补集（即保留集）在 CIFAR-10 和 CIFAR-100 上被训练模型后，测试精度与 EL2N、GraNd 等 SOTA coreset 选择方法相当，甚至在 CIFAR-100 的 90%/95% 选择率下超越了全量训练的模型精度。这表明最坏遗忘集的本质是"非核心子集"——最难遗忘的数据恰好是训练中最不重要的冗余数据。

### 更多验证

- **多数据集**：CIFAR-100、CelebA、Tiny ImageNet 上结论一致
- **多模型**：VGG、ResNet-50 上趋势相同
- **类级遗忘**：在 ImageNet / ResNet-18 上成功识别最难遗忘的类别
- **提示级遗忘**：在 UnlearnCanvas / Latent Diffusion 上成功应用于生成模型的概念遗忘
- **偏差数据集案例**：在 CelebA 发色预测任务中，最坏遗忘集大量包含（Blond + Female）样本——这些是数据中虚假相关性最强的"容易学"样本，印证了非核心子集假说

## 亮点与洞察

- **SignSGD 的巧妙利用**：利用 sign 函数导数为零将二阶 BLO 退化为一阶优化，这一简化既优雅又实用，是全文最精彩的技术贡献。仅需一阶梯度即可完成原本代价极高的双层优化
- **评估范式转变**：从"随机遗忘→算指标"转向"对抗选择遗忘集→暴露最坏情况"，为 MU 领域提供了更可靠的压力测试基准
- **MU 与 Coreset 的深层联系**：最难遗忘的数据 ≈ 训练中最不重要的数据，这一发现将数据选择与机器遗忘两个看似独立的领域连接起来
- **方法鲁棒性排名反转**：重标签方法在随机评估下看似优秀，在最坏情况下却全面崩溃，这对 MU 方法选择有重要指导意义

## 局限性与改进方向

1. **下层遗忘策略固定为 FT**：BLO 下层统一使用 FT+梯度上升策略，但不同 MU 方法有各自不同的遗忘目标函数，理想做法应针对每种方法单独求解其最坏遗忘集
2. **计算开销**：上层每次迭代都需完整执行下层遗忘训练（K 个 epoch），大规模数据上总开销仍然较高
3. **二值松弛的近似误差**：连续松弛加 top-m 选择可能导致次优遗忘集，特别是当 $\mathbf{w}$ 分布不够尖锐时
4. **领域覆盖有限**：仅涵盖视觉任务（分类 + 文生图），大语言模型的知识遗忘场景尚未探索

## 与相关工作的对比

- 与 MIA 等现有对抗指标不同，本文从**数据选择**角度评估 MU，与其他效果指标正交互补
- 与 SalUn 等基于显著性的遗忘方法不同，本文关注的是**评估侧**而非方法侧的改进
- BLO 框架借鉴了 coreset selection 中的数据选择思想，但目标函数针对遗忘难度而非训练效率

## 启发与延伸

- 该框架可直接应用于 LLM 知识遗忘的安全性评估，例如评测 RLHF 后的安全边界是否真的不可逆
- Coreset 与 MU 的交叉联系值得深入挖掘：最重要的训练数据 ≈ 最难遗忘的数据，可用于理解数据对模型的影响机制
- 对抗评估思想可推广到联邦遗忘、图神经网络遗忘等其他 MU 变体

## 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次系统性提出最坏遗忘集识别问题 |
| 技术深度 | ⭐⭐⭐⭐⭐ | SignSGD 简化 BLO 的推导精巧优雅 |
| 实验充分度 | ⭐⭐⭐⭐ | 多数据集、多模型、多场景全面验证 |
| 写作质量 | ⭐⭐⭐⭐ | 问题动机清晰，公式推导严谨 |
| 实用价值 | ⭐⭐⭐⭐ | 为 MU 提供了标准化的对抗评估工具 |

<!-- RELATED:START -->

## 相关论文

- [Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [MUNBa: Machine Unlearning via Nash Bargaining](../../ICCV2025/image_generation/munba_machine_unlearning_via_nash_bargaining.md)
- [Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs](../../ICLR2026/image_generation/model_collapse_is_not_a_bug_but_a_feature_in_machine_unlearning_for_llms.md)
- [Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](../../ICCV2025/image_generation/invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)
- [NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)

<!-- RELATED:END -->
