---
title: >-
  [论文解读] Robust Tabular Foundation Models
description: >-
  [AAAI 2026][自监督学习][tabular foundation model] 提出 RTFM——一种模型无关的对抗训练框架，通过在合成数据生成器的参数空间上做 min-max 优化（最大化 TFM 与传统树模型之间的"最优性差距"），仅用不到 10 万额外合成数据集就显著提升了 TabPFN V2 在多个表格基准上的表现。
tags:
  - AAAI 2026
  - 自监督学习
  - tabular foundation model
  - adversarial training
  - synthetic data
  - distributionally robust optimization
---

# Robust Tabular Foundation Models

**会议**: AAAI 2026  
**arXiv**: [2512.03307](https://arxiv.org/abs/2512.03307)  
**代码**: 待确认  
**领域**: 自监督 / 表格基础模型  
**关键词**: tabular foundation model, adversarial training, synthetic data, distributionally robust optimization

## 一句话总结

提出 RTFM——一种模型无关的对抗训练框架，通过在合成数据生成器的参数空间上做 min-max 优化（最大化 TFM 与传统树模型之间的"最优性差距"），仅用不到 10 万额外合成数据集就显著提升了 TabPFN V2 在多个表格基准上的表现。

## 研究背景与动机

### 表格数据深度学习的困境

近年来，深度学习在计算机视觉和自然语言处理领域取得了巨大成功，但在结构化表格数据上，梯度提升树（XGBoost、CatBoost 等）长期占据主导地位。多项大规模基准研究表明，深度学习方法在表格任务上尚未全面超越树模型，这一"表格数据之谜"驱动了大量新方法的出现。

### 表格基础模型（TFM）的崛起

以 TabPFN 为代表的表格基础模型采用 **上下文学习（in-context learning, ICL）** 范式：模型接收带标签的训练样本和待预测的测试样本作为输入序列，**零样本** 地在毫秒级完成新数据集上的预测。其核心训练策略是利用 **结构因果模型（SCM）** 生成大量合成数据集进行预训练。TabPFN V2 在不少数据集上已经超越传统树模型，但在某些类型的数据集上仍然落后。

### 核心洞察：固定先验的局限

现有 TFM（TabPFN、Mitra、TabICL）均从 **固定的先验分布** 中采样 SCM 参数来生成训练数据。然而固定先验不可避免地 **欠表示** 参数空间中的某些区域——例如特定的特征维度、分类特征比例或非线性程度。这导致模型在具有相似结构的真实数据集上表现退化。

本文的关键洞察是：既然数据生成器的参数可以被显式参数化，那么就可以从 **对抗鲁棒性** 的视角来审视训练过程——让"对手"主动寻找模型表现最差的参数区域，然后重点在这些区域上训练模型。

## 方法详解

### 整体框架

RTFM 是一个 **两阶段迭代优化** 框架（如 Figure 1 所示），包含 **最大化阶段（参数搜索）** 和 **最小化阶段（模型训练）**，二者交替进行直到收敛：

1. **最大化阶段**：冻结模型权重 $\mathbf{W}$，使用黑盒优化算法（Optuna + TPE）在 SCM 参数空间 $\mathcal{P}$ 中搜索使"最优性差距"最大的参数配置。
2. **最小化阶段**：根据搜索到的参数及其最优性差距构造 softmax 采样分布 $Q$，按此分布生成训练数据，微调模型以缩小差距。

### 最优性差距（Optimality Gap）的定义

传统对抗训练直接最大化模型损失，但这可能将优化引向"任何模型都难以学习"的数据集区域——这并不是模型需要改进的地方。RTFM 的核心创新在于最大化的是 **最优性差距** 而非绝对损失：

$$\delta_\theta(\mathbf{W}) = \mathbb{E}_{\phi \sim p(\Phi;\theta)}\left[\mathcal{L}_{PFN}(\mathbf{W};\phi) - H_\phi(Z_y|Z_x)\right]$$

其中 $H_\phi(Z_y|Z_x)$ 是贝叶斯最优预测器可达到的条件交叉熵下界。由于条件熵在实际中难以估计，论文用 **多个强基线模型（XGBoost、CatBoost、Random Forest 等）的最小交叉熵损失** 来近似：

$$\widehat{\delta}_\theta(\mathbf{W}) = \mathbb{E}_{\phi}\left[\mathcal{L}_{PFN}(\mathbf{W};\phi) - \min_{k \in [e]} \mathcal{L}_{PFN}(f_k, \phi)\right]$$

这给出了最优性差距的一个下界。当该下界为正时，模型确实有改进空间。

### 分布鲁棒优化（DRO）公式化

直接对单一参数配置做最大化容易导致过拟合。论文将问题提升到 **分布鲁棒优化** 框架：允许"对手"选择参数空间上的一个分布 $Q$，而非单一参数，同时施加最小熵约束 $H(Q) \geq H_{min}$ 以避免分布退化到点分布。

论文证明（见附录 C），DRO 问题的最优解恰好是 **softmax 分布**：

$$q_i^* \propto \exp(\eta \cdot \widehat{\delta}_{\theta_i}(\mathbf{W}))$$

温度参数 $\eta$ 由最小熵 $H_{min}$ 和各参数的最优性差距唯一确定，可通过一维搜索（如二分法）高效求解。实际中令 $H_{min} = c \log(n_{trials})$，其中 $c \in (0,1)$ 为超参数。

### 最大化阶段的具体实现

- 使用 Optuna 框架中的 **Tree-structured Parzen Estimator (TPE)** 作为黑盒优化器，进行 $n_{trials}=100$ 轮搜索。
- 每轮对提议的参数 $\theta_i$，采样 $n_{ds}=20$ 个合成数据集，用 $e=7$ 个基线模型分别拟合并评估，计算平均最优性差距。
- 关键加速：每对（数据集, 基线模型）的拟合完全独立，可在 $n_{ds} \times e = 140$ 个 CPU 核上并行执行，单试次仅需数秒。

### 最小化阶段的具体实现

- 根据最大化阶段得到的 $\{(\theta_i, \widehat{\delta}_{\theta_i})\}$ 构造 softmax 采样分布 $Q$。
- 每个训练 batch 从 $Q$ 中采样参数 $\theta_i$，再从 $p(\Phi;\theta_i)$ 中采样生成器和数据集。
- 使用学习率 $1 \times 10^{-5}$、batch size 64，每轮训练 3000 步。
- 完整 max-min 循环进行 30 个 epoch。
- **自蒸馏机制**：第 5 个 epoch 后，将原始 TabPFN 模型加入基线池，防止模型在某些区域"遗忘"原始能力。

### 参数空间

SCM 通过随机初始化的 MLP 实现，可调参数包括：层数 $l$、隐藏层大小 $h$、激活函数 $a$、分类特征比例 $r_{cat}$ 等。这些超参数的分布各自被参数化（例如 $r_{cat} \sim \text{TruncNorm}(\mu_{r_{cat}}, 0, 1)$），所有参数化的均值组成总参数向量 $\theta$。

## 实验关键数据

### 表 1：TabPertNet 基准结果

| 指标 | Log. Reg. | MLP | Random Forest | CatBoost | XGBoost | TabPFN | TabPFN (RTFM) |
|------|-----------|-----|---------------|----------|---------|--------|---------------|
| Mean Rank AUC | 5.1 | 4.6 | 4.0 | 3.8 | 4.6 | 3.2 | **2.7** |
| Mean Norm. AUC | 0.4253 | 0.5005 | 0.6481 | 0.6663 | 0.5222 | 0.7483 | **0.8167** |
| Rank-1 Wins | 1 | 8 | 5 | 7 | 5 | 11 | **17** |

RTFM 在 TabPertNet 上将 TabPFN 的平均归一化 AUC 从 0.7483 提升至 **0.8167**（+6.8%），Rank-1 胜出次数从 11 增至 17。

### 表 2：TabArena 基准结果

| 指标 | Log. Reg. | MLP | Random Forest | CatBoost | XGBoost | TabPFN | TabPFN (RTFM) |
|------|-----------|-----|---------------|----------|---------|--------|---------------|
| Mean Rank AUC OVO | 4.9 | 6.3 | 4.8 | 3.4 | 4.5 | 2.2 | **1.9** |
| Mean Norm. AUC OVO | 0.4277 | 0.1801 | 0.5761 | 0.7749 | 0.5918 | 0.9031 | **0.9298** |
| Rank-1 Wins | 2 | 0 | 0 | 2 | 0 | 5 | **12** |

在 TabArena 上 RTFM 同样全面领先，Rank-1 胜出从 5 次跃升至 12 次。Wilcoxon 检验在两个基准上均表明 RTFM 相对原始 TabPFN 的提升具有统计显著性（TabPertNet: $p=0.0023$, TabArena: $p=0.0103$）。

## 亮点与洞察

1. **最优性差距 vs 绝对损失**：最核心的设计洞察——对抗训练不应最大化绝对损失，而应最大化与可达最优之间的差距。这避免了将训练资源浪费在"本身就不可学"的数据分布上。

2. **合成数据的极高效率**：仅用 9 万额外合成数据集（不到 TabPFN 原始预训练数据的 1%）即实现显著提升，说明"精准定位弱点"远比"大规模数据堆叠"高效。

3. **模型无关性**：RTFM 框架不依赖特定模型架构，理论上可应用于任何 TFM（如 Mitra、TabICL），也可扩展到回归任务。

4. **"跳跃"现象**：在 TabPFN 原本落后于树模型的数据集中（rank > 2），约 20-21% 的情况下 RTFM 使其直接跃升为排名第一的模型，说明对抗训练有效补齐了特定短板。

5. **自蒸馏防遗忘**：第 5 个 epoch 后将原始模型加入基线池的设计巧妙地缓解了对抗训练可能带来的性能退化问题。

## 局限性

1. **仅验证了分类任务**：虽然框架理论上可扩展到回归，但实验仅在分类基准上进行，回归场景的效果未知。
2. **SCM 限于 MLP**：当前仅使用 MLP 作为数据生成器，尚未纳入树结构 SCM，可能限制了生成数据的多样性和对树模型擅长区域的覆盖。
3. **计算开销不可忽视**：最大化阶段每轮需要拟合 $n_{trials} \times n_{ds} \times e = 14000$ 个基线模型，虽然高度可并行，但仍需 256 核 CPU + A100 GPU 的大规模算力。
4. **最优性差距估计偏差**：用有限个基线模型的最佳表现来近似贝叶斯最优值，本质上是一个下界估计，可能低估真实差距，导致某些真正薄弱的区域被遗漏。
5. **仅以 TabPFN V2 为实验对象**：未在其他 TFM（Mitra、TabICL）上验证泛化性。

## 相关工作与启发

- **TabPFN 系列**（Hollmann et al. 2023, 2025）：开创了基于 Prior-Fitted Network 的表格基础模型范式，本文直接在其基础上做对抗微调。
- **Wu & Bergman 2025**：同期工作，也尝试在 GAN 式训练中调整 SCM 权重，但仅关注特定类别 SCM 的权重调整，范围较窄。RTFM 提供了更通用的参数空间对抗优化框架。
- **DRO（分布鲁棒优化）**（Rahimian & Mehrotra 2019）：RTFM 的理论支柱之一。将 DRO 引入合成数据生成领域是新颖的应用。
- **Madry et al. 2019**：经典对抗训练工作，RTFM 将其思想从输入空间对抗迁移到了数据生成器参数空间对抗。

**启发**：这种"在合成数据生成器参数空间上做对抗搜索"的思路具有很强的通用性——不仅限于表格数据，任何基于合成数据预训练的基础模型都有可能借鉴此框架来定位和补强薄弱领域。

## 评分

⭐⭐⭐⭐ (4/5)

**理由**：论文提出了一个理论扎实、实践高效的 TFM 对抗训练框架，DRO 视角的引入有较强的理论贡献，实验结果统计显著。主要扣分点在于实验范围偏窄（仅分类、仅 TabPFN、仅 MLP-SCM），对框架通用性的实证支撑不足。

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Test-Time Canonicalization by Foundation Models for Robust Perception](../../ICML2025/self_supervised/test-time_canonicalization_by_foundation_models_for_robust_perception.md)
- [\[ICML 2025\] Towards Benchmarking Foundation Models for Tabular Data With Text](../../ICML2025/self_supervised/towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [\[AAAI 2026\] FedGRPO: Privately Optimizing Foundation Models with Group-Relative Rewards from Domain Clients](fedgrpo_privately_optimizing_foundation_models_with_group-relative_rewards_from_.md)
- [\[AAAI 2026\] Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)
- [\[NeurIPS 2025\] Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](../../NeurIPS2025/self_supervised/mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)

<!-- RELATED:END -->
