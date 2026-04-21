---
title: >-
  [论文解读] Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch
description: >-
  [ECCV 2024][类别不平衡] 本文提出 RECD 算法，通过蒙特卡洛近似估计未标注数据的未知类别分布，基于估计分布重新平衡分类器，并引入特征聚类压缩缓解特征图不平衡，在标注-未标注数据类别分布失配的半监督学习场景中取得 SOTA 性能。
tags:
  - ECCV 2024
  - 类别不平衡
  - 半监督学习
  - 类别分布估计
  - 分布失配
  - 特征聚类压缩
---

# Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch

**会议**: ECCV 2024  
**arXiv**: 无公开 arXiv  
**代码**: 无  
**领域**: 半监督学习  
**关键词**: 类别不平衡, 半监督学习, 类别分布估计, 分布失配, 特征聚类压缩

## 一句话总结

本文提出 RECD 算法，通过蒙特卡洛近似估计未标注数据的未知类别分布，基于估计分布重新平衡分类器，并引入特征聚类压缩缓解特征图不平衡，在标注-未标注数据类别分布失配的半监督学习场景中取得 SOTA 性能。

## 研究背景与动机

**领域现状**：类别不平衡半监督学习（Class-Imbalanced Semi-Supervised Learning, CISSL）是半监督学习中的重要课题，旨在从少量标注数据和大量未标注数据中学习，同时应对类别分布不平衡的挑战。近年来的 CISSL 方法（如 DARP、DASO、ABC 等）取得了显著进展。

**现有痛点**：现有 CISSL 算法通常显式或隐式地假设未标注数据的类别分布与标注数据一致（distribution match）。然而在实际应用中，这一假设往往不成立——标注数据可能严重倾斜于某些类别（因为标注成本不同），而未标注数据可能有完全不同的类别分布。当分布失配（distribution mismatch）时，基于错误分布假设的方法会在训练中给各类别分配不正确的权重，导致分类性能显著退化。

**核心矛盾**：要正确地重新平衡分类器，需要知道整体训练数据的真实类别分布——但未标注数据的类别分布恰恰是未知的。现有方法要么假设分布已知，要么用简单启发式估计（如利用模型预测的类别频率），但这些估计在模型训练早期极不可靠，形成鸡生蛋的问题。

**本文目标** (1) 如何在不依赖分布假设的情况下可靠估计未标注数据的类别分布？(2) 如何基于估计的分布有效重新平衡分类器？(3) 如何同时在特征空间层面缓解类别不平衡？

**切入角度**：作者提出利用蒙特卡洛近似方法，基于模型对未标注样本的预测类别概率来估计未标注数据的类别分布。这种估计方法随着模型改进而逐步准确，形成正反馈循环。同时在特征空间层面引入特征聚类压缩（Feature Clusters Compression）来缓解少数类别特征空间的稀疏问题。

**核心 idea**：通过蒙特卡洛近似估计未标注数据的类别分布，基于此重新平衡分类器，同时用特征聚类压缩缓解表示空间的不平衡。

## 方法详解

### 整体框架

RECD 在标准半监督学习框架（如 FixMatch）之上增加两个核心组件：(1) 类别分布估计模块——利用模型对未标注数据的预测概率，通过蒙特卡洛近似估计未标注数据的类别分布；(2) 特征聚类压缩模块——对少数类别的特征表示进行聚类并压缩，增加少数类别特征的密度。最终基于估计的混合类别分布（标注+未标注）重新平衡分类器的决策边界。

### 关键设计

1. **蒙特卡洛类别分布估计**:

    - 功能：估计未标注数据中各类别的样本比例
    - 核心思路：对于一批未标注数据 $\{u_i\}_{i=1}^{N_u}$，模型输出每个样本属于各类别的预测概率 $p(y=c|u_i)$。未标注数据的类别 $c$ 的比例可以通过蒙特卡洛近似：$\hat{\pi}_c = \frac{1}{N_u} \sum_{i=1}^{N_u} p(y=c|u_i)$。随着训练进行，模型预测逐渐准确，分布估计也随之改善。为提高估计稳定性，使用指数移动平均（EMA）对多个 mini-batch 的估计进行平滑
    - 设计动机：直接用模型预测的硬标签统计类别频率在模型不够准确时波动大，而软概率的蒙特卡洛近似更加平滑稳定。同时避免了对未标注数据分布的任何先验假设

2. **基于估计分布的分类器重新平衡**:

    - 功能：根据估计的真实类别分布调整分类器的决策边界
    - 核心思路：在标准训练中，分类器倾向于偏向多数类别。知道了数据的真实类别比例后，可以在逻辑回归层进行后验校正（logit adjustment）。具体而言，将分类器的 logit 值调整为 $\tilde{z}_c = z_c - \tau \log(\hat{\pi}_c^{total})$，其中 $\hat{\pi}_c^{total}$ 是标注和未标注数据的混合类别比例估计。这种调整等价于在贝叶斯框架下将先验从均匀分布修正为估计的真实分布
    - 设计动机：经典的重新平衡方法（如 logit adjustment）需要已知类别分布。本文通过估计模块提供了所需的分布信息，使得 logit adjustment 可应用于分布未知的半监督场景

3. **特征聚类压缩（Feature Clusters Compression, FCC）**:

    - 功能：在特征空间层面缓解少数类别特征的稀疏和分散问题
    - 核心思路：少数类别由于训练样本少，其特征在嵌入空间中往往稀疏且分散，导致分类器学到的决策边界不够紧致。FCC 对每个类别的特征进行聚类（如 K-means），然后将每个聚类内的特征向聚类中心收缩，增大少数类别特征的密度。这种操作在不增加真实样本的情况下，使少数类别在特征空间中占据更紧致的区域，近似于过采样的效果但不会引入重复样本的过拟合风险
    - 设计动机：分类器层面的重新平衡调整了决策边界，但特征空间的不平衡仍然存在。FCC 从表示学习的角度补充解决了这一问题，使得两层面的重新平衡协同工作

### 损失函数 / 训练策略

RECD 基于 FixMatch 框架，总损失包含三部分：(1) 标注数据的交叉熵损失（带 logit adjustment）；(2) 未标注数据的一致性正则化损失（强增广预测与弱增广伪标签的交叉熵）；(3) FCC 的辅助正则化损失。类别分布估计通过 EMA 更新，logit adjustment 的温度系数 $\tau$ 作为超参数控制调整强度。

## 实验关键数据

### 主实验

| 数据集 | 不平衡比 | 分布 | RECD | DARP | DASO | ABC |
|--------|---------|------|------|------|------|-----|
| CIFAR-10 | 100 | match | SOTA | 次优 | - | - |
| CIFAR-10 | 100 | mismatch | SOTA | 显著下降 | 显著下降 | 下降 |
| CIFAR-100 | 50 | match | SOTA | 次优 | - | - |
| CIFAR-100 | 50 | mismatch | SOTA | 显著下降 | 下降 | 下降 |
| STL-10 | 20 | mismatch | SOTA | 下降 | 下降 | 下降 |
| Food-101 | 50 | mismatch | SOTA | 下降 | 下降 | - |

### 消融实验

| 配置 | 准确率 | 说明 |
|------|-------|------|
| Full RECD | 最优 | 完整模型 |
| w/o 分布估计 | 下降 2-5% | 使用标注数据分布替代，失配时退化明显 |
| w/o FCC | 下降 1-2% | 特征空间不平衡未解决 |
| w/o logit adjust | 下降 3-4% | 分类器决策边界偏移 |
| 硬标签估计替代MC | 下降 1-2% | 软概率估计更稳定 |

### 关键发现

- 分布失配是现有 CISSL 方法的"阿喀琉斯之踵"：当假设不满足时，几乎所有方法性能都显著退化，而 RECD 保持稳健
- 蒙特卡洛估计的类别分布随训练收敛，后期估计误差极小
- FCC 对少数类别的提升最为明显，尤其在极端不平衡比（100:1）下
- 在分布匹配（match）场景下，RECD 也能取得有竞争力的结果，说明方法的通用性
- EMA 平滑对估计稳定性至关重要，直接使用单批次估计波动过大

## 亮点与洞察

- **分布无关的设计理念**：RECD 不对未标注数据的分布做任何假设，而是自适应地估计和利用，这使其在"分布匹配"和"分布失配"两种场景下都有效。这种设计理念可推广到其他需要利用未标注数据但分布未知的学习范式
- **蒙特卡洛近似 + EMA 的简洁估计方案**：不需要复杂的分布建模或生成模型，仅利用模型自身的预测概率就能有效估计类别分布。这种"自举"式估计策略简单有效
- **双层面重新平衡**：同时在分类器层面（logit adjustment）和特征空间层面（FCC）解决不平衡问题，两者互补

## 局限与展望

- 蒙特卡洛分布估计在训练初期（模型尚不准确时）可能不可靠，可能需要 warm-up 策略
- FCC 的聚类操作增加了计算开销，在大规模数据集上的效率需要验证
- 未讨论开集场景（未标注数据包含标注集未涵盖的新类别）下的表现
- logit adjustment 的温度系数 $\tau$ 是超参数，不同数据集可能需要不同设置
- 未与最新的基于对比学习的半监督方法进行对比

## 相关工作与启发

- **vs DARP**: DARP 通过对齐伪标签分布与估计的先验分布来缓解不平衡，但假设先验已知或可从标注数据推断。RECD 直接估计未标注数据分布，更灵活
- **vs CReST**: CReST 通过类别重新平衡的自训练迭代处理不平衡，但也假设未标注数据分布已知。RECD 处理了更一般的分布失配场景
- **vs Logit Adjustment (Menon et al.)**: 经典 logit adjustment 需要已知类别先验，RECD 将其扩展到先验未知的半监督场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 分布失配下的 CISSL 关注不足，蒙特卡洛估计+双层重新平衡的组合有新意
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、多种不平衡比和分布设置的全面评估
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法动机充分
- 价值: ⭐⭐⭐⭐ 解决了 CISSL 实际部署中的关键假设问题，实用性强

<!-- RELATED:START -->

## 相关论文

- [ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-Agnostic Counting](abc_easy_as_123_a_blind_counter_for_exemplar-free_multi-class_class-agnostic_cou.md)
- [STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning](stsp_spatial-temporal_subspace_projection_for_video_class-incremental_learning.md)
- [Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](../../CVPR2025/others/distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)
- [Rethinking Data Bias: Dataset Copyright Protection via Embedding Class-Wise Hidden Bias](rethinking_data_bias_dataset_copyright_protection_via_embedding_class-wise_hidde.md)
- [Prediction-Powered Semi-Supervised Learning with Online Power Tuning](../../NeurIPS2025/others/prediction-powered_semi-supervised_learning_with_online_power_tuning.md)

<!-- RELATED:END -->
