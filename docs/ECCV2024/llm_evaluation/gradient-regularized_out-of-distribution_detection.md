---
title: >-
  [论文解读] Gradient-Regularized Out-of-Distribution Detection
description: >-
  [ECCV 2024][OOD检测] 提出 GReg/GReg+，通过正则化 OOD 评分函数的输入梯度范数来学习评分流形的局部平滑性，并结合基于能量评分的聚类采样策略选取高信息量辅助样本，在 CIFAR 和 ImageNet OOD 检测基准上取得 SOTA。
tags:
  - ECCV 2024
  - OOD检测
  - 梯度正则化
  - 能量评分
  - Lipschitz分析
  - 辅助数据采样
---

# Gradient-Regularized Out-of-Distribution Detection

**会议**: ECCV 2024  
**arXiv**: [2404.12368](https://arxiv.org/abs/2404.12368)  
**代码**: [https://github.com/o4lc/Greg-OOD](https://github.com/o4lc/Greg-OOD)  
**领域**: 其他  
**关键词**: OOD检测, 梯度正则化, 能量评分, Lipschitz分析, 辅助数据采样

## 一句话总结

提出 GReg/GReg+，通过正则化 OOD 评分函数的输入梯度范数来学习评分流形的局部平滑性，并结合基于能量评分的聚类采样策略选取高信息量辅助样本，在 CIFAR 和 ImageNet OOD 检测基准上取得 SOTA。

## 研究背景与动机

神经网络在遇到分布外（OOD）数据时常产生过度自信的错误预测，OOD 检测旨在区分分布内（ID）数据和 OOD 数据。

现有方法的局限：
- **后验方法**（MSP、ODIN、ReAct、DICE、LINe）：不使用辅助数据，通过推理时的统计量区分 ID/OOD，性能受限
- **使用辅助数据的方法**（OE、Energy Loss、POEM、DOS）：利用辅助 OOD 数据训练，但**仅关注评分函数的值**，未利用评分函数的局部结构信息
- **采样问题**：辅助数据集可能远大于 ID 数据，贪心采样会偏向特定区域导致泛化不足

核心动机来自一个简单观察（如论文 Figure 1 所示）：即使两个 OOD 样本 $x_1, x_2$ 的评分值相同，如果 $x_2$ 处的评分函数梯度很大（曲面陡峭），其邻域中的样本就容易被误检测为 ID。因此，仅优化评分值是不够的，还需要控制评分函数的局部光滑性。

## 方法详解

### 整体框架

GReg+ 由两个核心组件构成：
1. **梯度正则化（GReg）**：在能量损失基础上增加一个惩罚评分函数输入梯度范数的正则项
2. **基于能量的聚类采样**：对辅助 OOD 数据进行特征空间聚类后，从每个簇中按能量评分选取最具信息量的样本

### 关键设计

1. **梯度正则化损失**：基于能量评分函数 $S_\text{En}(x) = -\text{LSE}(f(x))$，在标准能量损失 $\mathcal{L}_{S_\text{En}}$ 的基础上，增加梯度范数正则项：

   $$\mathcal{L}_{\nabla S_\text{En}} = \mathbb{E}_{x_\text{in}} \| \mathbb{I}_{S_\text{En}(x_\text{in}) \leq m_\text{in}} \nabla_{x_\text{in}} S_\text{En}(x_\text{in}) \| + \mathbb{E}_{x_\text{aux}} \| \mathbb{I}_{S_\text{En}(x_\text{aux}) \geq m_\text{aux}} \nabla_{x_\text{aux}} S_\text{En}(x_\text{aux}) \|$$

   设计要点：**仅对已正确检测的样本施加梯度正则**——对于 ID 样本，只在其能量已足够低（正确归类为 ID）时惩罚梯度；对于 OOD 样本，只在其能量已足够高（正确归类为 OOD）时惩罚梯度。这样能量损失专注于"学会"检测，梯度损失专注于"稳固"已学到的检测。

   总损失函数：$\mathcal{L} = \mathcal{L}_\text{CE} + \lambda_S \mathcal{L}_S + \lambda_{\nabla S} \mathcal{L}_{\nabla S}$

2. **基于能量的聚类采样**：当辅助数据集很大时（如 300K RandomImages），需要有效选取样本。算法流程：

    - 提取所有 OOD 样本特征 $z_i = h(x_i)$ 并 L2 归一化为 $\hat{z}_i$
    - 使用 K-Means 将 $n_\text{OOD}$ 个样本聚为 $k = n_\text{ID}$ 个簇
    - 从每个簇中选取**能量最低的样本**（供能量损失 $\mathcal{L}_S$ 使用，需要提升的）和**能量最高的样本**（供梯度损失 $\mathcal{L}_{\nabla S}$ 使用，需要稳固的）

   聚类保证空间多样性，能量选择保证样本信息量，两者互补。

3. **理论保证（Lipschitz 分析）**：如果评分函数在 $x$ 的 $\varepsilon$-邻域内满足局部 Lipschitz 条件，则正确分类为 ID 的样本 $x$ 具有认证半径：

   $$\varepsilon^* = \min\left\{\varepsilon, \frac{\gamma - S(x)}{L_S}\right\}$$

   减小局部 Lipschitz 常数 $L_S$ 可扩大认证半径。对于 ReLU 网络，局部 Lipschitz 常数等于梯度范数，因此**惩罚梯度范数等价于控制局部 Lipschitz 常数**，为方法提供了鲁棒性理论基础。

### 损失函数 / 训练策略

- GReg（无采样）：在预训练模型上微调 20 epoch，使用 SGD + cosine annealing
- GReg+（带采样）：从头训练 50 epoch（lr=0.1）+ 微调 10 epoch（lr=0.01）
- 超参设置：$\lambda_S = 0.1$，$\lambda_{\nabla S} = 1$
- CIFAR 辅助数据：300K RandomImages，每 epoch 对每个 mini-batch 做聚类
- ImageNet 辅助数据：每 epoch 从剩余 990 类中随机采样 600K 张

## 实验关键数据

### 主实验

**CIFAR-10（平均 FPR95↓ / AUROC↑）：**

| 方法 | ResNet FPR95 | ResNet AUROC | WRN FPR95 | DenseNet FPR95 |
|------|-------------|-------------|-----------|--------------|
| Energy Loss | 11.14 | 97.53 | 13.11 | 11.26 |
| OpenMix | 22.24 | 96.26 | 21.92 | 22.86 |
| **GReg** | **7.90** | **97.95** | **7.95** | **7.93** |

**ImageNet（10 类 ID，FPR95↓ / AUROC↑ 平均）：**

| 方法 | FPR95↓ | AUROC↑ |
|------|--------|--------|
| LINe | 39.48 | 91.29 |
| DOS | 49.31 | 91.97 |
| Energy Loss | 45.23 | 88.63 |
| GReg | 47.26 | 90.13 |
| **GReg+** | **35.08** | **92.06** |

GReg+ 在 ImageNet 上将最佳对比方法的 FPR95 降低了 4% 以上。

### 消融实验

| 配置 | CIFAR-10 FPR95↓ | CIFAR-10 AUROC↑ | 说明 |
|------|-----------------|-----------------|------|
| OE | 21.76 | 95.80 | 基线 |
| OE + Grad | 18.79 | 96.32 | +梯度正则 |
| Energy | 11.26 | 97.43 | 基线 |
| Energy + Grad | 7.93 | 98.12 | +梯度正则，**相对改进 30%** |
| POEM | 29.15 | 94.18 | 基线 |
| POEM + Grad | 23.87 | 95.41 | +梯度正则 |

### 关键发现

- **梯度正则化具有通用性**：可即插即用地提升 OE、Energy、POEM 等多种方法的性能
- **能量损失与梯度损失互补**：能量损失负责拉开 ID/OOD 评分差距，梯度损失负责平滑局部流形
- **采样方法显著提升 CIFAR-100 和 ImageNet 性能**：GReg+ 比 GReg 在困难场景下贡献更大
- **梯度正则不损害 ID 精度**：同时提升 OOD 检测能力和 ID 分类精度
- **正则化在训练过程中抑制梯度范数增长**（Figure 3）：无 GReg 时梯度范数快速上升，有 GReg 时增长平缓

## 亮点与洞察

1. **从评分值到评分流形**：将关注点从评分函数的输出值扩展到其局部几何结构（梯度范数），是视角上的有意义创新
2. **理论与实践统一**：Lipschitz 分析为梯度正则提供了认证鲁棒性理论支撑，对于 ReLU 网络更是精确成立
3. **聚类采样设计精巧**：能量最低（待提升）和能量最高（待稳固）的双端采样策略与两个损失项精确匹配
4. **方法简洁且通用**：GReg 作为正则项可叠加到任何基于评分函数的 OOD 训练方法上

## 局限与展望

- 梯度计算引入额外反向传播开销（需要对输入求梯度），在大规模训练中可能较慢
- ImageNet 实验仅使用 10 类作为 ID，未验证全量 1K 类场景
- 理论分析基于局部 Lipschitz 假设，对于非分段线性激活函数的网络不完全适用
- 聚类采样需要预先提取所有辅助数据的特征，对于极大规模辅助集（如 ImageNet-21K）可能有扩展性问题
- 未讨论与近年更强的 post-hoc 方法（如 ASH、SHE 等）的结合

## 相关工作与启发

- **Energy Score**（Liu et al., NeurIPS 2020）：GReg 直接构建在能量评分之上
- **DOS**（ICLR 2024）：使用 K-Means 多样性采样的 SOTA 方法，GReg+ 的采样策略与之对比有改进
- **Certified Robustness**：梯度正则化与认证鲁棒性文献的 Lipschitz 分析建立了联系
- 核心启发：smoothness 作为设计理念可推广到其他检测任务（如 anomaly detection、novelty detection）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 利用评分函数梯度范数做正则化是新颖的视角，理论分析增加了深度
- **实验充分度**: ⭐⭐⭐⭐ — 多架构、多数据集、多消融的全面实验，还展示了对不同方法的通用性
- **写作质量**: ⭐⭐⭐⭐ — 动机直观（Figure 1 的 2D 示例），理论推导清晰
- **价值**: ⭐⭐⭐⭐ — 梯度正则化是即插即用的改进，实用价值高；但 ImageNet 实验规模偏小

<!-- RELATED:START -->

## 相关论文

- [Enhancing Out-of-Distribution Detection with Extended Logit Normalization](../../CVPR2026/llm_evaluation/enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)
- [OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](../../CVPR2025/llm_evaluation/oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](../../AAAI2026/llm_evaluation/graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection](../../ICCV2025/llm_evaluation/discopatch_taming_adversarially-driven_batch_statistics_for_improved_out-of-dist.md)
- [Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)

<!-- RELATED:END -->
