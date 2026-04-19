---
title: >-
  [论文解读] ModHiFi: Identifying High Fidelity Predictive Components for Model Modification
description: >-
  [NeurIPS 2025][模型压缩][model modification] 提出 Subset Fidelity 度量和 ModHiFi 框架，通过理论证明 Lipschitz 连续网络的局部重构误差线性上界全局误差，无需训练数据、损失函数或梯度，仅用合成数据即可识别模型中的高保真 (HiFi) 组件，统一实现结构化剪枝和类别遗忘两大任务。
tags:
  - NeurIPS 2025
  - 模型压缩
  - model modification
  - 剪枝
  - machine unlearning
  - subset fidelity
  - Lipschitz continuity
  - synthetic data
---

# ModHiFi: Identifying High Fidelity Predictive Components for Model Modification

**会议**: NeurIPS 2025  
**arXiv**: [2511.19566](https://arxiv.org/abs/2511.19566)  
**代码**: [DhruvaKashyap/modhifi](https://github.com/DhruvaKashyap/modhifi)  
**作者**: Dhruva Kashyap, Chaitanya Murti, Pranav Nayak, Tanay Narshana, Chiranjib Bhattacharyya (IISc, HP AI Lab, Google)
**领域**: model_compression  
**关键词**: model modification, structured pruning, machine unlearning, subset fidelity, Lipschitz continuity, synthetic data

## 一句话总结

提出 Subset Fidelity 度量和 ModHiFi 框架，通过理论证明 Lipschitz 连续网络的局部重构误差线性上界全局误差，无需训练数据、损失函数或梯度，仅用合成数据即可识别模型中的高保真 (HiFi) 组件，统一实现结构化剪枝和类别遗忘两大任务。

## 研究背景与动机

开放权重模型日益普及，但原始训练数据和损失函数通常不可得。这使得模型修改（剪枝、遗忘、去偏等）面临根本性挑战——现有方法大多依赖梯度或真实标签，在无损失函数、无训练数据的设置下不可行。

**现有方法的局限**：
- **视觉模型剪枝**：大多需要原始训练数据做微调，数据不可用时无法进行；L2 范数剪枝简单但效果差
- **LLM 剪枝**（SliceGPT、ShortGPT）：依赖校准数据集，且方法通常架构特定，不能泛化到遗忘等其他任务
- **类别遗忘**（Jia et al.）：需要梯度和微调，耗时且依赖原始训练数据
- **统一框架缺失**：剪枝和遗忘通常被独立研究，缺乏统一的组件重要性度量

**核心问题**：能否仅通过分布式访问（合成数据），在无梯度、无损失函数条件下，有效识别模型中对预测性能关键的组件？

## 方法详解

### 核心理论：局部到全局的误差传播

**统一抽象**：将 CNN 和 Transformer 统一建模——每层的输出通道 $c$ 都可分解为输入贡献的加性和：
$$\boldsymbol{Y}_c^l(\boldsymbol{X}) = \sum_{i=1}^{c_{in}^l} \boldsymbol{A}_{ci}^l(\boldsymbol{X})$$

其中 CNN 的 $\boldsymbol{A}_{ci}^l$ 是卷积操作，Transformer 的 $\boldsymbol{A}_{ci}^l$ 是 FFN 中间神经元的线性投影。

**Local-to-Global 定理**：对于 Lipschitz 连续网络，修改第 $l$ 层参数后的全局预测误差被局部重构误差线性上界：
$$\mathbb{E}[\|\mathrm{N}_\theta(\mathrm{X}) - \mathrm{N}_{\theta \odot M^l}(\mathrm{X})\|^2] \leq \mathcal{O}\left(\sum_c \mathbb{E}[\|\boldsymbol{Y}_c^l - \sum_{i \in C} m_{ci}^l \boldsymbol{A}_{ci}^l\|^2]\right)$$

**关键洞察**：与现有文献断言 Transformer 不是 Lipschitz 连续的相反，作者证明 well-trained Transformer 确实满足 Lipschitz 连续性（Corollary B.4），因此该定理同时适用于 CNN、ViT 和 LLM。

### Subset Fidelity 度量

定义子集 $C$ 对输出通道 $c$ 的保真度：
$$\mathrm{FS}_c^l(C) = \max_{\delta_c^l} \left(1 - \frac{\mathbb{E}[\|\boldsymbol{Y}_c^l - \sum_{i \in C} \delta_{ci}^l \boldsymbol{A}_{ci}^l\|^2]}{\mathbb{E}[\|\boldsymbol{Y}_c^l\|^2]}\right)$$

**性质**：有界性 $0 \leq \mathrm{FS} \leq 1$；单调性（子集越大保真度越高）。

**单例保真度的闭式解**：
$$s_{ci}^l = \mathrm{FS}_c^l(\{i\}), \quad \alpha_{ci}^l = \frac{\mathbb{E}[\langle \boldsymbol{Y}_c^l, \boldsymbol{A}_{ci}^l \rangle]}{\mathbb{E}[\|\boldsymbol{A}_{ci}^l\|^2]}$$

**最优性条件**：当输入贡献两两不相关时，按单例保真度排序的 NAIVE 选择策略是精确最优的（等价于 NP-hard 的 k-MFS 问题的精确解）。

### ModHiFi 算法框架

**ModHiFi-P（结构化剪枝）**：
1. 用合成数据前向传播，计算每层每个输入通道的单例保真度
2. 选择保真度最高的 $k$ 个通道作为 HiFi 集合
3. 将不在 HiFi 集合中的通道权重置零（删除）
4. 用闭式补偿项 $\delta^*$ 调整保留通道的权重，无需梯度微调

**ModHiFi-U（类别遗忘）**：
1. 仅用遗忘类别的样本计算各组件的保真度
2. 选择保真度最高的组件（即对该类最关键的组件）
3. 将这些组件的权重置零，破坏模型对该类的预测能力
4. 保留类的性能因未触及其关键组件而基本不受影响

**关键优势**：两个任务互为对偶——剪枝保留 HiFi 组件，遗忘删除 HiFi 组件。统一的保真度度量使同一算法框架适用于两种截然不同的任务。

## 实验关键数据

### Table 1: ImageNet 上 ResNet50 结构化剪枝对比

| 方法 | Accuracy | FLOP 压缩 | 参数压缩 | CPU 加速 | GPU 加速 |
|------|----------|----------|----------|----------|----------|
| Unpruned | 76.1 | 1x | 1x | 1x | 1x |
| GReg-2 | 73.9 | 3.02x | 2.31x | 1.36x | 1.53x |
| OTO | 74.7 | 2.86x | 2.81x | 1.25x | 1.45x |
| ThiNet | 71.6 | 3.46x | 2.95x | 1.38x | 1.50x |
| DFPC (54) | 73.80 | 3.46x | 2.65x | 2.37x | 2.38x |
| **ModHiFi-P** | **76.70** | 2.17x | 1.47x | **1.69x** | **1.70x** |
| **ModHiFi-P** (高压缩) | **73.82** | **3.66x** | **3.05x** | **2.42x** | **2.38x** |

在相同精度水平下，ModHiFi-P 实现了最优的加速比（比 DFPC 高约 11%），且无需原始训练数据。高压缩设置下 73.82% 精度对应 3.66x FLOP 压缩和 2.42x CPU 加速，超越所有基线。

### Table 3: Llama-2-7B 结构化剪枝

| 稀疏度 | 方法 | WikiText PPL | ARC-e | ARC-c | PIQA | WinoG. | HellaS. | Avg |
|--------|------|-------------|-------|-------|------|--------|---------|-----|
| 0% | Dense | 5.12 | 74.58 | 46.25 | 79.11 | 69.06 | 75.99 | 69.00 |
| 10% | SliceGPT | 6.46 | 56.14 | 35.33 | 69.53 | 64.80 | 59.02 | 59.96 |
| 10% | **ModHiFi-P-Alpaca** | 6.36 | **71.42** | **42.06** | **76.44** | **68.19** | **71.67** | **65.96** |
| 20% | ShortGPT | 14.32 | 58.33 | 38.05 | 72.58 | 65.51 | 65.27 | 59.95 |
| 20% | SliceGPT | 8.13 | 50.08 | 31.14 | 64.85 | 62.04 | 48.84 | 51.39 |
| 20% | **ModHiFi-P-Alpaca** | 9.38 | **64.73** | **38.22** | **72.79** | **64.64** | **62.70** | **60.62** |
| 30% | ShortGPT | 33.21 | 48.65 | 32.85 | 64.31 | 64.33 | 56.13 | 53.25 |
| 30% | SliceGPT | 10.96 | 44.19 | 27.47 | 58.71 | 57.46 | 41.27 | 45.82 |
| 30% | **ModHiFi-P-Alpaca** | 14.78 | **53.15** | **32.50** | **66.59** | **59.35** | **50.61** | **52.44** |

10% 稀疏度时平均精度 65.96%，大幅超越 SliceGPT 的 59.96%（+6.0%）。20% 和 30% 稀疏度时均为最优。Alpaca 合成数据比 WikiText 校准效果更好。

### Table 4: CIFAR-10 类别遗忘

| 模型 | 方法 | Forget Acc | Remain Acc | 时间 (秒) |
|------|------|------------|------------|----------|
| ResNet-50 | Base | 94.99 | 94.99 | - |
| ResNet-50 | Gradient Ascent | 6.59 | 93.44 | 30 |
| ResNet-50 | Jia et al. | 3.54 | 94.14 | 363 |
| ResNet-50 | **ModHiFi-U** | **0.20** | 92.98 | **10** |
| Swin-T | Jia et al. | 1.20 | 90.69 | 235 |
| Swin-T | ModHiFi-U | 8.83 | 73.57 | 2 |

ResNet-50 上 ModHiFi-U 实现完全遗忘（0.2% forget acc），速度比 Jia et al. 快 36 倍（10s vs 363s），且无需微调。Swin-T 上不微调效果稍差，但加 3 epoch 合成微调后大幅提升。

### HiFi 组件存在性验证

经验观察：所有评估模型中，每层不到 20% 的输入通道即可达到大于等于 0.8 的保真度。噪声实验更直观地验证：扰动 20% HiFi 组件导致精度下降约 12%，而扰动 80% 非 HiFi 组件仅下降 1%。

## 亮点

- **理论基础扎实**：首次证明 Lipschitz 连续网络的局部重构误差线性上界全局误差，并纠正了"Transformer 不是 Lipschitz 连续"的既有认知，为 well-trained Transformer 建立了 Lipschitz 连续性
- **统一框架**：同一个 Subset Fidelity 度量和 ModHiFi 算法同时适用于结构化剪枝和类别遗忘两个任务（互为对偶），跨 CNN 和 Transformer 架构通用，无需架构特定设计
- **无需梯度/损失/训练数据**：仅通过合成数据的前向传播即可识别关键组件，闭式补偿项消除了微调需求，这在实际部署中有很强的适用性
- **实际加速显著**：ImageNet 上比 SOTA 快约 11%（实际 CPU/GPU 加速），CIFAR-10 遗忘比基线快 36 倍

## 局限性

- **单例保真度的最优性依赖不相关假设**：实际网络中输入贡献通常存在相关性，NAIVE 选择策略在高度相关场景下可能不是最优
- **Swin-T 遗忘不微调效果差**：在更复杂的 Transformer 架构上，不微调时遗忘效果显著下降（forget acc 8.83%），说明方法对架构的复杂度有敏感性
- **仅分析 FFN 部分**：理论框架仅覆盖 Transformer 的 FFN 模块，未分析 Multi-Head Attention 的组件重要性，留作未来工作
- **合成数据质量影响**：虽然比 L2 剪枝+微调更鲁棒，但低质量合成数据仍会降低性能
- **仅验证了剪枝和遗忘两个任务**：虽框架理论上适用于去偏、持续学习等任务，但未提供实验验证

## 相关工作

- **结构化剪枝（视觉）**：GReg-2（正则化约束）、ThiNet（贪心通道选择）、DFPC（无数据剪枝+合成微调）、DepGraph（依赖图分析）——大多需要原始训练数据或架构特定设计
- **LLM 剪枝**：SliceGPT（正交投影降维）、ShortGPT（删除冗余层）、Wanda（权重乘激活值）——需要校准数据，不适用于遗忘
- **类别遗忘**：Gradient Ascent（简单但粗暴）、Jia et al.（基于 Fisher 信息的选择性遗忘）——需要梯度和微调，耗时且需要原始数据
- **ModHiFi 的定位**：首个统一剪枝和遗忘的无梯度、无损失函数框架，理论驱动而非启发式，合成数据即可工作

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — Subset Fidelity 概念新颖，Local-to-Global 理论将局部度量与全局性能严格关联；纠正 Transformer Lipschitz 连续性的认知具有独立理论价值
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 CNN/ViT/LLM、剪枝/遗忘双任务、多数据集；但 LLM 遗忘实验缺失，Swin-T 结果不太理想
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨，统一符号系统清晰；但数学符号密集，阅读门槛较高
- 价值: ⭐⭐⭐⭐⭐ — 解决了无训练数据/无损失函数下的模型修改这一实际且重要的问题，统一框架的思想对后续工作有显著启发

<!-- RELATED:START -->

## 相关论文

- [Predictive Data Selection: The Data That Predicts Is the Data That Teaches](../../ICML2025/model_compression/predictive_data_selection_the_data_that_predicts_is_the_data_that_teaches.md)
- [Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](../../ICLR2026/model_compression/discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation](../../CVPR2025/model_compression/curriculum_coarse-to-fine_selection_for_high-ipc_dataset_distillation.md)
- [Vision-centric Token Compression in Large Language Model](vision-centric_token_compression_in_large_language_model.md)
- [A Granular Study of Safety Pretraining under Model Abliteration](a_granular_study_of_safety_pretraining_under_model_abliteration.md)

<!-- RELATED:END -->
