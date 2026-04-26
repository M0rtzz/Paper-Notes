---
title: >-
  [论文解读] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers
description: >-
  [AAAI2026][模型压缩][class-incremental learning] 提出 Sequential Learning with Drift Compensation (SLDC)，通过学习潜在空间转换算子（线性/弱非线性）来补偿预训练 ViT 在类增量学习中因序列微调导致的分布漂移，结合知识蒸馏后性能接近联合训练上界。
tags:
  - AAAI2026
  - 模型压缩
  - class-incremental learning
  - distribution drift
  - Transformer
  - 知识蒸馏
  - sequential fine-tuning
---

# Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers

**会议**: AAAI2026  
**arXiv**: [2511.09926](https://arxiv.org/abs/2511.09926)  
**代码**: [raoxuan98-hash/sldc](https://github.com/raoxuan98-hash/sldc)  
**领域**: model_compression  
**关键词**: class-incremental learning, distribution drift, vision transformer, knowledge distillation, sequential fine-tuning

## 一句话总结

提出 Sequential Learning with Drift Compensation (SLDC)，通过学习潜在空间转换算子（线性/弱非线性）来补偿预训练 ViT 在类增量学习中因序列微调导致的分布漂移，结合知识蒸馏后性能接近联合训练上界。

## 背景与动机

近年来，基于预训练 ViT 的类增量学习（CIL）研究表明，序列微调（SeqFT）后用高斯近似分布精炼分类器是一种有效的策略（如 SLCA/SLCA++）。然而，这一范式存在一个关键缺陷：**序列优化共享 backbone 参数会导致分布漂移**——之前任务学到的类特征分布与更新后模型的特征空间不再匹配，随着任务增加，分类器的效果逐渐退化。

现有工作大多通过蒸馏、模型集成、梯度投影等方式来"预防"表征漂移，而本文另辟蹊径，关注的是**一旦漂移发生后如何补偿其负面影响**。这一视角在 PTM-based CIL 中尚属新颖。

## 核心问题

1. **分布漂移问题**：SeqFT 在新任务上微调后，旧类的高斯分布（均值 $\mu_c$、协方差 $\Sigma_c$）不再与当前特征空间对齐，直接用旧分布采样合成特征来精炼分类器会产生偏差
2. **非线性变换过拟合**：直接用 MLP 学习前后特征空间的映射会过拟合，变换后的分布精度反而不如线性映射
3. **样本有限性**：exemplar-free 设定下无法保存旧任务数据，算子估计的样本多样性可能不足

## 方法详解

### 整体框架

SLDC 包含三个阶段：
1. **序列微调**：使用 LoRA 对 ViT backbone 在当前任务上微调（可选添加蒸馏 → SeqKD）
2. **分布补偿**：学习转换算子将旧类高斯分布对齐到新的特征空间
3. **分类器精炼**：从补偿后的高斯分布中采样合成特征，训练统一分类器

### 潜在空间转换算子

定义转换算子 $\mathcal{P}_{t-1 \to t}: \mathcal{F}_{t-1} \to \mathcal{F}_t$，将任务 $t-1$ 的特征空间映射到任务 $t$ 的特征空间。理想情况下，该算子能将旧类高斯分布的一阶矩（均值）和二阶矩（协方差）正确传播到新空间。

实际场景中无法访问完整输入空间，因此仅用当前任务数据 $\mathcal{D}_t$ 和前后两个冻结模型 $\mathcal{F}_{t-1}$、$\mathcal{F}_t$ 来近似估计。

### α₁-SLDC（线性变体）

对当前任务数据分别提取前后模型的特征矩阵，做 L2 归一化后，通过正则化最小二乘求解线性算子：

$$\mathbf{A}_t = \tilde{F}^t (\tilde{F}^{t-1})^\top \left( \tilde{F}^{t-1} (\tilde{F}^{t-1})^\top + \gamma_{\alpha_1} I_d \right)^{-1}$$

为应对小样本场景，进一步用启发式重加权平滑：$\mathbf{A}_t = (1-w)\mathbf{A}_t + wI_d$，其中 $w = \exp(-n_t / (\alpha_{\text{temp}} d))$。

对旧类高斯分布的补偿为闭式更新：$\mu_c \leftarrow \mathbf{A}_t \mu_c$，$\Sigma_c \leftarrow \mathbf{A}_t \Sigma_c \mathbf{A}_t^\top$。

### α₂-SLDC（弱非线性变体）

假设理想转换算子介于纯线性和完全非线性之间，构造弱非线性变换：

$$\mathcal{T}(f) = c_1 \mathbf{A} f + c_2 \psi(f)$$

其中 $c_1 + c_2 = 1$，$\mathbf{A}$ 为可学习矩阵，$\psi(f)$ 为两层 ReLU MLP。优化目标加入正则项 $\gamma_{\alpha_2}(c_1 - 1)^2$ 来控制非线性部分的贡献，防止过拟合。

由于弱非线性变换不再有闭式高斯传播公式，采用 **Monte Carlo 采样**估计补偿后分布：从旧高斯中抽 $N \gg d$ 个样本，经 $\mathcal{T}$ 变换后重新估计均值和协方差。

### β₁/β₂-SLDC（蒸馏增强变体）

在微调阶段加入特征级知识蒸馏来约束表征更新：

$$\mathcal{L}_{\text{All}} = \mathcal{L}_{\text{CE}} + \gamma_{\text{kd}} \mathcal{L}_{\text{KD}} + \gamma_{\text{norm}} \mathcal{L}_{\text{Norm}}$$

蒸馏减轻表征漂移 + SLDC 补偿残余分布漂移，二者互补。

### 辅助无标签数据增强（ADE）

当任务数据有限时，可利用任意来源的无标签辅助数据来丰富转换算子的估计。ADE 不需要标签且不保存旧任务数据，与 exemplar-free CIL 框架兼容。

## 实验关键数据

在四个数据集（CUB-200、Cars-196、CIFAR-100、ImageNet-R）上进行 10-task CIL 评估，使用两种预训练 ViT-B/16（MoCo-V3 自监督 / ImageNet-21K 有监督）。

**MoCo-V3 预训练的主要结果（Last-Acc）：**

| 方法 | CUB-200 | Cars-196 | CIFAR-100 | ImageNet-R |
|------|---------|----------|-----------|------------|
| Joint-Training（上界） | 81.82 | 81.16 | 88.86 | 75.95 |
| SeqFT（基线） | 64.40 | 60.42 | 73.36 | 61.37 |
| α₂-SLDC | 78.98 (+14.58) | 77.53 (+17.11) | 81.75 (+8.39) | 71.38 (+10.01) |
| β₂-SLDC | **81.82** (+4.85) | **80.10** (+6.23) | 85.16 (+4.81) | **73.01** (+6.08) |
| β₂-SLDC + ADE | **82.32** | **80.61** | **86.12** | **73.14** |
| SLCA++ | 75.48 | 69.71 | 84.77 | 69.01 |
| CoFiMA | 77.65 | 69.51 | 87.44 | 70.87 |

**关键发现：**

- β₂-SLDC 在 CUB-200 上达到 81.82%，与联合训练完全持平
- 在所有数据集上，β-SLDC 与联合训练的差距仅在 +0.50% 到 -3.29% 之间
- α₂-SLDC 相比 SeqFT 基线提升巨大（+8~17%），说明分布补偿极为有效
- MLP 直接做转换（MLPDC）在有蒸馏时反而降低性能（如 CUB-200 下降 4.41%），验证了过拟合问题
- ADE 在大多数场景下能进一步提升性能

## 亮点

1. **视角新颖**：不是预防漂移而是补偿漂移，在 CIL 领域是一种独特的思路
2. **理论与实践结合好**：弱非线性假设有 NTK 理论支撑，线性变体有闭式高斯传播的理论保证
3. **性能接近联合训练上界**：这是 CIL 研究的标志性结果，证明了 SeqFT + 分布补偿路线的可行性
4. **方法通用性强**：SLDC 可作为即插即用模块集成到现有 SeqFT 方法中
5. **MLP 过拟合的观察有价值**：揭示了分布补偿中非线性程度需要精心控制，弱非线性是更好的平衡点

## 局限与展望

1. **高斯假设的局限**：深层特征的分布未必是高斯的，对于多模态或长尾分布的类别，高斯近似可能不够准确
2. **转换算子的任务数据依赖**：若当前任务数据与旧任务数据分布差异极大，用当前任务数据估计的算子可能不适用于旧类
3. **线性变体在细粒度数据集上效果有限**：α₁-SLDC 在 Cars-196 上反而下降 7.75%（ImageNet-21K 预训练），说明线性假设不总成立
4. **弱非线性中 MLP 结构固定**：两层 ReLU MLP 的选择较为粗糙，更复杂的架构搜索可能有帮助
5. **ADE 需要额外数据**：虽然无需标注，但辅助数据的获取和选择仍需设计

## 与相关工作的对比

| 维度 | SLCA/SLCA++ | CoFiMA | RanPAC | SLDC |
|------|-------------|--------|--------|------|
| backbone 更新 | 低学习率/LoRA | 模型平均 | 冻结 | LoRA + 蒸馏 |
| 漂移处理 | 隐式（慢更新） | 模型融合 | 无（冻结） | 显式补偿算子 |
| 分类器精炼 | 高斯采样 | 高斯采样 | 随机投影 | 补偿后高斯采样 |
| 计算开销 | 低 | 中 | 低 | 中（需求解算子） |
| 任务 ID 依赖 | 否 | 否 | 否 | 否 |

SLDC 的核心优势在于直接建模特征空间的演化，可与蒸馏互补，而非简单替代。

## 启发与关联

1. **分布补偿思路可推广**：除了高斯假设外，可以探索对更复杂分布（如混合高斯、normalizing flow）的补偿
2. **弱非线性的设计哲学**：在过拟合和欠拟合之间寻找平衡点的思路值得在其他领域借鉴（如域适应、迁移学习中的特征对齐）
3. **与 prompt-based CIL 方法结合**：SLDC 目前基于 SeqFT 路线，能否与 L2P/CODA-Prompt 等方法结合值得探索
4. **接近联合训练的意义**：这一结果暗示 CIL 的核心瓶颈并非遗忘本身，而是分布漂移没有被正确处理

## 评分
- 新颖性: ⭐⭐⭐⭐ （补偿漂移的视角新，弱非线性假设有独到见解）
- 实验充分度: ⭐⭐⭐⭐⭐ （4 数据集 × 2 预训练模型 × 多变体对比，实验设计严谨全面）
- 写作质量: ⭐⭐⭐⭐ （定义清晰、推导完整，符号较多但组织合理）
- 价值: ⭐⭐⭐⭐ （性能接近联合训练上界是里程碑式结果，方法可即插即用）

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[AAAI 2026\] Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge](put_the_space_of_lora_initialization_to_the_extreme_to_preserve_pre-trained_know.md)
- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](../../CVPR2025/model_compression/cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [\[AAAI 2026\] Stratified Knowledge-Density Super-Network for Scalable Vision Transformers](stratified_knowledge-density_super-network_for_scalable_vision_transformers.md)

<!-- RELATED:END -->
