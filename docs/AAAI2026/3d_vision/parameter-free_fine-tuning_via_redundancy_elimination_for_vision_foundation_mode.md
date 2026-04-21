---
title: >-
  [论文解读] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models
description: >-
  [AAAI 2026][3D视觉][视觉基础模型] 发现视觉基础模型（SAM/SAM2/DINOv2）中存在大量冗余通道，提出无需更新任何参数的微调方法：通过基于输出差异的通道选择算法找到最优替换对，用有效通道替换冗余通道来增强下游任务的特征表示，平均提升 mIoU 5-11 个点。
tags:
  - AAAI 2026
  - 3D视觉
  - 视觉基础模型
  - 无参数微调
  - 通道冗余
  - SAM
  - 特征选择
---

# Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models

**会议**: AAAI 2026  
**arXiv**: [2504.08915](https://arxiv.org/abs/2504.08915)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 视觉基础模型, 无参数微调, 通道冗余, SAM, 特征选择

## 一句话总结

发现视觉基础模型（SAM/SAM2/DINOv2）中存在大量冗余通道，提出无需更新任何参数的微调方法：通过基于输出差异的通道选择算法找到最优替换对，用有效通道替换冗余通道来增强下游任务的特征表示，平均提升 mIoU 5-11 个点。

## 研究背景与动机

视觉基础模型（VFMs）如 SAM、DINOv2 在大规模数据上训练后具有强大的通用视觉表示能力。将其适配到下游任务通常需要微调参数：

- **全参数微调**：更新所有参数，计算成本高
- **参数高效微调**（PEFT/LoRA/Adapter）：更新少量参数（数千~数百万），但仍需反向传播和计算图维护

**关键观察**（Table 1 的控制实验）：在 SAM 的 PerSeg 数据集上，将某些通道的激活值设为 0：
- Channel 6 设为 0：mIoU 不变（50.6 → 50.6），说明该通道冗余
- Channel 216 设为 0：mIoU 反而提升（50.6 → 52.7），说明该通道甚至是有害的
- Channel 175/19/189 设为 0：mIoU 下降，说明这些通道对任务有效

**根因分析**：VFMs 在大规模数据上学到的通用特征中，很多对特定下游任务是无关甚至有害的。这种冗余是因为模型需要在众多任务间泛化。

**核心问题**：能否不修改任何模型参数，仅通过选择、复用和增强已有特征来适配下游任务？

## 方法详解

### 整体框架

与传统微调方法形成鲜明对比：
- **(a) 传统解码器微调**：更新解码器参数使预训练特征适配任务
- **(b) 传统编码器微调**：更新编码器参数修改预训练特征
- **(c) 本文方法**：不更新任何参数，仅将冗余通道替换为更有效的通道

流程：搜索数据集 → 编码器提取特征 → 逐对通道替换 → 比较输出差异 → 构建字典 → 搜索最优组合 → 应用替换

### 关键设计

1. **问题形式化**

   目标：找到最优替换对组合 $P^*$ 使在下游数据集 $S$ 上性能最大化：
   $$P^* = \arg\max_P \text{mIoU}(S, P)$$
   其中 $P = \{(i,j)_1, (i,j)_2, ..., (i,j)_k\}$，$(i,j)$ 表示用第 $j$ 通道替换第 $i$ 通道。

   直接枚举所有组合不可行：$C=256$ 时需 $2^{C^2}$ 次推理。

2. **通道选择算法**

   **三大降低搜索开销的策略**：

   **(1) 基于输出差异搜索**：
   给定搜索数据集 $\mathbf{S}$，编码器输出特征 $X \in \mathbb{R}^{D \times C \times W \times H}$。对每个替换对 $(i,j)$，计算：
   $$\Delta\text{Acc}_{(i \to j)} = D(X') - D(X)$$
   其中 $D(X)$ 和 $D(X')$ 分别是原始和替换后特征经解码器的输出。

   构建字典 $\mathcal{D} = \{(i,j): \Delta\text{Acc}_{(i \to j)}\}$，选取 top $N$ 对组成 $\mathcal{D}_{topN}$。

   再在 $\mathcal{D}_{topN}$ 中遍历所有组合（$2^N - 1$ 种）找到最优组合 $P^*$。

   **复杂度降低**：从 $2^{C^2}$ 降至 $C^2 + 2^N - 1$（$N=10$ 时仅 $\sim$65,536 + 1,023 次推理）。

   **(2) 样本缩减**：仅使用 50 张图片作为搜索数据集。

   **(3) 特征存储**：预存编码器特征，每次推理只修改存储特征并送入解码器，避免重复编码。

   **设计动机**：单对替换的输出差异可以预测该对在组合中的贡献，先筛选后组合的策略在保持搜索有效性的同时大幅降低计算量。仅需推理无需反向传播，GPU 显存开销极低。

3. **通道替换的实现**

   给定替换对 $(i,j)$，特征变换为：
   $$X'_{d,c,w,h} = X_{d, f_{i \to j}(c), w,h}$$
   其中 $f_{i \to j}(\cdot)$ 是将第 $i$ 通道映射为第 $j$ 通道的映射函数。

   这不是随机洗牌（shuffle），而是有选择性地用有效通道替换冗余通道，过程完全确定。

### 损失函数 / 训练策略

搜索阶段使用与基线相同的 Dice + CE loss 进行输出评估。注意：搜索过程仅涉及模型推理，**不需要任何梯度计算和反向传播**。

实现细节：
- 搜索数据集：50 张随机采样图像
- $N = 10$（top-N 替换对）
- 基线微调对比实验使用 25 epochs、Adam 优化器、初始学习率 $10^{-4}$

## 实验关键数据

### 主实验

**对 SAM 各版本的无参数微调效果**（9 个数据集平均 mIoU）：

| 模型 | 骨干 | 参数量 | 基线 Avg | +Ours Avg | 提升 Δ |
|------|------|--------|----------|-----------|--------|
| SAM | ViT-B | 91M | 49.14 | 58.08 | **+8.94** |
| SAM | ViT-L | 308M | 56.15 | 67.61 | **+11.46** |
| SAM | ViT-H | 636M | 55.54 | 60.68 | +5.14 |
| SAM2 | Hiera-T | 39M | 57.29 | 65.63 | **+8.34** |
| SAM2 | Hiera-S | 46M | 61.04 | 68.69 | +7.65 |
| SAM2 | Hiera-B+ | 81M | 61.62 | 66.94 | +5.32 |
| SAM2 | Hiera-L | 224M | 67.77 | 73.53 | +5.76 |

不更新任何参数即提升 5-11 个 mIoU 点，效果惊人。

**叠加已有微调方法后的效果**：

| 微调方法 | 基线 Avg | +Ours Avg | 额外提升 |
|----------|----------|-----------|----------|
| Decoder-only | 73.61 | 74.62 | +1.01 |
| SAMed (LoRA) | 78.56 | 79.72 | **+1.16** |
| SAM-COBOT | 78.73 | 79.32 | +0.59 |
| SAM-Adapter | 72.89 | 73.80 | +0.91 |
| SAM-PARSER | 60.96 | 65.39 | **+4.43** |
| DoRA | 79.12 | 79.92 | +0.80 |

说明即使经过参数微调，模型中仍存在通道冗余，本方法可作为即插即用模块进一步提升。

### 消融实验

**计算开销对比**：

| 方法 | GPU 显存 (GB) | 可训练参数 (K) |
|------|--------------|---------------|
| Encoder-only | 34.6 | 89,670 |
| Decoder-only | 13.7 | 4,057 |
| MedSAM | 34.7 | 93,735 |
| SAMed (LoRA) | 28.9 | 147 |
| SAM-PARSER | 15.9 | 0.5 |
| **Ours** | **11.1** | **0** |

GPU 显存使用最低（11.1 GB vs 其他方法 13.7-34.7 GB），参数量为零。

**替换对数量的影响**：增加替换对数量通常提升性能，COCO 数据集上 6 对时性能达峰。

**扩展到其他视觉任务**：

| 模型 | 骨干 | NYUv2 MSE↓ / AbsRel↓ / δ₁↑ | CIFAR Acc↑ |
|------|------|---------------------------|-----------|
| DINOv2 | ViT-S | 0.225/0.126/0.893 | 80.41 |
| +Ours | ViT-S | **0.209/0.112/0.907** | **80.81** |
| DINOv2 | ViT-B | 0.210/0.110/0.900 | 88.08 |
| +Ours | ViT-B | **0.193/0.095/0.916** | **88.49** |

在深度估计和图像分类上同样有效。

### 关键发现

- 有效通道的特征图呈现更清晰的结构、边缘和纹理；冗余通道则模糊、嘈杂（见 Figure 5 可视化）
- 某些通道具有跨域一致性：如 Channel 19 在自然/医疗/伪装场景中都有效，Channel 20/98/162/226 都冗余
- 较大模型（ViT-H、Hiera-L）的提升幅度略小，可能是因为更大的模型冗余程度相对较低
- 对域内数据集（自然图像）的提升大于域外数据集（医疗图像），与 SAM 的训练数据分布一致

## 亮点与洞察

1. **颠覆性的范式创新**：首次证明 VFM 可以完全无参数微调——不需要梯度、不需要反向传播、不需要任何新增参数，仅通过"换通道"就能显著提升下游性能
2. **极低的计算门槛**：仅需 11.1 GB 显存和模型推理，实际可在消费级 GPU 上运行，大幅降低了 VFM 适配的门槛
3. **与 PEFT 方法的正交互补**：可作为即插即用后处理步骤，在已微调模型上额外提升 0.5-4.4 个 mIoU 点
4. **通道冗余的深层洞察**：揭示了基础模型中普遍存在的特征冗余现象，为理解大模型的特征利用效率提供了新视角
5. **跨任务泛化**：从分割扩展到深度估计和分类，从 SAM 扩展到 DINOv2，验证了方法的通用性

## 局限与展望

- 搜索过程仍需遍历 $C^2$（~65536）对和 $2^N - 1$ 种组合，虽然仅需推理但在大规模数据上仍有时间成本
- 搜索数据集的选取可能影响最优替换对，50 张图片的代表性对不同数据集可能不足
- $N=10$ 的选择较为固定，未探索自适应确定 N 的方法
- 通道替换是"硬替换"，未探索更柔和的通道权重调整方案
- 仅在编码器最后一层的特征上操作，未探索多层通道替换的可能性
- 目前仅验证了 SAM/SAM2/DINOv2，对 CLIP、MAE 等其他 VFM 的泛化性未知

## 相关工作与启发

- **SAM-PARSER (2024)**：将可训练参数压缩到仅 512 个，本文更进一步到零参数
- **ShuffleNet**：通道洗牌用于训练时跨组信息融合，与本文目标和方法本质不同
- **Channel-Exchanging Network**：多模态融合中的通道交换，本文是单模态内的冗余消除
- **Network Pruning**：剪枝去除冗余，但通常需要重训练，本文不需要
- **启发**：基础模型的特征冗余是普遍现象，"减法"（去冗余）有时比"加法"（加参数）更有效；该思路可推广到 NLP 大模型的适配

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （零参数微调范式在 VFM 领域首次提出，思路大胆且有效）
- 实验充分度: ⭐⭐⭐⭐⭐ （9 数据集 × 7 骨干 × 6 微调方法组合 + 深度/分类扩展）
- 写作质量: ⭐⭐⭐⭐ （动机清晰，实验设计周全，可视化分析到位）
- 价值: ⭐⭐⭐⭐⭐ （实用性极强，计算门槛极低，与现有方法正交互补）

<!-- RELATED:START -->

## 相关论文

- [VGGT-DP: Generalizable Robot Control via Vision Foundation Models](vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)
- [Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](adapt-as-you-walk_through_the_clouds_training-free_online_te.md)
- [On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](../../NeurIPS2025/3d_vision/on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](../../CVPR2025/3d_vision/fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [GIQ: Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra](../../ICLR2026/3d_vision/giq_benchmarking_3d_geometric_reasoning_of_vision_foundation_models_with_simulat.md)

<!-- RELATED:END -->
