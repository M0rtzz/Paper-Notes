---
title: >-
  [论文解读] InfMAE: A Foundation Model in the Infrared Modality
description: >-
  [ECCV 2024][自监督学习][红外基础模型] 提出 InfMAE——首个红外模态基础模型，构建了 30 万张红外图像数据集 Inf30，设计信息感知掩码策略和多尺度编码器，在红外语义分割、目标检测和小目标检测三个下游任务上超越现有方法。
tags:
  - ECCV 2024
  - 自监督学习
  - 红外基础模型
  - 掩码自编码器
  - 信息感知掩码
  - 多尺度编码器
  - 自监督预训练
---

# InfMAE: A Foundation Model in the Infrared Modality

**会议**: ECCV 2024  
**arXiv**: [2402.00407](https://arxiv.org/abs/2402.00407)  
**代码**: 无  
**领域**: 自监督学习 / 红外视觉  
**关键词**: 红外基础模型, 掩码自编码器, 信息感知掩码, 多尺度编码器, 自监督预训练

## 一句话总结

提出 InfMAE——首个红外模态基础模型，构建了 30 万张红外图像数据集 Inf30，设计信息感知掩码策略和多尺度编码器，在红外语义分割、目标检测和小目标检测三个下游任务上超越现有方法。

## 研究背景与动机

红外成像利用物体热辐射成像，在低光照、黑暗、烟雾等恶劣条件下具有不可替代的优势，广泛应用于自动驾驶、安防监控、军事侦察等领域。然而，当前红外视觉社区面临两个核心问题：

**缺乏红外基础模型**：可见光模态已有 MAE、DINO、BEiT 等基础模型，视频模态有 VideoMAE，遥感有 Scale-MAE，点云有 Point-M2AE，但红外模态尚无专属基础模型。直接将可见光 MAE 迁移到红外任务效果不佳，原因在于两种模态存在显著差异。

**红外图像的固有特性**：
   - **信息量低**：红外图像缺乏丰富的纹理和颜色细节。作者计算信息熵发现，Inf30 平均信息熵为 6.44，而 ImageNet-1K 为 7.19。
   - **目标与背景同温**：斑马线、电线杆等物体因与周围环境温度相近，在红外图像中几乎不可见。
   - **直接用随机掩码（vanilla MAE）会导致大量掩码落在信息贫乏区域**，不利于表示学习。

**缺乏大规模红外数据集**：现有红外数据集规模小、场景单一，不足以支撑基础模型的预训练。

这三个问题共同驱动了 InfMAE 的设计：专属数据集 + 针对性掩码策略 + 适配的模型架构。

## 方法详解

### 整体框架

InfMAE 由三个核心模块组成：
1. **掩码块生成模块**：信息感知掩码策略，优先保留信息丰富区域
2. **多尺度编码器模块**：CNN + Transformer 混合架构，输出多尺度特征
3. **红外解码器模块**：融合多尺度表示进行图像重建

### 关键设计

1. **Inf30 数据集构建**

   从多个网站收集约 50 万张红外图像，经过两步预处理：
   - **去重**：选取锚图像，去除场景高度相似的冗余图像
   - **质量过滤**：移除宽高均小于 20 像素的低分辨率图像

   最终得到 **305,241 张红外图像**，涵盖天空、海洋、森林、城市、郊区等多种场景，包含船舶、车辆、行人、建筑等目标。分辨率范围从 40×23 到 6912×1024。

2. **信息感知掩码策略（Information-Aware Masking, IAM）**

   针对红外图像信息分布不均匀的特点，设计了基于信息量的自适应掩码策略，避免随机掩码过多遮挡信息贫乏区域：

   - 输入图像 $x \in \mathbb{R}^{H \times W \times C}$ 通过卷积（kernel=16, stride=16）得到强度特征图
   - 沿通道取均值得到灰度值图 $I \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16}}$
   - 将 $N$ 个 token 按灰度值**降序排列**（值越大信息越丰富）
   - 以采样步长 $S$ 等间隔采样：采样到的为可见 token，其余为掩码 token
   - 将掩码模板上采样 2 倍和 4 倍，生成 mask\_block1 和 mask\_block2 供多尺度编码器使用

   这种策略无需额外的语义感知分支（不同于 ATTMask、SemMAE），简单有效地确保信息密集区域被保留。

3. **多尺度编码器模块**

   受 MCMAE 和 ConvNeXt 启发，设计三层编码器：

   - **Encoder Layer 1**（CNN）：Patch Embedding 1 → Conv Attention + FFN → 特征 $F_1 \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times C_1}$，与 mask\_block1 相乘实现掩码
   - **Encoder Layer 2**（CNN）：Patch Embedding 2 → Conv Attention + FFN → 特征 $F_2 \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8} \times C_2}$，与 mask\_block2 相乘
   - **Encoder Layer 3**（ViT Transformer）：展平 + 选择可见 token → 自注意力 → 学习到的可见 token $T_s \in \mathbb{R}^{N_v \times C_3}$

   这种设计让低层 CNN 处理局部纹理（即使红外纹理较少），高层 Transformer 捕获全局语义，多尺度特征图 $F_1, F_2, F_3, F_4$ 可直接用于各种下游任务的特征金字塔。

4. **红外解码器模块**

   将多尺度可见 token $T_1, T_2, T_s$ 归一化到相同维度后相加，与掩码 token 拼接，恢复原始顺序后送入解码器重建图像。

   关键设计：**解码器深度仅为 2 层**，远少于标准 MAE 的 8 层。这是因为红外图像缺乏纹理和颜色细节，深解码器反而过拟合。

### 损失函数 / 训练策略

重建损失为掩码区域的 MSE：

$$\mathcal{L}_{mse} = \frac{1}{M} \sum_{i=1}^{M} (y_i - x_i)^2$$

预训练配置：
- 400 epoch，cosine 学习率 + 40 epoch warmup
- AdamW 优化器，base lr = $1.5 \times 10^{-4}$，weight decay = 0.05
- 批大小 256，4 × A100 GPU
- 编码器层数：2（CNN1）+ 2（CNN2）+ 11（ViT），768维，12头
- 默认采样步长 $S=4$

## 实验关键数据

### 主实验

**红外语义分割**（MSRS-inf 数据集，UperNet head）：

| 方法 | Backbone | mIoU(%) | mAcc(%) | vs.之前SOTA |
|------|----------|---------|---------|------------|
| DDRNet | - | 67.3 | 73.3 | - |
| Vanilla MAE | ViT-B | 71.4 | 78.2 | - |
| MCMAE | ViT-B | 72.1 | 79.8 | 之前最优 |
| **InfMAE** | **ViT-B** | **74.3** | **82.5** | **+2.2 mIoU** |

**红外目标检测**（M3FD-inf 数据集，Mask R-CNN head）：

| 方法 | Backbone | mAP(%) | AP50(%) |
|------|----------|--------|---------|
| Sparse R-CNN | ResNet50 | 48.3 | 79.4 |
| Vanilla MAE | ViT-B | 51.4 | 83.4 |
| MCMAE | ViT-B | 55.7 | 88.4 |
| **InfMAE** | **ViT-B** | **56.2** | **88.1** |

**红外小目标检测**（IRSTD-1k 数据集）：

| 方法 | Backbone | AUC(%) | F1(%) | IoU(%) |
|------|----------|--------|-------|--------|
| DNANet | - | 87.8 | 76.4 | 61.8 |
| MCMAE | ViT-B | 90.8 | 78.4 | 64.5 |
| **InfMAE** | **ViT-B** | **91.2** | **79.5** | **66.0** |

### 消融实验

**模块消融**（UperNet分割 + Mask R-CNN检测）：

| IAM | Multi-scale | Seg_mIoU(%) | Det_AP50(%) | 说明 |
|:---:|:---:|------|------|------|
| ✗ | ✗ | 71.4 | 81.5 | baseline (vanilla MAE) |
| ✓ | ✗ | 72.0 | 78.9 | 仅信息感知掩码 |
| ✗ | ✓ | 72.1 | 86.3 | 仅多尺度编码器 |
| ✓ | ✓ | **74.3** | **88.1** | 两者结合最佳 |

**解码器深度**：

| Decoder Depth | Seg_mIoU(%) | Det_AP50(%) | 说明 |
|:---:|------|------|------|
| **2** | **74.3** | **88.1** | 红外图像不需深解码器 |
| 4 | 72.9 | 87.9 | 性能下降 |
| 8 | 73.2 | 87.6 | 继续下降 |
| 12 | 74.0 | 87.2 | 过深反而不利 |

### 关键发现

- 信息感知掩码和多尺度编码器各自贡献约 0.6-0.7 mIoU 提升，组合后提升 2.9 mIoU
- 解码器深度 2 层最优，验证了红外图像"信息简单"的特性——深解码器不增反减
- 掩码步长 $S=4$ 平衡了信息保留和掩码比例
- 预训练数据量从 10 万到 30 万持续提升，但 50 万（未去重）反而下降——数据质量和多样性比规模更重要
- 400 epoch 预训练已接近 1600 epoch 的性能，收益递减

## 亮点与洞察

- **填补空白**：首个红外模态专属基础模型，从数据集到模型架构的完整方案
- **简单有效的掩码策略**：利用灰度值排序 + 等间隔采样，无需额外分支即可实现信息感知掩码
- **解码器深度洞察**：浅解码器对信息稀疏模态更优，这一发现可推广到其他低信息密度模态
- **多任务验证**：在分割、检测、小目标检测三种下游任务上均验证有效

## 局限与展望

- Inf30 数据集仅 30 万图像，相比 ImageNet-1K (128万) 仍较小
- 信息感知掩码基于灰度值简单排序，未考虑空间结构信息（如边缘、轮廓）
- 仅验证了 ViT-B 规模，未探索更大模型（ViT-L/H）的效果
- 未考虑红外-可见光跨模态场景，如融合预训练
- 数据集多样性受限于公开红外数据源，某些场景（如医疗红外、工业检测）可能覆盖不足

## 相关工作与启发

- 与 Scale-MAE（遥感）、VideoMAE（视频）类似，InfMAE 展示了 MAE 框架适配特定模态的范式：模态特性分析 → 定制掩码策略 → 适配编码/解码结构
- 信息熵分析为理解不同模态之间的差距提供了量化工具
- 多尺度 CNN+Transformer 编码器的设计思路（MCMAE启发）可用于其他需要多尺度特征的模态
- 浅解码器的发现暗示：重建目标的复杂度应匹配模态数据的内在复杂度

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个红外基础模型，信息感知掩码简洁有效
- **实验充分度**: ⭐⭐⭐⭐ — 3个下游任务，丰富消融（模块/深度/步长/epoch/数据量）
- **写作质量**: ⭐⭐⭐⭐ — 信息熵分析铺垫动机，整体逻辑清晰
- **价值**: ⭐⭐⭐⭐ — 为红外视觉社区提供了标准化的基础模型和数据集，具有基础设施价值

<!-- RELATED:START -->

## 相关论文

- [MarineInst: A Foundation Model for Marine Image Analysis with Instance Visual Description](marineinst_a_foundation_model_for_marine_image_analysis_with_instance_visual_des.md)
- [Griffin: Towards a Graph-Centric Relational Database Foundation Model](../../ICML2025/self_supervised/griffin_towards_a_graph-centric_relational_database_foundation_model.md)
- [MOMO: Mars Orbital Model — Foundation Model for Mars Orbital Applications](../../CVPR2026/self_supervised/momo_mars_orbital_model_foundation_model_for_mars_orbital_applications.md)
- [Spikingformer: A Key Foundation Model for Spiking Neural Networks](../../AAAI2026/self_supervised/spikingformer_a_key_foundation_model_for_spiking_neural_networks.md)
- [Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection](../../ICML2025/self_supervised/foundation_model_insights_and_a_multi-model_approach_for_superior_fine-grained_o.md)

<!-- RELATED:END -->
