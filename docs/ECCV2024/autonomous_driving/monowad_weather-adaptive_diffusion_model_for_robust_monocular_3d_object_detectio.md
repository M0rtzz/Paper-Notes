---
title: >-
  [论文解读] MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection
description: >-
  [ECCV 2024][自动驾驶][单目3D检测] 提出 MonoWAD，通过天气码本学习晴天知识作为参考，结合天气自适应扩散模型将雾效建模为噪声进行特征增强，实现在各种天气条件下鲁棒的单目3D目标检测。
tags:
  - ECCV 2024
  - 自动驾驶
  - 单目3D检测
  - 天气自适应
  - 扩散模型
  - 天气码本
  - 雾天检测
---

# MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection

**会议**: ECCV 2024  
**arXiv**: [2407.16448](https://arxiv.org/abs/2407.16448)  
**代码**: [有](https://github.com/VisualAIKHU/MonoWAD)  
**领域**: 自动驾驶  
**关键词**: 单目3D检测, 天气自适应, 扩散模型, 天气码本, 雾天检测

## 一句话总结

提出 MonoWAD，通过天气码本学习晴天知识作为参考，结合天气自适应扩散模型将雾效建模为噪声进行特征增强，实现在各种天气条件下鲁棒的单目3D目标检测。

## 研究背景与动机

单目3D目标检测仅依赖单张RGB图像估计3D边界框，因其成本低廉而被广泛用于自动驾驶。然而，现有方法**几乎只关注理想天气（晴天）下的检测性能**，当面对雾天等恶劣天气时，性能急剧下降。

实验数据直观说明了这一问题的严重性：在作者的复现实验中，多个SOTA方法在雾天KITTI上的AP3D几乎归零：
- MonoGround: 晴天 Easy 25.24 → 雾天 **0.00**
- DID-M3D: 晴天 Easy 22.98 → 雾天 **1.15**
- MonoDETR: 晴天 Easy 28.84 → 雾天 **7.40**

这一性能悬崖式下降的根本原因在于：雾的**密集散射和光吸收**严重退化了视觉信息，而单目检测完全依赖视觉特征，没有LiDAR等深度传感器的补充。

作者识别出两个关键问题：**(1) 如何量化输入图像需要多大程度的改善？** 晴天只需少量增强，雾天需要大量增强；**(2) 如何引导特征表示的增强方向？** 增强需要有方向性——朝着"清晰天气"的方向增强。

现有去雾方法（image-level或feature-level dehazing）虽然在雾天有一定改善，但**一刀切地去雾会损害晴天性能**，因为它们无法根据天气条件动态调整增强力度。

## 方法详解

### 整体框架

MonoWAD 包含三个核心组件：
1. 骨干网络（DLA-102）提取输入特征 $x^c$ 或 $x^f$
2. **天气码本(Weather Codebook)** $\mathcal{Z}$：记忆晴天知识，为任意输入生成天气参考特征 $x^r$
3. **天气自适应扩散模型(Weather-Adaptive Diffusion)**：基于 $x^r$ 动态增强输入特征
最后由检测模块（transformer encoder-decoder）输出3D检测结果。

### 关键设计

1. **天气码本(Weather Codebook)**:

    - 可学习码本 $\mathcal{Z} = \{z_k\}_{k=1}^K$，$K=4096$ 个槽位，每个 $z_k \in \mathbb{R}^{1 \times c}$（$c=256$）
    - 输入特征通过卷积层得到 $\hat{x}^c$ 或 $\hat{x}^f$，对每个空间位置找最近码字进行量化：$x^{r(c)} = \mathbf{q}(\hat{x}^c) := \arg\min_{z_k \in \mathcal{Z}} \|\hat{x}^c_{ij} - z_k\|$
    - **Clear Knowledge Embedding (CKE) Loss**：用KL散度使码本输出接近晴天特征的通道分布——$\mathcal{L}_{cke} = D_{KL}(s^c \| s^{r(c)})$，其中 $s^c, s^{r(c)}$ 分别是GAP+softmax后的通道概率
    - **Weather-Invariant Guiding (WIG) Loss**：确保雾天和晴天输入产生相同的参考特征——$\mathcal{L}_{wig} = \|x^{r(c)} - x^{r(f)}\|_2^2$
    - 总CKR损失：$\mathcal{L}_{ckr} = \mathcal{L}_{cke} + \mathcal{L}_{wig}$
    - 设计动机：码本作为"天气记忆库"，训练时记住晴天的视觉知识模式；推理时无论输入是晴天还是雾天，都能回忆出相同的"晴天参考"，这个参考特征隐含了"需要增强多少"的信息

2. **天气自适应扩散模型**:

    - **核心创新：用雾分布代替高斯噪声**。定义雾分布 $\mathcal{F} = x^f - x^c$（雾天与晴天特征之差），作为扩散模型的噪声
    - 前向过程：逐步向晴天特征 $x^c_0$ 添加雾噪声——$q(x^c_t | x^c_{t-1}) = \mathcal{F}(x^c_t; \sqrt{1-\beta_t}x^c_{t-1}, \beta_t \mathbf{I})$
    - 反向过程：条件自编码器 $\boldsymbol{\epsilon}_\theta(x^c_t, t, x^r)$ 估计雾变量，**以天气参考特征 $x^r$ 作为条件**
    - 通过**交叉注意力机制**融合当前特征和参考特征：
    $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$
      其中 $Q = W_i^q \cdot \bar{x}^c_t$，$K = W_i^k \cdot \bar{x}^r$，$V = W_i^v \cdot \bar{x}^r$
    - 设计动机：传统扩散用高斯噪声，不理解天气语义；将雾效果建模为噪声，使扩散的前向/反向过程直接对应加雾/去雾过程。天气参考特征作为注意力的Key/Value，自动控制增强力度——晴天输入与参考差异小，增强少；雾天与参考差异大，增强多

3. **天气自适应增强损失(WAE Loss)**:

    - $\mathcal{L}_{wae} = \mathbb{E}_{x^c, \boldsymbol{\epsilon}_n \sim \mathcal{F}, t}[\|\boldsymbol{\epsilon}_n - \boldsymbol{\epsilon}_\theta(x^c_t, t, x^r)\|_2^2]$
    - 确保扩散模型准确估计雾变量
    - 设计动机：标准DDPM的MSE目标，但噪声分布替换为雾分布

### 损失函数 / 训练策略

- 总损失：$\mathcal{L}_{Total} = \mathcal{L}_{OD} + \lambda_1 \mathcal{L}_{ckr} + \lambda_2 \mathcal{L}_{wae}$
- $\lambda_1 = \lambda_2 = 1$，$\mathcal{L}_{OD}$ 为标准3D检测损失（分类+回归+深度）
- 训练数据：KITTI原始清晰图像 + 合成的foggy KITTI配对图像
- 雾图像基于DORN深度估计 + 物体距离合成
- 扩散步数 $T=15$，单卡RTX 4090训练120 epochs

## 实验关键数据

### 主实验（KITTI验证集，Car类别）

| 方法 | 雾天 AP3D Easy | 雾天 Mod. | 雾天 Hard | 晴天 AP3D Easy | 晴天 Mod. | 平均 Mod. |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| GUPNet | 2.74 | 2.19 | 2.16 | 22.76 | 16.46 | 9.33 |
| DID-M3D | 1.15 | 0.61 | 0.64 | 22.98 | 16.12 | 8.37 |
| MonoGround | 0.00 | 0.00 | 0.06 | 25.24 | 18.69 | 9.35 |
| MonoDTR | 16.89 | 11.86 | 9.87 | 24.52 | 18.57 | 15.22 |
| MonoDETR | 7.40 | 5.74 | 4.53 | 28.84 | 20.61 | 13.18 |
| **MonoWAD** | **27.17** | **19.57** | **16.21** | **29.10** | **21.08** | **20.33** |

Virtual KITTI多天气结果：

| 方法 | 雾天 AP3D Easy | 雨天 Easy | 日落 Easy |
|------|:---:|:---:|:---:|
| MonoDTR | 8.79 | 11.73 | 9.86 |
| MonoDETR | 4.50 | 6.61 | 7.08 |
| **MonoWAD** | **13.33** | **14.12** | **13.38** |

### 消融实验

各模块贡献：

| 配置 | 雾天 AP3D Easy | 雾天 Mod. | 晴天 Easy | 晴天 Mod. | 说明 |
|------|:---:|:---:|:---:|:---:|------|
| Baseline | 13.75 | 9.61 | 22.63 | 17.16 | 无任何增强 |
| +WAD | 25.62 | 18.66 | 26.34 | 19.17 | 仅扩散模型 |
| **+WC+WAD** | **27.17** | **19.57** | **29.10** | **21.08** | 完整MonoWAD |

扩散步数T的影响：

| T | 雾天 AP3D Easy | 雾天 Mod. | 晴天 Easy | 晴天 Mod. |
|---|:---:|:---:|:---:|:---:|
| 无 | 13.75 | 9.61 | 22.63 | 17.16 |
| 5 | 23.57 | 17.91 | 26.03 | 19.21 |
| 10 | 25.28 | 18.49 | 26.79 | 19.90 |
| **15** | **27.17** | **19.57** | **29.10** | **21.08** |
| 20 | 24.54 | 18.29 | 24.85 | 18.54 |

与去雾方法对比（基于MonoDTR）：

| 方法 | 雾天 Mod. | 晴天 Mod. | 平均 Mod. |
|------|:---:|:---:|:---:|
| MonoDTR | 11.86 | 18.57 | 15.22 |
| +RIDCP(图像去雾) | 12.41 | 17.89 | 15.15 |
| +DENet(特征去雾) | 17.44 | 5.70 | 11.57 |
| **MonoWAD** | **19.57** | **21.08** | **20.33** |

### 关键发现

- MonoWAD在雾天性能飙升的同时，**晴天性能也有提升**（29.10 vs baseline 22.63），这是传统去雾方法做不到的
- 去雾方法（DENet）虽然雾天提升17.44，但晴天暴跌到5.70——说明去雾操作对清晰图像有害
- 扩散步数T=15最优，T=20时出现过拟合退化
- 天气码本贡献显著：在WAD基础上加入WC，雾天AP3D Easy从25.62提升到27.17
- t-SNE可视化显示MonoWAD的晴天/雾天特征几乎重合，而其他方法呈明显分离

## 亮点与洞察

- **将雾效作为扩散噪声的思路非常巧妙**：传统扩散用高斯噪声是无语义的，而雾分布$\mathcal{F} = x^f - x^c$天然携带天气退化信息，使得扩散过程直接学会"去雾"
- **天气码本的自适应机制**：通过WIG Loss确保不同天气输入产生相同的参考特征，使得扩散模型的cross-attention自然实现自适应增强——不需要显式判断天气类型
- **晴天+雾天双赢**：这是相比去雾方法最大的优势，因为增强力度由天气参考特征自动调节

## 局限与展望

- **推理速度慢**：T=15时144ms/image，远慢于MonoDETR的38ms/image，扩散的迭代特性是瓶颈
- **需要配对训练数据**：训练阶段需晴天-雾天配对图像来计算雾分布，限制了对其他天气的直接扩展
- **雾图像为合成数据**：基于深度估计合成雾天图像，与真实雾场景存在domain gap
- 未来方向：探索无需配对图像的训练方式、加速扩散推理（如DDIM）、扩展到更多天气类型的联合处理

## 相关工作与启发

- **MonoDTR** (CVPR 2022): 深度引导的transformer单目检测器，本文的主要对比基线
- **MonoDETR** (ICCV 2023): DETR风格的单目3D检测，晴天表现最佳但雾天下降严重
- **Foggy Cityscapes** (Sakaridis et al.): 雾天合成方法的先驱，本文的雾合成遵循类似协议
- **DDPM/Conditional Diffusion**: 本文的扩散模型在噪声定义和条件机制上做了天气感知的改造

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将天气变化建模为扩散噪声是一个有洞察力的创新，码本设计也很精巧
- **实验充分度**: ⭐⭐⭐⭐⭐ — KITTI验证/测试、Virtual KITTI多天气、与去雾方法对比、多种扩散模型对比、步数消融、t-SNE可视化，极为完善
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清楚，方法motivation阐述充分
- **价值**: ⭐⭐⭐⭐ — 直击自动驾驶的实际痛点，但推理速度和配对数据需求限制了即时部署

<!-- RELATED:START -->

## 相关论文

- [GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)
- [SimPB: A Single Model for 2D and 3D Object Detection from Multiple Cameras](simpb_a_single_model_for_2d_and_3d_object_detection_from_multiple_cameras.md)
- [Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](../../ICCV2025/autonomous_driving/adaptive_dual_uncertainty_optimization_boosting_monocular_3d.md)
- [Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)
- [OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection](open_object-wise_position_embedding_for_multi-view_3d_object_detection.md)

<!-- RELATED:END -->
