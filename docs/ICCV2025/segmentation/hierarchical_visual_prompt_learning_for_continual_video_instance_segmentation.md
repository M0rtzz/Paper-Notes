---
title: >-
  [论文解读] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation
description: >-
  [ICCV 2025][语义分割][持续学习] 提出持续视频实例分割（CVIS）新问题，设计分层视觉提示学习（HVPL）模型，通过帧级和视频级两个层面的遗忘补偿机制，有效缓解旧类别的灾难性遗忘。 视频实例分割（VIS）需要在连续帧中同时检测、分割和跟踪物体实例。然而，现有方法假设类别集合固定不变，这与现实场景不符——新类别会…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "持续学习"
  - "视频实例分割"
  - "视觉提示学习"
  - "灾难性遗忘"
  - "正交梯度校正"
---

# Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation

**会议**: ICCV 2025  
**arXiv**: [2508.08612](https://arxiv.org/abs/2508.08612)  
**代码**: [GitHub](https://github.com/JiahuaDong/HVPL)  
**领域**: 图像分割  
**关键词**: 持续学习, 视频实例分割, 视觉提示学习, 灾难性遗忘, 正交梯度校正

## 一句话总结

提出持续视频实例分割（CVIS）新问题，设计分层视觉提示学习（HVPL）模型，通过帧级和视频级两个层面的遗忘补偿机制，有效缓解旧类别的灾难性遗忘。

## 研究背景与动机

视频实例分割（VIS）需要在连续帧中同时检测、分割和跟踪物体实例。然而，现有方法假设类别集合固定不变，这与现实场景不符——新类别会持续出现。如果直接在新类别上微调，模型会严重遗忘已学过的旧类别（即灾难性遗忘）。

**现有方法的不足**：
1. 传统VIS方法需要保留所有旧类别的视频数据进行重训练，计算成本和内存需求极高
2. 简单将持续学习（CL）与VIS结合仅能在帧级别缓解遗忘，无法利用全局视频上下文来解决视频级别的遗忘问题
3. 知识蒸馏方法在CVIS中因实例间外观变化大而效果不佳，多任务递增进一步加剧了蒸馏的难度

**核心动机**：需要一种无需存储旧任务训练数据（rehearsal-free）的方法，同时从帧级和视频级两个维度解决灾难性遗忘。

## 方法详解

### 整体框架

HVPL以预训练的Mask2Former作为帧级检测器（参数冻结），在其基础上引入两个层级的提示学习机制：
- **帧级**：任务特定帧提示 + 正交梯度校正（OGC）模块
- **视频级**：任务特定视频提示 + 视频上下文解码器（含GSS层和MSA层）

### 关键设计

1. **任务特定帧提示 + 正交梯度校正（OGC）**：

    - 为每个新任务 $t$ 引入可学习帧提示 $\mathbf{P}_{\mathrm{frm}}^{t} \in \mathbb{R}^{L_p^f \times D}$，编码新类别的全局实例信息
    - OGC的核心思想：更新帧提示的梯度 $\triangle\mathbf{P}$ 应满足 $\triangle\mathbf{P}(\mathcal{O}^{t-1})^\top = \mathbf{0}$，即梯度投影到旧任务特征空间的正交补空间
    - 通过对旧任务代表性特征空间 $\mathcal{O}^{t-1}$ 做SVD分解得到正交基 $\mathbf{V}_0^{t-1}$，将梯度投影为 $\triangle\mathbf{P}^* = \triangle\mathbf{P}\hat{\mathbf{V}}_0^{t-1}(\hat{\mathbf{V}}_0^{t-1})^\top$
    - 弹性阈值 $\xi \in [0,1]$ 控制正交空间大小，平衡新任务学习和旧任务保留（实验最优 $\xi=0.7$）
    - 设计动机：确保新任务学习不会干扰旧类别特征空间中的注意力计算，从数学上保证帧级遗忘的最小化

2. **图引导状态空间模块（GT-SSM）**：

    - 位于视频上下文解码器的GSS层中，用于捕获跨帧的结构化类间关系
    - 构建无向 $\varphi$-连接图 $\mathcal{G}$，通过最小生成树 $\mathcal{G}_T$ 消除噪声边
    - 基于状态空间模型的离散化变量 $\bar{\mathbf{A}}_j = \exp(\triangle_j \mathbf{A}_j)$，通过图遍历计算隐藏状态
    - 提出动态遍历策略（Algorithm 1），将复杂度从 $\mathcal{O}(N_v^2)$ 降至 $\mathcal{O}(N_v)$
    - 设计动机：传统Mamba使用固定遍历路径，无法建模跨帧实例间的内在结构关系；GT-SSM通过图结构自适应遍历，捕获语义相关实例间的信息传播

3. **任务特定视频提示 + MSA层**：

    - 可学习视频提示 $\mathbf{P}_{\mathrm{vid}}^{t} \in \mathbb{R}^{L_p^v \times D}$ 编码视频级全局上下文
    - MSA层以视频提示为Query，GSS层输出的增强特征为Key/Value，通过多头自注意力捕获任务特定的全局视频上下文
    - 输出的视频提示特征 $\mathbf{F}_{\mathrm{vid}}^{L_m}$ 整合了跨帧结构关系和全局语义信息
    - 设计动机：帧级补偿忽略了不同帧中实例之间的内在关系，缺乏全局语义上下文使得视频级遗忘难以解决

### 损失函数 / 训练策略

- 采用Mask2Former的标准目标损失
- 训练时冻结Mask2Former的骨干网络、像素解码器和Transformer解码器参数
- 首次任务使用帧/视频提示长度 $L_p^f = L_p^v = 100$，后续任务使用 $L_p^f = L_p^v = 10$
- 测试时拼接所有任务的帧提示和视频提示进行联合推理
- 旧任务特征空间 $\mathcal{O}^{t-1}$ 的存储不超过0.1M，可忽略不计

## 实验关键数据

### 主实验

| 数据集/设置 | 指标 | HVPL | ECLIPSE (CVPR'24) | CoMFormer+NeST | 提升 |
|---|---|---|---|---|---|
| OVIS 15-5 | AP | **11.09** | 6.62 | 4.44 | +4.47 |
| OVIS 15-5 | FAP↓ | **9.56** | 28.08 | 32.25 | -18.52 |
| OVIS 15-10 | AP | **11.92** | 6.82 | 5.87 | +5.10 |
| YT-VIS 2021 30-10 | AP | **43.24** | 29.68 | 28.70 | +13.56 |
| YT-VIS 2021 20-4 | AP | **35.05** | 30.52 | 15.37 | +4.53 |
| YT-VIS 2019 20-5 | AP | **34.79** | 32.62 | 21.39 | +2.17 |
| YT-VIS 2019 20-2 | AP | **31.68** | 30.24 | 10.29 | +1.44 |

### 消融实验

| 配置 | AP (30-10) | FAP↓ | 说明 |
|---|---|---|---|
| Base（仅微调Mask2Former） | 27.53 | 96.48 | 无任何抗遗忘机制 |
| +TFP（帧提示） | 30.02 | 74.88 | 帧提示缓解部分遗忘 |
| +TFP+TVP（+视频提示+MSA） | 39.07 | 31.05 | 视频级补偿大幅提升 |
| +TFP+TVP+GSS | 40.37 | 30.40 | GSS层进一步建模跨帧关系 |
| Full（+OGC） | **43.24** | **15.69** | OGC正交梯度校正最终完成 |

### 关键发现

- HVPL可训练参数仅0.92M，远少于知识蒸馏方法的41.92M，但性能远优
- 遗忘率FAP和FAR1大幅降低，例如在OVIS 15-5上FAP仅9.56%（ECLIPSE为28.08%）
- 在多任务递增（如YT-VIS 2021 20-4共6个任务）中表现尤为突出，各增量步骤均领先
- $\xi=0.7$ 为平衡新旧任务学习的最优阈值，$\xi$ 在0.2-1.0范围内性能稳定

## 亮点与洞察

- **新问题定义**：首次形式化定义了CVIS问题，填补了持续学习在视频实例分割中的空白
- **分层遗忘补偿**：帧级（OGC正交投影）+ 视频级（GT-SSM + 视频提示）的双层设计很有系统性
- **数学推导严谨**：从注意力等价性出发推导梯度正交条件，理论保证强
- **GT-SSM**：将Mamba与图结构结合用于跨帧关系建模是一个新颖思路
- **极低参数量**：0.92M参数即可大幅超越42M参数的方法，参数效率极高

## 局限与展望

- 仅使用ResNet-50作为骨干网络，未验证在更强骨干（如Swin-L）上的效果
- CVIS设置中不考虑旧任务数据回放（rehearsal-free），实际应用中少量回放可能进一步提升
- GT-SSM中最小生成树的构建依赖余弦相似度，对于外观变化剧烈的场景可能不够鲁棒
- 未探讨与大语言模型结合处理大规模持续任务（作者在结论中提及为未来工作）

## 相关工作与启发

- 与ECLIPSE（CVPR'24）同为提示学习方法，但HVPL增加了视频级补偿，性能提升显著
- 正交梯度投影思想源自持续学习中的OGD系列方法，用到提示学习是创新
- GT-SSM融合了Vision Mamba和图神经网络，可启发其他需要跨帧结构建模的任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次提出CVIS问题，分层提示学习框架设计新颖
- **实验充分度**: ⭐⭐⭐⭐ 三个数据集多种设置，消融充分，但缺乏更强骨干实验
- **写作质量**: ⭐⭐⭐⭐ 理论推导清晰，符号定义规范
- **价值**: ⭐⭐⭐⭐⭐ 对持续学习在视频分割中的应用具有开拓意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](cavis_context-aware_video_instance_segmentation.md)
- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[ICCV 2025\] Region-based Cluster Discrimination for Visual Representation Learning](region-based_cluster_discrimination_for_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
