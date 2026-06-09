---
title: >-
  [论文解读] Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes
description: >-
  [CVPR 2025][3D视觉][网格显著性] 本文提出Mesh Mamba，首个基于状态空间模型（SSM）的统一网格显著性预测模型，通过纹理对齐、子图嵌入和双向SSM，实现对有纹理和无纹理3D网格的高质量视觉注意力预测，并构建了首个系统对比有/无纹理条件下显著性差异的数据集。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "网格显著性"
  - "状态空间模型"
  - "Mamba"
  - "纹理对齐"
  - "子图嵌入"
---

# Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes

**会议**: CVPR 2025  
**arXiv**: [2504.01466](https://arxiv.org/abs/2504.01466)  
**代码**: [https://github.com/kaviezhang/MeshMamba](https://github.com/kaviezhang/MeshMamba)  
**领域**: 3D视觉  
**关键词**: 网格显著性, 状态空间模型, Mamba, 纹理对齐, 子图嵌入

## 一句话总结
本文提出Mesh Mamba，首个基于状态空间模型（SSM）的统一网格显著性预测模型，通过纹理对齐、子图嵌入和双向SSM，实现对有纹理和无纹理3D网格的高质量视觉注意力预测，并构建了首个系统对比有/无纹理条件下显著性差异的数据集。

## 研究背景与动机

1. **领域现状**：3D网格显著性预测旨在识别网格表面吸引视觉注意力的区域。现有方法主要针对无纹理网格，基于几何特征（曲率、法线等）进行预测。数据集方面，已有VR眼动追踪实验收集的显著性数据，但缺乏纹理条件的系统研究。

2. **现有痛点**：(1) 现有数据集要么只关注无纹理网格，要么只考虑简单顶点颜色，缺乏对相同模型在有/无纹理条件下显著性差异的系统研究；(2) 现有方法无法同时处理有纹理和无纹理网格；(3) 点云和网格方法难以建模全局上下文。

3. **核心矛盾**：几何结构和纹理信息共同影响视觉注意力分布，但两者的交互作用尚不清楚，且缺乏统一框架同时利用这两类特征。

4. **本文目标** (1) 构建对比数据集；(2) 设计统一模型同时处理两类网格；(3) 有效建模局部和全局特征。

5. **切入角度**：利用Mamba/SSM的线性复杂度和全局建模能力，结合图卷积处理局部拓扑关系，通过子图嵌入保持网格拓扑结构。

6. **核心 idea**：用子图嵌入保持拓扑、双向SSM实现全局上下文建模，统一解决有/无纹理网格的显著性预测。

## 方法详解

### 整体框架
输入为3D三角网格（可选纹理图像），输出为每个三角面的显著性值。模型由三部分组成：图卷积编码器（纹理对齐+几何特征提取+图卷积）→ 子图嵌入 → Mamba Block（双向SSM+特征扩散聚合）→ 特征传播（投票插值上采样）。

### 关键设计

1. **纹理对齐模块（Texture Alignment）**:

    - 功能：将2D纹理图像特征精确映射到3D网格表面
    - 核心思路：利用UV映射将三角面对应到纹理图像的像素位置，然后通过隐式表示（latent code map）在高维特征域实现对齐。对每个三角面均匀采样其UV范围内的纹理特征，采用双线性隐式插值处理亚像素位置以防止特征不连续。每个三角面根据其长宽比调整采样范围，确保不失真的感受野。
    - 设计动机：直接使用顶点颜色（vertex color）表示纹理信息过于粗糙，无法捕获复杂纹理的视觉细节。通过连续隐式表示实现精确的纹理-几何对齐。

2. **子图嵌入（Subgraph Embedding）**:

    - 功能：将网格分割为保持拓扑的局部patch，作为SSM的输入token序列
    - 核心思路：先用最远点采样（FPS）选择 $L$ 个中心面，然后对每个中心面使用随机游走采样（RWS）采集长度为 $M$ 的子图，生成保持连通性的局部patch。与KNN聚类不同，RWS沿网格边游走采样，天然保持邻接关系，其随机性还增强了模型的抗噪声鲁棒性。
    - 设计动机：传统patch embedding（如KNN聚类）会破坏网格的拓扑连接关系。网格不同于图像的规则网格结构，不能简单地切割为矩形patch。子图嵌入保持了固有的位置关系和几何完整性。

3. **双向SSM的Mamba Block**:

    - 功能：对子图patch token序列进行全局上下文建模
    - 核心思路：引入可学习的 $[cls]$ token和位置编码扩展序列，然后通过特征扩散和聚合操作扩展局部特征的感受野——为每个token生成 $l$ 个伪邻接面进行特征传播。关键是使用双向SSM：$z_t = SSM_+(f(z_{t-1})) + SSM_-(f(z_{t-1})) + f(z_{t-1})$，同时考虑正向和反向的序列上下文，克服了单向SSM在方向偏差上的不足。
    - 设计动机：单向SSM只能从一个方向处理序列，而网格上的patch token没有天然的线性顺序。双向处理确保每个token都能融合全局上下文信息，增强对整体结构和内容的理解。

### 损失函数 / 训练策略
使用L1损失函数训练，AdamW优化器，初始学习率1e-3，每50个epoch衰减0.1倍，共训练150个epoch。数据集80%训练、20%测试。

## 实验关键数据

### 主实验

无纹理网格显著性预测（仅几何特征）：

| 方法 | CC↑ | SIM↑ | KLD↓ | SE↓ |
|------|-----|------|------|-----|
| PointTrans | 0.5114 | 0.6861 | 0.3475 | 0.0314 |
| MeshNet | 0.5423 | 0.7000 | 0.3390 | 0.0309 |
| Mamba3D | 0.5993 | 0.7088 | 0.3345 | 0.0285 |
| **Ours** | **0.6140** | **0.7171** | **0.3067** | **0.0284** |

有纹理网格显著性预测（几何+纹理）：

| 方法 | CC↑ | SIM↑ | KLD↓ | SE↓ |
|------|-----|------|------|-----|
| PointTrans | 0.5201 | 0.6817 | 0.3578 | 0.0297 |
| MeshNet | 0.5605 | 0.7002 | 0.3371 | 0.0286 |
| Mamba3D | 0.5013 | 0.6769 | 0.3655 | 0.0372 |
| **Ours** | **0.6305** | **0.7232** | **0.2888** | **0.0265** |

有纹理条件下提升尤为显著：相比Mamba3D，CC提升25.8%，KLD降低21.0%。

### 消融实验

| 配置 | CC↑ | SIM↑ | KLD↓ | SE↓ |
|------|-----|------|------|-----|
| Full model | 0.6305 | 0.7232 | 0.2888 | 0.0265 |
| w/o Texture | 0.6066 | 0.7113 | 0.3134 | 0.0267 |
| w/o Shape | 0.5403 | 0.6903 | 0.3634 | 0.0324 |
| w/o Subgraph (用KNN) | 0.6237 | 0.7208 | 0.3048 | 0.0298 |
| w/o Feature D&A | 0.5889 | 0.7106 | 0.3123 | 0.0294 |
| w/o SSM- | 0.6203 | 0.7186 | 0.2935 | 0.0272 |
| w/o SSM+ | 0.6199 | 0.7164 | 0.2947 | 0.0275 |
| w/ Backbone-T (Transformer) | 0.6113 | 0.7204 | 0.2975 | 0.0270 |

### 关键发现
- **形状特征贡献最大**：去掉三角面形状特征（w/o Shape）CC下降14.3%（0.6305→0.5403），是所有消融中影响最大的
- **纹理对有纹理网格至关重要**：加入纹理后CC从0.6066提升到0.6305；但对无纹理网格反而有害
- **子图嵌入优于KNN**：子图嵌入比KNN聚类在KLD上好5.2%，验证了保持拓扑的重要性
- **双向SSM缺一不可**：去掉任一方向都会降低性能，正向和反向SSM贡献大致相当
- **计算复杂度线性增长**：FLOPs随子图数和子图长度线性增长，具有良好的可扩展性

## 亮点与洞察
- **数据集贡献**：首个系统对比同一3D模型在有/无纹理条件下显著性差异的数据集，60名参与者的VR眼动实验，方法论扎实。这为研究纹理对视觉注意力的影响提供了宝贵资源。
- **子图嵌入设计**：用随机游走替代KNN来分割网格patch是非常聪明的做法，自然保持了拓扑信息。这个思路可以推广到任何需要在非欧几何数据上做patch化的任务。
- **纹理-几何交互发现**：实验发现纹理信息对无纹理网格显著性预测无益甚至有害，但对有纹理网格显著改善——说明模型确实学到了条件依赖的特征表示。

## 局限与展望
- 数据集规模有限，Free3D资产库的多样性可能不足以反映所有场景
- 仅考虑静态网格的显著性，未涉及动态/交互场景下的视觉注意力
- 随机游走采样的子图可能难以覆盖网格的所有关键区域，FPS可能遗漏细节丰富的小区域
- 论文未讨论模型在不同分辨率网格上的表现和泛化能力
- 显著性驱动的网格简化应用（Sec 5.4）只是初步展示，可进一步深入

## 相关工作与启发
- **vs Mamba3D**: Mamba3D直接在点云上应用SSM，本文在网格的子图上应用，保持了拓扑信息，在有纹理网格上优势明显（CC 0.6305 vs 0.5013）
- **vs SAL3D**: SAL3D基于PointNet2，仅处理无纹理网格。本文统一框架处理两种网格类型，且性能全面领先
- **vs DiffusionNet**: DiffusionNet使用热核签名和扩散算子处理网格，但在显著性预测任务上表现不佳，可能因为缺少全局上下文建模

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将Mamba/SSM应用于网格显著性预测，子图嵌入设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 自建数据集+公开数据集双验证，16个baseline对比，详细消融，交叉验证设计
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但部分公式符号较密集
- 价值: ⭐⭐⭐⭐ 数据集和方法论贡献都很实在，对3D视觉感知研究有直接推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](../../NeurIPS2025/3d_vision/mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[CVPR 2025\] Textured Gaussians for Enhanced 3D Scene Appearance Modeling](textured_gaussians_for_enhanced_3d_scene_appearance_modeling.md)
- [\[ICCV 2025\] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](../../ICCV2025/3d_vision/ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)
- [\[CVPR 2025\] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter](pma_towards_parameter-efficient_point_cloud_understanding_via_point_mamba_adapte.md)
- [\[ICCV 2025\] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction](../../ICCV2025/3d_vision/meshmamba_state_space_models_for_articulated_3d_mesh_generation_and_reconstructi.md)

</div>

<!-- RELATED:END -->
