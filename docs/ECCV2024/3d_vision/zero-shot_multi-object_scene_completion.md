---
title: >-
  [论文解读] Zero-Shot Multi-Object Scene Completion
description: >-
  [ECCV 2024][3D视觉] 提出OctMAE，一种融合Octree U-Net和隐空间3D MAE的混合架构，从单张RGB-D图像实现高质量近实时的多物体场景形状补全，通过遮挡掩码策略和3D旋转位置编码显著提升效率和泛化能力。
tags:
  - ECCV 2024
  - 3D视觉
---

# Zero-Shot Multi-Object Scene Completion

**会议**: ECCV 2024  
**arXiv**: [2403.14628](https://arxiv.org/abs/2403.14628)  
**代码**: [项目页](https://sh8.io/#/oct_mae)  
**领域**: 3D视觉

## 一句话总结

提出OctMAE，一种融合Octree U-Net和隐空间3D MAE的混合架构，从单张RGB-D图像实现高质量近实时的多物体场景形状补全，通过遮挡掩码策略和3D旋转位置编码显著提升效率和泛化能力。

## 研究背景与动机

- 现有单物体形状补全方法在复杂多物体真实场景中表现不佳
- 以物体为中心的方法需要类别特定的先验，受限于少数类别
- VoxFormer扩展MAE到3D但使用密集体素（内存受限于低分辨率）
- 缺乏大规模多类别3D场景补全数据集
- **核心问题**：如何实现跨大量物体类别的零样本多物体场景补全

## 方法详解

### 整体框架

1. 使用预训练ResNeXt-50提取2D图像特征，通过深度图反投影到3D获取特征+坐标
2. 将3D点特征转换为Octree表示（LoD-9, 512³分辨率）
3. Octree U-Net编码到LoD-5的隐空间
4. 隐空间3D MAE处理编码特征+遮挡掩码token
5. Octree U-Net解码器恢复到LoD-9，逐层预测占据、最终层预测SDF和法线

### 关键设计

**OctMAE架构（核心创新）**：
- Octree U-Net负责高效的局部特征编码/解码（LoD-9到LoD-5）
- 3D MAE在LoD-5的隐空间做全局推理（token数降至数百~数千量级）
- 结合了CNN的局部感知和Transformer的全局理解

**遮挡掩码策略**：
- 不在所有空体素放置mask token（密集掩码→内存爆炸），仅在被遮挡的体素处放置
- 通过深度测试确定哪些体素位于物体背后
- 使掩码token数量大幅减少，支持使用full attention而非deformable attention

**3D RoPE（旋转位置编码）**：
- 将三维坐标分别编码为旋转矩阵R(p^x), R(p^y), R(p^z)
- 组成块对角矩阵应用于每个attention层的query/key
- 比可学习相对位置编码高效（后者需N'×N'次计算），比绝对位置编码更泛化

**大规模数据集构建**：
- 从Objaverse筛选12K+ 3D模型（601类），补充GSO数据
- 使用BlenderProc物理摆放+真实光照渲染1M张图像
- 涵盖手持尺寸物体（4~40cm），填补了零样本多物体场景补全数据集空白

### 损失函数

$$\mathcal{L} = \mathcal{L}_{nrm} + \mathcal{L}_{SDF} + \sum_{h \in \{5,6,7,8,9\}} \mathcal{L}_{occ}^h$$

法线L2损失 + SDF L2损失 + 每个LoD的占据二值交叉熵。逐层剪枝空体素避免不必要计算。

## 实验关键数据

### 主实验

| 方法 | 3D表示 | Synthetic CD↓ | YCB-V CD↓ | HB CD↓ | HOPE CD↓ |
|------|--------|---------------|-----------|--------|----------|
| VoxFormer | Dense | 44.54 | 30.32 | 34.84 | 47.75 |
| ConvONet | Dense | 23.68 | 32.87 | 26.71 | 20.95 |
| MCC | Implicit | 43.37 | 35.85 | 19.59 | 17.53 |
| AICNet | Dense | 15.64 | 12.26 | 11.87 | 11.40 |
| Minkowski | Sparse | 11.47 | 8.04 | 8.81 | 8.56 |
| OCNN | Sparse | 9.05 | 7.10 | 7.02 | 8.05 |
| **OctMAE** | **Sparse** | **6.48** | **6.40** | **6.14** | **6.97** |

### 消融实验

位置编码对比（合成数据集）：

| 类型 | CD↓ | F1↑ | NC↑ |
|------|-----|-----|-----|
| 无位置编码 | 11.32 | 0.778 | 0.808 |
| CPE | 9.91 | 0.785 | 0.811 |
| APE | 8.61 | 0.782 | 0.825 |
| RPE | 7.81 | 0.804 | 0.830 |
| **RoPE** | **6.48** | **0.839** | **0.848** |

3D注意力机制对比（HOPE数据集）：

| 方法 | 遮挡掩码 | CD↓ | F1↑ |
|------|----------|-----|-----|
| 3D Deformable | ✗ | 12.14 | 0.703 |
| Neighbor Attn | ✗ | 9.26 | 0.727 |
| Octree Attn | ✗ | 7.99 | 0.752 |
| Octree Attn | ✓ | 7.54 | 0.772 |
| **Full Attn** | **✓** | **6.97** | **0.803** |

### 关键发现

- OctMAE在所有4个数据集上全面SOTA，且仅在合成数据上训练就能零样本泛化到真实场景
- 3D RoPE对性能贡献巨大：CD从11.32降至6.48
- 遮挡掩码策略使full attention成为可能，效果显著优于deformable attention
- 稀疏表示（Octree/Minkowski）全面优于密集/隐式表示
- 隐空间3D MAE是泛化性的关键：相同U-Net架构（Minkowski/OCNN）加入MAE后大幅提升

## 亮点与洞察

- 在隐空间做MAE是Scale up 3D Transformer的有效策略——LoD-5下token数可控
- 遮挡掩码是面向场景补全任务的精妙设计：只在最需要生成的地方放token
- 3D RoPE在3D vision中的应用值得关注，比传统位置编码更高效且泛化
- 大规模Objaverse数据集的创建对社区有独立贡献价值

## 局限性

- 需要前景分割掩码作为输入
- 对极端遮挡场景（仅见一小片）的补全仍有挑战
- Octree表示对非常薄的结构（如线缆）可能丢失细节
- 近实时但仍非严格实时

## 评分

- 新颖性：⭐⭐⭐⭐ — Octree+MAE混合架构+遮挡掩码策略
- 有效性：⭐⭐⭐⭐⭐ — 全面SOTA+零样本泛化
- 实用性：⭐⭐⭐⭐ — 机器人场景理解的实际需求
- 推荐度：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)
- [Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)

<!-- RELATED:END -->
