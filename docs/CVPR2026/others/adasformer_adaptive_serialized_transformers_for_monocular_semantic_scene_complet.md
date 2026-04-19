---
title: >-
  [论文解读] AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments
description: >-
  [CVPR 2026][语义场景补全] 提出AdaSFormer，一种针对室内单目语义场景补全(MSSC)的序列化Transformer框架，通过自适应序列化注意力(可学习偏移量)、中心相对位置编码和卷积调制层归一化三个核心设计，在NYUv2和Occ-ScanNet上达到SOTA。
tags:
  - CVPR 2026
  - 语义场景补全
  - Transformer
  - 自适应注意力
  - 室内场景
  - 单目RGB
---

# AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments

**会议**: CVPR 2026  
**arXiv**: [2603.25494](https://arxiv.org/abs/2603.25494)  
**代码**: https://github.com/alanWXZ/AdaSFormer (有)  
**领域**: 3D视觉 / 室内场景理解  
**关键词**: 语义场景补全, 序列化Transformer, 自适应注意力, 室内场景, 单目RGB

## 一句话总结
提出AdaSFormer，一种针对室内单目语义场景补全(MSSC)的序列化Transformer框架，通过自适应序列化注意力(可学习偏移量)、中心相对位置编码和卷积调制层归一化三个核心设计，在NYUv2和Occ-ScanNet上达到SOTA。

## 研究背景与动机

**领域现状**：单目语义场景补全从单张RGB图像预测完整3D场景的体素占据和语义标签。室外(自动驾驶)场景已有大量研究，但室内MSSC因空间布局复杂和严重遮挡而更具挑战。

**现有痛点**：现有室内方法主要依赖CNN架构——局部感受野无法建模长程依赖，3D卷积核增大计算开销立方增长。Transformer虽能建模全局上下文，但直接应用于密集3D体素计算和内存开销巨大。

**核心矛盾**：室内场景需要强全局上下文推理（推断遮挡区域的几何和语义），但高分辨率3D体素使Transformer的 $O(N^2)$ 复杂度不可行。

**切入角度**：序列化Transformer将不规则3D数据转为有序序列，通过局部分组将复杂度降至 $O(N \cdot G)$，但现有方法的分组方案固定，感受野受限。

**核心idea**：引入可学习偏移量自适应调整序列化起点→不同层获得不同感受野→更灵活的空间表示。

## 方法详解

### 整体框架
单目RGB图像→2D编码器(EfficientNet)+深度估计→3D投影→3D编码器(多个AdaSFormer块交替Transformer和卷积)→轻量解码器→SSC输出。

### 关键设计

1. **自适应序列化注意力(ASA)**:

    - 功能：通过可学习偏移自适应调整序列化起点，获得更灵活的感受野
    - 核心思路：假设patch大小 $P$，引入 $K$ 个可学习参数表示偏移值（均匀间隔 $P/K$）。用**Straight-Through Gumbel-Softmax**实现可微离散选择：$\mathbf{y}_{soft} = \text{softmax}((\mathbf{l} + \mathbf{g})/\tau)$，前向用硬选择 $\mathbf{y}_{hard}$，反向通过 $\mathbf{y}_{soft}$ 传梯度。温度退火策略 $\tau_t = \max(\tau_{min}, \tau_{init} \cdot \exp(-\alpha t))$ 逐步增强离散性
    - 设计动机：不同起点显著改变感受野覆盖——可能完全覆盖单个物体或同时包含多个物体的空间关系。Swin Transformer的窗口偏移固定且不可学习，序列化注意力沿1D序列操作，偏移空间更广更灵活

2. **中心相对位置编码(CRPE)**:

    - 功能：编码每个体素与场景中心的空间关系，捕捉信息丰富度
    - 核心思路：计算场景中心 $\mathbf{c}$（所有占据体素坐标均值），计算每个体素相对场景中心的偏航角差 $\Delta\theta$ 和俯仰角差 $\Delta\phi$，拼接后过MLP作为注意力偏置
    - 设计动机：CNN组件已编码局部位置信息，额外的位置编码应侧重空间信息分布——距离场景中心不同位置的结构和语义信息丰富度不同

3. **卷积调制层归一化(CMLN)**:

    - 功能：桥接CNN和Transformer的异构特征表示
    - 核心思路：$\text{CMLN}(h_i | X_{voxel}) = \gamma(X_{voxel}) \odot \frac{h_i - \mu_i}{\sigma_i} + \beta(X_{voxel})$，归一化参数 $\gamma, \beta$ 由体素特征通过小MLP生成
    - 设计动机：Transformer和CNN提取根本不同的特征类型，直接交替使用导致学习困难，需要自适应特征统计调制

### 损失函数 / 训练策略
标准SSC损失（交叉熵+场景补全IoU相关损失）。

## 实验关键数据

### 主实验（NYUv2 数据集）

| 方法 | 会议 | SC IoU% | SSC mIoU% |
|------|------|---------|-----------|
| MonoScene | CVPR'22 | 42.51 | 26.94 |
| NDC-Scene | ICCV'23 | 44.17 | 29.03 |
| ISO | ECCV'24 | 47.11 | 31.25 |
| MonoMRN | ICCV'25 | 53.16 | 26.80* |
| **AdaSFormer (Ours)** | CVPR'26 | **SOTA** | **SOTA** |

*注：MonoMRN在SC IoU上强但SSC mIoU较低，AdaSFormer在两个指标上均达到SOTA。

### 消融实验（NYUv2）

| 配置 | SC IoU | SSC mIoU |
|------|--------|----------|
| 基线 (标准序列化Transformer) | 基准 | 基准 |
| + ASA (可学习偏移) | +提升 | +提升 |
| + CRPE (中心相对编码) | +提升 | +提升 |
| + CMLN (调制归一化) | +提升 | +提升 |
| 全部组合 | **最优** | **最优** |

### 关键发现
- 自适应序列化注意力是最关键组件——可学习偏移比固定偏移提升显著
- 中心相对位置编码在室内场景中特别有效——室内场景的结构更以中心为导向
- CMLN解决了直接CNN-Transformer交替的特征不匹配问题
- 在NYUv2和Occ-ScanNet两个数据集上均达到SOTA
- 相比全3D Transformer内存和计算开销大幅减小

## 亮点与洞察
- **可学习序列化偏移**：用Gumbel-Softmax让离散的序列化起点选择变可微，这是对序列化Transformer的通用改进，可迁移到点云分割和3D检测
- **空间信息丰富度编码**：不同于标准位置编码记录绝对/相对位置，CRPE编码的是空间信息密度——距离场景中心更远的区域通常信息更稀疏
- **CNN-Transformer异构特征桥接**：CMLN为混合架构设计中的特征统计不匹配问题提供了优雅的解决方案

## 局限与展望
- 仅在室内场景上验证（NYUv2较小），更大规模室内数据集的效果待验证
- 深度估计质量对整体性能影响大，端到端训练需确保深度网络和补全网络的协同
- 场景中心用占据体素均值计算可能不鲁棒——如果占据分布偏斜怎么办？
- 可学习偏移的K个候选值是预定义的等间距，自适应间距可能更优

## 相关工作与启发
- **vs MonoScene/NDC-Scene/ISO**: 全CNN架构缺乏全局推理能力，本文引入Transformer弥补
- **vs OctFormer/PTv3**: 通用序列化Transformer设计，本文增加了可学习偏移以适应SSC
- **vs Swin Transformer**: Swin的窗口偏移固定且限于2D，本文的序列化偏移沿1D序列操作更灵活

## 评分
- 新颖性: ⭐⭐⭐⭐ 可学习序列化偏移有创意，CRPE和CMLN设计合理
- 实验充分度: ⭐⭐⭐⭐ NYUv2和Occ-ScanNet验证全面，但室内数据集规模较小
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示直观
- 价值: ⭐⭐⭐ 室内SSC方向改进，但应用场景相对狭窄

<!-- RELATED:START -->

## 相关论文

- [Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion](../../AAAI2026/others/towards_temporal_fusion_beyond_the_field_of_view_for_camera-based_semantic_scene.md)
- [SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [The Counting Power of Transformers](../../ICLR2026/others/the_counting_power_of_transformers.md)
- [Tab-PET: Graph-Based Positional Encodings for Tabular Transformers](../../AAAI2026/others/tab-pet_graph-based_positional_encodings_for_tabular_transformers.md)
- [What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution](what_is_wrong_with_synthetic_data_for_scene_text_recognition_a_strong_synthetic_.md)

<!-- RELATED:END -->
