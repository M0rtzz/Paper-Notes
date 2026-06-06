---
title: >-
  [论文解读] STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding
description: >-
  [CVPR 2026][3D视觉][4D点云视频] STS-Mixer 首次将图傅里叶变换（GFT）引入 4D 点云视频理解，通过频域分解捕获不同尺度的几何结构（低频=全局形状、高频=局部细节），与时空信息混合后在动作识别和语义分割上达到 SOTA。
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "4D点云视频"
  - "图傅里叶变换"
  - "频谱表示"
  - "动作识别"
  - "语义分割"
---

# STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding

**会议**: CVPR 2026  
**arXiv**: [2604.11637](https://arxiv.org/abs/2604.11637)  
**代码**: [https://github.com/Vegetebird/STS-Mixer](https://github.com/Vegetebird/STS-Mixer)  
**领域**: 3D视觉  
**关键词**: 4D点云视频, 图傅里叶变换, 频谱表示, 动作识别, 语义分割

## 一句话总结
STS-Mixer 首次将图傅里叶变换（GFT）引入 4D 点云视频理解，通过频域分解捕获不同尺度的几何结构（低频=全局形状、高频=局部细节），与时空信息混合后在动作识别和语义分割上达到 SOTA。

## 研究背景与动机

**领域现状**：4D 点云视频包含 3D 空间+时间信息，现有方法（P4Transformer、PST-Transformer 等）在时空域建模短期和长期动态。

**现有痛点**：现有方法仅在时空域工作，难以捕获点云的底层几何特性——抽象形状和局部-全局上下文。点云的不规则无序性使得标准频域变换（如 DCT）不适用。

**核心矛盾**：时空域能建模运动动态但缺少对静态几何结构的显式建模，而几何结构（全局形状、局部细节）对理解 4D 场景至关重要。

**切入角度**：图傅里叶变换（GFT）天然适合不规则点云——通过图拉普拉斯的特征分解将点云转换到频域，不同频带捕获不同尺度的几何结构。

**核心 idea**：将 4D 点云分解为多频带信号（低/中/高频），各频带捕获不同几何特征，与时空信息混合实现全面表示学习。

## 方法详解

### 整体框架
输入 4D 点云视频 → 4D 点卷积编码局部时空 → GFT 变换到频域 → 频谱滤波器分解为低/中/高频 → IGFT 逆变换回空间域得到三套频带特定点云 → STS-Mixer 块处理（FA-Attention 频带内细化 + FM-MLP 频带间交互） → MLP 输出预测。

### 关键设计

1. **图傅里叶变换频域分解**:

    - 功能：将点云几何结构显式分解为多尺度信息
    - 核心思路：以每帧点云构建 KNN 图，计算归一化图拉普拉斯矩阵的特征分解，特征向量按特征值排序形成频率基。将点坐标投影到这些基上得到 GFT 系数，用频带滤波器分为低/中/高频段，各段 IGFT 逆变换回空间域得到频带特定的点云重建
    - 设计动机：频带拒绝实验证实低频保留全局形状、高频编码细节，这种分离可以让网络分别处理不同尺度的几何信息

2. **频率感知注意力（FA-Attention）**:

    - 功能：频带内独立细化各频段的表示
    - 核心思路：对每个频带（低/中/高频）独立应用自注意力，让同一频段内的点互相关注，捕获该尺度特有的几何模式
    - 设计动机：不同频段的几何信息语义不同（全局 vs 局部），独立处理避免了混合带来的信息干扰

3. **频率混合 MLP（FM-MLP）**:

    - 功能：促进不同频段之间的信息交换
    - 核心思路：将三个频段的特征沿频率维度拼接，通过 MLP 交换信息，再拆分回各频段。这实现了频带间的互相增强
    - 设计动机：虽然各频段捕获的信息不同，但它们描述的是同一个物体/场景，互相补充可以产生更全面的理解

### 损失函数 / 训练策略
动作识别用交叉熵损失，语义分割用带 Lovász-softmax 的交叉熵损失。

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | STS-Mixer | 之前SOTA | 提升 |
|-------------|------|-----------|----------|------|
| MSR-Action3D 动作识别 | Acc | SOTA | PST-Transformer | 提升 |
| NTU RGB+D 60 动作识别 | Acc | SOTA | PPTr | 提升 |
| Synthia 4D 语义分割 | mIoU | SOTA | PST-Transformer | 提升 |

### 消融实验

| 配置 | 准确率 | 说明 |
|------|--------|------|
| Full STS-Mixer | 最优 | 时空+频谱 |
| 仅时空(无GFT) | 下降 | 缺乏几何结构建模 |
| w/o FA-Attention | 下降 | 频带内细化缺失 |
| w/o FM-MLP | 下降 | 频带间交互缺失 |
| 单频带 | 下降 | 多频带分解必要 |

### 关键发现
- 频谱表示与时空表示高度互补——各自捕获不同方面的信息
- 低频对动作识别贡献最大（全局形状区分动作类别），高频对精细分割更重要
- 三频带分解比两频带效果更好，频带数继续增加收益递减

## 亮点与洞察
- **首次频域视角看 4D 点云**：GFT 为点云理解开辟了新的信息维度，类似于 RGB 图像中的频域处理
- **频带拒绝的直观验证**：通过"去掉某个频带看重建效果"直观展示了各频带的信息含义

## 局限与展望
- GFT 计算（特征分解）在大规模点云上可能成为瓶颈
- 频带数和滤波器参数需要手动设定
- 未来可探索自适应频带分解和更高效的频谱方法

## 相关工作与启发
- **vs P4Transformer/PST-Transformer**: 纯时空域建模，忽略了频域几何信息
- **vs PointGST/PointWavelet**: 仅处理静态点云的频域方法，未拓展到 4D 视频

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将 GFT 引入 4D 点云理解，视角独特
- 实验充分度: ⭐⭐⭐⭐ 动作识别+语义分割两个任务，多数据集验证
- 写作质量: ⭐⭐⭐⭐ 频域分析清晰直观
- 价值: ⭐⭐⭐⭐ 为 4D 理解开辟了新维度

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](../../ICCV2025/3d_vision/ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)
- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[CVPR 2026\] SparseCam4D: Spatio-Temporally Consistent 4D Reconstruction from Sparse Cameras](sparsecam4d_spatio-temporally_consistent_4d_reconstruction_from_sparse_cameras.md)
- [\[CVPR 2026\] STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction](stac_plug-and-play_spatio-temporal_aware_cache_compression_for_streaming_3d_reco.md)
- [\[CVPR 2026\] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)

</div>

<!-- RELATED:END -->
