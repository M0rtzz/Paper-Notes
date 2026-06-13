---
title: >-
  [论文解读] Consistency-aware Self-Training for Iterative-based Stereo Matching
description: >-
  [CVPR 2025][3D视觉][立体匹配] 首次提出面向迭代式立体匹配的一致性感知自训练框架（CST-Stereo），通过多分辨率预测一致性滤波和迭代预测一致性滤波评估伪标签可靠性，结合软加权损失有效利用无标签真实数据提升模型性能和泛化能力。 领域现状 领域现状：迭代式方法（如RAFT-Stereo、IGEV-Stere…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "立体匹配"
  - "自训练"
  - "伪标签过滤"
  - "一致性感知"
  - "迭代优化"
---

# Consistency-aware Self-Training for Iterative-based Stereo Matching

**会议**: CVPR 2025  
**arXiv**: [2503.23747](https://arxiv.org/abs/2503.23747)  
**代码**: 无  
**领域**: 3D视觉 / 立体匹配  
**关键词**: 立体匹配, 自训练, 伪标签过滤, 一致性感知, 迭代优化

## 一句话总结

首次提出面向迭代式立体匹配的一致性感知自训练框架（CST-Stereo），通过多分辨率预测一致性滤波和迭代预测一致性滤波评估伪标签可靠性，结合软加权损失有效利用无标签真实数据提升模型性能和泛化能力。

## 研究背景与动机

### 领域现状

**领域现状**：迭代式方法（如RAFT-Stereo、IGEV-Stereo）已成为立体匹配主流，但严重依赖标注数据

### 现有痛点

**现有痛点**：高质量立体标注获取成本高昂，现有标注数据多为合成数据，在真实场景泛化性差

### 核心矛盾

**核心矛盾**：已有自训练方法仅适用于代价体方法，无法直接应用于无完整代价体的迭代式方法

### 解决思路

**解决思路**：已有伪标签过滤策略采用硬阈值二值选择，既丢弃有价值的困难样本，又无法区分阈值内伪标签的可靠性差异

### 补充说明

**补充说明**：核心观察：**误差较大的区域在模型预测中表现出更明显的振荡特性**

## 方法详解

### 整体框架

CST-Stereo采用教师-学生自训练框架：教师模型在无标签数据上生成伪标签，一致性感知软滤波模块（CSF）评估伪标签可靠性，学生模型在强增强输入下用软加权损失学习，教师通过EMA更新。

### 关键设计

1. **多分辨率预测一致性滤波器（MRPCF）**:
    - 功能：从空间维度评估伪标签可靠性
    - 核心思路：将输入图像分别上采样、保持原尺寸、下采样后送入教师模型，计算三种分辨率预测的像素级方差 $\sigma_{i,j}$，再通过sigmoid映射转为可靠性权重 $w_{rc} = 1/(1 + e^{-\varepsilon_1(\sigma - \tau_1)})$
    - 设计动机：不同分辨率下预测不一致的像素往往是不可靠的（如边缘、遮挡区域），一致的像素更可靠

2. **迭代预测一致性滤波器（IPCF）**:
    - 功能：从时序维度评估伪标签可靠性
    - 核心思路：计算迭代后半段相邻迭代回合预测差异的平均值 $\Delta_{i,j} = \frac{1}{\lceil n/2 \rceil} \sum_{k} |P^{k+1}_{i,j} - P^k_{i,j}|$，同样通过sigmoid映射为权重 $w_{ic}$
    - 设计动机：迭代后期仍然振荡的像素表明多源信息不一致、模型无法收敛到确定值，预测不可靠

3. **一致性感知软加权损失**:
    - 功能：结合两种滤波器的权重指导学生模型训练
    - 核心思路：$w_{soft} = w_{rc} \odot w_{ic}$，损失为 $L_{st} = w_{soft} \odot |\hat{P} - P^O|$，乘法确保只有两个滤波器都认为可靠的像素才获得高权重
    - 设计动机：避免硬阈值丢弃有价值的困难样本，同时降低噪声伪标签的影响

### 损失函数 / 训练策略

- 先在标注数据集上预训练获得初始权重
- 自训练阶段：学生接收强增强图像，教师不做增强
- EMA更新教师：$\theta_T \leftarrow \lambda \theta_T + (1-\lambda) \theta_S$
- 适用于多种迭代式基线：RAFT-Stereo、CREStereo、IGEV-Stereo、Selective-Stereo等

## 实验关键数据

### 主实验（在线排行榜）

| 方法 | KITTI2015 D1-all ↓ | Middlebury bad1.0 ↓ | ETH3D bad1.0 ↓ |
|------|-------------------|---------------------|----------------|
| RAFT-Stereo | 1.96 | 9.37 | 2.44 |
| IGEV-Stereo | 1.59 | 9.41 | 1.12 |
| Selective-IGEV | 1.55 | 6.53 | 1.23 |
| **CST-Stereo** | **1.50** | **6.23** | **1.02** |

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| 无过滤基线 | 性能下降 | 直接使用伪标签导致噪声累积 |
| 仅MRPCF | 有提升 | 捕获空间分辨率敏感的误差区域 |
| 仅IPCF | 有提升 | 捕获迭代振荡相关的误差区域 |
| MRPCF + IPCF（硬阈值） | 次优 | 丢弃困难样本 |
| MRPCF + IPCF（软权重） | **最优** | 兼顾可靠性评估和困难样本利用 |

### 关键发现

- 方法具有通用性，可以即插即用地提升多种迭代式基线（RAFT-Stereo +35%降低误差，CREStereo +21%等）
- 在域内、域适应、域泛化三种场景下均有效
- 在KITTI2015、Middlebury、ETH3D多个公开排行榜上取得已发表方法中的SOTA
- 多分辨率一致性和迭代一致性是互补的：MRPCF擅长边缘区域，IPCF擅长模糊区域

## 亮点与洞察

- "误差区域振荡特性"这一观察非常直觉且可视化验证充分
- 软阈值设计比硬二值过滤优雅得多，保留了困难样本的训练价值
- 方法不依赖代价体，首次将自训练扩展到迭代式立体匹配全家族
- 两种一致性滤波器从空间和时序两个正交维度评估可靠性

## 局限与展望

- 多分辨率预测需要三次前向传播，训练计算开销增大
- 滤波器中的软阈值参数 $\tau_1, \tau_2$ 和缩放因子 $\varepsilon_1, \varepsilon_2$ 需要手动设定
- 未探索更复杂的权重融合策略（如学习的融合权重）
- 可以尝试将观察推广到光流估计等其他密集预测任务

## 相关工作与启发

- 延续 StereoBase、PCT-Stereo 等自训练立体匹配工作，但首次针对迭代式方法
- 软过滤思想可借鉴到其他伪标签学习场景（如半监督语义分割）
- 一致性观察可能对理解迭代式模型的不确定性有更深层意义

## 评分

- 新颖性: ⭐⭐⭐⭐ 振荡特性观察新颖，两种互补滤波器设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 多基线、多场景、多数据集验证，消融详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，观察→设计的逻辑链完整
- 价值: ⭐⭐⭐⭐ 即插即用提升迭代式立体匹配，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[CVPR 2026\] PIP-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching](../../CVPR2026/3d_vision/pip-stereo_progressive_iterations_pruner_for_iterative_optimization_based_stereo.md)
- [\[CVPR 2025\] FoundationStereo: Zero-Shot Stereo Matching](foundationstereo_zero-shot_stereo_matching.md)
- [\[CVPR 2025\] SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Input](spatialdreamer_self-supervised_stereo_video_synthesis_from_monocular_input.md)
- [\[NeurIPS 2025\] U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching](../../NeurIPS2025/3d_vision/u-can_unsupervised_point_cloud_denoising_with_consistency-aware_noise2noise_matc.md)

</div>

<!-- RELATED:END -->
