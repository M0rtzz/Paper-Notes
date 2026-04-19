---
title: >-
  [论文解读] RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation
description: >-
  [CVPR 2026][图像分割][视频语义分割] 提出 RS-SSM，通过频域分析提取各通道的特定信息分布特征（CwAP），并自适应反转遗忘门矩阵来补充性精炼 SSM 状态空间压缩时丢失的时空细节（FGIR），在 4 个视频语义分割基准上达到 SOTA 且保持高效率。
tags:
  - CVPR 2026
  - 图像分割
  - 视频语义分割
  - 状态空间模型
  - 遗忘门精炼
  - 频域分析
  - Mamba
---

# RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.24295](https://arxiv.org/abs/2603.24295)  
**代码**: https://github.com/zhoujiahuan1991/CVPR2026-RS-SSM  
**领域**: 语义分割 / 视频理解  
**关键词**: 视频语义分割, 状态空间模型, 遗忘门精炼, 频域分析, Mamba

## 一句话总结

提出 RS-SSM，通过频域分析提取各通道的特定信息分布特征（CwAP），并自适应反转遗忘门矩阵来补充性精炼 SSM 状态空间压缩时丢失的时空细节（FGIR），在 4 个视频语义分割基准上达到 SOTA 且保持高效率。

## 研究背景与动机

1. **领域现状**：视频语义分割（VSS）要求对视频每帧每像素赋予语义标签，并保持时序一致性。早期方法用光流建模帧间运动但计算昂贵且噪声大，Transformer 方法通过全局注意力聚合时空信息但复杂度为二次方。近年来，基于状态空间模型（SSM）的方法以线性复杂度高效压缩和传播时空语义信息。

2. **现有痛点**：SSM 通过固定大小的状态空间压缩序列信息，这在保留共性语义（全局结构、平滑区域）时是有效的，但不可避免地丢失了特定信息（边界、纹理、局部变化），导致分割结果仅能粗略定位物体位置，细节模糊。

3. **核心矛盾**：状态空间压缩的本质是用有限维隐状态表示无限长序列，遗忘门 $\bar{A}_d = \exp(\Delta_d A_d)$ 决定了对历史信息的衰减程度。较小的遗忘门值导致激进的压缩，而接近 1 的值保留更多信息——对于像素级 VSS，标准遗忘门的衰减策略会系统性地丢失高频细节。

4. **本文目标** 在保持 SSM 线性复杂度优势的同时，补偿其状态空间压缩过程中遗失的时空特定信息。

5. **切入角度**：观察到不同通道的隐状态包含不同量的特定信息，可以通过频域分析量化各通道的高频能量占比，然后反转遗忘门来重点精炼高频信息丰富的通道。

6. **核心 idea**：通过频域分析定位"特定信息丰富"的通道，自适应反转遗忘门来补偿性恢复被压缩遗忘的时空细节。

## 方法详解

### 整体框架

输入视频帧 $\{I_t\}_{t=1}^T$ 经图像编码器提取特征 $\{M_t\}$，送入 $L$ 层双路径 SSM 层。每层包含两个 SSM 模块 $\theta_1, \theta_2$：$\theta_2$ 是标准 SSM 用于提取共性语义，$\theta_1$ 通过 CwAP 和 FGIR 模块精炼 $\theta_2$ 遗忘的特定信息。两路输出拼接后经 MLP 融合，最终送入线性分割解码器生成逐帧分割掩码。

### 关键设计

1. **通道幅值感知器 (CwAP)**:

    - 功能：量化每个通道中特定信息（高频细节）的含量
    - 核心思路：对投影特征 $H_t \in \mathbb{R}^{D \times H \times W}$ 做 2D FFT 变换为频域，计算幅值谱 $H_t^m$。将频域按归一化频率半径分为 $K$ 个频带，低频带对应共性语义，高频带对应特定信息。统计每个通道在最高 $k_h$ 个高频带的归一化能量占比，得到频谱特征 $F_t \in \mathbb{R}^D$，反映各通道的特定信息量。
    - 设计动机：不同通道的隐状态编码不同语义层次的信息。通过频域析出高频能量占比，提供了一个不涉及标签的无监督度量来定位"细节丰富"的通道。

2. **通道信息损失 ($\mathcal{L}_{ci}$)**:

    - 功能：跨样本对齐特定信息的通道分布
    - 核心思路：将频谱特征 L2 归一化后，计算 batch 内所有帧对的余弦相似度矩阵 $\mathbf{S}_{i,j}$，损失为 $\mathcal{L}_{ci} = 1 - \frac{1}{|\mathcal{B}|^2} \sum_i \sum_j \mathbf{S}_{i,j}$，即最大化平均余弦相似度。
    - 设计动机：如果不同样本的特定信息分散在不同通道，FGIR 难以统一精炼。通过对齐通道分布，让特定信息集中在相似的通道子集中，使后续精炼更高效一致。

3. **遗忘门信息精炼器 (FGIR)**:

    - 功能：自适应反转遗忘门，引导 SSM 聚焦被遗忘的特定信息
    - 核心思路：利用 CwAP 提取的频谱特征 $F_t$，自适应地修改 $\theta_1$ 的遗忘门矩阵。核心操作是"反转遗忘门"——将标准 SSM $\theta_2$ 中衰减最严重（遗忘最多）的通道，在 $\theta_1$ 中对应地保留更多信息。具体而言，$\theta_2$ 的遗忘门指示了哪些通道被大量压缩，FGIR 基于频谱特征权重反转这些遗忘门值，使 $\theta_1$ 在这些通道上执行对 $\theta_2$ 的补偿性精炼。
    - 设计动机：双路径设计让 $\theta_2$ 负责共性语义的常规 SSM 建模，$\theta_1$ 专注于补全 $\theta_2$ 丢失的细节。遗忘门反转确保两个 SSM 的关注点互补而非冗余。

### 损失函数 / 训练策略

总损失包括标准的语义分割交叉熵损失和通道信息损失 $\mathcal{L}_{ci}$。图像编码器使用 SegFormer 预训练权重。输入分辨率 480×853 计算 GFLOPs 和 FPS。双路径 SSM 层数 $L$ 为超参数。频带数 $K$ 和高频带数 $k_h$ 是 CwAP 的关键超参。

## 实验关键数据

### 主实验

在 VSPW、NYUv2、CamVid 三个数据集上与现有方法对比。根据论文摘要和方法描述，RS-SSM 在保持高效率（线性复杂度）的同时达到最佳或次佳分割精度：

| 数据集 | 指标 | RS-SSM 表现 | 说明 |
|--------|------|-------------|------|
| VSPW | mIoU | SOTA | 大规模视频语义分割基准 |
| NYUv2 | mIoU | SOTA | 室内场景分割 |
| CamVid | mIoU | SOTA | 驾驶场景视频分割 |
| 4 基准整体 | mIoU + GFLOPs + FPS | 精度最优且效率保持 | 线性复杂度 vs Transformer 二次复杂度 |

（注：缓存文件中实验数据表格截断，具体数值需参见原文 Table 1）

### 消融实验

根据方法设计，消融实验应包括：

| 配置 | 说明 |
|------|------|
| Full RS-SSM | 完整模型，CwAP + FGIR + $\mathcal{L}_{ci}$ |
| w/o CwAP | 去掉频域通道分析，FGIR 无频谱特征引导 |
| w/o FGIR | 去掉遗忘门反转，仅做标准双路径 SSM |
| w/o $\mathcal{L}_{ci}$ | 去掉通道对齐损失 |
| Single-path SSM | 去掉双路径，退化为标准 SSM（如 TV3S） |

### 关键发现

- **遗忘门反转是核心贡献**：通过可视化更新门 $\bar{B}_d$ 可以看到，标准 SSM $\theta_2$ 在细节区域信息量严重衰减，而反转遗忘门后的 $\theta_1$ 能有效恢复这些被遗忘的边界和纹理细节。
- **CwAP 频域分析提供了有效的通道选择信号**：高频能量占比高的通道确实对应更多的边界和纹理信息，频谱特征是一个无监督且计算高效的通道重要性度量。
- **通道信息损失的对齐效果**：跨样本对齐使得 FGIR 可以一致地精炼相似的通道子集，避免了每个样本的精炼方向不一致导致的训练不稳定。
- **线性复杂度优势**：相比 Transformer 方法在长视频上的二次复杂度瓶颈，RS-SSM 保持了 SSM 的线性效率。

## 亮点与洞察

- **"遗忘的反面即精炼"的思想非常巧妙**：不是试图阻止 SSM 遗忘（这会破坏其压缩优势），而是用一个互补的 SSM 专门恢复被遗忘的内容。这种"分工互补"的双路径设计比简单增大状态空间维度更优雅。
- **频域作为通道分析工具**：用 FFT 幅值谱的高频能量占比来量化"特定信息量"，原理直观（高频=边界/纹理）、计算高效（FFT 是 O(n log n)）、无需标签。这种频域分析可迁移到任何需要区分"粗粒度 vs 细粒度"信息的场景。
- **SSM 在视觉领域的系统性改进**：识别了 SSM 在像素级任务中的关键瓶颈（遗忘细节），并提供了针对性解决方案，对 Mamba 在视觉领域的进一步发展有指导意义。

## 局限与展望

- 双路径设计增加了参数量和计算量（虽然仍是线性复杂度），需要验证在更大分辨率/更长视频下的效率优势是否仍然成立
- CwAP 的频带数 $K$ 和高频带数 $k_h$ 需要手动设定，对不同数据集可能需要调整
- 仅在预定义的 4 个 VSS 基准上验证，未在 panoptic/instance 级别的视频分割上测试
- 遗忘门反转是否会导致某些长距离依赖信息的过度保留，造成"信息过载"
- 可以探索可学习的频域选择机制，替代固定的频带划分

## 相关工作与启发

- **vs TV3S**: TV3S 是首个将 SSM 应用于 VSS 的方法，但忽略了状态空间压缩导致的细节丢失。RS-SSM 直接针对这一问题，通过遗忘门反转实现补偿精炼
- **vs Transformer VSS (CFFM/MRCFA)**: Transformer 方法通过全局注意力天然保留细节，但复杂度为二次方。RS-SSM 在线性复杂度下通过双路径+频域引导达到可比甚至更优的精度
- **vs VideoMamba**: VideoMamba 是通用视频骨干，RS-SSM 专门为像素级分割任务设计了遗忘门精炼策略，是任务特定的 SSM 改进

## 评分

- 新颖性: ⭐⭐⭐⭐ 遗忘门反转+频域通道分析的组合是新颖的，"在 SSM 中精炼被遗忘细节"的视角有启发性
- 实验充分度: ⭐⭐⭐⭐ 4 个基准覆盖面广，但缓存中具体数值截断，更新门可视化提供了直观验证
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，双路径架构和频域分析的可视化清晰
- 价值: ⭐⭐⭐⭐ 为 SSM 在像素级视觉任务中的应用提供了系统性解决方案，对 Mamba 在视觉领域的发展有推动作用

<!-- RELATED:START -->

## 相关论文

- [Exploiting Temporal State Space Sharing for Video Semantic Segmentation](../../CVPR2025/segmentation/exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [DefMamba: Deformable Visual State Space Model](../../CVPR2025/segmentation/defmamba_deformable_visual_state_space_model.md)
- [GroupMamba: Efficient Group-Based Visual State Space Model](../../CVPR2025/segmentation/groupmamba_efficient_group-based_visual_state_space_model.md)
- [VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](videomt_encoder_only_video_segmentation.md)
- [RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)

<!-- RELATED:END -->
