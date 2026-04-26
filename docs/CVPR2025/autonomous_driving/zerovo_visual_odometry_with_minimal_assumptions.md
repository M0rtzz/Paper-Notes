---
title: >-
  [论文解读] ZeroVO: Visual Odometry with Minimal Assumptions
description: >-
  [CVPR 2025][自动驾驶][视觉里程计] 本文提出 ZeroVO，一种基于 Transformer 的单目视觉里程计方法，通过免标定的几何感知网络结构、语言先验融合和半监督训练范式，在 KITTI、nuScenes、Argoverse 2 和自建 GTA 数据集上实现了超过 30% 的零样本泛化性能提升。
tags:
  - CVPR 2025
  - 自动驾驶
  - 视觉里程计
  - 零样本泛化
  - 语言先验
  - 半监督学习
  - 免标定
---

# ZeroVO: Visual Odometry with Minimal Assumptions

**会议**: CVPR 2025  
**arXiv**: [2506.08005](https://arxiv.org/abs/2506.08005)  
**代码**: https://zvocvpr.github.io/ (有)  
**领域**: 自动驾驶 / 视觉里程计  
**关键词**: 视觉里程计, 零样本泛化, 语言先验, 半监督学习, 免标定

## 一句话总结
本文提出 ZeroVO，一种基于 Transformer 的单目视觉里程计方法，通过免标定的几何感知网络结构、语言先验融合和半监督训练范式，在 KITTI、nuScenes、Argoverse 2 和自建 GTA 数据集上实现了超过 30% 的零样本泛化性能提升。

## 研究背景与动机

**领域现状**：单目视觉里程计（VO）旨在从连续图像帧中估计相机的相对位姿（旋转+平移）。学习型 VO 方法通过在特定数据集上训练神经网络来替代传统几何方法，但通常在训练集上效果好、泛化到新场景时表现急剧下降。

**现有痛点**：现有方法上存在多重限制：(1) 依赖预先标定或已知的相机内参；(2) 在同一数据集上训练和评估，无法反映真实的泛化能力；(3) 在恶劣条件（夜间、雨天、强光反射、镜头脏污等）下迅速失效，特征追踪断裂、优化不稳定。能够跨相机配置、跨场景零样本泛化的 VO 方法仍然缺乏。

**核心矛盾**：单目 VO 面临尺度模糊性——从 2D 图像恢复 3D 平移时无法确定绝对尺度。传统方法通过已知相机内参和深度约束来缓解，但这要求知道精确的相机参数。一旦相机配置变化（不同车辆、不同设备），方法就需要重新标定。

**本文目标**：设计一种"最小假设"的 VO 框架——不需要相机标定、不需要在目标数据集上微调、不需要真实的深度/光流标签——在自动驾驶中实现零样本泛化。

**切入角度**：不依赖单一线索，而是融合多种互补的图像级先验（估计的光流、度量深度、相机内参、语言描述），并通过 cross-attention 机制让网络自适应地处理各线索中的噪声。

**核心 idea**：用语言先验为几何估计提供高层语义补充，用半监督训练从大规模无标注视频中学习泛化能力，用多模态伪标签选择机制过滤噪声。

## 方法详解

### 整体框架
给定两帧连续图像，ZeroVO 提取多模态特征（光流、度量深度、估计内参、语言描述），通过语言和几何引导的 Transformer 融合模块生成统一表示，最终由双分支 MLP 解码器分别预测平移向量和旋转矩阵。可选的半监督训练阶段利用 YouTube 驾驶视频进行自训练。

### 关键设计

1. **免标定的几何感知编码**:

    - 功能：无需已知相机内参即可提取有效的几何特征
    - 核心思路：利用 WildCamera 模型从单张图像估计相机内参 $\hat{\mathbf{K}}$，并编码为图像尺寸的内参图 $\mathbf{I}_{\hat{K}}(u,v) = \frac{|u-c_U|}{f_U} + \frac{|v-c_V|}{f_V}$。用 Metric3Dv2 结合估计内参生成度量深度图 $\hat{\mathbf{D}}$，通过反投影得到 3D 点云 $\hat{\mathbf{D}}_{3D}$。同时将 2D 光流反投影为 3D 场景流 $\hat{\mathbf{F}}_{3D}$。这些 3D 估计虽然有噪声，但作为结构化先验被送入 Transformer 进行鲁棒融合
    - 设计动机：传统方法将内参作为已知常数使用，一旦不准确就崩溃。ZeroVO 将内参作为另一种带噪声的特征输入，让网络学习在噪声条件下仍能做出合理推断

2. **语言引导的深度和光流精炼**:

    - 功能：利用高层语义信息改善底层几何估计
    - 核心思路：用 LLaVA-NeXT 为每帧生成场景描述（包含道路类型、天气、动态对象等），通过 Sentence Transformer 编码为 15×768 的语言特征矩阵 $\mathbf{Z}_l$。然后通过 cross-attention 先将语言特征注入深度和光流特征：$\mathbf{Z} = \text{CA}(\text{PE}([\hat{\mathbf{D}}, \mathbf{I}_{\hat{K}}]), \mathbf{Z}_l)$（语言→深度），再将精炼后的特征与 3D 几何信息融合：$\mathbf{Z}_D = \text{CA}(\text{PE}(\hat{\mathbf{D}}_{3D}), \mathbf{Z})$。光流分支类似处理
    - 设计动机：在明和暗、雨和晴等条件下，深度和光流估计的可靠性差异巨大。语言描述提供了条件的上下文信息（"夜晚雨天"），帮助网络动态调整对几何特征的信任程度。实验中语言模块带来了 1.3 PQ 的 ATE 降低

3. **多模态伪标签半监督训练**:

    - 功能：利用大规模无标注视频提升泛化能力
    - 核心思路：分两阶段。第一阶段在标注数据（nuScenes-OneNorth）上监督训练得到教师模型。第二阶段用教师模型对 YouTube 驾驶视频生成伪标签，但伪标签有噪声。设计两种过滤机制：(1) 几何一致性过滤——用估计的位姿和深度做视角合成，通过 SSIM 评估合成图像与真实图像的相似度，归一化 SSIM < 0.5 的样本被过滤；(2) 语言引导过滤——在时间窗口 $H=10$ 内比较首尾帧的语言特征相似度，相似度过高说明场景变化小（信息冗余），予以过滤
    - 设计动机：直接用所有伪标签训练反而性能下降，因为冗余和噪声样本污染了学习。双重过滤确保只保留高质量、高信息量的伪标签

### 损失函数 / 训练策略
监督损失 $\mathcal{L} = \|\mathbf{t} - \hat{\mathbf{t}}\|_2^2 - \log(p(\mathbf{R}|\hat{\Psi}))$，其中平移用 MSE 回归，旋转用 Matrix Fisher 分布的负对数似然。使用 SGD 优化器训练100 epoch，batch size 16，初始学习率 0.001。数据增强使用随机裁剪缩放（RCR，模拟不同内参）和水平翻转。

## 实验关键数据

### 主实验

| 方法 | KITTI ATE↓ | nuScenes ATE↓ | Argoverse ATE↓ | GTA ATE↓ |
|------|-----------|-------------|---------------|---------|
| XVO | 168.43 | 8.30 | 5.70 | 28.02 |
| M+DS (Metric3D+DROID) | 154.77 | 10.46 | 8.51 | 12.96 |
| ZeroVO | 105.07 | 6.79 | 4.10 | 8.55 |
| **ZeroVO+** | **104.69** | **6.03** | **3.05** | **8.24** |
| TartanVO* | 103.07 | 6.26 | 7.03 | 3.82 |
| DPVO* | 78.53 | 2.66 | 1.59 | 4.33 |

*注：TartanVO/DPVO 使用真实尺度对齐（特权信息），非度量尺度预测。

### 消融实验

| Flow | Depth | Lang | SSL | Filter | KITTI ATE | nuScenes ATE |
|------|-------|------|-----|--------|-----------|-------------|
| ✓ | | | | | 174.24 | 12.54 |
| ✓ | ✓ | | | | 123.42 | 8.40 |
| ✓ | ✓ | ✓ | | | 105.07 | 6.79 |
| ✓ | ✓ | ✓ | ✓ | | 117.49 | 7.53 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **104.69** | **6.03** |

### 关键发现
- **度量深度是尺度恢复的关键**：加入深度模块后 KITTI ATE 从 174 降到 123（-29%），尺度误差大幅降低
- **语言先验有显著帮助**：在所有数据集上一致性地降低了 1-2 个 ATE 点，在恶劣条件（夜间/雨天）下效果更明显
- **无过滤的半监督训练反而有害**：ATE 从 105 升到 117，说明伪标签噪声会污染学习。加入双重过滤后 ATE 降到 105 以下
- 条件分析显示夜间和强光是最具挑战性的场景（ATE 10-13 vs 白天 3.6），ZeroVO+ 在这些条件下仍大幅优于基线
- ZeroVO+ 在所有度量尺度方法中是最优的，且在部分指标上接近使用真实尺度对齐的 DPVO

## 亮点与洞察
- **语言作为 VO 的语义先验**：这是首次将 VLM 的语言描述引入视觉里程计任务。语言提供了"场景是什么"的高层理解（夜间城市道路 vs 晴天高速公路），帮助网络在低层几何特征不可靠时保持鲁棒。这个思路可推广到其他几何估计任务（深度估计、场景流等）
- **内参作为可学习特征而非固定参数**：将通常作为已知常量的相机内参变成一种带噪声的输入特征，让网络学习容忍内参误差，大大拓展了方法的适用范围
- **GTA 数据集的价值**：新引入的高保真合成数据集包含真实数据集中罕见的极端场景（暴雪、镜头水珠、夜间沙漠），是评估鲁棒性的理想补充

## 局限与展望
- 推理速度较慢：ZeroVO+ 仅 0.6 FPS（主要受限于 LLaVA-NeXT 的 0.7 FPS），距离实时部署差距很大。LiteZeroVO+（不用语言模块）可达 5 FPS 但性能有下降
- 深度估计的失败会传播到 VO——在天空反射、玻璃表面、镜头脏污等场景，Metric3Dv2 的深度预测严重失真
- 两帧方法天然存在累积漂移问题，缺乏全局优化（如回环检测）。虽然 ZeroVO 可作为更大 SLAM 系统的前端，但论文未展示集成效果
- 语言描述是预计算的，未来可探索更轻量的语义编码方案替代 LLaVA + Sentence Transformer 的重型管线

## 相关工作与启发
- **vs TartanVO**：使用随机裁剪模拟不同内参来增强泛化，但只能预测相对尺度。ZeroVO 利用度量深度恢复绝对尺度
- **vs XVO**：也用多模态架构和自训练，但 ZeroVO 增加了语言先验和更好的伪标签过滤，在所有数据集上一致性超越
- **vs DROID-SLAM (M+DS)**：M+DS 利用 Metric3Dv2+DROID-SLAM 的多帧优化，理论上更强但在实际恶劣条件下不如 ZeroVO+ 稳定——多帧优化在追踪断裂时会整体失败

## 评分
- 新颖性: ⭐⭐⭐⭐ 语言先验引入 VO、免标定设计和多模态伪标签过滤都有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集含新建GTA、详尽消融、条件分析、噪声鲁棒性测试
- 写作质量: ⭐⭐⭐⭐ 方法描述详细清晰，补充材料非常丰富
- 价值: ⭐⭐⭐⭐ 为实际部署中的免标定、跨域 VO 提供了有力方案

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion](../../ECCV2024/autonomous_driving/dvlo_deep_visuallidar_odometry_with_localtoglobal_featu.md)
- [\[ECCV 2024\] DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion and Bi-directional Structure Alignment](../../ECCV2024/autonomous_driving/dvlo_deep_visual-lidar_odometry_with_local-to-global_feature_fusion_and_bi-direc.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](../../ICCV2025/autonomous_driving/splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[CVPR 2025\] Towards In-the-Wild 3D Plane Reconstruction from a Single Image](towards_in-the-wild_3d_plane_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] Segment Anything, Even Occluded](segment_anything_even_occluded.md)

<!-- RELATED:END -->
