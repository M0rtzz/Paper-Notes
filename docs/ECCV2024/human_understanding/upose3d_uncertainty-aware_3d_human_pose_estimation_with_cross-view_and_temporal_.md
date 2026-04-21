---
title: >-
  [论文解读] UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues
description: >-
  [ECCV 2024][人体理解][多视角3D姿态估计] 提出UPose3D，一种基于不确定性感知的多视角3D人体姿态估计方法，通过Normalizing Flow建模2D关键点不确定性、可扩展的跨视角点云投影融合策略和合成数据训练的Pose Compiler模块，在无需3D标注的情况下取得OoD场景下SOTA表现，且在InD场景下与使用3D监督的方法竞争。
tags:
  - ECCV 2024
  - 人体理解
  - 多视角3D姿态估计
  - 不确定性建模
  - 跨视角融合
  - 时序建模
  - 合成数据
---

# UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues

**会议**: ECCV 2024  
**arXiv**: [2404.14634](https://arxiv.org/abs/2404.14634)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: 多视角3D姿态估计, 不确定性建模, 跨视角融合, 时序建模, 合成数据

## 一句话总结
提出UPose3D，一种基于不确定性感知的多视角3D人体姿态估计方法，通过Normalizing Flow建模2D关键点不确定性、可扩展的跨视角点云投影融合策略和合成数据训练的Pose Compiler模块，在无需3D标注的情况下取得OoD场景下SOTA表现，且在InD场景下与使用3D监督的方法竞争。

## 研究背景与动机
多视角3D人体姿态估计在无标记动作捕捉中至关重要，特别是在游戏和电影制作中需要亚厘米级精度。传统方法先独立预测各视角2D关键点，再通过三角化得到3D坐标，但严重依赖各视角2D预测的精度，在遮挡和复杂肢体交互时表现不佳。RANSAC等异常值缓解技术只能提供有限的鲁棒性。

核心矛盾有三：(1) **可扩展性差**：现有跨视角融合方法（如Epipolar Transformers）随摄像机数量增加计算复杂度急剧上升；(2) **依赖3D标注**：端到端方法（如MTF-Transformer）需要大量3D标注数据训练，这在户外和野外场景极其稀缺；(3) **泛化性弱**：训练在固定室内场景的模型难以推广到新的摄像机配置和环境。

本文的核心idea：**利用两个来源的2D关键点不确定性（图像级RLE + Pose Compiler精炼后）来增强鲁棒性，通过合成训练数据避免对3D标注的依赖，通过极线几何投影实现对任意摄像机数量的可扩展融合**。

## 方法详解

### 整体框架
UPose3D由四个阶段组成：(1) RLE-based 2D姿态估计器从每个视角提取关键点μ̂和不确定性σ̂；(2) 跨视角投影：利用极线几何将各视角关键点投影到参考视角形成2D点云；(3) Pose Compiler模块利用跨视角和时序信息精炼关键点位置μ̂'和不确定性σ̂'；(4) 利用两阶段的预测通过MLE迭代优化得到最终3D姿态。

### 关键设计
1. **基于RLE的不确定性建模**: 

    - 使用Residual Log-likelihood Estimation (RLE) 头建模关键点分布P_Θ(x|I)
    - RLE通过normalizing flow学习关键点的概率分布，同时输出位置预测μ̂和不确定性σ̂
    - 相比heatmap方法，RLE计算效率更高且可微分，能直接用于后续MLE优化
    - 设计动机：不确定性提供了每个预测的可信度，使系统能自动discountch低置信度的预测，增强对遮挡和噪声的鲁棒性

2. **可扩展的跨视角点云融合**: 

    - 对每个关键点k，利用基础矩阵F_ij将视角j的预测投影到参考视角i的极线上，找到极线上最近点作为投影结果
    - 对所有视角重复此过程，形成2D点云C_{i,k,t}
    - 使用基于Point Cloud Transformer的编码器处理点云，保留坐标信息的残差连接
    - 关键优势：无论摄像机数量多少，每个参考视角只需处理一个点云，计算量与视角数线性相关而非二次方
    - 设计动机：Epipolar Transformer需要对每对视角做特征融合，扩展到更多摄像机代价巨大

3. **Pose Compiler模块**: 

    - 点云编码器：4层multi-head self-attention，hidden dim 64，提取跨视角特征
    - 时空编码器：4层criss-cross transformer块，分别沿时间轴和关节轴做注意力，近似全时空依赖同时节省内存
    - 输出通过RLE头提供精炼后的位置μ̂'和不确定性σ̂'
    - 训练数据：从AMASS动捕数据集在线合成多视角训练数据（随机摆放摄像机、投影3D关键点、添加噪声增augmentation）
    - 设计动机：criss-cross注意力比全注意力内存效率更高，而合成训练数据策略使模型不受真实多视角3D数据集的限制

### 损失函数 / 训练策略
最大似然估计(MLE)损失函数：
$$\mathcal{L}_{mle} = -\log \prod_{i \in V} P_\Theta(u_i | \mathcal{I})|_{u_i = \hat{\mu}_i} - \log \prod_{i \in V} P_{\Theta'}(u_i | \mathcal{C})|_{u_i = \hat{\mu}'_i}$$

其中u_i是3D变量U在各视角的投影。通过DLT初始化U，使用L-BFGS迭代优化。这种方式无需3D标注即可训练——通过最大化投影似然间接对齐3D姿态。

训练策略：
- 2D姿态估计器在Human3.6m+MPII上微调
- Pose Compiler在AMASS合成数据上训练：AdamW优化器，lr 4e-5，warm-up + cosine annealing，约6小时在NVIDIA 2080 RTX
- 在线数据合成：随机摆放最多8个摄像机，随机旋转和镜像增强运动数据

## 实验关键数据

### 主实验
Human3.6m数据集InD结果对比(mm)：

| 方法 | 监督类型 | Backbone | Frames | MPJPE↓ | PA-MPJPE↓ |
|------|---------|----------|--------|--------|-----------|
| Learnable Triangulation | 3D | ResNet152 | 1 | 20.7 | 17.0 |
| AdaFuse | 3D | ResNet152 | 1 | 19.5 | - |
| MTF-Transformer | 3D | CPN | 27 | 25.8 | - |
| DLT | 2D | CPN | 1 | 30.5 | 27.6 |
| **UPose3D** | **2D** | **CPN** | **1** | **26.9** | **24.1** |
| **UPose3D** | **2D** | **CPN** | **27** | **26.4** | **23.4** |

RICH数据集OoD结果对比(mm)：

| 方法 | 训练源 | MPJPE↓ | PA-MPJPE↓ |
|------|--------|--------|-----------|
| AdaFuse | (H3.6m+MPII, H3.6m) | 524.0 | 85.8 |
| UPose3D (T=27) | (H3.6m+MPII, H3.6m) | 51.8 | 43.6 |
| HRNet-W48+DLT | (COCO, N/A) | 66.0 | 55.1 |
| **UPose3D (T=27)** | **(COCO, AMASS)** | **34.7** | **32.0** |

### 消融实验
Human3.6m数据集消融(T=27)：

| 配置 | MPJPE↓ | PA-MPJPE↓ | 说明 |
|------|--------|-----------|------|
| UPose3D (完整) | 26.42 | 23.42 | - |
| w/o compiler | 37.14 | 33.90 | +10.7mm，compiler至关重要 |
| w/o image branch | 69.90 | 50.97 | 仅用compiler，缺少原始预测 |
| w/o compiler uncertainty | 26.42 | 23.58 | 推理时去除compiler不确定性 |
| w/o image uncertainty | 27.61 | 24.88 | +1.2mm，图像不确定性重要 |
| w/o uncertainty (全部) | 48.11 | 41.20 | +21.7mm，退化为简单三角化 |

计算效率对比：

| 方法 | Params(M)↓ | FLOPs-4cam(G)↓ | FLOPs-10cam(G)↓ |
|------|-----------|----------------|-----------------|
| Learnable Triangulation | 80.6 | 716.1 | 1326.9 |
| Epipolar Transformers | 68.1 | 406.5 | 1016.2 |
| AdaFuse | 69.7 | 595.0 | 1487.6 |
| **UPose3D** | **65.4** | **208.7** | **517.7** |

### 关键发现
- **不确定性建模是核心**：完全去除不确定性后MPJPE暴增21.7mm，说明不确定性是区分低质量预测的关键
- **OoD表现惊艳**：在RICH数据集上，AdaFuse的MPJPE高达524mm，而UPose3D仅34.7mm——差距超过10倍
- AdaFuse在OoD场景失败是因为它需要所有视角预测都在合理范围内，单个噪声视角就会导致巨大三角化误差
- **可扩展性优秀**：UPose3D的FLOPs在10个摄像机时仅为其他方法的35-50%，推理时间几乎不随摄像机数量增加
- 2D姿态估计器的选择至关重要：CPN明显优于ResNet152，说明backbone质量是基础
- 时序窗口有帮助但增益有限：27帧比1帧仅降低0.5mm MPJPE

## 亮点与洞察
- **不确定性的双重利用**: 同时从图像分支和Pose Compiler分支获取不确定性，在MLE优化中自然地加权各预测的可信度，是一个优雅的框架设计
- **合成数据训练策略**: 完全避免了对真实3D标注数据的依赖，从AMASS动捕数据在线合成多视角训练数据，实现了跨数据集的泛化
- **极线几何点云化**: 将跨视角问题转化为点云处理问题，是一个颇有创意的抽象
- **一个反直觉的发现**: 简单DLT三角化配合强2D预测器(CPN)就能超越一些复杂方法（如AdaFuse+noisy views），说明基础很重要
- 计算效率优势随摄像机数量增加而愈发显著

## 局限与展望
- MLE优化阶段使用L-BFGS，计算成本有波动且不适合实时应用
- 合成训练数据的多样性仍受限于AMASS数据集中的动作类型
- 未涉及多人场景——仅处理单人多视角
- criss-cross注意力虽然高效但相对简化了全时空建模
- 未来可探索：(1)用深度网络估计Hessian矩阵加速MLE，(2)引入场景深度和轨迹等额外模态

## 相关工作与启发
- RLE (Residual Log-likelihood Estimation) 的关键点不确定性建模对后续3D重建工作有启发
- 跨视角点云投影融合策略可推广到任何多视角几何问题
- 合成数据训练 + 真实数据推理的范式在多视角姿态估计中被验证有效
- 与AdaFuse的对比揭示了一个重要教训：在InD表现优秀的方法可能在OoD场景完全崩溃，鲁棒性评估至关重要
- 对markerless动作捕捉行业有直接应用价值

## 评分
- 新颖性: ⭐⭐⭐⭐ (跨视角点云融合和双源不确定性建模有新意)
- 实验充分度: ⭐⭐⭐⭐⭐ (InD/OoD均有评估，消融详尽，可扩展性分析全面)
- 写作质量: ⭐⭐⭐⭐ (结构清晰但部分符号较多，阅读门槛稍高)
- 价值: ⭐⭐⭐⭐⭐ (解决了实际问题：无需3D标注+OoD鲁棒+可扩展)

<!-- RELATED:START -->

## 相关论文

- [3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)
- [WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)
- [ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation](aden_adaptive_density_representations_for_sparseview_camera.md)

<!-- RELATED:END -->
