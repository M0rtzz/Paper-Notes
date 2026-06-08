---
title: >-
  [论文解读] InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset
description: >-
  [CVPR 2026][自动驾驶][相机位姿估计] 本文提出 InCaRPose，一个基于冻结 ViT 骨干和 Transformer 解码器的车内相对相机位姿估计模型，仅在合成数据上训练即可泛化到真实车内环境，实现绝对度量级翻译预测和实时推理（>45 FPS）…
tags:
  - "CVPR 2026"
  - "自动驾驶"
  - "相机位姿估计"
  - "车内感知"
  - "鱼眼相机"
  - "合成到真实迁移"
  - "Transformer"
---

# InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset

**会议**: CVPR 2026  
**arXiv**: [2604.03814](https://arxiv.org/abs/2604.03814)  
**代码**: [https://github.com/felixstillger/InCaRPose](https://github.com/felixstillger/InCaRPose)  
**领域**: 自动驾驶 / 3D视觉  
**关键词**: 相机位姿估计, 车内感知, 鱼眼相机, 合成到真实迁移, Transformer

## 一句话总结

本文提出 InCaRPose，一个基于冻结 ViT 骨干和 Transformer 解码器的车内相对相机位姿估计模型，仅在合成数据上训练即可泛化到真实车内环境，实现绝对度量级翻译预测和实时推理（>45 FPS），同时发布了配套的真实世界高畸变车内测试数据集 In-Cabin-Pose。

## 研究背景与动机

**领域现状**：相机外参标定是计算机视觉的基础任务。在汽车车内监控（ICAM）场景中，相机用于驾驶员监控、乘员姿态感知和安全气囊控制。现有方法依赖几何特征匹配和对极几何，或者大规模训练的深度学习模型。

**现有痛点**：车内环境有几个独特挑战：(1) 后视镜安装的相机会随驾驶员调节而频繁改变外参；(2) 车内相机通常使用广角/鱼眼镜头，带来严重畸变；(3) 相机工作在近红外（NIR）光谱，与常见 RGB 图像不同；(4) 安全气囊控制需要在碰撞后 15-50ms 内获得乘员位置，需要绝对度量级的翻译估计和实时推理。现有的 Reloc3r 等方法只能预测翻译方向而非绝对距离，且需要大规模训练数据。

**核心矛盾**：现有通用位姿估计模型要么需要大量训练数据和特定相机内参，要么只提供无尺度的翻译方向，无法满足车内安全应用对"小数据训练 + 绝对度量翻译 + 鱼眼畸变处理 + 实时推理"的综合需求。

**本文目标** (1) 仅用合成数据训练就能在真实车内环境中准确估计相对位姿；(2) 直接处理高畸变鱼眼图像，无需去畸变；(3) 预测绝对度量级翻译；(4) 实时推理以支持时间敏感的安全应用。

**切入角度**：将问题重新表述为参考-相对位姿估计（reference-relative pose estimation），避免依赖特定车辆坐标系。利用冻结的自监督 ViT 骨干（DINOv3）提取域不变特征，以小量合成数据训练即可跨域迁移。

**核心 idea**：用冻结 DINOv3 骨干 + Transformer 交叉注意力解码器 + 轻量预测头，在合成车内数据上训练实现真实车内的度量级相对位姿估计。

## 方法详解

### 整体框架

输入为参考视图和目标视图两张图像，通过冻结的 ViT 骨干提取 patch 级特征，送入 Transformer 交叉注意力解码器融合两视图信息，最后通过 MLP 预测头回归相对旋转和翻译。骨干完全冻结，仅训练解码器和预测头。可选地预测双向位姿以增强几何一致性监督。

### 关键设计

**1. 冻结 ViT 骨干 + 参考-相对公式化：用预训练特征绕过"小数据 + 跨域"双重困境。**

车内场景的尴尬在于：合成数据好造但和真实 NIR 鱼眼图像有域差，真实数据又稀缺到不够从头训练一个位姿网络。InCaRPose 的解法是把特征提取整个交给一个冻结的自监督 ViT——DINOv3（也支持 DINOv2、DUNE），对参考视图和目标视图分别抽 patch 级特征，骨干参数全程不更新。这么做有两层考量：一是 DINOv3 的预训练特征本身就跨域鲁棒，合成图和真实图在它眼里差别没那么大，迁移自然就发生了；二是冻结骨干能挡住训练早期随机初始化的解码器/预测头反传回来的噪声梯度，避免这些噪声把骨干好不容易学到的精细特征冲坏。

另一半设计是把任务从"估计绝对位姿"改写成参考-相对位姿。模型不去预测某个全局车辆坐标系下的姿态，而是给定一个标定好的参考位姿 $T_{v1}$，只回归相对变换 $T_{rel}$，使 $T_{v2} = T_{v1} \cdot T_{rel}$，训练标签即 $T_{rel} = T_{v1}^{-1} T_{v2}$。好处是它彻底和具体车型的坐标系解耦——换一辆车只需换参考位姿，同一个模型直接复用，不必为每种车重新训练，这对量产部署是关键。

**2. Transformer 交叉注意力解码器 + 2D RoPE：在小数据下把两视图的几何关系学稳。**

两视图的特征抽出来后，真正决定位姿的是它们之间的空间对应关系，这正是交叉注意力擅长的事。骨干 tokens 先经线性投影，再过多层解码器块，每块里自注意力负责精炼单视图特征、交叉注意力负责让一个视图去关注另一个视图的对应区域，配 LayerNorm 和残差连接稳住训练（用了 12 个注意力头、MLP 扩展比 4）。位置信息上，它没有去学一套位置 token，而是用 2D RoPE 把空间位置以旋转的形式注入 query 和 key——因为训练数据有限时，学出来的位置编码容易过拟合，而 RoPE 是固定的几何先验，外推和泛化都更稳。降维和特征融合则交给残差卷积瓶颈。

**3. 多输出表示 + 双向预测：让网络从一致性约束里榨出额外监督。**

旋转该用什么形式回归并不是无关紧要的。模型支持五种参数化——旋转向量、内蕴/外蕴欧拉角（6D）、四元数（7D）、旋转矩阵（12D）——并在后处理里强制输出是合法旋转：四元数做归一化，旋转矩阵走 SVD 正交化并锁定 $\det = +1$；实验下来四元数表现最好。更关键的是训练时打开双向预测：网络同时回归正向和反向的相对位姿，正反两个方向理应互逆，这个一致性本身就是一道免费的监督信号，逼网络学到自洽的双向变换关系，也省去了对图像顺序做数据增强。推理时这一支可以关掉，直接换来更高的吞吐。

### 损失函数 / 训练策略

使用 AdamW 优化器，学习率 $1 \times 10^{-6}$，权重衰减 $1 \times 10^{-5}$，batch size 8。合成训练数据来自 Blender 渲染的 11 辆不同车辆（8 训练 3 验证），随机放置人偶和物品，均匀采样旋转（±80° x/y, ±50° z）和翻译（±20cm 各轴），约 5000 纯旋转对和 1500 旋转+翻译对。使用 ColorJitter 数据增强防止过拟合，鱼眼图像直接处理不去畸变（缩放+零填充保持完整视场角）。

## 实验关键数据

### 主实验

**In-Cabin-Pose 真实数据集**:

| 模型 | 旋转误差 (°) Median | 翻译误差 (m) Median | 方向误差 (°) Median |
|------|---------------------|---------------------|---------------------|
| InCaRPose-Small224 | 4.43 | 0.08 | 37.74 |
| InCaRPose-Base224 | 3.55 | 0.09 | 42.45 |
| **InCaRPose-Large224** | **2.75** | **0.07** | **23.46** |
| Reloc3r224 (未去畸变) | 12.73 | – | 76.79 |
| Reloc3r512 (去畸变) | 3.23 | – | 13.05 |
| SIFT Matching (去畸变) | 4.83 | – | 28.30 |

**7-Scenes 室内数据集**:

| 模型 | 旋转误差 (°) Mean | 翻译误差 (m) Median |
|------|-------------------|---------------------|
| RelPoseNet | 9.30 | 0.21 |
| Relformer | 6.27 | 0.18 |
| RelPoseGNN | 5.20 | 0.17 |
| Reloc3r224 | 7.96 | – |
| **InCaRPose-Large224** | **2.55** | **0.13** |

### 消融实验

| 配置 | 旋转 Mean (°) | 翻译 Mean (m) | 说明 |
|------|---------------|---------------|------|
| InCaRPose-Small | 6.11 | 0.11 | 最快但精度较低 |
| InCaRPose-Base | 4.91 | 0.12 | 中等 |
| InCaRPose-Large | **4.15** | **0.10** | 最佳精度 |
| DINOv3-Base | 4.91 | 0.12 | 标准骨干 |
| DUNE-Base504 | **3.87** | 0.12 | DUNE 骨干略优 |

**推理速度（RTX 4090 单卡）**:

| 配置 | FPS |
|------|-----|
| InCaRPose-Small224 | ~70 |
| InCaRPose-Base224 | ~67 |
| InCaRPose-Large224 | >45 |

### 关键发现

- InCaRPose-Large 在真实车内数据上旋转误差仅 2.75°，翻译误差 0.07m，在仅使用合成训练数据的情况下实现了出色的合成到真实迁移
- 在 7-Scenes 上平均旋转误差 2.55°，比 Reloc3r 的 7.96° 低 65%，且提供度量翻译
- 冻结骨干是关键：DINOv3 的预训练特征具有强跨域泛化能力
- 更大骨干对高畸变图像的改善更显著（车内数据 small→large 从 6.11° 降到 4.15°），而在 7-Scenes 标准图像上差异较小
- 所有配置都保持实时性能（>45 FPS），Small/Base 接近 70 FPS

## 亮点与洞察

- **合成数据训练+真实泛化**：仅用约 6500 合成图像对就在真实车内环境中取得了优秀表现，这得益于冻结 DINOv3 骨干的域不变特征。这种"冻结基础模型 + 轻量任务头"的模式在数据稀缺场景下极具参考价值
- **端到端鱼眼处理**：不去畸变直接处理鱼眼图像是务实的设计——边缘区域的几何线索对位姿估计很重要，去畸变会丢失信息。且在实际部署中省去了去畸变的计算开销
- **参考-相对公式化的通用性**：避免了车辆特定坐标系的问题，使同一个模型可以跨车型部署，这对量产级应用非常重要

## 局限与展望

- 翻译方向误差仍然较大（中值 23.46°），特别是 z 轴方向的极端运动下翻译估计仍具挑战性
- 真实测试数据集仅来自单一车辆内部，跨车型泛化能力有待更多验证
- 当前支持的翻译范围限于车内摄像头的调整范围（±20cm），大范围位移场景未验证
- 未来可以探索多帧时序信息的利用或与 IMU 的融合

## 相关工作与启发

- **vs Reloc3r（大规模训练位姿估计）**: Reloc3r 在大量数据上训练但只预测翻译方向，无法提供度量距离。InCaRPose 数据量小得多但提供度量翻译，更适合安全应用
- **vs SIFT + RANSAC（经典方法）**: 经典方法在畸变图像上性能退化且存在尺度歧义。InCaRPose 直接端到端处理畸变图像
- **vs PoseNet / SCR（绝对位姿方法）**: 绝对方法需要场景特定训练或密集 3D 重建。InCaRPose 的参考-相对方法更灵活

## 评分

- 新颖性: ⭐⭐⭐ 方法层面创新有限（冻结骨干+Transformer+MLP），贡献主要在问题定义和数据集
- 实验充分度: ⭐⭐⭐⭐ 真实车内数据集、7-Scenes、Cambridge Landmarks 三个数据集验证，速度分析详细
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清晰，工程细节充分
- 价值: ⭐⭐⭐⭐ 数据集和问题定义对车内感知社区有直接价值，满足实际安全需求

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)
- [\[CVPR 2026\] EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets](expanding_mmwave_datasets_for_human_pose_estimation_with_unlabeled_data_and_lida.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)
- [\[CVPR 2026\] VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation](vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)
- [\[CVPR 2025\] LiSu: A Dataset and Method for LiDAR Surface Normal Estimation](../../CVPR2025/autonomous_driving/lisu_a_dataset_and_method_for_lidar_surface_normal_estimation.md)

</div>

<!-- RELATED:END -->
