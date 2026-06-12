---
title: >-
  [论文解读] VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation
description: >-
  [CVPR2026][自动驾驶][位姿估计] 提出 VIRD，通过双轴变换（极坐标变换 + 上下文增强位置注意力）构建视图不变表示，在无方向先验条件下实现 SOTA 的跨视角位姿估计，在 KITTI 上位置和方向误差分别降低 50.7% 和 76.5%。
tags:
  - "CVPR2026"
  - "自动驾驶"
  - "位姿估计"
  - "view-invariant representation"
  - "polar transformation"
  - "注意力机制"
---

# VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation

**会议**: CVPR2026  
**arXiv**: [2603.12918](https://arxiv.org/abs/2603.12918)  
**代码**: 待确认  
**领域**: 自动驾驶  
**关键词**: cross-view pose estimation, view-invariant representation, polar transformation, positional attention, autonomous driving localization

## 一句话总结

提出 VIRD，通过双轴变换（极坐标变换 + 上下文增强位置注意力）构建视图不变表示，在无方向先验条件下实现 SOTA 的跨视角位姿估计，在 KITTI 上位置和方向误差分别降低 50.7% 和 76.5%。

## 研究背景与动机

**全局定位是自动驾驶的关键需求**：准确的全局定位对自动驾驶和移动机器人至关重要，是实现真实世界导航的基础能力。

**GNSS 在城市场景中不可靠**：在密集城市区域中，GNSS 信号因遮挡和多径效应严重退化，定位精度大幅下降。

**跨视角位姿估计作为替代方案**：利用地理参考的卫星图像估计地面相机的 3-DoF 位姿是有前景的替代方案，但地面视图与卫星视图之间存在巨大视角差异。

**现有方法依赖方向先验**：早期方法假设已知粗略方向，在窄搜索空间内迭代优化，但实际中方向先验常不准确或不可用，导致收敛到次优解。

**语义方法忽略空间对应**：近期全方向 CVPE 方法通过语义相似性（交叉注意力、对比学习）缩小视角差距，但忽略了空间对应关系，视角差距未根本解决。

**几何变换方法各有缺陷**：极坐标变换仅解决水平对齐忽略垂直轴，投影变换依赖相机参数且在垂直结构（如建筑物）周围产生严重伪影。

## 方法详解

### 整体框架

VIRD 是一个全方向跨视角位姿估计框架，核心是用"双轴变换"把地面图和卫星图统一到同一个视图不变描述子上。先用预训练 CNN(VGG16 / EfficientNet-B0)分别提取地面特征 $F_g \in \mathbb{R}^{C \times H \times W_g}$ 和卫星特征 $F_s \in \mathbb{R}^{C \times A \times A}$；接着分两步对齐——极坐标变换把卫星特征的方位角映射到水平轴(水平对齐)，上下文增强位置注意力(CEPA)再消除垂直轴的不对齐（CEPA 建立在位置注意力 PA 之上）；对齐后的特征沿垂直方向压缩、展平成方向感知的 1D 描述子 $D_g$ 和 $D_{s2p}$，先用余弦相似度匹配拿到粗位姿，再由回归模块预测残差精化。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    GN["地面图像 → CNN 特征 F_g"]
    SN["卫星图像 → CNN 特征 F_s"]
    SN --> POLAR["极坐标变换<br/>方位角→水平轴(水平对齐)"]
    subgraph V["垂直轴对齐"]
        direction TB
        PA["位置注意力(PA)<br/>共享虚拟轴学垂直坐标映射,免相机参数"] --> CEPA["上下文增强位置注意力(CEPA)<br/>用地面上下文按方向自适应"]
    end
    POLAR --> PA
    GN --> CEPA
    GN --> DESC
    CEPA --> DESC["垂直压缩+展平<br/>1D 视图不变描述子 D_g / D_s2p"]
    DESC -->|训练| RECON["视图重建损失<br/>原始+交叉重建,逼描述子编码垂直结构"]
    DESC --> MATCH["余弦相似度匹配 → 粗位姿"]
    MATCH --> REG["回归模块预测残差 → 3-DoF 位姿"]
```

### 关键设计

**1. 极坐标变换：解决水平轴对齐**

跨视角的根本难点是地面视图和卫星视图差异巨大，得先把方位关系对上。VIRD 以候选位置为中心对卫星特征图做极坐标采样，把方位角和径向距离分别映射到水平和垂直轴。变换宽度 $W_s = \frac{2\pi}{\text{HFoV}} \cdot W_g$ 确保不同视场角下的尺度一致。这一步只管水平方向，把"转了多少角度"对齐，但垂直方向的结构对应还没解决。

**2. 位置注意力(PA)：不靠相机参数学垂直轴映射**

垂直轴对齐若走投影变换就要依赖相机参数和深度、还会在建筑物等垂直结构周围产生伪影。PA 改用注意力来学：定义三组正弦位置编码——共享虚拟编码 $P_a$、地面编码 $P_g$、卫星编码 $P_{s2p}$，通过注意力学习垂直坐标间的映射

$$\mathcal{A}_v = \text{Softmax}\left(\frac{(P_a W_v^Q)(P_v W_v^K)^\top}{\sqrt{d_k}}\right)$$

关键是把位置注意力重新解释成"通过一根共享虚拟轴学习跨视图的垂直坐标变换"，完全不需要相机参数。

**3. 上下文增强位置注意力(CEPA)：让垂直变换随场景内容自适应**

PA 有个隐含假设——所有水平方向的垂直变换都一致，但实际不同方向上的垂直结构(建筑高低、地形起伏)千差万别。CEPA 用地面特征的局部上下文来增强注意力权重

$$\mathcal{A}_{g'} = \mathcal{A}_g + \text{Softmax}\left(\Phi(\mathcal{A}_g \oplus F_g)\right)$$

卷积层 $\Phi$ 处理拼接后的注意力权重和特征，使模型能按场景上下文自适应地变换地面特征，而不是对所有方向一刀切。

**4. 视图重建损失：逼描述子真正编码垂直结构**

光对齐还不够，描述子得真的把视图不变的结构信息学进去。训练时要求每个描述子同时重建原始视图和交叉视图图像——原始重建损失 $\mathcal{L}_{\text{origin}}$ 保证保真，交叉重建损失 $\mathcal{L}_{\text{cross}}$ 强制描述子捕获跨视图共享的垂直结构，从而提升视图不变性。消融显示这一项对方向估计贡献最大。

### 损失函数

总损失由三部分组成：

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{match}} + \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{recon}} = \alpha_1 \mathcal{L}_{\text{origin}} + \alpha_2 \mathcal{L}_{\text{cross}}$：视图重建损失（$\ell_1$-loss）
- $\mathcal{L}_{\text{match}}$：InfoNCE 匹配损失，鼓励 GT 位姿处的高相似度
- $\mathcal{L}_{\text{reg}}$：位姿残差的 $\ell_2$-loss

## 实验

### 主要结果

**KITTI 数据集**（无方向先验，跨区域设置，EfficientNet-B0）：

| 方法 | 位置均值(m)↓ | 位置中值(m)↓ | 方向均值(°)↓ | 方向中值(°)↓ |
|------|:---:|:---:|:---:|:---:|
| HighlyAccurate | 15.50 | 16.02 | 89.84 | 89.85 |
| SliceMatch | 14.85 | 11.85 | 23.64 | 7.96 |
| CCVPE | 13.94 | 10.98 | 77.84 | 63.84 |
| FG2 | 13.58 | 11.72 | 90.12 | 90.42 |
| **VIRD (Ours)** | **11.12** | **5.41** | **22.03** | **1.87** |

**VIGOR 数据集**（无方向对齐，跨区域设置，EfficientNet-B0）：

| 方法 | 位置均值(m)↓ | 位置中值(m)↓ | 方向均值(°)↓ | 方向中值(°)↓ |
|------|:---:|:---:|:---:|:---:|
| CCVPE | 5.41 | 1.89 | 27.78 | 13.58 |
| DenseFlow | 7.67 | 3.67 | 17.63 | 2.94 |
| FG2† | 5.95 | 2.40 | 28.41 | 2.20 |
| **VIRD (Ours)** | **4.61** | **1.55** | **16.50** | **1.17** |

### 消融实验

**双轴变换消融**（KITTI 跨区域，VGG16）：

| 变换策略 | 位置中值(m) | 方向中值(°) |
|----------|:---:|:---:|
| 投影变换 S2G | 10.59 | 3.84 |
| 投影变换 G2S | 15.20 | 5.44 |
| 仅极坐标 | 11.75 | 4.00 |
| 极坐标 + PA | 9.76 | 3.44 |
| 极坐标 + CEPA | **8.88** | **3.36** |

**模型组件消融**（KITTI 跨区域，VGG16）：

| 配置 | 位置中值(m) | 方向中值(°) |
|------|:---:|:---:|
| Polar + CEPA | 8.88 | 3.36 |
| + $\mathcal{L}_{\text{origin}}$ | 8.29 | 3.31 |
| + $\mathcal{L}_{\text{cross}}$ | 8.10 | 3.21 |
| + 两者 | 7.90 | 3.05 |
| + 两者 + 回归 | **7.05** | **2.22** |

### 关键发现

- 双轴变换显著优于所有单一几何变换基线，PA 和 CEPA 的逐步引入均带来一致提升
- 视图重建损失对方向估计贡献最大，均值方向误差降低 39.1%；交叉重建比原始重建更有效
- 回归模块将位置中值降低 0.85m、方向中值降低 0.83°，但方向均值略升（因反方向预测被精化得更远）
- VIRD 在不同方向噪声水平（±10° 到 ±180°）下均保持低误差，鲁棒性显著优于 CCVPE 和 FG2

## 亮点

- **双轴变换策略设计精巧**：将跨视角匹配的难题分解为水平（极坐标变换）和垂直（位置注意力）两个可处理的子问题，避免了投影变换对相机参数和深度信息的依赖
- **CEPA 模块直觉清晰有效**：将位置注意力重新解释为通过共享虚拟轴的坐标变换，再通过上下文增强适应不同方向的垂直结构变化
- **视图重建损失设计巧妙**：通过要求描述子重建原始和交叉视图来强制视图不变性，同时帮助模型聚焦于跨视图共享结构
- **无需方向先验**：在全 360° 搜索空间下实现 SOTA，实际应用价值高

## 局限性

- 仅处理 3-DoF 位姿（x, y, yaw），假设 pitch 和 roll 可忽略，在复杂地形中可能不成立
- 在 VIGOR 有对齐方向的设置下，位置精度不如 FG2 等利用 3D 结构信息的方法
- 回归模块在粗匹配方向错误时可能放大误差（反方向问题）
- 仅在 KITTI 和 VIGOR 两个数据集上验证，泛化到更多城市场景和国家的能力未知
- 训练需要在每个候选位姿上进行极坐标变换，候选数量增大时计算开销可能较高

## 相关工作

- **有限角度 CVPE**：HighlyAccurate、PIDLoc 等使用 LM 算法或神经位姿优化器在窄角度范围内迭代优化，受限于方向先验
- **全方向 CVPE**：SliceMatch 使用基于内容的交叉注意力，CCVPE 使用对比学习，但都侧重语义相似性而忽略空间对应
- **几何变换**：极坐标变换（Shi & Li 等）解决水平对齐；投影变换（HighlyAccurate 等）尝试同时处理两轴但依赖相机参数且有伪影
- **FG2**：利用高度感知的 3D 点选择缩小视角差距，但稀疏点匹配导致方向估计退化

## 评分

- 新颖性: ⭐⭐⭐⭐ — 双轴变换策略和 CEPA 模块设计新颖，对位置注意力的重解释有启发性
- 实验充分度: ⭐⭐⭐⭐ — 两个主流数据集全面对比，消融完整，包括鲁棒性分析和可视化
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，图示直观
- 价值: ⭐⭐⭐⭐ — 无方向先验下 SOTA，对实际自动驾驶定位有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)
- [\[CVPR 2026\] CycleBEV: Regularizing View Transformation Networks via View Cycle Consistency for Bird's-Eye-View Semantic Segmentation](cyclebev_regularizing_view_transformation_networks_via_view_cycle_consistency_fo.md)
- [\[CVPR 2026\] IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration](igasa_integrated_geometry-aware_and_skip-attention_modules_for_enhanced_point_cl.md)
- [\[CVPR 2026\] CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection](coin3d_revisiting_configuration-invariant_multi-camera_3d_object_detection.md)
- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)

</div>

<!-- RELATED:END -->
