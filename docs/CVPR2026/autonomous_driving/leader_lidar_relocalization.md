---
title: >-
  [论文解读] LEADER: Learning Reliable Local-to-Global Correspondences for LiDAR Relocalization
description: >-
  [CVPR 2026][自动驾驶][LiDAR重定位] LEADER 通过鲁棒的投影式几何编码器（偏航不变）和截断相对可靠性损失（抑制不可靠点），在 LiDAR 重定位任务上分别实现 24.1% 和 73.9% 的位置误差相对降低。
tags:
  - CVPR 2026
  - 自动驾驶
  - LiDAR重定位
  - 场景坐标回归
  - 偏航不变
  - 可靠性估计
  - 点云
---

# LEADER: Learning Reliable Local-to-Global Correspondences for LiDAR Relocalization

**会议**: CVPR 2026  
**arXiv**: [2604.11355](https://arxiv.org/abs/2604.11355)  
**代码**: [https://github.com/JiansW/LEADER](https://github.com/JiansW/LEADER)  
**领域**: 自动驾驶  
**关键词**: LiDAR重定位, 场景坐标回归, 偏航不变, 可靠性估计, 点云

## 一句话总结
LEADER 通过鲁棒的投影式几何编码器（偏航不变）和截断相对可靠性损失（抑制不可靠点），在 LiDAR 重定位任务上分别实现 24.1% 和 73.9% 的位置误差相对降低。

## 研究背景与动机

**领域现状**：LiDAR 重定位在自动驾驶中至关重要。主流方法分为"检索+配准"（需存储密集点云地图）和基于学习的回归方法，后者又分为绝对位姿回归（APR）和场景坐标回归（SCR）。

**现有痛点**：(1) 检索+配准方法存储和通信开销大；(2) APR 精度有限；(3) 现有 SCR 网络架构不具备偏航不变性，车辆转弯时性能下降；(4) 所有预测点被同等对待，退化区域（无纹理、动态物体）的错误对应点严重干扰位姿估计。

**核心矛盾**：自动驾驶场景中偏航旋转频繁且场景中大量点不适合重定位（动态物体、重复纹理），但现有方法既不能处理旋转又不能区分点的可靠性。

**核心 idea**：设计偏航不变的几何编码器 + 点级可靠性量化，共同提升 SCR 的鲁棒性。

## 方法详解

### 整体框架
原始点云 → 地面平面估计+平面矫正 → 圆柱投影（偏航维度变为平移） → 体素化 → 循环稀疏卷积提取多尺度特征 → 多头最大回归器输出场景坐标+可靠性分数 → 笛卡尔恢复 → 截断相对可靠性损失训练 → 推理时高可靠性点驱动 RANSAC 位姿估计。

### 关键设计

1. **鲁棒投影式几何编码器（RPGE）**:

    - 功能：提取偏航不变的多尺度几何特征
    - 核心思路：将点云通过圆柱投影变换为 $(x^p = s \cdot \arctan2(y', x'), y^p = \sqrt{x'^2 + y'^2}, z^p = z')$，偏航旋转在投影空间中变为 x 方向的平移。体素化后用循环稀疏卷积处理偏航边界的不连续性（边界处的特征循环填充），初始特征仅用距离、高度和反射强度（不含偏航相关坐标）
    - 设计动机：标准卷积在偏航边界处产生不连续特征，循环卷积保证了环形连续性；排除偏航相关坐标保证了初始特征的旋转不变性

2. **截断相对可靠性损失（TRR）**:

    - 功能：建模点级可靠性，抑制退化区域的干扰
    - 核心思路：网络同时预测场景坐标和可靠性分数 $u_i$，通过 arctan 缩放和截断将分数转化为自归一化权重 $w_i$，高可靠性点获得大权重、低可靠性点被抑制。损失 $\mathcal{L}_{TRR} = \sum w_i \mathcal{L}_{raw,i}$
    - 设计动机：场景中不是所有点都适合重定位——动态物体、弱纹理区域的预测本质上不可靠，让网络学会"放弃"这些点而专注于可靠特征

3. **多头最大回归器**:

    - 功能：从编码特征回归场景坐标
    - 核心思路：将 512 维特征投影到 k×512 维（k=4 头），每维取各头最大值，堆叠 5 层这样的操作后接全连接层输出 3D 坐标+可靠性分数
    - 设计动机：多头最大池化增强了特征的鲁棒性和表达力

### 损失函数 / 训练策略
截断相对可靠性损失（TRR）端到端训练。推理时选取 top-k 可靠性分数的点用 RANSAC 估计 6-DoF 位姿。

## 实验关键数据

### 主实验

| 数据集 | 指标 | LEADER | 之前SOTA | 相对降低 |
|--------|------|--------|----------|---------|
| Oxford RobotCar | 位置误差(m) | 0.63 | 0.83 (LightLoc) | -24.1% |
| NCLT | 位置误差(m) | 0.19 | 0.72 (SGLoc→LiSA) | -73.9% |
| Oxford | 方向误差(°) | 1.11 | 1.12 | -0.9% |

### 消融实验

| 配置 | Oxford位置误差 | NCLT位置误差 | 说明 |
|------|--------------|-------------|------|
| Full LEADER | 0.63 | 0.19 | 完整模型 |
| w/o 循环卷积 | 增大 | 增大 | 偏航边界不连续 |
| w/o TRR | 增大 | 增大 | 不可靠点干扰 |
| w/o 投影变换 | 增大 | 增大 | 偏航不变性缺失 |

### 关键发现
- NCLT 数据集上改进最为显著（73.9%），因为 NCLT 包含更多偏航变化和退化区域
- TRR 损失学到的可靠性分数与直觉一致——建筑立面等稳定结构高、地面和植被低
- SCR 方法整体优于 APR 方法，因为显式利用了几何约束

## 亮点与洞察
- **圆柱投影+循环卷积**：将偏航旋转问题优雅地转化为平移等变问题，计算上高效且理论上合理
- **自学习可靠性**：网络自动学会区分可靠/不可靠区域，无需手动标注或语义先验

## 局限与展望
- 仅处理偏航旋转，俯仰和翻滚变化未显式建模
- 可靠性阈值的自适应选择仍可改进
- 未来可结合语义信息进一步提升可靠性估计

## 相关工作与启发
- **vs LiSA**: LiSA 用语义先验区分点贡献，LEADER 用学习的可靠性分数，无需额外语义标注
- **vs RALoc**: RALoc 也处理旋转但方式不同，LEADER 的投影方法更自然

## 评分
- 新颖性: ⭐⭐⭐⭐ 投影变换+可靠性损失的组合简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 两个权威数据集上大幅领先
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰
- 价值: ⭐⭐⭐⭐ 对自动驾驶 LiDAR 定位有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion](../../ECCV2024/autonomous_driving/dvlo_deep_visual-lidar_odometry_with_local-to-global_feature_fusion_and_bi-direc.md)
- [\[CVPR 2026\] BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images](bev-sld_self-supervised_scene_landmark_detection_for_global_localization_with_li.md)
- [\[AAAI 2026\] Beta Distribution Learning for Reliable Roadway Crash Risk Assessment](../../AAAI2026/autonomous_driving/beta_distribution_learning_for_reliable_roadway_crash_risk_a.md)
- [\[ECCV 2024\] DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion and Bi-directional Structure Alignment](../../ECCV2024/autonomous_driving/dvlo_deep_visual-lidar_odometry_with_local-to-global_feature_fusion_and_bi-direc.md)
- [\[CVPR 2026\] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_p.md)

</div>

<!-- RELATED:END -->
