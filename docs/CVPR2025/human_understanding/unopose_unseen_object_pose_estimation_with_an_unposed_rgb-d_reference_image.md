---
title: >-
  [论文解读] UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image
description: >-
  [CVPR 2025][人体理解][未知物体位姿估计] 提出 UNOPose 方法和基准，仅使用单张无位姿的 RGB-D 参考图像即可估计未知物体的 6DoF 相对位姿，通过 $SE(3)$ 不变参考坐标系和重叠感知匹配实现了与依赖 CAD 模型方法相当的性能。 - 现有物体位姿估计方法大多依赖 CAD 模型或多张参考视图来…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "未知物体位姿估计"
  - "单参考图像"
  - "SE(3)不变性"
  - "点云配准"
  - "重叠预测"
---

# UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image

**会议**: CVPR 2025  
**arXiv**: [2411.16106](https://arxiv.org/abs/2411.16106)  
**代码**: [GitHub](https://github.com/shanice-l/UNOPose)  
**领域**: 人体理解  
**关键词**: 未知物体位姿估计, 单参考图像, SE(3)不变性, 点云配准, 重叠预测

## 一句话总结

提出 UNOPose 方法和基准，仅使用单张无位姿的 RGB-D 参考图像即可估计未知物体的 6DoF 相对位姿，通过 $SE(3)$ 不变参考坐标系和重叠感知匹配实现了与依赖 CAD 模型方法相当的性能。

## 研究背景与动机

- 现有物体位姿估计方法大多依赖 CAD 模型或多张参考视图来覆盖目标物体的外观，标注和准备成本高
- 实例级和类别级方法只能处理已知物体/已知类别，在开放世界应用中有明显局限
- 单张参考图像的设定面临巨大挑战：相对位姿可在整个 $SE(3)$ 空间变化，不再有多视图选择最近锚点的简化
- 遮挡、传感器噪声和极端几何形状可能导致视点间重叠区域很小
- 已有的相对位姿估计方法（如 3DAHV、DVMNet）仅使用 RGB 模态预测 3DoF 旋转，无法估计完整的 6DoF 位姿（包含平移）
- 需要一种低成本、仅需单张 RGB-D 参考即可工作的通用位姿估计方案

## 方法详解

### 整体框架

UNOPose 采用从粗到精的范式：（1）利用 SAM+DINOv2 进行未知物体分割，从查询图像中定位目标物体；（2）将 RGB-D 图像反投影为 3D 点云，通过 $SE(3)$ 不变全局参考坐标系（GRF）标准化物体表示后，进行粗匹配获取初始位姿估计；（3）在初始对齐后的密集点云上进行精细匹配，利用局部参考坐标系（LRF）编码捕获细粒度几何结构，最终通过 RANSAC 求解精确位姿。

### 关键设计

**1. $SE(3)$ 不变全局参考坐标系（GRF）**

- **功能**：消除物体位姿和尺度变化对匹配的影响，标准化物体表示
- **核心思路**：通过7DoF变换 $\{\mathbf{R}_G, \mathbf{t}_G, s_G\}$ 将点云转到标准坐标系。原点设为物体中心（平移不变），半径归一化为1（尺度不变），旋转由物体中心法向量（SVD 最小奇异值对应向量）确定 z 轴，投影到切平面的加权向量和确定 x 轴
- **设计动机**：单参考图设定下相对位姿可覆盖整个 $SE(3)$ 空间，需要首先消除位姿和尺度变化才能有效建立对应关系。相比需要复杂网络或 PPF 特征的方法，GRF 变换计算高效

**2. 重叠感知对应关系建立**

- **功能**：在部分-部分匹配场景中识别可靠对应点，抑制非重叠区域的干扰
- **核心思路**：网络额外预测每个点是否处于重叠区域的置信度 $\hat{O}_Q^c, \hat{O}_P^c$，将其与特征描述子逐元素相乘后计算相关矩阵 $\mathbf{X}^c = \text{softmax}[(\hat{O}_Q^c \odot \hat{F}_Q^c)(\hat{O}_P^c \odot \hat{F}_P^c)^\top]$，同时引入可学习背景 token 处理无对应点
- **设计动机**：单参考图场景下遮挡、视角差异大导致重叠比例不可预知，不加区分的匹配会引入大量错误对应，需要自动调整每个对应点的权重

**3. 层级几何编码（GRF + LRF）**

- **功能**：粗匹配后对密集点云进行精细匹配，捕获局部几何细节
- **核心思路**：在精细阶段，对每个点构建局部邻域并计算其局部参考坐标系（LRF），方式与 GRF 类似但作用于局部点集。结合全局位置编码（mini-PointNet）和 LRF 编码，前者提供全局位置上下文，后者捕获细粒度局部几何结构，两者互补
- **设计动机**：粗匹配后残余误差需要利用细粒度几何特征来精确修正，LRF 保证了局部描述子的旋转不变性

### 损失函数 / 训练策略

- 粗阶段：对应矩阵的负对数似然损失 + 重叠预测的二元交叉熵损失
- 精细阶段：对应矩阵损失 + 位姿回归损失（ADD-style）
- 使用 GeoTransformer 作为几何编码器，DINOv2 作为颜色特征编码器
- 通过 $N_H$ 个三元组点对假设采样和评分选择最佳粗位姿

## 实验关键数据

### 主实验

基于 BOP Challenge 的 AR_BOP 指标（YCB-V + LM-O + TUD-L 平均）：

| 方法 | 参考类型 | AR_BOP |
|------|---------|--------|
| ICP（经典方法） | 单参考 | 13.8 |
| FPFH + RANSAC | 单参考 | 28.5 |
| DVMNet | 单参考 | 42.9 |
| **UNOPose** | **单参考** | **70.9** |
| ZTE-PPF (CAD-based) | CAD模型 | 69.0 |
| Koenig-PPF (CAD-based) | CAD模型 | 75.1 |

### 消融实验

| 配置 | YCB-V AR | LM-O AR | TUD-L AR |
|------|----------|---------|----------|
| w/o GRF | 62.1 | 43.2 | 71.8 |
| w/o Overlap Predictor | 68.3 | 49.7 | 80.5 |
| w/o LRF (fine) | 70.2 | 51.4 | 82.1 |
| Full UNOPose | **73.8** | **55.2** | **83.7** |

### 关键发现

1. UNOPose（70.9% AR_BOP）超越了基于 CAD 模型的 ZTE-PPF（69.0%），且仅需单张无位姿参考
2. GRF 贡献最大，移除后性能显著下降，证实了 $SE(3)$ 不变标准化的关键性
3. 相比传统方法（ICP 13.8%, FPFH 28.5%），学习方法在单参考设定下优势巨大
4. 重叠预测器在低重叠场景（大视角差异）中提升尤为显著

## 亮点与洞察

- 首次将未知物体位姿估计的参考需求降低到单张无位姿 RGB-D 图像，极大简化了部署流程
- GRF 的设计简洁优雅，利用点云协方差矩阵 SVD 构建不变坐标系，计算高效且无需学习
- 构建了基于 BOP Challenge 的标准化评测基准，便于社区评估和比较
- 单张参考超越部分 CAD-based 方法的结果令人惊喜

## 局限与展望

- GRF 对对称物体不鲁棒，法向量方向可能模糊
- 深度数据的噪声水平和传感器类型会显著影响性能
- 遮挡严重时，重叠区域过小仍可能导致失败
- 未来可扩展到少样本（few-shot）参考设定，进一步提升鲁棒性
- 可探索将 RGB-only 推广为纯 RGB 设定（无需深度）

## 相关工作与启发

- **FoundationPose / MegaPose**: 利用 CAD 模型渲染多视图做位姿估计，本文证明单张参考也能达到相当水平
- **SAM-6D**: 建立 3D-3D 对应关系的方法，UNOPose 借鉴了其背景 token 机制
- **GeoTransformer**: 用于点云配准的几何 Transformer，是 UNOPose 特征提取的骨干
- 启发：在机器人操作等场景中，用户拍一张照片即可让机器人识别和定位物体，极大降低了应用门槛

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 单参考无位姿设定是全新的，GRF 和重叠预测设计有效
- **实验充分度**: ⭐⭐⭐⭐ — BOP 标准评测，消融全面，与多种基线对比
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，数学推导严谨
- **价值**: ⭐⭐⭐⭐⭐ — 显著降低了未知物体位姿估计的应用门槛，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] One2Any: One-Reference 6D Pose Estimation for Any Object](one2any_one-reference_6d_pose_estimation_for_any_object.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[CVPR 2025\] Co-op: Correspondence-based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)
- [\[ECCV 2024\] FoundPose: Unseen Object Pose Estimation with Foundation Features](../../ECCV2024/human_understanding/foundpose_unseen_object_pose_estimation_with_foundation_features.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)

</div>

<!-- RELATED:END -->
