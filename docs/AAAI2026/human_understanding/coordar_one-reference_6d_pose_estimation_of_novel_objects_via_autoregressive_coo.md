---
title: >-
  [论文解读] CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation
description: >-
  [AAAI 2026][人体理解][6D位姿估计] 提出 CoordAR，将单参考视图 6D 位姿估计中的 3D-3D 对应关系建模为离散 token 的自回归生成问题，通过坐标图 token 化、模态解耦编码和自回归 Transformer 解码器，在多个基准上显著超越现有单视图方法，并对对称、遮挡等挑战场景展现强鲁棒性。
tags:
  - AAAI 2026
  - 人体理解
  - 6D位姿估计
  - 自回归
  - 坐标图
  - VQ-VAE
  - 单参考视图
---

# CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation

**会议**: AAAI 2026  
**arXiv**: [2511.12919](https://arxiv.org/abs/2511.12919)  
**代码**: [项目页面](https://sjtu-visys-team.github.io/CoordAR)  
**领域**: 人体理解  
**关键词**: 6D位姿估计, 自回归, 坐标图, VQ-VAE, 单参考视图

## 一句话总结

提出 CoordAR，将单参考视图 6D 位姿估计中的 3D-3D 对应关系建模为离散 token 的自回归生成问题，通过坐标图 token 化、模态解耦编码和自回归 Transformer 解码器，在多个基准上显著超越现有单视图方法，并对对称、遮挡等挑战场景展现强鲁棒性。

## 研究背景与动机

6D 物体位姿估计（恢复刚体物体的旋转和平移）是 CV/机器人领域基础任务，广泛应用于 AR、机器人操作和工业自动化。核心挑战：

**依赖 3D 模型**：实例级方法需为每个物体训练专用网络，类别级方法限于已知类别，类别无关方法仍需 CAD 模型

**单参考视图设定**（one-reference）：仅用一张已知位姿的参考图估计新视角下物体位姿，更贴近现实但极具挑战

现有单参考方法的局限：
- **One2Any** 使用卷积解码器回归连续坐标图，卷积的局部感受野限制了全局一致性
- 连续回归无法处理对称性和遮挡导致的**固有歧义**——对称物体存在多个合法解，直接回归会产生错误的平均结果

## 方法详解

### 整体框架

CoordAR 由三个主要阶段组成：

1. **模态解耦编码**：分别编码参考/查询 RGB 图像和参考坐标图（ROC map）
2. **特征融合**：堆叠融合块将参考信息注入查询特征
3. **自回归解码**：以融合特征为条件，自回归生成离散坐标 token，再反 token 化为像素级坐标图

最终通过 Umeyama 算法从坐标对应关系恢复刚体变换。

### 关键设计

**1. 参考物体坐标（ROC）图表示**

参考视图的 ROC 图 $\mathbf{X}^R$ 通过深度反投影得到，查询视图的 ROC 图 $\mathbf{X}^Q = \mathbf{S}\mathbf{T}_{RQ}\mathbf{\Pi}^{-1}(\mathcal{D}_Q)[\mathcal{M}_Q=1]$ 编码了参考-查询间的相对变换。两个 ROC 图都表示参考物体坐标系下的 3D 点，提供了像素级的 3D-3D 对应关系。位姿估计问题转化为：**估计查询图像的 ROC 图 $\mathbf{X}^Q$**。

**2. 模态解耦编码（Modality-Decoupled Encoding）**

不同于 One2Any 的 role-specific 编码（将 RGB 和 ROC 图拼接输入同一编码器），CoordAR 为不同模态分配独立编码器：
- **共享 RGB 编码器**：同时处理参考和查询的 RGB 图像
- **ROC 编码器**：专门处理参考的坐标图

这避免了 RGB 和坐标图这两种结构模式差异巨大的模态被混合编码。在融合块的交叉注意力中也按模态分别计算亲和度，减小 RGB-ROC 域差距。

**3. 坐标图 Token 化（VQ-VAE）**

采用 VQ-VAE 将连续坐标图编码为离散 token 序列 $\{s_1, \ldots, s_{h \cdot w}\}$，其中每个 token $s_* \in \mathcal{V}$（预训练码本）。核心优势：

- 在离散空间上建立**概率分布**，每个 patch 位置输出码本上的分类分布
- 对称物体可通过多模态分布表达多个合法解，而非被迫回归出错误的平均值
- 遮挡区域可通过高熵分布表达不确定性

**4. 自回归 Transformer 解码器**

token 生成建模为掩码自回归模型：

$$p(s) = \prod_{k=1}^K p(S_k | S_{<k}, C_{\mathbf{F}})$$

其中 $S_k$ 为第 $k$ 步生成的 token 集合，$C_{\mathbf{F}}$ 为位置对齐的条件特征。训练目标为最小化负对数似然：

$$\mathcal{L}_{\text{AR}} = -\sum_k^K [\log(p(S_k | S_{<k}, C_{\mathbf{F}}))]$$

解码器由多个自注意力块组成，以可学习 mask token 为输入，参考已生成 token 和位置对齐条件特征预测后续 token。推理时支持多步生成（64/16/4/1步），步数可灵活权衡精度与速度。

**5. 位姿恢复**

生成的 ROC 图 $\hat{\mathbf{X}}^Q$ 经逆归一化得到参考坐标系下的 3D 点 $\hat{\mathbf{P}}_R^Q$，与查询深度图反投影的 3D 点 $\mathbf{P}_Q^Q$ 像素对齐，通过 Umeyama 算法求解最优刚体变换。

### 损失函数 / 训练策略

- 主要损失：自回归负对数似然 $\mathcal{L}_{\text{AR}}$
- VQ-VAE tokenizer 预训练后冻结
- 在 FoundationPose 数据集和 OO3D-9D 子集上训练
- 假设提供 ground-truth mask

## 实验关键数据

### 主实验

**Table 1: Real275 & Toyota-Light（真实新物体泛化）**

| 方法 | 模态 | 参考数 | Real275 AR | Real275 ADD(-S) | Toyota AR | Toyota ADD(-S) |
|---|---|---|---|---|---|---|
| One2Any | RGBD | 1 | 54.9 | 41.0 | 42.0 | 34.6 |
| **CoordAR** | RGBD | 1 | **71.0** | **82.2** | **62.5** | **82.6** |

AR 提升 +16.1/+20.5，ADD(-S) 提升 +41.2/+48.0，提升幅度巨大。

**Table 3: LINEMOD（大视角变化）**

| 方法 | 参考数 | 平均 ADD(-S) |
|---|---|---|
| One2Any | 1 | 52.6 |
| **CoordAR** | 1 | **75.0** |
| OnePose++ | 200 | 76.9 |

仅用 1 张参考图即接近使用 200 张参考图的 OnePose++。

### 消融实验

**Table 4: 关键设计选择消融（LINEMOD）**

| 组件/变体 | ADD(-S) | AR |
|---|---|---|
| 卷积解码器 | 70.9 | 59.7 |
| 自回归解码器 | **73.1** | **61.6** |
| 无自回归（并行 token） | 60.7 | 52.1 |
| 无 token 化（连续回归） | 56.4 | 48.7 |
| Role-specific 编码 | 61.6 | 49.8 |
| 模态解耦编码 | **73.6** | **61.9** |

三个核心设计（token 化、自回归、模态解耦编码）各自贡献显著，模态解耦编码影响最大（+12.0 AR）。

**Table 5: 推理速度-精度权衡**

| 步数 | 时间(s) | ADD(-S) |
|---|---|---|
| 64 步 | 0.63 | 75.0 |
| 16 步 | 0.25 | 75.0 |
| 4 步 | 0.13 | 74.7 |
| 1 步 | 0.10 | 74.3 |

1 步推理仅损失 0.7% 精度，达到近实时速度（0.10s/帧）。

### 关键发现

- 自回归生成的坐标图空间一致性远优于并行生成（可视化中非自回归方法出现颜色断裂）
- 离散 token 化在对称物体（碗）和遮挡物体（杯）上表现尤其突出，能有效避免歧义平均
- 在 YCB-V 遮挡场景中，ADD-S AUC 显著优于 One2Any，但 ADD AUC 因纹理对称物体评估偏差略低

## 亮点与洞察

1. **离散化 + 自回归 = 解决歧义**：将连续坐标回归转为离散概率预测的设计直觉敏锐，直接解决了对称/遮挡的核心问题
2. **模态解耦 > 角色解耦**：按数据模态（RGB vs 坐标）而非功能角色（参考 vs 查询）分配编码器，效果差异显著
3. **灵活的速度-精度权衡**：1-64 步推理仅 0.7% 精度差距，1 步即接近实时
4. **极简位姿恢复**：利用 Umeyama 算法从密集对应关系直接解析位姿，无需迭代优化

## 局限与展望

1. 依赖 GT mask 输入，实际应用需结合分割网络
2. VQ-VAE tokenizer 需预训练，码本大小和 patch 分辨率的选择影响上限
3. 在纹理丰富但几何对称的物体上（如 YCB-V 的汤罐），ADD 指标受评估协议偏差影响
4. 自回归多步生成速度仍慢于 One2Any 的并行方式（0.63s vs 0.09s）

## 相关工作与启发

- **One2Any**：最直接的 baseline，用卷积回归连续坐标图，CoordAR 在其基础上引入 token 化+自回归
- **VQ-VAE / MAGVIT**：离散 token 化在图像/视频生成中的成功经验被巧妙迁移到坐标图领域
- **启发**：自回归建模密集对应关系的思路可扩展到光流估计、场景流等任务；模态解耦编码的结论值得多模态融合任务借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将自回归坐标图生成用于 6D 位姿估计，范式创新
- 技术深度: ⭐⭐⭐⭐ — token 化+自回归+模态解耦的组合设计完整
- 实验充分度: ⭐⭐⭐⭐⭐ — 4 个数据集，详细消融（5 组），速度分析
- 写作质量: ⭐⭐⭐⭐ — 问题定义形式化清晰，消融分析深入

<!-- RELATED:START -->

## 相关论文

- [One2Any: One-Reference 6D Pose Estimation for Any Object](../../CVPR2025/human_understanding/one2any_one-reference_6d_pose_estimation_for_any_object.md)
- [COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](../../CVPR2026/human_understanding/cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](../../ICCV2025/human_understanding/mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)
- [VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation](vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)
- [MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Network](mvgd-net_a_novel_motion-aware_video_glass_surface_detection_network.md)

<!-- RELATED:END -->
