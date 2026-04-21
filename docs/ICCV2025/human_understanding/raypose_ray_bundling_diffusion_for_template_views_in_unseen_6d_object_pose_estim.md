---
title: >-
  [论文解读] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation
description: >-
  [人体理解] 将未见物体6D位姿估计重新建模为射线对齐问题，提出物体中心的射线参数化方案，运用扩散变换器从多个已知位姿模板中推断查询图像的6D位姿。
tags:
  - 人体理解
---

# RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2510.18521](https://arxiv.org/abs/2510.18521)
- **代码**: [项目页面](https://demianhj.github.io/projects/RayPose)
- **领域**: 人体理解
- **关键词**: 6D物体位姿估计, 未见物体, 扩散模型, 射线参数化, 多视图推理

## 一句话总结

将未见物体6D位姿估计重新建模为射线对齐问题，提出物体中心的射线参数化方案，运用扩散变换器从多个已知位姿模板中推断查询图像的6D位姿。

## 研究背景与动机

6D物体位姿估计在机器人抓取、AR/VR、自动驾驶等应用中至关重要。现有方法面临两个核心问题：

**模板匹配式方法的局限**：传统方法先检索最匹配模板再对齐，但检索失败直接导致位姿不准。这种分步流程将问题分解为模板检索 → 对应点估计 → 位姿预测 → 可选精化，每步错误会累积。

**单视图约束不足**：多视图几何为3D感知提供核心约束，但现有单目方法无法利用多视图约束。虽然模板图像本身就是带位姿的多视图，但现有方法仅用于分类匹配而非几何推理。

**扩散模型的适用性差距**：RayDiffusion 等方法在相机位姿估计中展示了强大的泛化能力，但相机位姿定义在大尺度世界坐标系中，而物体位姿在紧凑的物体中心空间，直接迁移效果不佳。

## 方法详解

### 整体框架

RayPose 接受一张未见物体的查询图像和一组已知位姿的模板图像作为输入，通过扩散变换器逐步去噪结构化的2D位姿图来预测物体6D位姿。核心包括：Query Encoder、Template Encoder（含 Multiview Fuser）和 Diffusion Transformer Decoder。

### 旋转参数化——物体中心射线表示

与 RayDiffusion 的相机中心射线不同，本文提出物体中心射线：将物体中心视为虚拟针孔相机，射线从物体中心射向相机坐标系。方向向量集合为：

$$\mathcal{M}_R = \{d_1, \ldots, d_n\}$$

每个方向向量归一化为单位长度，映射到 $p \times p \times 3$ 的均匀2D网格上。这种表示将任意旋转矩阵 $R$ 映射到单位球面上的唯一结构化网格。旋转恢复通过SVD求解最优对齐：

$$R^* = \arg\min_{R \in \text{SO}(3)} \sum_{i=1}^n \|Rd_i^* - d_i\|^2$$

### 平移参数化——密集平移偏移图

扩展 Scale-Invariant Translation Estimation (SITE) 到 patch 级密集平移图。给定物体平移 $t = [t_x, t_y, t_z]$ 和相机内参 $K$，投影物体中心到图像坐标 $[o_x, o_y, 1]^T = Kt$，构建密集归一化平移偏移图：

$$\mathcal{M}_T = \left(\frac{u - o_x}{w}, \frac{v - o_y}{h}, \frac{t_z}{r_z}\right)$$

其中 $w, h$ 为边界框宽高，$r_z$ 为放大比例。这种解耦表示使旋转和平移可独立预测。

### 多视图模板条件扩散

**模板编码器**：冻结 DINOv2 提取图像特征，View Encoder 使用三个 Fourier 编码器处理旋转图、平移图和2D归一化边界框坐标，Multiview Fuser 通过自注意力聚合多视图信息。

**扩散过程**：前向过程对位姿图加高斯噪声 $\mathcal{M}_t = \sqrt{\alpha_t}\mathcal{M}_0 + \sqrt{1-\alpha_t}\epsilon$。网络训练预测清洁位姿图而非噪声：

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t,\epsilon}[\|\mathcal{M}_0 - \epsilon_\theta(\mathcal{M}_t, t, \mathcal{F}_C)\|_2^2]$$

**粗到细策略**：使用不同模板分布训练粗、细预测器。粗预测器随机采样模板，细预测器在查询位姿附近 ±30° 采样。推理时先粗后细，无需修改网络。

### 损失函数

旋转损失：$\mathcal{L}^R = \lambda_{\text{recon}}\mathcal{L}_{\text{recon}}^R + \lambda_{\cos}\mathcal{L}_{\cos}^R + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}^R$

其中 $\mathcal{L}_{\text{reg}}^R$ 为射线一致性损失，保证相邻射线间几何一致性：

$$\mathcal{L}_{\text{reg}}^R = \frac{1}{|\mathcal{N}_r|}\sum_{(i,j)\in\mathcal{N}_r}(\alpha_{ij} - \alpha_{ij}^*)^2$$

平移损失包括密集图重建和3D平移L1监督。总损失 $\mathcal{L} = \lambda_{\text{rot}}\mathcal{L}^R + \lambda_{\text{trans}}\mathcal{L}^T$。

## 实验

### BOP 基准主实验

| 方法 | 精化 | 多假设 | LM-O | T-LESS | TUD-L | IC-BIN | YCB-V | 平均 |
|------|------|--------|------|--------|-------|--------|-------|------|
| ZS6D | ✗ | ✗ | 29.8 | 21.0 | — | — | 32.4 | 27.7 |
| MegaPose | ✗ | ✗ | 22.9 | 17.7 | 25.8 | 15.2 | 28.1 | 21.9 |
| GenFlow | ✗ | ✗ | 25.0 | 21.5 | 30.0 | 16.8 | 27.7 | 24.2 |
| OSOP | ✗ | ✗ | 31.2 | — | — | — | 33.2 | 32.2 |

### 消融实验

| 设置 | 说明 | 性能影响 |
|------|------|---------|
| Camera-centric rays → Object-centric rays | 物体中心射线表示 | 旋转精度显著提升 |
| 无角度一致性损失 | 移除射线正则化 | 旋转精度下降 |
| 粗预测 → 粗细结合 | 粗到细策略 | 性能进一步提升 |
| 密集平移图 vs SITE | 密集偏移图 | 平移精度更高 |

### 关键发现

1. 物体中心射线表示相比相机中心 Plücker 坐标更适合物体位姿估计，因为解耦了内参影响
2. 多假设采样（从不同噪声初始化）有效捕获多模态分布，对对称物体尤为重要
3. 粗到细策略无需修改架构即可提升性能，体现了扩散框架的灵活性
4. 射线一致性损失作为几何正则化，保证预测射线图的结构完整性

## 亮点与洞察

1. **独特的问题重构**：将位姿估计视为射线束对齐而非模板匹配+对应点估计，更好利用多视图几何约束
2. **优雅的参数化设计**：物体中心射线和密集平移图将6D位姿映射到2D结构化表示，天然适配扩散模型的像素级去噪
3. **灵活的推理策略**：粗到细预测仅需改变模板采样分布，同一网络可用于不同精度需求
4. **理论根基扎实**：SVD 旋转恢复保证输出在 SO(3) 流形上

## 局限性

1. 依赖 CAD 模型渲染模板，限制了完全无模型场景的适用性
2. 扩散模型多步去噪导致推理速度较慢
3. 对严重遮挡和截断场景的鲁棒性有待验证

## 相关工作

- **模板方法**：MegaPose、GigaPose、OSOP 等渲染-比较策略
- **基础模型方法**：FoundPose (DINOv2)、ZeroPose (ImageBind+SAM)
- **扩散位姿估计**：RayDiffusion、PoseDiffusion、DiffusionNOCS

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 射线束扩散范式和物体中心参数化是全新贡献
- **技术深度**: ⭐⭐⭐⭐⭐ — 旋转/平移参数化数学严谨，损失设计周密
- **实验充分性**: ⭐⭐⭐⭐ — 多数据集评估和详细消融
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，公式规范

<!-- RELATED:START -->

## 相关论文

- [CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](../../CVPR2025/human_understanding/crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties](../../CVPR2025/human_understanding/multi-sensor_object_anomaly_detection_unifying_appearance_geometry_and_internal_.md)
- [SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)
- [OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM](orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [RapVerse: Coherent Vocals and Whole-Body Motion Generation from Text](rapverse_coherent_vocals_and_whole-body_motion_generation_from_text.md)

<!-- RELATED:END -->
