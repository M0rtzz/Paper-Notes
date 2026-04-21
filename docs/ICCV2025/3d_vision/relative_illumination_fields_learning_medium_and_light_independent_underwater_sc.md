---
title: >-
  [论文解读] Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes
description: >-
  [3D视觉] 提出相对光照场（Relative Illumination Fields），通过在相机局部坐标系中用MLP建模非均匀光照分布，结合体积介质表示，实现对水下场景的干净重建——去除光源和介质的影响。
tags:
  - 3D视觉
---

# Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2504.10024](https://arxiv.org/abs/2504.10024)
- **代码**: 未公开
- **领域**: 3D视觉
- **关键词**: 水下NeRF, 光照场, 散射介质, 联合优化, 颜色恢复

## 一句话总结

提出相对光照场（Relative Illumination Fields），通过在相机局部坐标系中用MLP建模非均匀光照分布，结合体积介质表示，实现对水下场景的干净重建——去除光源和介质的影响。

## 研究背景与动机

水下环境对3D重建提出了独特挑战：

**散射与衰减**：水体对光进行波长相关的吸收和散射，相当于在物体和相机间增加了一个"额外物体"

**动态非均匀光照**：深海（>几十米）缺乏阳光，机器人携带的人工光源随相机移动，产生强烈的非均匀后向散射光锥

**现有方法局限**：
   - SeaThru-NeRF等方法假设均匀光照（类似雾模型），无法处理人工光源场景
   - DarkGS等方法需要已知光源模型和标定，实用性受限
   - 最接近的工作假设单点光源位于相机中心，过于简化

**核心观察**：在共动光源场景中，光照分布在相机局部坐标系中保持恒定——重要的不是每个光源的具体参数，而是在相机视锥内某点接收到的累积光照。

## 方法详解

### 整体框架

在Nerfstudio/Nerfacto基础上扩展，包含三个组件：全局NeRF MLP、局部光照场MLP和介质表示。

### 1. 局部光照场表示

定义在相机局部坐标系中的MLP $\mathcal{F}^l_\Theta$：

$$\boldsymbol{\alpha} = \mathcal{F}^l_\Theta(\phi_{\text{Hash}}(\boldsymbol{x}^c), \phi_{\text{SH}}(\boldsymbol{n}^c))$$

- $\boldsymbol{x}^c$：采样点在相机坐标系中的位置
- $\boldsymbol{n}^c$：表面法线在相机坐标系中的方向（由密度场梯度导出）
- $\boldsymbol{\alpha}$：光照强度因子（每颜色通道独立）

坐标变换：$\mathbf{x}^c = {}^c\mathbf{T}_w \cdot \mathbf{x}^w$，$\boldsymbol{n}^c = {}^c\mathbf{R}_w \cdot \boldsymbol{n}^w$

**简化假设**：忽略BRDF（假设水下大多数自然材料为Lambertian），忽略光源可见性（灯通常靠近相机，阴影投在物体背面）。

### 2. 体积介质表示

将渲染公式分解为物体颜色和介质颜色：

$$\boldsymbol{C}(\boldsymbol{r}(t)) = \sum_i^N \boldsymbol{C}_i^{\text{obj}}(\boldsymbol{r}(t)) + \boldsymbol{C}_i^{\text{med}}(\boldsymbol{r}(t))$$

其中：
- 物体分量：$\boldsymbol{C}_i^{\text{obj}} = T_i^{\text{obj}} \cdot T_i^{\text{attn}} \cdot (1 - e^{-\sigma_i^{\text{obj}}\delta_i}) \cdot \boldsymbol{c}_i^{\text{obj}}$
- 介质分量：$\boldsymbol{C}_i^{\text{med}} = T_i^{\text{obj}} \cdot T_i^{\text{bs}} \cdot (1 - e^{-\boldsymbol{\sigma}^{\text{bs}}\delta_i}) \cdot \boldsymbol{c}^{\text{med}}$

介质参数 $\boldsymbol{\sigma}^{\text{attn}}$（衰减系数）、$\boldsymbol{\sigma}^{\text{bs}}$（后向散射系数）和 $\boldsymbol{c}^{\text{med}}$（介质颜色）作为全局可优化参数。

### 3. 最终模型

将光照场与介质模型结合：

$$\boldsymbol{C}(\boldsymbol{r}(t)) = \sum_i^N \boldsymbol{\alpha}_i (C_i^{\text{obj}}(\boldsymbol{r}(t)) + \boldsymbol{C}_i^{\text{med}}(\boldsymbol{r}(t)))$$

光照因子 $\boldsymbol{\alpha}_i$ 同时作用于物体和介质颜色。

### 损失函数

采用RawNeRF策略处理HDR图像：

$$\mathcal{L} = \sum_{\boldsymbol{r} \in \mathcal{R}} \|\frac{\hat{C}(\boldsymbol{r}(t)) - C(\boldsymbol{r}(t))}{\text{sg}(\hat{C}(\boldsymbol{r}(t))) + \epsilon}\|^2$$

其中 $\text{sg}(\cdot)$ 为stop-gradient，$\epsilon = 10^{-3}$。

## 实验

### 主实验：共动光源与介质去除

在五个数据集上验证（合成空气、合成水下、真实水下），包含1-4个共动光源：
- **DarkGS数据集**（空气中）：无需人工标定即可恢复干净场景，而DarkGS需要光源标定
- **Beyond NeRF水下数据集**：成功分离光锥、水体散射和物体颜色
- **自采真实水下数据**：在1m×2m水缸中用GoPro拍摄，成功去除光照和介质效果

### 消融实验：单通道 vs 三通道光照场

| 配置 | 合成空气 L2↓ | 合成水下 L2↓ | 真实水下 L2↓ |
|------|------|------|------|
| 单通道 $\alpha$ | 9.554 | 12.780 | 28.944 |
| **三通道 $\boldsymbol{\alpha}$** | **10.143** | **11.879** | **30.077** |

### 关键发现
1. **三通道光照场在水下更优**：因为光在到达物体前已经历波长相关衰减，需要逐通道建模
2. **空气中单/三通道差异不大**：无介质衰减时，假设所有光源同色，单通道即可
3. **无需光源标定**：完全从数据学习光照分布，是key advantage

## 亮点与洞察

1. **"相对光照"的优雅观察**：在共动光源场景中，相机局部坐标系的光照分布保持不变，这个观察大大简化了问题
2. **完全无标定**：不需要知道光源数量、位置、强度分布，仅从多视角观测中学习
3. **模块化设计**：去掉介质项即可用于空气中暗光场景

## 局限性

- 忽略了阴影效应（光源可见性），当阴影显著时表现受限
- 场景中从未被照亮的区域无法恢复
- 需要对数据集设置尺度因子以适应不同动态范围

## 相关工作

- **水下NeRF**: SeaThru-NeRF, UW-NeRF
- **暗光重建**: DarkGS, RawNeRF
- **重光照NeRF**: NeRFactor, S3-NeRF

## 评分

- 新颖性: ⭐⭐⭐⭐ (局部光照场的核心观察非常巧妙)
- 技术深度: ⭐⭐⭐⭐ (物理模型推导严谨)
- 实验质量: ⭐⭐⭐ (数据集规模有限，竞争方法少)
- 实用价值: ⭐⭐⭐⭐ (水下机器人视觉的实际需求)

<!-- RELATED:START -->

## 相关论文

- [PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes](placeit3d_language-guided_object_placement_in_real_3d_scenes.md)
- [InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [LayerLock: Non-collapsing Representation Learning with Progressive Freezing](layerlock_non-collapsing_representation_learning_with_progressive_freezing.md)
- [Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)

<!-- RELATED:END -->
