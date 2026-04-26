---
title: >-
  [论文解读] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos
description: >-
  [CVPR 2025][3D视觉][3D高斯] 提出 RigGS，一种无需模板先验的自动化骨架驱动建模方法，从单目视频中提取 3D 骨架并绑定 3D 高斯表示，支持新视角合成、姿态编辑、运动插值和运动迁移。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯
  - 骨架提取
  - 铰接体建模
  - 动态场景
  - 无模板先验
---

# RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos

**会议**: CVPR 2025  
**arXiv**: [2503.16822](https://arxiv.org/abs/2503.16822)  
**代码**: [项目页面](https://yaoyx689.github.io/RigGS.html)  
**领域**: 3D视觉  
**关键词**: 3D高斯, 骨架提取, 铰接体建模, 动态场景, 无模板先验

## 一句话总结

提出 RigGS，一种无需模板先验的自动化骨架驱动建模方法，从单目视频中提取 3D 骨架并绑定 3D 高斯表示，支持新视角合成、姿态编辑、运动插值和运动迁移。

## 研究背景与动机

铰接体的骨架绑定(Rigging)是影视、游戏和 AR/VR 中的核心技术，传统方法依赖手工设计或参数化模板模型（如 SMPL、MANO、SMAL），面临以下挑战：

- **模板依赖限制通用性**：SMPL 只能处理人体，MANO 只能处理手部，无法处理穿着配饰的人体、多样化动物或机器人等非标准对象
- **依赖高质量 3D 重建**：现有无模板方法通常先重建 3D 网格再提取骨架，强烈依赖重建质量；基于 Medial Axis Transform 的方法需要精确的表面重建
- **骨架稠密缺乏语义**：优化方法虽可提取骨架但结果往往过于稠密，缺乏简洁的语义结构
- **Neural Bones 缺乏可编辑性**：BANMo 等方法使用可学习骨骼表示运动，但这些骨骼不携带语义信息，难以用于创建合理的新动作
- **3D 数据稀缺**：从点云序列提取骨架的方法受限于有限的 3D 训练数据

## 方法详解

### 整体框架

RigGS 采用三阶段流程：(1) 初始化阶段——使用骨架感知节点控制的变形场驱动 canonical 3D 高斯表示，完成 4D 重建并获得候选骨架节点；(2) 骨架构建阶段——通过 coarse-to-fine 的启发式算法从候选节点中提取稀疏骨架树；(3) 骨架驱动建模阶段——使用可学习蒙皮权重和姿态相关的细节变形驱动 3D 高斯。

### 关键设计1: 骨架感知节点控制变形 — 同时完成重建和候选骨架提取

**功能**: 在初始 4D 重建过程中，同时获得 canonical 3D 高斯表示和有骨架语义的候选控制节点。

**核心思路**: 定义一组可学习的骨架感知节点 $\mathbf{C} = \{\mathbf{c}\}$，每个节点在时间 $t$ 拥有旋转 $\tilde{\mathbf{R}}_\mathbf{c}^t$ 和平移 $\tilde{\mathbf{t}}_\mathbf{c}^t$（通过 MLP 预测），每个高斯的变形由其 $k$ 近邻节点的加权混合决定：

$$\mu_i^t = \sum_{\mathbf{c} \in \mathcal{N}(\mu_i)} w_{\mu_i,\mathbf{c}} (\tilde{\mathbf{R}}_\mathbf{c}^t (\mu_i - \mathbf{c}) + \mathbf{c} + \tilde{\mathbf{t}}_\mathbf{c}^t)$$

权重基于高斯核距离计算。同时引入 2D 骨架投影约束 $L_\text{proj}^t$ 保证节点与物体中轴对齐。

**设计动机**: 不同于先重建后提取骨架的两步法，该设计将骨架信息融入重建过程，减少对 3D 重建质量的依赖。2D 骨架投影约束比直接从 3D 模型提取中轴更简单更鲁棒。

### 关键设计2: Coarse-to-Fine 骨架构建 — 融合几何、运动和语义信息

**功能**: 从稠密的骨架感知节点中提取出稀疏、具有语义意义的骨架树结构。

**核心思路**: 首先选择最接近平均姿态的时刻作为新的 canonical shape。然后通过 FPS 采样获得均匀分布的节点子集，构建基于平均帧间距离 $\beta_{ij} = \sum_t \|\mathbf{c}_i^t - \mathbf{c}_j^t\| / |\mathcal{I}|$ 的最小生成树。接着移除冗余分支、合并相邻交叉点得到稠密骨架树。最后利用 DINOv2 的语义标签确保对称性，通过 BFS 确定根节点和父子关系，得到最终的稀疏骨架树 $\mathcal{T} = \{\mathcal{J}, \mathcal{A}\}$。

**设计动机**: 综合利用几何（空间距离）、运动（跨帧距离变化）和语义（DINOv2 特征）三方面信息，比纯几何方法更能获得合理的骨架结构。

### 关键设计3: 骨架驱动动态建模 — 可学习蒙皮 + 姿态细节变形

**功能**: 通过骨架驱动的变形场实现高质量渲染和灵活的姿态编辑。

**核心思路**: 使用 LBS (Linear Blend Skinning) 进行粗变形：$\hat{\mu}_i^t = \mathbf{T}_1^t (\sum_{j} \hat{\omega}_{i,j} \mathbf{T}_j^t \bar{\mu}_i)$，其中蒙皮权重 $\hat{\omega}_{i,j}$ 由可学习的缩放因子 $\eta_{i,j}$（MLP 学习）和到骨骼的距离共同决定。在此基础上添加姿态相关的细节变形 $\delta_{i,t} = F_\Pi(\mu_i, \{\hat{\mathbf{r}}_j^t\})$，最终位置为 $\mu_i^t = \hat{\mu}_i^t + \delta_{i,t}$。

**设计动机**: LBS 提供全局刚性变形，但对布料褶皱等细节不足；姿态相关（而非时间相关）的细节变形模块确保了在创建新动作时也能产生合理的细节。正则化项 $L_\text{detail}$ 控制细节变形量级。

### 损失函数

$$L^t = L_\text{render}^t + w_{\tilde{\text{proj}}}^t L_{\tilde{\text{proj}}}^t + w_\text{detail}^t L_\text{detail}^t + w_\text{id} L_\text{id}^t$$

其中 $L_\text{render}$ 为 $\ell_1$ + D-SSIM 渲染损失，$L_{\tilde{\text{proj}}}$ 为自适应权重的骨架投影约束，$L_\text{detail}$ 为细节变形正则化，$L_\text{id}$ 约束 canonical 时刻的骨架姿态接近恒等变换。

## 实验关键数据

### 主实验: 新视角合成 (PSNR/SSIM/LPIPS)

| 方法 | D-NeRF | DG-Mesh |
|------|--------|---------|
| D-NeRF | 30.48 / 0.973 / 0.049 | 28.17 / 0.957 / 0.078 |
| TiNeuVox | 32.60 / 0.983 / 0.044 | 31.95 / 0.967 / 0.048 |
| 4D-GS | 33.25 / 0.989 / 0.023 | 33.96 / 0.979 / 0.027 |
| SC-GS | **43.04** / 0.998 / 0.007 | **38.96** / 0.993 / 0.014 |
| AP-NeRF | 30.94 / 0.970 / 0.035 | 31.83 / 0.967 / 0.046 |
| **RigGS** | 40.82 / 0.996 / 0.011 | 37.65 / 0.991 / 0.017 |

### ZJU-MoCap 真实数据集对比

| 方法 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| AP-NeRF | 25.62 | 0.919 | 0.093 |
| **RigGS** | **33.54** | **0.975** | **0.033** |

### 消融实验

| 设置 | 效果 |
|------|------|
| 各向异性 vs 各向同性高斯 | 各向异性质量更高但新姿态泛化差，选择各向同性 |
| 无 2D 投影损失 | 骨架无法嵌入高斯形状内部 |
| 固定权重投影损失 | 部分帧骨架突出形状外 |

### 关键发现

- RigGS 在渲染质量上接近 SC-GS（差 ~2dB），但 SC-GS 不支持骨架编辑
- 在 ZJU-MoCap 上大幅超越 AP-NeRF（+7.92dB），证明减少对 3D 重建质量的依赖是正确的
- 自适应投影损失权重通过 3-sigma 规则过滤不准确的 2D 骨架，有效避免错误引导

## 亮点与洞察

1. **将骨架提取融入重建过程**而非先重建后提取，从根本上降低了对重建质量的依赖
2. **多信息融合的启发式骨架简化**结合几何、运动和语义(DINOv2)信息，比纯几何方法更鲁棒
3. **姿态相关而非时间相关的细节变形**确保了对新姿态的泛化能力
4. **工程完成度高**：提供了交互式 GUI 支持实时编辑和渲染

## 局限与展望

- 在稀疏视角、不准确相机位姿或剧烈运动情况下效果受限
- 未建模姿态相关的外观变化（如光照和材质变化）
- 骨架的语义编辑（如通过文本或图像指导）是有趣的未来方向
- 与 SC-GS 相比渲染质量略低，因为骨架驱动的变形场自由度更低

## 相关工作与启发

- **SC-GS**: 使用 512 个控制点的变形场，自由度高但不支持语义编辑
- **AP-NeRF**: 类似的无模板骨架提取方法，但依赖 TiNeuVox 重建和 MAT 骨架提取
- **BANMo / BAGS**: 使用 neural bones 表示变形，但缺乏树状骨架结构的语义信息
- **DINOv2**: 为骨架简化提供了有效的视觉语义特征

## 评分

⭐⭐⭐⭐ — 完整的无模板铰接体骨架化流程设计精妙，三阶段方法逻辑清晰。虽然渲染质量略低于自由变形方法，但骨架带来的可编辑性是独特价值。在真实数据集上大幅超越同类方法 AP-NeRF。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](iaao_interactive_affordance_learning_for_articulated_objects_in_3d_environments.md)
- [\[CVPR 2025\] Textured Gaussians for Enhanced 3D Scene Appearance Modeling](textured_gaussians_for_enhanced_3d_scene_appearance_modeling.md)
- [\[CVPR 2025\] Category-Agnostic Neural Object Rigging](category-agnostic_neural_object_rigging.md)
- [\[CVPR 2025\] UnCommon Objects in 3D](uncommon_objects_in_3d.md)
- [\[NeurIPS 2025\] URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model](../../NeurIPS2025/3d_vision/urdf-anything_constructing_articulated_objects_with_3d_multimodal_language_model.md)

<!-- RELATED:END -->
