---
title: "Multi-View Pose-Agnostic Change Localization with Zero Labels"
conference: "CVPR 2025"
arxiv: "2412.03911"
arxiv_url: "https://arxiv.org/abs/2412.03911"
code: "https://MV-3DCD.github.io"
domain: "3d_vision"
keywords: ["change detection", "3D Gaussian Splatting", "pose-agnostic", "unsupervised", "multi-view", "scene understanding"]
---

# Multi-View Pose-Agnostic Change Localization with Zero Labels

## 一句话总结

提出首个无标签、位姿无关的多视角变化检测方法，通过在 3D Gaussian Splatting 中嵌入变化通道（change-aware 3DGS），融合多视角的特征感知和结构感知变化掩码，在复杂多物体场景中实现 1.7× mIoU 和 1.5× F1 的 SOTA 性能提升，并能为未见视角生成变化掩码。

## 研究背景与动机

自主智能体需要检测环境变化以更新地图和重新规划任务，但现有变化检测方法面临三大挑战：

1. **位姿约束**：传统方法要求变化前后的图像精确对齐（如固定相机、平面场景），不适用于机器人随机轨迹采集的场景
2. **标注依赖**：监督学习方法需要昂贵的变化标注数据集，且在分布偏移时性能急剧下降
3. **单视角的脆弱性**：现有的位姿无关方法（OmniPoseAD、SplatPose）使用 NeRF/3DGS 渲染参考视角后做逐图比较，但容易受到视角相关的假阳性干扰（反射、阴影、遮挡区域）

核心关键观察：单视角方法产生大量因视角变化引起的假阳性，如果利用多个视角的信息并在 3D 场景中融合变化信息，可以有效抑制这些视角相关的噪声。

## 方法详解

### 整体框架

整个流程分为五步：
1. 用参考场景图像构建 3DGS 参考场景表示
2. 将推理场景图像注册到参考场景的同一坐标系
3. 渲染与推理视角对齐的参考图像，生成特征感知+结构感知的候选变化掩码
4. 在推理场景的 3DGS 中嵌入变化通道，学习多视角一致的变化表示
5. 从 change-aware 3DGS 渲染最终的多视角变化掩码

### 关键设计

#### 1. 特征感知 + 结构感知候选变化掩码

**特征感知变化掩码**：
- 使用预训练的 DINOv2 提取渲染图像和推理图像的密集特征
- 计算特征差异：$D^k = \sum_{j=1}^d |f_{ren}^{k,j} - f_{inf}^{k,j}|$
- 归一化后阈值 0.5 过滤低值假变化

**结构感知变化掩码**：
- 使用 SSIM（结构相似度指数）比较渲染和推理图像
- 二值化低 SSIM 区域：$M_S^k = \mathbf{1}(\text{SSIM}(I_{ren}^k, I_{inf}^k) \leq 0.5)$

**联合掩码**：通过逐元素相乘实现"双重投票"滤波：$M_{F,S}^k = M_F^k \cdot M_S^k$

关键洞察：特征级（DINOv2）和像素级（SSIM）的变化检测是互补的——前者对语义敏感但对光照不敏感，后者对结构/光照变化敏感。

#### 2. Change-Aware 3D Gaussian Splatting

核心贡献——在 3DGS 中为每个 Gaussian 添加两个变化参数：
- **变化幅度 $\tilde{c}$**：表示该 Gaussian 在场景中捕获的变化程度
- **变化不透明度 $\tilde{\alpha}$**：控制该 Gaussian 对变化掩码渲染的贡献

关键设计决策：变化幅度使用**零阶球谐函数系数**（SH degree=0），而非标准 3DGS 的三阶。

原因：场景变化是视角无关的，而大多数视角相关的变化（反射、阴影、微小对齐偏差）是假阳性。使用低阶 SH 可以：
- 利用多视角信息学习真实变化区域
- 不过拟合视角相关的假阳性

用参考场景的 3DGS 初始化推理场景的 change-aware 3DGS，然后用推理图像和候选变化掩码联合优化所有参数。

#### 3. 数据增强策略

利用 change-aware 3DGS 反向操作：
- 从推理场景 3DGS 渲染参考视角的图像
- 将渲染图与原始参考图比较，生成额外的变化掩码
- 将新旧掩码合并重新优化变化通道

这相当于"反向检测"——从两个方向发现变化，增加了学习变化通道的训练数据。

### 损失函数

标准 3DGS 优化目标（L1 + D-SSIM 损失）扩展到变化通道：
- RGB 重建损失用于更新场景外观参数
- 变化通道的 L1 + D-SSIM 损失用于学习 $\tilde{c}$ 和 $\tilde{\alpha}$

## 实验关键数据

### 主实验表

**MAD-Real 数据集（单物体）**：

| 方法 | mIoU↑ | F1↑ | AUROC↑ |
|------|-------|-----|--------|
| OmniPoseAD | 0.064 | 0.115 | 0.937 |
| SplatPose | 0.077 | 0.123 | 0.898 |
| Feature Diff. | 0.052 | 0.089 | 0.967 |
| **Ours** | **0.132** | **0.210** | 0.953 |

**PASLCD 数据集（多物体场景，平均）**：

| 方法 | mIoU↑ | F1↑ |
|------|-------|-----|
| OmniPoseAD | 0.168 | 0.262 |
| SplatPose | 0.173 | 0.281 |
| CYWS-2D | 0.273 | 0.398 |
| Feature Diff. | 0.264 | 0.386 |
| **Ours** | **0.461** | **0.612** |

### 消融实验

**球谐函数阶数影响**：

| SH Degree | mIoU↑ | F1↑ | FP↓ |
|-----------|-------|-----|-----|
| 0 | 最优 | 最优 | 最低 |
| 3 | 较低 | 较低 | ~3× |

使用 SH degree=0 可减少约 70% 的假阳性。

**推理视角数量影响**：
- 仅 5 张推理图像即可实现 Feature Diff. 的 1.8× mIoU
- 性能随视角数增加而提升（5→10→15→25）

### 关键发现

1. **多视角方法大幅超越单视角**：在 PASLCD 上实现 1.7× mIoU 和 1.5× F1 的提升
2. **可为未见视角生成变化掩码**：这是现有方法无法做到的全新能力
3. **SH degree=0 是关键**：变化应被建模为视角无关的，低阶 SH 有效抑制了假阳性
4. **CYWS-2D + Change-3DGS 提升 44%**：证明 change-aware 3DGS 可作为任何变化掩码方法的多视角扩展
5. **ChangeSim 上 1.7× 改善**：在工业仿真场景中同样有效

## 亮点与洞察

1. **将变化检测从 2D "image-pair" 提升到 3D "scene"**：通过在 3DGS 中嵌入变化通道，变化检测从逐图比较升级为场景级理解
2. **SH degree=0 的精妙设计**：基于"真实变化视角无关、假阳性视角相关"的假设，用球谐函数阶数约束来实现去噪——优雅地利用了 3DGS 的表示灵活性
3. **特征+结构双重变化检测**：DINOv2 特征和 SSIM 的互补性被清晰论证，前者抗光照变化，后者捕获精细结构差异
4. **贡献真实世界数据集 PASLCD**：10 个真实场景（室内/室外）、500 个标注掩码、91 个变化实例，显著推动了该领域

## 局限性

1. **颜色级表面变化检测困难**：如液体溅洒、颜色替换等不改变 3D 结构的变化，DINOv2 特征可能不敏感
2. **小物体在大场景中容易遗漏**：Playground 和 Lunch Room 场景中的小变化检测效果较差
3. **变化掩码过估计**：由于特征图 patch-to-pixel 插值，真实变化的掩码边界通常比真值大
4. **COLMAP 注册失败时无法工作**：如推理场景极暗或与参考场景重叠不足
5. **计算成本**：需要训练两个 3DGS（参考+推理），加上 DINOv2 的特征提取

## 相关工作与启发

- **3D Gaussian Splatting** [Kerbl et al., 2023]：本文将其从"渲染工具"扩展为"变化检测工具"
- **OmniPoseAD / SplatPose**：前序工作利用 3D 表示实现位姿无关的异常检测，但仅做单视角比较
- **DINOv2**：作为特征感知变化掩码的强大骨干，其在多种场景下的零次泛化能力是方法成功的基础
- **SSIM**：传统图像质量指标被创新性地用于变化检测，与深度学习特征形成互补
- 启发：3D 表示不仅能做渲染和重建，还能作为多视角信息融合的媒介

## 评分

⭐⭐⭐⭐ (8.5/10)

- 创新性：⭐⭐⭐⭐⭐ — 首个多视角无标签位姿无关变化检测，change-aware 3DGS 概念新颖
- 实用性：⭐⭐⭐⭐ — 对机器人环境理解和基础设施监测有直接价值
- 实验充分度：⭐⭐⭐⭐⭐ — 三个数据集、多个基线、详细消融、视角数量分析
- 写作清晰度：⭐⭐⭐⭐⭐ — 方法描述层次分明，图表配合优秀

<!-- RELATED:START -->

## 相关论文

- [MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [Changes in Real Time: Online Scene Change Detection with Multi-View Fusion](../../CVPR2026/3d_vision/changes_in_real_time_online_scene_change_detection_with_multi-view_fusion.md)
- [Category-Agnostic Neural Object Rigging](category-agnostic_neural_object_rigging.md)
- [Cross-View Completion Models are Zero-shot Correspondence Estimators](cross-view_completion_models_are_zero-shot_correspondence_estimators.md)
- [CoMatcher: Multi-View Collaborative Feature Matching](comatcher_multi-view_collaborative_feature_matching.md)

<!-- RELATED:END -->
