---
title: >-
  [论文解读] PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes
description: >-
  [3D视觉] 提出语言引导的真实3D场景中物体放置任务（PlaceIt3D），包含基准测试、大规模数据集和基于3D LLM的基线方法PlaceWizard，实现对场景、物体和语言指令的联合推理。
tags:
  - 3D视觉
---

# PlaceIt3D: Language-Guided Object Placement in Real 3D Scenes

| 信息 | 内容 |
|------|------|
| 会议 | ICCV2025 |
| arXiv | [2505.05288](https://arxiv.org/abs/2505.05288) |
| 代码 | [nianticlabs/placeit3d](https://nianticlabs.github.io/placeit3d/) |
| 领域 | 3D视觉 / 语言引导的物体放置 |
| 关键词 | 3D场景理解, 物体放置, 多模态大语言模型, 点云, 语言引导 |

## 一句话总结

提出语言引导的真实3D场景中物体放置任务（PlaceIt3D），包含基准测试、大规模数据集和基于3D LLM的基线方法PlaceWizard，实现对场景、物体和语言指令的联合推理。

## 研究背景与动机

### 问题定义

给定一个3D重建的点云场景、一个3D资产和一条自然语言指令，目标是找到满足指令的物体放置位置和朝向。这个任务需要同时解决四个交织的挑战：

**一对多歧义性**：合法放置通常不唯一，多个位置都可满足指令

**精确的几何与物理推理**：许多约束是3D几何性质的，不能仅从2D投影推断

**跨模态联合理解**：需要同时理解场景、资产形状和语言指令

**对噪声点云的鲁棒性**：测试时没有特权元数据（如场景图、干净几何）

### 现有方法的局限

- **3D视觉定位**方法通常只匹配单一正确位置，无法处理一对多问题
- **合成场景生成**方法依赖布局、场景图等特权信息
- **图像级放置**方法仅预测2D放置区域，缺乏内在3D推理能力
- 并发工作FirePlace聚焦于合成清洁环境，无法处理重建场景的噪声

## 方法详解

### 任务形式化

输入：点云场景 $\mathbf{X} \in \mathbb{R}^{N \times 6}$（含3D坐标和颜色）、3D资产、文本指令。输出：3D平移向量 $\mathbf{t}$ 和偏航角 $\alpha$。

简化假设：场景和资产的垂直方向已知，资产始终放置在水平面上，仅考虑绕垂直轴的旋转。

### 约束体系

- **物理可行性**：物体不能与场景网格相交，且必须放在某个表面上
- **空间约束**：near/adjacent、on、between、above/below
- **旋转约束**：物体朝向某个锚点
- **可见性约束**：物体在锚点的视线内或被遮挡

合法放置必须同时满足所有约束：

$$\mathcal{M} = \bigcap_{c \in \mathcal{C}} \mathcal{M}_c$$

### PlaceWizard 整体架构

PlaceWizard基于Reason3D改进，主要包含以下模块：

**1. 场景编码**
- 使用点编码器提取特征 $F_X \in \mathbb{R}^{N \times d}$
- 添加位置嵌入特征 $F_X^{pos}$
- **均匀空间池化**替代原始Superpoints：使用最远点采样选取 $M$ 个中心点，每个点分配到最近中心。保留更细粒度的空间信息，避免Superpoints将水平/垂直面聚合为单一特征
- 通过Q-Former投射到LLM嵌入空间

**2. 资产编码**
- 使用Point-BER编码器（在Objaverse上预训练）编码资产点云
- 序列特征经Max-pooling得到单一嵌入
- 分别编码资产在X/Y/Z轴上的尺寸
- 通过MLP投射到LLM空间

**3. 放置解码器**
- LLM输出三个特殊token：[LOC]、[ANC]、[ROT]
- 通过自注意力和交叉注意力层处理token特征与场景/资产特征
- 三个预测头：
    - **放置Mask头**：预测 $\mathcal{M}_{loc} \in [0,1]^N$，有效放置区域
    - **旋转头**：预测 $\mathcal{M}_{rot} \in [0,1]^{N \times 8}$，每个点的8个离散旋转角有效性
    - **锚点Mask头**：预测 $\mathcal{M}_{anc} \in [0,1]^N$，定位语言中提到的锚点（辅助任务）

### 损失函数

总损失由四部分组成：

$$\mathcal{L} = \mathcal{L}_{seg}(\bar{\mathcal{M}}_{loc}, \mathcal{M}_{loc}) + \mathcal{L}_{rot} + \mathcal{L}_{seg}(\bar{\mathcal{M}}_{anc}, \mathcal{M}_{anc}) + \mathcal{L}_L$$

- 分割损失：$\mathcal{L}_{seg} = \text{BCE} + \text{Dice}$
- 旋转损失：$\mathcal{L}_{rot} = \text{BCE}(\bar{\mathcal{M}}_{rot}, \mathcal{M}_{rot})$
- LLM损失：$\mathcal{L}_L = \text{CE}(\bar{Y}, Y)$

### 推理过程

取 $\mathcal{M}_{loc}$ 中最大值对应的点，加上资产高度一半的偏移得到平移向量；取该点处 $\mathcal{M}_{rot}$ 的argmax得到旋转角。

## 实验关键数据

### 基准测试统计

PlaceIt3D-benchmark包含3,500个评测样例（142个ScanNet场景 × 20个资产），PlaceIt3D-dataset包含100,505个训练样例（565个场景 × 20个资产）。

### 主实验结果

| 方法 | 物理可行性 | 空间 | 旋转 | 可见性 | 语言遵从 | 全局准确率 | 完整成功率 |
|------|--------|------|------|--------|----------|-----------|-----------|
| OpenMask3D + rules | 61.6 | 28.6 | 6.5 | 53.4 | 21.8 | 29.2 | 11.7 |
| OpenMask3D + LLM | 5.8 | 35.3 | 10.5 | 61.5 | 18.4 | 26.7 | 1.6 |
| Reason3D (A) | 53.9 | 37.5 | 6.6 | 57.0 | 18.1 | 44.8 | 13.2 |
| **PlaceWizard (G)** | **58.8** | **56.6** | **17.3** | **61.2** | **25.9** | **54.9** | **15.0** |

PlaceWizard在所有整体指标上显著超越两种基线方法。LLM基线方法缺乏对3D几何的直接访问，物理可行性极低（5.8%）。

### 消融实验

| 变体 | 关键改动 | 语言遵从 | 全局准确率 | 完整成功率 |
|------|---------|----------|-----------|-----------|
| A (Reason3D) | Superpoints | 18.1 | 44.8 | 13.2 |
| B | 均匀池化 | 18.4 | 48.9 | 10.1 |
| C | +位置嵌入 | 20.0 | 50.4 | 10.9 |
| E | +锚点预测 | 22.2 | 42.5 | 12.3 |
| F | +旋转预测 | 20.8 | 51.0 | 11.4 |
| G (PlaceWizard) | +解码器资产特征 | **25.9** | **54.9** | **15.0** |

关键发现：
- 均匀空间池化比Superpoints提升全局准确率 +4.1%
- 位置嵌入进一步提升 +1.5%
- 锚点预测作为辅助任务改善语言遵从 +2.2%
- 将资产编码引入解码器是最大提升来源

## 亮点与洞察

1. **新颖任务定义**：首次系统定义了真实3D场景中语言引导物体放置问题，考虑了一对多歧义性
2. **完整生态**：同时提供基准测试（3,500例）、大规模训练集（100K+）和baseline方法
3. **端到端设计**：避免了规则基方法在推理时的昂贵碰撞检测，更具可扩展性
4. **均匀空间池化**的设计很巧妙，解决了Superpoints将大面积平面聚合为单特征的问题

## 局限性

- 仅支持水平面放置，无法处理如"把时钟挂在墙上"的场景
- 数据集基于人工规则生成，缺乏人工验证，边缘情况质量有限
- 未处理语言指令与实际场景不一致的情况
- 最严格指标（完整成功率）整体偏低（最高15.0%），任务本身极具挑战性

## 相关工作与启发

- **Reason3D**：PlaceWizard的基础架构，提供3D视觉定位能力
- **OpenMask3D**：开放词汇3D定位方法，作为对比基线
- **ScanNet**：提供真实房间的3D扫描数据
- 该任务对机器人操作、AR应用有直接价值，如通过语言指令让机器人放置物体

## 评分

⭐⭐⭐⭐ — 定义了有价值的新任务，贡献完整（任务+数据+方法），但当前方法性能仍有较大提升空间。

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/3d_vision/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)
- [\[ICCV 2025\] Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes](relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)
- [\[ICCV 2025\] Demeter: A Parametric Model of Crop Plant Morphology from the Real World](demeter_a_parametric_model_of_crop_plant_morphology_from_the_real_world.md)
- [\[ICCV 2025\] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)

<!-- RELATED:END -->
