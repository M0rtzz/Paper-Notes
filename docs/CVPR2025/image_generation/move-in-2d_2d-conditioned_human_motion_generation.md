---
title: >-
  [论文解读] Move-in-2D: 2D-Conditioned Human Motion Generation
description: >-
  [CVPR 2025][motion generation] 提出以 2D 场景图像+文本为条件的人体运动生成新任务，构建 30 万视频规模的 HiC-Motion 数据集，训练基于 in-context conditioning 的扩散 Transformer 模型，生成可投影到场景的人体运动序列。
tags:
  - CVPR 2025
  - motion generation
  - diffusion model
  - scene-aware
  - human video
  - SMPL
---

# Move-in-2D: 2D-Conditioned Human Motion Generation

**会议**: CVPR 2025  
**arXiv**: [2412.13185](https://arxiv.org/abs/2412.13185)  
**代码**: [项目页](https://hhsinping.github.io/Move-in-2D)  
**领域**: image_generation  
**关键词**: motion generation, 2D scene conditioning, diffusion transformer, in-context learning, HiC-Motion

## 一句话总结

定义 2D 场景图像+文本条件下的人体运动生成新任务，构建 30 万级 HiC-Motion 数据集，通过 in-context conditioning 扩散 Transformer 生成可自然投影到场景的运动序列，赋能下游人体视频生成。

## 研究背景与动机

**领域现状**: 人体视频生成进展显著，最有效的方法依赖预定义的人体运动序列（如 OpenPose、DensePose）作为控制信号。现有方法通常从其他视频中提取运动，限制了动作类型和全局场景适配。

**现有痛点**:
- 仅文本条件的运动生成（MDM、MLD）无法保证与目标场景的空间兼容性
- 3D 场景条件的方法（HUMANISE、SceneDiff）依赖 3D mesh/点云，获取成本高，且主要限于简单室内动作
- 缺乏包含运动序列、文本描述和场景图像三要素的大规模数据集

**核心矛盾**: 需要场景感知的运动生成，但 3D 场景获取代价高昂；2D 图像无处不在但此前无人将其作为运动生成的条件。

**本文目标**: 提出以单张 2D 场景图像为条件的运动生成范式，使生成的运动投影到 2D 平面后与场景自然兼容。

**切入角度**: 用 2D 图像替代 3D 场景作为条件模态，极大扩展可用场景范围（室内外、野外），并构建对应数据集训练扩散模型。

**核心 idea**: 2D 场景图像提供了语义和空间布局信息，无需 3D 重建即可生成场景兼容的人体运动。

## 方法详解

### 整体框架

1. 输入: 背景场景图像 $s$ + 文本提示 $p$
2. 运动表示: 256 帧序列，每帧包含 23 个 SMPL 关节的 6D 旋转 $\theta_b$、全局朝向 $\theta_g$、相机平移 $\pi \in \mathbb{R}^3$
3. 模型: 基于扩散 Transformer 的条件生成，支持 CFG

### 关键设计

#### 1. HiC-Motion 数据集构建

从 3000 万开放域网络视频中筛选出 30 万包含单人运动的视频：
- 通过 Keypoint R-CNN 和 OpenPose 检测过滤单人视频
- 保留运动帧数 > 256 的视频
- 用 4D-Humans 提取 SMPL 格式的伪真值运动
- 用 Mask R-CNN + 基础 inpainting 模型去除人体获取背景图
- 覆盖 1000+ 类别（日常活动、体育等），远超前作

#### 2. 多条件 Transformer（Multi-Conditional Transformer）

三种条件注入机制：
- **In-context conditioning**: 文本 token（CLIP-B 编码）和场景 token（DINO-B 编码为 240 个 patch token）拼接到运动序列作为额外 token
- **AdaLN**: 扩散时间步通过自适应层归一化注入，增强时间平滑性
- **Cross-attention**: 备选方案，但实验证明 in-context 更优

最终架构: 8 个 Transformer block，512 隐藏维度，4 注意力头，1000 步余弦噪声调度。

#### 3. 两阶段训练策略

- **第一阶段**: 全量 30 万视频训练 60 万迭代，学习场景语义和多样运动生成
- **第二阶段**: 混合数据集（60% 大运动 + 40% 固定背景）微调 60 万迭代，解耦相机运动影响

### 损失函数

MSE 重建损失 $\mathcal{L}_{mse} = \mathbb{E}_{x_0, t} \| x_0 - \mathcal{M}(x_t | t, c) \|^2$，配合 CFG（联合对文本和场景条件的 guidance）。

## 实验关键数据

### 主实验：定量评估

| 方法 | FID↓ | Accuracy↑ | Diversity↑ | Multimodality↑ |
|------|------|-----------|------------|----------------|
| MDM | 164.6 | 0.325 | 24.8 | 18.9 |
| MLD | 85.9 | 0.322 | 25.1 | 19.5 |
| SceneDiff（3D） | 543.8 | 0.203 | 4.2 | 3.9 |
| HUMANISE（3D） | 159.9 | 0.225 | 23.3 | 20.0 |
| MDM+（HiC训练）| 46.0 | 0.620 | 23.0 | 17.6 |
| **Ours** | **44.6** | **0.661** | **26.0** | 20.1 |

### VLM 自动化评估（GPT-4o 打分，0-5 分）

| 方法 | Scene-Align↑ | Text-Align↑ | Quality↑ | Total↑ |
|------|-------------|-------------|----------|--------|
| MDM | 2.25 | 1.35 | 1.50 | 5.10 |
| MLD | 2.85 | 1.95 | 1.90 | 6.70 |
| **Ours** | **3.55** | **2.70** | **2.85** | **9.10** |

### 消融实验：条件注入方式

| Timestep | Text | Scene | FID↓ | Accuracy↑ |
|----------|------|-------|------|-----------|
| AdaLN | In-Context | **In-Context** | **44.6** | **0.661** |
| AdaLN | In-Context | Cross-Attn | 47.7 | 0.567 |
| In-Context | In-Context | In-Context | 62.9 | 0.554 |

### 关键发现

1. 在 HiC-Motion 数据集上训练的 MDM+ 较原始 MDM 的 FID 降低 72%，验证了大规模数据的重要性
2. In-context conditioning 对场景和文本都优于 cross-attention，因为共享 token 空间促进跨模态交互
3. Ours vs Ours-scene: 文本条件使 Accuracy 提升 37%，但降低了 Multimodality（文本约束减少多样性）

## 亮点与洞察

- **新任务定义**: 2D 场景条件运动生成填补了文本-only 和 3D 场景之间的空白，实用性强
- **数据集规模突破**: 30 万运动序列，远超 HumanML3D（14.6K）和 Motion-X（81K），覆盖室内外多样场景
- **In-context learning 的跨域应用**: 将 LLM 的 in-context 范式迁移到运动扩散模型，文本和图像 token 在共享空间中交互
- **两步式视频生成管线**: 先生成运动控制信号→再用 Champ/Gen-3 渲染最终视频，质量显著优于 SVD 直接生成
- **2D 投影的设计选择**: 模型额外预测相机平移参数 $\pi$，使运动可通过透视投影自然映射到图像平面

## 局限性

1. 不控制相机运动——生成运动中的位移可能混入相机平移效果
2. 两步式视频生成管线未与运动生成联合优化
3. 从互联网视频提取的运动是伪真值（4D-Humans），存在系统性噪声
4. 场景兼容性依赖隐式学习，无法显式保证物理合理性（如不穿透地面）

## 相关工作与启发

- **MDM/MLD**: 文本条件运动生成的主流方法，但缺乏场景感知能力
- **HUMANISE**: 3D 场景+文本条件的先驱，但限于 643 个 ScanNet 室内场景
- **HiC 数据集系列**: 大规模人体中心视频数据集的构建范式启发了 HiC-Motion
- **启发**: 2D 条件范式可扩展到其他场景交互任务（机器人操控规划、AR 内容生成），降低 3D 重建门槛

## 评分

⭐⭐⭐⭐ — 新任务定义有价值，数据集工程扎实，实验充分（含 VLM 评估和视频生成应用）；方法本身（扩散 Transformer + in-context）较为标准，技术新颖性中等。
