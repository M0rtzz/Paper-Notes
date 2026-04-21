---
title: >-
  [论文解读] SceneCrafter: Controllable Multi-View Driving Scene Editing
description: >-
  [CVPR 2025][自动驾驶][多视角一致性] SceneCrafter 提出了一个基于多视角扩散模型的驾驶场景编辑框架，通过 teacher-student 两阶段训练范式生成高质量合成配对数据，支持天气/时间全局编辑和前景目标增删的局部编辑，同时保持跨相机的 3D 几何一致性。
tags:
  - CVPR 2025
  - 自动驾驶
  - 多视角一致性
  - 驾驶场景编辑
  - 扩散模型
  - 合成数据生成
  - 传感器仿真
---

# SceneCrafter: Controllable Multi-View Driving Scene Editing

**会议**: CVPR 2025  
**arXiv**: [2506.19488](https://arxiv.org/abs/2506.19488)  
**代码**: 无  
**领域**: 自动驾驶 / 场景编辑  
**关键词**: 多视角一致性, 驾驶场景编辑, 扩散模型, 合成数据生成, 传感器仿真

## 一句话总结

SceneCrafter 提出了一个基于多视角扩散模型的驾驶场景编辑框架，通过 teacher-student 两阶段训练范式生成高质量合成配对数据，支持天气/时间全局编辑和前景目标增删的局部编辑，同时保持跨相机的 3D 几何一致性。

## 研究背景与动机

**领域现状**：自动驾驶仿真需要逼真的传感器数据来评估全栈系统。现有方法分为基于重建的（神经场/原语）和基于生成的（图像/视频生成模型）两大类。重建方法忠实于真实场景但缺乏灵活编辑能力，纯生成方法缺乏与真实场景的关联性。

**现有痛点**：图像编辑在驾驶仿真中面临三大独特挑战：(1) 多相机之间需保持 3D 一致性；(2) 训练数据几乎全是"有车街道"，模型难以学到"空旷街道"的先验；(3) 获取全局/局部编辑的配对训练数据非常困难。

**核心矛盾**：编辑模型需要配对数据进行监督训练，但真实配对数据（如同一场景在不同天气/不同车辆布局下的图片对）极难获取。直接使用 Prompt-to-Prompt 或 RePaint 等现有方法在多视角驾驶场景下效果不佳。

**本文目标**：构建一个统一的多视角驾驶场景编辑器，同时支持全局编辑（天气/时间）和局部编辑（目标增删），并保持跨视角的几何一致性。

**切入角度**：作者采用 teacher-student 架构——先训练 teacher 模型生成高质量合成配对数据，再用这些数据训练统一的 student 编辑模型。这种间接方式绕过了直接获取配对数据的困难。

**核心 idea**：通过改进的 Prompt-to-Prompt（替换 self-attention 而非 cross-attention）生成全局编辑配对数据，通过 masked training + multi-view repaint + alpha blending 生成局部编辑配对数据，最终蒸馏为统一的场景编辑模型。

## 方法详解

### 整体框架

SceneCrafter 的 pipeline 分两大阶段：第一阶段训练两个 teacher 模型分别生成全局和局部编辑的合成配对数据；第二阶段用生成的 100 万对合成数据训练统一的 student 编辑模型。输入是多视角源图像（8 个相机）及编辑条件，输出是编辑后的多视角图像。

### 关键设计

1. **Teacher 模型的多模态条件控制**:

    - 功能：支持天气、时间、HD 地图、目标框等多种条件的场景生成
    - 核心思路：全局条件（天气用 CLIP 文本编码、时间用太阳角度位置编码）和局部条件（HD 地图用 PerceiverIO 降维到 512 token、目标框用 MLP 编码）统一通过 cross-attention 注入 U-Net。前景 mask 与 raymap 沿通道维度 concat 到输入。训练时对每种条件以 10% 概率 dropout，提升鲁棒性
    - 设计动机：丰富的条件信号将场景几何结构锚定在生成过程中，使得在修改高层属性（如天气）时能保留精细的几何细节

2. **改进的 Prompt-to-Prompt 生成全局编辑配对数据**:

    - 功能：生成天气/时间变化的几何一致配对数据
    - 核心思路：与原始 P2P 冻结 cross-attention 权重不同，SceneCrafter 冻结所有 self-attention 层的权重，因为全局编辑应影响图像所有区域但保留像素级布局。此外引入更多条件信号（目标框、HD 地图）提升几何一致性，并只使用白天作为源图像时间以获得最佳生成质量。生成的源目标对随机翻转顺序作为训练数据
    - 设计动机：在多视角驾驶场景中没有文本 token 可供操作，self-attention 权重的替换能更好地在像素级保持几何一致性

3. **Masked Training + Multi-View Repaint + Alpha Blending 生成局部编辑配对数据**:

    - 功能：生成目标增删的多视角一致配对数据
    - 核心思路：Masked training 在训练时将前景区域噪声设为零、只对背景去噪并只在背景像素计算损失 $\mathbf{z}_t = (1-\mathbf{m}) \odot (\alpha_t \mathbf{z}_0 + \sigma_t \epsilon) + \mathbf{m} \odot \mathbf{z}_0$，使模型学到"空街道"先验。Multi-view repaint 在每步反向过程中同时处理所有视角的前景区域，保证多视角一致。Alpha blending 将"空街道"和"满街道"按采样的 mask 混合，生成任意数量目标的配对数据
    - 设计动机：直接用 RePaint 擦除目标效果差，因为模型是在"满街道"数据上训练的，没有空街道先验。Masked training 以自监督方式巧妙地从有车数据中学习无车先验

### 损失函数 / 训练策略

Student 模型将源图像的 latent 直接 concat 到去噪 latent 中（而非 cross-attention），实验证明 concat 方式对像素级条件效果更好（FID 降低 13.1）。模型从全局编辑 teacher 初始化权重，在 128 个 TPU v5 上训练 100K 迭代，学习率 $1e^{-5}$，batch size 128。推理使用 50 步去噪 + classifier-free guidance。

## 实验关键数据

### 主实验

| 方法 | 时间编辑 FID↓ | 时间编辑 CLIP↑ | 时间编辑 用户偏好↑ | 天气编辑 FID↓ | 天气编辑 CLIP↑ | 天气编辑 用户偏好↑ |
|------|-------------|--------------|----------------|-------------|--------------|----------------|
| SDEdit | 60.4 | 0.204 | 2.7% | 78.3 | 0.203 | 1.8% |
| P2P* | 46.8 | 0.223 | 13.6% | 55.4 | 0.207 | 12.7% |
| **SceneCrafter** | **37.2** | **0.220** | **83.6%** | **38.9** | **0.221** | **85.5%** |

局部编辑（FID↓）：2D-RePaint 移除/插入 30.6/31.9，MV-RePaint 26.0/28.5，SceneCrafter **23.5/21.7**。

### 消融实验

| 替换 Self-Attn | 增加条件 | 白天源图 | FID↓ | CLIP↑ |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | 57.1 | 0.204 |
| ✓ | ✗ | ✗ | 41.5 | 0.202 |
| ✓ | ✓ | ✗ | 39.9 | 0.214 |
| ✓ | ✓ | ✓ | **36.2** | **0.223** |

### 关键发现

- 替换 self-attention 权重比 cross-attention 能更好保持像素级几何一致性
- 引入更多条件信号（目标框+HD地图）显著提升可控性
- 只用白天作源图像可大幅提升生成质量
- Concat 方式注入源图像优于 cross-attention 方式（FID 50.3 vs 37.2）
- 完整模型的 3D LPIPS 达到 0.187，与真实数据的 0.186 相当

## 亮点与洞察

- **Masked training** 非常巧妙——在几乎所有训练数据都包含前景目标的情况下，通过差异化噪声策略让模型自监督地学到了空街道先验
- 替换 self-attention 而非 cross-attention 的发现具有启发性，因为驾驶场景的条件不是文本 token
- 使用 box 条件而非 mask 条件进行目标编辑，在小目标上效果更好，避免了分割不精确导致的边界问题
- Teacher-student 训练范式有效解决了配对数据获取的难题

## 局限与展望

- 依赖 Waymo 私有数据（约 1400 万段驾驶视频），不易复现
- 编辑能力受限于 teacher 模型的生成质量
- 只处理静态场景编辑，未扩展到视频时序一致性
- 使用 box 条件虽好但无法精确控制目标外观细节
- 可探索扩展到更多编辑类型（如道路结构变化）

## 相关工作与启发

- **Prompt-to-Prompt** 的多视角扩展策略值得参考：替换 self-attention 实现几何保持
- **RePaint** 在多视角场景的不足启发了 masked training 方案
- Teacher-student 合成数据范式可推广到其他缺乏配对数据的编辑任务

## 评分

- **新颖性**: 8/10 — Masked training 和 self-attention 替换策略新颖
- **实验充分度**: 8/10 — 消融完整，提出了新 3D 一致性指标，有用户研究
- **写作质量**: 8/10 — 结构清晰，动机推导自然
- **价值**: 8/10 — 对自动驾驶仿真有直接工程价值

<!-- RELATED:START -->

## 相关论文

- [HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles](../../CVPR2026/autonomous_driving/horizonforge_driving_scene_editing_with_any_trajectories_and_any_vehicles.md)
- [Controllable 3D Outdoor Scene Generation via Scene Graphs](../../ICCV2025/autonomous_driving/controllable_3d_outdoor_scene_generation_via_scene_graphs.md)
- [UniScene: Unified Occupancy-centric Driving Scene Generation](uniscene_unified_occupancy-centric_driving_scene_generation.md)
- [Generating Multimodal Driving Scenes via Next-Scene Prediction](generating_multimodal_driving_scenes_via_next-scene_prediction.md)
- [Distilling Multi-modal Large Language Models for Autonomous Driving](distilling_multi-modal_large_language_models_for_autonomous_driving.md)

<!-- RELATED:END -->
