---
title: >-
  [论文解读] Decoupled Diffusion Sparks Adaptive Scene Generation
description: >-
  [ICCV2025][自动驾驶][场景生成] 提出 Nexus，一个基于解耦扩散的自适应驾驶场景生成框架，通过独立噪声状态实现目标导向与实时响应的统一，将位移误差降低 40%，并构建了包含 540 小时安全关键驾驶数据的 Nexus-Data。 - 自动驾驶数据集中多样性至关重要，但关键长尾场景极为稀缺 - 现有场景生成方法…
tags:
  - "ICCV2025"
  - "自动驾驶"
  - "场景生成"
  - "解耦扩散"
  - "自动驾驶仿真"
  - "安全关键场景"
  - "交通布局"
---

# Decoupled Diffusion Sparks Adaptive Scene Generation

**会议**: ICCV2025  
**arXiv**: [2504.10485](https://arxiv.org/abs/2504.10485)  
**代码**: [opendrivelab.com/Nexus](https://opendrivelab.com/Nexus)  
**领域**: 自动驾驶  
**关键词**: 场景生成, 解耦扩散, 自动驾驶仿真, 安全关键场景, 交通布局

## 一句话总结

提出 Nexus，一个基于解耦扩散的自适应驾驶场景生成框架，通过独立噪声状态实现目标导向与实时响应的统一，将位移误差降低 40%，并构建了包含 540 小时安全关键驾驶数据的 Nexus-Data。

## 研究背景与动机

- 自动驾驶数据集中多样性至关重要，但关键长尾场景极为稀缺
- 现有场景生成方法面临两个核心矛盾：
    - **全序列扩散**（如 SceneDiffuser）：能实现目标导向生成（通过 inpainting），但无法及时响应环境变化
    - **自回归预测**（如 GUMP）：能实时响应环境反馈，但无法感知目标状态进行精确控制
- 现有方法无法同时提供**实时响应性 (Reactivity)** 和**目标导向性 (Goal Orientation)**
- 公开数据集主要包含安全驾驶行为，缺少足够的高风险场景

## 方法详解

### 核心思想：噪声作为软掩码

- 关键洞察：将不同噪声水平视为不同程度的掩码
    - 低噪声 token → 已确定的目标/历史状态（类似 hard mask）
    - 高噪声 token → 待生成的未来状态
- 统一了扩散模型（噪声轴掩码）和自回归预测（时间轴掩码）为**三轴掩码建模**
- 每个 token 拥有独立噪声状态，形成噪声矩阵 $\mathbf{k} \in (0,1]^{A \times \mathcal{T}}$

### 场景编码与表示

- **Agent Tensor**: $\mathbf{x} \in \mathbb{R}^{A \times \mathcal{T} \times D}$，包含坐标、朝向、速度、尺寸
- **Map Tensor**: $\mathbf{c} \in \mathbb{R}^{L \times N \times D'}$，道路拓扑信息
- 使用 Perceiver IO 将地图编码为固定长度 token
- Agent token 添加随机独立噪声后，使用二维旋转位置编码（物理时间 + 去噪步骤）

### 噪声掩码训练（目标导向）

- 训练时为每个 token 独立采样噪声水平，而非全序列统一添加噪声
- 模型学习从部分软掩码 token 恢复完整序列，跟随低噪声 token 提供的引导
- 优化目标：$\forall \mathbf{k} \in (0,1]^{A \times \mathcal{T}}, \min_\theta \mathbb{E} \|(\epsilon - \epsilon_\theta(g(\mathbf{x}^0, \mathbf{k}); \mathbf{c}, \mathbf{k}))\|_2^2$
- 推理时：历史和目标设为低噪声，其他设为高噪声，即可实现条件生成

### Diffusion Transformer 架构

- 基于 DiT 构建，包含多种注意力交互：
    - **地图交叉注意力**：agent 查询 map，建模 agent-map 交互
    - **时间注意力**：捕获轨迹连续性
    - **空间注意力**：建模空间交互（跟车、让行等）
- 有效性掩码排除无效/跳过的 token
- 使用 AdaLN 条件化 transformer block

### 噪声感知调度（实时响应）

- 定义调度矩阵 $\mathcal{K} \in [\mathbf{k}]^M$，编码每个 agent 在每个时步的噪声水平变化
- **Chunk 机制**：
    - 每个 chunk 包含历史帧、待去噪帧和可选目标 token
    - 每个去噪步骤后，最低噪声 token 弹出（生成完成），高噪声帧推入
    - 环境变化可直接覆写 agent 状态并降低噪声
- **调度策略**：
    - **金字塔调度**：token 从 chunk 一端进出，逐步生成
    - **梯形调度**：token 从两端进出，支持双向目标引导
    - 两种策略响应时间仅 0.16 秒（vs 全序列的 4.96 秒）

### 行为对齐的分类器引导

- 在每个去噪步骤应用纠正函数：
    - 沿中心线反方向分离重叠 agent（避碰）
    - 平滑轨迹
    - 将 agent 拉向最近车道（保持在道路上）

### Nexus-Data: 安全关键场景数据集

- 使用 MetaDrive 仿真器 + ScenarioNet 格式化场景
- 基于 CAT 对抗学习生成高风险交互（cut-in、急刹、碰撞）
- 自动化过滤：仅 36.9% 产生有效碰撞，再过滤离路、无效轨迹等
- 最终收集 540 小时高质量安全关键驾驶场景

## 实验关键数据

### 主要生成性能对比（nuPlan 数据集，8 秒预测）

| 方法 | ADE↓ | 离路率↓ | 碰撞率↓ | 不稳定性↓ | 时间(s) |
|------|------|---------|---------|-----------|---------|
| IDM | 10.52 | 9.85 | 10.17 | 6.30 | 2.16 |
| Diffusion Policy | 7.80 | 13.9 | 14.92 | 12.71 | 6.59 |
| SceneDiffuser | 5.99 | 8.53 | 11.78 | 9.64 | 5.34 |
| GUMP | 1.93 | 7.73 | 7.85 | 16.18 | 5.59 |
| **Nexus** | **1.28** | 6.89 | **1.62** | **4.63** | 2.79 |
| **Nexus-Full** | **1.12** | **6.25** | **1.56** | **3.17** | 2.93 |

### 调度策略对比

| 调度策略 | ADE↓ | 响应时间(s) | 总时间(s) |
|----------|------|-------------|-----------|
| 自回归 | 1.48 | 4.96 | 79.36 |
| 全序列 | 1.28 | 4.96 | 4.96 |
| 金字塔 | 1.53 | **0.16** | 7.68 |
| 梯形 | 1.39 | **0.16** | 6.20 |
| 梯形+反馈 | **1.17** | **0.16** | 6.20 |

### 消融实验

| 组件 | ADE↓ (条件生成) |
|------|-----------------|
| Baseline (Diffusion Policy) | 7.53 |
| + 噪声掩码训练 | 3.42 |
| + 位置编码 | 1.44 |
| + Nexus-Data | 1.32 |
| + 分类器引导 | 1.25 |

### 闭环驾驶世界生成器评测

| 方法 | 响应式得分↑ | 碰撞得分↑ | 进度得分↑ |
|------|------------|-----------|-----------|
| Oracle (GT) | 82.8 | 89.5 | 97.0 |
| Diffusion Policy | 61.6 | 81.9 | 90.2 |
| SceneDiffuser | 57.2 | 74.7 | 91.6 |
| **Nexus** | **73.0** | **84.9** | **95.0** |

### 数据增强效果

使用 Nexus 生成合成数据增强规划模型训练，闭环得分从 48.11 提升至 57.86（+20%）。

## 优势与局限

**优势**：
- 首次统一目标导向与实时响应的场景生成
- 噪声掩码训练巧妙地将扩散与自回归预测统一
- 响应时间仅 0.16 秒，适合在线闭环仿真
- 碰撞率仅 1.56%，远低于所有基线方法
- Nexus-Data 提供了大规模安全关键场景训练数据

**局限**：
- 仅生成结构化交通布局，不直接合成视频
- 闭环训练端到端驾驶模型尚需视频合成支持
- 合成数据过少（3×）反而降低性能，需要足够的规模（30×+）

## 个人思考

- 将噪声状态视为软掩码是非常优雅的统一框架，本质上回答了"扩散模型和自回归模型能否统一"的问题
- chunk 滑动窗口的设计使得扩散模型第一次具备了真正的在线生成能力
- Nexus-Data 的构建思路（对抗学习 + 自动过滤）为安全关键场景的大规模获取提供了范例
- 数据增强实验（60× 合成数据 → +20% 闭环得分）有力地说明了场景生成的实际价值
- 与 NeRF 的结合展示了从布局生成到视觉渲染的完整链路，是 world model 的重要组成部分

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [\[ICCV 2025\] Controllable 3D Outdoor Scene Generation via Scene Graphs](controllable_3d_outdoor_scene_generation_via_scene_graphs.md)
- [\[NeurIPS 2025\] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](../../NeurIPS2025/autonomous_driving/cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](../../NeurIPS2025/autonomous_driving/x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [\[ICCV 2025\] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion](distilling_diffusion_models_to_efficient_3d_lidar_scene_completion.md)

</div>

<!-- RELATED:END -->
