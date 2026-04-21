---
title: >-
  [论文解读] CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation
description: >-
  [CVPR 2026][自动驾驶][对抗场景生成] 提出CompoSIA框架，通过对结构(Structure)、身份(Identity)、动作(Action)三因素的解耦控制，基于视频扩散模型生成可组合的对抗驾驶场景，实现身份编辑FVD降低17%、下游planner碰撞率提升173%，有效暴露自动驾驶系统的隐藏失败模式。
tags:
  - CVPR 2026
  - 自动驾驶
  - 对抗场景生成
  - 视频扩散模型
  - 解耦控制
  - 身份注入
  - 动作控制
  - 安全测试
---

# CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation

**会议**: CVPR 2026  
**arXiv**: [2603.12864](https://arxiv.org/abs/2603.12864)  
**代码**: [Yifever20002/CompoSIA](https://github.com/Yifever20002/CompoSIA)  
**领域**: 自动驾驶  
**关键词**: 对抗场景生成, 视频扩散模型, 解耦控制, 身份注入, 动作控制, 安全测试

## 一句话总结

提出CompoSIA框架，通过对结构(Structure)、身份(Identity)、动作(Action)三因素的解耦控制，基于视频扩散模型生成可组合的对抗驾驶场景，实现身份编辑FVD降低17%、下游planner碰撞率提升173%，有效暴露自动驾驶系统的隐藏失败模式。

## 研究背景与动机

自动驾驶系统的安全验证需要大量覆盖"长尾"危险场景的测试数据，但这类场景在现实中极其罕见且采集代价高。生成式方法成为构建对抗场景的关键途径，其核心需求是**对场景中各要素的独立可控编辑**——能"替换车辆外观而不改变轨迹"、"改变运动而保持身份"、"插入新视角而不破坏一致性"。

现有方法在可控性上存在明显局限：

- **MagicDrive-V2**：结构/身份/动作通过共享pathway处理，控制信号相互纠缠。编辑某一因素时不可避免地影响其他因素，难以实现精准的单因素编辑
- **DriveEditor**：支持部分编辑操作，但缺乏novel-view synthesis能力，只能在已有视角上操作，无法生成新摄像头角度的场景
- **通用视频生成模型(SVD等)**：身份保持(identity preservation)能力弱，生成过程中参考物体的外观特征容易丢失或漂移

CompoSIA的核心洞察是：**结构、身份、动作三个因素本质上具有不同的信息特性**——结构是全局空间布局、身份是局部外观纹理、动作是时序运动轨迹——因此需要用不同的机制分别注入，而不是用统一表示混合处理。

## 方法详解

### 整体框架

CompoSIA构建在Stable Video Diffusion (SVD)之上，通过三个专门设计的模块分别控制结构、身份和动作：

1. **结构条件**：3D bounding box序列 → 定义场景空间布局
2. **Noise-level Identity Injection**：单帧参考图 → 注入目标物体外观
3. **Hierarchical Dual-Branch Action Control**：轨迹条件 → 控制物体运动

三者在扩散模型的不同层级和不同机制上注入，天然避免信号干扰。

### 模块1：Noise-level Identity Injection

**问题**：如何从单张参考图提取身份信息，并在视频生成时保持外观一致性，同时不受参考图中姿态的影响？

**方案**：构建"替换-干净"(replaced-clean)训练对机制：

- **训练数据构造**：取训练视频的某帧，将目标物体区域替换为同类别但不同外观的物体→形成replaced clip；原始视频为clean clip。模型学习"接受replaced clip中的身份信息，输出clean clip的结果"
- **噪声水平注入策略**：在不同的去噪时间步$t$，对参考帧施加不同程度的噪声后再注入。高噪声水平(大$t$)时身份信息被模糊→模型依赖结构条件确定布局；低噪声水平(小$t$)时身份细节清晰→模型精确复制外观纹理
- **姿态无关性**：通过在不同噪声水平下训练，模型自动学会从参考图中提取pose-agnostic的身份表征——高噪声时只保留颜色/纹理等低频特征，自然过滤掉姿态信息

这种设计比IP-Adapter等方法更简洁，不需要额外的编码器，直接利用扩散模型自身的噪声机制实现信息解耦。

### 模块2：Hierarchical Dual-Branch Action Control

**问题**：物体动作既包括局部形变(转弯时的车身倾斜)又包括全局轨迹(直行/左转/右转)，如何同时精确控制两者？

**方案**：双分支层次化控制：

**分支1 - AdaLN局部残差控制**：
- 将轨迹条件通过Adaptive Layer Normalization (AdaLN)注入到扩散模型的中间层
- AdaLN以残差方式工作：$\gamma \cdot \text{LayerNorm}(x) + \beta$，其中$\gamma, \beta$由轨迹条件计算
- 优势：参数量少、收敛快，适合捕捉逐帧的局部运动变化
- 类比：类似于LoRA的"小幅度修正"思路

**分支2 - Camera Attention with PRoPE全局轨迹控制**：
- 引入Positional Rotary Position Embedding (PRoPE)，将3D空间位置编码嵌入注意力计算
- Camera Attention处理多视角一致性：在自注意力中同时考虑不同相机视角间的空间关系
- 优势：长程依赖建模能力强，适合全局轨迹的时序一致性
- PRoPE将相机外参矩阵编码为旋转位置嵌入，使模型感知3D空间中的相对位置关系

**层次化融合**：AdaLN负责"怎么动"(局部形变)，Camera Attention负责"往哪去"(全局轨迹)，两者互补。

### 模块3：结构条件

- 输入为3D bounding box序列，覆盖整个视频时间跨度
- 每个bbox定义物体在3D空间中的位置、尺寸和朝向
- 通过ControlNet-like的方式注入扩散模型，提供coarse-level的空间约束
- 结构条件仅约束"物体在哪"，不限制外观和动作细节

### 训练策略

- 基于nuScenes数据集训练，利用其多视角标注和3D bbox GT
- 三个模块分阶段训练：先固定SVD权重训练结构条件→加入Identity Injection→最后加入Action Control
- 损失函数为标准扩散去噪损失$\mathcal{L} = \mathbb{E}_{t,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2]$，其中$c$包含三种条件

## 实验关键数据

### 生成质量

| 方法 | FVD↓ | FID↓ | 身份保持度↑ |
|------|------|------|-----------|
| MagicDrive-V2 | 基线 | 基线 | 基线 |
| DriveEditor | — | — | — |
| **CompoSIA** | **-17%** | 最优 | 最优 |

### 可控性指标

| 操控维度 | 指标 | 改进幅度 |
|---------|------|---------|
| 身份编辑 | FVD | -17% vs 最强基线 |
| 旋转控制 | 旋转误差 | -30% |
| 平移控制 | 平移误差 | -47% |

### 下游安全测试

将CompoSIA生成的对抗场景用于自动驾驶planner的stress testing：

| 指标 | 正常场景 | CompoSIA对抗场景 | 变化 |
|------|---------|-----------------|------|
| 3秒碰撞率 | 基线 | +173% | **大幅暴露安全隐患** |

这一结果表明生成的对抗场景确实能有效探测到planner在正常数据上无法暴露的失败模式。

### 消融实验

| 消融条件 | FVD变化 | 说明 |
|---------|--------|------|
| 去除Noise-level策略(直接concat) | 恶化 | 身份-姿态耦合，生成姿态被参考图锁定 |
| 去除AdaLN分支(仅Camera Attention) | 恶化 | 局部运动细节缺失 |
| 去除Camera Attention(仅AdaLN) | 恶化 | 长程轨迹不一致 |
| 固定噪声水平(不分级) | 恶化 | 无法平衡身份细节和姿态灵活性 |

## 亮点与洞察

- **三因素解耦的哲学**：结构(空间)、身份(外观)、动作(时序)各有对应的最优注入机制——3D bbox做空间约束、噪声水平做外观注入、双分支做时序控制。这种"因素-机制对齐"比统一处理优雅得多
- **Noise-level Identity Injection极为巧妙**：不引入任何额外编码器，仅利用扩散模型自身的噪声schedule实现信息粒度控制。高噪声=低频信息(颜色/纹理)，低噪声=高频信息(细节/edges)，天然实现pose-agnostic的外观提取
- **实际安全价值突出**：碰撞率+173%不是"指标涨了"，而是"真的找到了planner在正常测试中不会暴露的bug"——这是自动驾驶安全验证的切实需求
- **PRoPE的3D感知**：将相机外参编码为位置嵌入，优雅地解决了多视角一致性问题

## 局限与展望

1. **训练数据依赖nuScenes**：nuScenes规模有限(~1000场景)，生成多样性受限于训练分布。迁移到更大规模数据集(如Waymo Open)可能进一步提升
2. **仅支持vehicle类别**：当前主要针对车辆的身份编辑和动作控制，对行人、自行车等其他道路参与者的支持需要额外验证
3. **3D bbox标注依赖**：结构条件需要精确的3D标注，限制了在无标注数据上的应用。可探索从2D检测自动估计3D布局
4. **时序长度限制**：SVD基础模型限制了生成视频的时序长度，长片段场景(如完整交叉路口通过)可能需要分段生成+拼接
5. **对抗场景的合理性验证**：碰撞率提升说明场景有效，但部分生成场景的物理合理性(如不可能的加速度)需要进一步约束
6. **多Agent交互**：当前更偏向单Agent编辑，多物体间的交互动力学(如避让行为)建模尚不充分

## 相关工作与启发

- **MagicDrive系列**：驾驶场景生成的代表工作 → CompoSIA在可控性上显著超越
- **DriveEditor**：支持编辑但缺乏新视角合成 → CompoSIA的Camera Attention补齐此能力
- **SVD (Stable Video Diffusion)**：基础视频生成模型 → CompoSIA展示了如何在其上构建精细控制
- **IP-Adapter**：身份保持的另一条路线 → Noise-level方案更简洁且不需额外编码器
- **启发**：解耦控制的思路可推广到更广泛的场景编辑任务(室内/机器人环境)。随机生成→有目的的对抗生成是安全验证的核心需求

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4.0 | 三因素解耦框架设计精妙，Noise-level Identity Injection有原创性 |
| 实用性 | 4.5 | 直接服务安全验证，碰撞率+173%证明实际价值 |
| 实验充分度 | 4.0 | 生成质量+可控性+下游任务三层验证，消融充分 |
| 写作质量 | 4.0 | 结构清晰，三因素解耦的叙事逻辑自然 |
| **综合** | **4.1** | 自动驾驶安全验证方向的强工作，技术设计与应用场景高度契合 |

<!-- RELATED:START -->

## 相关论文

- [Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](composing_driving_worlds_through_disentangled_cont.md)
- [Steerable Adversarial Scenario Generation through Test-Time Preference Alignment (SAGE)](../../ICLR2026/autonomous_driving/steerable_adversarial_scenario_generation_through_test-time_preference_alignment.md)
- [Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)
- [MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving](meanfuser_fast_one-step_multi-modal_trajectory_generation_and_adaptive_reconstru.md)
- [Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors](points-to-3d_structure-aware_3d_generation_with_point_cloud_priors.md)

<!-- RELATED:END -->
