---
title: >-
  [论文解读] IGen: Scalable Data Generation for Robot Learning from Open-World Images
description: >-
  [CVPR 2026][机器人][机器人学习] IGen 从单张开放世界图像出发，通过3D场景重建→VLM任务规划→SE(3)动作生成→点云合成→帧渲染，自动生成大规模视觉-动作训练数据，仅用生成数据训练的策略即可完成真实世界操作。
tags:
  - "CVPR 2026"
  - "机器人"
  - "机器人学习"
  - "数据生成"
  - "开放世界图像"
  - "视觉运动策略"
  - "3D重建"
---

# IGen: Scalable Data Generation for Robot Learning from Open-World Images

**会议**: CVPR 2026  
**arXiv**: [2512.01773](https://arxiv.org/abs/2512.01773)  
**代码**: [https://chenghaogu.github.io/IGen/](https://chenghaogu.github.io/IGen/)  
**领域**: 机器人  
**关键词**: 机器人学习, 数据生成, 开放世界图像, 视觉运动策略, 3D重建

## 一句话总结
IGen 从单张开放世界图像出发，通过3D场景重建→VLM任务规划→SE(3)动作生成→点云合成→帧渲染，自动生成大规模视觉-动作训练数据，仅用生成数据训练的策略即可完成真实世界操作。

## 研究背景与动机
1. **领域现状**：通用机器人策略需要大规模视觉-动作配对数据，但真实数据采集昂贵且局限于特定环境。
2. **现有痛点**：Real-to-Sim方法需要对物理工作空间进行显式重建；视频生成方法无法提供显式动作且对复杂长任务表现差。
3. **核心矛盾**：开放世界图像极其丰富多样但缺乏机器人相关的动作信息，无法直接用于策略学习。
4. **本文目标**：从非结构化的开放世界图像中自动生成接地的视觉-动作数据。
5. **切入角度**：将2D像素转化为结构化3D表示，然后利用VLM进行任务规划和动作生成。
6. **核心idea**：图像→3D点云+关键点→VLM高层规划+低层控制→SE(3)轨迹→点云序列合成→帧渲染。

## 方法详解

### 整体框架
三阶段pipeline：（1）场景重建：将输入图像转化为可操作的机器人工作空间（3D点云+空间关键点）；（2）动作规划：VLM推理任务指令生成高层计划和低层控制；（3）观测合成：基于SE(3)轨迹合成动态点云序列并逐帧渲染。

### 关键设计

1. **从像素到结构化3D表示**:
    - 功能：将非结构化2D图像转化为机器人可理解的3D表示
    - 核心思路：用单目几何基础模型估计深度→VLM识别任务相关物体→SAM分割物体掩码→DINOv2提取特征+K-means聚类得到空间关键点。对操作目标物体用3D生成重建完整形状，背景用图像修复+深度反投影生成点云。
    - 设计动机：机器人操作需要3D空间理解，直接在2D图像上规划无法提供物理接地的动作。

2. **基于VLM的空间规划**:
    - 功能：从任务指令生成可执行的机器人动作序列
    - 核心思路：VLM将场景理解和任务描述转化为高层计划（如"抓取→移动→放置"），再映射为低层SE(3)末端执行器位姿序列。利用空间关键点作为动作的空间锚点。
    - 设计动机：VLM具有强大的场景理解和推理能力，可以将自然语言指令接地到3D空间中的具体操作。

3. **无仿真的点云合成**:
    - 功能：生成动作一致的视觉观测序列
    - 核心思路：用SE(3)轨迹对场景点云进行刚体运动变换，生成操作过程中的动态点云序列。然后逐帧渲染为RGB观测。避免了传统方法需要构建完整物理仿真环境的开销。
    - 设计动机：基于点云的刚体合成比物理仿真器轻量级得多，且对渲染质量的要求更宽松。

### 损失函数 / 训练策略
使用生成的视觉-动作数据训练标准模仿学习策略（如ACT、DP3等），标准行为克隆损失。

## 实验关键数据

### 主实验

| 评估维度 | 指标 | IGen | TesserAct | Cosmos | 说明 |
|---------|------|------|----------|--------|------|
| 视觉保真度 | 一致性评分 | 高 | 中 | 低 | 更接近真实 |
| 动作质量 | 指令遵循+物理对齐 | 最优 | 次优 | 差 | 生成动作更合理 |
| 策略迁移 | 真实任务成功率 | 可比/优于真实数据 | - | - | 纯生成数据有效 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full IGen | 最优 | 完整pipeline |
| w/o 3D重建 | 显著下降 | 3D理解是基础 |
| w/o 空间关键点 | 下降 | 关键点提供空间锚定 |
| 2D生成替代 | 下降 | 2D方法缺乏物理接地 |

### 关键发现
- 仅用IGen生成的数据训练的策略能在真实世界中成功执行操作任务，无需任何真实采集数据。
- 在某些场景下，IGen生成数据训练的策略甚至超越了真实数据训练的策略，可能因为场景多样性更高。
- 与视频生成方法相比，IGen生成的动作更物理一致且指令遵循度更高。

## 亮点与洞察
- **"图像即数据源"**的理念极具吸引力：互联网图像是最丰富的视觉资源。
- **无仿真的点云合成**避免了传统Real-to-Sim pipeline中最耗时的仿真环境构建步骤。
- 3D表示 + VLM规划的组合在保持轻量级的同时提供了物理接地。

## 局限与展望
- 依赖单目深度估计，精度受限于估计模型。
- 刚体运动假设限制了对软体/流体操作的建模。
- 对复杂物理交互（如接触力反馈）的建模不足。

## 相关工作与启发
- **vs RoLA**: RoLA也从开放图像生成数据，但依赖物理属性估计且限于简单交互。IGen通过VLM规划支持更复杂的任务。
- **vs TesserAct/Cosmos**: 基于视频生成的方法缺乏显式动作，IGen提供完整的视觉-动作对。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从开放世界图像到机器人数据的完整pipeline是新颖贡献
- 实验充分度: ⭐⭐⭐⭐ 三维度评估+真实世界验证
- 写作质量: ⭐⭐⭐⭐ pipeline描述清晰，对比充分
- 价值: ⭐⭐⭐⭐⭐ 有潜力根本性改变机器人数据获取方式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning](como_learning_continuous_latent_motion_from_internet_videos_for_scalable_robot_l.md)
- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](../../NeurIPS2025/robotics/dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)
- [\[CVPR 2026\] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)
- [\[CVPR 2026\] RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset](radar_closedloop_robotic_data_generation_via_seman.md)
- [\[CVPR 2025\] A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning](../../CVPR2025/robotics/a_data-centric_revisit_of_pre-trained_vision_models_for_robot_learning.md)

</div>

<!-- RELATED:END -->
