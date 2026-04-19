---
title: >-
  [论文解读] SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration
description: >-
  [NeurIPS 2025][自动驾驶][simulation] 提出 SimWorld-Robotics (SWR)，一个基于 Unreal Engine 5 的大规模城市仿真平台，支持程序化生成无限逼真城市环境，并以此构建了多模态导航（SimWorld-MMNav）和多机器人搜索（SimWorld-MRS）两个新 benchmark，揭示了当前 VLM 在户外城市任务中的严重能力缺陷。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - simulation
  - embodied-ai
  - urban-navigation
  - multi-robot
  - benchmark
---

# SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration

**会议**: NeurIPS 2025  
**arXiv**: [2512.10046](https://arxiv.org/abs/2512.10046)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: simulation, embodied-ai, urban-navigation, multi-robot, benchmark

## 一句话总结

提出 SimWorld-Robotics (SWR)，一个基于 Unreal Engine 5 的大规模城市仿真平台，支持程序化生成无限逼真城市环境，并以此构建了多模态导航（SimWorld-MMNav）和多机器人搜索（SimWorld-MRS）两个新 benchmark，揭示了当前 VLM 在户外城市任务中的严重能力缺陷。

## 背景与动机

1. **室内偏重**：当前具身 AI 模拟器（Habitat 3、RoboTHOR、BEHAVIOR 等）集中在桌面/家居场景，缺乏户外城市环境支持。
2. **城市仿真不足**：CARLA、AirSim 专注自动驾驶且不支持程序化城市生成；MetaDrive、MetaUrban 可扩展但渲染质量差，缺乏真实感。
3. **动态交互缺失**：现有户外模拟器的行人行为仅限随机行走，无法反映真实城市中的交通规则、社会导航等复杂动态。
4. **多智能体受限**：大部分模拟器仅支持单类智能体（车辆或机器人），不支持人-车-机器人的统一控制与异步协作。
5. **导航 benchmark 单一**：Vision-and-Language Navigation 多为静态室内或街景全景，缺少多模态指令（语言+视觉提示）在动态城市中的联合评估。
6. **多机器人协作空白**：已有 benchmark（RoCo、RobotSlang 等）要么限于小环境，要么不涉及语言通信+物理导航的联合协作。

## 方法详解

### 3.1 程序化城市生成

SWR 的城市生成分四阶段：
- **道路网络**：基于优先队列的增长策略，平衡分支与深度，支持路端连接与交叉口验证
- **建筑放置**：沿道路进行碰撞感知采样 + 贪心间隙填充，最大化覆盖率
- **街道元素**：上下文感知地放置树木、锥桶、长椅、停放车辆等
- **动态交通元素**：集成车辆和行人，支持概率性路口路径选择，引入自然变异性

### 3.2 具身智能体

SWR 统一支持三类智能体：人类（26种动作）、车辆、机器人（滑板车/四足机器人），采用异步多智能体控制框架，每个智能体独立行动，不阻塞其他智能体。观测空间包括 RGB、深度、语义分割和 3D 包围盒。

### 3.3 行人与交通仿真

- 车辆运动由 PID 控制器驱动，参数经验调优
- 行人使用轻量级模型，根据角度差渐进调整朝向
- 路口采用概率路由策略，增强场景多样性

### 3.4 SimWorld-MMNav 多模态导航 Benchmark

机器人需沿多模态指令（语言描述 + 视觉提示图像）在大规模动态城市中导航至目标。指令分四类：方向对齐、沿路行进、路口转弯、到达目的地。分 Easy（无障碍物）和 Hard（含行人、交通信号）两个难度。评估指标包括成功率（SR）、子任务SR、距离进度和安全指标（碰撞、闯红灯次数）。

### 3.5 SimWorld-MRS 多机器人搜索 Benchmark

两个机器人在陌生城市中需通过自然语言通信 + 物理导航来定位并会面。主机器人拥有城市地标记忆（20+地标），跟随者无先验知识。评估 CSR（协作成功率）和任务进度。

### 3.6 SimWorld-20K 数据集

100 个训练地图，平均面积 2 km²，200 条 A* oracle 轨迹（平均长度 > 2.5 km），总计 20K 训练步。环境面积是 MetaUrban 的 100×，轨迹长度 1.2× 以上。

## 实验结果

### 表1：SimWorld-MMNav Easy 设置结果

| 模型 | SR(%) ↑ | 子任务SR(%) ↑ | 距离进度(%) ↑ |
|------|---------|-------------|-------------|
| GPT-4o | 0 | 33.07 | 15.60 |
| Gemini 2.5 Flash | 0 | 37.06 | 31.29 |
| GPT-o3 | 5.0 | 42.50 | 38.43 |
| GPT-o3-pro | 8.3 | 46.35 | 39.46 |
| QwenVL 2.5 7B | 0 | 16.86 | 7.82 |
| QwenVL 2.5 72B | 0 | 23.80 | 17.50 |
| QwenVL 2.5 7B (ft) | **4.0** | **52.45** | **53.63** |

**发现**：所有零样本非推理模型 SR 均为 0；推理模型（o3/o3-pro）首次实现非零 SR；微调后 7B 模型全面超越闭源大模型。

### 表2：SimWorld-MRS 多机器人搜索结果

| 模型 | 方法 | CSR(%) ↑ | 任务进度(%) ↑ |
|------|------|---------|-------------|
| GPT-4o | Oracle Planner | 65.00 | 76.90 |
| Gemini 2.5 Flash | Oracle Planner | 54.55 | 75.84 |
| GPT-4o | RoCo | 33.33 | 22.93 |
| QwenVL 72B | RoCo | 11.11 | 35.94 |

**发现**：即使用 Oracle 规划器，GPT-4o 也仅 65% CSR；RoCo 策略因语言歧义导致执行偏差，性能大幅下降。

### 失败模式分析

- **路口距离误判**（53.33%）：VLM 无法准确估计到路口的距离
- **目的地视角匹配失败**（60.00%）：从不同视角识别同一地标困难
- **转弯模式误解**（42.86%）：高层指令如"在路口右转"缺乏具体动作细节

## 亮点

- 首个同时支持程序化生成、真实感渲染、人-车-机器人异步控制的城市仿真平台
- 支持 26 种人类动作，远超现有模拟器（≤10 种）
- 两个 benchmark 设计精妙，系统性暴露了 VLM 在空间推理、动态避障、多智能体通信方面的短板
- SimWorld-20K 为城市导航训练提供了首个大规模数据集，微调效果显著

## 局限性

- 仅关注户外环境，不支持室内场景
- 人类智能体动作空间虽优于前人但仍有限，未利用运动生成模型
- SimWorld-20K 训练数据基于 oracle 轨迹，缺乏纠错/强化学习数据
- Hard 设置下推理模型因延迟过高无法实时避障
- 渲染基于 UE5 但未量化 sim-to-real 迁移效果

## 相关工作对比

### vs CARLA
CARLA 作为经典自动驾驶模拟器，不支持程序化城市生成，建筑多样性低且重复度高，仅支持车辆和简单行人控制（2种动作）。SWR 支持无限城市生成、三类智能体统一异步控制、26种人类动作，在可扩展性和真实感上全面超越。

### vs MetaUrban
MetaUrban 支持程序化生成和微出行任务，但环境面积仅 0.02 km²、渲染质量差（非真实感）。SWR 环境面积达 2 km²（100×），基于 UE5 实现真实感渲染，且支持车辆和行人交通系统，任务复杂度和评估维度远超 MetaUrban。

### vs RoCo/RobotSlang
RoCo 限于桌面操作任务，不涉及环境探索；RobotSlang 评估机器人通信但限于小环境。SimWorld-MRS 首次在大规模城市环境中评估多机器人语言通信+物理导航的联合协作能力。

## 评分

- ⭐⭐⭐⭐ 新颖性：首个全功能城市具身 AI 仿真平台
- ⭐⭐⭐⭐ 技术质量：工程实现完整，benchmark 设计系统全面
- ⭐⭐⭐⭐ 实验充分度：多模型多设置评估，失败模式分析深入
- ⭐⭐⭐ 可复现性：平台基于 UE5，部署门槛较高

<!-- RELATED:START -->

## 相关论文

- [SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic](../../ECCV2024/autonomous_driving/sledge_synthesizing_driving_environments_with_generative_models_and_rule-based_t.md)
- [Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](../../ICCV2025/autonomous_driving/leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)
- [URB -- Urban Routing Benchmark for RL-Equipped Connected Autonomous Vehicles](urb_--_urban_routing_benchmark_for_rl-equipped_connected_autonomous_vehicles.md)
- [SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)
- [Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion](layer-wise_modality_decomposition_for_interpretable_multimodal_sensor_fusion.md)

<!-- RELATED:END -->
