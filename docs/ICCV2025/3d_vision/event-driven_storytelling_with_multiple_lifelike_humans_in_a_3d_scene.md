---
title: >-
  [论文解读] Event-Driven Storytelling with Multiple Lifelike Humans in a 3D Scene
description: >-
  [ICCV 2025][3D视觉][Multi-Human Motion] 提出基于事件驱动的LLM框架，将3D场景中多角色行为规划分解为叙述者逐事件生成和事件解析器的精细空间推理两个模块，首次实现了大规模多房间3D场景中4-5+角色的长时序自然交互运动生成。
tags:
  - ICCV 2025
  - 3D视觉
  - Multi-Human Motion
  - LLM Planning
  - Event-Based
  - Scene-Aware
  - 3D Scene Graph
---

# Event-Driven Storytelling with Multiple Lifelike Humans in a 3D Scene

**会议**: ICCV 2025  
**arXiv**: [2507.19232](https://arxiv.org/abs/2507.19232)  
**代码**: 见项目页面  
**领域**: 3D视觉 / 多智能体运动生成 / LLM规划  
**关键词**: Multi-Human Motion, LLM Planning, Event-Based, Scene-Aware, 3D Scene Graph  

## 一句话总结
提出基于事件驱动的LLM框架，将3D场景中多角色行为规划分解为叙述者逐事件生成和事件解析器的精细空间推理两个模块，首次实现了大规模多房间3D场景中4-5+角色的长时序自然交互运动生成。

## 研究背景与动机

3D场景中生成逼真的数字人运动对VR、游戏、电影至关重要。然而现有运动合成工作局限性明显：
- 大多关注**单角色**独立运动，或仅扩展到与环境交互/与另一人交互中的**单一类型**
- 缺乏泛化性和可扩展性：无法同时生成多角色在拥挤场景中的合理交互
- 正确分配每个角色在正确时间的行为需要**广泛的动态上下文和空间推理**

**LLMs的潜力与挑战**：
- LLMs在高级规划和行为建模方面表现出色
- 但直接生成命令会遭受幻觉问题或定位错误（基础模型的已知局限）
- 需要精心设计框架以降低推理复杂度

**核心难点**：如何在保持可扩展性的同时，实现对3D空间的深入理解和多角色行为的协调规划？

## 方法详解

### 整体框架
系统围绕中间表示 **事件(event)** $e$ 运作：
- **高级行为规划模块**（LLM驱动）：以事件为单位进行规划
- **低级运动合成模块**：将事件转化为3D角色运动

### 高级行为规划模块

包含三个LLM子模块：

**1. 场景描述器 (Scene Describer)**
- 从3D场景 $\mathcal{S}$ 自动提取3D场景图（空间关系的JSON格式）
- 用DBSCAN对物体进行区域聚类，发现功能性空间（如餐厅区、学习区）
- 通过in-context learning引导LLM生成包含区域上下文信息的场景描述 $\mathcal{D}$

**2. 叙述者 (Narrator)**
- 每次生成一个事件，基于场景描述 $\mathcal{D}$、事件历史 $\mathcal{H}$ 和可选用户指令 $\mathcal{T}$
- 事件用半叙事自然语言描述，聚焦上下文而非精确坐标
- 接收事件状态反馈（'ongoing'/'completed'），驱动时间线推进
- 采用chain-of-thought推理：先分析当前规划状态和其他角色状态，再生成事件

**3. 事件解析器 (Event Parser)**
- 将高级事件描述转化为低级细节：$e=(\mathcal{C}_e, \{p_i\}, \{d_i\}, \{a_i\})$
    - $\mathcal{C}_e$：涉及的角色集合
    - $p_i \in \mathbb{R}^2$：目标位置
    - $d_i \in \mathbb{R}^1$：目标朝向
    - $a_i$：目标动作标签
- **Python编程结构提示**：提供空间推理工具函数（如 `get_distance_between()`），LLM通过编写代码来推理空间关系
- **区域条件位置采样**：先输出语义描述（如"离接待台最远的椅子"），再在对应区域内采样坐标，弥补LLM在坐标级推理上的弱点

### 低级运动合成模块
- 在2D网格地图上用windowed cooperative A*算法规划无碰撞多智能体路径
- 运动匹配(motion matching)快速生成目标位置的动作
- 群组事件（如聊天、握手）：先到的角色保持idle等待，所有角色到齐后同步执行

## 实验

### 基准评估

| 模型 | 总成功率(执行率) | OA | RC | SS |
|------|----------------|-----|-----|-----|
| **Ours (GPT-4o)** | **0.90 (0.98)** | **0.93 (0.99)** | **0.90 (0.98)** | **0.92 (0.98)** |
| w/o Event | 0.82 (0.92) | 0.88 (0.92) | 0.86 (0.93) | 0.77 (0.92) |
| Object List | 0.51 (0.85) | 0.61 (0.78) | 0.28 (0.88) | 0.65 (0.97) |
| Scene Graph | 0.82 (0.96) | 0.80 (0.96) | 0.82 (0.94) | 0.87 (0.95) |

OA=物体排列推理, RC=区域上下文推理, SS=场景状态推理

### 跨LLM引擎泛化

| 模型 | GPT-4o | GPT-4o mini | Llama-3.1-70B |
|------|--------|-------------|---------------|
| Ours | 0.90 | 0.74 | 0.72 |
| w/o Event | 0.82 | 0.60 | 0.60 |
| Object List | 0.51 | 0.34 | 0.35 |

**关键发现**：
- 事件驱动设计在所有LLM引擎上一致优于无事件变体（+8%~14%成功率）
- 场景描述器显著优于简单物体列表（+31%~39%），特别是在区域上下文推理上（0.90 vs 0.28）
- 使用较弱LLM（GPT-4o mini、Llama-3.1-70B）时性能下降可控
- Scene Graph变体在SS维度表现接近（0.87 vs 0.92），但RC维度有差距（0.82 vs 0.90）

## 亮点与洞察
1. **事件驱动分解**：将复杂的多角色长时序规划分解为逐事件决策，显著降低LLM的推理复杂度
2. **模块化降低耦合**：叙述者只管宏观叙事，事件解析器只管精确空间定位，各司其职
3. **编程式空间推理**：巧妙利用Python函数作为LLM的空间推理工具，弥补LLM坐标级推理的弱点
4. **区域功能发现**：DBSCAN聚类发现场景中的功能区域，提供超越物体列表的空间理解

## 局限性
- 运动合成模块依赖现有框架（motion matching），运动质量受限于动作库
- 3D场景需要实例分割标注作为输入
- 评估基准规模有限（40个测试用例），更大规模/更复杂场景的泛化性待验证
- LLM的推理延迟可能影响实时交互应用
- 未处理角色间的物理接触动作（如递交物品、推拉等）

## 相关工作
- 上下文运动合成：Digital Life Project处理双人场景但未扩展到更多角色
- LLM规划：SMART-LLM处理多机器人但场景理解简单（仅物体列表）
- 场景理解：3D场景图提供空间关系但缺乏区域功能推理

## 评分
- **创新性**: ★★★★☆ — 事件驱动+LLM模块化设计首次解决大规模多角色问题
- **实用性**: ★★★★☆ — 适用于VR/游戏/电影中的虚拟场景填充
- **实验**: ★★★☆☆ — 提出新基准但规模较小，缺乏与已有方法的直接对比
- **写作**: ★★★★☆ — 框架清晰，示例丰富

<!-- RELATED:START -->

## 相关论文

- [Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [TriDi: Trilateral Diffusion of 3D Humans, Objects, and Interactions](tridi_trilateral_diffusion_of_3d_humans_objects_and_interactions.md)
- [ETCH: Generalizing Body Fitting to Clothed Humans via Equivariant Tightness](etch_generalizing_body_fitting_to_clothed_humans_via_equivariant_tightness.md)
- [3D Test-time Adaptation via Graph Spectral Driven Point Shift](3d_test-time_adaptation_via_graph_spectral_driven_point_shift.md)
- [Event-based Tiny Object Detection: A Benchmark Dataset and Baseline](event-based_tiny_object_detection_a_benchmark_dataset_and_baseline.md)

<!-- RELATED:END -->
