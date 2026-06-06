---
title: >-
  [论文解读] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model
description: >-
  [CVPR 2026][自动驾驶][交通场景生成] 提出 TTSG，一个无需训练的模块化框架，能够直接从自由格式自然语言描述生成逼真的交通场景，通过 LLM 驱动的提示分析、道路检索、智能体规划和计划感知道路排序算法，无需预定义路线或生成点，在 SafeBench 上实现最低 3.5% 平均碰撞率。
tags:
  - "CVPR 2026"
  - "自动驾驶"
  - "交通场景生成"
  - "自然语言驱动"
  - "大语言模型"
  - "自动驾驶仿真"
  - "CARLA"
---

# Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model

**会议**: CVPR 2026  
**arXiv**: [2409.09575](https://arxiv.org/abs/2409.09575)  
**代码**: [https://basiclab.github.io/TTSG](https://basiclab.github.io/TTSG)  
**领域**: 自动驾驶 / 场景生成  
**关键词**: 交通场景生成, 自然语言驱动, 大语言模型, 自动驾驶仿真, CARLA

## 一句话总结

提出 TTSG，一个无需训练的模块化框架，能够直接从自由格式自然语言描述生成逼真的交通场景，通过 LLM 驱动的提示分析、道路检索、智能体规划和计划感知道路排序算法，无需预定义路线或生成点，在 SafeBench 上实现最低 3.5% 平均碰撞率。

## 研究背景与动机

交通场景数据集（如 nuScenes、Waymo）为自动驾驶模型提供了丰富的多模态驾驶日志，但真实世界数据收集受安全限制和可控性不足的制约。CARLA、MetaDrive 等仿真平台提供了安全、可扩展的实验环境，但现有的场景生成方式存在明显不足：随机采样缺乏针对性的控制力来系统评估特定失效模式和边缘案例；基于日志回放的方法受限于收集数据的分布，难以生成新颖场景。

近期的指令驱动仿真方法虽然增强了可控性，但存在三个核心痛点：(1) LCTGen、ProSim 等依赖结构化输入，无法处理自由格式自然语言；(2) ChatScene 仅关注智能体规划，仍需用户手动指定生成点和地图位置；(3) 所有先前工作都忽略了交通信号、静态道路物体和天气等环境条件。

核心矛盾在于：用户希望用自然语言描述复杂场景（如"一辆消防车从左侧道路驶来，此时自车正在右转"），但现有系统缺乏将自由文本落地为空间有效、语义连贯布局的能力，尤其是在没有预定义位置的情况下组合场景。

TTSG 的核心 idea：设计一个免训练的模块化流水线，将 LLM 嵌入严格控制的管道中执行结构化、可行的场景分解，并通过计划感知的道路排序算法确保智能体动作与道路几何的一致性。

## 方法详解

### 整体框架

TTSG 包含五个阶段：(1) 提示分析——LLM 将输入文本分解为结构化场景元素；(2) 道路候选检索——基于分析结果从预构建的道路图中检索候选道路；(3) 智能体规划——LLM 确定每个智能体的类型、动作和相对位置；(4) 道路排序——评估候选道路与智能体计划的兼容性；(5) 场景生成——通过自定义渲染模块将所有信息转化为可执行的交通场景。

### 关键设计

1. **道路图构建与智能体集 (Road Graph & Agent Set)**:

    - 功能：构建道路信息的结构化数据库，支持自动生成点选择
    - 核心思路：将 CARLA 内置地图转换为 OpenDRIVE 格式，解析道路特征（交通信号、静态物体、交叉口、车道配置等），组织为图结构，其中边表示道路连通性。智能体按类型分类（常规车辆、紧急车辆、行人等），支持从各类别中随机选择实例
    - 设计动机：图结构使得可以高效查询道路属性和邻接关系（如可转弯性、道路连接），为自动化场景生成提供基础设施

2. **提示分析与计划感知道路排序 (Prompt Analysis & Plan-Aware Road Ranking)**:

    - 功能：将自由文本转化为结构化场景表示，并选择最优道路
    - 核心思路：提示分析阶段让 LLM 将输入分解为交通信号、物体、智能体配置等显式组件，作为后续阶段的上下文知识。道路排序阶段对每条候选道路计算与智能体计划的匹配分数 $r^* = \arg\max_{r \in R_c} \sum_{a \in A} \mathbf{1}_{\{\text{match}(r,a)\}}$，选择满足最多智能体条件的道路。若多条道路得分相同则随机选择，确保排序-随机策略兼顾对齐和多样性
    - 设计动机：直接使用 CoT 需要大量 token，分析式策略在保持相当质量的同时将 token 用量从 1022 降至 682。道路排序解决了无排序时随机选路导致场景与描述不匹配的问题

3. **智能体规划与序列事件支持 (Agent Planning & Sequential Events)**:

    - 功能：生成详细的多智能体行为计划，支持多阶段事件组合
    - 核心思路：LLM 为每个智能体确定八个方向之一的朝向、具体动作（直行、停车等）和相对距离。支持定义智能体间的相对位置关系（"position"属性，值越小表示越靠前）。序列事件通过迭代规划实现：先执行完整管道生成初始事件，将结束位置作为后续事件的起始点，仅需重新执行提示分析和智能体规划
    - 设计动机：支持复杂的多智能体交互场景（如"两辆车挡住自车"）和时间连续的多阶段场景（如"左转后被两辆车堵住"），这是先前工作无法实现的

### 损失函数 / 训练策略

TTSG 是完全免训练的框架，不涉及模型训练。下游应用中，使用 soft-actor-critic 模型训练自驾智能体时，TTSG 生成的场景作为训练环境。每个阶段后应用格式验证，检查所有键、类型和值的正确性，错误时自动重新提交 LLM 修正。

## 实验关键数据

### 主实验

| 方法 | SO 碰撞率↓ | LC 碰撞率↓ | ULT 碰撞率↓ | 平均碰撞率↓ | 平均驾驶分↑ |
|--------|------|------|----------|------|------|
| Learning-to-Collide | 0.120 | 0.510 | 0.000 | 0.210 | 0.822 |
| AdvSim | 0.230 | 0.430 | 0.050 | 0.270 | 0.796 |
| Adversarial-Trajectory | 0.140 | 0.300 | 0.000 | 0.150 | 0.867 |
| ChatScene | 0.030 | 0.110 | 0.100 | 0.080 | 0.905 |
| **TTSG (本文)** | **0.021** | **0.085** | **0.000** | **0.035** | **0.914** |

### 消融实验

| 配置 | AA (Avg)↑ | RA (Avg)↑ | 说明 |
|------|---------|------|------|
| w/o analysis | 0.833 | 0.775 | 去掉分析阶段 |
| w/ analysis | 0.925 | 0.875 | 加入分析阶段 |
| w/ analysis + CoT | **0.975** | **0.940** | 分析+CoT 混合 |
| w/o ranking (SA) | 0.560 | - | 无道路排序 |
| w/ ranking (SA) | **0.800** | - | 有道路排序 |

### 关键发现

- **道路排序贡献最大**：场景准确率从 0.560 提升到 0.800，提升 43%，说明智能体-道路对齐是场景质量的关键
- **分析策略高效且可组合**：分析阶段在 token 减少 33% 的情况下达到接近 CoT 的质量，且与 CoT 组合可进一步提升
- **跨 LLM 泛化性强**：从轻量开源模型 Gemma3-12B 到 Claude-Sonnet-3.5 都能有效运行，Claude 达到近乎完美的规划精度
- **驾驶描述增强**：仅用 20 个关键场景微调，CIDEr 分数在推理方面从 18.4 提升到 51.9（+33.5 points）

## 亮点与洞察

- **免训练的端到端场景生成**是最大亮点——完全不需要任何模型训练，仅通过 LLM 的推理能力和结构化管道就能从自然语言生成可执行的 CARLA 场景。巧妙之处在于将 LLM 嵌入受约束的管道中避免了幻觉问题
- **计划感知排序策略**简单但非常有效——一个简单的匹配计数指标就将场景准确率从 56% 提升到 80%，说明问题的关键不在于复杂算法而在于正确的问题分解
- **序列事件组合机制**可以迁移到机器人任务规划、游戏场景设计等需要多阶段连贯场景的领域

## 局限与展望

- 完全依赖 LLM 的语言理解能力，对模糊或矛盾的描述缺乏鲁棒的错误恢复机制
- 智能体行为模式相对简单（7 种基本动作），无法表达连续的复杂驾驶行为（如渐进式变道）
- 仅在 CARLA 上验证，未验证在其他仿真器（如 MetaDrive）上的泛化性
- 道路图是静态预构建的，无法动态创建新的道路布局或交通设施

## 相关工作与启发

- **vs ChatScene**: ChatScene 需要手动指定生成点和初始位置，TTSG 完全自动化；碰撞率从 0.08 降至 0.035
- **vs LCTGen**: LCTGen 依赖结构化输入格式，TTSG 支持完全自由格式的自然语言
- **vs CTG++**: CTG++ 用 LLM 生成代码级损失函数指导扩散模型，更复杂但灵活性不如 TTSG 的模块化设计

## 评分

- 新颖性: ⭐⭐⭐⭐ 计划感知道路排序和免训练模块化管道设计新颖，但核心组件（LLM 规划）不算突破性
- 实验充分度: ⭐⭐⭐⭐ SafeBench 评测、消融、多 LLM 对比、多样性测试，涵盖面广
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富，流水线每步都有明确解释
- 价值: ⭐⭐⭐⭐ 对自动驾驶场景生成有实际价值，免训练的特点降低了使用门槛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning](nord_a_data-efficient_vision-language-action_model_that_drives_without_reasoning.md)
- [\[ACL 2025\] Embracing Large Language Models in Traffic Flow Forecasting](../../ACL2025/autonomous_driving/embracing_large_language_models_in_traffic_flow_forecasting.md)
- [\[CVPR 2026\] Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](drive_my_way_preference_alignment_of_vision-language-action_model_for_personaliz.md)
- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)

</div>

<!-- RELATED:END -->
