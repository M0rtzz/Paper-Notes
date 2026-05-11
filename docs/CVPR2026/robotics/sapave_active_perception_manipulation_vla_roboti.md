---
title: >-
  [论文解读] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics
description: >-
  [CVPR 2026][机器人][主动感知] SaPaVe提出端到端主动操作框架，通过解耦相机动作和操作动作的底层到顶层训练策略，先用20万对语义相机控制数据学习主动感知先验，再联合优化实现主动操作，在真实世界中超越π₀和GR00T N1达31.25%成功率提升。
tags:
  - CVPR 2026
  - 机器人
  - 主动感知
  - VLA模型
  - 解耦动作空间
  - 3D空间注入
  - 人形机器人
---

# SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics

**会议**: CVPR 2026  
**arXiv**: [2603.12193](https://arxiv.org/abs/2603.12193)  
**代码**: [https://lmzpai.github.io/SaPaVe](https://lmzpai.github.io/SaPaVe)  
**领域**: 多模态VLM / 机器人  
**关键词**: 主动感知, VLA模型, 解耦动作空间, 3D空间注入, 人形机器人

## 一句话总结
SaPaVe提出端到端主动操作框架，通过解耦相机动作和操作动作的底层到顶层训练策略，先用20万对语义相机控制数据学习主动感知先验，再联合优化实现主动操作，在真实世界中超越π₀和GR00T N1达31.25%成功率提升。

## 研究背景与动机

1. **领域现状**：主动感知和操作是机器人与复杂场景交互的核心能力。现有VLM（如Qwen2.5-VL、Gemini 2.5 Pro）已提升语义理解能力，VLA模型（如π₀、GR00T N1）致力于端到端桥接视觉-语言-动作。
2. **现有痛点**：
    - VLM做主动感知多将其建模为VQA任务（从离散候选中选最优视角），无法进行连续精细的相机控制。
    - VLA模型通常在固定最优头部相机视角下训练和评估，对视角变化敏感，缺乏主动视角调整能力。
    - 直接在VLA中增加相机动作到统一动作空间会产生冲突，且需要大量昂贵的真实世界主动感知+操作数据。
3. **核心矛盾**：主动操作需要"语义主动感知"（根据任务策略调整视角获取关键信息）和"主动视角执行"（在动态视角下保持鲁棒操作）两种能力的紧密耦合，但数据稀缺和动作空间冲突使得现有方法难以兼顾。
4. **本文目标** 如何在数据高效的前提下，让机器人同时学会语义驱动的主动视角调整和视角变化下的鲁棒操作。
5. **切入角度**：关键洞察是相机运动是跨具身平台通用的（embodiment-agnostic），可以先独立学习再联合优化，从而实现底层到顶层的高效训练。
6. **核心 idea**：解耦相机动作与操作动作，先用大规模语义相机运动数据建立主动感知先验，再联合优化实现数据高效的主动操作。

## 方法详解

### 整体框架
SaPaVe基于VLA模型架构，输入RGB图像和任务指令，输出解耦的双路动作：头部相机动作 $A_{head}$（pitch/yaw调整）和操作动作 $A_{other}$（26-DoF关节位置增量，对应Unitree G1双臂双手）。采用动作分块策略(action chunking)预测时域为 $k$ 的动作序列，确保时序一致性和平滑执行。

### 关键设计

1. **解耦动作头 + 相机适配器（Decoupled Action Heads & Camera Adapter）**

    - 功能：在不破坏VLM原有语义能力的前提下，让模型分别学会相机控制和操作动作。
    - 核心思路：相机适配器使用LoRA加在VLM上学习语义主动感知先验，不改变原始VLM权重。解耦动作头设置两个独立的denoising decoder，分别输出相机动作和操作动作。这种轻量解耦设计使模型能快速准确地学习两类动作，避免统一动作空间的相互干扰。
    - 设计动机：直接在现有VLA动作空间中加入相机运动会破坏大规模固定视角操作数据上学到的先验知识。相机运动本质上与具身形态无关，用独立适配器学习更高效且不影响操作能力。实验证实，全量微调VLM学相机运动反而不如用轻量适配器（Tab.5），因为适配器可以保留VLM的高层语义信息。

2. **通用空间知识注入（Universal Spatial Knowledge Injection）**

    - 功能：增强模型在动态视角变化下的3D空间感知和鲁棒操作能力。
    - 核心思路：使用一个继承自强大前馈3D几何模型的Universal Spatial Encoder，支持任意种类的3D几何信息（深度图、相机内外参等）作为输入，无需重新训练或修改架构。编码后的空间token与VLM输出token逐元素相加，混合token注入到解耦动作头的动作去噪过程中。
    - 设计动机：VLA模型缺乏3D几何先验，在主动视角变化时无法保持一致的空间理解。直接注入多种3D信息可以从根本上提升模型对视角变化的鲁棒性。实验показ去掉该模块后即使简单的遮挡抓取任务也掉15%成功率（Tab.5）。

3. **两阶段底层到顶层训练策略**

    - 功能：数据高效地将主动感知和主动操作能力层层建立。
    - 核心思路：
        - **Stage 1（语义主动感知对齐）**：仅用ActiveViewPose-200K数据集训练相机适配器和相机动作解码器，损失为MSE $\mathcal{L}_{stage1} = \mathcal{L}_{MSE}(A_{head,t}, A_{head,t}^*)$。此阶段让模型获得强大的语义驱动视角调整先验。
        - **Stage 2（主动操作微调）**：冻结相机适配器，用混合数据（ActiveViewPose-200K + 机器人操作数据）训练解耦动作头，$\mathcal{L}_{stage2} = \lambda_{head}\mathcal{L}_{head} + \lambda_{other}\mathcal{L}_{other}$。
    - 设计动机：一次性联合训练需要大量稀缺的主动操作数据。先在大量容易获取的纯视角数据上建立感知先验，再用少量操作数据微调，实现数据高效的迁移学习。

### 损失函数 / 训练策略
- Stage 1: 仅MSE损失监督相机动作预测
- Stage 2: 加权MSE损失同时监督相机和操作动作，冻结相机适配器保护Stage 1学到的先验
- 动作分块(action chunking)确保时序平滑

## 实验关键数据

### 主实验：语义主动感知评估

| 方法 | Val | Test1 | Test2 | 平均 |
|------|-----|-------|-------|------|
| Qwen2.5-VL-72B | 63.9 | 65.1 | 58.0 | 62.3 |
| Multi-SpatialMLLM | 72.8 | 74.3 | 63.6 | 70.2 |
| Gemini-2.5-Pro | 73.3 | 76.5 | 68.2 | 72.7 |
| **SaPaVe (2B)** | **85.5** | **89.1** | **78.3** | **84.3** |

真实世界主动操作（成功率%）：

| 方法 | 遮挡抓放 | 视野外抓放 | 遮挡关节操作 | 视野外关节操作 | 平均 |
|------|---------|----------|------------|------------|------|
| π₀ | 55 | 45 | 45 | 35 | 45.00 |
| GR00T-N1 | 60 | 55 | 50 | 50 | 53.75 |
| **SaPaVe** | **90** | **85** | **85** | **80** | **85.00** |

### 消融实验

| 配置 | 遮挡抓放 | 视野外抓放 | 遮挡操作 | 视野外操作 | 平均 |
|------|---------|----------|---------|----------|------|
| Full Model | 90 | 85 | 85 | 80 | 85.00 |
| w/o Stage 1 | 65 | 55 | 50 | 45 | 53.75 |
| w/o Stage 2 | 75 | 60 | 70 | 60 | 66.25 |
| w/o 解耦动作头 | 80 | 70 | 70 | 65 | 71.25 |
| w/o 相机适配器 | 80 | 75 | 70 | 70 | 73.75 |
| w/o 空间知识注入 | 75 | 75 | 65 | 60 | 68.75 |

### 关键发现
- Stage 1贡献最大，去掉后平均掉31.25%（85→53.75），尤其是视野外任务几乎减半，说明主动感知先验是核心。
- 通用空间知识注入去掉后掉16.25%，连简单的遮挡抓取都掉15%，说明3D信息对抗视角变化至关重要。
- 仅2B参数的SaPaVe在语义感知上超越72B的Qwen2.5-VL和Gemini 2.5 Pro，说明语义主动感知不是通用VLM的涌现能力，需要专门训练。
- 固定相机+手腕相机的组合仍远不如主动相机，尤其是视野外任务（gap > 40%），说明"更多视角"不如"主动控制视角"。

## 亮点与洞察
- **"相机运动是embodiment-agnostic的"这一洞察**是全文最核心的insight，由此推导出解耦+底层到顶层的训练策略，优雅且有效。可以迁移到其他需要分离通用能力和具身特定能力的机器人学习场景。
- **ActiveViewPose-200K数据集**的构建流程（4K高质量资产 + 启发式动作生成 + GPT-4o指令生成 + 人工精炼）既高效又可复现，为社区提供了一个填补空白的评测基准。
- 真实世界中超越π₀ 40%、GR00T-N1 31.25%的绝对成功率提升，说明主动操作能力不是简单增加动作维度就能解决的。

## 局限与展望
- 仅在Unitree G1人形机器人上验证，对其他机器人形态（如单臂、移动底盘）的迁移性未验证。
- 当前相机动作仅2-DoF（pitch/yaw），未考虑平移等更复杂的视角调整。
- ActiveViewPose-200K是半自动构建的静态场景数据，真实世界的动态遮挡变化可能需要更多数据。
- 可以探索在线学习机制，让机器人在执行中持续改进主动感知策略。

## 相关工作与启发
- **vs π₀ [6]**: π₀是强通用VLA但缺乏主动感知能力，直接微调加入相机动作效果差（成功率仅45%），SaPaVe通过解耦策略达85%。
- **vs GR00T-N1 [5]**: 同样缺乏主动感知先验，虽然是专为人形设计但主动操作上被SaPaVe超越31.25%。
- **vs NBV方法 [7,54]**: 传统Next-Best-View方法非端到端且缺少语义输入，SaPaVe端到端整合语义理解和连续相机控制。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "解耦+底层到顶层"策略和ActiveManip-Bench都是开创性贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 仿真+真实世界+消融+泛化全覆盖，baseline选择精准
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，实验分析深入
- 价值: ⭐⭐⭐⭐⭐ 填补了VLA模型在主动操作领域的空白，数据集和基准对社区价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](lada_robotic_manipulation.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)
- [\[CVPR 2026\] Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior](boosting_vision-language-action_finetuning_with_feasible_action_neighborhood_pri.md)

</div>

<!-- RELATED:END -->
