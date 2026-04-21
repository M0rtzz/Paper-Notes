---
title: >-
  [论文解读] Robotic Visual Instruction
description: >-
  [CVPR 2025][机器人][视觉指令] 提出 Robotic Visual Instruction (RoVI)，一种以手绘箭头和圆圈为核心的视觉指令范式，替代自然语言来指导机器人操作，并设计 VIEW pipeline 将2D视觉指令转化为3D动作序列，在真实环境中达到87.5%成功率。
tags:
  - CVPR 2025
  - 机器人
  - 视觉指令
  - 人机交互
  - 手绘符号
  - 机器人操作
  - VLM
---

# Robotic Visual Instruction

**会议**: CVPR 2025  
**arXiv**: [2505.00693](https://arxiv.org/abs/2505.00693)  
**代码**: https://robotic-visual-instruction.github.io/  
**领域**: 机器人 / 人机交互  
**关键词**: 视觉指令, 人机交互, 手绘符号, 机器人操作, VLM

## 一句话总结

提出 Robotic Visual Instruction (RoVI)，一种以手绘箭头和圆圈为核心的视觉指令范式，替代自然语言来指导机器人操作，并设计 VIEW pipeline 将2D视觉指令转化为3D动作序列，在真实环境中达到87.5%成功率。

## 研究背景与动机

**领域现状**：当前人机交互主要依赖自然语言，借助LLM将文本指令转化为机器人动作。也有一些工作使用图像条件策略（如目标图像、轨迹图像）来传达空间信息。

**现有痛点**：自然语言在描述空间细节（精确位置、方向、距离）时天然不足，容易产生歧义和冗余。例如"把柠檬移到土豆的下方附近"这种指令很难精确传达位置。此外某些公共场景（如图书馆、医院）不适合语音交互。而目标图像方法要求用户提供任务完成后的最终状态图像，轨迹方法要求用户想象并绘制末端执行器的完整运动路径，这些对用户来说都不友好。

**核心矛盾**：用户友好性与空间精度之间的矛盾——自然语言好用但不精确，图像/轨迹精确但不好用。

**本文目标**：设计一种同时兼顾用户友好性、可解释性和时空精度的人机交互方式，并构建完整的从视觉指令到机器人动作的处理流程。

**切入角度**：人们日常生活中就会通过手绘箭头和圈示来传达空间信息（比如在地图上画路线），这种以目标物体为中心的符号语言可以自然地编码时空信息。

**核心 idea**：用2D手绘符号（箭头表示运动轨迹和方向、圆圈表示交互区域、颜色/数字表示时序）来替代自然语言进行机器人任务定义，再利用VLM理解这些符号并转化为可执行的3D动作序列。

## 方法详解

### 整体框架

RoVI系统的输入是一张在初始观测图像上叠加的手绘视觉指令图像，输出是机器人的3D动作序列。整个流程包含三个核心组件：(1) VLM负责理解RoVI并生成层次化的语言响应和可执行代码；(2) 关键点模块从RoVI符号中提取空间约束；(3) 基于关键点的低层策略执行具体动作。

### 关键设计

1. **RoVI 视觉指令范式设计**:

    - 功能：定义了一套简洁的视觉符号语言来编码机器人操作的时空信息
    - 核心思路：所有操作被分解为三种基本运动——从A到B（箭头表示）、旋转物体（圆圈+箭头）、拾取/选择（圆圈标记）。箭头分解为尾部（起始点$p_0$）、轴部（中间路径点）和头部（终点$p_n$）。不同颜色（绿→蓝→粉）表示多步操作的时序关系，数字标注用于双臂系统。还设计了松散风格和几何风格两种绘制方式。
    - 设计动机：将3D坐标的时序序列压缩到人类可理解的2D视觉语言中，解决自然语言的空间模糊性问题。实验表明结构化的几何风格比松散风格对VLM理解更友好。

2. **VIEW (Visual Instruction Embodied Workflow) Pipeline**:

    - 功能：将2D手绘视觉指令转化为机器人的3D可执行动作
    - 核心思路：VLM接收RoVI图像和初始观测图像，通过Chain-of-Thought推理生成层次化输出：粗粒度任务预测→细粒度规划→可执行Python函数。同时关键点模块使用YOLOv8检测箭头和圆圈的关键点，提供空间约束。最终代码函数和关键点坐标结合，通过RGB-D相机映射到3D空间后执行。
    - 设计动机：相比端到端策略直接输出SE(3)参数，语言化的动作表示在不同任务和环境间泛化能力更强。使用YOLOv8检测RoVI符号而非环境物体，使系统不受环境变化和干扰物影响。

3. **基于关键点的低层策略**:

    - 功能：根据关键点序列生成并执行机器人末端执行器的运动
    - 核心思路：将2D关键点通过RGB-D深度数据映射到3D坐标$p'_i \in \mathbb{R}^3$，然后映射为SE(3)空间中的末端执行器位姿序列。在每个时间步最小化代价函数 $\mathcal{L}_i(t) = \alpha_i \delta_{trans}(t) + (1-\alpha_i)\delta_{rot}(t)$，其中$\alpha_i$区分平移和旋转操作。当代价低于阈值$\epsilon$时切换到下一个关键点。
    - 设计动机：将平移和旋转统一到同一框架中，通过$\alpha_i$自适应切换，能够处理复杂的多步组合动作。

### 损失函数 / 训练策略

RoVI Book数据集包含15K图文问答对，基于Open-X Embodiment数据集构建。使用LoRA对LLaVA-7B/13B进行微调，学习率2e-4，训练1个epoch。数据覆盖64%单步任务和36%多步任务，包含5种基本操作技能。对RoVI进行数据增强（3-8种变体，不同路径、风格、线宽）。

## 实验关键数据

### 主实验

| 方法 | 真实环境平均成功率 | 仿真环境平均成功率 |
|------|-------------------|-------------------|
| VoxPoser | 43.8% | - |
| CoPa | 45.0% | - |
| VIEW-GPT4o | 82.5% | - |
| VIEW-LLaVA-13B (RoVI Book) | **87.5%** | - |
| RT-1-X | - | 20% |
| Octo-goal-image | - | 13.3% |
| Octo-language | - | 3% |
| VIEW* | - | **76.6%** |

### 消融实验

| 配置 | 任务规划准确率 | 说明 |
|------|---------------|------|
| GPT-4o (零样本) | 81% | 最强商用模型 |
| Gemini-1.5 Pro | 68% | 仿真表现较弱 |
| Claude 3.5 Sonnet | 70% | 多步任务准确率下降 |
| LLaVA-13B (RoVI Book) | 38% | 规划准确率低但执行成功率高 |
| 小模型 (<13B) | 0% | 完全无法理解RoVI |
| 松散绘制风格 | 74% | - |
| 几何绘制风格 | **80%** | 结构化风格更利于VLM理解 |

### 关键发现

- LLaVA-13B在任务规划准确率(38%)远低于GPT-4o(81%)，但在动作执行层面表现相当甚至更好(87.5% vs 82.5%)。这是因为可执行函数将动作和序列错误映射掉了，不受感知错误影响。
- VIEW在杂乱环境和轨迹跟随任务中显著优于语言指令方法，因为关键点模块提供了像素级精度的空间约束。
- 所有小于13B参数的模型完全无法理解RoVI，表明理解这种视觉符号需要足够的模型容量。

## 亮点与洞察

- **以物体为中心的符号设计**：仅用箭头、圆圈、颜色、数字四种基本元素就能编码复杂的多步操作，设计极度简洁。这种设计思路可迁移到其他需要精确空间表达的场景（如手术机器人指令）。
- **VLM理解RoVI后的代码生成**：让VLM输出Python代码函数而非直接动作参数，提供了很好的可调试性和可解释性。
- **关键点模块检测RoVI符号而非环境物体**：巧妙避开了开放词汇物体检测在杂乱环境中的困难，使得系统对环境变化和干扰物鲁棒。

## 局限与展望

- 当前RoVI仍需用手写笔在平板/电脑上绘制，交互便利性有提升空间，未来可考虑AR/手势等更自然的输入方式
- 2D到3D的映射依赖深度相机的精度，在遮挡严重或深度不准的场景可能失效
- 颜色编码方案限制了可支持的最大步数，且假设背景色较暗以保证符号可见性
- 对双臂协作操作的支持还比较初步，复杂的协同任务可能需要更丰富的符号语义

## 相关工作与启发

- **vs VoxPoser/CoPa**: 这两种方法仍依赖自然语言输入+物体检测，在杂乱环境和需要精确空间对齐的任务中表现不佳。RoVI通过像素级的视觉指令避免了语言的空间模糊性。
- **vs 目标图像/轨迹策略**: 目标图像方法对用户不友好（需要知道最终状态），轨迹方法让用户难以想象完整运动过程。RoVI取了折中——用简单符号传达关键空间信息。
- **vs RT-1-X/Octo**: 端到端VLA模型在泛化到新任务时严重受限，而VIEW的模块化设计（VLM+关键点+低层策略）提供了更好的泛化能力。

## 评分

- 新颖性: ⭐⭐⭐⭐ 提出了全新的视觉符号指令范式，思路简洁优雅但符号设计本身不算复杂
- 实验充分度: ⭐⭐⭐⭐ 11个任务覆盖真实和仿真环境，有多角度消融和对比，但规模不算大
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，Teaser图一目了然
- 价值: ⭐⭐⭐⭐ 开辟了视觉符号交互的新方向，但实际部署的便利性和用户接受度仍待验证

<!-- RELATED:START -->

## 相关论文

- [Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation](mitigating_the_human-robot_domain_discrepancy_in_visual_pre-training_for_robotic.md)
- [Instruction-based Image Manipulation by Watching How Things Move](instruction-based_image_manipulation_by_watching_how_things_move.md)
- [VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions](../../NeurIPS2025/robotics/vitrix-clipin_enhancing_fine-grained_visual_understanding_in_clip_via_instructio.md)
- [RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](roboground_robotic_manipulation_with_grounded_vision-language_priors.md)
- [3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](3d-mvp_3d_multiview_pretraining_for_manipulation.md)

<!-- RELATED:END -->
