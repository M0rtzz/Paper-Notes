---
title: >-
  [论文解读] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs
description: >-
  [ICLR 2026][多模态][空间推理] 提出 SpinBench，一个以认知科学为基础的诊断性基准测试，通过 7 类渐进式空间推理任务（从物体识别到视角转换）系统评估 37 个 VLMs 的空间理解能力，揭示了模型存在的自我中心偏差、旋转理解薄弱等系统性缺陷。
tags:
  - ICLR 2026
  - 多模态
  - 空间推理
  - 视角转换
  - 心理旋转
  - 视觉语言模型
  - benchmark
---

# SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs

**会议**: ICLR 2026  
**arXiv**: [2509.25390](https://arxiv.org/abs/2509.25390)  
**代码**: [https://spinbench25.github.io/](https://spinbench25.github.io/)  
**领域**: 多模态视觉语言模型 (Multimodal VLM)  
**关键词**: 空间推理, 视角转换, 心理旋转, 视觉语言模型, benchmark

## 一句话总结

提出 SpinBench，一个以认知科学为基础的诊断性基准测试，通过 7 类渐进式空间推理任务（从物体识别到视角转换）系统评估 37 个 VLMs 的空间理解能力，揭示了模型存在的自我中心偏差、旋转理解薄弱等系统性缺陷。

## 研究背景与动机

空间推理是人类认知的基本组成部分，也是具身智能体在物理世界中操作的关键能力。虽然视觉语言模型（VLMs）在视觉理解方面取得了令人瞩目的进展，但它们的空间推理能力仍然知之甚少且诊断不足。

现有方法的几个关键问题：

**评估纠缠问题**：现有应用（导航、操纵、自动驾驶等）主要反映端到端性能，空间推理与高层语言和规划目标纠缠在一起，无法直接测试模型是否真正理解旋转、平移、物体相对位姿和视角变化等几何基元。

**数据集偏差依赖**：不清楚 VLMs 是否真正具有空间推理能力，还是依赖数据集偏差和浅层模式匹配。

**现有基准不足**：已有的空间推理基准（如 CLEVR、BLINK、SpaCE-10 等）在以下方面存在不足：缺乏对视角的控制变化、缺少参考框架变化的测试、不支持多帧推理、将空间推理与功能和物理常识纠缠在一起。

本文的核心研究问题源自认知科学的基础洞察——经典的心理旋转实验（Shepard & Metzler, 1971）证明空间认知通常依赖于模拟的、基于意象的过程。由此引出核心问题：**VLMs 能否进行这种基于意象的空间推理，还是仅限于符号和语言关联？**

## 方法详解

### 整体框架

SpinBench 围绕**视角转换（perspective taking）**这一核心挑战设计。视角转换是一种高度整合性能力，需要识别跨视角物体、定位相对位置、以及模拟心理变换。SpinBench 将这种高级推理能力分解为一组有针对性的诊断类别，每个类别代表支持视角转换的基本空间推理能力。

为减少混淆因素，所有任务都在**水平 2D 平面**中定义。排除了垂直关系（如上/下），视角变化限制为围绕场景的水平轨道。

### 关键设计

1. **七大任务类别的渐进式结构**：

    - **身份匹配（Identity Matching）**：评估模型是否能在不同视角下一致识别同一物体。这是跨视图推理的前提条件。405 个样本。
    - **物体关系定位（Object-Relation Grounding）**：测试在单张静态图像中理解物体相对配置的能力，包括方向关系（左/右、前/后）和距离关系（近/远）。636 个样本，是最大的类别。
    - **动态平移（Dynamic Translation）**：评估对线性物体位移的推理能力。给定两帧时间有序的图像，模型需识别物体相对观察者的移动方向。156 个样本。
    - **动态旋转（Dynamic Rotation）**：专注于旋转变换。模型需判断旋转方向（如顺时针 vs 逆时针）。353 个样本。
    - **规范视角选择（Canonical View Selection）**：检验模型是否能映射物体的不同规范视角（给定正面，选择左/右/后视图）。358 个样本。
    - **心理旋转（Mental Rotation）**：测试模型能否模拟物体变换——给定物体和指定的旋转角度/方向，选择正确的结果配置。78 个样本。
    - **视角转换（Perspective Taking）**：SpinBench 的核心任务，要求在视角变化下推理整个场景。包括两个子类型：(S) 从新视角选择正确场景图像；(T) 预测视角转换下物体关系如何变换。613 个样本。

2. **四大数据域的多样性覆盖**：

    - **Infinigen 合成场景**（54.1%）：使用 Infinigen 在 Isaac Sim 环境中生成的室内桌面多物体场景，物体来自 YCB 数据集
    - **ABO 物体**（23.7%）：来自 Amazon Berkeley Objects 数据集的日用品，提供 360° 视图
    - **汽车**（9.2%）：Multi-View Car Dataset 的车辆旋转序列
    - **人脸**（13.0%）：Stereo Face Database 的 8 种姿态

3. **精细控制变量**：

    - **参考框架变化**：测试自我中心（egocentric）vs 物体中心（allocentric）参考框架的切换能力
    - **对称增强**：通过翻转关系和答案生成逻辑等价变体（如"左→右"）
    - **句法增强**：在保持语义的前提下重新表述问题
    - **前提条件变体**：区分视觉定位失败和语言推理失败

### 评估指标

- **原始准确率**：正确回答的比例
- **Cohen's kappa (κ)**：校正偶然性的准确率，考虑选项基数差异，实现跨任务公平比较
- **成对一致性**：衡量模型在逻辑等价问题对上是否产生一致结果

## 实验关键数据

### 主实验

评估了 37 个 VLMs，包括 4 个商业模型和 33 个开源模型。

| 模型 | 总体准确率 | 一致性 | 完美率 |
|------|-----------|--------|--------|
| InternVL3-38B | 73.8% | 95.7% | 71.1% |
| InternVL3.5-38B | 71.9% | 95.3% | 75.1% |
| InternVL3-14B | 70.3% | 91.4% | 63.7% |
| GPT-4.1 | 69.8% | 85.9% | 59.5% |
| GPT-4o | 67.8% | 79.6% | 51.2% |
| Claude Sonnet 4 | 64.8% | 71.7% | 42.8% |
| 人类 | 91.2% | - | - |

### 关键分类任务表现

| 任务类别 | 最佳模型κ | 一般模型表现 |
|---------|----------|------------|
| 物体关系定位 | >0.6 | 表现最佳，多数模型可靠 |
| 身份匹配 | 双模分布 | 小模型接近随机，大模型接近完美 |
| 动态旋转 | 困难 | 大多数模型表现较差 |
| 心理旋转 | 接近随机 | 大多数模型在或低于随机水平 |
| 视角转换 | 接近随机 | 最具挑战性 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 自我中心 vs 物体中心 | κ 差异可达 1.6 | Molmo-7B：自我中心 0.94，物体中心 -0.66 |
| CoT 推理（Cosmos-Reason1） | 平均 +0.221 κ | 视角转换任务最大提升 +0.650 |
| CoT 推理（SpaceOm） | 平均 +0.118 κ | 物体关系定位获益最多 |
| 有前提 vs 无前提 | 41% 模型低于随机 | 即使在纯语言推理时仍有系统性失败 |

### 关键发现

1. **系统性自我中心偏差**：模型在动态旋转任务中表现出强烈的观察者视角偏差。自我中心任务上表现最好的模型，在物体中心变体上表现最差，表明训练数据中第一人称视觉描述主导了归纳偏差。

2. **旋转理解严重不足**：心理旋转和视角转换任务中，大多数模型在或低于随机水平，表明缺乏用于旋转变换的鲁棒内部表示。

3. **涌现能力**：身份匹配展现出明确的涌现特性——小模型停留在随机水平，而大模型（7B-8B+）达到接近完美的准确率，表明跨图像 3D 抽象只有在模型达到足够容量时才可能。

4. **准确率-一致性强相关**：整体准确率和一致性之间存在强正相关（Pearson r=0.874, p<0.05），表明不能维持"A 在 B 左边"等于"B 在 A 右边"等价性的模型缺乏真正的空间理解。

5. **人类-VLM 难度对齐**：人类响应时间与 VLM 准确率之间存在显著负相关（r=-0.54, p<0.05），验证了 SpinBench 捕捉了人类和 VLMs 共享的空间推理困难。

## 亮点与洞察

1. **认知科学驱动的基准设计**：灵感来自 Shepard & Metzler 的心理旋转实验，将空间推理分解为渐进式的认知子技能，这种分层结构在之前的基准中是缺失的。

2. **诊断价值超越排行榜**：SpinBench 不仅提供分数排名，更重要的是提供一个诊断性框架，精确识别 VLMs 在空间推理中哪里成功、哪里失败，以及这些技能如何在视角转换中组合。

3. **与其他基准的互补性**：与 MindCube、ViewSpatial-Bench、OmniSpatial 和 SpaCE-10 的整体相关性较弱且不显著，验证了 SpinBench 捕捉了独特的基础能力，而非一般空间智能。

4. **实用价值**：对具身 AI 的直接指导——参考框架推理或旋转理解的失败可能导致导航、操纵和其他安全关键任务中的崩溃。

## 局限与展望

1. **空间概念覆盖不全**：尚未覆盖其他重要空间概念，如包含关系（in）、支撑关系（on）、垂直关系（above/below）。

2. **任务限于水平平面**：所有任务定义在 2D 水平平面中，排除了高度差异和垂直关系，限制了评估的全面性。

3. **规模相对较小**：总共 2,599 个样本，某些类别（如心理旋转仅 78 个）样本量较小。

4. **缺少训练集引导**：当前主要是评估基准，未探索如何利用基准数据来训练改进空间推理能力。

5. **合成数据偏重**：Infinigen 合成场景占比超过 54%，可能与真实世界场景存在分布差异。

## 相关工作与启发

- **CLEVR**：早期合成诊断数据集，仅使用简单 3D 形状
- **MindCube**：强调认知映射，关注模型如何跨场景跟踪空间信息
- **ViewSpatial-Bench**：聚焦视点依赖的定位
- **OmniSpatial**：基于认知心理学的空间任务，但将空间推理与功能和物理常识纠缠
- **SpatialReasoner / SSR / APC**：通过显式 3D 表示增强空间推理
- **MetaSpatial / SpatialVLM**：通过强化学习或大规模预训练增强空间理解
- 启发方向：现有纯语言方法在空间推理上存在根本性限制，需要**超越语言的结构化推理**

## 评分

- 新颖性: ⭐⭐⭐⭐ （认知科学驱动的渐进式诊断设计很有创意，但基准测试这一形式已较常见）
- 实验充分度: ⭐⭐⭐⭐⭐ （37 个模型、人类对照、一致性分析、缩放规律、相关性分析，非常全面）
- 写作质量: ⭐⭐⭐⭐⭐ （结构清晰，分析深入，展示方式专业）
- 价值: ⭐⭐⭐⭐ （对 VLM 空间推理的系统诊断具有重要指导意义，尤其对具身 AI 方向）

<!-- RELATED:START -->

## 相关论文

- [Through the Lens of Contrast: Self-Improving Visual Reasoning in VLMs](through_the_lens_of_contrast_self-improving_visual_reasoning_in_vlms.md)
- [Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](../../NeurIPS2025/multimodal_vlm/ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)
- [OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [SpatiaLab: Can Vision-Language Models Perform Spatial Reasoning in the Wild?](spatialab_can_vision-language_models_perform_spatial_reasoning_in_the_wild.md)

<!-- RELATED:END -->
