---
title: >-
  [论文解读] SITE: towards Spatial Intelligence Thorough Evaluation
description: >-
  [ICCV2025][机器人][空间智能] 本文提出 SITE，一个基于认知科学三重分类体系的空间智能综合基准，涵盖 8,068 个多选 VQA 任务（覆盖 31 个数据集、图像+视频），评估结果显示当前最强 VLM（GPT-4o）在整体空间推理上仍落后人类专家约 32%，且 VLM 的空间智能与机器人操控任务的成功率呈高度正相关（Pearson $r=0.902$）。
tags:
  - ICCV2025
  - 机器人
  - 空间智能
  - VLM基准测试
  - 多视角推理
  - 认知科学
  - 具身AI
---

# SITE: towards Spatial Intelligence Thorough Evaluation

**会议**: ICCV2025  
**arXiv**: [2505.05456](https://arxiv.org/abs/2505.05456)  
**代码**: https://github.com/SITE-project-page (待确认)  
**领域**: 机器人 / 空间智能 / VLM评测  
**关键词**: 空间智能, VLM基准测试, 多视角推理, 认知科学, 具身AI

## 一句话总结
本文提出 SITE，一个基于认知科学三重分类体系的空间智能综合基准，涵盖 8,068 个多选 VQA 任务（覆盖 31 个数据集、图像+视频），评估结果显示当前最强 VLM（GPT-4o）在整体空间推理上仍落后人类专家约 32%，且 VLM 的空间智能与机器人操控任务的成功率呈高度正相关（Pearson $r=0.902$）。

## 研究背景与动机

**领域现状**：空间智能（Spatial Intelligence, SI）是认知科学的核心能力之一，涵盖对空间关系的可视化、操控与推理。当前大型视觉语言模型（VLM）在通用 VQA 上已取得显著进展，但其空间推理能力仅被零散地纳入 MME、MMBench 等综合基准中，缺乏系统性评估。

**现有痛点**：已有空间推理基准存在明显不足——CVBench 缺少视角变换任务，3DSRBench 仅限单张图像，VSI-Bench 仅涵盖室内场景的视频。这些基准都只覆盖了空间智能的部分维度，无法给出全面画像。

**核心矛盾**：认知科学中空间智能至少有三种主要分类体系（尺度分类、视角分类、2x2 intrinsic/extrinsic × static/dynamic），但没有任何现有基准同时覆盖这三套分类，尤其缺乏视角变换（spatial orientation）和动态场景下的空间推理任务。

**本文目标**：构建一个同时满足三种认知科学分类体系的综合空间智能基准，弥补视角变换和动态推理任务的空白，并系统评估当前 VLM 的空间推理盲区。

**切入角度**：采用"自下而上 + 自上而下"的双路策略——自下而上从 30 个已有数据集中筛选空间相关任务，自上而下从认知科学分类体系中发现缺失维度并设计新任务。

**核心 idea**：用认知科学的三套空间智能分类体系驱动基准设计，结合已有任务筛选和新任务创建（Ego-Exo 视角关联 + 帧排序），实现对 VLM 空间智能的全面系统评估。

## 方法详解

### 整体框架
SITE 的构建分为两条路径：**自下而上（Bottom-Up）**从 30 个已有视觉数据集中筛选空间相关任务，得到 6,943 个 QA 对；**自上而下（Top-Down）**基于认知科学分类体系发现任务未覆盖的维度，设计了两类新任务（1,125 个 QA 对），最终合并为 8,068 个标准化多选 VQA 任务。

### 关键设计

1. **三重认知科学分类体系**:

    - 功能：用三套互补的 SI 分类系统来指导基准设计的覆盖范围
    - 核心思路：(a) **尺度分类**——figural（小于人体、单视角可感知）、vista（房间级）、environmental（需导航才能感知），不同尺度涉及不同的脑区和机制；(b) **VZ/SO 分类**——spatial visualization（不依赖自身视角的心理旋转）vs spatial orientation（想象从不同视角观察）；(c) **2×2 分类**——intrinsic/extrinsic × static/dynamic
    - 设计动机：单一分类无法全面覆盖空间智能的各个方面，三套互补分类确保评估无死角

2. **自下而上的数据收集与筛选**:

    - 功能：从 22 个图像数据集和 8 个视频数据集中筛选空间相关任务
    - 核心思路：先用数据集自带标签过滤，再用 GPT-4o 进行两阶段筛选——先用文本部分快速筛选，再联合视觉模态二次筛选。最终生成 6 个粗粒度空间任务类别：Counting、Relationship Reasoning、Localization、3D Information、Multi-View Reasoning、Movement Prediction
    - 设计动机：不同数据集标签体系不同，需要统一分类；223K 初始样本经筛选后做分层采样确保类别平衡

3. **自上而下的新任务设计（Ego-Exo4D）**:

    - 功能：基于 Ego-Exo4D 数据集设计两类填补空白的新任务
    - **Ego-Exo 视角关联**：给定自我中心视角图像，从候选中选出匹配的外部视角图像（或反向），考察外在-静态维度的视角变换能力
    - **帧顺序重排**：从视频中提取起止帧和中间关键帧并打乱顺序，要求模型推断正确的时间顺序，考察外在-动态维度的时空推理
    - 设计动机：分析发现已有任务严重缺乏视角变换（spatial orientation）和动态场景推理，而这恰恰是空间智能的核心因素

4. **Chance-Adjusted Accuracy (CAA) 指标**:

    - 功能：设计一个消除随机猜测偏差的评估指标
    - 核心思路：$\mathcal{CAA} = (\sum X_i - \sum \frac{1}{n_i}) / (N - \sum \frac{1}{n_i})$，其中 $n_i$ 是第 $i$ 题的选项数。CAA=1 表示全对，CAA=0 表示与随机一样，CAA<0 表示比随机差
    - 设计动机：不同题目选项数不同（2-6 个），直接用准确率会引入偏差

### 损失函数 / 训练策略
SITE 是一个评估基准而非训练方法，不涉及损失函数设计。评估使用 GPT-4o 作为自动评判器解析 VLM 输出。

## 实验关键数据

### 主实验

| 模型 | Overall CAA | Counting | Localization | 3D Info | Multi-View | Relation | Movement |
|------|------------|----------|-------------|---------|------------|----------|----------|
| Human | 67.5 | 66.0 | 83.3 | 54.7 | 87.5 | 73.0 | 52.5 |
| GPT-4o | 37.8 | 44.6 | 56.0 | 26.9 | 22.0 | 54.6 | 18.4 |
| InternVL-2.5-8B | 32.8 | 47.1 | 37.0 | 23.2 | 9.05 | 47.6 | 28.7 |
| Qwen2.5-VL-7B | 31.4 | 52.6 | 44.1 | 9.42 | 1.08 | 51.5 | 18.9 |
| Gemini-1.5-Pro | 32.5 | 48.0 | 45.8 | 25.3 | 5.33 | 48.8 | 18.4 |
| LLaVA-OV-7B | 30.2 | 51.8 | 38.5 | 22.4 | 9.40 | 55.3 | 9.18 |

### 新任务（View Association & Frames Reordering）

| 模型 | View Assoc. ego2exo | View Assoc. exo2ego | Reorder ego2exo | Reorder exo2ego |
|------|---------------------|---------------------|------------------|-----------------|
| Human | 100 | 100 | 98 | 96 |
| GPT-4o | 35.70 | 20.70 | -2.01 | -5.16 |
| Qwen2.5-VL-7B | 5.09 | -3.80 | 7.63 | 4.23 |
| InternVL-2.5-8B | -5.56 | 5.91 | 5.22 | -0.66 |

### 空间智能与机器人任务相关性

| 模型 | SITE CAA | L2 Dist ↓ | Success Rate ↑ |
|------|----------|-----------|----------------|
| LLaVA-OV-0.5B | 18.4 | 0.268 | 0.0% |
| LLaVA-OV-7B | 30.2 | 0.142 | 0.0% |
| Qwen2.5-VL-3B | 29.5 | 0.139 | 0.0% |
| Qwen2.5-VL-7B | 31.4 | 0.030 | 38.0% |

Pearson 相关系数 $r = 0.902$，表明空间智能得分与机器人操控能力高度正相关。

### 关键发现
- **Multi-View Reasoning 是 VLM 最大短板**：所有模型在该类别上 CAA 均低于 10%（GPT-4o 为 22%），而人类达到 87.5%，差距超过 65%
- **3D 理解也是持续挑战**：大多数 VLM 在 3D Information Understanding 上低于 15%
- **视角变换任务几乎全军覆没**：在帧排序任务上，多数 VLM 的 CAA 为负数（比随机猜还差），说明当前 VLM 完全不具备跨视角时序推理能力
- **模型规模有帮助但不够**：7B 版本一致优于小版本，但最强开源模型仍大幅落后人类

## 亮点与洞察
- **认知科学驱动的基准设计范式**：不是简单汇总数据集，而是用三套认知科学分类体系系统性地审视覆盖范围并针对性补充任务。这种"先建分类框架、再查缺补漏"的思路可以迁移到其他能力评估基准的设计中
- **SI 与具身 AI 的强相关性**（$r=0.902$）是一个重要发现：说明在 VQA 基准上的空间推理得分可以作为预测机器人操控能力的代理指标，为 VLM 的实用部署提供了一个低成本评估手段
- **Ego-Exo 视角关联任务的设计非常巧妙**：利用 Ego-Exo4D 数据集天然的多视角同步拍摄，构造了一个人类能轻松做到（100%）但 VLM 几乎失败的任务，精准定位了 VLM 在空间理解上的根本缺陷

## 局限与展望
- **缺少 3D 输入模态**：所有任务都是 2D 图像/视频输入，没有评估点云、深度图等 3D 原始输入下的空间推理
- **人类标注规模有限**：上界评估仅用 7 名参与者在小子集上测量，代表性不够
- **评估依赖 GPT-4o**：用 LLM 解析 VLM 输出可能引入额外噪声
- **SI 与具身 AI 的实验规模较小**：仅在 LIBERO-Spatial 一个任务上用 4 个模型验证相关性，需要更大规模验证
- **可改进方向**：可以增加交互式空间推理任务（如需要多步导航才能回答的问题）；可以引入 3D 点云或深度图作为输入模态；可以评估最新的推理增强 VLM（如 o1-style 模型）

## 相关工作与启发
- **vs CVBench**: CVBench 仅关注 vista 尺度的空间关系，缺乏视角变换任务。SITE 在三个尺度和视角变换上都有覆盖，更全面
- **vs VSI-Bench**: VSI-Bench 用视频评估空间推理但仅限室内场景。SITE 涵盖室内外多种场景，且包含 figural 尺度
- **vs 3DSRBench**: 仅限单图像，无法评估动态场景中的空间推理。SITE 同时包含图像和视频输入
- 本文的 CAA 指标设计简洁有效，适合推广到其他不等选项数的多选基准中

## 评分
- 新颖性: ⭐⭐⭐⭐ 认知科学三重分类驱动基准设计的思路新颖，但基准类工作创新空间有限
- 实验充分度: ⭐⭐⭐⭐ 评估了 9 个 VLM，包含人类上界和具身 AI 相关性分析
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、叙述逻辑流畅，认知科学背景介绍详尽
- 价值: ⭐⭐⭐⭐ 揭示了 VLM 空间智能的短板，但作为基准需要后续迭代维护才能持续发挥价值

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] On the Generalization Capacities of MLLMs for Spatial Intelligence](../../ICLR2026/robotics/on_the_generalization_capacities_of_mllms_for_spatial_intelligence.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[NeurIPS 2025\] Rethinking the Simulation vs. Rendering Dichotomy: No Free Lunch in Spatial World Modelling](../../NeurIPS2025/robotics/rethinking_the_simulation_vs_rendering_dichotomy_no_free_lunch_in_spatial_world_.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](../../NeurIPS2025/robotics/learning_spatial-aware_manipulation_ordering.md)
- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](../../NeurIPS2025/robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)

<!-- RELATED:END -->
