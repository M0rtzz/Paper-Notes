---
title: >-
  [论文解读] 4D-Bench: Benchmarking Multi-Modal Large Language Models for 4D Object Understanding
description: >-
  [ICCV 2025][视频理解][4D 理解] 4D-Bench 是首个评估多模态大语言模型（MLLM）4D 物体理解能力的基准，包含 4D 物体问答和描述两大任务，揭示了即使 SOTA GPT-4o 也仅达 63% 准确率（人类基线 91%），暴露了当前 MLLM 在多视角时序推理上的显著不足。
tags:
  - ICCV 2025
  - 视频理解
  - 4D 理解
  - MLLM 评测
  - 多视角时序推理
  - benchmark
  - 视觉问答
---

# 4D-Bench: Benchmarking Multi-Modal Large Language Models for 4D Object Understanding

**会议**: ICCV 2025  
**arXiv**: [2503.17827](https://arxiv.org/abs/2503.17827)  
**代码**: [https://4dbench.github.io/](https://4dbench.github.io/)  
**领域**: 视频理解 / 多模态  
**关键词**: 4D 理解, MLLM 评测, 多视角时序推理, benchmark, 视觉问答

## 一句话总结

4D-Bench 是首个评估多模态大语言模型（MLLM）4D 物体理解能力的基准，包含 4D 物体问答和描述两大任务，揭示了即使 SOTA GPT-4o 也仅达 63% 准确率（人类基线 91%），暴露了当前 MLLM 在多视角时序推理上的显著不足。

## 研究背景与动机

4D 数字资产（动态 3D 物体）在数字孪生、增强现实、游戏等领域日益重要。随着 MLLM（如 GPT-4o、Qwen2-VL）在 2D 图像/视频理解上取得长足进步，一个自然的问题是：**这些模型能否理解 4D 物体？**

然而，当前存在一个关键空白：

**无公开的 4D 语言理解基准**：现有基准要么关注 2D 图像/视频（忽视多视角理解），要么关注静态 3D 场景（忽视时序动态）

**4D 理解的独特挑战**：
   - **多视角歧义**：同一物体从不同视角观察呈现不同外观，需要综合多视角信息
   - **时序演变**：物体部件随时间运动，需要跟踪和推理
   - **跨视角-时间的联合推理**：如图 1 所示，机器人的右手在某些视角模糊、最终消失，回答问题需要选择正确视角、定位部件、追踪变化

**反事实测试**：4D-Bench 的合成物体可提供违反物理规律或常识的反事实数据（如 6 条腿的蜘蛛、球从洞中滚出），测试 MLLM 是否真正理解输入而非依赖记忆

**核心 idea**：将 4D 物体渲染为多视角视频，直接输入现有 MLLM 进行评测，无需建立新的 4D 理解模型。通过精心设计的评测任务暴露 MLLM 的具体短板。

## 方法详解

### 整体框架

4D-Bench 由两大任务组成：
1. **4D 物体问答（QA）**：751 个四选一问题，736 个 4D 物体
2. **4D 物体描述（Captioning）**：580 个 4D 物体，每个 5 条人工标注描述

### 关键设计

#### 1. 数据收集与清洗

数据源自 Objaverse-XL 的动态 3D 物体，经过两阶段清洗：

- **运动分析**：通过像素变化检测识别运动边界，提取有效视频段，确保只包含动态物体
- **视觉质量评估**：人工标注数千张图像为高/低质量，微调 CLIP 图像编码器作为质量分类器，多视角投票过滤低质量物体

每个 4D 物体渲染 24 个视角的视频。

#### 2. 问答任务设计（5 个子任务）

| 子任务 | 评测内容 | 独特挑战 |
|-------|---------|---------|
| **外观（Appearance）** | 视觉属性分析 | 合成/虚构物体偏离真实世界训练分布 |
| **动作（Action）** | 细粒度运动检测 | 需多视角观察才能判断运动方向 |
| **计数（Object Counting）** | 动态场景中精确计数 | 物体出现/消失 + 跨视角遮挡 |
| **空间关系（Spatial Relationship）** | 跨视角空间配置理解 | 不同角度的空间关系不同 |
| **时序关系（Temporal Relationship）** | 时间演变和顺序理解 | 需在时间和视角两个维度联合推理 |

#### 3. 标注流程

**QA 标注**：采用混合方式
- 专业标注团队初始标注（保留率从 92% 降至 62.5%，凸显质量控制难度）
- 后续用 GPT-4o/Qwen2-VL 生成候选 QA → Qwen2-VL 7B 初筛 → 纯文本盲测（Qwen2.5 + Llama 3.1，删去两者都能正确回答的） → 人工终审
- 最终保留 751 个高质量 QA 对

**描述标注**：纯人工标注，5 位专业标注员独立为每个物体写描述，审核员确保描述捕捉重要细节。

#### 4. 评测设置

- 从 24 视角中均匀选取 $K=3$ 个视角
- 每个视角采样 $N=6$ 帧 → 输入为 $3 \times 6 = 18$ 帧
- 描述任务使用 GPT-4o 作为评估器，分别给出 GPT-Appearance 和 GPT-Action 评分（0-5 分）

### 损失函数 / 训练策略

4D-Bench 是评测基准，不涉及模型训练。

## 实验关键数据

### 主实验

**4D 物体问答准确率（%）**：

| 模型 | 计数 | 时序关系 | 动作 | 空间关系 | 外观 | 总体 |
|-----|------|---------|------|---------|------|------|
| MiniGPT4-Video | 22.05 | 26.43 | 22.90 | 22.39 | 22.06 | 23.17 |
| Qwen2-VL 7B | 38.58 | 56.43 | 57.94 | 58.96 | 71.32 | 56.99 |
| LLaVA-Video 72B | 54.33 | 58.57 | 57.48 | 66.42 | 77.21 | 62.32 |
| GPT-4o | 44.09 | 59.29 | 63.55 | 69.40 | 77.21 | **62.98** |
| 所有模型平均 | 37.29 | 49.29 | 49.37 | 53.57 | 63.92 | 50.69 |
| **人类** | **88.98** | **89.29** | **94.39** | **91.04** | **89.71** | **91.08** |

GPT-4o 与人类基线差距近 **28 个百分点**。

### 消融实验（视角数和采样频率的影响）

| 设置变化 | 准确率变化（Gemini 1.5 Flash） |
|---------|--------------------------|
| 1 视角 → 3 视角（固定 6 帧） | 41.3% → 53.7%（+12.4%） |
| 1 帧 → 6 帧（固定 3 视角） | 46.3% → 53.7%（+7.4%） |
| 3 视角 → 6 视角 | 53.7% → 下降（信息冗余） |
| 6 帧 → 9 帧 | 提升可忽略 |

结论：问题确实需要多视角+时序信息，但超过 3 视角/6 帧后信息冗余反而干扰模型。

**描述任务 GPT-Eval 评分**：

| 模型 | GPT-Appearance | GPT-Action | GPT-Eval |
|-----|---------------|-----------|---------|
| Qwen2-VL 72B | 3.324/5 | 2.791/5 | 3.057/5 |
| Gemini 1.5 Pro | 3.311/5 | 2.983/5 | 3.147/5 |
| GPT-4o | 3.507/5 | 3.258/5 | 3.382/5 |
| **人类** | **3.772/5** | **3.879/5** | **3.826/5** |

### 关键发现

1. **计数是最难的任务**：所有模型平均仅 **37.29%**（接近随机猜测 25%），需要跨视角信息整合来消除遮挡
2. **外观理解 >> 动作理解**：外观平均 63.92% vs 动作 49.37%，差距约 15%
3. **开源 vs 闭源的差距在动作理解上更大**：外观方面开源模型已接近闭源水平，但动作理解差距显著
4. **反事实数据暴露模型的"记忆依赖"**：面对 6 条腿蜘蛛或物理违反场景，所有 MLLM 都给出错误答案，说明它们依赖世界知识而非真正理解视觉输入
5. **鲁棒性良好**：改变帧排列顺序（视角优先 vs 时间优先）或添加时间戳对结果影响微小

## 亮点与洞察

- **填补 4D-语言理解评测空白**：在 3D（静态）和 2D 视频（单视角）之间开辟了全新的评测维度
- **反事实测试的巧妙设计**：合成数据天然提供了超越真实世界的 OOD 评测，是 2D benchmark 无法做到的
- **数据质量把控严格**：混合标注流程（人工+MLLM+盲测+终审）确保问题确实需要多视角时序推理
- **实用的评测发现**：计数能力差 → 需要更好的跨视角对应关系建模；动作弱 → 需要更强的时序编码器

## 局限与展望

- 当前使用多视角视频拼接作为 4D 输入，而非原生 4D 表示（如点云序列、4D Gaussian Splatting）——受限于当前 MLLM 的输入模态
- 数据集规模相对有限（751 QA + 580 captioning），可能不足以支撑全面的统计结论
- 物体来自 Objaverse-XL，以合成为主，外观和运动可能与真实世界有分布差异
- 未评测专门的 3D/4D 理解模型（如 3D-LLM），仅评测了通用 2D MLLM

## 相关工作与启发

- **MVBench** [Li et al., 2024]：视频理解多任务 benchmark，但仅单视角
- **ScanQA** [Azuma et al., 2022]：3D 场景问答，但仅静态场景
- **T3Bench** [He et al., 2023]：评测 Text-to-3D 生成，关注生成质量而非理解
- **4DGS** [Wu et al., 2024]：4D 高斯溅射，提供 4D 表示但缺少语言理解评测
- 启发：未来 MLLM 需要原生 4D 输入支持（而非多视角视频代理），以及更强的时序建模能力

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首个 4D 物体理解 benchmark，问题定义开创性
- 技术深度：⭐⭐⭐ — 以评测为主，方法论贡献相对有限
- 实验充分度：⭐⭐⭐⭐⭐ — 14 个 MLLM、5 个子任务、多维分析（视角数/帧数/排列/反事实）
- 实用性：⭐⭐⭐⭐ — 为 MLLM 的 4D 理解能力提供了明确的改进方向

<!-- RELATED:START -->

## 相关论文

- [Aligning Effective Tokens with Video Anomaly in Large Language Models](aligning_effective_tokens_with_video_anomaly_in_large_language_models.md)
- [DisTime: Distribution-based Time Representation for Video Large Language Models](distime_distribution-based_time_representation_for_video_large_language_models.md)
- [Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)
- [DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)

<!-- RELATED:END -->
