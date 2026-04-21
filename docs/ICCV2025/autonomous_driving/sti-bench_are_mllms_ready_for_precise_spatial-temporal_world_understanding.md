---
title: >-
  [论文解读] STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?
description: >-
  [ICCV 2025][自动驾驶][MLLM 基准] 提出 STI-Bench，一个评估多模态大语言模型（MLLM）精确时空理解能力的基准，涵盖桌面/室内/户外三大场景、8类静态+动态任务超 2000 道 QA 对，揭示当前最强 MLLM（Gemini-2.5-Pro）平均准确率仅 41.4%，在精确空间量化和时序动态理解上存在根本性不足。
tags:
  - ICCV 2025
  - 自动驾驶
  - MLLM 基准
  - spatial-temporal understanding
  - 具身智能
  - 视频问答
---

# STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?

**会议**: ICCV 2025  
**arXiv**: [2503.23765](https://arxiv.org/abs/2503.23765)  
**代码**: [项目主页](https://mint-sjtu.github.io/STI-Bench.io/)  
**领域**: 自动驾驶  
**关键词**: MLLM 基准, spatial-temporal understanding, 具身智能, 视频问答, 自动驾驶

## 一句话总结

提出 STI-Bench，一个评估多模态大语言模型（MLLM）精确时空理解能力的基准，涵盖桌面/室内/户外三大场景、8类静态+动态任务超 2000 道 QA 对，揭示当前最强 MLLM（Gemini-2.5-Pro）平均准确率仅 41.4%，在精确空间量化和时序动态理解上存在根本性不足。

## 研究背景与动机

MLLM 越来越多地被用作具身 AI 和自动驾驶的端到端解决方案，但一个关键问题被忽视：

**语义理解 vs 精确时空理解的鸿沟**：现有评估主要聚焦 2D 视觉感知和语义 QA，但具身任务需要精确的 3D 空间度量和物理运动理解

**现有基准的局限**：
   - VSI-Bench 仅涵盖有限场景和任务类型
   - EmbodiedBench 等依赖仿真环境
   - 没有基准同时覆盖静态空间度量和动态运动分析，且使用真实世界数据

**核心追问**：**MLLM 是否真正准备好进行精确的时空世界理解？**

设计理念：使用视频（而非点云）作为输入，因为 (1) GPT-4o/Gemini 等主流模型接受图像/视频输入 (2) 视频在日常中更普遍且包含足够推理时空的信息。

## 方法详解

### 整体框架

STI-Bench 包含 300+ 真实世界视频、2064 道 QA 对，覆盖三大场景（Desktop/Indoor/Outdoor）和八大任务。采用五选一选择题形式，随机基线准确率 20%。

### 关键设计

1. **八大任务体系**: 分为静态理解（3类）和动态理解（5类）：

   **静态理解**：
    - **尺寸度量（Dim. Meas.）**：从 2D 像素推断物理尺寸，如"这个箱子多高？"
    - **空间关系（Spatial Relation）**：判断物体间相对位置，如"椅子在桌子左边还是右边？"
    - **3D 视频定位（3D Video Grounding）**：根据语义描述检索 3D 边界框

   **动态理解**：
    - **位移与路径长度（Disp. & P.L.）**：跨帧追踪运动距离
    - **速度与加速度（Speed & Acc.）**：整合空间位移和时间间隔计算运动参数
    - **自中心朝向（Ego Orient.）**：理解相机水平方位变化
    - **轨迹描述（Traj. Desc.）**：将运动路径抽象为自然语言
    - **位姿估计（Pose Est.）**：给定初始位姿，估计特定时刻的相机位姿

2. **数据来源与多尺度设计**: 三个数据集覆盖不同空间尺度：

    - **Waymo**（户外）：自动驾驶场景，分米-米级精度
    - **ScanNet**（室内）：室内 3D 场景重建，厘米-分米级
    - **Omni6DPose**（桌面）：6D 物体位姿估计，毫米级

3. **精细化干扰项设计**: 根据场景类型设定不同误差范围（Desktop: 0.5-5cm, Indoor: 5-50cm, Outdoor: 0.5-5m），使用对数采样生成干扰项：

    $e = E_{min} \cdot (E_{max}/E_{min})^u, \quad u \sim \mathcal{U}(0,1)$

   通过加权平均调整干扰项与正确答案的距离，确保最小距离恰好等于采样误差值 $e$。设计动机：不同场景精度要求差异巨大，统一的干扰项生成会导致某些场景过于简单或过于困难。

4. **构建流水线**: 数据集标注 → 自动 QA 生成（MLLM 辅助） → 多轮人工质检 → 选项随机打乱

### 损失函数 / 训练策略

本文是 benchmark 论文，不涉及模型训练。评估协议：统一采样 30 帧（Claude 为 20 帧），提示中标明采样 FPS，直接比对模型选择与 ground truth。

## 实验关键数据

### 主实验（全模型对比）

| 模型 | 排名 | Avg | Dim.Meas. | Spatial Rel. | 3D Ground. | Disp.&PL | Speed&Acc | Ego Orient. | Traj.Desc. | Pose Est. |
|------|------|-----|-----------|-------------|------------|----------|-----------|-------------|------------|-----------|
| Gemini-2.5-Pro | 1 | 41.4 | 38.7 | 53.8 | 36.9 | 33.9 | 33.1 | **52.5** | 47.4 | 50.4 |
| Qwen2.5-VL-72B | 2 | 40.7 | 31.5 | 47.6 | 39.1 | 25.1 | 38.4 | 43.8 | 51.3 | 60.6 |
| Claude-3.7-Sonnet | 3 | 40.5 | 29.8 | 45.5 | 35.7 | 28.9 | 38.8 | 40.0 | 47.4 | **62.6** |
| GPT-4o | 8 | 34.8 | 27.1 | 51.8 | 29.0 | 23.2 | 35.4 | 33.7 | 32.0 | 53.6 |
| MiniCPM-V-2.6 | 10 | 26.9 | 27.7 | 44.5 | 29.0 | 19.0 | 25.7 | 7.0 | 30.8 | 35.6 |

- 最佳模型 Gemini-2.5-Pro 平均仅 **41.4%**（随机 baseline 20%）
- 精确量化任务最难：尺寸度量 38.7%、位移路径 33.9%、速度加速度 33.1%
- 相对理解任务较好：空间关系 53.8%、自中心朝向 52.5%

### 跨场景对比

| 模型 | Outdoor | Indoor | Desktop | Overall |
|------|---------|--------|---------|---------|
| Qwen2.5-VL-72B | **50.6** | 35.0 | 33.5 | 40.7 |
| Gemini-2.5-Pro | 48.7 | 37.1 | **35.8** | **41.4** |
| Claude-3.7-Sonnet | 47.2 | **38.2** | 32.3 | 40.5 |
| GPT-4o | 41.4 | 33.1 | 27.3 | 35.2 |

- 所有模型在**户外**表现最好，**桌面**最差
- 推测与训练数据分布有关（自动驾驶数据远多于桌面操作数据）

### 消融/错误分析

| 错误类型 | 占比 | 影响任务 |
|---------|------|---------|
| 空间量化不准确 | ~40% | 尺寸度量、距离估计、3D定位 |
| 时序动态理解错误 | ~35% | 位移、速度、轨迹描述 |
| 跨模态融合不足 | ~25% | 所有依赖指令或初始条件的任务 |

### 关键发现

- **MLLM 的"精确时空理解"远未成熟**：最强模型仅 41.4%，大量任务接近随机猜测
- **三大根本性不足**：
  1. **空间量化不准确**：缺乏视觉尺寸参考、2D 像素到物理度量的映射困难
  2. **时序动态理解错误**：无法有效追踪跨帧运动、区分相机运动与物体运动
  3. **跨模态融合不足**：误解时间约束（如"从1s到18s"）、无法整合初始条件
- **模型尺度效应显著**：72B/78B 模型（40%+）显著优于 7B 模型（26-35%）
- **户外 > 室内 > 桌面**：可能反映训练数据偏向

## 亮点与洞察

- **填补了精确时空评估的空白**：此前的 benchmark 要么关注语义、要么仅覆盖部分任务，STI-Bench 首次系统化覆盖 8 类精确时空任务
- **多尺度场景设计精心**：从毫米级桌面到米级户外，全面测试不同空间尺度下的精度
- **错误分析深入**：不仅报告数字，还通过 Gemini-2.5-Pro 的推理过程分析了三类根本性错误模式，为未来改进提供方向
- **对数采样干扰项**：确保选项差异分布均匀，避免人为制造过于简单/困难的问题

## 局限与展望

- 2000+ QA 对的规模相对较小，可能不足以进行统计显著性检验
- 仅测试了视频输入（30帧均匀采样），未探索不同采样策略的影响
- 未包含多传感器融合场景（如 LiDAR+Camera）
- 桌面场景数据来源单一（Omni6DPose），多样性有限
- 可增加更多动态任务如物体交互预测、碰撞预测等

## 相关工作与启发

- **VSI-Bench** 是最接近的前作，但仅覆盖有限场景和空间任务
- 结果表明 MLLM 的"世界模型"能力存在严重不足，这对依赖 MLLM 做端到端自动驾驶或机器人操作的研究路线提出警示
- 未来改进方向：引入 3D 几何先验（深度估计预训练）、物理引擎辅助训练、更密帧采样

## 评分

- 新颖性：⭐⭐⭐⭐ — 首个全面覆盖精确时空理解的 benchmark
- 技术深度：⭐⭐⭐ — benchmark 论文技术深度有限，但任务设计和干扰项生成有 thoughtful 设计
- 实验充分度：⭐⭐⭐⭐⭐ — 10 个模型（含 4 个闭源 SOTA）+ 跨场景 + 跨任务 + 深入错误分析
- 实用价值：⭐⭐⭐⭐ — 为具身 AI 和自动驾驶社区提供重要评估工具和改进方向

<!-- RELATED:START -->

## 相关论文

- [Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning](../../AAAI2026/autonomous_driving/dual-branch_spatial-temporal_self-supervised_representation_for_enhanced_road_ne.md)
- [EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)
- [MCAM: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding](mcam_multimodal_causal_analysis_model_for_ego-vehicle-level_driving_video_unders.md)
- [Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)

<!-- RELATED:END -->
