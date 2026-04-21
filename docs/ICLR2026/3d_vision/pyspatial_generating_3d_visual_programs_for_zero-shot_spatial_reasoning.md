---
title: >-
  [论文解读] pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning
description: >-
  [ICLR 2026][3D视觉][视觉编程] pySpatial 是一个视觉编程框架，让 MLLM 通过生成 Python 代码自动调用 3D 空间工具（3D 重建、相机位姿恢复、新视角渲染等），将有限的 2D 图像输入转化为可交互探索的 3D 场景，实现零样本、即插即用的显式 3D 空间推理，在 MindCube 基准上以 58.56% 的整体准确率超越 GPT-4.1-mini 12.94%、超越 VLM-3R 16.5%，并成功驱动真实四足机器人完成室内导航。
tags:
  - ICLR 2026
  - 3D视觉
  - 视觉编程
  - 3D重建
  - 空间推理
  - 零样本
  - 机器人导航
---

# pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning

**会议**: ICLR 2026  
**arXiv**: [2603.00905](https://arxiv.org/abs/2603.00905)  
**代码**: [项目页面](https://pySpatial.github.io)  
**领域**: 3D空间推理/视觉编程  
**关键词**: 视觉编程, 3D重建, 空间推理, 零样本, 机器人导航

## 一句话总结

pySpatial 是一个视觉编程框架，让 MLLM 通过生成 Python 代码自动调用 3D 空间工具（3D 重建、相机位姿恢复、新视角渲染等），将有限的 2D 图像输入转化为可交互探索的 3D 场景，实现零样本、即插即用的显式 3D 空间推理，在 MindCube 基准上以 58.56% 的整体准确率超越 GPT-4.1-mini 12.94%、超越 VLM-3R 16.5%，并成功驱动真实四足机器人完成室内导航。

## 研究背景与动机

**领域现状**：MLLM（GPT-4o、Claude 等）在图像描述、视频理解等任务上表现卓越，但在 3D 空间推理方面仍然非常薄弱。最新研究显示，MLLM 在多视角空间推理任务（如"从视角 1 到视角 2 需要怎么移动？"）上的表现仅略高于随机猜测。

**现有痛点**：

1. **训练数据瓶颈**：MLLM 在海量图文对上预训练，但显式 3D 空间监督数据极度稀缺且标注成本高，导致模型难以建立语言与 3D 空间结构的可靠对应
2. **隐式推理不可靠**：现有方法（如认知地图、思维链）依赖 MLLM 的"隐式想象"来构建空间模型，效果有限且不可控
3. **单视角局限**：SpatialVLM、SpatialRGPT 等方法仅处理单视角空间理解，无法应对多视角推理
4. **需要微调**：特化空间模型（如 VLM-3R）需要在合成数据上微调，不具备即插即用的灵活性

**核心矛盾**：MLLM 缺乏对 3D 世界的显式几何理解，仅靠隐式推理无法可靠解决空间问题。

**本文方案**：不让 MLLM 隐式想象 3D ，而是通过视觉编程范式让 MLLM 生成 Python 代码调用 3D 工具，显式构建、探索和推理 3D 场景——将"想象"转化为"计算"。

## 方法详解

### 整体框架

pySpatial 的工作流程分三步：

$$\text{图像序列} + \text{语言查询} \xrightarrow{\text{代码代理 } \mathcal{F}} \text{Python 程序 } z \xrightarrow{\text{解释器 } \mathcal{E}} \text{中间输出 } O \xrightarrow{\text{MLLM } \mathcal{M}} \text{最终回答 } r$$

1. **程序生成**：代码代理（默认 GPT-4o）根据查询 $q$ 生成调用 pySpatial API 的 Python 程序 $z = \mathcal{F}(q)$
2. **程序执行**：Python 解释器执行程序，调用 3D 工具产生中间结果 $O = \mathcal{E}(z, \mathcal{I})$（文本/图像/渲染视图）
3. **最终推理**：MLLM 综合原始图像、程序输出和查询生成最终答案 $r = \mathcal{M}(\mathcal{I}, O, q)$

### 关键设计一：模块化空间工具 API

pySpatial 定义了一套简洁的 Python API，将底层复杂实现封装为高层语义操作：

| API 函数 | 功能 | 默认参数 |
|----------|------|----------|
| `reconstruct(scene)` | 从图像序列进行 3D 重建 | - |
| `describe_camera_motion(recon)` | 将相机位姿描述为自然语言 | - |
| `synthesize_novel_view(recon, pose)` | 从任意视角渲染新视图 | - |
| `rotate_right/left(ext, angle)` | 左/右旋转相机位姿 | 45° |
| `move_forward/backward(ext, dist)` | 前/后移动相机 | 0.3 |
| `turn_around(ext)` | 相机 180° 转向 | - |

**3D 重建工具**：根据任务需求选用 CUT3R（度量尺度，用于真实导航）或 VGGT（归一化空间，用于基准评测），通过反投影将像素映射到世界坐标：

$$\mathbf{X}_i = \mathbf{G}_n^{-1} \pi^{-1}(\mathbf{p}_i, D_n(\mathbf{p}_i), K^{-1})$$

**相机运动描述**：将相机位姿矩阵转换为自我中心运动的自然语言标签（前进、后退、左转等八个方向），通过计算世界坐标系下的位移在第一个相机帧中的偏航角 $\theta = \text{atan2}(d_x, d_z) \cdot 180/\pi$ 并离散化。

**新视角合成**：基于重建点云 $\mathcal{P}$ 和目标相机位姿进行光栅化渲染，代理发出 `rotate_left`、`turn_around` 等高层指令时自动转换为偏航旋转再渲染。

### 关键设计二：零样本视觉程序生成

pySpatial 的核心优势在于零样本——无需任何梯度训练：

- 代码代理仅需接口文档和少量查询-代码示例（in-context learning）即可工作
- 代理不接触模型权重、文件 I/O 或渲染后端等内部实现
- 使用结构化输出：先自然语言推理，再合成 Python 代码
- 生成的程序本身即为可解释的推理过程——可直接检查、调试或修改

### 关键设计三：即插即用的框架设计

- 适用于开源和闭源 MLLM（GPT-4o、GPT-4.1-mini、Claude 等均可作为代码代理或最终推理器）
- 3D 重建模块可替换（CUT3R / VGGT / DUSt3R 等）
- 所有实验在单张 NVIDIA A6000 Ada GPU 上完成

## 实验结果

### 主实验：MindCube 全集（21K+ 问题）

| 方法 | 类型 | 整体 | Rotation | Among | Around |
|------|------|:---:|:---:|:---:|:---:|
| Random (chance) | - | 32.35 | 36.36 | 32.29 | 30.66 |
| LLaVA-OneVision-7B | 开源 MLLM | 47.43 | 36.45 | 48.42 | 44.09 |
| DeepSeek-VL2-Small | 开源 MLLM | 47.62 | 37.00 | 50.38 | 26.91 |
| GPT-4o | 商用 MLLM | 38.81 | 32.65 | 40.17 | 29.16 |
| GPT-4.1-mini | 商用 MLLM | 45.62 | 37.84 | 47.22 | 34.56 |
| Claude-4-Sonnet | 商用 MLLM | 44.75 | 48.42 | 44.21 | 47.62 |
| VLM-3R | 特化空间模型 | 42.09 | 36.73 | 44.22 | 24.45 |
| **pySpatial (Ours)** | 视觉编程 | **58.56** | **43.20** | **60.54** | **48.10** |

pySpatial 以 58.56% 的整体准确率全面碾压所有基线：比最强开源 MLLM（DeepSeek-VL2-Small）高 10.94%，比 GPT-4.1-mini 高 12.94%，比微调过的 VLM-3R 高 16.47%。在最具挑战性的 Among 类别（推理中心物体与所有周围物体的关系）上达到 60.54%，其他方法均未超过 50%。

### MindCube-1k 子集对比

| 方法 | 整体 | Rotation | Among | Around |
|------|:---:|:---:|:---:|:---:|
| GPT-4o | 42.29 | 35.00 | 43.00 | 46.40 |
| Chain-of-Thought | 40.48 | 32.00 | 36.00 | 58.00 |
| Cognitive Map | 41.43 | 37.00 | 41.67 | 44.40 |
| ViperGPT | 36.95 | 20.50 | 41.00 | 40.40 |
| VADAR | 40.76 | 33.50 | 40.67 | 46.80 |
| VADAR + 3D重建 | 35.62 | 31.00 | 36.83 | 36.40 |
| **pySpatial** | **62.35±1.18** | **41.83±2.34** | **64.89±2.60** | **72.67±3.30** |

pySpatial 以约 20% 的优势超越所有心理模型方法和视觉编程基线。值得注意的是，给 VADAR 加上 3D 重建模块后性能反而下降（40.76→35.62），说明不是有了 3D 信息就能推理——需要合理的 API 设计才能有效利用 3D 几何。

### Omni3D-Bench 单视角泛化

| 方法 | numeric(ct) | numeric(other) | y/n | multi-choice | Total |
|------|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 28.1 | 35.5 | 66.7 | 57.2 | 42.9 |
| VADAR | - | - | - | - | 41.5 |
| ViperGPT | - | - | - | - | 27.8 |
| **pySpatial** | - | - | - | - | **45.3** |

即使在单视角设置下，pySpatial 仍然超越 GPT-4o 和所有视觉编程方法，验证了框架跨设置的泛化能力。

## 论文评价

### 优点

1. **范式创新**：将"隐式想象"转化为"显式计算"的思路极为清晰，通过视觉编程桥接 MLLM 与 3D 世界
2. **零样本即 SOTA**：无需任何训练即在多个基准上大幅超越微调过的特化模型，展现了强大的泛化能力
3. **可解释性强**：生成的 Python 程序本身就是推理过程的精确记录，便于调试和审计
4. **实际应用验证**：四足机器人室内导航实验展示了从学术benchmark到真实世界的可行性

### 不足

1. 依赖 GPT-4o 作为代码代理，API 调用成本高且受限于商用模型的可用性
2. 3D 重建质量直接影响下游推理，在纹理贫乏或重复纹理场景下可能失效
3. 渲染新视角基于点云光栅化，遮挡区域会出现空洞，影响 MLLM 的后续推理

### 评分

⭐⭐⭐⭐ — 视觉编程范式在 3D 空间推理中的优雅应用，方法简洁有效，零样本性能令人印象深刻，为 MLLM 的具身智能落地提供了实用路径。

<!-- RELATED:START -->

## 相关论文

- [Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation](../../CVPR2026/3d_vision/context-nav_context-driven_exploration_and_viewpoint-aware_3d_spatial_reasoning_.md)
- [SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](../../CVPR2025/3d_vision/seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)
- [Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding](../../CVPR2026/3d_vision/masking_matters_unlocking_the_spatial_reasoning_capabilities_of_llms_for_3d_scen.md)
- [Learning Multi-View Spatial Reasoning from Cross-View Relations](../../CVPR2026/3d_vision/learning_multi-view_spatial_reasoning_from_cross-view_relations.md)
- [Quantized Visual Geometry Grounded Transformer](quantized_visual_geometry_grounded_transformer.md)

<!-- RELATED:END -->
