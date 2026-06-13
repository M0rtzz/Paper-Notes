---
title: >-
  [论文解读] MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent
description: >-
  [ICCV 2025][视频生成][图像到视频生成] 提出 MotionAgent，通过运动场代理（Motion Field Agent）将文本中的运动描述转化为物体轨迹和相机外参，再经解析式光流合成模块统一为光流图，实现仅凭文本输入即可对 I2V 生成中的物体运动和相机运动进行细粒度精确控制。 现有 I2V（Image-t…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "图像到视频生成"
  - "运动场代理"
  - "光流"
  - "细粒度运动控制"
  - "扩散模型"
---

# MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent

**会议**: ICCV 2025  
**arXiv**: [2502.03207](https://arxiv.org/abs/2502.03207)  
**代码**: [GitHub](https://github.com/leoisufa/MotionAgent)  
**领域**: 视频生成  
**关键词**: 图像到视频生成, 运动场代理, 光流, 细粒度运动控制, 扩散模型

## 一句话总结

提出 MotionAgent，通过运动场代理（Motion Field Agent）将文本中的运动描述转化为物体轨迹和相机外参，再经解析式光流合成模块统一为光流图，实现仅凭文本输入即可对 I2V 生成中的物体运动和相机运动进行细粒度精确控制。

## 研究背景与动机

现有 I2V（Image-to-Video）生成模型在视觉质量上已取得显著进展，但仅通过文本输入实现精确运动控制仍是一个开放问题。当前方法存在三类局限：

**纯文本编码器控制**（如 DynamiCrafter、CogVideoX）：通过文本编码器将文本嵌入注入视觉特征，只能实现整体粗粒度控制，难以做到对视频中每个元素的细粒度运动控制。此外，高质量文本-运动对齐的训练数据极为稀缺。

**专用运动控制模块**（如 DragNUWA、CameraCtrl）：设计特定模块分别控制物体运动或相机运动，但通常只能控制单一运动类型，难以同时兼顾；且输入控制条件（如点轨迹、相机外参矩阵）需要专业知识，对普通用户不友好。

**中间表示方法**（如 Motion-I2V、MOFA-Video）：通过光流等中间表示控制多种运动类型，但仍需用户手动提供轨迹或相机参数，使用门槛高。

MotionAgent 的核心动机：**能否让用户只需提供自然语言文本描述，就能自动、精确地同时控制视频中的物体运动和相机运动？** 将 LLM Agent 的推理能力与几何化的光流合成相结合，实现"文本→运动场→可控视频"的端到端流程。

## 方法详解

### 整体框架

MotionAgent 由两大模块组成：

1. **运动场代理（Motion Field Agent）**：将文本中的运动信息解析并转化为物体轨迹和相机外参两种显式中间表示。
2. **可控 I2V 生成模型**：包含解析式光流合成模块和光流适配器，将中间表示转化为统一光流以控制基础 I2V 扩散模型（SVD）。

### 运动场代理的关键设计

#### Step 1: 视频运动分解
Agent 分析文本中的运动信息，将其拆分为**物体运动描述**和**相机运动描述**两部分，实现运动类型的解耦与独立控制。

#### Step 2: 物体轨迹绘制
分为两个子任务：

- **物体识别**：Agent 从文本中提取动态物体描述，调用 Grounded-SAM 进行开放世界目标检测与分割，将检测结果以半透明掩膜的形式叠加在输入图像上，反馈给 Agent 以确认目标物体。
- **轨迹绘制**：Agent 根据检测结果确定轨迹起点，采用**网格选择策略**（将图像划分为 $N \times M$ 网格，每个格标记整数编号），Agent 通过选择网格编号确定轨迹各点，依次连接形成完整轨迹。这种方式比直接生成坐标更稳定可靠。

#### Step 3: 相机外参生成
Agent 根据相机运动文本描述和输入图像直接生成相机外参 $E$。文本指定相机路径，图像帮助 Agent 判断运动幅度（例如广阔风景需要大幅运动，特写镜头只需微调）。平移量 $T$ 约束在 $(-1, 1)$，后续根据深度图重新缩放。

#### Step 4: 反思机制（Rethinking，可选）
Agent 分析已生成的视频，回顾此前每一步的决策，形成反馈闭环，根据文本与视频的不一致之处修正物体轨迹和相机外参，进一步提升生成质量。

### 解析式光流合成模块

该模块的核心是将物体运动和相机运动在 3D 空间中进行几何合成：

1. **深度估计与3D提升**：使用 Metric3D 估计输入图像深度图 $D$，将每个像素 $I^0$ 反投影到 3D 空间获得初始3D位置 $P^0$。
2. **物体运动光流**：用 CMP 从物体轨迹估计稠密光流 $F_{obj}$，结合深度图计算物体运动引起的 3D 位置偏移 $O$，得到移动后的 3D 位置 $P^1 = P^0 + O$。
3. **相机运动重投影**：根据相机外参 $E$ 将 $P^1$ 重投影到对应图像坐标：$I^1 = \Pi(E P^1)$。
4. **统一光流**：计算像素偏移作为统一光流：$F = I^1 - I^0$，该光流同时包含物体运动和相机运动信息。

### 损失函数与训练

- 基础 I2V 模型采用冻结的 SVD（Stable Video Diffusion）。
- 光流适配器采用 MOFA-Video 提出的架构，但需要**微调**以适配统一光流（因为统一光流与真实光流存在域差异）。
- 微调数据制备：用 Unimatch 估计真实光流，用 DROID-SLAM 计算相机外参，去除相机运动引起的光流后获得纯物体运动光流，进行稀疏采样得到物体轨迹，再用解析合成方法重新计算统一光流作为训练输入。
- 训练配置：32 张 A800 GPU，AdamW 优化器，学习率 $2 \times 10^{-5}$，分辨率 $512 \times 512$，随机采样 24 帧（步长 4）。

## 实验关键数据

### 主实验：通用 I2V 生成（VBench）

| 方法 | I2V Score | Video-Text Camera Motion | Subject Consistency | Motion Smoothness | Dynamic Degree |
|------|-----------|-------------------------|--------------------|--------------------|----------------|
| VideoCrafter | 88.95 | 33.60 | 97.86 | 98.00 | 22.60 |
| DynamiCrafter | 97.98 | 35.81 | 95.69 | 97.38 | 47.40 |
| SVD (基线) | 96.93 | – | 95.42 | 98.12 | 43.17 |
| **MotionAgent** | **97.51** | **81.91** | **96.10** | **98.93** | 16.67 |

**关键发现**：MotionAgent 在 Video-Text Camera Motion 指标上达到 **81.91%**，远超其他方法（第二名 DynamiCrafter 仅 35.81%），证明了对相机运动的精确控制能力。

### 主实验：可控 I2V 生成（自建 Benchmark）

| 方法 | Object Movement Q&A | Complex Camera Motion | Total Scores |
|------|--------------------|-----------------------|--------------|
| DynamiCrafter | 29.38 | 8.22 | 16.58 |
| CogVideoX | 26.47 | 20.62 | 22.93 |
| Pyramid Flow | 30.96 | 6.18 | 15.97 |
| **MotionAgent** | **45.69** | **77.76** | **65.10** |
| **MotionAgent (Rethinking)** | **49.58** | **89.04** | **73.45** |

**关键发现**：MotionAgent 总分 65.10%，大幅领先第二名 CogVideoX（22.93%）；加入 Rethinking 后进一步提升至 73.45%。

### 消融实验

| 变体 | Object Movement Q&A | Complex Camera Motion | Dynamic Degree |
|------|--------------------|-----------------------|----------------|
| 无检测工具（多轮对话识别） | 34.33 | 75.11 | 20.53 |
| 无物体运动 | 10.51 | 75.95 | 8.42 |
| 无相机运动 | 38.20 | 0.30 | 29.95 |
| 无光流合成（直接相加） | 30.07 | 64.92 | 27.89 |
| 无适配器微调 | 40.56 | 48.27 | 28.47 |
| **完整模型** | **45.69** | **77.76** | **32.11** |

**消融关键发现**：
- Grounded-SAM 辅助检测比纯多轮对话物体识别效果更好（+11.36 Q&A）。
- 解析式光流合成比直接叠加光流在两项指标上分别高 15.62 和 12.84 个百分点。
- 光流适配器微调对复杂相机运动提升显著（+29.49）。
- Rethinking 机制对复杂相机运动提升明显（+11.28）。

### 不同 LLM 和基础模型的鲁棒性

| Agent LLM | 基础模型 | Total Scores |
|-----------|---------|--------------|
| GPT-4o | SVD | 65.10 |
| Qwen2 | SVD | 61.73 |
| Llama3 | SVD | 63.80 |
| GPT-4o | Motion-I2V | 61.00 |

替换不同 LLM 后性能差异不大，证明方法对 Agent 后端模型具有良好鲁棒性。

## 亮点与洞察

1. **Agent + 几何的巧妙融合**：不是直接用 LLM 端到端生成视频控制信号，而是让 Agent 生成结构化中间表示（轨迹 + 外参），再通过几何方法合成光流，结合了 LLM 的语义理解能力和几何计算的精确性。
2. **网格选择替代坐标生成**：将轨迹绘制转化为网格编号选择问题，降低了 LLM 输出连续坐标的难度，提高了轨迹生成的鲁棒性。
3. **反思机制**：引入"生成→评估→修正"的闭环反馈，类似 Agent 的 self-reflection，对复杂相机运动的提升尤为显著（+11.28）。
4. **解耦设计**：物体运动和相机运动的解耦使得错误不会相互传播（如物体识别错误不影响相机运动控制）。
5. **自建评估基准**：针对现有评测缺乏运动语义对齐评估的问题，构建了专门的可控 I2V 评测集，包含 432 个物体运动提示和 662 个复杂相机运动提示。

## 局限性

1. **Dynamic Degree 下降**：精确运动控制导致未被文本提及的物体保持静止，使得 VBench 的动态程度指标下降（16.67 vs SVD 的 43.17），在需要丰富自然运动的场景中可能显得不够生动。
2. **依赖外部工具链较多**：流水线中涉及 Grounded-SAM、Metric3D、CMP、Unimatch、DROID-SLAM 等多个外部模型，系统复杂度高，任一环节的误差都可能累积传播。
3. **推理成本较高**：需要调用 LLM Agent 进行多步推理 + 可选的 Rethinking 迭代 + 多个外部模型推理，整体延迟和计算成本较高端到端方法更大。
4. **光流表示的固有局限**：光流作为 2D 运动表示，在处理遮挡、新内容出现、大幅度运动等场景时存在天然局限。
5. **评估指标自建**：Object Movement Q&A 依赖 GPT-4o 打分，可能存在评分偏差；且自建 benchmark 的覆盖面和通用性有待验证。

## 相关工作与启发

- **MOFA-Video**：提供了光流适配器的基础架构，MotionAgent 在此基础上引入微调策略以适配统一光流。
- **Motion-I2V**：使用光流扩散模型生成中间表示，但精度不如 MotionAgent 的解析式合成方法。
- **ChatCam**：通过对话导航相机运动，但只控制相机，不涉及物体运动。
- **ObjCtrl-2.5D**：将 2D 轨迹扩展到 2.5D，但需要手动输入轨迹和深度。
- **启发方向**：Agent 框架可推广至更多生成任务（3D 场景生成、长视频生成）；Rethinking 机制可与 RLHF 结合进一步优化；光流表示可升级为 3D scene flow 以处理更复杂的运动。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | ⭐⭐⭐⭐ | 首次将 LLM Agent 与几何光流合成结合实现纯文本可控 I2V |
| 技术深度 | ⭐⭐⭐⭐ | 解析式光流合成模块设计精巧，3D 空间合成思路清晰 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | VBench 评测 + 自建 benchmark + 消融 + 用户研究 + 鲁棒性验证，非常全面 |
| 实用价值 | ⭐⭐⭐⭐ | 降低了可控视频生成的使用门槛，但系统复杂度是落地障碍 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，流水线图示详尽 |
| 总评 | ⭐⭐⭐⭐ | 解决了重要问题，方法新颖且实验扎实，是 I2V 可控生成方向的一篇优质工作 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](../../AAAI2026/video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[ICCV 2025\] ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering](etva_evaluation_of_text-to-video_alignment_via_fine-grained_question_generation_.md)
- [\[NeurIPS 2025\] DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](../../NeurIPS2025/video_generation/densedpo_finegrained_temporal_preference_optimization_for_vi.md)
- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](../../CVPR2025/video_generation/motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[ICCV 2025\] DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation](dh-facevid-1k_a_large-scale_high-quality_dataset_for_face_video_generation.md)

</div>

<!-- RELATED:END -->
