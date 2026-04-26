---
title: >-
  [论文解读] Multi-Object Sketch Animation by Scene Decomposition and Motion Planning
description: >-
  [ICCV 2025][模型压缩][草图动画] MoSketch 首次解决多物体草图动画问题，通过 LLM 场景分解 + LLM 运动规划 + 运动精炼网络 + 组合式 SDS 四个模块，以分治策略处理物体感知运动建模和复杂运动优化两大挑战，无需任何训练数据实现高质量多物体草图动画。
tags:
  - ICCV 2025
  - 模型压缩
  - 草图动画
  - 多物体动画
  - LLM运动规划
  - Score Distillation Sampling
  - 组合生成
---

# Multi-Object Sketch Animation by Scene Decomposition and Motion Planning

**会议**: ICCV 2025  
**arXiv**: [2503.19351](https://arxiv.org/abs/2503.19351)  
**代码**: 无（计划开源）  
**领域**: 扩散模型  
**关键词**: 草图动画、多物体动画、LLM运动规划、Score Distillation Sampling、组合生成

## 一句话总结

MoSketch 首次解决多物体草图动画问题，通过 LLM 场景分解 + LLM 运动规划 + 运动精炼网络 + 组合式 SDS 四个模块，以分治策略处理物体感知运动建模和复杂运动优化两大挑战，无需任何训练数据实现高质量多物体草图动画。

## 研究背景与动机

**领域现状**：草图动画（Sketch Animation）将静态草图转化为动态视频，广泛应用于 GIF 设计、卡通制作和日常娱乐。近期方法 Live-Sketch（CVPR 2024）使用向量草图表示和 Score Distillation Sampling（SDS）进行无数据训练的动画生成，FlipSketch（CVPR 2025）通过 DDIM 反转和微调 T2V 模型生成光栅化草图动画。两者在单物体草图动画上表现优秀。

**现有痛点**：从单物体到多物体草图动画存在本质性困难：(1) Live-Sketch 不具备物体感知的运动建模能力，无法处理物体间的关系和交互（如倒水时水量应减少），且 T2V 扩散模型难以被 SDS 有效引导来生成多物体复杂运动；(2) FlipSketch 的 DDIM 反转无法充分捕获多物体草图的外观，且其微调数据由 Live-Sketch 合成，多物体场景极少且质量差。

**核心矛盾**：多物体草图动画面临两大挑战——**物体感知运动建模**（需要考虑物体间的相对运动、交互和物理约束）和**复杂运动优化**（T2V 扩散模型难以通过 SDS 有效引导多物体的复杂运动）。现有方法都未能同时解决这两个问题。

**本文目标**：提出一种无需训练数据的多物体草图动画方法，同时解决物体感知运动建模和复杂运动优化两大挑战。

**切入角度**：利用 LLM 的先验知识进行场景理解和运动规划（解决运动建模问题），结合组合式 SDS 将复杂运动分解为简单运动逐一优化（解决优化问题）。

**核心 idea**：采用分治策略——LLM 负责"高层规划"（场景分解和粗粒度运动），神经网络负责"底层精炼"（细粒度运动），组合 SDS 负责"分而优化"（将复杂运动拆解为简单运动分别引导）。

## 方法详解

### 整体框架

MoSketch 基于向量草图表示（每个笔画是三次 Bézier 曲线），通过 SDS 迭代优化。输入是向量草图 $P \in \mathbb{R}^{n \times 2}$ 和文本指令 $Y$，输出是所有控制点的移动序列 $\Delta Z \in \mathbb{R}^{n \times f \times 2}$。四个模块依次工作：LLM 分解场景→LLM 规划运动→运动精炼网络生成细粒度运动→组合 SDS 优化全局。

### 关键设计

1. **LLM 场景分解（LLM-based Scene Decomposition）**:

    - 功能：作为整个方法的基础，识别物体、获取位置、将复杂运动分解为简单运动
    - 核心思路：给定草图 $P$ 和文本 $Y$，让 GPT-4 识别出 $m$ 个独立物体和 $r$ 个分解后的简单运动描述 $\{Y_k\}_{k=1}^r$（每个简单运动仅涉及 1-2 个物体）。使用 Grounding DINO 检测各物体的边界框 $B_0 \in \mathbb{R}^{m \times 4}$，然后根据笔画中心点到边界框的距离将每个控制点分配给最近的物体。约束 $m < 7, r < 5$
    - 设计动机：将不可直接建模的多物体复合运动分解为可被 T2V 模型有效处理的简单运动，为后续运动规划和组合优化提供基础结构

2. **LLM 运动规划 + 运动精炼网络**:

    - 功能：LLM 生成粗粒度物体级运动规划，精炼网络生成细粒度运动
    - 核心思路：**运动规划**：GPT-4 根据草图、文本指令和初始位置 $B_0$，在推理步骤后生成 $f$ 帧中所有物体的边界框序列 $B \in \mathbb{R}^{m \times f \times 4}$，转换为粗粒度物体运动 $\Delta Z_c$。GPT-4 被引导充分考虑惯性、重力等物理约束。**精炼网络**：在 Live-Sketch 基础上改进，将草图级运动替换为物体级运动。物体边界框序列 $B$ 和控制点 $P$ 分别经过 MLP 得到隐表示，通过 Transformer 建模物体间关系得到物体嵌入 $\hat{B}$ 和点嵌入 $\hat{P}$。物体嵌入预测各物体的 7 参数仿射变换（平移、缩放、剪切、旋转）得到精细物体运动 $\Delta Z_o$，点嵌入通过物体特定的 MLP 预测各点位移得到点运动 $\Delta Z_p$。最终动画 $\Delta Z = \Delta Z_c + \Delta Z_o + \Delta Z_p$
    - 设计动机：LLM 拥有对物体交互、物理约束的先验知识，适合做高层运动规划（如炮弹发射后应沿抛物线运动），但输出精度有限。精炼网络通过 Transformer 建模物体关系，能对粗规划做细粒度修正（如炮弹到达目标时的爆炸效果）

3. **组合式 SDS（Compositional SDS）**:

    - 功能：确保多物体的复杂运动在 SDS 迭代优化中被有效引导
    - 核心思路：在标准 SDS 损失 $\mathcal{L}_{SDS}$（使用完整文本 $Y$ 和完整动画 $\Delta Z$）基础上，额外为每个分解出的简单运动 $Y_k$ 计算独立的 SDS 损失 $\mathcal{L}_{SDS-k}$。具体做法是从完整动画中提取 $Y_k$ 涉及物体的子视频 $\Delta Z_k$，然后用 T2V 模型对 $(\Delta Z_k, Y_k)$ 计算 SDS。总损失：$\mathcal{L}_{CSDS} = \mathcal{L}_{SDS} + \sum_{k=1}^r \mathcal{L}_{SDS-k}$
    - 设计动机：T2V 扩散模型在处理简单运动（如"球飞向篮筐"）时比处理复杂多物体运动（如"球员投篮并将球投入篮筐同时队友跑向前场"）更可靠。组合 SDS 将复杂问题分解为模型擅长的简单问题，逐一提供有效梯度引导

### 损失函数 / 训练策略

使用 Adam 优化器，学习率 5e-3，权重衰减 1e-2，迭代 500 步，单卡 RTX 3090 Ti 约 1 小时。隐藏维度 128，Transformer 2 层，帧数 16。无需训练数据，完全通过 SDS 从预训练 T2V 模型获取梯度信号。

## 实验关键数据

### 主实验

在 60 个多物体草图测试集上的定量比较：

| 方法 | Text-Video Align↑ | Motion Smooth↑ | Sketch-Video Align↑ | Dynamic Degree↑ |
|------|-------------------|----------------|---------------------|----------------|
| CogVideoX (I2V) | 0.141 | 0.610 | 0.747 | - |
| DynamiCrafter (I2V) | 0.184 | 0.771 | 0.868 | - |
| FlipSketch | 0.199 | 0.704 | 0.839 | - |
| Live-Sketch | 0.207 | 0.897 | 0.956 | 0.266 |
| **MoSketch** | **0.218** | **0.914** | **0.977** | **0.283** |

### 消融实验

| 配置 | Text-Video↑ | Motion Smooth↑ | Sketch-Video↑ | Dynamic↑ | 说明 |
|------|------------|----------------|---------------|----------|------|
| w/o 运动规划 | 0.212 | 0.955 | 0.959 | 0.083 | 外部运动几乎为零 |
| w/o 精细物体运动 | 0.212 | 0.909 | 0.964 | 0.266 | 外部运动不够精细 |
| w/o 点运动 | 0.203 | 0.971 | 0.971 | 0.200 | 完全缺乏内部运动 |
| w/o 物体感知网络 | 0.205 | 0.932 | 0.968 | 0.266 | 运动不够精细 |
| w/o 组合SDS | 0.207 | 0.911 | 0.966 | 0.267 | 缺乏细节 |
| **MoSketch (完整)** | **0.218** | **0.914** | **0.977** | **0.283** | 最优 |

### 关键发现

- **LLM 运动规划至关重要**：去掉运动规划后 Dynamic Degree 从 0.283 骤降至 0.083，说明不靠 LLM 的粗规划几乎无法产生有意义的外部运动（如发射炮弹）
- **三层运动缺一不可**：$\Delta Z_c$（粗运动）、$\Delta Z_o$（精细物体运动）、$\Delta Z_p$（点运动）分别控制大范围位移、外部运动精化和内部形变，消融任一层都会导致特定维度的运动缺失
- **组合 SDS 提升运动细节**：去掉组合 SDS 后 Text-Video Align 从 0.218 降至 0.207，说明分解优化确实帮助 T2V 模型更好地引导复杂运动
- **方法对点分配误差有一定鲁棒性**：即使 Grounding DINO 的物体定位或点分配有小误差，最终结果仍然可视觉compelling
- **FlipSketch 和 I2V 方法在草图域严重失败**：I2V 方法因为草图和自然图像的域差距无法保持草图外观，FlipSketch 的光栅化表示也无法保持多物体草图细节

## 亮点与洞察

- **LLM 作为物理直觉引擎**：利用 GPT-4 对现实世界物体运动的先验知识来规划运动，包含了惯性、重力、碰撞等物理约束的理解。这种"LLM 规划 + 网络执行"的范式可以推广到其他需要物理理解的生成任务
- **分治策略的完整闭环**：场景分解→运动规划→运动精炼→组合优化，四个模块形成了一个从高层到底层的分治闭环，每个层级解决一个明确的子问题
- **无需训练数据的方法设计**：在没有多物体草图动画数据集的情况下，通过 SDS + LLM 先验实现了端到端的动画生成，展示了预训练模型组合使用的强大能力

## 局限与展望

- **点分配质量影响较大**：当 Grounding DINO 的物体检测错误较大时（如哥斯拉的尾巴被错误归属到"城市"），最终动画质量会严重下降
- **LLM 运动规划可能出错**：GPT-4 对某些运动的理解有偏差（如守门员应该朝足球移动），错误的粗规划无法被精炼网络完全修正
- **T2V 模型的运动认知限制**：T2V 模型对某些特殊运动（如"打架"）可能缺乏理解，导致相关动画生成失败
- 优化过程较慢，单个动画约需 1 小时

## 相关工作与启发

- **vs Live-Sketch (CVPR 2024)**: Live-Sketch 是 MoSketch 的直接基础，处理单物体效果好但缺乏物体感知能力。MoSketch 在其之上增加了场景分解、运动规划和组合优化，在多物体场景上全面超越
- **vs FlipSketch (CVPR 2025)**: FlipSketch 使用光栅化表示和微调方法，在多物体场景中外观保持失败。MoSketch 继承向量表示保持了草图外观完整性
- **vs LLM-grounded Video Diffusion**: 在 T2V 领域，LLM-grounded VDM 等工作也利用 LLM 做轨迹规划。MoSketch 将此思路首次引入草图动画，并增加了组合 SDS 来处理 SDS 范式下的多物体优化问题

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次解决多物体草图动画问题，LLM 分治 + 组合SDS 设计新颖
- 实验充分度: ⭐⭐⭐⭐ 定量定性对比全面，消融覆盖所有模块，但测试集规模较小（60个草图）
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详尽，图示丰富直观
- 价值: ⭐⭐⭐⭐ 开辟了多物体草图动画新方向，但应用场景相对小众

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization](../../NeurIPS2025/model_compression/find_your_needle_small_object_image_retrieval_via_multi-object_attention_optimiz.md)
- [\[NeurIPS 2025\] DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method](../../NeurIPS2025/model_compression/deltaflow_an_efficient_multi-frame_scene_flow_estimation_method.md)
- [\[CVPR 2025\] Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch](../../CVPR2025/model_compression/sketch_down_the_flops_towards_efficient_networks_for_human_sketch.md)
- [\[ICCV 2025\] VQ-SGen: A Vector Quantized Stroke Representation for Creative Sketch Generation](vq-sgen_a_vector_quantized_stroke_representation_for_creative_sketch_generation.md)
- [\[ICCV 2025\] MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion](motionfollower_editing_video_motion_via_score-guided_diffusion.md)

<!-- RELATED:END -->
