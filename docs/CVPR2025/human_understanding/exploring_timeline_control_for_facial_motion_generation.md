---
title: >-
  [论文解读] Exploring Timeline Control for Facial Motion Generation
description: >-
  [CVPR 2025][人体理解][面部动作生成] 本文首次提出面部动作生成的时间线控制方式——用户指定多轨道时间轴上各面部动作的精确帧区间，通过TICC时序聚类实现省力的帧级面部动作标注，并设计base-branch扩散模型在解耦各面部区域的同时保留自然耦合，生成精确对齐时间线且自然流畅的面部动作。
tags:
  - "CVPR 2025"
  - "人体理解"
  - "面部动作生成"
  - "时间线控制"
  - "面部动作标注"
  - "扩散模型"
  - "TICC时序聚类"
---

# Exploring Timeline Control for Facial Motion Generation

**会议**: CVPR 2025  
**arXiv**: [2505.20861](https://arxiv.org/abs/2505.20861)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: 面部动作生成, 时间线控制, 面部动作标注, 扩散模型, TICC时序聚类

## 一句话总结

本文首次提出面部动作生成的时间线控制方式——用户指定多轨道时间轴上各面部动作的精确帧区间，通过TICC时序聚类实现省力的帧级面部动作标注，并设计base-branch扩散模型在解耦各面部区域的同时保留自然耦合，生成精确对齐时间线且自然流畅的面部动作。

## 研究背景与动机

**领域现状**：生成逼真的面部动作在数字人和影视制作中有广泛需求。现有方法主要使用音频或文本作为控制信号——音频驱动方法只能生成与音频同步的动作，文本驱动方法只能通过时间副词（如"然后"）做粗粒度时序描述。

**现有痛点**：用户经常需要的精细控制——如"在第10-30帧抬眉，同时在第14-43帧微笑"——是现有控制方式无法实现的。音频信号绑定了节奏，文本信号缺乏帧级精度。规则方法（如直接调blendshape曲线）虽能精确控制时间，但生成的动作不自然，偏离真实运动分布。

**核心矛盾**：实现帧级时间控制需要帧级的面部动作标注数据，但这样的标注代价极高。现有方法用ChatGPT总结时间序列或用阈值判断blendshape值，但前者无法确定精确起止帧，后者对复杂动作（如眉毛运动）难以设定可靠阈值，且忽略了多个运动描述符之间的关系。

**本文目标** (1) 如何省力地获取帧级面部动作区间标注？(2) 如何设计生成模型使其既能精确对齐时间线、又能保持动作的自然性？(3) 如何处理面部不同区域动作之间的耦合与解耦平衡？

**切入角度**：利用TICC（基于Toeplitz逆协方差的时序聚类）自动将连续面部运动时间序列分割为离散的动作区间，并将相似动作模式聚类。人工只需检查每个聚类的少量样本即可确定动作类型，大幅减少标注工作量。

**核心 idea**：通过时序聚类自动标注面部动作区间，然后用base-branch扩散模型从时间线生成自然且时间精确的面部运动。

## 方法详解

### 整体框架

系统包含两个核心部分：(1) 面部动作帧级标注流程：提取ARKit blendshape时间序列→TICC分割聚类→人工审查聚类标签→得到帧级标注；(2) 时间线驱动生成模型：base网络编码全局运动耦合→branch网络为各面部区域独立生成去耦动作→组合得到完整面部运动→扩散渲染器进行真实感渲染。支持ChatGPT将自然语言转换为时间线实现文本控制。

### 关键设计

1. **基于TICC的帧级面部动作标注**:

    - 功能：以极低人力成本实现帧级面部动作区间标注
    - 核心思路：提取每段视频的ARKit blendshape系数作为面部运动描述符（眉毛用browDown/browInnerUp/browOuterUp，眼睛用eyeBlink/eyeSquint/eyeWide，嘴巴用mouthSmile/mouthStretch/mouthFrown）。将多段视频的时间序列用"null序列"（值为-1的长度100序列）分隔后拼接成单一长序列，输入TICC算法。TICC同时完成两件事：(a) 将时间序列分割为若干动作模式区间，每个区间有明确的起止帧；(b) 将相似模式的区间聚类到一起。人工只需检查每个聚类的几个代表性样本，即可确定该聚类代表的动作类别。
    - 设计动机：避免了逐帧手动标注的天量工作。相比阈值法，TICC自动考虑多个描述符之间的关系，不需要为复杂动作手动设定阈值。实验显示眉毛/眼睛/嘴巴的标注macro-F1分别达到0.90/0.91/0.87。

2. **Base-Branch扩散生成模型**:

    - 功能：在精确对齐时间线的同时保持面部动作的自然耦合
    - 核心思路：生成模型分为base网络和多个branch网络。Base网络接收所有面部区域的时间线和含噪运动，通过Transformer编码器将全局运动耦合编码为base特征。Branch网络分为上半脸（眼+眉+注视）、下半脸（嘴+下巴）、姿态&其他三个分支，每个分支仅接收对应区域的时间线和base特征（姿态分支例外，接收所有区域时间线因为头部姿态与所有面部运动耦合）。时间线通过cross-attention引导运动生成，每一层都重新引入初始时间线token以防止时序信息被改变。
    - 设计动机：面部不同区域的运动是耦合的（如微笑时眯眼和微微低眉），这种耦合对自然性至关重要。但完全耦合生成会降低精度（如生成微笑时被迫同时生成低眉动作，与用户指定的抬眉冲突）。Base-Branch设计让base网络学习全局耦合，branch网络实现区域解耦，在精度和自然性之间取得平衡。

3. **时间线Token常驻注入 + Classifier-Free Guidance**:

    - 功能：增强运动与时间线的精确对齐并提升泛化能力
    - 核心思路：在base/branch网络的每一层Transformer编码器中，时间线token始终使用初始的timeline token而非上一层的输出token。训练时使用classifier-free guidance，以0.5的概率独立drop每个面部区域的条件，另有0.1的概率drop全部条件和0.1的概率保留全部条件。条件被drop时，该区域时间线值设为-1。
    - 设计动机：如果允许时间线token在各层间迭代更新，时序精度信息会被逐层稀释。常驻注入确保每一层都能获得原始的精确时序信息。Classifier-free guidance增强模型对部分条件缺失情况的鲁棒性和泛化能力，最优drop概率为0.5。

### 损失函数 / 训练策略

使用标准扩散去噪损失 $\mathcal{L}_{denoise} = \mathbb{E}_{t,M_{(0)},C}[\|M_{(0)} - \mathcal{G}(M_{(t)}, t, C)\|^2]$，直接预测原始信号而非噪声。使用FaceVerse 3DMM系数作为运动表示。数据集为RealTalk（692段真实对话视频，约60万帧）。网络包含8层Transformer，其中base网络和branch网络共享结构但参数独立，最优配置为base 6层+branch 2层。

## 实验关键数据

### 主实验

| 方法 | Var→ | FID$_{fm}$↓ | FID$_{\Delta fm}$↓ | SND↓ | TAS↑ |
|------|------|------------|----------------|------|------|
| w/o branch | 0.68 | 7.39 | 0.14 | 7.53 | 0.66 |
| w/o base | 0.64 | 12.4 | 0.18 | 12.58 | 0.81 |
| all decoup. | 0.41 | 28.4 | 0.23 | 28.63 | 0.69 |
| **Ours** | **0.70** | **4.54** | **0.09** | **4.63** | **0.84** |

TAS(Timeline Alignment Score)评估时间线对齐精度，SND评估运动自然性。GT的Var为0.73。

### 消融实验

| 配置 | TAS↑ | SND↓ | 说明 |
|------|------|------|------|
| w/o time con. (去掉常驻注入) | 0.79 | 5.48 | 时序精度下降 |
| branchL1 (branch仅1层) | 0.76 | 6.36 | branch太浅解耦不足 |
| branchL2 (Ours) | **0.84** | **4.63** | 最优平衡 |
| branchL4 (branch 4层) | 0.83 | 6.76 | base太浅耦合不足 |
| drop 0 (无CFG) | 0.78 | 7.01 | 泛化差 |
| drop 0.5 (Ours) | **0.84** | **4.63** | 最优 |
| drop 0.7 | 0.68 | 4.21 | 条件信号太弱精度低 |

### 关键发现

- TICC标注质量很高：眉毛macro-F1=0.90，眼睛0.91，嘴巴0.87。用AU替代blendshape后F1降至0.73，说明描述符精度至关重要。
- Base-Branch设计缺一不可：去掉branch（仅base）TAS降至0.66（精度差），去掉base（仅branch）SND升至12.58（不自然）。完全解耦所有区域到独立branch会导致最差的FID(28.4)和不自然的动作。
- 时间线常驻注入将TAS从0.79提升到0.84。
- 用户研究：89%的评价认为生成动作准确对齐时间线，86%认为动作自然。
- 支持ChatGPT文本→时间线转换，实现自然语言控制面部动作生成。

## 亮点与洞察

- **时间线控制是面部动作生成的新范式**：相比音频和文本控制，时间线提供了帧级精度的时序控制能力。这种控制粒度在影视制作和动画中具有极高实用价值。
- **TICC用于面部动作标注非常巧妙**：利用时序聚类算法同时完成分割和聚类两个任务，将帧级标注问题转化为"检查K个聚类的代表性样本"的轻量任务，大幅降低标注成本。
- **Base-Branch设计精准平衡了耦合与解耦**：这个架构思想可以迁移到其他需要部分解耦的多条件生成任务，如人体动作生成中四肢的独立控制与全身协调。

## 局限与展望

- 标注只覆盖了有限的动作类别（眉毛3类、眼睛4类、嘴巴4类等），无法描述更细粒度的面部微动作。
- TICC的聚类数和beta参数需要手动调优，不同面部区域的最优参数不同。
- 训练数据来自说话场景，生成结果偶尔会出现不受控的说话动作。
- 渲染器（扩散渲染）的质量和身份保持能力限制了最终视频的实用性。
- 目前仅处理对称面部动作（只用左侧blendshape系数），扩展到非对称表情（如挤眉弄眼）是一个方向。

## 相关工作与启发

- **vs AgentAvatar/InstructAvatar**: 这些文本驱动方法只能用时间副词粗略描述动作时序，无法实现帧级控制。时间线控制从根本上解决了控制粒度的问题。
- **vs 人体动作时间线控制 (TEACH等)**: 人体动作时间线控制通过先生成片段再拼接实现，但面部动作变化快速频繁，拼接策略会产生不自然的过渡。本文的base-branch扩散模型直接在完整时间线上一次性生成，更适合面部场景。
- **vs 规则方法**: 规则方法可以精确控制时间但动作不自然。本文通过学习真实数据分布，在保持时间精度的同时确保动作自然性。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 开创性地提出面部动作时间线控制，TICC标注方案巧妙
- 实验充分度: ⭐⭐⭐⭐ 消融详尽，用户研究完善，但缺乏与其他方法的定量对比
- 写作质量: ⭐⭐⭐⭐ 整体清晰，但标注和生成两部分的衔接可以更紧凑
- 价值: ⭐⭐⭐⭐ 面部动作精细控制的需求真实且重要，方法有实际应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ControlFace: Harnessing Facial Parametric Control for Face Rigging](controlface_harnessing_facial_parametric_control_for_face_rigging.md)
- [\[CVPR 2025\] PersonaBooth: Personalized Text-to-Motion Generation](personabooth_personalized_text-to-motion_generation.md)
- [\[AAAI 2026\] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](../../AAAI2026/human_understanding/soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)
- [\[CVPR 2025\] MotionReFit: Dynamic Motion Blending for Versatile Motion Editing](motionrefit_motion_editing.md)
- [\[CVPR 2025\] SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance](semgeomo_dynamic_contextual_human_motion_generation_with_semantic_and_geometric_.md)

</div>

<!-- RELATED:END -->
