---
title: >-
  [论文解读] Human Motion Instruction Tuning
description: >-
  [CVPR 2025][人体理解][人体运动理解] LLaMo 提出了一种保留运动原始表示（而非转化为语言 token）的多模态指令微调框架，通过同时处理视频、运动序列和文本输入来增强模型对复杂人体行为的理解和预测能力。 领域现状：当前的多模态大语言模型（MLLM）在处理图像、文本等模态时取得了显著进展。在人体运动理解领域…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "人体运动理解"
  - "指令微调"
  - "多模态框架"
  - "运动序列"
  - "视频分析"
---

# Human Motion Instruction Tuning

**会议**: CVPR 2025  
**arXiv**: [2411.16805](https://arxiv.org/abs/2411.16805)  
**代码**: [https://github.com/ILGLJ/LLaMo](https://github.com/ILGLJ/LLaMo)  
**领域**: 人体理解 / 多模态VLM  
**关键词**: 人体运动理解, 指令微调, 多模态框架, 运动序列, 视频分析

## 一句话总结
LLaMo 提出了一种保留运动原始表示（而非转化为语言 token）的多模态指令微调框架，通过同时处理视频、运动序列和文本输入来增强模型对复杂人体行为的理解和预测能力。

## 研究背景与动机

**领域现状**：当前的多模态大语言模型（MLLM）在处理图像、文本等模态时取得了显著进展。在人体运动理解领域，研究者开始探索将运动序列（如骨骼关节点序列）融入大语言模型中，以实现运动描述生成、动作识别和行为预测等任务。

**现有痛点**：传统的指令微调方法在处理非语言输入（如视频或运动序列）时，通常将其转换为语言 token。这种 tokenization 过程会丢失运动特有的细节信息，例如精细的关节运动轨迹、时序连续性和空间协调性。这导致模型在理解复杂人体行为时精度不足。

**核心矛盾**：语言表示的离散性与运动数据的连续性之间存在根本性冲突。将连续的运动信号强行映射到离散 token 空间会造成不可逆的信息损失，尤其在涉及专业活动分析（如体育动作、医疗康复）时更为明显。

**本文目标**：设计一种能够在原始形态下处理运动数据的多模态框架，避免 tokenization 导致的信息损失，同时支持灵活的多模态指令交互。

**切入角度**：作者观察到运动数据本身蕴含丰富的时空结构信息，如果能在保留原始表示的前提下与语言模态对齐，就能让 LLM 更好地"理解"运动语义。

**核心 idea**：用原始运动表示替代 token 化运动表示进行指令微调，构建同时支持视频、运动和文本三模态的人体运动理解助手 LLaMo。

## 方法详解

### 整体框架
LLaMo（Large Language and Human Motion Assistant）是一个多模态框架。输入端接受三种模态：视频帧序列、人体运动序列（骨骼关节点数据）和文本指令。框架通过专门的编码器分别提取视频和运动特征，然后通过对齐模块将这些特征映射到 LLM 的输入空间，最终由大语言模型进行统一的推理和文本生成。

### 关键设计

1. **原始运动表示保留（Native Motion Representation）**:

    - 功能：避免将运动序列转换为离散语言 token，直接以连续向量形式输入模型
    - 核心思路：使用专门的运动编码器提取运动序列的时空特征表示，保留关节运动轨迹的连续性和空间关系。运动编码器基于 Transformer 架构，对输入的骨骼关节序列 $\mathbf{M} \in \mathbb{R}^{T \times J \times 3}$（T 为帧数，J 为关节数）进行时序和空间维度的特征聚合
    - 设计动机：传统方法将运动信号离散化为 token 会导致精细运动信息丢失。保留原始表示能更完整地保留时空结构，使 LLM 能够基于更丰富的운동语义进行推理

2. **多模态对齐模块（Multimodal Alignment Module）**:

    - 功能：将不同模态的特征表示对齐到 LLM 的统一语义空间
    - 核心思路：通过可学习的投影层（projection layers）将视频特征和运动特征分别映射到与 LLM word embedding 同维度的空间。采用两阶段训练策略：第一阶段冻结 LLM 只训练投影层完成模态对齐；第二阶段对整个模型进行端到端微调
    - 设计动机：视频和运动数据处于不同的特征空间，需要统一对齐才能让 LLM 在同一语义框架下同时理解视觉外观和运动结构信息

3. **视频-运动协同分析（Video-Motion Co-Analysis）**:

    - 功能：同时处理视频和运动数据，捕获互补信息
    - 核心思路：视频提供外观、场景和上下文信息，运动序列提供精确的身体姿态和动力学信息。模型通过 cross-attention 机制让两种模态的特征在 LLM 内部交互融合，使推理过程能够综合利用外观线索和运动线索
    - 设计动机：单一模态各有局限——视频在遮挡和光照不良时信息缺失，运动数据缺少场景上下文。协同分析能弥补各自短板，提升对复杂行为的理解能力

### 损失函数 / 训练策略
采用两阶段训练：预训练阶段使用大规模运动-文本配对数据训练对齐模块，以自回归语言建模损失为主；指令微调阶段在高质量人体行为分析数据上进行端到端训练，使用标准的 next-token prediction loss。训练数据涵盖体育分析、日常行为识别等多个领域。

## 实验关键数据

### 主实验

| 任务/数据集 | 指标 | LLaMo | 之前SOTA | 提升 |
|------------|------|-------|----------|------|
| 运动描述生成 (HumanML3D) | BLEU-4 | 15.8 | 13.2 | +2.6 |
| 运动描述生成 (HumanML3D) | CIDEr | 42.3 | 37.1 | +5.2 |
| 行为识别 (NTU-RGBD) | Top-1 Acc | 89.7% | 86.3% | +3.4% |
| 专业活动分析 | F1 Score | 83.5 | 79.2 | +4.3 |

### 消融实验

| 配置 | BLEU-4 | CIDEr | 说明 |
|------|--------|-------|------|
| Full LLaMo | 15.8 | 42.3 | 完整模型（原始运动表示） |
| w/ Motion Tokenization | 13.5 | 36.8 | 使用 token 化运动表示，降幅显著 |
| w/o Video Input | 14.1 | 39.0 | 去掉视频输入，仅用运动数据 |
| w/o Motion Input | 12.7 | 34.5 | 去掉运动输入，仅用视频 |
| w/o Two-stage Training | 14.3 | 38.7 | 跳过预训练直接微调 |

### 关键发现
- 原始运动表示 vs token 化表示的差距非常显著（CIDEr 提升 5.5 点），验证了保留运动原始形态的核心价值
- 运动输入的贡献大于视频输入（去掉运动降 7.8 vs 去掉视频降 3.3），说明在运动理解任务中骨骼数据更为关键
- 两阶段训练策略带来约 3.6 点 CIDEr 提升，表明模态对齐预训练是必要的
- 在体育分析和专业活动等高复杂度场景中，LLaMo 的优势更加明显

## 亮点与洞察
- **原始表示保留**的设计理念具有普适性——不仅适用于运动序列，也启发了其他连续信号（如音频、传感器数据）与 LLM 对接的思路。关键价值在于避免了离散化的信息瓶颈
- **视频-运动协同**思路非常务实。视频和运动骨骼数据在实际应用中经常可以同时获取（如动作捕捉系统），充分利用两种互补信息源是合理选择
- 多模态指令微调的范式为人体中心 AI 系统提供了灵活可扩展的基础架构

## 局限与展望
- LLaMo 的 GitHub 仓库目前实质内容较少（仅 README），开源完整度不足，难以复现
- 运动数据的获取依赖骨骼提取或动作捕捉设备，在 wild 场景下的适用性有限
- 对细粒度运动差异（如同一动作的不同风格）的区分能力未充分验证
- 可探索将该框架扩展到运动生成任务（text-to-motion），形成双向理解-生成能力

## 相关工作与启发
- **vs MotionGPT**: MotionGPT 将运动转为离散 token 再输入 LLM，LLaMo 则保留原始表示。LLaMo 在保留运动细节方面更好，但 MotionGPT 的 token 方案在运动生成上更灵活
- **vs Video-LLaVA**: Video-LLaVA 专注于视频理解，LLaMo 额外引入运动骨骼模态，在人体行为分析上更专精
- 该工作为将 LLM 应用于运动科学、体育分析和康复医学等人体中心领域提供了有价值的技术路线

## 评分
- 新颖性: ⭐⭐⭐⭐ 保留原始运动表示的思路有启发性，但整体架构是 LLaVA 系框架的自然延伸
- 实验充分度: ⭐⭐⭐ 涵盖多个任务但缺乏与更多近期 baseline 的详细对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 为人体运动理解的多模态方向开辟了实用路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Facial Affective Behavior Analysis with Instruction Tuning](../../ECCV2024/human_understanding/facial_affective_behavior_analysis_with_instruction_tuning.md)
- [\[CVPR 2025\] SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [\[CVPR 2025\] SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance](semgeomo_dynamic_contextual_human_motion_generation_with_semantic_and_geometric_.md)
- [\[CVPR 2025\] Stochastic Human Motion Prediction with Memory of Action Transition and Action Characteristic](stochastic_human_motion_prediction_with_memory_of_action_transition_and_action_c.md)

</div>

<!-- RELATED:END -->
