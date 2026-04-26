---
title: >-
  [论文解读] AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video
description: >-
  [CVPR 2026][人体理解][音视频推理] 提出AVATAR框架，通过离策略训练架构（分层重放缓冲区）和时间优势塑形（TAS，U形加权强调推理链首尾）两个核心组件改进GRPO，解决其数据低效、优势消失和均匀信用分配三大问题，在音视频推理基准上显著超越GRPO基线。
tags:
  - CVPR 2026
  - 人体理解
  - 音视频推理
  - GRPO改进
  - 离策略强化学习
  - 时间优势塑形
  - 多模态大语言模型
---

# AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video

**会议**: CVPR 2026  
**arXiv**: [2508.03100](https://arxiv.org/abs/2508.03100)  
**代码**: https://people-robots.github.io/AVATAR/ (有)  
**领域**: 人类理解 / 多模态推理  
**关键词**: 多模态推理, 强化学习, GRPO, 音视频理解, 时间优势塑造

## 一句话总结
提出AVATAR框架，通过离线策略训练架构（分层回放缓冲区）和时间优势塑造(TAS)策略解决GRPO在多模态视频推理中的数据低效、优势消失和均匀信用分配三大问题，在音视频理解基准上显著超越标准GRPO（OmniBench +3.7，样本效率提升5倍）。

## 研究背景与动机

**领域现状**：多模态大语言模型(MLLM)需要对齐视频、音频和语言模态以支持长时推理。GRPO已成为增强推理能力的有力方法，在数学等可验证领域表现出色。

**现有痛点**：GRPO在开放域视频任务上存在三大限制——(1) 在线策略（on-policy）导致数据低效，尤其视频数据标注昂贵；(2) **优势消失问题**：当组内所有响应奖励相同时（全对或全错），优势变为零，学习信号消失；(3) **均匀信用分配**：所有token接收相同奖励，忽略了推理链中不同阶段的重要性差异。

**核心矛盾**：视频推理中，初始规划（定位声音来源）和最终综合（结合视听线索识别说话者）是关键阶段，但GRPO平等对待所有token，稀释了梯度信号。

**切入角度**：(1) Transformer的attention sink效应——初始token持续获得注意力作为规划参考；(2) 最终token对综合答案至关重要。

**核心idea**：离线策略架构用分层回放缓冲区解决数据效率和优势消失；U型抛物线加权(TAS)强调推理链首尾阶段。

## 方法详解

### 整体框架
三阶段RL训练管线：冷启动SFT(S0)→视觉推理RL(S1)→音视频推理RL(S2)→音频目标定位RL(S3)。每阶段使用不同数据集和奖励配置。

### 关键设计

1. **离线策略架构（Off-Policy Architecture）**:

    - **分层回放缓冲区**：大小10K，三层固定容量——Easy(25%)、Medium(35%)、Hard(40%)。层分配基于prompt的移动平均奖励 $\bar{R}(q)$，动态分位数决定阈值
    - **解决优势消失**：从缓冲区采样包含成功和失败轨迹的训练组，确保组内奖励多样性→非零优势→持续梯度更新
    - **提示机制(Hinting)**：当prompt持续困难（低 $\bar{R}(q)$）且策略KL散度低时，触发预计算提示引导exploration
    - 混合训练：$\mathcal{J}_{AVATAR} = \mathcal{J}_{on\text{-}policy} + \alpha \cdot \mathcal{J}_{off\text{-}policy}$，离线项用重要性采样校正策略漂移

2. **时间优势塑造(TAS)**:

    - 功能：对推理链中不同位置的token施加不同权重
    - 核心思路：U型抛物线加权函数 $w_t = 1.0 + \lambda_{TAS} \cdot (2\tilde{t} - 1)^2$，其中 $\tilde{t} = t/(L-1) \in [0,1]$。首尾token权重为 $1.0 + \lambda_{TAS}$，中间token权重为1.0
    - 成型优势：$A_{i,t}^{TAS} = w_{i,t} \cdot A_i$
    - 设计动机：Transformer的attention sink效应和综合阶段的关键性，强化规划和综合阶段的学习信号

3. **多源奖励函数**:

    - **格式奖励** $R_{format}$：验证<think>...</think><answer>...</answer>格式
    - **准确度奖励** $R_{acc}$：非零数值任务用rMAE提供密集奖励
    - **自奖励** $R_{self}$：组内多数投票产生伪正确答案用于共识学习
    - **步骤推理评判** $R_{judge}$：冻结的VLM评判器(InternVL3-2B)评估推理质量

### 损失函数 / 训练策略
各阶段奖励配置不同：S1用0.5×格式+0.5×准确度；S2加入自奖励；S3加入推理评判。

## 实验关键数据

### 主实验（Qwen2.5-Omni基线）

| 模型 | OmniBench | DailyOmni | AV-Counting | WorldSense |
|------|-----------|-----------|-------------|------------|
| Qwen2.5-Omni (基线) | 44.2 | 44.0 | 22.3 | 44.2 |
| + GRPO | 45.4 (+1.2) | 44.8 (+0.8) | 22.8 (+0.5) | 45.1 (+0.9) |
| **+ AVATAR** | **49.1 (+4.9)** | **47.0 (+3.0)** | **23.1 (+0.8)** | **46.0 (+1.8)** |

AVATAR vs GRPO: OmniBench **+3.7**, Video-Holmes +1.9。样本效率5倍（减少80%生成量达到目标性能）。

### 消融实验

| 配置 | OmniBench | MMVU | Video-Holmes |
|------|-----------|------|-------------|
| GRPO (基线) | 45.4 | 56.6 | 39.0 |
| + Off-Policy | 47.2 | 57.5 | 39.8 |
| + TAS | 47.8 | 57.9 | 40.2 |
| + Both (AVATAR) | **49.1** | **58.2** | **40.5** |

### 关键发现
- 离线策略和TAS各自独立贡献显著，组合后进一步提升
- AVATAR对两个不同基线(Ola-7B和Qwen2.5-Omni)都有效，证明方法无关模型
- 在AV-Odyssey和IntentBench等更难的基准上也有提升
- 回放缓冲区的分层设计避免了Easy样本被频繁淘汰，保持训练多样性

## 亮点与洞察
- **优势消失问题的优雅解决**：通过分层回放缓冲区引入组内奖励多样性，从根本上避免了零优势梯度。这个思路对所有GRPO类方法都有价值
- **TAS的简洁性**：仅用一个U型抛物线函数就有效强化了推理链的关键阶段，无需训练critic网络
- **三阶段渐进训练**：从视觉推理→音视频推理→精细定位的课程安排，逐步增加任务难度

## 局限与展望
- TAS的抛物线形状是手工设计的，可能不是最优形状——能否自动学习每个token的重要性？
- 提示机制依赖预计算的hint，增加了数据准备成本
- 当前离线数据的重要性采样比可能不稳定，需clipping保护
- 仅在选择题形式上验证，开放式视频问答的表现有待探索

## 相关工作与启发
- **vs Video-R1**: Video-R1用时间对比奖励但沿用标准GRPO，仍有优势消失问题
- **vs DAPO**: DAPO通过修改采样减少均匀组，但对困难query仍有零梯度问题
- **vs HumanOmni**: 用LLM评判奖励但均匀信用分配，AVATAR通过TAS解决了信用分配问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 离线GRPO和TAS创意好，但各组件相对独立
- 实验充分度: ⭐⭐⭐⭐⭐ 多基线、多基准、95%置信区间、样本效率分析
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，但论文较长
- 价值: ⭐⭐⭐⭐ 对GRPO训练范式有通用改进价值
# AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video

**会议**: CVPR 2026  
**arXiv**: [2508.03100](https://arxiv.org/abs/2508.03100)  
**代码**: https://people-robots.github.io/AVATAR/ (有)  
**领域**: 人体理解 / 多模态推理  
**关键词**: 音视频推理, GRPO改进, 离策略强化学习, 时间优势塑形, 多模态大语言模型

## 一句话总结
提出AVATAR框架，通过离策略训练架构（分层重放缓冲区）和时间优势塑形（TAS，U形加权强调推理链首尾）两个核心组件改进GRPO，解决其数据低效、优势消失和均匀信用分配三大问题，在音视频推理基准上显著超越GRPO基线。

## 研究背景与动机

1. **领域现状**：MLLM需要对齐视频、音频和语言模态来支持长时推理。GRPO作为RL方法已展示出增强推理的潜力，但在开放式视频域有显著局限。
2. **GRPO三大问题**：
    - **数据低效**：在策略方法，每次更新后丢弃经验，在昂贵视频标注数据上浪费严重
    - **优势消失**：组内奖励方差坍缩时（全正确或全错误），优势归零，学习信号消失
    - **均匀信用分配**：对推理链中所有token施加相同奖励，忽视规划阶段（开头）和综合阶段（结尾）的关键性
3. **切入角度**：从RL算法设计角度系统解决GRPO的三个结构性缺陷。
4. **核心idea**：离策略架构+分层重放缓冲区解决前两个问题；TAS的U形位置加权解决第三个问题。

## 方法详解

### 整体框架
三阶段RL训练流程：Stage 0 SFT冷启动 → Stage 1 视觉推理RL → Stage 2 音视频推理RL → Stage 3 音频目标定位RL。每阶段使用不同数据集和奖励配置。

### 关键设计

1. **离策略架构与分层重放缓冲区**:
    - 功能：复用历史经验提高样本效率，保证组内奖励多样性
    - 核心思路：维护容量10K的分层缓冲区 $\mathcal{B}$，分三层：Easy(25%)、Medium(35%)、Hard(40%)。每个prompt的移动平均奖励 $\bar{R}(q)$ 决定层级分配。混合目标 $\mathcal{J}_{AVATAR} = \mathcal{J}_{on} + \alpha \cdot \mathcal{J}_{off}$，离策略部分用重要性采样比 $r_i^{off} = \pi_\theta(o_i|q) / \pi_{\theta_{off}}(o_i|q)$ 校正策略漂移
    - 设计动机：(1) Hard层容量最大，确保困难样本被反复训练；(2) 历史成功/失败轨迹混合入组，保证奖励方差非零，解决优势消失问题

2. **Hinting机制**:
    - 功能：当prompt持续困难且策略停止探索时，注入预计算提示引导
    - 核心思路：监控 $D_{KL}(\pi_\theta || \pi_\beta)$，当 $\bar{R}(q)$ 低且KL低时触发hint（如"先定位发声物体，再计数"），由Qwen2.5-VL-72B预生成
    - 设计动机：帮助agent逃离局部最优，保持训练在有挑战但可解的区间

3. **时间优势塑形(TAS)**:
    - 功能：对推理链中不同位置的token施加不同权重的优势
    - 核心思路：U形抛物线加权 $w_t = 1.0 + \lambda_{TAS} \cdot (2\tilde{t} - 1)^2$，其中 $\tilde{t} = t/(L-1)$。序列首尾（规划+综合阶段）获得最高权重 $1+\lambda_{TAS}$，中间token权重为1.0。token级优势 $A_{i,t}^{TAS} = w_{i,t} \cdot A_i$
    - 设计动机：基于Transformer的注意力沉降效应（初始token持续获得注意力）和最终token在综合答案中的关键作用，强化推理链的首尾

4. **多奖励函数设计**:
    - 格式奖励 $R_{format}$：验证推理格式
    - 准确度奖励 $R_{acc}$：用rMAE做密集奖励
    - 自奖励 $R_{self}$：多数投票产出的伪正确答案做一致性奖励
    - 推理质量判断 $R_{judge}$：冻结InternVL3-2B评估推理过程

### 损失函数 / 训练策略
在标准GRPO目标上混合离策略项，用TAS替换均匀优势。三阶段训练逐步从视觉推理过渡到音视频推理再到精细定位。

## 实验关键数据

### 主实验（多基准对比）

| 模型 | OmniBench | MMVU | Video-Holmes | AV-Odyssey |
|------|-----------|------|-------------|------------|
| Qwen2.5-Omni (基线) | 44.2 | - | - | 29.8 |
| + GRPO | 45.4 (+1.2) | - | - | 31.3 (+1.5) |
| + **AVATAR** | **49.1 (+4.9)** | - | - | **32.1 (+2.3)** |
| Ola-7B (基线) | 45.3 | - | - | 25.6 |
| + GRPO | 46.8 (+1.5) | - | - | 27.0 (+1.4) |
| + **AVATAR** | **47.2 (+1.9)** | - | - | **28.8 (+3.2)** |

AVATAR vs GRPO on Qwen2.5-Omni: OmniBench +3.7, Video-Holmes +1.9, 同时只需**80%更少的生成补全**达到目标性能。

### 消融实验

| 组件 | OmniBench | DailyOmni | 说明 |
|------|-----------|-----------|------|
| GRPO (基线) | 45.4 | 44.8 | |
| + Off-policy only | +1.5 | +1.2 | 离策略架构贡献 |
| + TAS only | +1.0 | +0.8 | 时间塑形贡献 |
| + Both (AVATAR) | **+3.7** | **+2.2** | 两者互补 |

### 关键发现
- AVATAR在两个基础模型(Qwen2.5-Omni和Ola-7B)上都一致有效，证明方法的模型无关性
- 样本效率提升5×：需要80%更少的生成补全即可达目标性能
- 离策略和TAS的增益互补而非重叠
- 所有改进均附95%置信区间(bootstrap)，统计可靠

## 亮点与洞察
- **系统性解决GRPO缺陷**：很好地将RL中的经典问题（离策略学习、信用分配、探索-利用）工程化应用到MLLM训练中
- **TAS的U形加权简洁有效**：理论上对齐了Transformer的注意力模式，实现上只需一行公式修改，无需额外网络或critic
- **Hinting机制的实用性**：利用大模型（72B）为小模型预计算学习引导，是一种实用的teacher-student RL策略

## 局限与展望
- TAS的U形是固定形状，不同任务/不同推理长度可能需要自适应的形状
- Hinting依赖外部大模型，在完全自主学习场景中不可用
- 仅在音视频QA任务上验证，对更长时间推理（如规划、决策）的效果未知
- 重放缓冲区的大小(10K)和层级比例(25/35/40)是手动设定的

## 相关工作与启发
- **vs 标准GRPO**: AVATAR是GRPO的直接改进，保持了GRPO的简洁性同时解决其三个结构性问题
- **vs Video-R1**: Video-R1使用时间对比奖励，AVATAR从训练算法角度优化，两者可组合
- **vs DAPO**: DAPO通过修改采样减少均匀组，但AVATAR通过离策略重放更根本地解决优势消失

## 评分
- 新颖性: ⭐⭐⭐⭐ 组合已有RL技术（离策略、信用分配），但在MLLM场景中的应用有新意
- 实验充分度: ⭐⭐⭐⭐ 多基准、多基础模型、统计检验、消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析清晰，三个限制→三个解决方案的对应关系明确
- 价值: ⭐⭐⭐⭐ 对MLLM RL训练的通用性改进，方法可广泛应用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[CVPR 2026\] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)
- [\[CVPR 2025\] FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video](../../CVPR2025/human_understanding/fate_full-head_gaussian_avatar_with_textural_editing_from_monocular_video.md)
- [\[CVPR 2026\] rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training](rppg_vqa_video_quality_assessment.md)
- [\[CVPR 2026\] LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining](lca_large-scale_codec_avatars_the_unreasonable_effectiveness_of_large-scale_avata.md)

<!-- RELATED:END -->
