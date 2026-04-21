---
title: >-
  [论文解读] Reinforce to Learn, Elect to Reason: A Dual Paradigm for Video Reasoning
description: >-
  [CVPR 2026][视频推理] 提出 RLER 双范式框架，训练阶段用 GRPO 配合三种新颖奖励（Frame-sensitive、Think-transparency、Anti-repetition）教模型生成结构化证据，推理阶段用无训练编排器在多候选之间基于证据一致性进行加权选举和自检，在 8 个视频基准上全面超越开源和 RL-based LMM，平均提升 6.3%，仅需约 3.1 个候选。
tags:
  - CVPR 2026
  - 视频推理
  - 强化学习
  - 证据驱动
  - 多候选选举
  - 测试时推理
---

# Reinforce to Learn, Elect to Reason: A Dual Paradigm for Video Reasoning

**会议**: CVPR 2026  
**arXiv**: [2604.04379](https://arxiv.org/abs/2604.04379)  
**代码**: 无  
**领域**: 视频理解 / 多模态推理 / 强化学习  
**关键词**: 视频推理, 强化学习, 证据驱动, 多候选选举, 测试时推理

## 一句话总结

提出 RLER 双范式框架，训练阶段用 GRPO 配合三种新颖奖励（Frame-sensitive、Think-transparency、Anti-repetition）教模型生成结构化证据，推理阶段用无训练编排器在多候选之间基于证据一致性进行加权选举和自检，在 8 个视频基准上全面超越开源和 RL-based LMM，平均提升 6.3%，仅需约 3.1 个候选。

## 研究背景与动机

1. **领域现状**：大型多模态模型（LMM）在视频理解上取得了显著进展，但推理仍然是"单次通过"——生成答案后不验证推理是否基于有效证据。即使 SOTA 模型也容易被轻微扰动（视频变化或措辞变化）打破。
2. **现有痛点**：
    - 现有基于 RL 的训练方法（如 Video-R1、VideoChat-R1）改善了推理能力，但多个推理轨迹之间很少检查证据一致性
    - 测试时扩展方法（best-of-N、beam search）提供了多样候选，但缺乏基于证据的系统性仲裁
    - 链式思考（CoT）很少被系统性地与关键帧和关系进行对照验证
3. **核心矛盾**：现有方法能展示模型"能推理"，但无法证明它"用正确的证据进行了推理"
4. **本文目标**：将视频推理从"答案驱动"转向"证据驱动"——训练时让模型发出结构化的机器可检查证据信号，推理时通过证据一致性选举出可靠答案
5. **切入角度**：分离"能够思考"和"正确思考"——训练负责塑造和增强推理能力（potentiation），推理负责通过证据选举来保证可靠性
6. **核心 idea**：训练阶段用奖励塑造让模型生成包含关键帧引用和结构化推理的输出，推理阶段用证据加权选举从多个候选中选出最可靠的答案

## 方法详解

### 整体框架

RLER 由两个对称阶段组成：**RLER-Training** 基于 GRPO 优化策略，用 5 种奖励信号塑造证据中心的结构化输出（含 `<think>`、`<answer>`、`<keyframes>` 标签）；**RLER-Inference** 通过多样化输入生成少量候选，解析结构、计算证据分数、加权选举、异常去除、自适应预算和最终裁判自检，形成完整的证据驱动决策闭环。基础模型为 Qwen2.5-VL-7B-Instruct，仅更新语言模型参数（LoRA r=8）。

### 关键设计

1. **三种新颖训练奖励**:

    - 功能：教模型生成结构化的、可机器解析的证据信号
    - 核心思路：
        - **Frame-sensitive Reward**：鼓励模型引用具体的关键帧。定义有效帧分数 $s_{fs}(o) = \text{clip}(\frac{|K(o)| - E(o)}{1 + |K(o)|}, 0, 1)$，其中 $K(o)$ 是有效帧索引集合，$E(o)$ 计算无效索引数。被答案正确性和格式有效性门控。
        - **Think-transparency Reward**：鼓励适度长度的推理链（非过短也非冗长）。用 $\sin^2(\pi \tilde{L}(o))$ 的单峰曲线在中等长度时给最大奖励。
        - **Anti-repetition Reward**：用 n-gram 去重抑制低信息密度的重复。$R_{ar}(o) = -\rho(o)$，其中 $\rho$ 是重复率。
    - 设计动机：这三种奖励精确对应视频推理的三个核心需求——"看哪里"、"说多少"、"怎么说"，且训练信号直接对接推理阶段的评分维度。

2. **证据对齐推理编排器 (RLER-Inference)**:

    - 功能：在不增大模型的情况下提升答案可靠性和可解释性
    - 核心思路：(1) 通过温度网格 {0.2, 0.7, 0.9}、五角裁剪、亮度扰动等生成多样候选；(2) 解析每个候选为 $(a_i, K_i, z_i, c_i)$（答案、关键帧、推理、置信度）；(3) 计算综合证据分数 $S_i = \frac{1}{4}(s_{fs}(o_i) + \tau(o_i) + (1-\rho(o_i)) + c_i)$；(4) 证据加权选举——计算答案支持者的证据交集（Jaccard 重叠），去除极端值后加权投票；(5) 早停条件：领先边际超过阈值 $\delta=0.08$ 且平均置信度超过 $\gamma=0.4$；(6) 裁判自检：一次性的证据充分性验证，必要时触发重加权。
    - 设计动机：单次推理无法保证证据质量，但通过结构化输出+证据一致性选举，可以在低计算成本下大幅提升可靠性（平均仅需 3.1 个候选/问题）。

3. **训练-推理对称设计**:

    - 功能：使训练奖励与推理评分形成对称闭环
    - 核心思路：训练阶段的 Frame-sensitive Reward、Think-transparency Reward 和 Anti-repetition Reward 分别对应推理阶段的 $s_{fs}$、$\tau$、$1-\rho$ 评分。训练塑造的特性正好是推理需要评估的维度。
    - 设计动机：避免训练-推理脱节，确保训练学到的能力（关键帧引用、适度思考、避免重复）在推理时被直接利用。

### 损失函数 / 训练策略

- 基于 GRPO，组大小 G=4，clip $\epsilon=0.2$，KL 系数 $\beta=0.04$
- 奖励权重：$w_{acc}=0.1, w_{fmt}=0.1, w_{fs}=0.2, w_{tt}=0.3, w_{ar}=0.3$
- 使用 LoRA (r=8, α=16)，AdamW (lr=1e-5)，在 Video-R1-260k 上训练 2 个 epoch
- 冻结视觉编码器和投影层，仅更新语言模型参数
- 16 帧均匀采样训练，32 帧标准推理，长视频 1fps 子采样

## 实验关键数据

### 主实验

| 基准 | 类型 | Qwen2.5-VL-7B | Video-R1 | VideoChat-R1.5 | **RLER** |
|------|------|---------------|----------|----------------|----------|
| VSIBench | 视频推理 | 37.4 | 35.8 | - | **43.3** |
| VideoMMMU | 视频推理 | 47.4 | 52.3 | 51.4 | **54.2** |
| VideoMME | 通用 | 65.1 | 59.3 | 67.1 | **68.5** |
| TempCompass | 通用 | 69.2 | 73.2 | - | **76.2** |
| MVBench | 通用 | 67.5 | 63.9 | 70.6 | **72.9** |
| LVBench | 长视频 | 42.0 | - | 48.4 | **50.7** |
| LongVideoBench | 长视频 | 56.0 | - | - | **63.0** |

### 消融实验

**训练奖励消融**

| 配置 | VSIBench | VideoMMMU |
|------|----------|-----------|
| 完整 RLER | **43.3** | **54.2** |
| w/o frame-sensitive | 41.0 (-2.3) | 52.1 (-2.1) |
| w/o think-transparency | 41.9 (-1.4) | 52.7 (-1.5) |
| w/o anti-repetition | 42.1 (-1.2) | 53.0 (-1.2) |
| w/o RLER-Inference | 41.7 (-1.6) | 52.5 (-1.7) |
| w/o GRPO (用 SFT) | 39.2 (-4.1) | 49.8 (-4.4) |

**推理组件消融（MVBench/LVBench）**

| 配置 | MVBench | LVBench | Avg K |
|------|---------|---------|-------|
| 完整 RLER | **72.9** | **50.7** | 3.1 |
| w/o diversity input | 69.1 | 46.5 | 1.0 |

### 关键发现

- **Frame-sensitive Reward 对推理类任务贡献最大**：VSIBench 下降 2.3%，因为该基准需要精确的空间推理和跨帧关联
- **GRPO vs SFT 差距巨大**：SFT 只学格式但缺乏奖励信号激发的深度推理能力，差距 4.1-4.4%
- **仅训练（w/o RLER-Inference）已经很强**：41.7/52.5 已超越大多数 RL-based LMM，证明奖励塑造本身就增强了推理能力
- **平均仅需 3.1 个候选/问题**就能获得全部收益（上限 K=8），得益于高效的早停机制
- 训练过程中出现了"Aha moment"——模型自发识别推理中的缺陷并启动自我纠正（"Wait. Let me re-evaluate."）

## 亮点与洞察

- **证据驱动的训练-推理闭环**是核心创新：同一套维度（帧引用、透明度、信息密度）在训练中作为奖励信号，在推理中作为评分标准，设计非常优雅。
- **在 VideoMME 上超越 GPT-4o（68.5 vs 67.9）**，是一个 7B 开源模型超过闭源大模型的亮点。
- **证据加权选举 vs 简单多数投票**：消融显示去掉证据权重后性能显著下降，证明基于证据的选举比简单投票更有效。
- **"Aha moment"的涌现**表明 RL 训练不仅改善了输出质量，还改变了模型的推理行为模式，这是一个值得深入研究的现象。

## 局限与展望

- 推理时多候选生成增加计算成本（虽然平均 3.1 个已很高效，但对延迟敏感场景仍是问题）
- 仅在 Qwen2.5-VL-7B 上验证，未测试更大模型或其他 LMM 架构
- 裁判自检（referee self-check）仅做一次，可以探索迭代验证
- 关键帧引用目前是弱监督的（奖励级别），未使用帧级标注
- 早停阈值 $\delta=0.08$ 和 $\gamma=0.4$ 在所有基准上共享，针对不同难度的数据集自适应调整可能更优

## 相关工作与启发

- **vs Video-R1**: Video-R1 引入了视频 RL 训练，但仅做单次推理。RLER 在训练端加入更细粒度的证据奖励并在推理端增加选举机制，全面超越（VSIBench: 43.3 vs 35.8）
- **vs VideoChat-R1.5**: 两者都用 RL 训练 + 测试时扩展，但 RLER 的证据对齐选举比简单的 beam search 或 best-of-N 更有效
- **vs test-time scaling**: 传统方法（best-of-N、MCTS）缺乏视频特有的证据评分机制，RLER 的帧引用和证据一致性评分填补了这一空白

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 训练-推理对称闭环的证据驱动范式是全新的框架设计，三种奖励的设计精准且互补
- 实验充分度: ⭐⭐⭐⭐⭐ 8个基准全面评估，训练和推理分别消融，辅助指标（EGS、TI、RR）量化证据质量
- 写作质量: ⭐⭐⭐⭐ 框架介绍清晰，公式推导完整，但论文较长需要耐心阅读
- 价值: ⭐⭐⭐⭐⭐ 对视频推理可靠性问题提出了系统性解决方案，7B 模型超越 GPT-4o 的实际价值很高

<!-- RELATED:START -->

## 相关论文

- [How LLMs Learn to Reason: A Complex Network Perspective](../../ICLR2026/reinforcement_learning/how_llms_learn_to_reason_a_complex_network_perspective.md)
- [CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)
- [ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering](reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)
- [Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)
- [VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](../../NeurIPS2025/reinforcement_learning/videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)

<!-- RELATED:END -->
