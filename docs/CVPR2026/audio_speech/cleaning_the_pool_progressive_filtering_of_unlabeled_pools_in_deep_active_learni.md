---
title: >-
  [论文解读] Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning
description: >-
  [CVPR 2026][语音][主动学习] 提出 Refine 集成主动学习方法，通过两阶段策略——渐进过滤（多策略迭代精炼无标签池）+ 覆盖选择（从精炼池中选择多样性高价值样本）——在不预知最佳策略的情况下一致超越单一 AL 策略和现有集成方法。
tags:
  - CVPR 2026
  - 语音
  - 主动学习
  - 集成策略
  - 渐进过滤
  - 基础模型
  - 覆盖选择
---

# Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning

**会议**: CVPR 2026  
**arXiv**: [2511.22344](https://arxiv.org/abs/2511.22344)  
**代码**: [GitHub](https://github.com/dhuseljic/dal-toolbox) (有)  
**领域**: Audio/Speech (主动学习)  
**关键词**: 主动学习, 集成策略, 渐进过滤, 基础模型, 覆盖选择

## 一句话总结

提出 Refine 集成主动学习方法，通过两阶段策略——渐进过滤（多策略迭代精炼无标签池）+ 覆盖选择（从精炼池中选择多样性高价值样本）——在不预知最佳策略的情况下一致超越单一 AL 策略和现有集成方法。

## 研究背景与动机

**领域现状**：预训练基础模型（DINOv2、CLIP）适配下游任务仍需标注数据。主动学习通过智能选择标注样本降低标注成本，但近年基准显示没有单一策略始终最优。

**现有痛点**：(a) 不同 AL 策略捕获不同"数据价值"观——不确定性 vs 代表性，无策略始终占优；(b) 选错策略可能比随机采样更差；(c) 现有集成方法（TCM/TAILOR/SelectAL）依赖启发式切换或学习调度，表现不稳定。

**核心矛盾**：AL 是 one-shot 问题（无试错机会），必须在不知最佳策略时做选择。

**本文目标**：设计无需学习的集成方法，自动整合多种互补策略优势。

**切入角度**：将重点从"选什么样本"转为"先清理池子去掉无价值样本"。

**核心 idea**：让多策略反复投票筛选——经多轮保留的样本必被至少一个策略认为有价值，未被任何策略选中的样本必无价值。

## 方法详解

### 整体框架

两阶段选择：(1) 渐进过滤迭代精炼无标签池；(2) 覆盖选择从精炼池中选最终批次。

### 关键设计

1. **渐进过滤 (Progressive Filtering)**：$R=5$ 轮迭代，每轮每个策略从 $\alpha=0.4$ 的随机子采样中选 $J=10$ 个批次，取所有批次的并集作为下一轮候选池：
    $\mathcal{C}_r = \bigcup_{m=1}^M \bigcup_{j=1}^J s_m(\text{SubSample}(\mathcal{C}_{r-1}, \alpha \cdot |\mathcal{C}_{r-1}|), b)$
   
   三个关键设计决策及动机：
    - **并集而非交集**：保留所有被任一策略认为有价值的样本，避免丢弃独特发现
    - **多轮迭代**：单轮仅简单拼接；多轮中无价值样本存活概率指数衰减
    - **子采样 $\alpha < 1$**：使确定性策略产生多样批次，降低内存需求

2. **覆盖选择 (Coverage-Based Selection)**：从精炼池 $\mathcal{C}_R$ 中用 UHerding 选最大化覆盖的批次：
    $\mathcal{B}^* = \arg\max_{\mathcal{B} \subset \mathcal{C}_R} \mathbb{E}_{\mathbf{x}}[\max_{\mathbf{x}' \in (\mathcal{L}_t \cup \mathcal{B})} k(\mathbf{x}, \mathbf{x}')]$
   设计动机：精炼后的池已是高价值候选集，只需覆盖式选择确保多样性。

3. **理论保证**：

    - 定理1（价值保留）：$P_r(\mathbf{x}) \geq 1 - (1 - \alpha \cdot \max_m p_{m,r}(\mathbf{x}))^J$
    - 定理2（指数衰减）：$\epsilon$-无价值样本经 $R$ 轮存活概率 $\leq (MJ\alpha\epsilon)^R$
    - 定理3（价值单调性）：$\mathbb{E}[V|\mathcal{C}_R] \geq ... \geq \mathbb{E}[V|\mathcal{C}_0]$

### 训练设置

- 3 个骨干：DINOv2-ViT-S/14, DINOv3-ViT-S/16, CLIP-ViT-B/16
- 冻结骨干+训练分类头，SGD LR 0.01, 200 epochs/cycle
- 20 AL 周期，10 次独立运行

## 实验关键数据

### 主实验 — 综合胜率

| Refine vs | 胜率 (3骨干×5数据集×10试验) |
|-----------|--------------------------|
| BAIT | 85% |
| UHerding | 80% |
| SelectAL | 100% |
| TAILOR | 100% |
| TCM | 98% |
| AutoAL | 97% |

### 消融实验 — 渐进过滤作为前处理

| 策略 | 从原始池 AULC | 从精炼池 AULC | 提升 |
|------|-------------|-------------|------|
| Random | 基线 | +3.7% | 仅过滤+随机即有效 |
| BAIT | 差于Random | **优于Random** | 过滤拯救失败策略 |
| AlfaMix | 基线 | +2.6% | 普遍受益 |
| UHerding | 高 | +0.7% | 强策略也受益 |

### 过滤轮数影响

| R (轮数) | CIFAR-10 AULC提升 | Snacks AULC提升 |
|---------|-------------------|-----------------|
| 1 | +3.02% | +6.91% |
| 3 | +3.72% | +7.22% |
| 5 | +3.71% | +7.79% |
| 9 | +3.78% | +8.43% |

### 关键发现

1. Refine 对所有单一策略和所有集成方法均有最高综合胜率。
2. 渐进过滤是通用前处理——任何 AL 策略应用于精炼池后都比应用于原始池更好。
3. BAIT 从原始池选差于随机，过滤后反转为优于随机——过滤去除了误导样本。
4. 仅 Margin+TypiClust 两策略过滤即可自动整合不确定性和代表性。
5. $\alpha \in [0.3, 0.9]$ 范围内性能稳定。

## 亮点与洞察

- **渐进过滤作为前处理**是最实用的贡献——零成本嫁接到任何 AL 策略。
- 理论分析优雅：三个定理分别保证价值保留、噪声去除、质量单调提升。
- "从不被任何策略选中的样本肯定无价值"——洞察简洁但深刻。
- 易于扩展：新策略直接加入集成，无需重训。

## 局限与展望

- 多次调用多个策略增加计算开销（虽可并行化）。
- 集成中多数策略失效时精炼池质量可能下降（Dopanim+CLIP 案例）。
- 未探索策略加权——当前等权，可根据历史表现动态调权。
- 主要在图像分类上验证，检测/分割待进一步测试。

## 相关工作与启发

- 与 TCM 的硬切换不同，Refine 通过渐进过滤自然实现软切换。
- 并集+迭代的设计可推广到其他需要集成多种启发式的场景。
- 为基础模型时代的 AL 提供了实用且理论上有保证的解决方案。

## 评分

- 新颖性: ⭐⭐⭐⭐ 渐进过滤思路简洁新颖，但是策略组合而非根本新范式
- 实验充分度: ⭐⭐⭐⭐⭐ 6数据集×3骨干×8单策略+4集成方法+充分消融+理论分析
- 写作质量: ⭐⭐⭐⭐⭐ 理论和实验完美互补，结构清晰
- 价值: ⭐⭐⭐⭐ 通用AL前处理步骤，实用价值高

<!-- RELATED:START -->

## 相关论文

- [PACE: Pretrained Audio Continual Learning](../../ICLR2026/audio_speech/pace_pretrained_audio_continual_learning.md)
- [Learning to Highlight Audio by Watching Movies](../../CVPR2025/audio_speech/learning_to_highlight_audio_by_watching_movies.md)
- [EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](../../ICLR2026/audio_speech/emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)
- [Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](../../AAAI2026/audio_speech/do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)
- [LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale](../../CVPR2025/audio_speech/livecc_learning_video_llm_with_streaming_speech_transcription_at_scale.md)

<!-- RELATED:END -->
