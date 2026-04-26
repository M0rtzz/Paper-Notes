---
title: >-
  [论文解读] ArrowGEV: Grounding Events in Video via Learning the Arrow of Time
description: >-
  [ACL 2026][视频理解][视频事件定位] 提出 ArrowGEV，一个受物理学"时间之箭"启发的强化学习框架，通过区分时间敏感和时间不敏感事件来建模视频中的时间方向性，提升 VLM 的事件定位精度和时序理解能力。
tags:
  - ACL 2026
  - 视频理解
  - 视频事件定位
  - 时间方向性
  - 强化学习
  - 视觉语言模型
  - 时序理解
---

# ArrowGEV: Grounding Events in Video via Learning the Arrow of Time

**会议**: ACL 2026  
**arXiv**: [2601.06559](https://arxiv.org/abs/2601.06559)  
**代码**: [有](https://arxiv.org/abs/2601.06559)（Code / Model / Data 均公开）  
**领域**: Video Understanding  
**关键词**: 视频事件定位, 时间方向性, 强化学习, 视觉语言模型, 时序理解

## 一句话总结

提出 ArrowGEV，一个受物理学"时间之箭"启发的强化学习框架，通过区分时间敏感和时间不敏感事件来建模视频中的时间方向性，提升 VLM 的事件定位精度和时序理解能力。

## 研究背景与动机

**领域现状**: 视频事件定位（GEV）是视频分析的基础任务，近年来 VLM 凭借端到端推理能力成为主流方法，通过大规模时间戳标注训练、时间 token 嵌入或视频分割适配来实现事件定位。

**现有痛点**: 现有方法仅在正向视频上对齐事件与时间戳，忽略了事件的内在时间结构和方向性。实验表明 VLM 无法区分正向和反向视频中事件语义的变化——例如"拿起杯子"反转后变为"放下杯子"，但模型仍然错误地在反向视频中定位原始事件。

**核心矛盾**: VLM 过度拟合文本时间戳而非视频语义，缺乏对事件时间方向性的理解，导致在需要时序推理的任务上泛化性不足。

**本文目标**: 通过显式建模时间方向性，提升 VLM 的事件定位精度和时序结构理解能力。

**切入角度**: 借鉴物理学中"时间之箭"概念，将事件分为时间敏感（反转改变语义）和时间不敏感（反转不变）两类，设计差异化的奖励信号。

**核心 idea**: 用反向视频作为额外训练信号——对时间敏感事件惩罚反向视频中的定位，对时间不敏感事件强制正反一致性。

## 方法详解

### 整体框架

基于 GRPO 强化学习框架，输入正向和反向视频，根据事件类别计算差异化奖励。训练后 VLM 不仅能准确定位正向视频事件，还能理解时间结构以增强鲁棒性。

### 关键设计

1. **事件时间方向性分类**:

    - 功能：将事件分为时间敏感和时间不敏感两类
    - 核心思路：用 LLM 推理判断事件类别 $c(q) \in \{\text{sensitive}, \text{insensitive}\}$，如"开门"是时间敏感的（反转变"关门"），"球在桌上"是时间不敏感的
    - 设计动机：不同类型事件在时间反转下的语义变化不同，需要差异化处理

2. **时间方向性奖励建模**:

    - 功能：结合定位精度和时间方向性的统一奖励函数
    - 核心思路：$r_{\text{grounding}} = r_{\text{acc}} + \lambda \cdot r_{\text{temp}}$，其中 $r_{\text{acc}}$ 使用 tIoU 评估正向定位精度，$r_{\text{temp}}$ 对不敏感事件奖励正反一致性（$S_c$），对敏感事件奖励差异性（$1-S_c$）
    - 设计动机：统一框架下同时优化定位精度和时间方向理解

3. **难度感知训练策略**:

    - 功能：动态调整样本权重和训练数据分布
    - 核心思路：权重调整 $w_i = \exp((1 - \text{avg\_tIoU})/\tau)$ 让模型聚焦困难样本；动态课程过滤在每个 epoch 结束时移除已掌握样本（最差 IoU > $\eta=0.7$）
    - 设计动机：训练过程中样本逐渐变简单，需要动态维持学习信号强度

### 损失函数 / 训练策略

最终奖励 $r_{\text{final}} = r_{\text{grounding}} + r_{\text{form}}(o)$，其中 $r_{\text{form}}$ 是格式奖励，要求输出 `<think>...</think><answer>$t_s$ to $t_e$</answer>` 模板。基于 Qwen2.5-VL-7B-Instruct，2 FPS 采样。

## 实验关键数据

### 主实验

| 方法 | Charades-STA R1@0.5 | ActivityNet R1@0.5 | TVGBench R1@0.5 |
|------|-------------------|-------------------|-----------------|
| Gemini-2.5-Pro | 25.5 | 31.9 | 25.7 |
| GPT-5 | 18.3 | 33.0 | 18.8 |
| TimeSuite* | 67.1 | - | - |
| ArrowGEV (本文) | **显著提升** | **显著提升** | **显著提升** |

### TDD 指标（时间方向性理解）

引入 Temporal Directionality Discrepancy (TDD) 指标：$\text{TDD}(m) = \frac{R1@m(\text{fwd}) - R1@m(\text{rev})}{R1@m(\text{fwd})}$。对时间敏感事件 TDD 应接近 1（能区分正反），对时间不敏感事件 TDD 应接近 0（正反一致）。

### 关键发现

- ArrowGEV 在三个 GEV 基准上均显著提升定位精度
- 大幅改善 VLM 对时间方向性的理解（TDD 指标）
- 在 OOD 通用视频理解和推理任务（TempCompass、MVBench、VideoMME 等）上也有提升
- 时间敏感事件在常用基准中占比显著，特别是 Charades-STA

## 亮点与洞察

- "时间之箭"概念从物理学引入视频理解，角度新颖且直觉清晰
- 利用反向视频作为"免费"的训练信号，不需额外标注
- 提出 TDD 指标，首次量化评估模型对事件时间方向性的理解
- 难度感知训练策略（权重调整 + 课程过滤）有效维持学习效率

## 局限与展望

- 事件分类依赖 LLM 推理，可能存在分类噪声
- 仅在 7B 模型上验证，更大模型的效果待探索
- 视频采样率 2 FPS 可能不足以捕捉快速事件
- 未来可探索更细粒度的时间方向性建模

## 相关工作与启发

- GRPO / DeepSeek-R1：RL 训练范式基础
- TimeSuite / ChatVTG：GEV 任务的监督学习方法
- 时间方向性相关的自监督学习（shuffle-and-learn、order prediction）
- 将时间方向性作为视频理解的基本归纳偏置是一个有前景的方向

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 物理学启发的时间方向性建模，视角独特
- 实验充分度: ⭐⭐⭐⭐ 三个 GEV 基准 + 六个通用基准，消融充分
- 写作质量: ⭐⭐⭐⭐ 动机清晰，pilot study 有说服力
- 价值: ⭐⭐⭐⭐ 揭示了 VLM 时间方向性理解的缺陷，提出有效方案

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Seeing the Arrow of Time in Large Multimodal Models](../../NeurIPS2025/video_understanding/seeing_the_arrow_of_time_in_large_multimodal_models.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](../../CVPR2026/video_understanding/how_should_video_llms_output_time.md)
- [\[CVPR 2025\] Localizing Events in Videos with Multimodal Queries](../../CVPR2025/video_understanding/localizing_events_in_videos_with_multimodal_queries.md)
- [\[ECCV 2024\] R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding](../../ECCV2024/video_understanding/r2tuning_efficient_imagetovideo_transfer_learning_for_video.md)
- [\[CVPR 2026\] Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](../../CVPR2026/video_understanding/learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)

<!-- RELATED:END -->
