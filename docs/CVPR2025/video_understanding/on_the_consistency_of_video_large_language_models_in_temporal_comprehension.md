---
title: >-
  [论文解读] On the Consistency of Video Large Language Models in Temporal Comprehension
description: >-
  [CVPR 2025][视频理解][视频大语言模型] 系统研究视频大语言模型 (Video-LLMs) 在时序理解中的预测一致性，发现当前模型在改述查询、时间偏移和自我验证等探测下一致性极差（接近随机水平），提出事件时序验证微调 (VTune) 方法通过显式考虑一致性显著改善 grounding 和一致性表现。
tags:
  - CVPR 2025
  - 视频理解
  - 视频大语言模型
  - 时序理解一致性
  - 视频时序定位
  - 鲁棒性评估
  - 指令微调
---

# On the Consistency of Video Large Language Models in Temporal Comprehension

**会议**: CVPR 2025  
**arXiv**: [2411.12951](https://arxiv.org/abs/2411.12951)  
**代码**: [github](https://github.com/minjoong507/Consistency-of-Video-LLM)  
**领域**: Video Understanding / Temporal Grounding  
**关键词**: 视频大语言模型, 时序理解一致性, 视频时序定位, 鲁棒性评估, 指令微调

## 一句话总结

系统研究视频大语言模型 (Video-LLMs) 在时序理解中的预测一致性，发现当前模型在改述查询、时间偏移和自我验证等探测下一致性极差（接近随机水平），提出事件时序验证微调 (VTune) 方法通过显式考虑一致性显著改善 grounding 和一致性表现。

## 研究背景与动机

视频大语言模型能定位语言查询对应的视频时刻（temporal grounding），但这种时序理解能力的**鲁棒性和可信度**尚未被充分研究。关键问题在于：当模型定位了一个视频片段后，其后续回答是否与初始定位一致？

实际观察表明，现有 Video-LLMs 在自我验证时表现极不一致——当被问到"该事件是否发生在你刚预测的时段内"时，模型往往给出矛盾的答案。这种不一致性说明模型的时序理解可能并非基于真正的视频内容理解，而是依赖语言先验。

现有的"时序感知"模型（如 TimeChat、VTimeLLM）虽在 grounding 指标上有进步，但在一致性方面改善有限。常规的 prompting（CoT、描述提示）和指令微调虽能提升定位性能，但对一致性改善不稳定。

核心动机：**需要一种显式考虑一致性的训练方法**，使模型在定位、改述定位和自我验证之间保持逻辑一致。

## 方法详解

### 整体框架

本文工作包含两个部分：(1) **一致性评估框架**——构建 Charades-CON 和 ActivityNet-CON 数据集，设计 4 种一致性探测（改述定位、偏移定位、整体验证、组合验证），评估 10 种模型；(2) **VTune 方法**——一种扩展指令微调的训练策略，将定位任务重新表述为验证过程，显式训练模型识别和纠正内容/时序变化。

### 关键设计

**1. 一致性评估探测 (Consistency Probes)**

- **功能**: 多维度检测模型时序理解的鲁棒性
- **核心思路**: 定义 4 种探测：(a) **Rephrased Grounding (R-Ground)**: 对查询语句改述后检查定位是否一致（测 IoU）；(b) **Shifted Grounding (S-Ground)**: 将视频内容时序偏移后检查模型能否跟随调整预测；(c) **Holistic Verification (H-Verify)**: 让模型验证查询是否在其预测的时段内（应答"Yes"），同时用不匹配查询测"No"；(d) **Compositional Verification (C-Verify)**: 验证模型能否确认查询中各组成元素（主语/动作/关系）
- **设计动机**: 单一 grounding 指标无法反映理解的可靠性。模型可能通过语言先验猜对时间戳但不真正理解内容，一致性探测能暴露这种"虚假理解"

**2. 数据集构建 (Charades-CON / ActivityNet-CON)**

- **功能**: 提供结构化的一致性评估数据
- **核心思路**: 从 Charades-STA 和 ActivityNet-Captions 各采样 500 个视频，用 GPT-4o-mini 生成对齐/不对齐/组合查询。对齐查询通过词替换、主被动转换、语序修改生成；不对齐查询对关键成分做细微修改。人工评估显示 92.2% 句子质量"Well-matched"
- **设计动机**: 需要高质量的改述和误导查询来客观测量一致性，而非依赖模型自身的理解

**3. Event Temporal Verification Tuning (VTune)**

- **功能**: 通过验证式指令微调显式提升模型的一致性
- **核心思路**: 超越简单的"预测时间戳"指令微调，VTune 训练模型完成三种验证任务：(a) **确认对齐**：确认改述查询与正确时段匹配；(b) **识别内容变化**：检测不对齐查询并指出哪些内容不匹配；(c) **识别时序变化**：检测时间偏移并纠正时间戳。在 Charades-STA 和 ActivityNet-Captions 训练集上构建验证式训练数据
- **设计动机**: 标准指令微调的 token likelihood 目标不直接优化一致性。VTune 通过显式要求模型处理对齐/不对齐变体，迫使模型建立查询与视觉内容之间的忠实映射

### 损失函数 / 训练策略

- 使用标准 LLM 训练目标（next token prediction）
- VTune 在原始 grounding 训练数据基础上增加验证任务数据
- 训练数据包含三类：grounding 指令 (G)、事件变化验证 (E)、时序变化验证 (T)
- 在 Video-LLaMA 和 TimeChat 两个模型上验证有效性

## 实验关键数据

### 主实验

10 种模型的一致性评估（Charades-CON, 相对一致性分数 %）：

| 模型 | Ground | R-Ground | S-Ground | H-Verify | C-Verify |
|------|--------|----------|----------|----------|----------|
| Video-LLaVA | 9.4 | 80.8% | 30.3% | 52.8% | 50.0% |
| TimeChat | 30.5 | 82.1% | 18.5% | 45.9% | 51.2% |
| VTimeLLM | 27.3 | 83.2% | 26.9% | 43.7% | 49.8% |
| GPT-4o | 28.5 | 74.3% | 32.8% | **62.4%** | **71.3%** |
| Gemini 1.5 | **34.6** | **85.7%** | **71.7%** | 65.8% | 70.8% |

### 消融实验

VTune vs 其他方法（TimeChat on Charades-CON）：

| 方法 | Ground | S-Ground | H-Verify | C-Verify |
|------|--------|----------|----------|----------|
| TimeChat (原始) | 30.5 | 5.6 | 14.0 | 15.6 |
| + CoT prompting | 28.7 | 7.1 | 13.5 | 14.4 |
| + Desc prompting | 33.3 | 7.3 | 19.9 | 20.6 |
| + Instruction Tuning | 55.8 | 10.5 | 16.7 | 25.7 |
| + **VTune** | **76.2** | **36.2** | **44.8** | **42.4** |

### 关键发现

1. **开源模型验证一致性接近随机水平**（约 50%）：模型能定位但不能可靠地验证自己的预测
2. **时序感知模型未能显著提升一致性**：VTimeLLM 比 Video-LLaMA grounding 高 7.3%，但验证一致性仅高 1.6%
3. **Shifted Grounding 暴露模型依赖语言先验**：大多数模型对视频时序偏移不敏感，预测几乎不变
4. **VTune 全面优于 prompting 和标准指令微调**：TimeChat 上 grounding 从 30.5→76.2，H-Verify 从 14.0→44.8
5. **闭源模型（GPT-4o, Gemini）一致性显著优于开源模型**，但仍有提升空间

## 亮点与洞察

1. **"一致性"视角切入时序理解评估**新颖且深刻：揭示了现有指标的盲区——高 grounding 分不等于真正理解
2. **S-Ground 实验的发现最有冲击力**：模型对视频内容时序偏移几乎不敏感，强烈暗示其依赖语言先验而非视觉理解
3. **VTune 的验证式重构设计**很有启发：将"预测+验证"统一到训练中，迫使模型建立双向理解

## 局限与展望

1. VTune 虽显著提升一致性，但绝对水平仍不高（H-Verify 约 55-60% 相对一致性）
2. 评估依赖 GPT-4o-mini 判断验证回答的正确性，引入自动化偏差
3. 仅在两个模型上验证 VTune，对更大规模模型的效果未知
4. 可探索将一致性直接纳入训练目标函数，而非仅通过数据增强

## 相关工作与启发

- **TimeChat / VTimeLLM**: 时序感知 Video-LLM，本文揭示其一致性不足
- **Chain-of-Thought prompting**: 在视频时序任务中改善不稳定，效果因模型而异
- **自然语言中 LLM 一致性研究**: 本文将一致性分析扩展到视频时序理解领域
- **DETR-style 时序定位模型**: 任务特定模型在一致性方面可能更稳定

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 从一致性角度评估 Video-LLM 时序理解是原创且重要的贡献
- **实验充分度**: ⭐⭐⭐⭐⭐ — 10 种模型、2 个数据集、4 种探测、多种解决方案对比、消融、人工验证
- **写作质量**: ⭐⭐⭐⭐ — 问题定义精准，实验分析深入
- **价值**: ⭐⭐⭐⭐⭐ — 对 Video-LLM 研究社区有重要警示意义，VTune 是可行的改进方向

<!-- RELATED:START -->

## 相关论文

- [Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [PAVE: Patching and Adapting Video Large Language Models](pave_patching_and_adapting_video_large_language_models.md)
- [VoCo-LLaMA: Towards Vision Compression with Large Language Models](voco-llama_towards_vision_compression_with_large_language_models.md)
- [Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](../../CVPR2026/video_understanding/understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)
- [Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Models](coarse_correspondences_boost_spatial-temporal_reasoning_in_multimodal_language_m.md)

<!-- RELATED:END -->
