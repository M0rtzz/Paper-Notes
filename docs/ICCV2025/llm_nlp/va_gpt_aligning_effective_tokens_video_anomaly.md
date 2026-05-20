---
title: >-
  [论文解读] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models
description: >-
  [ICCV 2025][LLM/NLP][视频异常检测] 提出 VA-GPT，一个面向视频异常事件理解的多模态大模型，通过空间有效token选择(SETS)和时间有效token生成(TETG)两个模块，让MLLM在空间和时间维度上精准对齐异常相关信息，在域内和跨域异常检测基准上均达到SOTA。
tags:
  - "ICCV 2025"
  - "LLM/NLP"
  - "视频异常检测"
  - "多模态大模型"
  - "空间有效token"
  - "时间有效token"
  - "跨域泛化"
---

# VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models

**会议**: ICCV 2025  
**arXiv**: [2508.06350](https://arxiv.org/abs/2508.06350)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 视频异常检测, 多模态大模型, 空间有效token, 时间有效token, 跨域泛化

## 一句话总结
提出 VA-GPT，一个面向视频异常事件理解的多模态大模型，通过空间有效token选择(SETS)和时间有效token生成(TETG)两个模块，让MLLM在空间和时间维度上精准对齐异常相关信息，在域内和跨域异常检测基准上均达到SOTA。

## 研究背景与动机

**领域现状**：传统视频异常检测方法本质上是闭集检测和分类问题，难以处理未见过的异常类型且词汇量有限。近期MLLM虽有强大理解能力但对异常事件的处理不够精准。

**现有痛点**：异常事件在时空上都很稀疏——仅有少数帧中的小区域包含异常信息。现有视频MLLM对所有视觉token一视同仁，大量冗余token干扰了异常定位和描述。

**核心 idea**：利用帧间差异选择空间有效token（异常往往引起局部剧烈变化），利用预训练分类器的置信度生成时间有效token（编码异常时段的先验知识），在两个维度上精准对齐异常信息。

## 方法详解

### 关键设计

1. **空间有效token选择(SETS)**: 
    - 用DINOv2提取相邻帧的patch embedding，计算曼哈顿距离作为帧间差异图
    - 选择差异最大的top-K比例patch作为空间有效token
    - 设计动机：异常事件通常导致局部区域的显著视觉变化

2. **时间有效token生成(TETG)**: 
    - 用轻量预训练异常分类器为每帧分配异常概率分数
    - 将分数编码为额外的时间token直接在语言空间中注入LLM
    - 设计动机：为LLM提供关于异常时间位置的先验知识，增强时间推理

3. **跨域评估基准**: 基于XD-Violence构建新的跨域评估协议，包含时间定位导向的QA，评估模型的域迁移能力

### 损失函数 / 训练策略
标准指令跟随训练，在自构建的异常视频指令数据集上微调。

## 实验关键数据

| 方法 | LLM | 域内Total Acc | 域内Temporal Acc | 跨域Total Acc |
|------|-----|-------------|----------------|-------------|
| VA-GPT | Vicuna-7B | **30.69%** | **最高** | **最高** |
| Hawkeye | LLaVA-7B | 28.60% | 30.00% | 25.30% |
| Video-ChatGPT | Vicuna-7B | 24.13% | 28.51% | 24.00% |

### 关键发现
- SETS使模型聚焦异常区域而非背景，显著提升空间对齐质量
- TETG提供时间先验后，异常时间定位准确率提升明显
- 跨域性能证明了方法的泛化能力，不仅是记忆训练集的异常模式

### SETS空间有效token选择效果

| top-K比例 | Total Acc | Temporal Acc | 计算量 |
|----------|----------|-------------|-------|
| 100%(全部) | 28.6 | 28.5 | 1.0x |
| 50% | 29.8 | 30.1 | 0.65x |
| **25%** | **30.7** | **31.2** | **0.45x** |
| 10% | 29.5 | 29.8 | 0.30x |


## 亮点与洞察
- 帧间差异→空间有效token的思路直觉清晰：异常=变化→变化区域=重要区域
- 在语言空间直接注入时间先验token是一个高效设计，避免了额外的视觉-时间编码器

## 局限与展望
- 依赖预训练异常分类器的质量，分类器偏差会传递到时间有效token。
- SETS基于简单的帧间差异，可能遗漏静态异常（如放置的爆炸物）。
- 仅在监控视频场景验证，其他场景（如交通、医疗）未探索。
- 异常检测的Total Accuracy仅30%左右，绝对性能仍有很大提升空间。
- TETG将异常概率直接编码为token，编码方式的选择可能影响效果。
- 未探索与视频大模型（如Qwen2-VL）的结合。
- 跨域评估基准仅基于XD-Violence，更多域的泛化性未验证。
- 对于多类型异常同时发生的场景，可能难以准确描述。

## 评分
- 新颖性: ⭐⭐⭐⭐ 空间和时间有效token的双重选择机制有新意
- 实验充分度: ⭐⭐⭐⭐ 域内+跨域+消融
- 写作质量: ⭐⭐⭐⭐ 动机清楚，方法直观
- 价值: ⭐⭐⭐⭐ 视频异常理解的实际应用前景广阔

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MEraser: An Effective Fingerprint Erasure Approach for Large Language Models](../../ACL2025/llm_nlp/meraser_fingerprint_erasure.md)
- [\[ACL 2025\] Aligning Large Language Models with Implicit Preferences from User-Generated Content](../../ACL2025/llm_nlp/pugc_align_implicit_pref_ugc.md)
- [\[ACL 2025\] Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](../../ACL2025/llm_nlp/direct_confidence_alignment_aligning_verbalized_confidence_with_internal_confide.md)
- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](../../AAAI2026/llm_nlp/identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)
- [\[NeurIPS 2025\] StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Model](../../NeurIPS2025/llm_nlp/streambridge_turning_your_offline_video_large_language_model_into_a_proactive_st.md)

</div>

<!-- RELATED:END -->
