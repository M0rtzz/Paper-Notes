---
title: >-
  [论文解读] AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization
description: >-
  [ICLR 2026][音频/语音][多模态情感理解] 针对多模态大语言模型在情感推理中的虚假关联和幻觉问题，提出 EmoReAlM 评测基准和 AVEm-DPO 偏好优化方法，通过构建针对性偏好对和文本先验正则化，在 DFEW/RAVDESS/EMER 上实现 6-19% 的零样本相对性能提升。
tags:
  - "ICLR 2026"
  - "音频/语音"
  - "多模态情感理解"
  - "偏好优化"
  - "DPO"
  - "幻觉缓解"
  - "视听推理"
---

# AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization

**会议**: ICLR 2026  
**arXiv**: [2602.07054](https://arxiv.org/abs/2602.07054)  
**代码**: [https://github.com/ihp-lab/AVERE](https://github.com/ihp-lab/AVERE)  
**领域**: 对齐RLHF  
**关键词**: 多模态情感理解, 偏好优化, DPO, 幻觉缓解, 视听推理

## 一句话总结
针对多模态大语言模型在情感推理中的虚假关联和幻觉问题，提出 EmoReAlM 评测基准和 AVEm-DPO 偏好优化方法，通过构建针对性偏好对和文本先验正则化，在 DFEW/RAVDESS/EMER 上实现 6-19% 的零样本相对性能提升。

## 研究背景与动机
情感理解是构建社会智能体的核心能力之一。近年来多模态大语言模型（MLLM）在情感识别任务上取得了显著进展，但仍存在两大关键挑战：

**挑战一：虚假关联（Spurious Associations）**。模型常将情感与无关的视听线索错误地关联，例如将画面中的黄色高领衫与"快乐"情绪挂钩，而非关注面部表情。这属于推理层面的错误。

**挑战二：幻觉（Hallucinations）**。语言模型骨干的文本先验驱动模型"编造"视听线索，比如声称视频中有"紧握拳头"来支撑"愤怒"判断，但实际画面中并不存在该动作。这属于感知层面的错误。

现有的多模态偏好优化方法（如 Vista-DPO）主要面向视频理解的通用场景，并未针对情感理解中的特殊问题进行设计。同时，缺少专门的评测工具来系统性地量化 MLLM 在情感场景下的虚假关联和幻觉现象。

**核心idea**：同时构建评测基准（EmoReAlM）和对齐方法（AVEm-DPO），引入针对虚假关联和幻觉的偏好对构造策略，并加入文本先验正则化，从根源上对齐模型的视听感知与情感推理能力。

## 方法详解

### 整体框架
这项工作两条腿走路：先用 EmoReAlM 基准把 MLLM 在情感任务上的失败模式量化出来，再用 AVEm-DPO 在 DPO 框架上做情感专属的偏好对齐。后者的关键不在算法本身，而在于针对虚假关联和幻觉精心构造的偏好数据，以及一个压制文本先验的正则项。

### 关键设计

**1. EmoReAlM 评测基准：区分"答对了但理由错了"。** 传统情感识别只看最终标签是否正确，无法暴露模型其实是靠无关线索或编造证据蒙对的。EmoReAlM 把情感推理拆成四类任务来逐项体检：基础推理（Reasoning Basic）查模型是否真的依据正确的视听线索做判断；压力测试（Stress Test）专门探模型会不会幻觉出画面里根本不存在的线索；模态一致性（Modality Agreement）检验它能否分辨视觉与听觉线索是否真正吻合；无幻觉检测则反向确认模型对真实存在的线索没有漏判。四类任务合起来才能把"虚假关联"（推理错）和"幻觉"（感知错）这两种隐性失败分别钉死。

**2. 双层偏好对构造：让 chosen/rejected 直击情感专属的错误。** AVEm-DPO 之所以比通用 DPO 有效，核心就在于偏好对怎么造。第一层是响应级：用基线模型对同一视频生成多个回答，再借 EmoReAlM 的评测维度自动筛出那些表现出虚假关联或幻觉的回答当 rejected，把依据真实线索的正确回答当 chosen，逼模型在"对的理由"和"错的理由"之间做对比。第二层是输入级：通过替换视频里的音频或视觉模态构造出不匹配的视听输入对，让模型学会判断到底哪条模态线索才真正与当前情感相关。响应级管"怎么说对"，输入级管"看哪里才对"，两层叠加才把对齐做细。

**3. 文本先验正则化：从根上掐断幻觉。** 幻觉的病根在语言模型骨干自带的文本偏见——它会在没有任何视听证据时，仅凭"愤怒常伴随紧握拳头"这类文本常识就编造出对应的视觉描述。针对这一点，AVEm-DPO 在 DPO 损失上额外加一个正则项，专门惩罚那些仅靠文本先验、缺乏真实多模态支撑就能生成的视觉/音频描述，从而把模型的输出拉回到真实输入上。这是个轻量但直接命中根源的设计。

### 损失函数 / 训练策略
总损失是标准 DPO 项加上上述文本先验惩罚项。整套流程在零样本设置下评估，不在目标数据集上做任何微调。为验证泛化性，方法在两个骨干上分别训练：Our base（基于 VITA-1.5 架构）和情感专用微调的 EmotionLLaMA，最终在 DFEW、RAVDESS、EMER 等数据集上测试。

## 实验关键数据

### 主实验

| 数据集 | 指标 | AVEm-DPO (Our base) | Naive-DPO | Vista-DPO | Base | 提升(vs Base) |
|--------|------|---------------------|-----------|-----------|------|---------------|
| DFEW   | WAR  | 58.54               | 55.67     | 56.42     | 56.78| +3.1%         |
| DFEW   | UAR  | 64.24               | 59.90     | 62.33     | 60.14| +6.8%         |
| RAVDESS| WAR  | 58.66               | 53.63     | 56.94     | 53.59| +9.5%         |
| EmoReAlM|Avg  | 83.3                | 68.1      | 76.9      | 65.1 | +28.0%        |

### 消融实验

| 配置 | EmoReAlM Avg | 说明 |
|------|-------------|------|
| Our base | 65.1 | 无偏好优化 |
| + Naive-DPO | 68.1 | 普通DPO,改善有限 |
| + Vista-DPO | 76.9 | 视频通用DPO |
| + AVEm-DPO | 83.3 | 情感专属设计,效果最佳 |

### 关键发现
- AVEm-DPO 在 EmoReAlM 上甚至超过了闭源 Gemini 2.5 Pro（70.3→83.3），说明针对性对齐极为有效
- 对 EmotionLLaMA 骨干同样有效，说明方法具有通用性
- 在 Stress Test（幻觉检测）子任务上提升最为显著（51.4→68.9），验证了文本先验正则化的作用
- 模态一致性任务上从 66.4 提升到 94.6，说明模型学会了真正利用跨模态信息

## 亮点与洞察
- 首个专门面向多模态情感推理的偏好优化方法，切入角度非常精准
- EmoReAlM 基准设计巧妙，四类任务全面剖析MLLM的情感推理弱点
- 双层偏好对构造（响应级+输入级）是一个值得借鉴的通用范式，可推广到其他需要细粒度对齐的多模态任务
- 文本先验正则化是缓解MLLM幻觉的一个轻量级但有效的方案
- Leaderboard 显示 AVEm-DPO 甚至能让开源模型在情感理解上胜过闭源Gemini 2.5 Pro
- 定性示例清楚展示了 AVEm-DPO 如何帮助模型聚焦真实的面部表情和语音语调，而非编造不存在的视觉线索

## 局限与展望
- 代码/模型虽已承诺公开但尚在准备中（HuggingFace checkpoint已发布：chaubeyG/AVERE-7B）
- 评测集规模和情感类别覆盖度可进一步扩展，当前主要关注基础情绪
- 仅在零样本设置下评估，少样本和微调设置值得探索
- 文本先验正则化的强度需要手动调节，自适应策略可改进
- 基准的音频线索评估依赖模型的音频理解能力，这本身就是一个挑战
- 偏好对构造的自动化程度可以进一步提升，当前仍需一定人工设计

## 相关工作与启发
- **vs Vista-DPO**: Vista-DPO是通用视频DPO，不针对情感场景设计，AVEm-DPO专门针对虚假关联和幻觉构造偏好对
- **vs EmotionLLaMA**: EmotionLLaMA通过情感数据微调，但仍有幻觉问题；AVEm-DPO在其基础上进一步对齐，两者是互补关系
- **vs Qwen 2.5 Omni**: 闭源模型在通用视听理解上强大，但在情感专项任务上仍不如AVEm-DPO
- **vs Naive-DPO**: 直接用通用DPO做偏好优化，效果有限（+3%），说明偏好对的质量比算法本身更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ 评测基准+对齐方法双重贡献，偏好对构造思路新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多骨干验证，消融完整
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，逻辑链完整
- 价值: ⭐⭐⭐⭐ 对多模态情感AI领域有直接推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)
- [\[ACL 2026\] Data-efficient Targeted Token-level Preference Optimization for LLM-based Text-to-Speech](../../ACL2026/audio_speech/data-efficient_targeted_token-level_preference_optimization_for_llm-based_text-t.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](../../AAAI2026/audio_speech/improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[ICLR 2026\] Improving Black-Box Generative Attacks via Generator Semantic Consistency](improving_black-box_generative_attacks_via_generator_semantic_consistency.md)
- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)

</div>

<!-- RELATED:END -->
