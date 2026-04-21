---
title: >-
  [论文解读] AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization
description: >-
  [ICLR 2026][LLM对齐][多模态情感理解] 针对多模态大语言模型在情感推理中的虚假关联和幻觉问题，提出 EmoReAlM 评测基准和 AVEm-DPO 偏好优化方法，通过构建针对性偏好对和文本先验正则化，在 DFEW/RAVDESS/EMER 上实现 6-19% 的零样本相对性能提升。
tags:
  - ICLR 2026
  - LLM对齐
  - 多模态情感理解
  - 偏好优化
  - DPO
  - 幻觉缓解
  - 视听推理
---

# AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization

**会议**: ICLR 2026  
**arXiv**: [2602.07054](https://arxiv.org/abs/2602.07054)  
**代码**: https://github.com/ihp-lab/AVERE (有)  
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
整个工作包含两大贡献：一个评测基准 EmoReAlM 和一个偏好优化方法 AVEm-DPO。

### 整体框架
AVEm-DPO 建立在 DPO（Direct Preference Optimization）框架之上，但在偏好对的构造和损失函数设计上做了情感专属的创新。整个流程是：(1) 分析 MLLM 在情感任务上的失败模式→(2) 设计 EmoReAlM 评测量化问题→(3) 构建针对性偏好数据→(4) 加入文本先验正则化进行对齐训练。

### 关键设计
1. **EmoReAlM Benchmark**: 专为评测 MLLM 情感推理能力设计的基准，包含四类任务：(a) 基础推理（Reasoning Basic）——评估模型是否通过正确的视听线索进行情感判断；(b) 压力测试（Stress Test）——辨别模型是否会幻觉出不存在的线索；(c) 模态一致性（Modality Agreement）——判断视觉和听觉线索是否真正一致；(d) 无幻觉检测——验证模型是否正确识别真实存在的线索。设计动机是现有评测无法区分"答对了但理由错误"的情况。
   
2. **偏好对构造策略**: AVEm-DPO 构造两类偏好对。第一类针对响应：将展示虚假关联或幻觉的响应作为 rejected，正确响应作为 chosen。第二类针对输入：通过文本引导构造不同的视听输入对，让模型学习区分哪些视听线索真正与情感相关。这种双层偏好构造是该方法的核心创新。

3. **文本先验正则化**: 额外加入正则化项惩罚模型对文本先验的过度依赖。当模型在没有对应视听证据的情况下仍生成与某种情感相关的描述时，该正则化会施加惩罚。这直接针对了幻觉的根源——语言模型骨干中学到的文本偏见。

### 损失函数 / 训练策略
在标准 DPO 损失基础上增加文本先验惩罚项。训练采用零样本设置，在 DFEW、RAVDESS、EMER 等情感数据集上评估。
具体来说，偏好对的构建过程中，先用基线模型生成多个响应，然后用 EmoReAlM 的评估维度自动筛选出展现虚假关联或幻觉的响应作为 rejected。对于输入级偏好对，通过替换视频中的音频/视觉模态来构造不匹配的输入对。
正则化项的形式为对仅依据文本先验就能生成的视觉/音频描述施加额外惩罚，鼓励模型依据真实的多模态输入来进行判断。
训练时使用了两个骨干网络验证方法的泛化性：Our base（基于VITA-1.5架构）和 EmotionLLaMA（情感专用微调模型）。

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

## 相关论文

- [ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](../../ACL2025/llm_alignment/aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)
- [LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](../../NeurIPS2025/llm_alignment/longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)
- [Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](dual-ipo_dual-iterative_preference_optimization_for_text-to-video_generation.md)
- [Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](../../ACL2025/llm_alignment/probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization](alignment_through_meta-weighted_online_sampling_bridging_the_gap_between_data_ge.md)

<!-- RELATED:END -->
