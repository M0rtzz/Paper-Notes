---
title: >-
  [论文解读] MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark
description: >-
  [ACL 2025][多模态][多模态基准] 本文引入MMMU-Pro，通过过滤纯文本可解的题目、将选项从4个增加到10个、引入"纯视觉输入"设置三步增强了MMMU基准的鲁棒性，导致模型性能下降16.8%~26.9%，更准确地反映了多模态模型的真实理解能力。
tags:
  - ACL 2025
  - 多模态
  - 多模态基准
  - MMMU
  - 视觉推理
  - VLM评估
  - CoT推理
---

# MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark

**会议**: ACL 2025  
**arXiv**: [2409.02813](https://arxiv.org/abs/2409.02813)  
**代码**: [排行榜](https://mmmu-benchmark.github.io/#leaderboard) (有)  
**领域**: 多模态 / VLM评估  
**关键词**: 多模态基准, MMMU, 视觉推理, VLM评估, CoT推理

## 一句话总结

本文引入MMMU-Pro，通过过滤纯文本可解的题目、将选项从4个增加到10个、引入"纯视觉输入"设置三步增强了MMMU基准的鲁棒性，导致模型性能下降16.8%~26.9%，更准确地反映了多模态模型的真实理解能力。

## 研究背景与动机

**领域现状**: MMMU是评估多模态大模型在大学级别多学科理解与推理能力的标准benchmark，覆盖30个学科11.5K题目。GPT-4o等模型已在MMMU上达到69.1%，但这些分数是否真正反映了深层多模态理解仍存疑。

**现有痛点**: (1) 部分MMMU题目可被纯文本LLM正确回答（无需图像）；(2) 4选1的选项空间太小，模型可通过猜测或选项间的线索推断答案；(3) 当前评估未测试模型"看"与"读"的整合能力。

**核心矛盾**: 现有benchmark分数可能高估了模型的真实多模态理解能力——模型可能在利用文本捷径而非真正理解图像内容。

**本文目标**: 构建更鲁棒的多模态评估基准，过滤捷径、增加难度、引入更贴近真实使用场景的评估设置。

**切入角度**: 三步增强流程——过滤、扩选项、纯视觉输入——系统性地堵住现有benchmark的漏洞。

**核心 idea**: 通过过滤文本可解题目、10选项增难、以及将题目嵌入截图的纯视觉设置，MMMU-Pro暴露了当前多模态模型在真正"看+读"整合能力上的巨大差距。

## 方法详解

### 整体框架

基于MMMU的三步构造流程：(1) 用4个强开源LLM过滤掉纯文本可解的题目；(2) 人工+GPT-4o将选项从4个扩展到10个，增加猜测难度；(3) 引入纯视觉输入设置，将题目嵌入截图/照片中呈现给模型。最终获得3460道题目（标准1730+视觉1730）。

### 关键设计

1. **文本可解题目过滤**
    - 功能：去除不需要图像就能回答的题目
    - 核心思路：用Llama3-70B、Qwen2-72B、Yi-1.5-34B、Mixtral-8×22B四个纯文本LLM各回答MMMU题目10次，若≥3个模型在>5次中答对则过滤
    - 设计动机：确保保留的题目真正需要视觉信息，消除文本捷径

2. **选项扩展至10个**
    - 功能：减少猜测成功的概率
    - 核心思路：GPT-4o生成候选选项，Claude 3.5过滤，两轮人工审核确保质量和多样性。同时再次过滤与图像无关的题目（去掉70题）
    - 设计动机：4选1有25%随机基线，10选1降至10%，迫使模型真正理解题意

3. **纯视觉输入设置**
    - 功能：测试模型整合视觉和文本信息的核心认知能力
    - 核心思路：人工标注者在多样的显示环境下截屏/拍照，变化背景、字体、字号，题目和选项全部以图像形式呈现
    - 设计动机：模拟真实用户场景（用户常截屏提问），挑战模型同时"看"文字和图像的能力

### 损失函数 / 训练策略

不涉及训练。本文是benchmark构造工作，评估使用Direct和CoT两种提示方式，报告较高分数。

## 实验关键数据

### 主实验

| 模型 | Standard(4选项) | Standard(10选项) | Vision | MMMU(Val) | Δ₁(10选项-MMMU) |
|------|---------------|-----------------|--------|-----------|----------------|
| GPT-4o (0513) | 64.7 | **54.0** | 49.7 | 69.1 | -15.1 |
| Claude 3.5 Sonnet | 63.7 | 55.0 | 48.0 | 68.3 | -13.3 |
| Gemini 1.5 Pro | 60.6 | 49.4 | 44.4 | 65.8 | -16.4 |
| InternVL2-76B | 55.0 | 41.9 | 38.0 | 58.3 | -16.4 |
| LLaVA-OneVision-72B | 52.3 | 38.0 | 24.0 | 56.8 | -18.8 |
| 人类专家(High) | 88.6 | 85.4 | 85.4 | 88.6 | -3.2 |
| 随机选择 | 24.9 | 12.8 | 12.4 | 22.1 | -9.3 |

### 消融实验

| 实验条件 | 结论 |
|---------|------|
| CoT vs Direct (Claude 3.5) | Standard: 42.7%→55.0% (+12.3%)，CoT显著提升 |
| CoT vs Direct (VILA-1.5) | 显著下降，因指令遵循能力不足导致格式错误 |
| OCR提示 vs 无OCR (Vision) | GPT-4o: 49.7% vs 49.4%，OCR提示几乎无影响 |
| CoT按学科 (GPT-4o) | 工程: +14.49%, 科学: +8.22%, 艺术设计: +1.58% |

### 关键发现

- 从MMMU到MMMU-Pro的性能下降幅度为16.8%~26.9%，表明现有模型在MMMU上的分数被高估
- 人类专家从MMMU到MMMU-Pro仅下降1.8~3.2%，而模型下降15-27%，差距被显著放大
- Vision设置比Standard(10选项)再降4-14%，开源模型（如LLaVA-OneVision-72B降14%）在文字图像整合上尤其薄弱
- OCR提示几乎无帮助，说明瓶颈不在文字提取而在于图像中文字和视觉元素的语义整合
- CoT在推理密集型学科（工程、科学）帮助最大，在主观性学科（艺术设计）效果有限甚至负面

## 亮点与洞察

- **纯视觉输入设置的创新性**：将题目嵌入截图测试"看+读"整合能力，贴近真实用户行为，暴露了模型在图文交织场景下的根本弱点
- **人机差距的放大镜**：MMMU上人机差距约20%，MMMU-Pro上扩大到30-40%，更真实地反映了多模态AI的现状
- **OCR不是瓶颈的发现**：强模型已具备良好OCR能力（>85%），Vision设置的难度来源于需要理解文字在视觉上下文中的语义关系

## 局限与展望

- 人类专家性能为基于原MMMU数据的近似估计，未在MMMU-Pro上直接评估
- 10选项的扩展引入了GPT-4o/Claude的偏见，可能影响选项质量
- 纯视觉输入的截图由人工拍摄而非程序生成，可能引入非预期的视觉线索（如背景颜色）
- 未包含自由响应题型（仅多选），不反映模型的开放式推理能力
- 学科分布可能不均匀，某些学科的题目可能过于依赖领域知识而非视觉理解

## 相关工作与启发

- **MMMU (Yue et al. 2024)**: MMMU-Pro的基础benchmark，11.5K大学级多学科多模态题目
- **Du et al. (2023) / Yuksekgonul et al. (2023)**: 揭示多模态模型利用捷径的研究，本文借鉴其发现系统性地构建更鲁棒的benchmark
- 启发：benchmark设计应该持续迭代升级以追赶模型能力的提升，纯视觉输入设置可以推广到其他多模态benchmark

## 评分

- **新颖性**: ⭐⭐⭐⭐（纯视觉输入设置是亮点，三步增强方法论有推广价值）
- **实验充分度**: ⭐⭐⭐⭐⭐（20+模型评估，CoT/OCR/学科维度全面分析）
- **写作质量**: ⭐⭐⭐⭐⭐（图表清晰，层次分明，动机阐述convincing）
- **价值**: ⭐⭐⭐⭐⭐（已成为多模态模型评估的标准benchmark之一，影响力大）

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating](longdocurl_multimodal_long_doc.md)
- [\[ACL 2025\] Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder](exploring_how_generative_mllms_perceive_more.md)
- [\[ACL 2025\] CoSyn: Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](cosyn_code_guided_synthetic_data.md)
- [\[ACL 2025\] CrafText Benchmark: Advancing Instruction Following in Complex Multimodal Open-Ended World](craftext_benchmark_advancing_instruction_following_in_complex_multimodal_open-en.md)
- [\[ACL 2025\] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)

<!-- RELATED:END -->
