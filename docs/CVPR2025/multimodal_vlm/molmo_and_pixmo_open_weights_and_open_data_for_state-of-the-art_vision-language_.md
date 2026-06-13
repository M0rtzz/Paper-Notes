---
title: >-
  [论文解读] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models
description: >-
  [CVPR 2025][多模态VLM][视觉语言模型] 本文提出Molmo系列VLM和PixMo数据集，完全不依赖闭源VLM的合成数据，通过创新的数据收集方式（语音描述图像、交互式问答标注、2D指向标注）从零构建高质量训练数据，其72B模型在学术基准和人类评估中超越Claude 3.5 Sonnet和Gemini 1.5 Pro，仅次于GPT-4o。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视觉语言模型"
  - "开放数据"
  - "指向能力"
  - "图像描述"
  - "多模态预训练"
---

# Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models

**会议**: CVPR 2025  
**arXiv**: [2409.17146](https://arxiv.org/abs/2409.17146)  
**代码**: [https://github.com/allenai/molmo](https://github.com/allenai/molmo)  
**领域**: 多模态VLM  
**关键词**: 视觉语言模型, 开放数据, 指向能力, 图像描述, 多模态预训练

## 一句话总结

本文提出Molmo系列VLM和PixMo数据集，完全不依赖闭源VLM的合成数据，通过创新的数据收集方式（语音描述图像、交互式问答标注、2D指向标注）从零构建高质量训练数据，其72B模型在学术基准和人类评估中超越Claude 3.5 Sonnet和Gemini 1.5 Pro，仅次于GPT-4o。

## 研究背景与动机

当前最强的VLM（GPT-4o、Gemini、Claude）均为闭源模型。开源模型要么性能落后（如早期LLaVA），要么严重依赖闭源模型生成的合成数据（如ShareGPT4V用GPT-4V生成caption），本质上是对闭源VLM的蒸馏。这导致学术界缺乏"如何从零构建高性能VLM"的基础知识。核心矛盾在于：高质量多模态数据的收集成本高昂，且众包标注质量难以保证。本文的切入角度是：通过创新的数据收集方法（模态转换trick——让标注者"说"而非"写"描述），以低成本获得高质量数据。核心idea：不靠蒸馏，靠高质量原创数据也能训出SOTA级VLM。

## 方法详解

### 整体框架

Molmo采用标准的VLM架构：预训练ViT图像编码器 + 视觉-语言连接器 + 预训练解码器LLM。训练分为两个阶段：(1) 在PixMo-Cap上预训练所有参数，学习详细图像描述能力；(2) 在PixMo数据集和开源学术数据集混合上微调，学习问答、指向、计数等多种能力。模型家族涵盖1B（MolmoE）到72B（基于Qwen2-72B）多个规模。

### 关键设计

1. **PixMo-Cap 语音描述数据收集**:
    - 功能：收集712k张图像的高质量详细描述（平均196词），作为预训练数据
    - 核心思路：让标注者对图像**语音描述60-90秒**，而非打字写caption。语音转文本后由纯文本LLM整理为高质量caption
    - 设计动机：直接让标注者打字写长caption效果差——他们倾向只关注几个显著元素，打长段耗时，且可能复制粘贴闭源VLM的输出。语音模态的"模态转换trick"让标注者自然提供更丰富的细节，且留有音频"收据"证明未使用VLM

2. **PixMo-Points 2D指向数据**:
    - 功能：收集2.3M个指向标注，支持模型通过指向像素来定位物体、计数和解释答案
    - 核心思路：标注者在图像中指出某类物体的所有实例（用点而非框），模型学习输出归一化坐标 $x, y \in [0, 100]$。计数时使用chain-of-thought方式——逐个指出每个目标，最后报总数
    - 设计动机：点标注比边界框和分割掩码快得多，且能更自然地支持计数（通过逐一指点）和视觉解释。未来可赋能VLM驱动的agent（机器人导航、UI操作等）

3. **重叠多裁剪策略（Overlapping Multi-Crop）**:
    - 功能：处理高分辨率图像，同时保证每个patch有邻域上下文
    - 核心思路：将图像分成多个正方形裁剪块（crop），各crop之间允许重叠，但只取非重叠部分的patch特征送入LLM。训练用12个crop，推理用36个
    - 设计动机：标准tiling策略中边界patch缺失邻域上下文（如自行车品牌名被裁断），重叠解决了这个问题。消融实验显示重叠将11-benchmark均值从75.7提升到76.9

### 损失函数 / 训练策略

- **两阶段简化**：跳过常见的"先冻结LLM只训连接器"的stage，直接端到端预训练所有参数。连接器使用更高学习率（2e-4 vs LLM的2e-5）和更短warmup（200步 vs 2000步）来补偿
- **文本专用Dropout**：预训练阶段仅对文本token施加残差dropout，不对图像token施加，迫使模型更依赖视觉输入而非语言先验
- **多标注高效训练**：同一图像的多个标注拼成一个长序列，用注意力掩码隔离不同标注，避免重复编码图像。这减少了2/3的图像处理量，训练时间减半以上
- **风格标签（Style Tag）**：对学术数据集加前缀标签（如"vqa2:"），让模型学会仅在评测时使用短答案风格，用户交互时保持自然风格

## 实验关键数据

### 主实验

| 数据集 | 指标 | Molmo-72B | GPT-4o | 提升 |
|--------|------|-----------|--------|------|
| AI2D | Acc | 96.3 | 94.2 | +2.1 |
| ChartQA | Acc | 87.3 | 85.7 | +1.6 |
| VQA v2.0 | Acc | 86.5 | 78.7 | +7.8 |
| DocVQA | Acc | 93.5 | 92.8 | +0.7 |
| TextVQA | Acc | 83.1 | 77.4 | +5.7 |
| RealWorldQA | Acc | 75.2 | 75.4 | -0.2 |
| MMMU | Acc | 54.1 | 69.1 | -15.0 |
| CountBenchQA | Acc | 91.2 | 87.9 | +3.3 |
| PixMo-Count | Acc | 85.2 | 59.6 | +25.6 |
| **11-avg** | | **81.2** | **78.5** | **+2.7** |

Molmo-72B在Elo人类评估中排名第2（1077），仅次于GPT-4o（1079）。

### 消融实验

| 配置 | cap F1 | 11-avg | 说明 |
|------|--------|--------|------|
| 单裁剪 vs 多裁剪+重叠 | 46.7→54.1 | 62.8→76.9 | 多裁剪+重叠大幅提升 |
| 无dropout vs 文本专用dropout | 53.0→54.1 | 76.2→76.9 | 文本专用dropout改善描述能力 |
| Pooling: 拼接 vs 注意力 | 53.7→54.1 | 76.1→76.9 | 注意力池化优于简单拼接 |
| ViT层: 单层 vs 双层拼接 | - | - | 第3倒数+第10倒数层拼接最优 |
| 数据量: 25% vs 100% PixMo-Cap | 51.6→54.1 | 76.3→76.9 | 更多数据持续提升 |

### 关键发现

- 不使用任何闭源VLM合成数据，仅靠高质量原创数据也能达到SOTA级性能
- 语音描述比文字描述产生更详细的caption（196词 vs COCO的11词）
- 指向能力使计数准确率大幅领先所有模型（PixMo-Count上85.2 vs GPT-4o的59.6）
- MolmoE-1B（混合专家模型）以极低参数量接近GPT-4V性能

## 亮点与洞察

- **数据收集创新**：语音描述的"模态转换"trick是一个极其巧妙的低成本高质量方案，既解决了标注者偷懒问题，又提供了音频防作弊凭证
- **指向即解释**：用2D点代替bounding box做视觉定位，大幅降低标注成本的同时开启了VLM作为agent的指向交互范式
- **完全开放的高性能VLM**：在"开放权重+开放数据"类别中遥遥领先，证明了不蒸馏闭源模型也能做到顶级性能

## 局限与展望

- 推理能力（MMMU、MathVista）显著弱于GPT-4o和Qwen系列，训练数据缺乏高级推理相关数据
- 指向能力在改变crop数量时泛化不佳，需要额外高分辨率后训练
- 语音描述方案依赖英语标注者，多语言扩展需要额外考虑
- 模型架构本身无创新，核心贡献在数据侧

## 相关工作与启发

- **vs LLaVA-OneVision**: LLaVA系列依赖ShareGPT4V等蒸馏数据，Molmo完全自建数据且性能更强
- **vs Qwen2-VL**: Qwen2-VL在OCR相关任务略胜，但在人类评估中远逊于Molmo（Elo 1037 vs 1077）
- **vs PaliGemma**: 同为开放模型，PaliGemma性能远低于Molmo，说明数据质量比架构更关键

## 评分

- 新颖性: ⭐⭐⭐⭐ 架构无创新但数据收集方法（语音描述+指向标注）非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 11个学术基准+325k人类评估+大量消融
- 写作质量: ⭐⭐⭐⭐⭐ 详尽清晰，实验细节和消融分析极其充分
- 价值: ⭐⭐⭐⭐⭐ 为开放VLM社区提供了从零构建SOTA模型的完整路线图

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Compositional Caching for Training-free Open-vocabulary Attribute Detection](compositional_caching_for_training-free_open-vocabulary_attribute_detection.md)
- [\[AAAI 2026\] O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model](../../AAAI2026/multimodal_vlm/o3slm_open_weight_open_data_and_open_vocabulary_sketch-language_model.md)
- [\[CVPR 2025\] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)
- [\[CVPR 2025\] RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness](rlaif-v_open-source_ai_feedback_leads_to_super_gpt-4v_trustworthiness.md)
- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)

</div>

<!-- RELATED:END -->
