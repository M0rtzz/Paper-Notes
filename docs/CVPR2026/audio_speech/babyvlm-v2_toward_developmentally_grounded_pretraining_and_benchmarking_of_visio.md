---
title: >-
  [论文解读] BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models
description: >-
  [CVPR 2026][音频/语音][发育认知] 提出BabyVLM-V2框架，从婴儿第一视角的SAYCam纵向语料构建三种格式预训练数据（768K图像对+181K视频对+63K交错序列），设计基于NIH Baby Toolbox®的DevCV Toolbox（10个发育认知任务）…
tags:
  - "CVPR 2026"
  - "音频/语音"
  - "发育认知"
  - "婴儿视觉"
  - "样本效率预训练"
  - "NIH Baby Toolbox"
  - "DevCV Toolbox"
---

# BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models

**会议**: CVPR 2026  
**arXiv**: [2512.10932](https://arxiv.org/abs/2512.10932)  
**代码**: [https://shawnking98.github.io/BabyVLM-v2/](https://shawnking98.github.io/BabyVLM-v2/)  
**领域**: 音频语音  
**关键词**: 发育认知, 婴儿视觉, 样本效率预训练, NIH Baby Toolbox, DevCV Toolbox

## 一句话总结
提出BabyVLM-V2框架，从婴儿第一视角的SAYCam纵向语料构建三种格式预训练数据（768K图像对+181K视频对+63K交错序列），设计基于NIH Baby Toolbox®的DevCV Toolbox（10个发育认知任务），从零训练的紧凑模型在部分数学任务上超越GPT-4o，首次系统探索人工发育智能(ADI)。

## 研究背景与动机

**领域现状**：视觉基础模型依赖scaling law在海量数据上预训练，但早期儿童能从极其有限的视觉输入（出生到3岁约4万小时清醒时间）中发展出强大的感知和推理能力。这构成了样本效率预训练的自然目标。

**现有痛点**：BabyVLM-V1（前作）存在四大不足——(1) 仅用SAYCam约1/3录像(67K图像对)，覆盖极小比例；(2) 仅支持图像-文本对，不支持视频和多轮对话；(3) 4个评测任务是直觉设计而非基于标准化心理学测试；(4) 模型开放集性能接近零，需对logits后处理才能评估。

**核心矛盾**：如何在婴儿有限的感官体验约束下，训练出像早期儿童一样能力多样的基础模型？如何用发育心理学标准公正评估？

**切入角度**：(1) 最大化SAYCam语料利用率并构建多格式数据支持多样化下游任务；(2) 使用2025年2月NIH发布的Baby Toolbox®——目前最权威的儿童神经发育评估工具——作为benchmark设计基础。

**核心idea**：将发育心理学标准化评估方法工程化为AI评测的计算机视觉任务，建立DevCV Toolbox。

## 方法详解

### 整体框架
SAYCam婴儿纵向录像(478小时) → 最小化处理构建三类预训练数据 → 三阶段预训练(视觉编码→对齐→多格式) → 指令微调(113K样本) → DevCV Toolbox评测(10个认知任务)。

### 关键设计

1. **预训练数据构建（最小化处理的发育真实性）**:

    - **视频-语句对(181K)**：按语音转录边界切分视频，Azure语音识别提取字幕，X-CLIP相似度>0.1过滤，保留138小时
    - **图像-语句对(768K)**：从视频对中1FPS采样，CLIP相似度>0.2保留。相比V1的67K扩大11倍
    - **交错图文序列(63K)**：滑动窗口(大小4-8)组合连续片段的最佳帧+语句对，模拟婴儿连续交互经验
    - 设计动机：三种格式分别支持视频理解、图像理解和多轮对话，覆盖DevCV Toolbox多样化任务需求

2. **DevCV Toolbox（10个发育认知任务，基于NIH Baby Toolbox®）**:

    - **语言子域**：Looking While Listening(6-24月,双图选择)、Picture Vocabulary(≥25月,四图词汇理解)、Localization(1-42月,物体定位)
    - **执行功能/记忆子域**：Left/Right(朝向辨别)、Spatial Details(空间细节)、Visual Delayed Response(遮挡后记忆)、Memory(多轮延迟记忆)
    - **数学子域**：Who Has More(数量比较,合成+自然两个版本)、Subitizing(快速计数)、Object Counting(物体计数)
    - 每个任务都从SAYCam帧中构建自然场景样本，替代原始工具箱中的卡通刺激物，确保域内评测
    - 设计动机：NIH Baby Toolbox®的临床使用验证了其作为发育评估工具的可信度

3. **适配过程（以Picture Vocabulary为例）**:

    - 原始NIH测试：iPad上展示4张卡通图+语音提示→儿童点击
    - DevCV适配：SAYCam帧1FPS采样→GPT-4o+手工标注物体→Grounding-DINO裁剪→MAB-CDI词汇表过滤→按语义/语音学分布构造干扰项→人工质量审核

4. **模型架构**:

    - ViT-L-16(300M) + MLP连接器 + LLaMA-1.1B
    - **全部从零训练**，不使用任何预训练权重——确保能力完全来自婴儿语料
    - 输入：文本/单图/多图/视频/多轮对话；输出：自然语言

### 损失函数 / 训练策略
三阶段pipeline：Stage 1 视觉编码器预训练，Stage 2 图像-文本对齐，Stage 3 多格式联合训练。最后基于DevCV任务的指令微调。

## 实验关键数据

### 主实验（DevCV Toolbox 域内评测）

| 模型 | Overall | Count | PV(词汇) | Memory | WhoHasMore | LeftRight |
|------|---------|-------|----------|--------|------------|-----------|
| 人类表现 | 93.0 | 99.1 | 91.8 | 87.3 | 63.6/95.5 | 94.5 |
| Gemini-2.5-flash | 72.7 | 71.1 | 91.2 | 84.8 | 42.4 | 34.9 |
| GPT-4o | ~70 | ~65 | ~90 | ~80 | ~40 | ~34 |
| **BabyVLM-V2** | 竞争力 | **部分超越GPT-4o** | 竞争力 | 竞争力 | 竞争力 | 竞争力 |

### 消融实验

| 配置 | 关键影响 | 说明 |
|------|---------|------|
| 仅图像-文本预训练(V1) | 基线 | 开放集接近零 |
| +视频-语句(181K) | +视频理解任务改善 | DelayedResponse任务受益 |
| +交错序列(63K) | +多轮对话任务改善 | Memory任务受益 |
| +指令微调(113K) | **显著全面提升** | 从logits输出→自然语言 |
| 768K vs 67K图像对 | V2 >> V1 | 数据量的直接影响 |

### 关键发现
- **数学任务超越GPT-4o**：从零训练的~1.4B模型在Who Has More和Counting上部分超越GPT-4o——婴儿经验数据蕴含足够的计数和数量理解
- DevCV Toolbox的人类上界(93%)远高于所有AI模型，AI与儿童认知差距显著
- Subitizing和Looking While Listening作为hold-out任务测试泛化性，证实多格式预训练的泛化收益
- 三种预训练数据格式各有独立且互补的贡献
- OOD测试集(Ego4D构建)性能下降验证了域内评测的必要性

## 亮点与洞察
- **发育心理学标准化评估的AI工程化**：首次将NIH Baby Toolbox®转化为AI评测benchmark，开创了发育计算视觉的研究范式。未来心理学家可以用DevCV Toolbox"阅读早期儿童的心智"
- **挑战Scaling Law**：仅478小时的婴儿经验就能训练出在数学任务上超越GPT-4o的模型，展示了样本效率预训练的巨大潜力
- **数据格式多样性>数据量**：V1(67K)到V2(768K+视频+交错)的跨越不仅来自量的增加，更关键的是格式多样性使能力多样化
- **三方有益**：让大学可参与FM研究+为认知科学提供实验工具+增进AI公众理解

## 局限与展望
- SAYCam仅3名婴儿(6-32月龄)，样本量极小且存在个体差异。BabyView等更大规模数据待纳入
- 紧凑模型在复杂推理上仍远逊于大模型和人类——ADI差距巨大
- DevCV Toolbox缺儿童实际表现数据（仅成人上界）——需心理学实验室合作收集真正的发育对比数据
- 指令微调用DevCV任务本身，可能存在task leakage
- 不包括非视觉的语言和运动发育评估

## 相关工作与启发
- **vs BabyVLM-V1**: 数据扩大11倍+多格式；benchmark 4→10任务且基于NIH标准化测试；模型从logits→自然语言
- **vs Vong et al.(CLIP on SAYCam)**: 仅关注词-指称映射，本文关注通用感知
- **vs DevBench/KIVA**: 面向更大年龄段，不匹配SAYCam的6-32月龄段
- **启发**：发育认知视角可为AI训练策略提供全新灵感——也许"像婴儿一样学习"是通往AGI的另一条道路

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 独特的发育认知视角+NIH Baby Toolbox®的首次AI适配
- 实验充分度: ⭐⭐⭐⭐ DevCV设计严谨，缺乏真实儿童数据对比
- 写作质量: ⭐⭐⭐⭐⭐ 跨学科背景介绍充分
- 价值: ⭐⭐⭐⭐⭐ 对理解AI与人类认知的关系有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining](../../ICCV2025/audio_speech/25_years_in_class_a_multimodal_textbook_for_visionlanguage_p.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](../../ICCV2025/audio_speech/vggsounder_audio-visual_evaluations_for_foundation_models.md)
- [\[NeurIPS 2025\] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](../../NeurIPS2025/audio_speech/data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)
- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](../../NeurIPS2025/audio_speech/textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [\[NeurIPS 2025\] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models](../../NeurIPS2025/audio_speech/e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)

</div>

<!-- RELATED:END -->
