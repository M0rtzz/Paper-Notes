---
title: >-
  [论文解读] HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation
description: >-
  [CVPR 2025][人物交互] HOIGen-1M 是首个面向人物交互 (HOI) 视频生成的百万级高质量数据集，通过高效数据筛选管线和 Mixture-of-Multimodal-Experts (MoME) 字幕策略解决了 HOI 视频数据稀缺和描述幻觉问题，并提出 CoarseHOIScore/FineHOIScore 两个评估指标来量化生成视频中交互的质量。
tags:
  - CVPR 2025
  - 人物交互
  - 文本到视频生成
  - 大规模数据集
  - 视频描述
  - 多模态大模型
---

# HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation

**会议**: CVPR 2025  
**arXiv**: [2503.23715](https://arxiv.org/abs/2503.23715)  
**代码**: https://liuqi-creat.github.io/HOIGen.github.io  
**领域**: 视频理解/视频生成  
**关键词**: 人物交互, 文本到视频生成, 大规模数据集, 视频描述, 多模态大模型

## 一句话总结

HOIGen-1M 是首个面向人物交互 (HOI) 视频生成的百万级高质量数据集，通过高效数据筛选管线和 Mixture-of-Multimodal-Experts (MoME) 字幕策略解决了 HOI 视频数据稀缺和描述幻觉问题，并提出 CoarseHOIScore/FineHOIScore 两个评估指标来量化生成视频中交互的质量。

## 研究背景与动机

**领域现状**：文本到视频 (T2V) 生成已取得巨大进展，Sora、Kling 1.5 等模型可以生成复杂场景。然而，人物交互 (HOI) 作为物理世界的基本组成部分，仍然是当前 T2V 模型的硬伤——即使超过 10B 参数的模型也难以准确生成简单的 HOI 视频（如"把行李箱搬上公交车"）。

**现有痛点**：(1) 缺乏大规模 HOI 视频数据——WebVid-10M 包含低质量水印视频，Panda-70M 含大量静态/模糊视频且多数不含 HOI；(2) 现有 HOI 感知数据集（CAD-120、BEHAVE 等）规模太小（数千到数万级），远达不到训练 T2V 模型所需的百万级；(3) 现有字幕方法要么太简短（12-13 词），要么不专门针对 HOI 设计，丢失交互细节；(4) 缺乏评估 HOI 视频生成质量的专用指标。

**核心矛盾**：T2V 模型在 HOI 场景下的表现远不如一般场景，根本原因是训练数据中缺乏大规模、高质量、描述精确的 HOI 视频。

**本文目标** 构建一个百万级高质量 HOI 视频数据集，设计精确的视频描述方法，以及提出 HOI 视频生成的评估框架。

## 方法详解

### 整体框架

HOIGen-1M 的构建包含三个核心部分：(1) 视频筛选管线——从 8000 万原始视频中自动筛选出高质量 HOI 视频；(2) MoME 字幕策略——利用多个多模态大模型互相验证，消除幻觉并生成精确描述；(3) 评估框架——提出 CoarseHOIScore 和 FineHOIScore 两个指标，从粗到细评估生成视频中的交互质量。

### 关键设计

1. **高效视频筛选管线**：
    - 功能：从 8000 万原始视频中高效筛选出包含 HOI 的高质量视频
    - 核心思路：五阶段级联过滤——(a) 元数据过滤（时长>1s、分辨率≥720p、帧率≥20FPS）；(b) OCR 过滤去除文字多的视频；(c) 美学评分过滤确保视觉质量；(d) 光流评分过滤确保适度运动（过高/过低都排除）；(e) MLLM+LLM 判断是否含 HOI（PLLaVA 生成描述 + Qwen2.5 判断交互）；最终 150 万视频进入人工验证阶段
    - 人工验证：7 名标注员用 8 周时间逐一检查交互是否明显、物体是否可见，最终得到 110 万视频

2. **Mixture-of-Multimodal-Experts (MoME) 字幕策略**：
    - 功能：生成精确且无幻觉的 HOI 视频描述
    - 核心思路：(a) 两个字幕专家（PLLaVA 和 Qwen2-VL）分别生成描述；(b) 一个决策专家（Llama3.1）判断两个描述是否一致——不一致则检测到幻觉；(c) 检测到幻觉时，引入第三个字幕专家聚焦争议区域，再由决策专家融合生成修正后的描述；(d) 无幻觉时，由决策专家选择信息更丰富的描述
    - 设计动机：单个 MLLM 在视频描述时不可避免会产生幻觉，通过多专家交叉验证可系统性地检测和消除幻觉

3. **CoarseHOIScore 和 FineHOIScore 评估指标**：
    - CoarseHOIScore：使用 HOI 检测器检测生成视频中是否存在 HOI 三元组（人、物体、动作），按帧统计超过置信度阈值的比例
    - FineHOIScore：基于 MLLM 评分，综合评估交互合理性、动作流畅度、人体真实度等多个维度，提供更细粒度的质量评估

### 损失函数/训练策略

本文主要贡献是数据集而非模型。在微调 T2V 模型验证数据集价值时，使用的是各模型原有的训练策略（如 CogVideoX-5B 的原始训练损失）。

## 实验关键数据

| 指标 | 数据 |
|------|------|
| 数据集规模 | 110 万+ 视频片段 |
| 视频时长 | 共 2200+ 小时 |
| 分辨率 | ≥720p |
| 平均描述长度 | 153.8 词（WebVid-10M: 12.0, Panda-70M: 13.2）|
| 物体种类 | 15,000+ |
| 交互动作种类 | 7,000+ |
| 微调后 CogVideoX-5B CoarseHOIScore | 接近商业软件 Kling 1.5 的水平 |
| 最佳商业模型 (Kling 1.5) CoarseHOIScore | 42.72% |
| 最佳开源模型 (CogVideoX-5B) CoarseHOIScore | 32.84% |
| Hailuo CoarseHOIScore | 39.56% |
| Dreamina CoarseHOIScore | 36.36% |
| 评估prompt数量 | 306 条（乐器、交通工具、厨具等）|
| 人工验证耗时 | 7 名标注员 × 8 周 |

## 亮点与洞察

1. **首个百万级 HOI 视频生成数据集**：填补了 T2V 领域在 HOI 方向的数据空白，所有视频均经过人工验证
2. **MoME 字幕策略的幻觉消除思路**：通过多个 MLLM 交叉验证而非依赖单个模型，是处理大规模自动标注中幻觉问题的有效范式
3. **HOI 评估指标的设计**：将 HOI 检测器引入生成评估是巧妙的跨任务迁移——比通用指标更能捕捉交互生成的核心质量
4. **实验揭示的 gap**：即使最先进的商业模型 Kling 1.5 在 CoarseHOIScore 上也仅 42.72%，说明 HOI 视频生成仍然是一个远未解决的难题

## 局限性

1. CoarseHOIScore 和 FineHOIScore 依赖现有 HOI 检测器和 MLLM 的能力，可能无法捕捉精细的交互质量差异
2. 数据集主要来源于公开视频，场景和拍摄条件可能存在偏差
3. 人工验证虽然保证了质量，但限制了进一步扩展的效率
4. 字幕平均 153.8 词虽然比现有数据集长很多，但对于复杂 HOI 场景仍可能不够详细

## 相关工作

- **T2V 数据集**：WebVid-10M（1000 万视频，短描述，有水印）、Panda-70M（7000 万视频，短描述，多静态）、OpenVid-1M（100 万视频，长描述，通用场景）
- **HOI 感知数据集**：BEHAVE（15.2K 帧，RGBD+SMPL）、HOI4D（4000 段第一人称 4D 视频）、GRAB（1334 段全身+手部动作捕捉序列）、MPHOI-72（72段多人活动视频）、PVSG（400段第一/第三人称视频）
- **T2V 模型**：Sora（分钟级视频生成）、CogVideoX（开源百万级训练）、Kling 1.5（商业级视频生成）、OpenSora/OpenSoraPlan（开源社区方案）
- **视频描述方法**：PLLaVA（视频多模态理解）、Qwen2-VL（视频描述生成）、MoME 策略（多专家交叉验证消除幻觉）
- **视频质量评估**：VBench（16维视频质量评估框架）、FID/FVD（通用生成质量指标）

## 评分

- 新颖性：⭐⭐⭐⭐（首个百万级 HOI 视频生成数据集 + MoME 幻觉消除 + 专用评估指标）
- 实用性：⭐⭐⭐⭐⭐（直接可用于提升 T2V 模型的 HOI 生成能力）
- 技术深度：⭐⭐⭐（数据集论文以工程为主，方法创新适中）
- 表达清晰度：⭐⭐⭐⭐（结构清晰，分析全面）

<!-- RELATED:START -->

## 相关论文

- [DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation](../../ICCV2025/video_generation/dh-facevid-1k_a_large-scale_high-quality_dataset_for_face_video_generation.md)
- [IDOL: Instant Photorealistic 3D Human Creation from a Single Image](idol_instant_photorealistic_3d_human_creation_from_a_single_image.md)
- [Video-Bench: Human-Aligned Video Generation Benchmark](video-bench_human-aligned_video_generation_benchmark.md)
- [TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](../../ICCV2025/video_generation/tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)
- [Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval](video-colbert_contextualized_late_interaction_for_text-to-video_retrieval.md)

<!-- RELATED:END -->
