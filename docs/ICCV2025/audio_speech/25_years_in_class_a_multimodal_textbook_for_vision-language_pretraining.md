---
title: >-
  [论文解读] 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining
description: >-
  [ICCV 2025][语音][视觉-语言预训练] 从YouTube教学视频中提取关键帧和文本（ASR+OCR），构建高质量交错图文格式的"多模态教材"数据集，用于VLM预训练，在知识密集型和推理任务上大幅领先网页爬取的交错数据集。
tags:
  - ICCV 2025
  - 语音
  - 视觉-语言预训练
  - 交错图文数据集
  - 教学视频
  - 多模态教材
  - 上下文学习
---

# 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining

**会议**: ICCV 2025  
**arXiv**: [2501.00958](https://arxiv.org/abs/2501.00958)  
**代码**: [https://github.com/DAMO-NLP-SG/multimodal_textbook](https://github.com/DAMO-NLP-SG/multimodal_textbook)  
**领域**: 多模态预训练  
**关键词**: 视觉-语言预训练, 交错图文数据集, 教学视频, 多模态教材, 上下文学习

## 一句话总结

从YouTube教学视频中提取关键帧和文本（ASR+OCR），构建高质量交错图文格式的"多模态教材"数据集，用于VLM预训练，在知识密集型和推理任务上大幅领先网页爬取的交错数据集。

## 研究背景与动机

现有的交错图文数据集（如MMC4、OBELICS）均通过网页爬取获得，存在三个核心问题：（1）**图文关联松散**——网页图片可能与上下文无关（如广告、logo）；（2）**图片序列缺乏逻辑连贯性**——网页内多张图片之间没有明确的推理关系；（3）**知识密度低**——新闻、娱乐等内容占据大量比例，缺乏基础学科知识。

与此同时，互联网上存在海量高质量教学视频（如数学几何课程），这些视频中讲师的逐帧演示配合详细口头讲解，天然形成了图文紧密对齐、逻辑连贯的"教材"结构。然而，这些资源在VLM训练中尚未被充分利用。微软Phi系列模型也证明了高质量教材级数据对LLM训练的关键性。

## 方法详解

### 整体框架

整个方法可概括为两个阶段：**教学视频的系统化收集**和**视频到教材的多级提取与过滤流水线**。最终产出一个包含650万关键帧和7.5亿文本token的交错图文数据集，覆盖数学、物理、化学等6个基础学科。

### 关键设计

1. **LLM驱动的知识分类体系与视频收集**:

    - 利用GPT-4o构建四层知识分类体系：学科→课程→子课程→知识点，涵盖6个学科、55门课程、3915个知识点
    - 以每个知识点为关键词从YouTube检索相关视频，取top-50并去重
    - 用LLM审查视频元数据（标题、描述、评论）过滤不相关/违规内容，最终收集159,565个视频
    - 设计动机：标准化分类体系确保覆盖面广、避免遗漏重要学科

2. **多级知识提取与过滤流水线（视频→教材）**:

    - **视频级**：用FFmpeg提取音频→whisper-large-v3转录ASR→Qwen2-72B改写优化ASR流畅度；LLM从相关性、知识密度、转录质量三个维度过滤低质视频，保留75K视频
    - **片段级**：用ASR时间戳将长视频分为10-20秒片段；用VideoLlama2生成片段描述，通过与ASR的文本相似度过滤无信息量的片段（如仅有讲师正面照的场景）
    - **关键帧级**：用SSIM算法对比连续帧检测显著变化，提取关键帧去除冗余；用InternVL2-40B对关键帧做OCR提取文本/公式/符号，并过滤低信息量关键帧和重复OCR
    - 设计动机：多级（粗到细）过滤逐步消除噪声——视频级过滤不相关内容，片段级过滤无视觉知识的场景，关键帧级去除冗余帧和低质OCR

3. **交错格式教材的组织**:

    - 将关键帧、OCR文本和精化ASR文本按时间顺序交错排列
    - 即便某片段的视觉内容被过滤，其ASR文本仍可保留——不浪费有价值的口述知识
    - 最终格式：$\{\text{frame}_1^{k_1}, \text{frame}_1^{k_2}, \text{ocr}_1, \text{asr}_1, \text{asr}_2, \text{asr}_3, \text{frame}_4^{k_1}, \text{ocr}_4, \text{asr}_4, \ldots\}$

### 训练策略

对LLaVA-1.5-7B做持续预训练（在558K配对数据对齐后），对Idefics2-8B分别做从头训练和持续预训练。为公平比较，从MMC4和OBELICS中采样等量样本（610K），使用相同训练参数。

## 实验关键数据

### 主实验

| 数据集/基准 | 设置 | MMC4 | OBELICS | Textbook-6.5M | 提升 |
|-------------|------|------|---------|---------------|------|
| ScienceQA-IMG | 0-shot | - | - | 26.3 | - |
| ScienceQA-IMG | 4-shot | 11.6 | 16.4 | **37.3** | +20.9 vs MMC4 |
| MathVista | 0-shot | 20.4 | 21.6 | **24.3** | +2.7 |
| MathVista | 1-shot | 30.0 | 28.5 | **43.4** | +14.9 vs OBELICS |
| OKVQA | 4-shot | 28.7 | 37.5 | **39.9** | +2.4 vs OBELICS |
| TextVQA | 4-shot | 20.9 | 32.2 | **33.5** | +1.3 vs OBELICS |
| 7基准平均 | 0-4shot | 10.9-21.9 | 10.7-26.2 | **15.5-30.8** | +3.2~+8.3 |

在Idefics2上持续预训练后，MathVista从27.6提升至29.7，MathVision从14.3提升至16.2。

### 消融实验

| 配置 | 1-shot平均准确率 | 说明 |
|------|-----------------|------|
| 完整方法（SSIM+ASR精化+OCR） | 31.1 | 最佳 |
| 去除ASR精化 | 26.2 (↓4.9) | 原始ASR口语化PPL高达16.86，损害语言能力 |
| 去除OCR | 28.8 (↓2.3) | OCR提供公式/符号等额外知识 |
| SSIM→像素级关键帧提取 | 22.1 (↓9.0) | 提取过多帧（18M），大量冗余 |
| SSIM→CLIP语义级提取 | 24.6 (↓6.5) | 提取过少帧（1.7M），丢失关键帧 |

### 关键发现

- **"作弊测试"**验证上下文感知能力：将测试样本本身放入few-shot示例中，在MathVista上Textbook达94.1%（MMC4仅72.6%），证明预训练在Textbook上的VLM能有效关注交错上下文中的信息
- **打乱图片顺序**实验表明：MMC4打乱后几乎无影响，OBELICS有中等下降，而Textbook性能大幅下降——证实视频源数据的图片序列具有强逻辑依赖，这对学习复杂知识和推理至关重要
- 指令微调后（LLaVA-665K），Textbook在MathVista上额外提升5.5%，是OBELICS（+2.4%）的两倍多

## 亮点与洞察

- **数据来源的范式转变**：从网页爬取转向教学视频挖掘，利用视频天然的时序一致性和讲解-演示对齐性
- In-sample Image SIM指标显示教材内图片相关性（0.686）是OBELICS（0.345）的两倍，且随图片数量增加保持稳定
- 每样本平均10.7张图片和1297个token，远超MMC4（5.7张/417 token）

## 局限与展望

- 数据集主要覆盖基础学科教育类视频，对通用领域VQA的zero-shot提升有限（需few-shot才显优势）
- ASR精化依赖大型LLM（Qwen2-72B），处理75K视频的计算成本较高
- 仅覆盖英语教学视频，多语言扩展尚未探索
- 未与大规模（十亿级）网页数据混合训练，最优数据配比有待研究

## 相关工作与启发

- Phi系列模型验证了"教材级"数据的重要性，本文将此思想从纯文本扩展到多模态
- 与OmniCorpus等多源数据集的差异核心在于视频的时序连贯性——未来可探索将会议演讲、实验录像等更多视频类型纳入
- "作弊测试"方法可作为通用的交错上下文感知评估工具

## 评分
- 新颖性: ⭐⭐⭐⭐ 从教学视频构建交错教材数据集的思路新颖，填补了视频源预训练数据的空白
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多基准、多消融、作弊测试和打乱实验设计巧妙
- 写作质量: ⭐⭐⭐⭐ 结构清晰，pipeline描述详细
- 价值: ⭐⭐⭐⭐ 数据集开源，pipeline可复用，对知识密集型VLM预训练有实际价值

<!-- RELATED:START -->

## 相关论文

- [How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects](how_would_it_sound_material-controlled_multimodal_acoustic_profile_generation_fo.md)
- [Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)
- [MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [VGGSounder: Audio-Visual Evaluations for Foundation Models](vggsounder_audio-visual_evaluations_for_foundation_models.md)

<!-- RELATED:END -->
