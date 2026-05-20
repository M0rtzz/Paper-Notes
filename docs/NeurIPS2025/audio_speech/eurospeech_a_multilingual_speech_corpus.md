---
title: >-
  [论文解读] EuroSpeech: A Multilingual Speech Corpus
description: >-
  [NeurIPS2025][音频/语音][多语言语音] 提出可扩展的开源 pipeline，从 22 个欧洲议会录音中自动构建 EuroSpeech 数据集——61K 小时、覆盖 22 种语言的高质量语音-文本对齐数据，其中 19 种语言超 1K 小时，微调 Whisper 后平均 WER 降低 41.8%。
tags:
  - "NeurIPS2025"
  - "音频/语音"
  - "多语言语音"
  - "议会录音"
  - "ASR"
  - "语音-文本对齐"
  - "数据集构建"
  - "低资源语言"
---

# EuroSpeech: A Multilingual Speech Corpus

**会议**: NeurIPS2025  
**arXiv**: [2510.00514](https://arxiv.org/abs/2510.00514)  
**代码**: [disco-eth/EuroSpeech](https://huggingface.co/datasets/disco-eth/EuroSpeech)  
**领域**: 语音处理 / 多语言数据集  
**关键词**: 多语言语音, 议会录音, ASR, 语音-文本对齐, 数据集构建, 低资源语言

## 一句话总结

提出可扩展的开源 pipeline，从 22 个欧洲议会录音中自动构建 EuroSpeech 数据集——61K 小时、覆盖 22 种语言的高质量语音-文本对齐数据，其中 19 种语言超 1K 小时，微调 Whisper 后平均 WER 降低 41.8%。

## 研究背景与动机

- **核心问题**: 多语言 ASR/TTS 模型依赖大规模标注语音数据，但公开可用的多语言数据集在语言覆盖上严重不均衡——大部分语言的数据量远低于训练所需的 1K 小时门槛
- **已有数据集不足**:
    - Common Voice 覆盖 133 种语言，但仅 8 种超 1K 小时
    - VoxPopuli 来自欧盟议会，仅 16 种语言、1.8K 小时，无语言超 1K 小时
    - Whisper 训练数据 680K 小时但不公开，MMS-Lab 覆盖 1107 种语言但仅私有
    - FLEURS 仅 1.4K 小时，面向 benchmark 而非训练
- **机遇**: 各国议会公开录音+文字记录是天然的多语言高质量语音来源，但数据格式碎片化、转录稿非逐字记录、录音时间长且未分段，现有 pipeline 难以规模化处理
- **动机**: 设计一套源无关(source-agnostic)的自动化 pipeline，处理非逐字转录、跨格式数据源，从而大规模构建均衡的多语言语音数据集

## 方法详解

### 1. 整体 Pipeline 架构

完整流程分为三个阶段：

1. **数据源收集与元数据整理**: 手动检查各国议会网站 → 编写定制采集脚本 → 生成标准化 CSV（含音频/视频 URL、转录稿链接、会议 ID）
2. **下载 Pipeline**: 基于调度架构(dispatch architecture)，将 URL 分发给专用 handler（直连下载、YouTube、动态页面等），支持断点续传、并行下载、PostgreSQL 状态跟踪
3. **对齐 Pipeline**: 音频分段 → ASR 转录 → 两阶段动态对齐 → CER 过滤 → 输出对齐数据集

### 2. 两阶段动态对齐算法（核心贡献）

针对议会转录稿中的非逐字文本、发言人标注、程序性文字等噪声，提出 coarse-to-fine 对齐策略：

**Stage 1 — 粗搜索(Coarse Search)**:
- 对每个 ASR 转录段，使用长度为 $n$（ASR 段词数）的滑动窗口
- 从上一段匹配位置 `last_end_idx` 开始顺序搜索
- 计算每个窗口与 ASR 文本的 CER，选取 CER < 30% 的首个候选；若无，则取 CER 最低的 $k=3$ 个候选

**Stage 2 — 细搜索(Refined Search)**:
- 对粗搜索得到的候选窗口进行局部优化
- 在起始位置 $\pm 15$ 词范围内遍历，窗口大小在 $(L-15, L+15)$ 范围测试
- 选取 CER 最小的 (start, size) 组合作为最终匹配

**回退机制(Fallback)**:
- 若局部搜索 CER 超阈值 $\theta$，从转录稿开头重新执行全局粗搜索，解决前序误对齐导致搜索位置偏移的问题
- 若全局搜索仍失败，执行默认匹配（在 `last_end_idx` 附近 refined search，无论 CER 多大都保留），确保数据集完整覆盖

### 3. 转录稿预处理

- 内置 PDF、DOCX、HTML、TXT、SRT 多格式解析器
- 可选的 LLM 清洗步骤（默认使用 Gemini Flash 2.0），去除发言人标签、程序性注释等非语音内容
- 德语测试中，LLM 清洗将对齐段中位 CER 从 12.3% 降至 9.7%

### 4. 多转录稿选择策略

议会可能为同一天提供多份格式不同的转录稿，且无法明确对应关系。解决方案：
- 将音频与所有候选转录稿 × 所有格式分别对齐
- 按中位 CER 选最佳格式
- 按可配置标准（最低 CER 或低于阈值的全部）选最终转录稿

### 5. CER 分层质量过滤

以 CER 为主要质量控制指标，在多个阈值下统计对齐时长：

| 过滤级别 | 对齐时长 | 占比 |
|---------|---------|------|
| 全部对齐 | 78.1K h | 100% |
| CER < 30% | 61.0K h | 78.2% |
| CER < 20% | 50.5K h | 65.4% |
| CER < 10% | 32.3K h | 41.0% |

CER < 20% 为主数据集（与 VoxPopuli 一致），音频段长 3-20 秒，采样率 16kHz。

## EuroSpeech 数据集分析

### 语言覆盖与规模

| 语言 | CER<20% 时长 | 语言 | CER<20% 时长 |
|------|-------------|------|-------------|
| 克罗地亚语 | 5615.8h | 保加利亚语 | 2200.1h |
| 丹麦语 | 5559.8h | 德语 | 2184.4h |
| 挪威语 | 3866.7h | 塞尔维亚语 | 1855.7h |
| 葡萄牙语 | 3293.5h | 芬兰语 | 1848.2h |
| 意大利语 | 2813.7h | 拉脱维亚语 | 1218.8h |
| 立陶宛语 | 2681.2h | 乌克兰语 | 1191.1h |
| 英语 | 2609.3h | 斯洛文尼亚语 | 1156.4h |
| 斯洛伐克语 | 2553.6h | 爱沙尼亚语 | 1014.9h |
| 希腊语 | 2395.4h | 波黑语 | 691.3h |
| 瑞典语 | 2312.8h | 冰岛语 | 647.4h |
| 法语 | 2249.8h | 马耳他语 | 613.0h |

**关键对比**: 在 12 种语言上超越已有公开 SOTA 数据集规模，其中 8 种语言首次突破 1K 小时门槛；5 种语言数据量是此前最佳的 10-100 倍（如立陶宛语 25h → 2681h，斯洛伐克语 61h → 2554h，马耳他语 44h → 613h）。

### 数据划分

按整个议会会议(session)为单位划分 train/dev/test，防止同一会议的片段泄露到不同集合中。

## 实验结果

### ASR 微调评估

在 6 种低资源语言上，每种仅取约 200 小时最低 CER 数据微调 Whisper v3 Turbo，在域外 FLEURS 测试集上评估：

| 语言 | 基线 WER | 微调 WER | 相对提升 |
|------|---------|---------|---------|
| 马耳他语 | 72.2% | 25.9% | **64.1%** |
| 冰岛语 | 20.0% | 15.0% | 25.0% |
| 立陶宛语 | 25.0% | 15.9% | 36.4% |
| 拉脱维亚语 | 19.3% | 11.1% | 42.5% |
| 斯洛文尼亚语 | 20.5% | 13.0% | 36.7% |
| 爱沙尼亚语 | 18.4% | 9.9% | **46.1%** |
| **平均** | **29.2%** | **15.1%** | **41.8%** |

- 马耳他语提升最为显著（64.1%），从几乎不可用变为实用水平
- 6 种语言均在 **域外** 测试集上获得大幅提升，验证了数据集的泛化价值
- 训练效率高：每种语言仅需 200h 数据、1.3-43h GPU 时间

### 计算成本

| 阶段 | 资源消耗 |
|------|---------|
| 视频下载 | ~3930 CPU·h |
| 转录稿获取 | ~280 CPU·h |
| 对齐处理 | ~5548 GPU·h (多种 GPU) |
| ASR 微调 | 1.3-43h/语言 (A6000) |

## 亮点与洞察

- **均衡覆盖是核心价值**: 与其追求极大的总时长，EuroSpeech 的差异化在于 22 种语言均超 500h——相比之下 Common Voice 133 种语言中仅 8 种超 1K 小时
- **两阶段对齐算法设计精巧**: 线性扫描保证效率，局部 refined search 保证精度，两级 fallback 保证鲁棒性，无需手动预处理转录稿
- **pipeline 的工程价值**: 模块化设计（数据源 CSV → 下载调度 → 对齐 → 过滤）使得扩展到新议会仅需编写元数据采集脚本，核心逻辑完全复用
- **LLM 辅助清洗降低门槛**: 用 Gemini Flash 自动清洗 PDF 转录稿中的非语音元素，减少了人工预处理
- **选择基线最差的语言做实验**: 刻意选 Whisper 表现最差的 6 种语言，既展示最大提升幅度，也是对 pipeline 鲁棒性的严格测试（ASR 越差→对齐噪声越大→pipeline 需更强容错）

## 局限性与改进方向

- **领域单一**: 仅覆盖议会正式演讲，语言风格为规划性/正式性，模型在日常对话、非正式场景中可能泛化不佳
- **方言/变体覆盖不足**: 议会语言倾向标准/官方变体，缺乏方言和社会语言学变体的代表性
- **对 ASR 模型的依赖**: 对齐质量受底层 ASR 能力制约，低资源语言的 ASR 本身就差，对齐可能引入系统性偏差
- **地理局限**: 仅覆盖欧洲 22 个国家，非洲、亚洲、南美洲的低资源语言未涉及
- **潜在滥用风险**: 数据含可识别的政治人物语音，可能被用于语音合成/深伪造，尽管议会语言风格在一定程度上限制了滥用范围
- **未来方向**: 扩展至对话式语音、加入说话人元数据、支持 24kHz 采样率的 TTS 版本

## 相关工作对比

| 数据集 | 语言数 | 总时长 | >1K h 语言数 | 公开 |
|--------|-------|--------|------------|------|
| Common Voice | 133 | 22.1K h | 8 | ✓ |
| VoxPopuli | 16 | 1.8K h | 0 | ✓ |
| YODAS | 149 | 369.5K h | 13 | ✓ |
| Whisper Data | 91 | 680K h | 16 | ✗ |
| MMS-Lab | 1107 | 44.7K h | 0 | ✗ |
| **EuroSpeech** | **22** | **61K h** | **19** | **✓** |

EuroSpeech 在公开数据集中实现了 >1K h 语言数最多（19 种），以较少的语言总数换取了每种语言的充分覆盖。

## 启发与思考

- **数据质量⇔数量的分层权衡**: CER 多级过滤的设计思路（10%/20%/30%）值得借鉴——不同应用对噪声容忍度不同（TTS 要高质量、ASR 预训练可用更多噪声数据）
- **公共数据的系统化利用**: 议会录音这类"政府公开但未被系统利用"的数据源，在其他模态（如法律文本、政策文档、手语视频）中同样存在
- **pipeline 思维**: 与其手工标注小规模高质量数据，不如设计鲁棒 pipeline 从大量噪声源中自动提取——这是当前大规模数据集构建的主流范式

## 评分
- 新颖性: ⭐⭐⭐ pipeline不算新但规模大
- 实验充分度: ⭐⭐⭐⭐ 多语言ASR验证
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐ 大规模多语言语音资源对社区有高价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[ACL 2025\] GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages](../../ACL2025/audio_speech/gigaspeech2_low_resource_asr.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](../../ACL2025/audio_speech/speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [\[ACL 2026\] Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs](../../ACL2026/audio_speech/mind_the_pause_disfluency-aware_objective_tuning_for_multilingual_speech_correct.md)
- [\[ACL 2025\] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis](../../ACL2025/audio_speech/tcsinger_2_customizable_multilingual_zero-shot_singing_voice_synthesis.md)

</div>

<!-- RELATED:END -->
