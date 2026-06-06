---
title: >-
  [论文解读] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation
description: >-
  [AAAI2026][音频/语音][factuality evaluation] 提出 CCFQA——首个覆盖 8 种语言、14,400 条完全平行语音-文本事实问答样本的跨语言跨模态基准，支持 QA/XQA/SQA/XSQA 四种任务设定，系统揭示了现有 MLLM 在语言和模态切换下的事实不一致性…
tags:
  - "AAAI2026"
  - "音频/语音"
  - "factuality evaluation"
  - "multilingual benchmark"
  - "spoken question answering"
  - "cross-lingual consistency"
  - "多模态"
---

# CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation

**会议**: AAAI2026  
**arXiv**: [2508.07295](https://arxiv.org/abs/2508.07295)  
**代码**: [yxduir/ccfqa](https://github.com/yxduir/ccfqa)  
**领域**: 音频语音  
**关键词**: factuality evaluation, multilingual benchmark, spoken question answering, cross-lingual consistency, multimodal LLM

## 一句话总结

提出 CCFQA——首个覆盖 8 种语言、14,400 条完全平行语音-文本事实问答样本的跨语言跨模态基准，支持 QA/XQA/SQA/XSQA 四种任务设定，系统揭示了现有 MLLM 在语言和模态切换下的事实不一致性；同时提出 LLM-SQA，以英语为桥接语言、仅 5-shot 即实现跨语言语音问答迁移，在 XSQA 上 F1 达 51.4 超越 GPT-4o-mini-Audio（45.7）。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）已广泛部署于多语言场景，但其事实可靠性评估严重滞后。现有事实性基准（SimpleQA、TruthfulQA、HaluEval）主要面向英语文本，多语言基准 KoLasSimpleQA 覆盖 9 种语言但仅限文本。语音侧基准（SD-QA、VoiceBench、SpeechIQ）则仅面向单语音模态且多为英语。

**现有痛点**：同一事实问题用不同语言提问（跨语言不一致）或用不同模态呈现（跨模态不一致），MLLM 常给出矛盾答案。例如 GPT-4o 系列在英语文本上回答正确的问题，换成日语语音后可能回答错误。但目前缺乏包含完全平行数据的基准来系统测量这种不一致性。

**核心矛盾**：多语言+多模态场景的叉乘导致评估空间爆炸——8 种语言 × 2 种模态 × 跨语言组合 = 完整评估单个 MLLM 需 128,000 次请求，构建这样的高质量平行数据集成本极高（需各语言母语者真人录音）。

**本文目标**：构建首个同时覆盖跨语言和跨模态的事实性评估基准，并提出利用英语 LLM 强事实推理能力的低成本迁移方案。

**切入角度**：利用现有英语 QA 数据集（MKQA + MOOCCubeX）→ GPT-4.1 翻译 + 人工校验 → 母语者录音 + ASR 质检，构建高质量平行数据；模型侧用英语桥接 + 5-shot 迁移突破语言壁垒。

**核心 idea**：完全平行的语音-文本多语言数据 + 英语桥接少样本迁移 = 系统诊断并缓解 MLLM 跨语言跨模态的事实不一致性。

## 方法详解

### 整体框架

CCFQA 包含两个正交维度：**基准构建**（诊断问题）和 **LLM-SQA 模型**（缓解问题）。基准通过跨语言数据构建→跨模态数据构建→质量控制三阶段生成 14,400 条平行样本。LLM-SQA 通过课程学习（ASR→SRT→SQA）预训练再 5-shot 跨语言迁移，利用英语的事实知识服务非英语语音问答。

### 关键设计1：跨语言数据构建——严格筛选+机翻+人工校验

数据源来自 MKQA（开放域 QA）和 MOOCCubeX（教育领域），涵盖人文、社会、自然科学、应用科学 4 大类共 20 个子领域。关键筛选标准：排除歧义问题（如"谁是现任首相？"）、敏感内容（PII/冒犯性语言）、事实错误问答、文化依赖问题（如"法定结婚年龄是多少？"）。筛选后的英语问答用 GPT-4.1 翻译为 7 种目标语言（中文、法语、日语、韩语、俄语、西班牙语、粤语），中/英/日做人工校验，其余语言用回译+审校确保准确性。

### 关键设计2：跨模态数据构建——真人录音+ASR迭代质检

招募 8 种语言的母语志愿者（男女均衡），按清晰、自然、匀速、中性语调的规范朗读录音。对低音量样本做音频增强，然后用 Whisper-large-v3 做 ASR 转录，计算 WER/CER 并与原文对比。WER 超过阈值的音频标记为异常并重录。这套"录制→ASR 检验→重录"的迭代流程保证了语音-文本精准对齐。最终语音质量：英语 WER 仅 3.2%，中文 CER 6.8%。

### 关键设计3：LLM-SQA模型——课程学习+英语桥接5-shot迁移

模型架构：冻结 Whisper 语音编码器（~635M）+ 可训练 Q-Former 适配器（80 queries, dim=768）+ MLP（合计~80.5M）+ GemmaX2-9B LLM（~9.2B）。LLM 用 LoRA（$r=16, \alpha=32$）微调，仅训练~8.9M 参数。

训练策略采用三阶段课程学习：
1. **ASR 阶段**：在 FLEURS 数据集上训练语音识别，建立语音-文本对齐基础
2. **SRT 阶段**：训练语音识别+翻译，学习跨语言映射
3. **SQA 阶段**：分两步——(a) 用~3,000 条合成英语语音-文本对监督微调学习 QA 任务结构；(b) 对每种目标语言仅用 5-shot 做跨语言迁移

核心机制：通过特殊指令标记（如 `<|qa|><|fra|>`）让模型将非英语语音问题内部转化为英语进行事实推理，再将答案翻译回目标语言。这样利用了 LLM 在英语上最强的知识储备，同时最小化了非英语标注数据的需求。

### 损失函数/训练策略

各阶段均使用标准的自回归交叉熵损失。ASR 和 SRT 在 FLEURS 上训练，SQA 用合成英语数据+5-shot 目标语言数据。训练在 4×A100（80GB）上运行一周。优化器 AdamW，peak lr = $1 \times 10^{-4}$，1000 步 warmup + 线性衰减。

## 实验关键数据

### 四种任务设定主实验（F1 / LLM Acc）

| 模型 | 文本QA Avg | 文本XQA Avg | 语音SQA Avg | 语音XSQA Avg |
|------|:---:|:---:|:---:|:---:|
| GPT-4o-mini | 63.9 / 64.4 | 59.7 / 62.2 | — | — |
| GPT-4o-mini-Audio | — | — | 47.7 / 40.4 | 45.7 / 38.6 |
| Phi-4-Multimodal | 18.0 / 13.8 | 18.0 / 15.3 | 18.5 / 22.0 | 21.0 / 5.7 |
| Qwen2-Audio | 27.9 / 30.6 | 24.2 / 19.2 | 27.7 / 17.0 | 24.1 / 10.7 |
| Qwen2.5-Omni-3B | 20.6 / 11.8 | — | 34.9 / 20.9 | 29.8 / 17.1 |
| Qwen2.5-Omni-7B | 46.5 / 38.2 | 42.1 / 34.5 | 44.0 / 33.2 | 38.5 / 29.5 |
| **LLM-SQA (本文)** | — | — | **52.0 / 40.3** | **51.4 / 39.7** |

### 跨语言与跨模态一致性（%，越高越一致）

| 模型 | 跨语言一致性 | 跨模态一致性 |
|------|:---:|:---:|
| GPT-4o-mini | 96.6 / 95.5 | 62.7 / 62.1 |
| Phi-4-Multimodal | 90.8 / 25.9 | 62.7 / 37.5 |
| Qwen2-Audio | 62.4 / 62.9 | 55.6 / 56.0 |
| Qwen2.5-Omni-3B | 75.4 / 81.8 | 56.5 / 52.0 |
| Qwen2.5-Omni-7B | 90.3 / 87.2 | 90.3 / 85.5 |

### 语音时长对SQA准确率的影响（LLM Acc）

| 模型 | 0-5s | 5-10s | 10-30s | 平均 |
|------|:---:|:---:|:---:|:---:|
| GPT-4o-mini-Audio | 38.6 | 42.6 | 40.1 | 40.4 |
| LLM-SQA (本文) | 40.3 | 40.8 | 36.1 | 40.3 |
| Qwen2.5-Omni-7B | 32.1 | 34.9 | 29.0 | 33.2 |
| Qwen2-Audio | 14.7 | 19.5 | 18.6 | 17.0 |
| Phi-4-Multimodal | 17.7 | 26.4 | 27.4 | 22.0 |

### 关键发现

- LLM-SQA 在 SQA/XSQA 上均超越所有开源基线，且 XSQA F1（51.4）大幅超越 GPT-4o-mini-Audio（45.7）——5-shot 迁移策略非常有效
- 跨语言挑战：从 QA→XQA，大多数模型性能显著下降，GPT-4o-mini 最稳定（96.6%一致性）
- 跨模态挑战：文本→语音普遍存在"模态鸿沟"，Qwen2.5-Omni-7B 跨模态一致性最高（90.3%），得益于 Omni 架构设计
- F1 高但 LLM Acc 低→幻觉问题（回答流畅但事实错误）；F1 低但 Acc 高→指令遵循问题（知道答案但格式不规范）
- 语言维度：英语/法语/西班牙语表现最好，韩语/粤语最差，反映了预训练数据分布偏倚
- 语音质量：英语 WER 3.2%，部分语言（法语 13.8%、俄语 18.2%、粤语 16.8%）因专有名词较高

## 亮点与洞察

- **填补评估空白**：CCFQA 是唯一同时支持跨语言（8语言）和跨模态（文本+语音）、数据完全平行的事实性基准，完整评估需 128K 请求，评估维度前所未有
- **诊断力强**：跨语言一致性和跨模态一致性两个正交维度精准定位模型弱点——如 Phi-4-Multimodal 跨语言 LLM Acc 一致性仅 25.9%，暴露其多语言 token 设计缺陷
- **极低标注成本的迁移方案**：LLM-SQA 仅用 5-shot 即实现跨语言语音QA迁移，利用英语作为事实知识的"通用接口"，思路简洁但效果显著
- **F1 与 LLM Acc 的解耦分析**：揭示了幻觉（高F1低Acc）和指令遵循（低F1高Acc）两种不同失败模式，为模型诊断提供新工具
- **高质量语音数据**：真人录音+ASR迭代质检流程确保了评估的可靠性

## 局限与展望

- 语言覆盖 8 种仍有限，缺少阿拉伯语、印地语等重要低资源语言和声调语言（除粤语外）
- 语音数据为实验室环境朗读式录音，未覆盖带口音、噪声、自然对话等真实场景，可能高估模型实际表现
- LLM-SQA 的英语桥接策略引入英语中心偏差——对于与英语语言学距离极远的语言（如韩语），迁移效果可能有瓶颈
- 基准规模（1,000 测试对 × 8语言）在部分子领域的样本量有限
- 仅评估事实性问答，未涉及语音推理、语音摘要等更复杂任务
- 未探索端到端多语言语音训练（而非英语桥接）是否能在大规模数据下表现更好

## 相关工作与启发

- **vs SimpleQA / Chinese SimpleQA**：仅单语文本。CCFQA 扩展到 8 语言 + 语音模态，首次实现跨模态事实一致性评估
- **vs KoLasSimpleQA**：覆盖 9 种语言但仅限文本，且问答非完全平行。CCFQA 的完全平行设计使得跨语言一致性可精确量化
- **vs VoiceBench / SpeechIQ**：仅英语语音、非开放式回答。CCFQA 支持 8 语言开放式语音问答
- **vs SD-QA**：5种语言的方言语音QA，但不支持跨语言评估。CCFQA 增加 XQA/XSQA 评估维度

| 基准 | 语言数 | 模态 | 数据量 | 支持任务 |
|------|:---:|:---:|:---:|:---:|
| SimpleQA | 1 | 文本 | 4,326 | QA |
| KoLasSimpleQA | 9 | 文本 | 2,147 | QA |
| VoiceBench | 1 | 语音 | 5,783 | SQA |
| SD-QA | 5 | 语音 | 11,109 | SQA |
| **CCFQA** | **8** | **文本+语音** | **14,400** | **QA/XQA/SQA/XSQA** |

**启发**：英语桥接策略在事实知识传递上有效，可推广至其他知识密集型多语言任务；课程学习（ASR→SRT→SQA）的渐进式训练可借鉴于多模态对齐场景；"模态鸿沟"现象值得深入研究——语音输入为何系统性地降低事实准确率？

## 评分

⭐⭐⭐⭐ (4/5)

综合评价：首个跨语言跨模态事实性评估基准，填补重要空白。基准构建流程严谨（GPT翻译+人工校验+真人录音+ASR质检）、评估维度全面（4种任务×2种指标×8种语言）、LLM-SQA 的 5-shot 迁移方案简洁有效。主要遗憾是语言覆盖和语音场景多样性仍有限，且模型创新（英语桥接）偏增量。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Probing Cross-modal Information Hubs in Audio-Visual LLMs](../../ICML2026/audio_speech/probing_cross-modal_information_hubs_in_audio-visual_llms.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[ICML 2026\] Position: Towards Responsible Evaluation for Text-to-Speech](../../ICML2026/audio_speech/position_towards_responsible_evaluation_for_text-to-speech.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)

</div>

<!-- RELATED:END -->
