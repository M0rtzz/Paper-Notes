---
title: >-
  ICML2026 音频/语音方向7篇论文解读
description: >-
  7篇ICML2026的音频/语音方向论文解读，涵盖语音、医学影像、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "音频/语音"
  - "论文解读"
  - "论文笔记"
  - "语音"
  - "医学影像"
  - "少样本学习"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**🧪 ICML2026** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (47)](../../ACL2026/audio_speech/index.md) · [📷 CVPR2026 (15)](../../CVPR2026/audio_speech/index.md) · [🔬 ICLR2026 (32)](../../ICLR2026/audio_speech/index.md) · [🤖 AAAI2026 (31)](../../AAAI2026/audio_speech/index.md) · [🧠 NeurIPS2025 (50)](../../NeurIPS2025/audio_speech/index.md) · [📹 ICCV2025 (11)](../../ICCV2025/audio_speech/index.md)

🔥 **高频主题：** 语音 ×5

**[Alethia: A Foundational Encoder for Voice Deepfakes](alethia_a_foundational_encoder_for_voice_deepfakes.md)**

:   Alethia 提出一种"瓶颈式掩码嵌入预测 + Flow-Matching 频谱生成"的双分支预训练范式，训出首个面向语音 deepfake 检测/定位/溯源的基础编码器，在 5 类任务 56 个数据集上显著超过 Wav2vec2/HuBERT/WavLM 等通用 SFM，并对未见过的歌声 deepfake 和真实扰动表现出强零样本鲁棒性。

**[MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)**

:   MECAT 用「多专家模型 + CoT 大模型推理」构造了 20k 条多视角细粒度音频字幕与 10 万条开放式 QA，并提出 DATE 指标（语义相似度 × 跨样本可区分度的调和平均），首次能稳定区分泛泛而谈与细节准确的音频模型输出。

**[MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)**

:   MedMosaic 用合成管道构造了一个覆盖生理声 + 真实/合成临床对话的医学音频 QA 基准（46,701 条 QA、10 种问题类型），系统评测 13 个音频/多模态模型，发现即使 Gemini-2.5-Pro 也只能拿到约 68.1% 加权准确率，揭示当代 LALM 在医学音频推理上的根本短板。

**[MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models](moshirag_asynchronous_knowledge_retrieval_for_full-duplex_speech_language_models.md)**

:   MoshiRAG 在 Moshi 这一全双工语音模型里加入一个特殊的 ⟨ret⟩ 触发 token，让模型边说边异步调用 LLM/搜索后端去取参考文档，利用"开口到关键词出现"的自然 keyword delay 把 2 秒以内的检索延迟完全藏起来，从而在 LlamaQ/WebQ/TriviaQA/HaluEval 上把语音模型的事实性拉到 GPT-4o Audio 量级，同时保留全双工实时性。

**[NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating](naaca_training-free_neuroauditory_attentive_cognitive_architecture_with_oscillat.md)**

:   用一套受皮层振荡启发的二维波动场（OWM）做实时显著性检测，给 Audio Language Model 在长音频上当一个"训练无关的注意力门"，只把真正显著的窗口送进 ALM，从而在 XD-Violence 上把 AP 从 53.5% 拉到 70.6%，同时减少约 40% 的 ALM 调用。

**[Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration](polyphonia_zero-shot_timbre_transfer_in_polyphonic_music_with_acoustic-informed_.md)**

:   Polyphonia 把 zero-shot 音色转换从单轨扩展到密集多轨混音：用盲源分离得到的 Ideal Ratio Mask（IRM）当外部声学先验，先在 pre-softmax 注意力 logit 里做"源插值 + 声学调制"，让目标声部（如人声）的频谱被新音色（如小提琴）替换的同时把背景伴奏严格保住，相比 SOTA 在 target alignment 上提升 15.5%。

**[Probing Cross-modal Information Hubs in Audio-Visual LLMs](probing_cross-modal_information_hubs_in_audio-visual_llms.md)**

:   作者用因果追踪 + 单模态主导框架揭示了音视频 LLM 中存在一类被称为"跨模态 sink token"的隐藏枢纽,绝大多数跨模态信息都凝聚在这些 token 上,据此提出训练免费的注意力放大策略显著缓解物体幻觉。
