---
title: >-
  AAAI2026 多语言/翻译方向 10篇论文解读
description: >-
  10篇AAAI2026 多语言/翻译方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🌐 多语言/翻译

**🤖 AAAI2026** · **10** 篇论文解读

**[Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)**

:   本文综合多项实证研究，揭示LLM安全机制在低资源语言和代码混合场景下的严重失效，并提出基于参数高效安全引导、文化驱动偏好数据和社区参与式对齐的资源感知蓝图。

**[Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation](consensus-aligned_neuron_efficient_fine-tuning_large_language_models_for_multi-d.md)**

:   提出 CANEFT，通过互信息（MI）识别 LLM 中跨域一致对齐的神经元（consensus-aligned neurons），仅微调这些神经元即可实现多域机器翻译的高效适应，在 3 个 LLM、10 个翻译域上超越 LoRA 等 PEFT 基线，且无需额外参数。

**[Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)**

:   本文提出LAHIS方法，仅需一次前向-后向传播即可高效识别多语言LLM中的语言特异性和语言通用性注意力头，并展示了通过调控这些头来实现跨语言注意力转移、缓解非目标语言生成问题，以及仅用14-20个可训练参数就能提升多语言QA性能的能力。

**[GloCTM: Cross-Lingual Topic Modeling via a Global Context Space](gloctm_cross-lingual_topic_modeling_via_a_global_context_space.md)**

:   提出GloCTM，通过双路径VAE架构（局部语言路径+全局上下文路径）结合Polyglot Augmentation（跨语言近邻词扩充输入）、KL散度内部对齐、统一解码器结构对齐和CKA语义对齐四重机制，在3个跨语言数据集上全面超越现有方法的主题质量和跨语言对齐度。

**[How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective](how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per.md)**

:   提出三元神经元分类（语言特定/语言相关/通用），将 LLM 多语言推理分为四阶段分析，发现多语言对齐通过增加语言相关神经元（减少语言特定神经元）来提升性能，且在未训练语言上也产生"自发多语言对齐"效应。

**[Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering](mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_.md)**

:   通过激活转向（activation steering）技术缓解 LLM 中的内容效应偏见——模型将内容可信度与形式逻辑有效性混淆的问题，提出 K-CAST（基于 kNN 的条件激活转向）方法，在不响应静态转向的模型上实现高达 15% 的形式推理准确率提升。

**[NADIR: Differential Attention Flow for Non-Autoregressive Transliteration in Indic Languages](nadir_differential_attention_flow_for_non-autoregressive_transliteration_in_indi.md)**

:   提出 NADIR，一种结合差分 Transformer 和混合专家（MoE）的非自回归（NAR）多语言音译架构，在印度语言音译任务上实现了 13× 以上的推理加速，同时将 NAR 模型的幻觉错误（重复、替换、遗漏、插入）大幅降低，缩小了与自回归模型之间的精度差距。

**[STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)**

:   提出 STELLAR 框架，通过语言自适应字形编码器和合成预训练+真实微调的两阶段训练策略，实现韩语/阿拉伯语/日语等低资源语言的场景文本编辑，并提出可解释的 TAS 指标无需 ground truth 评估字体/颜色/背景风格保持，韩语识别准确率从基线最高 22.1% 飙升至 80.4%。

**[ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect-to-Standard Translation](vidia2std_a_parallel_corpus_and_methods_for_low-resource_vietnamese_dialect-to-s.md)**

:   ViDia2Std 构建了首个覆盖越南全部 63 个省份的手工标注越南语方言-标准语平行语料库（13,000+ 句对），并评估了多种 seq2seq 模型在方言归一化任务上的表现，证明方言归一化作为预处理步骤能显著提升机器翻译和情感分析等下游任务的性能。

**[X-MuTest: A Multilingual Benchmark for Explainable Hate Speech Detection](x-mutest_a_multilingual_benchmark_for_explainable_hate_speech_detection_and_a_no.md)**

:   提出 X-MuTest，一个多语言可解释仇恨言论检测基准，覆盖多种语言和文化背景，评估 LLM 不仅检测仇恨言论的能力，更关注其提供可解释性理由的能力，发现当前模型在多语言和跨文化场景中存在显著性能差异。
