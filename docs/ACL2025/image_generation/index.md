---
title: >-
  ACL2025 图像生成方向 7篇论文解读
description: >-
  7篇ACL2025 图像生成方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**💬 ACL2025** · **7** 篇论文解读

**[A Unified Agentic Framework for Evaluating Conditional Image Generation](a_unified_agentic_framework_for_evaluating_conditional_image_generation.md)**

:   提出 CIGEval，一个基于大型多模态模型（LMM）的统一 Agent 评估框架，通过工具集成（Grounding、Highlight、Difference、Scene Graph）和分而治之的评估策略，在 7 种条件图像生成任务上达到与人类标注者相当的相关性（0.4625 vs 人类间 0.47），且仅用 2.3K 训练数据微调 7B 模型即超越 GPT-4o 版 SOTA。

**[D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Models](d-gen_automatic_distractor_generation_and_evaluation_for_reliable_assessment_of_.md)**

:   提出 D-GEN——首个开源干扰项生成模型（LLaMA微调，8B/70B），自动将开放式评测题转为多选题格式，配套排名对齐+熵分析两种评估方法验证干扰项质量，在 MMLU 上 Spearman's ρ=0.99 保持模型排名一致性。

**[Planning with Diffusion Models for Target-Oriented Dialogue Systems](difftod_diffusion_dialogue_planning.md)**

:   DiffTOD 将对话规划建模为轨迹生成问题，利用掩码扩散语言模型实现非顺序对话规划，并设计三种引导机制（词级/语义级/搜索级）灵活控制对话朝目标推进，在谈判/推荐/闲聊三种场景上显著超越基线。

**[FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation](flashaudio_rectified_flow_tta.md)**

:   将整流流（Rectified Flow）引入文本转音频生成，通过双焦采样器优化时间步分布、不混溶流减少数据-噪声总距离、锚定优化修正 CFG 引导误差，实现单步生成 FAD=1.49 超越百步扩散模型，生成速度达实时 400 倍。

**[Multimodal Pragmatic Jailbreak on Text-to-image Models](multimodal_pragmatic_jailbreak_on_text-to-image_models.md)**

:   提出"多模态语用越狱"（Multimodal Pragmatic Jailbreak）新型攻击方式，通过让T2I模型生成包含视觉文字的图像，使得图像内容和文字内容单独看都安全但组合后产生不安全内容，揭示了所有测试模型（包括DALL·E 3）均受此攻击影响。

**[OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching](ozspeech_one-step_zero-shot_speech_synthesis_with_learned-prior-conditioned_flow.md)**

:   提出OZSpeech，首个将最优传输条件流匹配(OT-CFM)与学习先验分布相结合实现单步采样的零样本TTS系统，在内容准确性(WER)、推理速度和模型大小上均大幅领先现有方法。

**[R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching](rvc_rhythm_voice_conversion.md)**

:   R-VC 是首个实现节奏可控的零样本语音转换系统，通过 Mask Transformer 时长模型建模目标说话人的节奏风格，结合 Shortcut Flow Matching 的 DiT 解码器实现仅 2 步采样的高效高质量语音生成，在 LibriSpeech 上 WER 3.51、说话人相似度 0.930。
