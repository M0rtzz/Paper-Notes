---
title: >-
  CVPR2026 音频/语音方向 19篇论文解读
description: >-
  19篇CVPR2026 音频/语音方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎵 音频/语音

**📷 CVPR2026** · **19** 篇论文解读

**[BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models](babyvlm-v2_toward_developmentally_grounded_pretraining_and_benchmarking_of_visio.md)**

:   提出BabyVLM-V2框架，从婴儿第一视角的SAYCam纵向语料构建三种格式预训练数据（768K图像对+181K视频对+63K交错序列），设计基于NIH Baby Toolbox®的DevCV Toolbox（10个发育认知任务），从零训练的紧凑模型在部分数学任务上超越GPT-4o，首次系统探索人工发育智能(ADI)。

**[BROTHER: Behavioral Recognition Optimized Through Heterogeneous Ensemble Regularization for Ambivalence and Hesitancy](brother_behavioral_recognition_optimized_through_heterogeneous_ensemble_regulari.md)**

:   提出一个高度正则化的多模态融合管线，通过视觉(SigLip2)、音频(HuBERT)、文本(F2LLM)及统计特征四模态的异质分类器委员会，结合带训练-验证差距惩罚的 PSO 硬投票集成，实现自然场景下矛盾与犹豫（A/H）行为的鲁棒视频级识别，在 ABAW10 测试集上取得 Macro F1 = 0.7465。

**[Cleaning the Pool: Progressive Filtering of Unlabeled Pools in Deep Active Learning](cleaning_the_pool_progressive_filtering_of_unlabeled_pools_in_deep_active_learni.md)**

:   提出 Refine 集成主动学习方法，通过两阶段策略——渐进过滤（多策略迭代精炼无标签池）+ 覆盖选择（从精炼池中选择多样性高价值样本）——在不预知最佳策略的情况下一致超越单一 AL 策略和现有集成方法。

**[Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)**

:   提出 MMHNet，一种基于层级结构和非因果 Mamba-2 的多模态层级网络，实现了在短片段（8秒）上训练、在长视频（5分钟以上）上生成高质量对齐音频的长度泛化能力，在 UnAV100 和 LongVale 基准上大幅超越现有方法。

**[GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization](gem-tfl_bridging_weak_and_full_supervision_for_forgery_localization_through_em-g.md)**

:   提出 GEM-TFL，通过两阶段分类-回归框架弥合弱监督与全监督之间的差距，用 EM 分解二元标签为多维潜在属性、训练无关的时序一致性精化、图扩散提案精化三大模块，在弱监督时序伪造定位上平均 mAP 提升 4-8%。

**[LaScA: Language-Conditioned Scalable Modelling of Affective Dynamics](lasca_language-conditioned_scalable_modelling_of_affective_dynamics.md)**

:   提出 LaScA 框架，利用大语言模型生成确定性语义词典为手工制作的面部和声学特征提供语义先验，通过冻结的句子编码器生成语义嵌入并与原始特征融合，在 Aff-Wild2 和 SEWA 数据集上的情感变化预测中一致性地超越纯特征基线，并在一致性、效率和可解释性上与端到端深度模型持平或更优。

**[Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding](omni-mmsi_toward_identity-attributed_social_interaction_understanding.md)**

:   提出 Omni-MMSI 任务——从原始音视频输入（而非预处理的 oracle 社交线索）理解多人社交交互，并设计 Omni-MMSI-R 参考引导流水线，通过工具生成身份归因社交线索 + 链式思维推理实现准确的社交交互理解。

**[OmniRet: Efficient and High-Fidelity Omni Modality Retrieval](omniret_efficient_and_high-fidelity_omni_modality_retrieval.md)**

:   提出首个支持文本-视觉-音频三模态组合查询的统一检索模型 OmniRet，通过共享媒体重采样器（Shared Media Resampler）提升计算效率，并引入注意力切片 Wasserstein 池化（ASWP）保留细粒度信息，在 13 个检索任务上取得 12 项领先。

**[OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)**

:   提出 Universal Holistic Audio Generation (UniHAGen) 任务和 OmniSonic 框架，通过 TriAttn-DiT 架构的三路交叉注意力和 MoE 门控机制，首次实现同时生成屏幕内/屏外环境声和人声的统一音频合成，在新构建的 UniHAGen-Bench 上全面超越 SOTA。

**[PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)**

:   提出PhaSR框架，通过双层物理先验对齐——全局级的PAN执行无参数Retinex分解抑制色彩偏差、局部级的GSRA利用差分注意力对齐DepthAnything深度先验和DINO-v2语义嵌入——实现从单光源直射阴影到多光源环境光场景的泛化阴影去除，在WSRD+和Ambient6K上达到SOTA且FLOPs最低。

**[Semantic Audio-Visual Navigation in Continuous Environments](semantic_audio-visual_navigation_in_continuous_environments.md)**

:   本文提出 SAVN-CE 任务，将语义音视觉导航扩展到连续3D环境中，并设计 MAGNet（记忆增强目标描述网络），通过融合历史上下文和自运动线索实现在目标声音消失后的稳健目标推理，成功率绝对提升最高达 12.1%。

**[Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion](solution_for_10th_competition_on_ambivalencehesitancy_ah_video_recognition_chall.md)**

:   针对第10届 ABAW 竞赛的矛盾/犹豫 (A/H) 视频识别任务，提出基于散度的多模态融合策略，通过计算视觉（AU）、音频（Wav2Vec 2.0）和文本（BERT）三个模态嵌入的逐对绝对差来显式建模跨模态冲突，在 BAH 数据集上以 Macro F1 0.6808 大幅超越基线 0.2827。

**[Talking Together: Synthesizing Co-Located 3D Conversations from Audio](talking_together_synthesizing_co-located_3d_conversations_from_audio.md)**

:   首次提出从单一混合音频流生成两个**共处同一3D空间**的对话参与者完整面部动画的方法，通过双流扩散架构（共享 U-Net + 跨注意力）、两阶段混合数据训练策略、LLM 驱动的文本-空间布局控制以及辅助眼神损失，实现自然的互视、转头和空间感知的双人对话3D动画合成。

**[Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](team_leya_in_10th_abaw_competition_multimodal_ambi.md)**

:   提出四模态（场景 VideoMAE + 人脸 EfficientNetB0 + 音频 Wav2Vec2.0/Mamba + 文本 EmotionDistilRoBERTa）融合管线，通过原型增强 Transformer 融合模块将各模态嵌入投影到共享 128 维空间并以原型分类辅助损失正则化，在 BAH 语料的最终测试集上以 5 模型集成达到 **71.43% Macro F1**，显著超越所有单模态基线。

**[Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](team_leya_in_10th_abaw_competition_multimodal_ambivalencehesitancy_recognition_a.md)**

:   提出面向第 10 届 ABAW 竞赛的多模态矛盾/犹豫（A/H）识别方法，整合场景、面部、音频和文本四种模态，通过 Transformer 融合模块和原型增强分类策略，最佳单模型 MF1 达 83.25%，最终测试集上五模型集成达 71.43%。

**[Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)**

:   提出 TSD 框架，将多模态特征显式分解为全局共享/成对共享/模态专属三个互补子空间，并通过子空间感知跨注意力融合模块自适应整合三层信息，在 CMU-MOSI/MOSEI 上全面 SOTA。

**[UNICBench: UNIfied Counting Benchmark for MLLM](unicbench_unified_counting_benchmark_for_mllm.md)**

:   推出UNICBench，首个统一的跨模态（图像/文本/音频）多层级计数基准，包含5,508+5,888+2,905共14,301个QA对及三级能力(Pattern/Semantic/Reasoning)×三级难度(Easy/Medium/Hard)分类，系统评估45个SOTA MLLM，揭示基本计数任务趋近但推理级和困难任务存在显著差距。

**[UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)**

:   提出首个统一的任意到任意交错多模态基准 UniM（31K 样本、7 种模态、30 个领域），配套三维评估体系和基于可追溯推理的智能体基线 UniMA，揭示现有 MLLM 在交错多模态范式下的严重不足。

**[Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods](unlocking_strong_supervision_a_data-centric_study_of_general-purpose_audio_pre-t.md)**

:   本文通过系统的数据中心实验证明音频预训练性能主要由标签/监督质量驱动而非模型设计，提出 Unified Tag System (UTS) 将语音、音乐、环境音统一到 800-3k 标签的高粒度词表中，UTS 训练的模型用 5 倍更少的数据在语音（VoxCeleb2）和音乐（MusicCaps）等域外任务上超越 AudioSet 基线。
