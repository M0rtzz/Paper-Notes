---
title: >-
  CVPR2026 LLM推理方向 13篇论文解读
description: >-
  13篇CVPR2026 LLM推理方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM推理

**📷 CVPR2026** · 共 **13** 篇

**[Beyond Geometry Artistic Disparity Synthesis For Immersive 2D-To-3D](beyond_geometry_artistic_disparity_synthesis_for_immersive_2d-to-3d.md)**

:   提出"艺术视差合成"新范式（Art3D），将2D-to-3D转换目标从几何精度转向艺术表达，通过双路径架构解耦全局深度风格与局部艺术效果，从专业3D电影数据中学习导演意图。

**[E-Comiq-Zh A Human-Aligned Dataset And Benchmark For Fine-Grained Evaluation Of ](e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_.md)**

:   构建首个面向中文电商海报的多维度质量评估框架 E-comIQ-ZH，包含18K专家标注数据集（含CoT推理链）、专用评估模型 E-comIQ-M（SFT+GRPO训练）和标准化基准 E-comIQ-Bench。

**[Eaglevision A Dual-Stage Framework With Bev-Grounding-Based Chain-Of-Thought For](eaglevision_a_dual-stage_framework_with_bev-grounding-based_chain-of-thought_for.md)**

:   提出EagleVision双阶段框架，宏观感知阶段用语义-视角融合DPP(SPF-DPP)在SE(3)空间联合优化语义相关性和视角多样性选择关键帧，微观验证阶段让模型在BEV平面上主动查询新视角帧进行迭代空间CoT推理（假设→查看→验证闭环），查询策略纯RL训练无需人工标注，在VSI-Bench和SQA3D上达开源SOTA。

**[Facecot Cot Reasoning Face Anti Spoofing](facecot_cot_reasoning_face_anti_spoofing.md)**

:   构建了首个面向人脸反欺骗（FAS）的大规模 VQA 数据集 FaceCoT（108 万样本，覆盖 14 种攻击类型），包含六层级 CoT 推理标注（从全局描述到局部推理到最终结论）；同时提出 CoT-Enhanced Progressive Learning (CEPL) 两阶段训练策略，在 11 个基准数据集上平均 AUC 提升 4.06%、HTER 降低 5.00%，超越所有 SOTA 方法。

**[Graze Grounded Refinement And Motion-Aware Zero-Shot Event Localization](graze_grounded_refinement_and_motion-aware_zero-shot_event_localization.md)**

:   提出GRAZE，一种完全无训练的时空事件定位管线——用Grounding DINO发现候选player-dummy交互对，通过运动感知的几何评分（位移幅度+方向余弦相似度）排序候选，再用SAM2掩码传播作为独立的像素级接触验证器（而非依赖检测置信度），配合两阶段后向精化恢复事件起始帧，在738个橄榄球练习视频上97.4%有效输出率、77.5%在±10帧内定位。

**[Graze Grounded Refinement And Motion-Aware Zero-Shot Generation](graze_grounded_refinement_and_motion-aware_zero-shot_generation.md)**

:   提出 GRAZE，一个无需训练的管线，利用 Grounding DINO 发现候选交互、SAM2 掩码重叠作为像素级接触验证器，在 738 段美式橄榄球训练视频中实现 97.4% 覆盖率和 ±10 帧内 77.5% 的接触起始帧定位精度。

**[Harnessing Chain-Of-Thought Reasoning In Multimodal Large Language Models For Fa](harnessing_chain-of-thought_reasoning_in_multimodal_large_language_models_for_fa.md)**

:   构建首个面向人脸反欺骗(FAS)的CoT-VQA数据集 FaceCoT（108万样本，14种攻击类型），并提出分两阶段渐进学习策略 CEPL，在11个FAS基准上平均AUC提升4.06%、HTER降低5.00%。

**[Rationale-Enhanced Decoding For Multi-Modal Chain-Of-Thought](rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)**

:   发现现有LVLM在CoT推理时实际上忽略了中间rationale的内容，提出 RED (Rationale-Enhanced Decoding)——将图像条件和rationale条件的next-token分布在logit层面相乘，理论上等价于KL约束奖励最大化的最优解，无需训练即可显著提升多模态推理准确率。

**[Red Rationale Enhanced Decoding Cot](red_rationale_enhanced_decoding_cot.md)**

:   发现现有 LVLM 在多模态 CoT 推理中会忽略生成的 rationale 内容（图像 token 主导注意力），提出 Rationale-Enhanced Decoding (RED)——将 CoT 重新表述为 KL 约束的 rationale 条件对数似然奖励最大化问题，最优解为将图像条件分布 $p(y|x,q)$ 和 rationale 条件分布 $p(y|r,q)^\lambda$ 相乘，无需训练即可显著提升多个基准上的推理性能。

**[Reinforcing Structured Chain-Of-Thought For Video Understanding](reinforcing_structured_chain-of-thought_for_video_understanding.md)**

:   提出 SDRL（Summary-Driven Reinforcement Learning），一种无需 SFT 的单阶段 RL 框架，通过结构化 CoT（Summarize→Think→Answer）和两个自监督机制（CVK 和 DVR）增强视频时序推理，在 7 个 VideoQA 基准上达到 SOTA。

**[Step-Cot Stepwise Visual Chain-Of-Thought For Medical Visual Question Answering](step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)**

:   构建首个对齐临床诊断工作流的结构化多步CoT医学推理数据集Step-CoT（10K+病例/70K QA对），并提出基于图注意力网络的教师-学生框架实现逐步推理监督，提升Med-VQA的准确性和可解释性。

**[Visref Visual Refocusing Test Time Scaling](visref_visual_refocusing_test_time_scaling.md)**

:   发现多模态推理模型在延长推理时会逐渐丢失对视觉token的注意力，提出VisRef在推理过程中主动重新注入与当前推理上下文语义相关的视觉token核心子集，在固定计算预算下比现有方法提升最高6.4%。

**[Visref Visual Refocusing While Thinking Improves Test-Time Scaling In Multi-Moda](visref_visual_refocusing_while_thinking_improves_test-time_scaling_in_multi-moda.md)**

:   提出 VisRef，一个无需训练的视觉重聚焦框架，在多模态推理每一步通过 DPP 选择与当前推理状态相关且多样的视觉token子集重新注入，配合基于熵的自适应停止准则，在三个视觉推理基准上比文本自反思方法提升最高6.4%。
