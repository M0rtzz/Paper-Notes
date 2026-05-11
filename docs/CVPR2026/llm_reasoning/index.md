---
title: >-
  CVPR2026 LLM 推理方向12篇论文解读
description: >-
  12篇CVPR2026的 LLM 推理方向论文解读，涵盖推理、多模态、人脸/视线、少样本学习、自动驾驶、问答等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**📷 CVPR2026** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (37)](../../ACL2026/llm_reasoning/index.md) · [🔬 ICLR2026 (65)](../../ICLR2026/llm_reasoning/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/llm_reasoning/index.md) · [🧠 NeurIPS2025 (66)](../../NeurIPS2025/llm_reasoning/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/llm_reasoning/index.md) · [🧪 ICML2025 (15)](../../ICML2025/llm_reasoning/index.md)

🔥 **高频主题：** 推理 ×10 · 多模态 ×2

**[Beyond Geometry: Artistic Disparity Synthesis for Immersive 2D-to-3D](beyond_geometry_artistic_disparity_synthesis_for_immersive_2d-to-3d.md)**

:   提出"艺术视差合成"新范式（Art3D），将2D-to-3D转换目标从几何精度转向艺术表达，通过双路径架构解耦全局深度风格与局部艺术效果，从专业3D电影数据中学习导演意图。

**[E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought](e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_.md)**

:   构建首个面向中文电商海报的多维度质量评估框架 E-comIQ-ZH，包含18K专家标注数据集（含CoT推理链）、专用评估模型 E-comIQ-M（SFT+GRPO训练）和标准化基准 E-comIQ-Bench。

**[EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence](eaglevision_a_dual-stage_framework_with_bev-grounding-based_chain-of-thought_for.md)**

:   提出EagleVision双阶段框架，宏观感知阶段用语义-视角融合DPP(SPF-DPP)在SE(3)空间联合优化语义相关性和视角多样性选择关键帧，微观验证阶段让模型在BEV平面上主动查询新视角帧进行迭代空间CoT推理（假设→查看→验证闭环），查询策略纯RL训练无需人工标注，在VSI-Bench和SQA3D上达开源SOTA。

**[FaceCoT: Harnessing Chain-of-Thought Reasoning in MLLMs for Face Anti-Spoofing](facecot_cot_reasoning_face_anti_spoofing.md)**

:   构建了首个面向人脸反欺骗（FAS）的大规模 VQA 数据集 FaceCoT（108 万样本，覆盖 14 种攻击类型），包含六层级 CoT 推理标注；提出 CoT-Enhanced Progressive Learning (CEPL) 两阶段训练策略（先视觉增强再联合训练），在 11 个跨域基准上平均 AUC 提升 4.06%、HTER 降低 5.00%。

**[GRAZE: Grounded Refinement and Motion-Aware Zero-Shot Event Localization](graze_grounded_refinement_and_motion-aware_zero-shot_event_localization.md)**

:   提出GRAZE，一种完全无训练的时空事件定位管线——用Grounding DINO发现候选player-dummy交互对，通过运动感知的几何评分（位移幅度+方向余弦相似度）排序候选，再用SAM2掩码传播作为独立的像素级接触验证器（而非依赖检测置信度），配合两阶段后向精化恢复事件起始帧，在738个橄榄球练习视频上97.4%有效输出率、77.5%在±10帧内定位。

**[Latent Chain-of-Thought World Modeling for End-to-End Autonomous Driving](latent_chain-of-thought_world_modeling_for_end-to-end_autonomous_driving.md)**

:   LCDrive 提出潜在链式思考（Latent CoT）框架，用动作提议token和世界模型预测token替代自然语言CoT进行推理，通过冷启动+RL后训练实现更低延迟、更好轨迹质量的端到端自动驾驶。

**[Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)**

:   发现现有LVLM在CoT推理时实际上忽略了中间rationale的内容，提出 RED (Rationale-Enhanced Decoding)——将图像条件和rationale条件的next-token分布在logit层面相乘，理论上等价于KL约束奖励最大化的最优解，无需训练即可显著提升多模态推理准确率。

**[Reinforcing Structured Chain-of-Thought for Video Understanding](reinforcing_structured_chain-of-thought_for_video_understanding.md)**

:   提出 SDRL（Summary-Driven Reinforcement Learning），一种无需 SFT 的单阶段 RL 框架，通过结构化 CoT（Summarize→Think→Answer）和两个自监督机制（CVK 和 DVR）增强视频时序推理，在 7 个 VideoQA 基准上达到 SOTA。

**[Step-CoT: Stepwise Visual Chain-of-Thought for Medical Visual Question Answering](step-cot_stepwise_visual_chain-of-thought_for_medical_visual_question_answering.md)**

:   构建首个对齐临床诊断工作流的结构化多步CoT医学推理数据集Step-CoT（10K+病例/70K QA对），并提出基于图注意力网络的教师-学生框架实现逐步推理监督，提升Med-VQA的准确性和可解释性。

**[Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)**

:   本文系统分析了多模态 CoT 模型中幻觉的成因，发现"发散思维"（associative reasoning）是幻觉的核心触发因素，并提出基于视觉熵的免训练检测+解码干预策略，在 Object HalBench 上将 CHAIRS 降低超过 30%，同时保持甚至提升通用推理能力。

**[Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)**

:   本文提出 Hallucination-as-Cue 分析框架，通过三种模态特定腐蚀策略（空白图像、随机图像、文本移除）系统研究 RL 后训练对多模态推理模型的真实作用机制，发现即使在 100% 腐蚀视觉输入下 GRPO 训练仍能显著提升推理性能，挑战了"RL 训练能有效利用视觉信息"的主流假设。

**[VisRef: Visual Refocusing while Thinking Improves Test-Time Scaling in Multi-Modal Large Reasoning Models](visref_visual_refocusing_test_time_scaling.md)**

:   提出 VisRef，一个免训练的视觉重聚焦框架——在多模态大推理模型（MLRM）的推理过程中，通过行列式点过程（DPP）在每步自适应选择与当前推理状态语义相关且视觉覆盖多样的 token 子集并重新注入，同时用基于熵的停止准则防止过度推理，在固定计算预算下将视觉推理准确率提升最高 6.4%。
