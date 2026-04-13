---
title: >-
  CVPR2026 强化学习方向 15篇论文解读
description: >-
  15篇CVPR2026 强化学习方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**📷 CVPR2026** · 共 **15** 篇

**[Acetone Bridging Words And Colors For Conditional Image Grading](acetone_bridging_words_and_colors_for_conditional_image_grading.md)**

:   提出AceTone，首个支持文本和参考图像多模态条件色彩调色的统一框架，通过VQ-VAE将3D-LUT压缩为64个离散token，训练VLM预测LUT token序列，再用GRPO强化学习对齐色彩相似度和美学偏好，在风格迁移和指令调色上LPIPS改善50%。

**[Anticipatory Planning For Multimodal Ai Agents](anticipatory_planning_for_multimodal_ai_agents.md)**

:   提出 TraceR1，一个两阶段 RL 框架：第一阶段通过轨迹级奖励优化让智能体学会"向前看几步"的前瞻性规划，第二阶段通过工具执行反馈做 grounded fine-tuning 来提升单步精度，在 7 个 GUI 和工具使用 benchmark 上取得了开源 SOTA。

**[Anydoc Enhancing Document Generation Via Large-Scale Htmlcss Data Synthesis And ](anydoc_enhancing_document_generation_via_large-scale_htmlcss_data_synthesis_and_.md)**

:   AnyDoc 提出了一个基于统一 HTML/CSS 表示的通用文档生成框架，通过自动化数据合成管线构建 265K 文档数据集 DocHTML，结合 SFT 和高度感知强化学习（HARL）微调多模态大模型，在意图到文档、文档反渲染和元素到文档三个任务上超越 GPT-4o 等基线。

**[Bridge Multimodal-To-Text Retrieval Via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)**

:   提出 BRIDGE 系统，通过 FORGE（RL 训练的查询对齐模型）将噪声多模态查询蒸馏为检索优化的纯文本查询，配合 LENS 推理增强检索器，在 MM-BRIGHT 上达到 29.7 nDCG@10，作为插件进一步将 Nomic-Vision 提升到 33.3，超越最佳纯文本检索器。

**[Cccaption Dual-Reward Reinforcement Learning For Complete And Correct Image Capt](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)**

:   提出 CCCaption 双奖励强化学习框架，通过 completeness reward（基于多 MLLM 生成的视觉 query 集）和 correctness reward（基于 caption 分解后的子 query 幻觉检测）联合优化图像描述的完整性和正确性，2B 模型超越 32B 基线。

**[Cross-Modal Identity Mapping Minimizing Information Loss In Modality Conversion ](cross-modal_identity_mapping_minimizing_information_loss_in_modality_conversion_.md)**

:   提出 Cross-modal Identity Mapping (CIM)，通过分析用 caption 检索到的图像的表示一致性（GRC）和与源图像的相关性（QIR）来量化图像描述中的信息损失，将其作为 RL 奖励信号训练 LVLM 生成细粒度且精确的描述，无需额外标注。

**[Graspldp Towards Generalizable Grasping Policy Via Latent Diffusion](graspldp_towards_generalizable_grasping_policy_via_latent_diffusion.md)**

:   提出 GraspLDP，将预训练抓取检测器的 grasp pose 先验和 graspness map 视觉线索注入潜在扩散策略框架，通过 VAE 编码的动作潜空间引导和自监督重建目标，显著提升抓取精度和泛化能力。

**[Lifelong Imitation Learning Multimodal Latent Rep](lifelong_imitation_learning_multimodal_latent_rep.md)**

:   提出终身模仿学习框架，通过多模态潜在回放（MLR）在冻结编码器的特征空间中存储和回放紧凑表示，并引入增量特征调整（IFA）机制用角距离约束维持任务间可分性，在LIBERO基准上AUC提升10-17点、遗忘降低最多65%。

**[Lifelong Imitation Learning With Multimodal Latent Replay And Incremental Adjust](lifelong_imitation_learning_with_multimodal_latent_replay_and_incremental_adjust.md)**

:   提出终身模仿学习框架，通过 Multimodal Latent Replay（在冻结编码器的潜空间中存储和回放紧凑多模态特征）和 Incremental Feature Adjustment（基于角距离的自适应间隔约束防止任务间表示漂移），在 LIBERO 基准上实现 AUC 提升 10-17 点、遗忘减少 65%。

**[Linking Perception Confidence And Accuracy In Mllms](linking_perception_confidence_and_accuracy_in_mllms.md)**

:   揭示 MLLM 的严重置信度失校准问题（视觉输入退化时准确率暴跌但置信度不变），提出 CDRL（基于原始-噪声图像对的置信度驱动 RL）进行感知敏感性训练，并利用校准后的置信度实现自适应测试时缩放（CA-TTS），在四个基准上平均提升 8.8%。

**[Msrl Scaling Generative Multimodal Reward Modeling](msrl_scaling_generative_multimodal_reward_modeling.md)**

:   提出多阶段强化学习（MSRL）方法，通过先在大规模文本偏好数据上学习奖励推理能力，再逐步迁移到多模态任务，解决多模态奖励模型训练中标注数据稀缺的瓶颈问题，在 VL-RewardBench 上将准确率从 66.6% 提升至 75.9%。

**[Msrl Scaling Generative Multimodal Reward Modeling Via Multi-Stage Reinforcement](msrl_scaling_generative_multimodal_reward_modeling_via_multi-stage_reinforcement.md)**

:   提出MSRL(Multi-Stage Reinforcement Learning)，通过多阶段RL扩展生成式多模态奖励建模——先在大规模文本偏好数据(400K)上做RL学习通用奖励推理能力，再经caption-based RL和跨模态知识蒸馏向多模态迁移，最后用少量多模态偏好数据微调适配，无需额外多模态标注即在VL-RewardBench上从66.6%提升到75.9%、GenAI-Bench上从70.2%到75.7%。

**[Reag Reasoning-Augmented Generation For Knowledge-Based Visual Question Answerin](reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)**

:   提出 ReAG，一个推理增强的多模态 RAG 方法，结合粗细粒度检索与 Critic 过滤模型减少噪声，并通过 GRPO 强化学习训练生成器进行显式推理，在知识密集型 VQA 上达到新 SOTA。

**[Rethinking Camera Choice An Empirical Study On Fisheye Camera Properties In Robo](rethinking_camera_choice_an_empirical_study_on_fisheye_camera_properties_in_robo.md)**

:   首次系统性地对腕部鱼眼相机在机器人操作模仿学习中的特性进行实证研究，围绕空间定位、场景泛化和硬件泛化三个核心问题揭示了宽视场角的优势与局限，并提出 Random Scale Augmentation (RSA) 策略解决跨相机迁移中的尺度过拟合问题。

**[See It Say It Sorted An Iterative Training-Free Framework For Visually-Grounded ](see_it_say_it_sorted_an_iterative_training-free_framework_for_visually-grounded_.md)**

:   提出一个无需训练的迭代框架，通过"看到-说出-排序"的循环过程增强大视觉语言模型的视觉定位多模态推理能力，在多个推理任务上显著提升性能。
