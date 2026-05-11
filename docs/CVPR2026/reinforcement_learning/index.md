---
title: >-
  CVPR2026 强化学习方向19篇论文解读
description: >-
  19篇CVPR2026的强化学习方向论文解读，涵盖多模态、强化学习、推理、Agent、机器人等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**📷 CVPR2026** · **19** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (34)](../../ACL2026/reinforcement_learning/index.md) · [🔬 ICLR2026 (139)](../../ICLR2026/reinforcement_learning/index.md) · [🤖 AAAI2026 (71)](../../AAAI2026/reinforcement_learning/index.md) · [🧠 NeurIPS2025 (172)](../../NeurIPS2025/reinforcement_learning/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/reinforcement_learning/index.md) · [🧪 ICML2025 (81)](../../ICML2025/reinforcement_learning/index.md)

🔥 **高频主题：** 多模态 ×5 · 强化学习 ×4 · 推理 ×4 · Agent ×3 · 机器人 ×3

**[AceTone: Bridging Words and Colors for Conditional Image Grading](acetone_bridging_words_and_colors_for_conditional_image_grading.md)**

:   提出AceTone，首个支持文本和参考图像多模态条件色彩调色的统一框架，通过VQ-VAE将3D-LUT压缩为64个离散token，训练VLM预测LUT token序列，再用GRPO强化学习对齐色彩相似度和美学偏好，在风格迁移和指令调色上LPIPS改善50%。

**[Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)**

:   提出 TraceR1，一个两阶段 RL 框架：第一阶段通过轨迹级奖励优化让智能体学会"向前看几步"的前瞻性规划，第二阶段通过工具执行反馈做 grounded fine-tuning 来提升单步精度，在 7 个 GUI 和工具使用 benchmark 上取得了开源 SOTA。

**[AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization](anydoc_enhancing_document_generation_via_large-scale_htmlcss_data_synthesis_and_.md)**

:   AnyDoc 提出了一个基于统一 HTML/CSS 表示的通用文档生成框架，通过自动化数据合成管线构建 265K 文档数据集 DocHTML，结合 SFT 和高度感知强化学习（HARL）微调多模态大模型，在意图到文档、文档反渲染和元素到文档三个任务上超越 GPT-4o 等基线。

**[BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)**

:   提出 BRIDGE 系统，通过 FORGE（RL 训练的查询对齐模型）将噪声多模态查询蒸馏为检索优化的纯文本查询，配合 LENS 推理增强检索器，在 MM-BRIGHT 上达到 29.7 nDCG@10，作为插件进一步将 Nomic-Vision 提升到 33.3，超越最佳纯文本检索器。

**[CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)**

:   提出 CCCaption 双奖励强化学习框架，通过 completeness reward（基于多 MLLM 生成的视觉 query 集）和 correctness reward（基于 caption 分解后的子 query 幻觉检测）联合优化图像描述的完整性和正确性，2B 模型超越 32B 基线。

**[Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning](cross-modal_identity_mapping_minimizing_information_loss_in_modality_conversion_.md)**

:   提出 Cross-modal Identity Mapping (CIM)，通过分析用 caption 检索到的图像的表示一致性（GRC）和与源图像的相关性（QIR）来量化图像描述中的信息损失，将其作为 RL 奖励信号训练 LVLM 生成细粒度且精确的描述，无需额外标注。

**[GeoWorld: Geometric World Models](geoworld_geometric_world_models.md)**

:   GeoWorld 将预测式世界模型的潜在表征从欧氏空间映射到双曲流形上，通过 Hyperbolic JEPA 保持几何结构和层级关系，并提出 Geometric Reinforcement Learning 来优化多步规划，在 CrossTask 和 COIN 上实现了约 3% SR（3步）和 2% SR（4步）的提升。

**[GraspLDP: Towards Generalizable Grasping Policy via Latent Diffusion](graspldp_towards_generalizable_grasping_policy_via_latent_diffusion.md)**

:   提出 GraspLDP，将预训练抓取检测器的 grasp pose 先验和 graspness map 视觉线索注入潜在扩散策略框架，通过 VAE 编码的动作潜空间引导和自监督重建目标，显著提升抓取精度和泛化能力。

**[Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](lifelong_imitation_learning_multimodal_latent_rep.md)**

:   提出终身模仿学习框架，通过多模态潜在回放（MLR）在冻结编码器的特征空间中存储和回放紧凑表示，并引入增量特征调整（IFA）机制用角距离约束维持任务间可分性，在LIBERO基准上AUC提升10-17点、遗忘降低最多65%。

**[MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](msrl_scaling_generative_multimodal_reward_modeling.md)**

:   提出多阶段强化学习（MSRL）方法，通过先在大规模文本偏好数据上学习奖励推理能力，再逐步迁移到多模态任务，解决多模态奖励模型训练中标注数据稀缺的瓶颈问题，在 VL-RewardBench 上将准确率从 66.6% 提升至 75.9%。

**[RADAR: Closed-Loop Robotic Data Generation via Semantic Planning and Autonomous Causal Environment Reset](radar_closedloop_robotic_data_generation_via_seman.md)**

:   提出RADAR——一个完全自主的闭环机器人操作数据生成引擎，通过VLM语义规划+GNN策略执行+VQA成功评估+FSM驱动的LIFO因果逆序环境重置四个模块，仅需2-5个人工演示即可持续生成高保真操作数据，在仿真中复杂长horizon任务达到90%成功率。

**[ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering](reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)**

:   提出 ReAG，一个推理增强的多模态 RAG 方法，结合粗细粒度检索与 Critic 过滤模型减少噪声，并通过 GRPO 强化学习训练生成器进行显式推理，在知识密集型 VQA 上达到新 SOTA。

**[Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)**

:   提出 ReAL 和 CGRO 两个模块，通过提取 MLLM 自回归推理过程中的异常相关 token 并聚合其视觉注意力来生成像素级异常图，再通过一致性引导的强化学习对齐推理与视觉证据，实现仅凭图像级监督的端到端异常检测、定位与可解释推理。

**[Reinforce to Learn, Elect to Reason: A Dual Paradigm for Video Reasoning](reinforce_to_learn_elect_to_reason_a_dual_paradigm_for_video_reasoning.md)**

:   提出 RLER 双范式框架，训练阶段用 GRPO 配合三种新颖奖励（Frame-sensitive、Think-transparency、Anti-repetition）教模型生成结构化证据，推理阶段用无训练编排器在多候选之间基于证据一致性进行加权选举和自检，在 8 个视频基准上全面超越开源和 RL-based LMM，平均提升 6.3%，仅需约 3.1 个候选。

**[Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation](rethinking_camera_choice_an_empirical_study_on_fisheye_camera_properties_in_robo.md)**

:   首次系统性地对腕部鱼眼相机在机器人操作模仿学习中的特性进行实证研究，围绕空间定位、场景泛化和硬件泛化三个核心问题揭示了宽视场角的优势与局限，并提出 Random Scale Augmentation (RSA) 策略解决跨相机迁移中的尺度过拟合问题。

**[RoboAgent: Chaining Basic Capabilities for Embodied Task Planning](roboagent_chaining_basic_capabilities_for_embodied_task_planning.md)**

:   提出 RoboAgent，一种能力驱动的具身任务规划框架，用单个 VLM 同时实现调度器和 5 种基本能力（探索引导、物体定位、场景描述、动作解码、经验总结），通过三阶段训练（SFT + DAgger + 专家引导 RL）在 EB-ALFRED 和 ALFWorld 上达到 SOTA。

**[See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs](see_it_say_it_sorted_an_iterative_training-free_framework_for_visually-grounded_.md)**

:   提出Evidence-Constrained Reweighting Decoding（ECRD）框架：在LVLM解码时维护动态文本证据池，通过分布协商重加权候选token，不确定时自动调用轻量视觉决策器提取微证据，无需训练即可在多个LVLM上显著减少视觉幻觉、提升推理准确率。

**[Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement](seeing_is_improving_visual_feedback_for_iterative_text_layout_refinement.md)**

:   VFLM 提出一个利用视觉反馈进行迭代优化的布局生成框架，通过结合 OCR 准确率的视觉奖励模型和强化学习训练，使多模态大语言模型能够"看到"渲染结果并反复修正，在文本排版质量上显著超越仅生成代码的方法。

**[Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification](specificity-aware_reinforcement_learning_for_fine-grained_open-world_classificat.md)**

:   提出 SpeciaRL——一种特异性感知的强化学习框架，通过基于在线 rollout 最佳预测的动态奖励信号，引导推理型大型多模态模型在开放世界细粒度图像分类中同时提升预测的特异性和正确性。
