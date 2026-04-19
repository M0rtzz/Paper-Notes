---
title: >-
  ICCV2025 强化学习方向 6篇论文解读
description: >-
  6篇ICCV2025 强化学习方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**📹 ICCV2025** · **6** 篇论文解读

**[Embodied Navigation with Auxiliary Task of Action Description Prediction](embodied_navigation_with_auxiliary_task_of_action_description_prediction.md)**

:   DescRL 将动作描述生成作为强化学习导航的辅助任务，通过从预训练的视觉-语言模型蒸馏知识来训练 ADPredictor，使导航智能体在生成可解释动作描述的同时提升导航性能，在语义音频-视觉导航（SAVNav）等多个任务上实现 SOTA。

**[mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs](mdp3_a_training-free_approach_for_list-wise_frame_selection_in_video-llms.md)**

:   提出 mDP3，一种免训练、模型无关的视频帧选择方法，通过条件高斯核在 RKHS 中估计帧相似度，结合行列式点过程（DPP）捕获查询相关性和列表级多样性，再通过马尔可夫决策过程（MDP）建模时序性，在多个长视频 benchmark 上以仅 8 帧输入显著超越均匀采样和现有帧选择方法。

**[NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation](navq_learning_a_q-model_for_foresighted_vision-and-language_navigation.md)**

:   提出 NavQ，一种前瞻性 VLN 智能体，通过 Q-model 在单次前向传播中预测每个候选动作的长期未来语义聚合特征（Q-feature），结合 A* 式搜索策略在目标导向导航中取得显著提升。

**[Progressor: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement](progressor_a_perceptually_guided_reward_estimator_with_self-supervised_online_re.md)**

:   提出Progressor框架，从无标注视频中自监督学习任务无关的奖励函数，通过预测任务进度分布提供稠密奖励信号，并在在线RL训练中通过对抗性push-back策略应对分布偏移问题。

**[R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization](r1-onevision_advancing_generalized_multimodal_reasoning_through_cross-modal_form.md)**

:   提出 R1-Onevision，通过跨模态推理管线将图像转换为形式化文本表示，结合 SFT + 基于规则的强化学习（GRPO）的两阶段后训练策略，显著提升视觉语言模型的多模态推理能力，在多个数学推理基准上超越 GPT-4o。

**[RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment](reinforcement_learning-guided_data_selection_via_redundancy_assessment.md)**

:   提出 RL-Selector，引入 ε-sample cover 概念量化样本冗余度，将数据选择建模为强化学习过程，通过轻量 A2C 策略网络自适应优化选择策略，在多个基准数据集上以更少数据达到接近甚至超越全量训练的泛化性能。
