---
title: >-
  CVPR2025 强化学习方向 6篇论文解读
description: >-
  6篇CVPR2025 强化学习方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎮 强化学习

**📷 CVPR2025** · **6** 篇论文解读

**[CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos](citywalker_learning_embodied_urban_navigation_from_web-scale_videos.md)**

:   利用互联网上超过 2000 小时的城市步行和驾驶视频，通过视觉里程计 (VO) 自动提取动作标签进行大规模模仿学习，训练出能在复杂动态城市环境中导航的具身智能体，真实部署成功率达 77.3%，显著超越现有方法。

**[Decision SpikeFormer: Spike-Driven Transformer for Decision Making](decision_spikeformer_spike-driven_transformer_for_decision_making.md)**

:   提出 DSFormer，首个用于离线强化学习的脉冲驱动 Transformer，设计了时序脉冲自注意力 (TSSA) 和位置脉冲自注意力 (PSSA) 来捕获 RL 中的时序/位置依赖，并引入渐进式阈值依赖批归一化 (PTBN) 解决归一化与脉冲特性的冲突，在 D4RL 基准上超越 ANN 对手且节省 78.4% 能耗。

**[Gazing at Rewards: Eye Movements as a Lens into Human and AI Decision-Making in Hybrid Visual Foraging](gazing_at_rewards_eye_movements_as_a_lens_into_human_and_ai_decision-making_in_h.md)**

:   提出Visual Forager（VF）模型，通过目标特征调制、目标价值调制和ViT-based Actor-Critic决策网络模拟人类混合视觉搜索任务中的眼动策略，在归一化得分上达到72.6%（人类87.4%），扫视大小仅差0.01°（4.06° vs 人类4.05°），首次揭示目标价值和出现率如何联合影响人类搜索决策。

**[Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning](neural_motion_simulator_pushing_the_limit_of_world_models_in_reinforcement_learn.md)**

:   提出 MoSim，一个基于刚体动力学先验和 Neural ODE 的世界模型，可在物理状态空间中进行高精度长时域预测，首次实现零样本强化学习——不需任何真实环境交互即可训练策略。

**[SkillMimic: Learning Basketball Interaction Skills from Demonstrations](skillmimic_learning_basketball_interaction_skills_from_demonstrations.md)**

:   提出 SkillMimic，一个纯数据驱动的框架，通过统一的 HOI 模仿奖励（特别是创新的接触图奖励）从动捕数据中学习多样的篮球交互技能，并通过高层控制器组合技能实现连续得分等复杂长程任务。

**[ThinkStream: Thinking in Streaming Video](thinking_in_streaming_video.md)**

:   提出 ThinkStream，采用 Watch-Think-Speak 范式实现流式视频的实时连续推理，通过 RCSM（推理压缩流式记忆）将推理 trace 作为紧凑语义锚点替代旧视觉 token，配合 Streaming RLVR 训练策略，在保持低延迟/低内存的同时超越现有在线视频模型。
