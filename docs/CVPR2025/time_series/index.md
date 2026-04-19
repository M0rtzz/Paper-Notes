---
title: >-
  CVPR2025 时间序列方向 5篇论文解读
description: >-
  5篇CVPR2025 时间序列方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**📷 CVPR2025** · **5** 篇论文解读

**[Competition-Aware CPC Forecasting with Near-Market Coverage](competition-aware_cpc_forecasting_with_near-market_coverage.md)**

:   将付费搜索CPC预测重构为"部分可观测竞争下的预测"问题，通过语义邻域（Transformer嵌入）、行为邻域（DTW对齐）和地理意图三类竞争代理逼近不可观测的竞争状态，在1811个关键词×127周的Google Ads数据上显示竞争感知增强在中长期预测（6/12周）上显著优于单变量和弱上下文baseline。

**[DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification](dejavid_encoder-agnostic_learned_temporal_matching_for_video_classification.md)**

:   提出 DejaVid，一种编码器无关的轻量级视频分类增强方法：将视频表示为变长时序嵌入序列 (TSE) 而非单个嵌入，通过学习每个时间步、每个特征维度的重要性权重，结合改进的可微分 DTW 算法做时序对齐分类，仅增加 <1.8% 参数就在 SSV2 达到 77.2%、K400 达到 89.1% 的 SOTA。

**[FLAVC: Learned Video Compression with Feature Level Attention](flavc_learned_video_compression_with_feature_level_attention.md)**

:   提出 FLAVC，在学习型视频压缩框架中引入特征级注意力机制，在特征域（而非像素域）中更高效地利用运动估计和时空上下文信息。

**[Learning Extremely High Density Crowds as Active Matters](learning_extremely_high_density_crowds_as_active_matters.md)**

:   将极高密度人群类比为物理学中的"主动物质"（active matter），从质量参差的野外视频中学习人群的集体动力学行为模式，用于人群分析和预测。

**[Reasoning in Visual Navigation of End-to-end Trained Agents: A Dynamical Systems Approach](reasoning_in_visual_navigation_of_end-to-end_trained_agents_a_dynamical_systems_.md)**

:   通过262个真实机器人导航episode的大规模实验，深入分析端到端RL训练的导航智能体内部涌现出的推理能力——包括类Kalman滤波的动力学模型、场景结构的潜在记忆、有限水平的规划能力以及与长期规划相关的价值函数。
