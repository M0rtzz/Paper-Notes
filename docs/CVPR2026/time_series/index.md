---
title: >-
  CVPR2026 时间序列方向 5篇论文解读
description: >-
  5篇CVPR2026 时间序列方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**📷 CVPR2026** · **5** 篇论文解读

**[A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)**

:   提出 DeltaTok 将连续帧的 VFM 特征差压缩为单个 delta token，配合 Best-of-Many 训练的 DeltaWorld 在单次前向传播中高效生成多样化未来预测，参数量仅为 Cosmos 的 1/35、FLOPs 仅为 1/2000，但在密集预测任务上表现更优。

**[Competition-Aware CPC Forecasting with Near-Market Coverage](competition-aware_cpc_forecasting_with_near-market_coverage.md)**

:   这篇论文把搜索广告中的 CPC 预测重新表述为“竞争状态部分不可观测”下的时间序列预测问题，用语义相似性、CPC 轨迹对齐和地理意图三个可观测代理去近似隐含竞争，再分别以协变量和图先验两种形式注入预测器，在中长期预测上显著优于纯自回归基线。

**[Competition-Aware CPC Forecasting with Near-Market Coverage](competitionaware_cpc_forecasting_with_nearmarket_c.md)**

:   将付费搜索广告中的 CPC（每次点击成本）预测重新定义为**部分竞争可观测性**问题，通过语义邻域、DTW 行为邻域和地理意图三类竞争代理信号，结合时序基础模型（Chronos-2/TimeGPT/Moirai）和时空 GNN，在 1,811 条关键词序列上实现了中长期预测精度的显著提升。

**[PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning](pfgnet_a_fully_convolutional_frequency-guided_peripheral_gating_network_for_effi.md)**

:   提出 PFGNet，一种纯卷积时空预测框架，通过像素级频率引导门控（PFG）动态调制多尺度大核外周响应并施加可学习中心抑制，模拟生物视觉的 center-surround 带通滤波机制，在 Moving MNIST、TaxiBJ、KTH、Human3.6M 四个基准上以极少参数和计算量达到 SOTA 或近 SOTA 性能。

**[STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)**

:   提出STCast框架，通过Spatial-Aligned Attention（SAA）用可学习的全球-区域分布替代静态边界来自适应融合全球大气信息到区域预报，并用Temporal Mixture-of-Experts（TMoE）按月动态路由专家增强时序建模，在全球预报、高分辨率区域预报、台风路径预测和集合预测四个任务上全面超越现有方法。
