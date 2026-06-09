---
title: >-
  CVPR2026 时间序列方向6篇论文解读
description: >-
  6篇CVPR2026的时间序列方向论文解读，涵盖时序预测、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "时间序列"
  - "论文解读"
  - "论文笔记"
  - "时序预测"
  - "对齐/RLHF"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**📷 CVPR2026** · **6** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (32)](../../ICML2026/time_series/index.md) · [💬 ACL2026 (7)](../../ACL2026/time_series/index.md) · [🔬 ICLR2026 (36)](../../ICLR2026/time_series/index.md) · [🤖 AAAI2026 (30)](../../AAAI2026/time_series/index.md) · [🧠 NeurIPS2025 (51)](../../NeurIPS2025/time_series/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/time_series/index.md)

🔥 **高频主题：** 时序预测 ×3

**[Competition-Aware CPC Forecasting with Near-Market Coverage](competition-aware_cpc_forecasting_with_near-market_coverage.md)**

:   这篇论文把搜索广告中的 CPC 预测重新表述为“竞争状态部分不可观测”下的时间序列预测问题，用语义相似性、CPC 轨迹对齐和地理意图三个可观测代理去近似隐含竞争，再分别以协变量和图先验两种形式注入预测器，在中长期预测上显著优于纯自回归基线。

**[L2GTX: From Local to Global Time Series Explanations](l2gtx_from_local_to_global_time_series_explanations.md)**

:   L2GTX 提出一种完全模型无关的局部到全局解释方法，通过从 LOMATCE 局部解释中提取参数化时间事件原语（趋势/极值），跨实例合并冗余聚类并以子模优化选取代表性实例，最终聚合为简洁的类级别全局解释，在6个时序分类数据集上保持稳定的全局忠实度。

**[PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning](pfgnet_a_fully_convolutional_frequency-guided_peripheral_gating_network_for_effi.md)**

:   提出 PFGNet，一种纯卷积时空预测框架，通过像素级频率引导门控（PFG）动态调制多尺度大核外周响应并施加可学习中心抑制，模拟生物视觉的 center-surround 带通滤波机制，在 Moving MNIST、TaxiBJ、KTH、Human3.6M 四个基准上以极少参数和计算量达到 SOTA 或近 SOTA 性能。

**[SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval](sattc_structure-aware_label-free_test-time_calibration_for_cross-subject_eeg-to-.md)**

:   提出SATTC，一个无标签的测试时校准头，通过几何专家（被试自适应白化+自适应CSLS）和结构专家（互最近邻+双向top-k排名+类别流行度）的乘积专家融合，在冻结的EEG和图像编码器上直接操作相似度矩阵，显著改善跨被试EEG-to-image检索的Top-1精度并降低hubness效应。

**[Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks](stable_spike_dual_consistency_optimization_via_bitwise_and_operations_for_spikin.md)**

:   提出 Stable Spike 双一致性优化框架，利用硬件友好的 AND 位运算从多时间步脉冲图中解耦稳定脉冲骨架，并注入振幅感知脉冲噪声增强泛化，在超低延迟(T=2)下将神经形态物体识别精度提升最高 8.33%。

**[STCast: Adaptive Boundary Alignment for Global and Regional Weather Forecasting](stcast_adaptive_boundary_alignment_for_global_and_regional_weather_forecasting.md)**

:   提出STCast框架，通过Spatial-Aligned Attention（SAA）用可学习的全球-区域分布替代静态边界来自适应融合全球大气信息到区域预报，并用Temporal Mixture-of-Experts（TMoE）按月动态路由专家增强时序建模，在全球预报、高分辨率区域预报、台风路径预测和集合预测四个任务上全面超越现有方法。
