---
title: >-
  AAAI2026 时间序列方向 31篇论文解读
description: >-
  31篇AAAI2026 时间序列方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🤖 AAAI2026** · **31** 篇论文解读

**[A Unified Shape-Aware Foundation Model for Time Series Classification](a_unified_shape-aware_foundation_model_for_time_series_class.md)**

:   提出 UniShape——一个面向时间序列分类的基础模型，通过 shape-aware adapter 自适应聚合多尺度判别性子序列（shapelet），并结合原型对比预训练在实例和 shape 两个层面学习可迁移的 shapelet 表示，在 128 个 UCR 数据集上以 3.1M 参数达到 SOTA（平均准确率 87.08%），同时提供良好的分类可解释性。

**[AirDDE: Multifactor Neural Delay Differential Equations for Air Quality Forecasting](airdde_multifactor_neural_delay_differential_equations_for_air_quality_forecasti.md)**

:   首个将神经延迟微分方程（NDDE）引入空气质量预测的框架，通过记忆增强注意力模块和物理引导的延迟演化函数，对污染物连续时间传播中的延迟效应进行建模，在三个数据集上平均 MAE 降低 8.79%。

**[iTimER: Reconstruction Error-Guided Irregularly Sampled Time Series Representation Learning](beyond_observations_reconstruction_error-guided_irregularly_sampled_time_series_.md)**

:   提出 iTimER，利用模型自身的重建误差分布作为学习信号——从观测点估计误差分布后采样生成未观测时刻的伪观测值，通过 Wasserstein 距离对齐观测/伪观测区域的误差分布 + 对比学习，在不规则采样时序的分类、插值、预测任务上全面超越 SOTA。

**[C3RL: Rethinking the Combination of Channel-independence and Channel-mixing from Representation Learning](c3rl_rethinking_the_combination_of_channel-independence_and_channel-mixing_from_.md)**

:   提出 C3RL，基于 SimSiam 对比学习框架将通道独立（CI）和通道混合（CM）策略视为同一数据的两个转置视图构建正样本对，通过孪生网络联合表示学习和预测学习，将 CI 模型的最佳性能率从 43.6% 提升到 81.4%，CM 模型从 23.8% 提升到 76.3%。

**[Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj](coherent_multi-agent_trajectory_forecasting_in_team_sports_with_causaltraj.md)**

:   提出CausalTraj——一种时序因果、基于似然的多智能体轨迹预测模型，通过逐步自回归建模智能体间时空交互，在NBA、篮球和橄榄球数据集上实现了联合指标（minJADE/minJFDE）的最优结果，同时保持有竞争力的单智能体精度。

**[CometNet: Contextual Motif-guided Long-term Time Series Forecasting](cometnet_contextual_motif-guided_long-term_time_series_forecasting.md)**

:   提出 CometNet，通过从完整历史序列中提取循环出现的"上下文 motif"构建 motif 库，再用 motif 引导的 MoE 架构动态关联当前窗口与相关motif进行预测，突破了有限回看窗口的感受野瓶颈，在8个数据集上显著超越 TimeMixer++、iTransformer 等 SOTA。

**[Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)**

:   提出 CONFETTI，一种面向多变量时间序列（MTS）分类的多目标反事实解释方法，通过结合类激活图（CAM）引导的子序列提取与 NSGA-III 多目标优化，在预测置信度、稀疏性和接近度三个目标间实现最优平衡，在 7 个 UEA 数据集上全面超越现有方法。

**[DeepBooTS: Dual-Stream Residual Boosting for Drift-Resilient Time-Series Forecasting](deepboots_dual-stream_residual_boosting_for_drift-resilient_time-series_forecast.md)**

:   提出 DeepBooTS，通过偏差-方差分解理论证明加权集成可降低方差从而缓解概念漂移，设计双流残差递减 boosting 架构，每个 block 的输出修正前一个 block 的残差，在多个数据集上平均提升 15.8%。

**[Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching](detecting_the_future_all-at-once_event_sequence_forecasting_with_horizon_matchin.md)**

:   提出DEF（Detection-based Event Forecasting），借鉴目标检测中DETR的匹配思想，通过匈牙利算法对齐预测与真实事件序列，实现高精度和高多样性的长程事件预测，在5个数据集上达到SOTA。

**[Finding Time Series Anomalies using Granular-ball Vector Data Description](finding_time_series_anomalies_using_granular-ball_vector_data_description.md)**

:   提出 Granular-ball One-Class Network (GBOC)，通过在潜在空间中自适应构建密度引导的粒球向量数据描述 (GVDD)，取代传统聚类或单一超球体假设，实现对时间序列正常行为的灵活建模和鲁棒异常检测。

**[FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)**

:   提出FreqCycle框架，通过FECF模块显式学习共享周期模式、SFPL模块增强中高频能量占比，并扩展为MFreqCycle处理耦合多周期性，在7个基准上达到SOTA性能与效率的最优平衡。

**[GAICo: A Deployed and Extensible Framework for Evaluating Diverse and Multimodal Generative AI Outputs](gaico_a_deployed_and_extensible_framework_for_evaluating_diverse_and_multimodal_.md)**

:   提出GAICo（Generative AI Comparator），一个已部署的、可扩展的开源Python库，为文本、结构化数据（规划序列、时间序列）和多媒体（图像、音频）提供统一的基于参考的评估框架，支持多模型比较、可视化与报告生成。

**[Harmonic Dataset Distillation for Time Series Forecasting](harmonic_dataset_distillation_for_time_series_forecasting.md)**

:   提出HDT（Harmonic Dataset Distillation for Time Series Forecasting），通过FFT将时间序列分解为正弦基底，在频域上通过谐波匹配（Harmonic Matching）对齐合成数据与原始数据的核心周期结构，实现强跨架构泛化和良好可扩展性的时间序列数据集蒸馏。

**[HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting](hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)**

:   提出 HN-MVTS，利用超网络(HyperNetwork)为每个通道生成特定的最后一层权重，在通道独立(CI)和通道依赖(CD)之间取得平衡，作为即插即用模块可提升 DLinear、PatchTST、TSMixer 等多种主干模型的预测精度，且不增加推理时间。

**[HydroDCM: Hydrological Domain-Conditioned Modulation for Cross-Reservoir Inflow Prediction](hydrodcm_hydrological_domain-conditioned_modulation_for_cross-reservoir_inflow_p.md)**

:   提出 HydroDCM，首次将域泛化(Domain Generalization)引入水文预测领域，通过空间元属性构建伪域标签指导对抗学习提取不变特征，再用 FiLM 适配器根据目标水库的地理信息调制特征，实现对未见水库的跨域入流预测。

**[IdealTSF: Can Non-Ideal Data Contribute to Enhancing Time Series Forecasting?](idealtsf_can_non-ideal_data_contribute_to_enhancing_the_performance_of_time_seri.md)**

:   提出 IdealTSF 框架，通过三阶段渐进式设计——负样本预训练（用稳定分布+多尺度噪声+结构删除模拟非理想数据）、正样本训练（混合平滑插值修复数据）、ECOS 优化器（对抗扰动引导到平坦极值）——使基础 attention 模型在含噪声/缺失的时序数据上获得约 10% 的性能提升。

**[Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths](interpreting_fedspeak_with_confidence_a_llm-based_uncertainty-aware_framework_gu.md)**

:   提出基于 LLM 的 uncertainty-aware 框架解读 Fedspeak（美联储语言）：通过货币政策传导路径的领域推理增强输入，引入 dynamic uncertainty decoding 模块量化预测置信度（Perceptual Uncertainty = Environmental Ambiguity × Cognitive Risk），在 FOMC 政策立场分析任务上达到 SOTA。

**[LoReTTA: A Low Resource Framework To Poison Continuous Time Dynamic Graphs](loretta_a_low_resource_framework_to_poison_continuous_time_dynamic_graphs.md)**

:   提出 LoReTTA，一种无需代理模型的两阶段对抗投毒攻击框架：先通过 16 种时序重要性度量稀疏化高影响力边，再用保度数负采样算法替换对抗边，在 4 个数据集 × 4 个 TGNN 模型上平均降低 29.47% 性能，同时逃避 4 种异常检测系统且抵御 4 种防御方法。

**[M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting](m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)**

:   提出 M2FMoE，通过傅里叶和小波双视角的频域混合专家建模常规与极端模式，结合跨视角共享频段分割器对齐两域语义、多分辨率自适应融合捕获多尺度信息、时序门控整合长短期特征，在 5 个水文极端事件数据集上无需极端事件标签即超越所有 SotA（含使用标签的方法），平均 RMSE 提升 22.3%。

**[Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering](mask_the_redundancy_evolving_masking_representation_learning_for_multivariate_ti.md)**

:   提出 EMTC 框架，通过 Importance-aware Variate-wise Masking (IVM) 动态屏蔽冗余时间戳，结合 Multi-Endogenous Views (MEV) 多视图生成与 cluster-guided contrastive learning，在 15 个 MTS 聚类基准上平均 F1 提升 4.85%。

**[Optimal Look-back Horizon for Time Series Forecasting in Federated Learning](optimal_look-back_horizon_for_time_series_forecasting_in_federated_learning.md)**

:   提出联邦学习场景下时间序列预测的最优回看窗口（look-back horizon）理论框架，通过合成数据生成器（SDG）和内禀空间表示，将预测损失分解为贝叶斯不可约误差和近似误差，证明总损失关于窗口长度是单峰的，最小充分窗口为最优解。

**[Predicting the Future by Retrieving the Past](predicting_the_future_by_retrieving_the_past.md)**

:   提出 PFRP（Predicting the Future by Retrieving the Past），构建全局记忆库(GMB)存储历史模式，通过预测性对比学习训练编码器实现高效检索，将检索到的全局预测与任意局部预测模型动态融合，在 7 个数据集上平均提升 8.4% 的预测性能。

**[ReCast: Reliability-aware Codebook Assisted Lightweight Time Series Forecasting](recast_reliability-aware_codebook_assisted_lightweight_time_series_forecasting.md)**

:   提出 ReCast，通过 patch 级向量量化将时间序列编码为离散嵌入，设计量化路径（预测规律结构）和残差路径（捕获不规则波动）的双路径架构，并引入基于分布鲁棒优化(DRO)的可靠性感知码本更新策略，在 8 个数据集上以轻量架构实现 SOTA 精度。

**[Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)**

:   首次论证了规范预对齐(CPA)在不规则多变量时间序列(IMTS)预测中不应被抛弃，提出 KAFNet 通过预卷积平滑、时间核聚合(TKA)压缩和频域线性注意力(FLA)三个模块解决 CPA 的效率问题，在 4 个 IMTS 数据集上实现 SOTA 精度，同时参数量减少 7.2 倍、训练推理加速 8.4 倍。

**[Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios](scaling_llm_speculative_decoding_non-autoregressive_forecasting_in_large-batch_s.md)**

:   提出 SpecFormer，一种融合单向和双向注意力的非自回归草稿模型架构，在大批次推理场景下通过降低对复杂前缀树的依赖、减少位置相关参数，实现了对 LLM 推理的一致性加速。

**[SELDON: Supernova Explosions Learned by Deep ODE Networks](seldon_supernova_explosions_learned_by_deep_ode_networks.md)**

:   提出SELDON，一种结合masked GRU-ODE编码器、隐式Neural ODE传播器和可解释高斯基函数解码器的连续时间VAE，用于稀疏、不规则采样的天文光变曲线预测，在仅观测20%数据时即可超越基线方法做出准确的多波段通量预测。

**[Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting](sonnet_spectral_operator_neural_network_for_multivariable_time_series_forecastin.md)**

:   提出 Sonnet，通过可学习小波变换将输入映射到时频域，引入基于谱相干性的多变量注意力（MVCA）建模变量间依赖关系，并利用 Koopman 算子进行稳定的时间演化预测，在 47 个预测任务中的 34 个取得最优，平均 MAE 降低 2.2%。

**[Task-Aware Retrieval Augmentation for Dynamic Recommendation](task-aware_retrieval_augmentation_for_dynamic_recommendation.md)**

:   提出 TarDGR 框架，通过任务感知的评估机制自动构建训练数据，训练 Graph Transformer 来评估历史子图的任务相关性，在推理时检索并融合任务相关子图以增强推荐的时序泛化能力。

**[Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)**

:   提出 DTAF 双分支框架，通过时域的非平稳 MoE 滤波器提取并去除异质非平稳模式、频域的频谱差分追踪频率漂移，并通过双分支注意力融合两个域的互补信息，实现鲁棒的非平稳时间序列预测。

**[Transparent Networks for Multivariate Time Series](transparent_networks_for_multivariate_time_series.md)**

:   提出 GATSM（Generalized Additive Time Series Model），一种透明的时间序列神经网络模型，通过共享权重的特征网络学习特征表示并用带掩码的多头注意力捕捉时序模式，在保持完全可解释性的同时达到与 Transformer 等黑箱模型可比的性能。

**[Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports](urban_incident_prediction_with_graph_neural_networks_integrating_government_rati.md)**

:   提出 URBAN（多视图多输出GNN模型），联合利用稀疏但无偏的政府检查评级数据和密集但有偏的众包报告数据来预测城市事件的真实潜在状态，在纽约市960万+报告和100万+检查数据上验证，预测相关性比仅用报告数据高5.3倍。
