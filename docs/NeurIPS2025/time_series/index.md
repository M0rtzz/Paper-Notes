---
title: >-
  NeurIPS2025 时间序列方向 52篇论文解读
description: >-
  52篇NeurIPS2025 时间序列方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🧠 NeurIPS2025** · 共 **52** 篇

**[A Graph Neural Network Approach For Localized And High-Resolution Temperature Fo](a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)**

:   提出一种 GCN-GRU 混合框架用于社区尺度（2.5km）高分辨率温度预报（1-48小时），在加拿大西南安大略三个区域上验证，最大区域平均 MAE 1.93°C、48h MAE 2.93°C，探索了 ClimateBERT 语言模型嵌入作为标准化输入的方案，为数据稀缺的全球南方地区提供可迁移的轻量级预报框架。

**[Abstain Mask Retain Core Time Series Prediction By Adaptive](abstain_mask_retain_core_time_series_prediction_by_adaptive.md)**

:   揭示了时间序列预测中"适当截断历史数据反而提升精度"的反直觉现象（冗余特征学习问题），基于信息瓶颈理论提出AMRC方法，通过自适应掩码损失和表征一致性约束来抑制冗余特征学习，作为模型无关的训练框架在多种架构上显著提升性能。

**[Aero A Redirection-Based Optimization Framework Inspired By Judo For Robust Prob](aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)**

:   AERO 提出受柔道"借力重定向"启发的优化框架，通过梯度投影、能量守恒和干扰预测将对抗性扰动重定向为有利优化方向，在概率太阳能价格预测上展示更稳定的收敛。

**[Attentionpredictor Temporal Patterns Matter For Kv Cache Com](attentionpredictor_temporal_patterns_matter_for_kv_cache_com.md)**

:   首个基于学习的 KV Cache 压缩方法，通过轻量级时空卷积模型预测下一 token 的注意力分数来动态识别关键 token，实现 13× KV cache 压缩和 5.6× cache offloading 加速，显著优于静态方法。

**[Benchmarking Probabilistic Time Series Forecasting Models On Neural Activity](benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)**

:   首次系统评测 12 个概率时间序列预测模型在小鼠皮层钙成像数据上的表现，发现 PatchTST 一致最优（信息性预测窗口达 1.5 秒），零样本基础模型（Chronos）完全失败但微调后竞争力强，揭示神经活动的内在可预测性上限约 1.5 秒。

**[Causal Masking On Spatial Data An Information-Theoretic Case For Learning Spatia](causal_masking_on_spatial_data_an_information-theoretic_case_for_learning_spatia.md)**

:   证明在空间数据（国际象棋棋盘FEN状态）上直接应用因果掩蔽训练单模态LLM，其表现优于先将数据线性化为序列（PGN棋步）后再应用因果掩蔽——FEN+因果掩蔽的Llama 1.3B达到~2630 Elo，而PGN+因果仅~2130 Elo。

**[Causaldynamics A Large-Scale Benchmark For Structural Discovery Of Dynamical Cau](causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)**

:   提出 CausalDynamics——迄今最大规模的动力系统因果发现 benchmark（14000+ 图、5000 万+ 样本），涵盖从 3 维混沌 ODE/SDE 到层级耦合系统再到拟真气候模型的三层渐进复杂度体系，并全面评估了 10 种 SOTA 因果发现算法，揭示当前深度学习方法在高维非线性动力系统上的不足。

**[Channel Matters Estimating Channel Influence For Multivariate Time Series](channel_matters_estimating_channel_influence_for_multivariate_time_series.md)**

:   提出 Channel-wise Influence (ChInf)——首个能量化多变量时间序列中不同通道对模型性能影响的影响函数方法，将 TracIn 从整体样本级分解到通道级，衍生出通道级异常检测和通道剪枝两个应用，在 5 个异常检测基准上排名第一。

**[Connecting The Dots A Machine Learning Ready Dataset For Ionospheric Forecasting](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)**

:   构建了首个ML-ready电离层预测数据集，整合SDO、太阳风、地磁指数和TEC观测等多源异构数据为统一的时间-空间结构，并基准测试了多种时空ML架构用于TEC预测。

**[Demandcast Global Hourly Electricity Demand Forecasting](demandcast_global_hourly_electricity_demand_forecasting.md)**

:   构建DemandCast——覆盖56个国家(2000-2025)的XGBoost全球小时电力需求预测框架，融合ERA5温度/GDP/人口等特征，归一化目标（年度分数）+时间分割评估，MAPE 9.2%。

**[Diffusion Transformers As Open-World Spatiotemporal Foundation Models](diffusion_transformers_as_open-world_spatiotemporal_foundation_models.md)**

:   提出 UrbanDiT，首个基于 Diffusion Transformer 的开放世界城市时空基础模型，通过统一的 prompt learning 框架整合异构数据类型（grid/graph）和多种任务（预测/插值/外推/填补），在多城市多场景下实现 SOTA 性能并展现强大的 zero-shot 泛化能力。

**[Diffusion Transformers For Imputation Statistical Efficiency And Uncertainty Qua](diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)**

:   本文从统计学习角度分析了条件扩散Transformer（DiT）在时间序列插补任务中的样本复杂度和不确定性量化性能，并提出混合掩码训练策略提升插补效果。

**[Ecocast A Spatio-Temporal Model For Continual Biodiversity And Climate Risk Fore](ecocast_a_spatio-temporal_model_for_continual_biodiversity_and_climate_risk_fore.md)**

:   提出EcoCast，基于Transformer的时空模型，整合Sentinel-2、ERA5和GBIF数据进行近期物种分布预测，配合EWC持续学习机制，在非洲鸟类分布预测上F1从0.31提升至0.65。

**[Exploring Neural Granger Causality With Xlstms Unveiling Temporal Dependencies I](exploring_neural_granger_causality_with_xlstms_unveiling_temporal_dependencies_i.md)**

:   提出 GC-xLSTM，利用 xLSTM 架构结合新颖的动态稀疏优化策略，在多变量时间序列中挖掘 Granger 因果关系，在多个数据集上取得 SOTA 性能。

**[Friren Beyond Trajectories -- A Spectral Lens On Time](friren_beyond_trajectories_--_a_spectral_lens_on_time.md)**

:   提出 Fern (Forecasting with Ellipsoidal RepresentatioN)，通过逐 patch 的椭球体传输（旋转-缩放-平移）替代传统轨迹预测，在混沌系统上大幅超越基线，并在标准 LTSF 基准上保持竞争力。

**[How Foundational Are Foundation Models For Time Series Forecasting](how_foundational_are_foundation_models_for_time_series_forecasting.md)**

:   通过合成数据与真实电力消耗数据的系统性实验，揭示时间序列基础模型(TSFM)的零样本泛化能力高度依赖于预训练数据分布，在领域偏移场景下仅49.5K参数的轻量专用模型SAMFormer从头训练即可超越500M+参数的微调TimesFM。

**[How Patterns Dictate Learnability In Sequential Data](how_patterns_dictate_learnability_in_sequential_data.md)**

:   提出基于预测信息（predictive information）的信息论框架来量化序列数据中时间模式的强度，推导出将预测信息与最小可达风险联系起来的理论界，从而区分"模型不够好"还是"数据本身就不可预测"。

**[Improving Time Series Forecasting Via Instance-Aware Post-Hoc Revision](improving_time_series_forecasting_via_instance-aware_post-hoc_revision.md)**

:   PIR 提出实例感知的事后修正框架——通过不确定性估计识别预测失败实例，用局部修正（协变量+外生变量 Transformer）和全局修正（检索相似训练实例加权平均）的残差组合，作为即插即用模块使 SparseTSF MSE 降低 25.87%，PatchTST 降低 8.99%。

**[In-Context Learning Of Stochastic Differential Equations With Foundation Inferen](in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)**

:   提出FIM-SDE（基础推断模型），一个预训练的识别模型，能够从噪声时间序列数据中进行零样本(in-context)估计低维SDE的漂移和扩散函数，并通过快速微调进一步超越所有基线方法。

**[Ioncast A Deep Learning Framework For Forecasting Ionospheric Dynamics](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_dynamics.md)**

:   提出 IonCast 框架，基于 GraphCast 启发的图神经网络架构，融合多源异构物理驱动数据，实现全球总电子含量（TEC）的高精度时空预测。

**[Ioncast A Deep Learning Framework For Forecasting Ionospheric Total Electron Con](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_total_electron_con.md)**

:   提出IonCast框架，包含基于GraphCast的GNN模型和ConvLSTM基线，融合多源异构空间天气数据（TEC图、太阳风、地磁指数、轨道力学等）进行全球电离层总电子含量（TEC）的时空预测，在地磁风暴条件下优于持续性基线和IRI经验模型。

**[Learning Time-Scale Invariant Population-Level Neural Representations](learning_time-scale_invariant_population-level_neural_representations.md)**

:   提出时间尺度增强预训练（TSAP）策略，通过在预训练阶段引入多种时间窗口长度的数据增强，使群体级神经信号基础模型对输入时间尺度具有不变性，在匹配和未见时间尺度上均显著提升解码性能。

**[Learning With Calibration Exploring Test-Time Computing Of Spatio-Temporal Forec](learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)**

:   提出 ST-TTC，一种轻量级测试时计算范式，通过频域相位-幅值校准器和闪电梯度更新机制，在推理阶段实时修正时空预测中的周期性偏差，无需修改骨干网络即可一致性提升多种模型性能。

**[Maestro Adaptive Sparse Attention And Robust Learning For Multimodal Dynamic Tim](maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)**

:   提出 MAESTRO 框架，通过符号化分词、自适应注意力预算、稀疏跨模态注意力和动态 MoE 路由，解决多模态时间序列中模态异质性和任意缺失的问题，在完整/缺失模态场景下均显著超越基线。

**[Martingale Score An Unsupervised Metric For Bayesian Rationality In Llm Reasonin](martingale_score_an_unsupervised_metric_for_bayesian_rationality_in_llm_reasonin.md)**

:   提出 Martingale Score 作为无监督度量指标，基于贝叶斯统计中的鞅性质(Martingale property)来量化 LLM 推理过程中的信念固化(belief entrenchment)现象，发现该现象普遍存在且与准确率下降显著相关。

**[Masfin A Multi-Agent System For Decomposed Financial Reasoning And Forecasting](masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)**

:   提出 MASFIN 多 agent 系统，将金融预测任务分解为多个子任务（宏观分析、行业分析、技术分析、情感分析等），由专门的 LLM agent 协作完成，实现比单一模型更准确和可解释的金融预测。

**[Neural Mjd Neural Non-Stationary Merton Jump Diffusion For Time Series Predictio](neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)**

:   提出 Neural MJD，用神经网络参数化非平稳 Merton 跳跃扩散模型，将预测建模为 SDE 仿真问题，结合时变 Itô 扩散（捕获连续漂移）和时变复合 Poisson 过程（建模突变跳跃），配合似然截断和 Euler-Maruyama with Restart 求解器实现可扩展学习与推理。

**[Nsw-Epnews A News-Augmented Benchmark For Electricity Price Forecasting With Llm](nsw-epnews_a_news-augmented_benchmark_for_electricity_price_forecasting_with_llm.md)**

:   提出首个融合新闻文本的电力价格预测基准 NSW-EPNews，系统评估传统模型和 LLM 在多模态电价预测中的表现，发现新闻特征对传统模型增益有限，而 LLM 存在严重幻觉问题。

**[Parallelization Of Non-Linear State-Space Models Scaling Up Liquid-Resistance Li](parallelization_of_non-linear_state-space_models_scaling_up_liquid-resistance_li.md)**

:   提出 LrcSSM，通过约束液态电阻-液态电容（LRC）网络的 Jacobian 矩阵为对角形式，实现非线性 RNN 的精确高效并行化，在长序列分类任务上超越 Transformer、LRU、S5 和 Mamba 等 SOTA 方法。

**[Physics-Informed Reduced Order Modeling Of Time-Dependent Pdes Via Differentiabl](physics-informed_reduced_order_modeling_of_time-dependent_pdes_via_differentiabl.md)**

:   提出Φ-ROM框架，将可微分PDE求解器嵌入非线性降阶模型的训练过程中，通过求解器反馈直接约束潜在空间动态，使模型在泛化到未见参数/初始条件、长时间外推、稀疏观测数据恢复等方面显著优于纯数据驱动ROM和其他物理信息方法。

**[Planu Large Language Model Reasoning Through Planning Under Uncertainty](planu_large_language_model_reasoning_through_planning_under_uncertainty.md)**

:   提出PlanU——一种在MCTS中用分位数分布建模节点回报、并通过Upper Confidence Bounds with Curiosity (UCC)分数平衡探索与利用的LLM决策方法，首次系统性地同时处理LLM不确定性和环境不确定性，在多个随机环境基准上显著优于现有方法。

**[Probability Calibration For Precipitation Nowcasting](probability_calibration_for_precipitation_nowcasting.md)**

:   提出了期望阈值校准误差（ETCE）作为降水临近预报中更合理的概率校准度量，并将计算机视觉中的后处理校准技术扩展到预报领域，通过结合前置时间条件的选择性缩放（Selective Scaling）方法将模型校准误差降低高达23.5%。

**[Rivermamba A State Space Model For Global River Discharge And Flood Forecasting](rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)**

:   首个能在 0.05°（~5.5km）全球网格上做 7 天河流流量预报的深度学习模型——用空间填充曲线将 3D 时空点序列化后输入双向 Mamba block，结合 ECMWF HRES 气象预报，在 1.5-500 年重现期洪水检测上 F1 =0.459 超越 LSTM（0.358）和物理模型 GloFAS。

**[Rotary Masked Autoencoders Are Versatile Learners](rotary_masked_autoencoders_are_versatile_learners.md)**

:   提出 RoMAE，将旋转位置编码（RoPE）扩展到连续位置并与掩码自编码器（MAE）结合，无需任何时间序列特定的架构修改即可在不规则时间序列、图像和音频等多种模态上达到或超越专用模型的性能。

**[Scalable Signature Kernel Computations For Long Time Series Via Local Neumann Se](scalable_signature_kernel_computations_for_long_time_series_via_local_neumann_se.md)**

:   提出 PowerSig，通过自适应截断的局部 Neumann 级数展开高效计算签名核（signature kernel），将内存从 $O(\ell^2)$ 降到 $O(\ell P)$，使签名核可扩展到单GPU上百万级长度的时间序列。

**[Scatterad Temporal-Topological Scattering Mechanism For Time Series Anomaly Dete](scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)**

:   提出"散射性"（scattering）作为异常检测的新归纳偏置——异常样本在高维表示空间中比正常样本分布更分散，通过双编码器（时间+拓扑）+ 超球面散射中心约束 + 对比融合学习时拓扑联合表示，在 6 个工业 IoT 数据集上 15/24 设置取得最佳。

**[Selective Learning For Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)**

:   提出选择性学习（Selective Learning）策略，通过不确定性掩码和异常掩码组成的双掩码机制筛选可泛化时间步计算 MSE 损失，在 8 个数据集上为 Informer 降低 37.4% MSE、TimesNet 降低 8.4%、iTransformer 降低 6.5%。

**[Sempo Lightweight Foundation Models For Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)**

:   提出SEMPO——仅用6.5M参数和83M时间点预训练的轻量级时间序列基础模型，通过能量感知频谱分解和混合提示Transformer，在零样本和少样本预测中超越参数量百倍以上的大型基础模型。

**[Simple And Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)**

:   提出 SE-HTGNN，通过动态注意力机制将时序建模融入空间学习，并用 LLM 初始化注意力系数，在异构时序图任务上实现 10 倍加速的同时保持最优预测精度。

**[Statistical Guarantees For High-Dimensional Stochastic Gradient Descent](statistical_guarantees_for_high-dimensional_stochastic_gradient_descent.md)**

:   将高维非线性时间序列的耦合技术引入在线学习，首次为常数学习率 SGD 及其 Ruppert-Polyak 平均变体在高维（$\ell^s$ 和 $\ell^\infty$ 范数下）提供了严格的矩收敛界和高概率集中界。

**[Strap Spatio-Temporal Pattern Retrieval For Out-Of-Distribution Generalization](strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)**

:   提出 StRap，一个检索增强的时空模式学习框架，通过构建空间/时间/时空三维模式库并在推理时检索相似模式注入模型，在流式时空图 OOD 任务上平均提升 7.17%。

**[Structured Temporal Causality For Interpretable Multivariate Time Series Anomaly](structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)**

:   提出OracleAD框架，通过为每个变量学习因果嵌入（LSTM编码+注意力池化）并构建稳定潜在结构（SLS）来建模正常状态下的变量间关系，结合预测误差和SLS偏离的双重评分机制实现可解释的多变量时间序列异常检测与根因定位。

**[Synthetic Series-Symbol Data Generation For Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)**

:   提出 Series-Symbol (S²) 数据生成机制和 SymTime 基础模型，通过符号表达式与时序数据的双模态对比学习预训练，在纯合成数据上训练即可在 5 大时序分析任务上与真实数据预训练的基础模型竞争。

**[Syntsbench Rethinking Temporal Pattern Learning In Deep Learning Models For Time](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)**

:   提出 SynTSBench，一个基于合成数据的时序预测模型评估框架，通过可编程特征配置（趋势/周期/噪声/依赖/多变量）和理论最优基准，系统揭示当前深度学习模型在各类时序模式上的能力边界。

**[Time-Imm A Dataset And Benchmark For Irregular Multimodal Multivariate Time Seri](time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)**

:   构建 Time-IMM 数据集——首个按因果机制分类不规则性的多模态多变量时序 benchmark（9 种不规则类型分为触发/约束/伪影三大类，9 个数据集），配套 IMM-TSF 预测库支持异步多模态融合，实验表明显式建模多模态在不规则时序上平均降低 MSE 6.71%，最高达 38.38%。

**[Time-O1 Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)**

:   提出 Time-o1，通过将标签序列变换为去相关且按重要性排序的主成分，解决时间序列预测中 TMSE 损失的自相关偏差和任务过载问题，实现与多种预测模型兼容的 SOTA 性能。

**[Timeperceiver An Encoder-Decoder Framework For Generalized Time-Series Forecasti](timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)**

:   提出 TimePerceiver 统一编码器-解码器框架，通过广义化预测任务（同时包含外推、插值和填补）以及潜在瓶颈编码器 + 查询式解码器设计，在 8 个标准基准上取得全面 SOTA。

**[Transformer Embeddings For Fast Microlensing Inference](transformer_embeddings_for_fast_microlensing_inference.md)**

:   本文将Transformer编码器与神经后验估计（NPE）结合，直接从稀疏、噪声、不等间隔的微引力透镜光变曲线中进行快速且校准良好的参数推断，速度比传统MCMC快10⁴倍以上。

**[Universal Spectral Tokenization Via Self-Supervised Panchromatic Representation ](universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)**

:   提出首个通用光谱 Tokenizer，通过连续波长嵌入和自监督重建目标，在原始波长网格上联合训练异构天文光谱数据（SDSS/DESI/GALAH/APOGEE），生成对齐、均匀且物理有意义的表征。

**[Walrus Wavelets For Long-Range Representation Using Ssms](walrus_wavelets_for_long-range_representation_using_ssms.md)**

:   提出 WaLRUS，基于 Daubechies 小波构建状态空间模型 (SSM)，作为 SaFARi 框架的新实现，扩展了 SSM 家族的多样性，在长程依赖建模中展现独特优势。

**[Wavelet Canonical Coherence For Nonstationary Signals](wavelet_canonical_coherence_for_nonstationary_signals.md)**

:   提出 WaveCanCoh 框架，将经典的典型相干分析（canonical coherence）扩展到小波域，基于多变量局部平稳小波（MvLSW）模型实现对非平稳多变量时间序列两组信号间时变、尺度特定的典型相干性估计。

**[Xlstm-Mixer Multivariate Time Series Forecasting By Mixing Via Scalar Memories](xlstm-mixer_multivariate_time_series_forecasting_by_mixing_via_scalar_memories.md)**

:   提出 xLSTM-Mixer，首次将扩展长短期记忆网络（sLSTM）与混合架构（Mixer）结合，通过时间混合、联合时间-变量混合和多视角混合三阶段架构实现多变量长期时间序列预测的 SOTA 性能，同时保持极低的内存占用。
