---
title: >-
  NeurIPS2025 时间序列方向58篇论文解读
description: >-
  58篇NeurIPS2025的时间序列方向论文解读，涵盖时序预测、扩散模型、推理、图神经网络、对抗鲁棒、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🧠 NeurIPS2025** · **58** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (5)](../../ACL2026/time_series/index.md) · [📷 CVPR2026 (6)](../../CVPR2026/time_series/index.md) · [🔬 ICLR2026 (37)](../../ICLR2026/time_series/index.md) · [🤖 AAAI2026 (35)](../../AAAI2026/time_series/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/time_series/index.md) · [🧪 ICML2025 (26)](../../ICML2025/time_series/index.md)

🔥 **高频主题：** 时序预测 ×31 · 扩散模型 ×3 · 推理 ×3 · 图神经网络 ×2 · 对抗鲁棒 ×2

**[A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)**

:   提出一种 GCN-GRU 混合框架用于社区尺度（2.5km）高分辨率温度预报（1-48小时），在加拿大西南安大略三个区域上验证，最大区域平均 MAE 1.93°C、48h MAE 2.93°C，探索了 ClimateBERT 语言模型嵌入作为标准化输入的方案，为数据稀缺的全球南方地区提供可迁移的轻量级预报框架。

**[Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](abstain_mask_retain_core_time_series_prediction_by_adaptive.md)**

:   揭示了时间序列预测中"适当截断历史数据反而提升精度"的反直觉现象（冗余特征学习问题），基于信息瓶颈理论提出AMRC方法，通过自适应掩码损失和表征一致性约束来抑制冗余特征学习，作为模型无关的训练框架在多种架构上显著提升性能。

**[AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting](aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)**

:   AERO提出受柔道"借力重定向"启发的优化范式，尝试将对抗扰动重定向为有利的优化信号而非直接抵抗，理论上通过15条公理和4个定理构建了基于能量守恒的梯度重定向系统，但实际实现大幅简化为带高斯噪声注入的动量SGD，仅在一个私有太阳能价格预测数据集上进行了无基线对比的验证。

**[AttentionPredictor: Temporal Patterns Matter for KV Cache Compression](attentionpredictor_temporal_patterns_matter_for_kv_cache_com.md)**

:   AttentionPredictor是首个学习型方法直接预测注意力模式以实现KV缓存压缩和关键token识别，通过轻量CNN捕捉注意力分数的时空模式，实现13倍KV缓存压缩和5.6倍推理加速，统一预测模型仅21KB可跨所有Transformer层共享。

**[Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity](benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)**

:   首次系统评测 12 个概率时间序列预测模型在小鼠皮层钙成像数据上的表现，发现 PatchTST 一致最优（信息性预测窗口达 1.5 秒），零样本基础模型（Chronos）完全失败但微调后竞争力强，揭示神经活动的内在可预测性上限约 1.5 秒。

**[BubbleFormer: Forecasting Boiling with Transformers](bubbleformer_forecasting_boiling_with_transformers.md)**

:   提出 BubbleFormer，基于分解时空轴注意力的 Transformer 架构用于预测沸腾动力学——包括难以预测的自主气泡成核事件，配合 BubbleML 2.0 数据集（160+ 高保真仿真），在多种流体、几何和壁面条件下实现准确的沸腾时空过程预测。

**[Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models](causal_masking_on_spatial_data_an_information-theoretic_case_for_learning_spatia.md)**

:   证明在空间数据（国际象棋棋盘FEN状态）上直接应用因果掩蔽训练单模态LLM，其表现优于先将数据线性化为序列（PGN棋步）后再应用因果掩蔽——FEN+因果掩蔽的Llama 1.3B达到~2630 Elo，而PGN+因果仅~2130 Elo。

**[CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)**

:   提出 CausalDynamics——迄今最大规模的动力系统因果发现 benchmark（14000+ 图、5000 万+ 样本），涵盖从 3 维混沌 ODE/SDE 到层级耦合系统再到拟真气候模型的三层渐进复杂度体系，并全面评估了 10 种 SOTA 因果发现算法，揭示当前深度学习方法在高维非线性动力系统上的不足。

**[Channel Matters: Estimating Channel Influence for Multivariate Time Series](channel_matters_estimating_channel_influence_for_multivariate_time_series.md)**

:   提出 Channel-wise Influence (ChInf)——首个能量化多变量时间序列中不同通道对模型性能影响的影响函数方法，将 TracIn 从整体样本级分解到通道级，衍生出通道级异常检测和通道剪枝两个应用，在 5 个异常检测基准上排名第一。

**[Connecting the Dots: 面向电离层预测的机器学习数据集](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)**

:   本文构建了一个开放的、机器学习就绪的电离层预测数据集，融合了8种异构数据源（太阳观测、地磁指数、TEC地图等），覆盖2010-2024年约14年时间，并基于此训练了LSTM、SFNO、GraphCast三种时空模型作为基准，实现了最长12小时的TEC预测。

**[Decomposition of Small Transformer Models](decomposition_of_small_transformer_models.md)**

:   将 Stochastic Parameter Decomposition (SPD) 扩展到 Transformer，设计适用于序列数据的因果重要性函数和新损失函数，在玩具 induction head 上恢复期望两步电路，在 GPT-2-small 上定位到"高尔夫""篮球"等可解释概念对应的 rank-1 参数子空间。

**[DemandCast: Global hourly electricity demand forecasting](demandcast_global_hourly_electricity_demand_forecasting.md)**

:   构建 DemandCast 开源机器学习框架，基于 XGBoost 融合历史电力需求、ERA5 温度和社会经济特征进行全球 56 个国家/地区的小时级电力需求预测，通过归一化目标变量（年度分数）实现跨国家可比，在时间外推测试集上达到 MAPE 9.2%。

**[Diffusion Transformers as Open-World Spatiotemporal Foundation Models](diffusion_transformers_as_open-world_spatiotemporal_foundation_models.md)**

:   提出 UrbanDiT，首个基于 Diffusion Transformer 的开放世界城市时空基础模型，通过统一的 prompt learning 框架整合异构数据类型（grid/graph）和多种任务（预测/插值/外推/填补），在多城市多场景下实现 SOTA 性能并展现强大的 zero-shot 泛化能力。

**[Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)**

:   本文从统计学习角度分析了条件扩散Transformer（DiT）在时间序列插补任务中的样本复杂度和不确定性量化性能，并提出混合掩码训练策略提升插补效果。

**[Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data](exploring_neural_granger_causality_with_xlstms_unveiling_temporal_dependencies_i.md)**

:   提出 GC-xLSTM，利用 xLSTM 架构结合新颖的动态稀疏优化策略，在多变量时间序列中挖掘 Granger 因果关系，在多个数据集上取得 SOTA 性能。

**[Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)**

:   揭示时间序列基础模型（TSFM）在工业场景中泛化失败的关键原因——频谱偏移（downstream 数据主频与预训练数据不重叠），通过工业级手游玩家参与预测任务和受控合成实验验证了这一假说。

**[Fern: Chaining Spectral Pearls — Ellipsoidal Forecasting Beyond Trajectories for Time Series](friren_beyond_trajectories_--_a_spectral_lens_on_time.md)**

:   提出 Fern (Forecasting with Ellipsoidal RepresentatioN)，通过逐 patch 的椭球体传输（旋转-缩放-平移）替代传统轨迹预测，在混沌系统上大幅超越基线，并在标准 LTSF 基准上保持竞争力。

**[How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)**

:   通过合成数据与真实电力消耗数据的系统性实验，揭示时间序列基础模型(TSFM)的零样本泛化能力高度依赖于预训练数据分布，在领域偏移场景下仅49.5K参数的轻量专用模型SAMFormer从头训练即可超越500M+参数的微调TimesFM。

**[How Patterns Dictate Learnability in Sequential Data](how_patterns_dictate_learnability_in_sequential_data.md)**

:   提出基于预测信息（predictive information）的信息论框架来量化序列数据中时间模式的强度，推导出将预测信息与最小可达风险联系起来的理论界，从而区分"模型不够好"还是"数据本身就不可预测"。

**[Human-Machine Ritual: Synergic Performance through Real-Time Motion Recognition](human-machine_ritual_synergic_performance_through_real-time_motion_recognition.md)**

:   提出一种轻量级实时动作识别系统，利用可穿戴 IMU 传感器 + MiniRocket 时序分类器实现 <50ms 延迟的舞者特定动作识别（96.05% 准确率），通过"具身记忆映射"将舞者的个人动作-声音关联编码到系统中，构建了一种尊重人体表达深度的人机协作表演范式。

**[Improving Time Series Forecasting via Instance-aware Post-hoc Revision (PIR)](improving_time_series_forecasting_via_instance-aware_post-hoc_revision.md)**

:   PIR 提出实例感知的事后修正框架——通过不确定性估计识别预测失败实例，用局部修正（协变量+外生变量 Transformer）和全局修正（检索相似训练实例加权平均）的残差组合，作为即插即用模块使 SparseTSF MSE 降低 25.87%，PatchTST 降低 8.99%。

**[In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)**

:   提出FIM-SDE（基础推断模型），一个预训练的识别模型，能够从噪声时间序列数据中进行零样本(in-context)估计低维SDE的漂移和扩散函数，并通过快速微调进一步超越所有基线方法。

**[IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_total_electron_con.md)**

:   提出IonCast框架，包含基于GraphCast的GNN模型和ConvLSTM基线，融合多源异构空间天气数据（TEC图、太阳风、地磁指数、轨道力学等）进行全球电离层总电子含量（TEC）的时空预测，在地磁风暴条件下优于持续性基线和IRI经验模型。

**[Learning Time-Scale Invariant Population-Level Neural Representations](learning_time-scale_invariant_population-level_neural_representations.md)**

:   提出时间尺度增强预训练（TSAP）策略，通过在预训练阶段引入多种时间窗口长度的数据增强，使群体级神经信号基础模型对输入时间尺度具有不变性，在匹配和未见时间尺度上均显著提升解码性能。

**[Learning with Calibration: Exploring Test-Time Computing of Spatio-Temporal Forecasting](learning_with_calibration_exploring_test-time_computing_of_spatio-temporal_forec.md)**

:   提出 ST-TTC，一种轻量级测试时计算范式，通过频域相位-幅值校准器和闪电梯度更新机制，在推理阶段实时修正时空预测中的周期性偏差，无需修改骨干网络即可一致性提升多种模型性能。

**[Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning](less_is_more_unlocking_specialization_of_time_series_foundation_models_via_struc.md)**

:   揭示预训练时间序列基础模型（TSFM）存在固有的任务相关稀疏性，提出"先剪枝再微调"范式——通过结构化剪枝移除任务无关参数，使剪枝后微调的小模型显著超越直接微调的原模型，甚至胜过强专用基线。

**[MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series](maestro_adaptive_sparse_attention_and_robust_learning_for_multimodal_dynamic_tim.md)**

:   提出 MAESTRO 框架，通过符号化分词、自适应注意力预算、稀疏跨模态注意力和动态 MoE 路由，解决多模态时间序列中模态异质性和任意缺失的问题，在完整/缺失模态场景下均显著超越基线。

**[Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning](martingale_score_an_unsupervised_metric_for_bayesian_rationality_in_llm_reasonin.md)**

:   提出 Martingale Score 作为无监督度量指标，基于贝叶斯统计中的鞅性质(Martingale property)来量化 LLM 推理过程中的信念固化(belief entrenchment)现象，发现该现象普遍存在且与准确率下降显著相关。

**[MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting](masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)**

:   提出 MASFIN 多 agent 系统，将金融预测任务分解为多个子任务（宏观分析、行业分析、技术分析、情感分析等），由专门的 LLM agent 协作完成，实现比单一模型更准确和可解释的金融预测。

**[Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)**

:   提出 MSFT（Multi-Scale FineTuning），通过因果分析揭示 naive 微调忽略尺度混淆问题，设计多尺度建模框架对 encoder-based 时间序列基础模型进行高效微调，显著超越 naive 微调和从头训练的 SOTA 方法。

**[Neural MJD: Neural Non-Stationary Merton Jump Diffusion for Time Series Prediction](neural_mjd_neural_non-stationary_merton_jump_diffusion_for_time_series_predictio.md)**

:   提出 Neural MJD，用神经网络参数化非平稳 Merton 跳跃扩散模型，将预测建模为 SDE 仿真问题，结合时变 Itô 扩散（捕获连续漂移）和时变复合 Poisson 过程（建模突变跳跃），配合似然截断和 Euler-Maruyama with Restart 求解器实现可扩展学习与推理。

**[NSW-EPNews: A News-Augmented Benchmark for Electricity Price Forecasting with LLMs](nsw-epnews_a_news-augmented_benchmark_for_electricity_price_forecasting_with_llm.md)**

:   提出首个融合新闻文本的电力价格预测基准 NSW-EPNews，系统评估传统模型和 LLM 在多模态电价预测中的表现，发现新闻特征对传统模型增益有限，而 LLM 存在严重幻觉问题。

**[Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling](parallelization_of_non-linear_state-space_models_scaling_up_liquid-resistance_li.md)**

:   提出 LrcSSM，通过约束液态电阻-液态电容（LRC）网络的 Jacobian 矩阵为对角形式，实现非线性 RNN 的精确高效并行化，在长序列分类任务上超越 Transformer、LRU、S5 和 Mamba 等 SOTA 方法。

**[Physics-informed Reduced Order Modeling of Time-dependent PDEs via Differentiable Solvers](physics-informed_reduced_order_modeling_of_time-dependent_pdes_via_differentiabl.md)**

:   提出Φ-ROM框架，将可微分PDE求解器嵌入非线性降阶模型的训练过程中，通过求解器反馈直接约束潜在空间动态，使模型在泛化到未见参数/初始条件、长时间外推、稀疏观测数据恢复等方面显著优于纯数据驱动ROM和其他物理信息方法。

**[PlanU: Large Language Model Reasoning through Planning under Uncertainty](planu_large_language_model_reasoning_through_planning_under_uncertainty.md)**

:   提出PlanU——一种在MCTS中用分位数分布建模节点回报、并通过Upper Confidence Bounds with Curiosity (UCC)分数平衡探索与利用的LLM决策方法，首次系统性地同时处理LLM不确定性和环境不确定性，在多个随机环境基准上显著优于现有方法。

**[Power Ensemble Aggregation for Improved Extreme Event AI Prediction](power_ensemble_aggregation_for_improved_extreme_event_ai_prediction.md)**

:   提出基于幂均值的自适应集成聚合方法，通过对生成式天气预测模型的集成成员得分施加非线性聚合（幂指数$p>1$），显著提升极端高温事件的分类性能，尤其在高分位数阈值下效果更佳。

**[Probability Calibration for Precipitation Nowcasting](probability_calibration_for_precipitation_nowcasting.md)**

:   提出了期望阈值校准误差（ETCE）作为降水临近预报中更合理的概率校准度量，并将计算机视觉中的后处理校准技术扩展到预报领域，通过结合前置时间条件的选择性缩放（Selective Scaling）方法将模型校准误差降低高达23.5%。

**[RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting](rivermamba_a_state_space_model_for_global_river_discharge_and_flood_forecasting.md)**

:   首个能在 0.05°（~5.5km）全球网格上做 7 天河流流量预报的深度学习模型——用空间填充曲线将 3D 时空点序列化后输入双向 Mamba block，结合 ECMWF HRES 气象预报，在 1.5-500 年重现期洪水检测上 F1 =0.459 超越 LSTM（0.358）和物理模型 GloFAS。

**[Rotary Masked Autoencoders are Versatile Learners](rotary_masked_autoencoders_are_versatile_learners.md)**

:   提出 RoMAE，将旋转位置编码（RoPE）扩展到连续位置并与掩码自编码器（MAE）结合，无需任何时间序列特定的架构修改即可在不规则时间序列、图像和音频等多种模态上达到或超越专用模型的性能。

**[Scalable Signature Kernel Computations for Long Time Series via Local Neumann Series Expansions](scalable_signature_kernel_computations_for_long_time_series_via_local_neumann_se.md)**

:   提出 PowerSig，通过自适应截断的局部 Neumann 级数展开高效计算签名核（signature kernel），将内存从 $O(\ell^2)$ 降到 $O(\ell P)$，使签名核可扩展到单GPU上百万级长度的时间序列。

**[ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)**

:   提出"散射性"（scattering）作为异常检测的新归纳偏置——异常样本在高维表示空间中比正常样本分布更分散，通过双编码器（时间+拓扑）+ 超球面散射中心约束 + 对比融合学习时拓扑联合表示，在 6 个工业 IoT 数据集上 15/24 设置取得最佳。

**[Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)**

:   提出选择性学习（Selective Learning）策略，通过不确定性掩码和异常掩码组成的双掩码机制筛选可泛化时间步计算 MSE 损失，在 8 个数据集上为 Informer 降低 37.4% MSE、TimesNet 降低 8.4%、iTransformer 降低 6.5%。

**[SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)**

:   提出SEMPO——仅用6.5M参数和83M时间点预训练的轻量级时间序列基础模型，通过能量感知频谱分解和混合提示Transformer，在零样本和少样本预测中超越参数量百倍以上的大型基础模型。

**[Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)**

:   提出 SE-HTGNN，通过动态注意力机制将时序建模融入空间学习，并用 LLM 初始化注意力系数，在异构时序图任务上实现 10 倍加速的同时保持最优预测精度。

**[Statistical Guarantees for High-Dimensional Stochastic Gradient Descent](statistical_guarantees_for_high-dimensional_stochastic_gradient_descent.md)**

:   将高维非线性时间序列的耦合技术引入在线学习，首次为常数学习率 SGD 及其 Ruppert-Polyak 平均变体在高维（$\ell^s$ 和 $\ell^\infty$ 范数下）提供了严格的矩收敛界和高概率集中界。

**[StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)**

:   提出 StRap 框架，通过构建包含空间、时间和时空三维键值对的模式库，在推理时检索与当前输入最相似的历史模式并自适应融合，有效应对流式时空数据中的分布偏移（STOOD）问题。

**[Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)**

:   提出OracleAD框架，通过为每个变量学习因果嵌入（LSTM编码+注意力池化）并构建稳定潜在结构（SLS）来建模正常状态下的变量间关系，结合预测误差和SLS偏离的双重评分机制实现可解释的多变量时间序列异常检测与根因定位。

**[Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)**

:   提出 Series-Symbol (S²) 数据生成机制和 SymTime 双模态基础模型，利用 Takens 定理和符号动力学理论生成无限规模的合成时序-符号配对数据（40M 对/50B token），通过跨模态对比学习预训练在 5 大时序任务上达到与真实数据预训练模型竞争的性能。

**[SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)**

:   提出SynTSBench合成数据驱动评估范式，通过可编程的特征配置和理论最优基准，系统评估时间序列预测模型在趋势、周期、依赖性、噪声鲁棒性等维度的实际建模能力。

**[The Human Brain as a Combinatorial Complex](the_human_brain_as_a_combinatorial_complex.md)**

:   提出一种数据驱动的框架，利用 S-信息和 O-信息等信息论度量从 fMRI 时间序列中直接构建组合复形（Combinatorial Complexes），将脑区间的高阶协同交互编码到拓扑结构中，为拓扑深度学习应用于脑网络分析奠定基础。

**[Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)**

:   构建 Time-IMM 数据集——首个按因果机制分类不规则性的多模态多变量时序 benchmark（9 种不规则类型分为触发/约束/伪影三大类，9 个数据集），配套 IMM-TSF 预测库支持异步多模态融合，实验表明显式建模多模态在不规则时序上平均降低 MSE 6.71%，最高达 38.38%。

**[Time-O1: Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)**

:   提出 Time-o1，通过将标签序列变换为去相关且按重要性排序的主成分，解决时间序列预测中 TMSE 损失的自相关偏差和任务过载问题，实现与多种预测模型兼容的 SOTA 性能。

**[TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)**

:   提出 TimePerceiver 统一编码器-解码器框架，通过广义化预测任务（同时包含外推、插值和填补）以及潜在瓶颈编码器 + 查询式解码器设计，在 8 个标准基准上取得全面 SOTA。

**[Transformer Embeddings for Fast Microlensing Inference](transformer_embeddings_for_fast_microlensing_inference.md)**

:   本文将Transformer编码器与神经后验估计（NPE）结合，直接从稀疏、噪声、不等间隔的微引力透镜光变曲线中进行快速且校准良好的参数推断，速度比传统MCMC快10⁴倍以上。

**[Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)**

:   提出首个通用光谱 Tokenizer，通过连续波长嵌入和自监督重建目标，在原始波长网格上联合训练异构天文光谱数据（SDSS/DESI/GALAH/APOGEE），生成对齐、均匀且物理有意义的表征。

**[WaLRUS: Wavelets for Long-range Representation Using SSMs](walrus_wavelets_for_long-range_representation_using_ssms.md)**

:   提出 WaLRUS，基于 Daubechies 小波构建状态空间模型 (SSM)，作为 SaFARi 框架的新实现，扩展了 SSM 家族的多样性，在长程依赖建模中展现独特优势。

**[Wavelet Canonical Coherence for Nonstationary Signals](wavelet_canonical_coherence_for_nonstationary_signals.md)**

:   提出 WaveCanCoh 框架，将经典的典型相干分析（canonical coherence）扩展到小波域，基于多变量局部平稳小波（MvLSW）模型实现对非平稳多变量时间序列两组信号间时变、尺度特定的典型相干性估计。

**[xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories](xlstm-mixer_multivariate_time_series_forecasting_by_mixing_via_scalar_memories.md)**

:   提出 xLSTM-Mixer，首次将扩展长短期记忆网络（sLSTM）与混合架构（Mixer）结合，通过时间混合、联合时间-变量混合和多视角混合三阶段架构实现多变量长期时间序列预测的 SOTA 性能，同时保持极低的内存占用。
