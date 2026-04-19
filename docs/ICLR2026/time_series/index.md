---
title: >-
  ICLR2026 时间序列方向 32篇论文解读
description: >-
  32篇ICLR2026 时间序列方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🔬 ICLR2026** · **32** 篇论文解读

**[Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)**

:   提出TATO框架，通过自动优化数据预处理 pipeline（包括上下文裁切、尺度归一化、异常值校正），让冻结的大型时序模型（LTM）在不微调的情况下适配不同下游领域，平均降低MSE 13.6%，最高65.4%。

**[Contextual and Seasonal LSTMs for Time Series Anomaly Detection](contextual_and_seasonal_lstms_for_time_series_anomaly_detection.md)**

:   针对单变量时间序列中现有方法难以检测的"小幅点异常"和"缓慢上升异常"，提出 CS-LSTMs 双分支架构——S-LSTM 在频域建模周期性演化、C-LSTM 在时域捕捉局部趋势，结合小波噪声分解策略，在四个基准上全面超越 SOTA 且推理速度提升 40%。

**[CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time.md)**

:   提出 CPiRi 框架，通过冻结的预训练时序编码器 + 轻量空间 Transformer + 通道打乱训练策略，实现通道排列不变 (CPI) 的跨通道关系建模，在 5 个基准上达到 SOTA 且通道打乱后性能几乎无损 ($\Delta$WAPE < 0.25%)。

**[Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)**

:   提出 Delta-XAI 统一框架，通过包装函数将14种现有XAI方法适配到在线时间序列预测变化解释场景，并提出 SWING（Shifted Window Integrated Gradients）方法，利用过去观测值构建积分路径以捕获时序依赖关系，在多种评估指标上持续优于现有方法。

**[EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)**

:   构建了基于日本 EDINET 十年年报的金融基准 EDINET-Bench，包含会计欺诈检测、盈利预测和行业分类三项专家级任务，发现即使是 SOTA LLM 也仅略优于逻辑回归。

**[Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)**

:   提出 Global Temporal Retriever（GTR），一个轻量级即插即用模块，通过维护自适应全局周期嵌入并利用绝对时间索引检索对齐全局周期信息，使任意预测模型突破回看窗口限制，有效捕获远超输入长度的全局周期模式。

**[Free Energy Mixer](free_energy_mixer.md)**

:   提出 Free Energy Mixer (FEM)，通过将注意力的值读取重新定义为自由能（log-sum-exp）优化问题，实现了逐通道的值感知后验选择，克服了标准注意力"无损存储但有损读取"的固有瓶颈，可即插即用替换 softmax/线性注意力/RNN/SSM，在 NLP、视觉和时间序列任务上一致提升。

**[From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting](from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)**

:   提出 Probabilistic Scenarios 范式，用模型直接输出有限个 {场景, 概率} 对取代采样，并用仅含三层平行线性层的 TimePrism 在5个基准数据集上取得9/10 SOTA。

**[GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-ser.md)**

:   提出 GTM，一个通过频域注意力机制捕获时间粒度感知特征的通用时序基础模型，结合混合掩码预训练策略，首次实现无需任务特定修改即可适配所有生成式时序任务。

**[HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming](hivid_llm-guided_video_saliency_for_content-aware_vod_and_live_streaming.md)**

:   提出 HiVid 框架，首次利用 LLM 作为人类代理为视频块生成内容重要性权重，通过感知模块（滑动窗口评分）、排序模块（LLM 引导归并排序去除评分偏差）和预测模块（多模态时间序列预测自适应延迟）实现内容感知流媒体传输，

**[Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative](language_in_the_flow_of_time_time-series-paired_texts_weaved_into_a_unified_temp.md)**

:   发现时间序列配对文本具有与时间序列相似的周期性（Chronological Textual Resonance），提出 TaTS 框架将文本表征转化为辅助变量，以即插即用方式增强任意现有时间序列模型的预测和插补性能。

**[Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)**

:   提出 ReIMTS，通过基于时间段的递归分割（而非重采样）来保留不规则多变量时间序列的原始采样模式，结合不规则感知的表示融合机制实现多尺度建模，作为插件在六种 IMTS 骨干上平均提升 27.1%。

**[PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)**

:   提出 PaAno，一种基于 patch 级表示学习的轻量时间序列异常检测方法，使用 1D-CNN 编码器 + triplet loss + pretext loss 学习 patch 嵌入空间，通过与记忆库中正常 patch 的距离计算异常分数，在 TSB-AD 基准上全面 SOTA，且仅需 0.3M 参数和数秒推理。

**[Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)**

:   提出TSRating框架，利用LLM从趋势/频率/幅度/模式四个维度对时间序列数据块做成对质量比较，通过Bradley-Terry模型转换为标量质量分数，并以MAML元学习在9个领域22个子集上训练TSRater模型（MOMENT编码器+MLP），实现高效、统一的跨域时间序列数据质量评估。

**[Reasoning on Time-Series for Financial Technical Analysis](reasoning_on_time-series_for_financial_technical_analysis.md)**

:   提出 Verbal Technical Analysis (VTA) 框架，结合 LLM 的语言推理能力与时间序列模型的模式捕捉能力，通过 Time-GRPO 强化学习优化推理链，并以推理属性条件化时序预测，实现了兼具准确性和可解释性的金融时间序列预测。

**[Relational Feature Caching for Accelerating Diffusion Transformers](relational_feature_caching_for_accelerating_diffusion_transformers.md)**

:   提出关系特征缓存（RFC）框架，通过利用DiT模块输入-输出特征之间的强相关性来增强缓存特征预测的精度，包括从输入变化估计输出变化幅度的RFE和用输入误差代理判断是否需要全量计算的RCS，在图像和视频生成任务上显著优于现有的基于时间外推的缓存方法。

**[Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)**

:   提出 Relational Transformer (RT) 架构，通过 task table prompting、cell tokenization 和 Relational Attention 机制，在多个关系数据库上预训练后可零样本迁移到未见过的数据集和任务，22M 参数模型零样本 AUROC 达到全监督方法的 93%，远超 27B LLM 的 84%。

**[ResCP: Reservoir Conformal Prediction for Time Series Forecasting](rescp_reservoir_conformal_prediction_for_time_series_forecasting.md)**

:   首次将储备计算（Echo State Network）引入保形预测，通过随机初始化ESN编码残差序列的时间动态，利用状态相似性自适应重加权历史残差构建局部预测区间，无需任何训练即在4个真实数据集上实现SOTA的Winkler分数，速度比HopCPT快20-80×。

**[Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)**

:   提出 xCPD 即插即用插件，将多变量时间序列的建模单元从"通道"细化到"通道-patch"，通过共享图傅里叶基做谱嵌入→按频率能量响应分组为低/中/高频段→动态 MoE 路由自适应选择频率特定滤波专家，可无缝集成到 CI/CD 任何现有模型上一致提升长短期预测性能，并支持零样本迁移。

**[scits scientific time series understanding and generation with llms](scits_scientific_time_series_understanding_and_generation_with_llms.md)**

:   提出SciTS基准覆盖12个科学领域43个任务54K+实例（长度从$10^0$到$10^7$、频率达10MHz），系统评估17个模型发现通用LLM比专用时序模型泛化更好但文本/图像编码各有局限，据此设计TimeOmni框架用多Patch专家+路由机制+Patch重编程显式建模时间动态并与LLM联合训练。

**[SCRAPL: Scattering Transform with Random Paths for Machine Learning](scrapl_scattering_transform_with_random_paths_for_machine_learning.md)**

:   针对多变量散射变换(ST)作为可微损失函数时因路径数P过多导致计算代价过高的问题，提出SCRAPL——每步仅随机采样一条路径并通过P-Adam（路径自适应动量）、P-SAGA（路径随机平均梯度）和θ-重要性采样三种方差缩减技术来稳定梯度，在无监督声音匹配任务上以接近全路径ST的精度、MSS级别的低计算成本实现了Pareto最优。

**[SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)**

:   提出首个时间序列预训练模型选择框架SwiftTS，使用双编码器架构独立嵌入数据集patch级时序特征和模型元信息（架构/拓扑/功能），通过patch级交叉注意力计算兼容性分数，结合horizon自适应专家组合和跨域/跨horizon元学习，在14个数据集×8个模型上以平均加权Kendall $\tau_\omega = 0.442$ 大幅超越所有基线。

**[T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)**

:   提出T1——CNN-Transformer混合架构，核心创新是Channel-Head Binding（CHead Attention）：共享Depthwise Conv为每个变量提取C种时序特征（趋势/周期/突变等），然后将每个CNN通道与一个注意力头一对一绑定，使跨变量信息传递在特征级别独立进行。当缺失导致某通道无法提取有效模式时，对应注意力头自动降权，实现无需显式设计的自适应缺失处理。在11个基准数据集上MSE平均降低46%，70%极端缺失下优势更大。

**[Tensor learning with orthogonal, Lorentz, and symplectic symmetries](tensor_learning_with_orthogonal_lorentz_and_symplectic_symmetries.md)**

:   本文给出了关于正交群 $O(d)$、不定正交群（含 Lorentz 群）和辛群 $Sp(d)$ 对张量对角作用下的等变多项式函数的完整参数化刻画，并将其应用于设计可学习的稀疏向量恢复算法，在多种数据生成假设下超越了已有的 sum-of-squares 谱方法。

**[Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)**

:   提出 Chroma——小型预训练时序模型组合（portfolio）框架：从通用模型通过后训练（post-training）产出频率/领域专家（训练加速 10×），测试时通过模型选择或贪心集成组合，4M 参数的 portfolio 在 Chronos Benchmark II 上匹配 205M-500M 参数的大型单体模型性能，同时推理计算远低于 test-time fine-tuning。

**[TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification](timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)**

:   提出TimeSliver——可解释性驱动的深度学习框架,联合利用原始时序数据和符号抽象(分箱)构建保持原始时间结构的表示,每个元素线性编码对应时间段对最终预测的贡献→赋予每个时间点正/负归因分数,在7个数据集上时间归因准确率超越其他方法11%,同时在26个UEA基准上预测性能持平SOTA。

**[Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)**

:   提出 iMOOE 框架，通过显式定义 PDE 系统中的"算子不变性 + 组合不变性"两层物理不变性原理，设计与之对齐的混合算子专家网络和频率增强的风险等式目标，在不需要任何测试时适应的条件下实现多种 OOD 情景下的 SOTA 零样本 PDE 动力学预测。

**[Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)**

:   提出ChannelTokenFormer（CTF），一个统一的Transformer框架同时解决真实世界多变量时序预测的三大挑战：(1) 通道间复杂依赖——channel token跨通道注意力；(2) 各通道异步采样——频域动态patching保持原始分辨率；(3) 测试时块缺失——训练时patch masking模拟+推理时直接移除全缺失patch，在ETT/SolarWind/Weather/EPA/CHS等6个数据集上全面SOTA。

**[TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time Series](tspulse_tiny_pre-trained_models_with_disentangled_representations_for_rapid_time.md)**

:   提出 TSPulse，仅 1M 参数的超轻量时间序列预训练模型，通过双空间掩码重建和双嵌入解耦策略，在分类（+5-16%）、异常检测（+20%）、插补（+50%）和相似性检索（+25%）四大任务上超越 10-100 倍大的模型。

**[调节 RNN 训练中的 Burn-in 阶段可提升性能](tuning_the_burn-in_phase_in_training_recurrent_neural_networks_improves_their_pe.md)**

:   从理论上证明了 RNN 训练中 burn-in 阶段长度 $m$ 对截断反向传播时间（TBPTT）训练性能的关键影响，建立了训练遗憾的上界估计，并通过系统辨识和时间序列预测实验验证，合理调节 burn-in 可将预测误差降低超过 60%。

**[VoT: 事件驱动推理与多层对齐解锁文本价值用于时间序列预测](unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)**

:   提出 VoT，一种通过事件驱动推理（利用 LLM 对外生文本进行结构化推理获取数值预测）和多层对齐（表征级内生文本对齐 + 预测级自适应频率融合）充分挖掘文本信息价值的多模态时间序列预测方法，在 10 个领域的真实数据集上全面超越现有方法。

**[WARP: 权重空间线性循环神经网络](weight-space_linear_recurrent_neural_networks.md)**

:   提出 WARP（Weight-space Adaptive Recurrent Prediction），将线性 RNN 的隐状态显式参数化为辅助 MLP 的权重和偏置，利用输入差分驱动线性递推来更新权重，结合非线性解码实现高效序列建模，在分类、预测和动力系统重建等任务上达到 SOTA。
