---
title: >-
  ICLR2026 时间序列方向 34篇论文解读
description: >-
  34篇ICLR2026 时间序列方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🔬 ICLR2026** · 共 **34** 篇

**[Adapt Data To Model Adaptive Transformation Optimization For Domain-Shared Time ](adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)**

:   提出TATO框架，通过自动优化数据预处理 pipeline（包括上下文裁切、尺度归一化、异常值校正），让冻结的大型时序模型（LTM）在不微调的情况下适配不同下游领域，平均降低MSE 13.6%，最高65.4%。

**[Contextual And Seasonal Lstms For Time Series Anomaly Detection](contextual_and_seasonal_lstms_for_time_series_anomaly_detection.md)**

:   针对单变量时间序列中现有方法难以检测的"小幅点异常"和"缓慢上升异常"，提出 CS-LSTMs 双分支架构——S-LSTM 在频域建模周期性演化、C-LSTM 在时域捕捉局部趋势，结合小波噪声分解策略，在四个基准上全面超越 SOTA 且推理速度提升 40%。

**[Cpiri Channel Permutation-Invariant Relational Interaction For Multivariate Time](cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time.md)**

:   提出 CPiRi 框架，通过冻结的预训练时序编码器 + 轻量空间 Transformer + 通道打乱训练策略，实现通道排列不变 (CPI) 的跨通道关系建模，在 5 个基准上达到 SOTA 且通道打乱后性能几乎无损 ($\Delta$WAPE < 0.25%)。

**[Cpiri Channel Permutation-Invariant Relational Interaction For Multivariate Time Se](cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)**

:   提出 CPiRi 框架，通过冻结预训练时序编码器 + 可训练置换等变空间模块 + 通道打乱训练策略，在不牺牲跨通道建模能力的前提下实现通道排序不变性（CPI），在多个交通基准上达到 SOTA。

**[Delta-Xai A Unified Framework For Explaining Prediction Changes In Online Time S](delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)**

:   提出 Delta-XAI 统一框架，通过包装函数将14种现有XAI方法适配到在线时间序列预测变化解释场景，并提出 SWING（Shifted Window Integrated Gradients）方法，利用过去观测值构建积分路径以捕获时序依赖关系，在多种评估指标上持续优于现有方法。

**[Edinet-Bench Evaluating Llms On Complex Financial Tasks Using Japanese Financial](edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)**

:   构建了基于日本 EDINET 十年年报的金融基准 EDINET-Bench，包含会计欺诈检测、盈利预测和行业分类三项专家级任务，发现即使是 SOTA LLM 也仅略优于逻辑回归。

**[Enhancing Multivariate Time Series Forecasting With Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)**

:   提出 Global Temporal Retriever（GTR），一个轻量级即插即用模块，通过维护自适应全局周期嵌入并利用绝对时间索引检索对齐全局周期信息，使任意预测模型突破回看窗口限制，有效捕获远超输入长度的全局周期模式。

**[Free Energy Mixer](free_energy_mixer.md)**

:   提出 Free Energy Mixer (FEM)，通过将注意力的值读取重新定义为自由能（log-sum-exp）优化问题，实现了逐通道的值感知后验选择，克服了标准注意力"无损存储但有损读取"的固有瓶颈，可即插即用替换 softmax/线性注意力/RNN/SSM，在 NLP、视觉和时间序列任务上一致提升。

**[From Samples To Scenarios A New Paradigm For Probabilistic Forecasting](from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)**

:   提出 Probabilistic Scenarios 范式，用模型直接输出有限个 {场景, 概率} 对取代采样，并用仅含三层平行线性层的 TimePrism 在5个基准数据集上取得9/10 SOTA。

**[Gtm A General Time-Series Model For Enhanced Representation Learning Of Time-Ser](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-ser.md)**

:   提出 GTM，一个通过频域注意力机制捕获时间粒度感知特征的通用时序基础模型，结合混合掩码预训练策略，首次实现无需任务特定修改即可适配所有生成式时序任务。

**[Gtm A General Time-Series Model For Enhanced Representation Learning Of Time-Series](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)**

:   提出 GTM，一个通过频域注意力机制捕获时间粒度感知特征、并通过混合掩码统一重建与自回归预训练目标的通用时间序列基础模型，在预测、补全、异常检测、分类等多任务上均达到 SOTA。

**[Hivid Llm-Guided Video Saliency For Content-Aware Vod And Live Streaming](hivid_llm-guided_video_saliency_for_content-aware_vod_and_live_streaming.md)**

:   提出 HiVid 框架，首次利用 LLM 作为人类代理为视频块生成内容重要性权重，通过感知模块（滑动窗口评分）、排序模块（LLM 引导归并排序去除评分偏差）和预测模块（多模态时间序列预测自适应延迟）实现内容感知流媒体传输，

**[Language In The Flow Of Time Time-Series-Paired Texts Weaved Into A Unified Temp](language_in_the_flow_of_time_time-series-paired_texts_weaved_into_a_unified_temp.md)**

:   发现时间序列配对文本具有与时间序列相似的周期性（Chronological Textual Resonance），提出 TaTS 框架将文本表征转化为辅助变量，以即插即用方式增强任意现有时间序列模型的预测和插补性能。

**[Learning Recursive Multi-Scale Representations For Irregular Multivariate Time S](learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)**

:   提出 ReIMTS，通过基于时间段的递归分割（而非重采样）来保留不规则多变量时间序列的原始采样模式，结合不规则感知的表示融合机制实现多尺度建模，作为插件在六种 IMTS 骨干上平均提升 27.1%。

**[Paano Patch-Based Representation Learning For Time-Series Anomaly Detection](paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)**

:   提出 PaAno，一种基于 patch 级表示学习的轻量时间序列异常检测方法，使用 1D-CNN 编码器 + triplet loss + pretext loss 学习 patch 嵌入空间，通过与记忆库中正常 patch 的距离计算异常分数，在 TSB-AD 基准上全面 SOTA，且仅需 0.3M 参数和数秒推理。

**[Rating Quality Of Diverse Time Series Data By Meta-Learning From Llm Judgment](rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)**

:   提出TSRating框架，利用LLM从趋势/频率/幅度/模式四个维度对时间序列数据块做成对质量比较，通过Bradley-Terry模型转换为标量质量分数，并以MAML元学习在9个领域22个子集上训练TSRater模型（MOMENT编码器+MLP），实现高效、统一的跨域时间序列数据质量评估。

**[Reasoning On Time-Series For Financial Technical Analysis](reasoning_on_time-series_for_financial_technical_analysis.md)**

:   提出 Verbal Technical Analysis (VTA) 框架，结合 LLM 的语言推理能力与时间序列模型的模式捕捉能力，通过 Time-GRPO 强化学习优化推理链，并以推理属性条件化时序预测，实现了兼具准确性和可解释性的金融时间序列预测。

**[Relational Feature Caching For Accelerating Diffusion Transformers](relational_feature_caching_for_accelerating_diffusion_transformers.md)**

:   提出关系特征缓存（RFC）框架，通过利用DiT模块输入-输出特征之间的强相关性来增强缓存特征预测的精度，包括从输入变化估计输出变化幅度的RFE和用输入误差代理判断是否需要全量计算的RCS，在图像和视频生成任务上显著优于现有的基于时间外推的缓存方法。

**[Relational Transformer Toward Zero-Shot Foundation Models For Relational Data](relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)**

:   提出 Relational Transformer (RT) 架构，通过 task table prompting、cell tokenization 和 Relational Attention 机制，在多个关系数据库上预训练后可零样本迁移到未见过的数据集和任务，22M 参数模型零样本 AUROC 达到全监督方法的 93%，远超 27B LLM 的 84%。

**[Rescp Reservoir Conformal Prediction For Time Series Forecasting](rescp_reservoir_conformal_prediction_for_time_series_forecasting.md)**

:   提出ResCP，首次将储备计算(Echo State Network)用于保形预测的残差重加权，通过储备状态间相似性自适应调权形成局部化预测区间，无需训练即可实现渐近条件覆盖保证，计算效率远超需训练的Transformer方法。

**[Routing Channel-Patch Dependencies In Time Series Forecasting With Graph Spectra](routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)**

:   提出 xCPD 即插即用插件，将多变量时间序列的建模单元从"通道"细化到"通道-patch"，通过共享图傅里叶基做谱嵌入→按频率能量响应分组为低/中/高频段→动态 MoE 路由自适应选择频率特定滤波专家，可无缝集成到 CI/CD 任何现有模型上一致提升长短期预测性能，并支持零样本迁移。

**[Scits Scientific Time Series Understanding And Generation With Llms](scits_scientific_time_series_understanding_and_generation_with_llms.md)**

:   提出SciTS基准覆盖12个科学领域43个任务54K+实例（长度从$10^0$到$10^7$、频率达10MHz），系统评估17个模型发现通用LLM比专用时序模型泛化更好但文本/图像编码各有局限，据此设计TimeOmni框架用多Patch专家+路由机制+Patch重编程显式建模时间动态并与LLM联合训练。

**[Scrapl Scattering Transform With Random Paths For Machine Learning](scrapl_scattering_transform_with_random_paths_for_machine_learning.md)**

:   提出SCRAPL——通过随机采样散射变换(ST)路径将计算量降低P倍的随机优化方案：结合路径自适应动量估计(P-Adam)、路径随机平均梯度(P-SAGA)和θ-重要性采样三种技术稳定单路径梯度的高方差，在DDSP无监督声音匹配任务上实现与全路径ST相近精度但计算效率提升数十倍。

**[Swiftts A Swift Selection Framework For Time Series Pre-Trained Models Via Multi](swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)**

:   提出SwiftTS——首个时间序列预训练模型选择框架：用双编码器架构独立嵌入数据集特征(patch级时序)和模型元信息(架构/训练目标),通过patch级交叉注意力计算兼容性分数,配合horizon自适应专家组合和跨域/跨horizon元学习增强OOD泛化,在14个数据集×8个模型上达SOTA选择性能。

**[T1 One-To-One Channel-Head Binding For Multivariate Time-Series Imputation](t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)**

:   提出T1——CNN-Transformer混合架构通过Channel-Head Binding(CHead Attention)实现鲁棒的多变量时序填充：CNN提取每个变量的多尺度时序特征(每个通道捕捉一种模式)，每个注意力头仅处理对应的一个CNN通道→实现特征级的选择性跨变量信息传递——当缺失导致某通道无法提取有效模式时，对应注意力头自动降权→在11个基准上MSE平均降低46%。

**[Tensor Learning With Orthogonal Lorentz And Symplectic Symmetries](tensor_learning_with_orthogonal_lorentz_and_symplectic_symmetries.md)**

:   本文给出了关于正交群 $O(d)$、不定正交群（含 Lorentz 群）和辛群 $Sp(d)$ 对张量对角作用下的等变多项式函数的完整参数化刻画，并将其应用于设计可学习的稀疏向量恢复算法，在多种数据生成假设下超越了已有的 sum-of-squares 谱方法。

**[Test-Time Efficient Pretrained Model Portfolios For Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)**

:   探索时间序列基础模型的替代范式：不训练单一大模型→而是构建小型预训练模型组合(portfolio)+测试时通过集成/选择组合,发现(1)专家模型组合(各自在特定域/频率上训练)持续优于独立训练的通用组合,(2)从通用模型后训练产出专家→训练计算减少10x,(3)集成/选择在测试时比微调更高效,性能媲美SOTA大型单体模型。

**[Timesliver Symbolic-Linear Decomposition For Explainable Time Series Classificat](timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)**

:   提出TimeSliver——可解释性驱动的深度学习框架,联合利用原始时序数据和符号抽象(分箱)构建保持原始时间结构的表示,每个元素线性编码对应时间段对最终预测的贡献→赋予每个时间点正/负归因分数,在7个数据集上时间归因准确率超越其他方法11%,同时在26个UEA基准上预测性能持平SOTA。

**[Towards Generalizable Pde Dynamics Forecasting Via Physics-Guided Invariant Lear](towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)**

:   提出iMOOE——面向零样本OOD泛化的PDE动力学预测的物理引导不变学习方法：显式定义PDE的两层不变性原则(1.组成算子不变 2.算子间组合关系不变)，设计不变对齐的算子专家混合架构捕获不变算子和组合关系，加频率增强的不变学习目标实现跨域风险均衡→在多种OOD场景(参数/初条件/外力/时间分辨率变化)上实现零样本SOTA。

**[Towards Robust Real-World Multivariate Time Series Forecasting A Unified Framewo](towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)**

:   提出ChannelTokenFormer——同时解决真实世界多变量时序的三大挑战的统一Transformer框架：(1)通道间复杂依赖→channel token跨通道注意力,(2)各通道异步采样→无需重采样/对齐→自然处理不同长度,(3)测试时块缺失→掩码引导注意力跳过缺失→从其他通道推断→在公开基准+真实工业数据上验证鲁棒性和精度。

**[Tspulse Tiny Pre-Trained Models With Disentangled Representations For Rapid Time](tspulse_tiny_pre-trained_models_with_disentangled_representations_for_rapid_time.md)**

:   提出 TSPulse，仅 1M 参数的超轻量时间序列预训练模型，通过双空间掩码重建和双嵌入解耦策略，在分类（+5-16%）、异常检测（+20%）、插补（+50%）和相似性检索（+25%）四大任务上超越 10-100 倍大的模型。

**[Tuning The Burn-In Phase In Training Recurrent Neural Networks Improves Their Pe](tuning_the_burn-in_phase_in_training_recurrent_neural_networks_improves_their_pe.md)**

:   从理论上证明了 RNN 训练中 burn-in 阶段长度 $m$ 对截断反向传播时间（TBPTT）训练性能的关键影响，建立了训练遗憾的上界估计，并通过系统辨识和时间序列预测实验验证，合理调节 burn-in 可将预测误差降低超过 60%。

**[Unlocking The Value Of Text Event-Driven Reasoning And Multi-Level Alignment For](unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)**

:   提出 VoT，一种通过事件驱动推理（利用 LLM 对外生文本进行结构化推理获取数值预测）和多层对齐（表征级内生文本对齐 + 预测级自适应频率融合）充分挖掘文本信息价值的多模态时间序列预测方法，在 10 个领域的真实数据集上全面超越现有方法。

**[Weight-Space Linear Recurrent Neural Networks](weight-space_linear_recurrent_neural_networks.md)**

:   提出 WARP（Weight-space Adaptive Recurrent Prediction），将线性 RNN 的隐状态显式参数化为辅助 MLP 的权重和偏置，利用输入差分驱动线性递推来更新权重，结合非线性解码实现高效序列建模，在分类、预测和动力系统重建等任务上达到 SOTA。
