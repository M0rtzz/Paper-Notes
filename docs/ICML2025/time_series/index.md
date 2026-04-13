---
title: >-
  ICML2025 时间序列方向 25篇论文解读
description: >-
  25篇ICML2025 时间序列方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📈 时间序列

**🧪 ICML2025** · 共 **25** 篇

**[Are Llms Prescient A Continuous Evaluation Using Daily News As The Oracle](are_llms_prescient_a_continuous_evaluation_using_daily_news_as_the_oracle.md)**

:   提出 Daily Oracle 基准——从每日新闻自动生成 QA 对来持续评估 LLM 的时间泛化和预测能力，揭示 LLM 性能随预训练数据过时而持续退化（True/False 题平均下降 21.55%），且 RAG 也无法完全挽救。

**[Breaking Silos Adaptive Model Fusion Unlocks Better Time Series Forecasting](breaking_silos_adaptive_model_fusion_unlocks_better_time_series_forecasting.md)**

:   提出 TimeFuse——一个样本级自适应模型融合框架，通过元特征描述输入时间序列特征并训练可学习融合器预测最优模型组合权重，在多个预测基准上对 SOTA 模型实现近乎普遍的改进（95.1% 样本优于最佳单模型）。

**[Causal Discovery From Conditionally Stationary Time Series](causal_discovery_from_conditionally_stationary_time_series.md)**

:   提出 SDCI（State-Dependent Causal Inference）——处理条件平稳时间序列的因果发现方法，通过离散潜状态变量建模非平稳行为，实现状态依赖的因果结构恢复，在粒子交互、基因调控网络和 NBA 球员运动预测中验证有效性。

**[Causality-Aware Contrastive Learning For Robust Multivariate Time-Series Anomaly](causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)**

:   提出 CAROTS——将因果关系融入对比学习的多变量时间序列异常检测框架，用因果保持增强作为正样本（正常变化），因果破坏增强作为负样本（模拟异常），训练编码器基于因果结构区分正常与异常。

**[Channel Normalization For Time Series Channel Identification](channel_normalization_for_time_series_channel_identification.md)**

:   提出通道归一化（Channel Normalization, CN），通过为每个通道分配独立的仿射变换参数来增强时间序列模型的通道可辨识性（CID），并扩展出自适应版本 ACN（动态调整参数）和原型版本 PCN（支持未知/可变通道数），在多种时间序列模型上实现显著性能提升。

**[Context Is Key A Benchmark For Forecasting With Essential Textual Information](context_is_key_a_benchmark_for_forecasting_with_essential_textual_information.md)**

:   提出 CiK 基准——将数值时间序列与精心设计的文本上下文配对，要求模型必须理解文本才能成功预测，评估统计模型、时序基础模型和 LLM 预测器，发现简单的 LLM 提示方法在需要上下文信息的任务上表现最优。

**[Customizing The Inductive Biases Of Softmax Attention Using Structured Matrices](customizing_the_inductive_biases_of_softmax_attention_using_structured_matrices.md)**

:   提出用高效结构化矩阵（BTT 和 MLR）替换 softmax attention 中的低秩打分函数，既解决了标准 attention 的低秩瓶颈问题，又通过 MLR 引入了距离依赖的计算偏置，在上下文回归、语言建模和长程时间序列预测上均取得改进。

**[Event-Aware Sentiment Factors From Llm-Augmented Financial Tweets A Transparent ](event-aware_sentiment_factors_from_llm-augmented_financial_tweets_a_transparent_.md)**

:   利用大语言模型对金融推文进行多标签事件分类标注，将非结构化社交媒体文本转化为结构化、可解释的事件驱动量化因子，发现特定事件类别（如谣言/投机）具有显著的负Alpha信号（Sharpe ratio低至-0.38）。

**[Hyperimts Hypergraph Neural Network For Irregular Multivariate Time Series Forec](hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)**

:   提出 HyperIMTS，利用超图结构表示不规则多元时间序列（IMTS）中的观测值和其依赖关系，通过三种消息传递机制（节点→超边、超边→超边、超边→节点）实现不规则性感知的时间和变量依赖学习，在 5 个 IMTS 数据集上达到 SOTA 且计算效率优于 padding 方法。

**[Imts Is Worth Time Times Channel Patches Visual Masked Autoencoders For Irregula](imts_is_worth_time_times_channel_patches_visual_masked_autoencoders_for_irregula.md)**

:   提出 VIMTS 框架，将不规则多变量时间序列（IMTS）转化为 time × channel 的类图像 patch 结构，借助在大规模 RGB 图像上预训练的视觉 MAE 的稀疏多通道建模能力，结合 GCN 跨通道补全与粗到细预测策略，在 IMTS 预测任务上实现 SOTA 性能和强 few-shot 能力。

**[Kan-Ad Time Series Anomaly Detection With Kolmogorov-Arnold Networks](kan-ad_time_series_anomaly_detection_with_kolmogorov-arnold_networks.md)**

:   KAN-AD 将时间序列异常检测重新建模为用光滑单变量函数逼近序列，用截断傅里叶展开替代 KAN 中的 B 样条避免局部扰动敏感性，以不到 1000 个参数在 4 个基准上平均提升 15% 检测精度。

**[Learning Soft Sparse Shapes For Efficient Time-Series Classification](learning_soft_sparse_shapes_for_efficient_time-series_classification.md)**

:   提出 SoftShape 模型，用基于贡献分数的软稀疏化替代传统硬筛选 shapelet 的方式，结合 MoE 驱动的 intra-shape 和 shared expert 的 inter-shape 双模式时序模式学习，在 128 个 UCR 数据集上取得 SOTA 分类精度。

**[Lightgts A Lightweight General Time Series Forecasting Model](lightgts_a_lightweight_general_time_series_forecasting_model.md)**

:   提出 LightGTS，利用时间序列固有的尺度不变周期性归纳偏置，通过 Periodical Tokenization 和 Periodical Parallel Decoding 两个核心技术，仅用不到 500 万参数就在 9 个基准数据集上的 zero-shot 和 full-shot 设定中取得了 SOTA 性能，比现有时序基础模型小 10-100 倍。

**[Lyapunov Learning At The Onset Of Chaos](lyapunov_learning_at_the_onset_of_chaos.md)**

:   提出 Lyapunov Learning 算法，通过将神经网络视为动力系统并在损失函数中加入 Lyapunov 指数正则项，将网络推向混沌边缘（edge of chaos），从而在非平稳时间序列发生 regime shift 时实现快速自适应，在 Lorenz 系统实验中将 post-shift MSE 降低约 96%。

**[Risk And Cross Validation In Ridge Regression With Correlated Samples](risk_and_cross_validation_in_ridge_regression_with_correlated_samples.md)**

:   利用随机矩阵理论和自由概率技术，为训练样本具有任意相关性的高维岭回归推导了精确的风险渐近公式，提出了修正的广义交叉验证估计器 CorrGCV，在样本相关条件下准确预测样本外风险。

**[Temporal Query Network For Efficient Multivariate Time Series Forecasting](temporal_query_network_for_efficient_multivariate_time_series_forecasting.md)**

:   提出Temporal Query(TQ)技术——用周期性移位的可学习向量作为注意力查询来捕捉全局变量间相关性，keys/values来自原始数据保留局部信息，单层注意力+浅MLP的TQNet即达SOTA。

**[Timepoint Accelerated Time Series Alignment Via Self-Supervised Keypoint And Des](timepoint_accelerated_time_series_alignment_via_self-supervised_keypoint_and_des.md)**

:   提出 TimePoint——受 2D 关键点检测启发但针对 1D 信号重新设计的自监督方法，通过学习时间序列的关键点和描述子实现稀疏表示，将 DTW 应用于稀疏关键点而非完整信号，在大幅加速对齐的同时通常提升对齐精度。

**[Timepro Efficient Multivariate Long-Term Time Series Forecasting With Variable- ](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)**

:   提出基于 Mamba 的 TimePro 模型，通过构建变量感知和时间感知的超级状态（hyper-state），自适应选择关键时间点来调节变量维度的隐状态，以线性复杂度实现高效的多变量长期时间序列预测。

**[Timing Temporality-Aware Integrated Gradients For Time Series Explanation](timing_temporality-aware_integrated_gradients_for_time_series_explanation.md)**

:   提出 TIMING 方法，通过引入时序感知的分段随机掩码基线改进 Integrated Gradients，同时设计新评估指标 CPD/CPP 解决现有时序 XAI 评估中正负归因相互抵消的问题，在多个真实数据集上全面超越现有基线。

**[Transpl Vq-Code Transition Matrices For Pseudo-Labeling Of Time Series Unsupervi](transpl_vq-code_transition_matrices_for_pseudo-labeling_of_time_series_unsupervi.md)**

:   提出 TransPL，通过将时间序列 patch 离散化为 VQ 码并构建类别-通道级转移矩阵，利用贝叶斯定理在目标域生成可解释伪标签，实现时间序列无监督域适应中平均 6.1% 准确率和 4.9% F1 的提升。

**[Understanding The Limits Of Deep Tabular Methods With Temporal Shift](understanding_the_limits_of_deep_tabular_methods_with_temporal_shift.md)**

:   揭示深度表格模型在时间分布偏移下失败的根因——训练滞后与验证偏差导致模型选择失效，以及模型表示丢失周期/趋势信息——并提出改进的时序划分策略和基于傅里叶级数的即插即用时间嵌入方法。

**[Wave Weighted Autoregressive Varying Gate For Time Series Forecasting](wave_weighted_autoregressive_varying_gate_for_time_series_forecasting.md)**

:   将经典统计学中的ARMA（自回归移动平均）结构引入自回归Transformer注意力机制，通过间接MA权重生成方法在不增加时间复杂度和参数量的前提下，解耦长短期时序模式，显著提升时间序列预测性能。

**[When Will It Fail Anomaly To Prompt For Forecasting Future Anomalies In Time Ser](when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_ser.md)**

:   提出 Anomaly to Prompt (A2P) 框架，通过异常感知预测 (AAF) 和合成异常提示 (SAP) 两大模块，首次有效解决时间序列中"未来异常预测"(Anomaly Prediction) 这一新任务——不仅预测未来信号走势，还能精准定位未来哪些时间点会出现异常。

**[When Will It Fail Anomaly To Prompt For Forecasting Future Anomalies In Time Seri](when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_seri.md)**

:   提出A2P框架解决"异常预测(AP)"新任务——预测未来哪些时间点会发生异常，通过Anomaly-Aware Forecasting让预测模型学习异常关系+Synthetic Anomaly Prompting用可学习prompt模拟多样异常模式。

**[Winner-Takes-All For Multivariate Probabilistic Time Series Forecasting](winner-takes-all_for_multivariate_probabilistic_time_series_forecasting.md)**

:   提出 TimeMCL，将 Multiple Choice Learning 的 Winner-Takes-All (WTA) 损失引入多变量概率时序预测，通过多头网络单次前向传播即可生成多样且具代表性的未来轨迹，兼顾预测质量与计算效率。
