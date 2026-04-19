---
title: >-
  ICML2025 LLM效率方向 24篇论文解读
description: >-
  24篇ICML2025 LLM效率方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM效率

**🧪 ICML2025** · **24** 篇论文解读

**[Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)](addressing_imbalanced_domain-incremental_learning_through_dual-balance_collabora.md)**

:   DCE 提出频率感知专家组 + 动态专家选择器的双阶段训练框架，同时解决域增量学习中域内类别不平衡和跨域类别分布偏移两个难题，在四个 benchmark 上达到 SOTA。

**[Autonomy-of-Experts Models (AoE)](autonomy-of-experts_models.md)**

:   AoE 提出让 MoE 中的 expert 基于自身内部激活范数自主决定是否处理输入（而非由外部 router 决定），通过低秩权重分解降低预计算开销，在 700M-4B 参数语言模型预训练中超越传统 MoE。

**[Consistency in Language Models: Current Landscape, Challenges, and Future Directions](consistency_in_language_models_current_landscape_challenges_and_future_direction.md)**

:   系统综述了 LLM 一致性研究的全景，提出包含逻辑一致性（否定/对称/传递）、语义一致性、事实/信息一致性和非逻辑一致性（道德/规范）的分类体系，分析了 2019-2025 年间评测方法的不足，并呼吁建立标准化多语言基准和跨学科方法。

**[CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering](costfilter-ad_enhancing_anomaly_detection_through_matching_cost_filtering.md)**

:   将立体匹配/光流估计中的**代价体滤波（cost volume filtering）**思想引入无监督异常检测（UAD），构造输入与模板之间的匹配代价体，并通过3D U-Net 加双流注意力引导进行去噪滤波，作为通用后处理插件可同时提升重建型和嵌入型 UAD 方法的性能，在 MVTec-AD 和 VisA 上取得 SOTA。

**[Curse of High Dimensionality Issue in Transformer for Long-context Modeling](curse_of_high_dimensionality_issue_in_transformer_for_long-context_modeling.md)**

:   本文从监督学习视角重新审视序列建模中的注意力冗余问题，提出了 Dynamic Group Attention (DGA) 机制，通过将不重要的 token 动态分组聚合来减少注意力计算中的冗余，在保持竞争性能的同时大幅降低推理延迟（LLaMA2-7B 在 16K 上下文下推理速度提升 2.42 倍）。

**[Curvature Enhanced Data Augmentation for Regression](curvature_enhanced_data_augmentation_for_regression.md)**

:   提出 CEMS（Curvature-Enhanced Manifold Sampling），利用数据流形的二阶近似（曲率信息）生成合成样本，用于回归任务的数据增强，在分布内和分布外场景均取得 SOTA 或接近 SOTA 的性能。

**[EasyInv: Toward Fast and Better DDIM Inversion](easyinv_toward_fast_and_better_ddim_inversion.md)**

:   提出 EasyInv，通过在反演过程中周期性地将当前 latent 状态与前一步 latent 状态加权聚合（类卡尔曼滤波），增强初始 latent 的影响力、抑制噪声累积误差，在不需要迭代优化的前提下达到与迭代方法相当甚至更好的反演质量，同时推理速度提升约 3 倍。

**[Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](efficient_length-generalizable_attention_via_causal_retrieval_for_long-context_l.md)**

:   本文提出 Grouped Cross-Attention (GCA) 机制，将 chunk 级别的因果检索（causal retrieval）集成到注意力中实现端到端可学习的检索器，构建的 Differentiable Retrieval-based Transformer (DRT) 在 16M 上下文的 passkey 检索测试中达到近乎完美的准确率，实现了训练长度 1000 倍的长度泛化。

**[Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)**

:   用离散信号处理 (DSP) 理论揭示 RoPE 隐式实现了非均匀 DFT 以编码周期性注意力，但线性层导致的频谱泄漏、激活函数导致的频谱畸变以及训练长度截断导致的欠训练频率分量会破坏这一周期性，据此提出 FoPE——将每个维度建模为傅里叶级数（而非单一频率）并裁剪破坏性低频分量为零频，在多个模型规模和任务上显著改善长度泛化。

**[Ladder Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference](ladder-residual_parallelism-aware_architecture_for_accelerating_large_model_infe.md)**

:   本文提出 Ladder Residual，一种简单的架构修改——将每个模块的输入从上一层的输出改为上上层的输出（错位残差），使模块计算与 AllReduce 通信解耦，从而实现通信与计算的重叠，在 70B 模型 8 卡 TP 推理中实现 29% 的端到端加速，且模型性能与标准 Transformer 持平。

**[Long-Short Alignment for Effective Long-Context Modeling in LLMs](long-short_alignment_for_effective_long-context_modeling_in_llms.md)**

:   本文从模型输出分布的角度提出长度泛化的新视角——长短对齐 (Long-Short Alignment)，指出不同长度输入的输出分布一致性是长度泛化的关键因素，提出 Long-Short Misalignment 度量并将其作为训练正则项，在合成任务和自然语言任务上均显著提升长上下文建模能力。

**[Mixture of Lookup Experts](mixture_of_lookup_experts.md)**

:   提出 MoLE（Mixture of Lookup Experts），将 MoE 中的路由专家输入从中间特征改为 embedding token，使专家可在推理前被重参数化为查找表（LUT）并卸载到存储设备，从而在保持 MoE 级别性能的同时实现与 dense 模型相当的推理速度和显存占用。

**[MoH: Multi-Head Attention as Mixture-of-Head Attention](moh_multi-head_attention_as_mixture-of-head_attention.md)**

:   本文将多头注意力（MHA）重新表述为求和形式，借鉴 MoE 思想提出 Mixture-of-Head Attention（MoH），通过路由器为每个 token 动态选择最相关的注意力头子集，仅激活 50%~90% 的头即可匹配甚至超越标准 MHA 性能，并证明预训练模型（如 LLaMA3-8B）可通过 continue-tuning 转换为 MoH 模型。

**[NExtLong: Toward Effective Long-Context Training without Long Documents](nextlong_toward_effective_long-context_training_without_long_documents.md)**

:   本文提出 NExtLong 框架，通过将文档分割为 meta-chunk 并在 chunk 之间插入从预训练语料检索的硬负例干扰文本来合成长上下文训练数据，迫使模型区分长距离依赖信息和干扰内容，在 HELMET 和 RULER 基准上比此前最佳的长上下文合成方法 Quest 平均提升 7.33%。

**[Online Sparsification of Bipartite-Like Clusters in Graphs](online_sparsification_of_bipartite-like_clusters_in_graphs.md)**

:   提出了一种**近线性时间的在线图稀疏化算法**，能在保留图的二部图式聚类（bipartite-like clusters）结构的前提下，将边数压缩到 $\widetilde{O}(n)$，同时适用于无向图和有向图，显著加速现有聚类算法。

**[PENCIL: Long Thoughts with Short Memory](pencil_long_thoughts_with_short_memory.md)**

:   提出 **PENCIL**（PENCIL ENables Context-efficient Inference and Learning），在自回归生成过程中引入受函数调用栈启发的**归约规则（reduction rule）**，递归地清除不再需要的中间推理步骤，使LLM能以多项式级上下文长度解决本需指数级上下文的计算难题。

**[Position: Theory of Mind Benchmarks are Broken for Large Language Models](position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)**

:   这篇 Position Paper 指出当前大多数 LLM Theory of Mind（ToM）基准只测“能否预测他人行为”（Literal ToM），却没有测“能否基于该预测采取最优响应”（Functional ToM），因此会系统性高估模型在真实交互中的适应能力。

**[Ranked Entropy Minimization for Continual Test-Time Adaptation](ranked_entropy_minimization_for_continual_test-time_adaptation.md)**

:   提出 Ranked Entropy Minimization (REM)，通过渐进式遮挡策略构建预测难度的显式排序结构，结合遮挡一致性损失和熵排序损失，解决了熵最小化方法在持续测试时自适应(CTTA)中的模型崩塌问题，同时保持了计算效率。

**[Rejecting Hallucinated State Targets during Planning](rejecting_hallucinated_state_targets_during_planning.md)**

:   本文系统识别了目标导向决策规划中生成器产生不可行目标（幻觉目标）导致的"妄想行为"类型，并设计了一种可行性评估器（feasibility evaluator）作为附加模块来识别和拒绝这些不可行目标，结合离策略学习规则、分布式架构和后见重标记数据增强，在不修改原始智能体的前提下显著减少妄想行为并提升OOD泛化性能。

**[Retraining-Free Merging of Sparse MoE via Hierarchical Clustering](retraining-free_merging_of_sparse_moe_via_hierarchical_clustering.md)**

:   提出 HC-SMoE，一种基于专家输出层次聚类的无需重训练专家合并框架，通过输出相似度度量和层次聚类实现 SMoE 模型的高效压缩，在 Qwen 和 Mixtral 上分别实现 25%-50% 的专家参数缩减并保持优越性能。

**[Safe Delta: Consistently Preserving Safety when Fine-Tuning LLMs on Diverse Datasets](safe_delta_consistently_preserving_safety_when_fine-tuning_llms_on_diverse_datas.md)**

:   Safe Delta提出了一种安全感知的后训练防御方法，通过估计安全退化程度、选择性保留delta参数以最大化效用同时限制安全损失、并施加安全补偿向量来弥补残余安全损失，在多种微调数据集（不同规模、任务类型）上一致地保持LLM安全性而不牺牲效用。

**[Scaling Inference-Efficient Language Models](scaling_inference-efficient_language_models.md)**

:   本文提出了推理感知的 Scaling Law，通过在 Chinchilla 损失函数中引入模型宽高比（aspect ratio）项来联合优化参数量、训练 token 数和模型形状，训练 63 个模型拟合该定律后指导设计了 Morph-1B 模型，在保持下游任务精度的同时实现 1.8× 推理延迟提升。

**[SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator](sepllm_accelerate_large_language_models_by_compressing_one_segment_into_one_sepa.md)**

:   SepLLM 发现分隔符 token（标点等）在注意力中占据主导地位，提出将文本段信息压缩到分隔符 token 中，通过数据依赖的稀疏注意力掩码仅保留 Initial + Separator + Neighboring tokens 的 KV cache，实现 50%+ 的 KV cache 压缩且性能几乎无损。

**[Star Attention: Efficient LLM Inference over Long Sequences](star_attention_efficient_llm_inference_over_long_sequences.md)**

:   提出Star Attention两阶段块稀疏注意力：第一阶段将上下文分块在多主机上局部注意力编码，第二阶段查询通过聚合全局注意力生成，无需微调即可兼容现有LLM，推理加速11倍且保持97-100%精度。
