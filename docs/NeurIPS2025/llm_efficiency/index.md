---
title: >-
  NeurIPS2025 LLM效率方向 43篇论文解读
description: >-
  43篇NeurIPS2025 LLM效率方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM效率

**🧠 NeurIPS2025** · **43** 篇论文解读

**[3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)**

:   在标准的draft-target两模型推测解码的中间插入一个"qualifier"模型，构成三层金字塔式解码架构（PyramidSD），利用模型家族天然的熵梯度来分级过滤token，以模糊接受准则放宽匹配阈值，实现最高1.91×的速度提升（在RTX 4090上达到124 tok/s）。

**[A Unified Framework for Establishing the Universal Approximation of Transformer-Type Architectures](a_unified_framework_for_establishing_the_universal_approxima.md)**

:   建立了统一的理论框架证明各类Transformer架构的万能逼近性(UAP)，核心条件仅两个——前馈层的非线性仿射不变性和注意力层的token可区分性——并利用解析性假设将后者简化为仅需检验两样本情况，成功覆盖softmax、RBF kernel、Performer、BigBird、Linformer等多种实用架构。

**[Advancing Expert Specialization for Better MoE](advancing_expert_specialization_for_better_moe.md)**

:   通过正交性损失（减少专家间投影重叠）和方差损失（增大路由分数差异）双目标优化，在不修改 MoE 架构的前提下将专家重叠减少 45%、路由方差提升 150%，11 个基准任务平均提升 23.79%，同时完全保持负载均衡。

**[Approximately Aligned Decoding](approximately_aligned_decoding.md)**

:   提出 Approximately Aligned Decoding (AprAD)，一种利用投机解码（speculative decoding）中的前缀选择算法来实现LLM受约束生成的方法——在遇到约束违反时，既不像约束生成那样仅回退一步（导致极端概率放大），也不像ASAp那样完全重新采样（计算成本过高），而是通过投机采样智能选择回退位置，在输出分布失真和计算效率之间取得良好平衡。

**[Constant Bit-Size Transformers Are Turing Complete](constant_bit-size_transformers_are_turing_complete.md)**

:   首次证明常数 bit 精度、固定参数数量的 Transformer（仅允许上下文窗口增长）是图灵完备的，并建立了精确的复杂度等价关系 WINDOW[s(n)] = SPACE[s(n)]，表明扩展上下文窗口——而非模型尺寸——已足以实现通用计算。

**[Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag.md)**

:   提出 branched training 方法直接实证测量临界 batch size (CBS)，发现 CBS 在训练早期快速增长后趋于平稳且不依赖模型规模，据此设计 batch size warmup 策略以 43% 更少的梯度步数达到同等甚至更优的训练 loss。

**[Deep Compositional Phase Diffusion for Long Motion Sequence Generation](deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)**

:   提出 Compositional Phase Diffusion 框架，在 ACT-PAE 建立的频域相位空间中用 SPDM 和 TPDM 分别处理语义对齐和过渡连续性，实现长程组合式动作序列生成，在 BABEL-TEACH 上达到 SOTA。

**[Dense Associative Memory with Epanechnikov Energy](dense_associative_memory_with_epanechnikov_energy.md)**

:   提出基于 Epanechnikov 核的 log-sum-ReLU（LSR）能量函数替代传统的 log-sum-exp（LSE），在 Dense Associative Memory 中首次实现了"精确记忆所有模式 + 同时涌现新的创造性局部极小"的共存，且保持指数级记忆容量。

**[DISC: Dynamic Decomposition Improves LLM Inference Scaling](disc_dynamic_decomposition_improves_llm_inference_scaling.md)**

:   DISC 提出了一种动态分解算法，在推理时根据每一步的 z-score（采样奖励的标准化最大值）自动、递归地调整推理步骤的粒度——困难步骤分更细、简单步骤一步跨过——可以即插即用地与贪心搜索、Beam Search、MCTS 结合，在 APPS、MATH、LiveCodeBench 上以更少的 token 预算达到更高的 pass@k。

**[Document Summarization with Conformal Importance Guarantees](document_summarization_with_conformal_importance_guarantees.md)**

:   首次将Conformal Prediction应用于文档摘要，通过校准句子重要性分数的阈值，为抽取式摘要提供用户可控的覆盖率($1-\alpha$)和召回率($\beta$)的严格统计保证，方法模型无关且仅需小规模校准集。

**[Dynamics of Spontaneous Topic Changes in Next Token Prediction with Self-Attention](dynamics_of_spontaneous_topic_changes_in_next_token_prediction_with_self-attenti.md)**

:   从理论和实验两方面研究自注意力模型中"自发主题切换"的动力学机制，证明在单层 self-attention 模型中：(1) 混合主题训练保持原主题的 token 优先级顺序；(2) 主题切换仅在低优先级 token 数量超过高优先级 token 时发生；(3) 更长输入和更模糊主题不会增加切换概率——与人类认知相反。

**[Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)**

:   提出首个无需训练的在线 LLM 路由算法 PORT，通过近似最近邻搜索估计查询特征，并在少量初始查询上一次性优化对偶变量作为路由权重，在有限 token 预算下实现接近离线最优 ($1-o(1)$ 竞争比) 的路由性能，平均较基线提升 3.55× 性能、1.85× 成本效率和 4.25× 吞吐量。

**[Frequency-Aware Token Reduction for Efficient Vision Transformer](frequency-aware_token_reduction_for_efficient_vision_transformer.md)**

:   从频域视角提出 frequency-aware token reduction，将 token 分为高频（HF）和低频（LF）两组，选择性保留 HF token 并将 LF token 聚合为 DC token，在缓解 rank collapse 的同时减少 ViT 的计算量，在 30% token 减少率下多个模型上超越现有 SOTA。

**[From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers](from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in.md)**

:   通过严格的理论分析证明了预训练数据的多样性（由"max-sum ratio"刻画）决定了单层Transformer学到的是可泛化的induction head还是无法OOD泛化的位置捷径，并给出了使模型学会induction head的最优预训练分布。

**[Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)**

:   提出层次化稀疏注意力（HSA）及 RAMba 架构，通过两阶段 token-to-chunk 相关性学习与硬件对齐 kernel 设计，让 Mamba 获得高效长程随机访问能力，仅在 4K 上下文预训练即可在 64M passkey retrieval 上达到 100% 准确率。

**[Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM](hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)**

:   提出层次均衡打包（HBP）方法，通过多级打包分组、均衡批处理、自适应序列并行和稳定损失归一化，解决长短上下文混合 SFT 中的注意力计算不均衡和通信浪费问题，在 DeepSeek-V2 (236B) 上实现 2.4× 训练加速且性能无损。

**[L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod.md)**

:   L-MTP 在多token预测（MTP）基础上引入跳跃机制，预测非相邻位置的token（如位置1,3,5,7而非1,2,3,4），通过"后向查找"解码策略复用先前预测填补空隙，在3B-12B模型上实现22%推理加速的同时保持或提升任务性能。

**[Learning in Compact Spaces with Approximately Normalized Transformer](learning_in_compact_spaces_with_approximately_normalized_transformer.md)**

:   提出 anGPT（近似归一化 Transformer），利用高维空间中向量范数的集中现象，用简单标量乘法替代逐层精确归一化，在消除权重衰减和学习率预热的同时实现了相比 GPT+（含 QK-norm）40% 的收敛加速，仅增加 3% 运行时开销。

**[Linear Attention for Efficient Bidirectional Sequence Modeling](linear_attention_for_efficient_bidirectional_sequence_modeling.md)**

:   提出 Lion 框架，首次系统地将线性 Transformer 扩展到双向序列建模，统一了全线性注意力、双向 RNN 和分块并行三种等价表示，在图像分类和 MLM 任务上训练速度比 SSM 快达 10 倍且性能可比 softmax Transformer。

**[Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)**

:   提出动态分层稀疏注意力 (DHSA)，通过自适应 chunk 分割 + chunk 级相似度预测 + 上采样到 token 级的分层框架，在不重训基座模型的前提下将密集注意力替换为稀疏注意力，在 Gemma2/3 上实现与密集注意力同等精度、20-60% prefill 延迟降低和 35% 峰值内存节省。

**[LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?](loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges.md)**

:   构建覆盖法律/金融/游戏/代码四大真实领域、长度16K-2M token的长依赖推理基准LooGLE v2，设计10类领域特定任务共1,934个QA实例，评估10个LLM发现最强模型GPT-4.1仅59.2%，揭示当前LLM在真实长依赖场景下的根本不足。

**[Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures](mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite.md)**

:   提出 Mozart 算法-硬件协同设计框架，通过专家聚类分配、细粒度流式调度和 3.5D 晶粒架构（NoP-Tree + 分层存储），在三个 MoE-LLM 上实现 1.9× 以上的训练加速。

**[Not All Splits Are Equal: Rethinking Attribute Generalization Across Unrelated Categories](not_all_splits_are_equal_rethinking_attribute_generalization_across_unrelated_ca.md)**

:   本文首次系统评估了属性预测任务中训练/测试划分策略对泛化性能的影响,提出了基于 LLM 语义分组、嵌入相似度、嵌入聚类和超类标签的四种渐进式难度划分方案,发现无监督聚类划分在不依赖标注的情况下实现了与真值超类划分相当的去泄漏效果,同时保留了更好的预测性能。

**[Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure](obliviator_reveals_the_cost_of_nonlinear_guardedness_in_concept_erasure.md)**

:   提出Obliviator——一种基于RKHS中HSIC最小化的后处理概念擦除方法，通过两步迭代优化逐步变形特征空间，首次实现对非线性对抗者的完全防护，同时量化了非线性防护的效用-擦除代价（utility-erasure trade-off），在多个PLM和数据集上显著优于现有方法。

**[OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)**

:   提出 OmniDraft 框架，通过在线 n-gram 缓存实现跨词表推测解码、混合蒸馏损失在线对齐草稿模型与目标模型、并结合自适应起草长度控制，使单个轻量 Llama-68M 模型可为 Vicuna-7B、Qwen2-7B、Llama3-8B 等不同目标模型提供推测解码加速（1.5-2x）。

**[On the Entropy Calibration of Language Models](on_the_entropy_calibration_of_language_models.md)**

:   系统研究语言模型的熵校准问题（生成文本的熵是否匹配在人类文本上的 log loss），发现由于数据分布的幂律特性（$\alpha \approx 1$），误差积累随模型规模的改善极为缓慢（scaling exponent $\approx -0.05$），并从理论上证明了在多项式时间内可以在不牺牲多样性的前提下校准熵。

**[On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)**

:   首次系统分析 MoE 在结构化复杂任务上的表达能力：证明浅层 MoE 可在低维流形上克服维度诅咒（近似速率由内在维度 $d$ 而非环境维度 $D$ 决定），深层 MoE 通过 $E$ 专家 × $L$ 层的分层组合可高效近似有 $E^L$ 段的分段函数，远超朴素上界 $LE$。

**[One Prompt Fits All: Universal Graph Adaptation for Pretrained Models](one_prompt_fits_all_universal_graph_adaptation_for_pretrained_models.md)**

:   理论证明表示级图提示（representation-level prompt）本质等价于线性探针，据此提出 UniPrompt——基于可学习 kNN 拓扑提示图的输入级方法，通过 bootstrapping 策略融合提示图和原图，在同域和跨域 few-shot 节点分类中一致超越现有图提示学习方法。

**[Plasticity as the Mirror of Empowerment](plasticity_as_the_mirror_of_empowerment.md)**

:   本文提出**广义有向信息（GDI）**作为度量智能体可塑性（plasticity）的信息论工具，揭示可塑性是赋权（empowerment）的"镜像"——两者使用相同度量、仅方向相反，并证明了两者之间存在严格的张力约束（tension bound）。

**[Silent Tokens, Loud Effects: Padding in LLMs](silent_tokens_loud_effects_padding_in_llms.md)**

:   系统性研究了padding token在未被正确掩码时对LLM的影响，发现即使少量padding也会漂移隐层表示、降低生成质量、不可预测地改变偏见，而128个padding token可将Llama-3.1-8B的有害提示攻击成功率从8%飙升到77.5%，本质上实现了jailbreak。

**[SkyLadder: Better and Faster Pretraining via Context Window Scheduling](skyladder_better_and_faster_pretraining_via_context_window_scheduling.md)**

:   通过上下文窗口短到长的渐进式调度策略 SkyLadder，在固定计算量下实现更优的预训练效率（节省 22% 训练时间）和更好的模型性能（+3.7%），反驳了"长上下文=好性能"的业界信念。

**[SPARTA Alignment: Collectively Aligning Multiple Language Models through Combat](sparta_alignment_collectively_aligning_multiple_language_models_through_combat.md)**

:   让多个LLM组成"斯巴达部落"互相竞技和互评，通过声誉加权的判断聚合生成偏好对，再用DPO迭代训练所有模型，在12个任务中的10个上超越Self-Rewarding等自对齐基线，平均提升7%。

**[Structure-Aware Spectral Sparsification via Uniform Edge Sampling](structure-aware_spectral_sparsification_via_uniform_edge_sampling.md)**

:   本文证明在具有良好聚类结构的图上（结构比 Υ(k) 足够大），**均匀边采样**即可保留谱聚类所需的谱子空间结构，无需昂贵的有效电阻预计算——这是首个关于均匀采样保持结构的可证明保证。

**[Technical Debt in In-Context Learning: Diminishing Efficiency in Long Context](technical_debt_in_in-context_learning_diminishing_efficiency_in_long_context.md)**

:   借鉴优化软件基准方法论，用性能比率精确量化ICL相对贝叶斯最优估计器的样本效率，发现存在"二分法"——少射下(≤15个演示)效率接近最优(仅多10%)而多射下(>40个演示)急剧恶化(多45%)，信息论分析证明这源于不可消除的非递减过剩风险，是ICL机制的内在限制。

**[Tensor Product Attention Is All You Need](tensor_product_attention_is_all_you_need.md)**

:   通过上下文张量分解将 Q/K/V 表示为低秩因子的加权和，将 KV 缓存压缩至原来的 1/10~1/16，同时在验证损失和下游任务精度上超越标准 MHA/MQA/GQA/MLA。

**[The Emergence of Sparse Attention: Impact of Data Distribution and Benefits of Repetition](the_emergence_of_sparse_attention_impact_of_data_distribution_and_benefits_of_re.md)**

:   通过理论分析和受控实验研究 sparse attention 的涌现机制，揭示涌现时间遵循关于序列长度和维度的幂律关系 $T_\epsilon \propto \sqrt{d} \cdot T$，并发现 in-context 和 cross-sample 两种数据重复策略都能加速涌现，为理解 LLM 能力涌现提供了统一的 sparse attention 视角。

**[The PokeAgent Challenge: Competitive and Long-Context Learning at Scale](the_pokeagent_challenge_competitive_and_long-context_learning_at_scale.md)**

:   提出 PokéAgent Challenge，一个基于宝可梦对战和RPG速通的双赛道大规模AI基准，通过NeurIPS 2025竞赛验证了专家RL方法远超通用LLM方法，并揭示宝可梦对战衡量的能力与现有49个LLM基准近乎正交。

**[Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels](tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels.md)**

:   提出 TFLA（Tiled Flash Linear Attention）算法，通过二层序列并行化和 tiling 优化，实现高效的线性 RNN/mLSTM 内核，相比 FlashAttention 3 和 Mamba 2 获得显著墙钟加速（训练 >2x vs Mamba 2），同时保持等价的模型精度。

**[UMoE: Unifying Attention and FFN with Shared Experts](umoe_unifying_attention_and_ffn_with_shared_experts.md)**

:   通过重新表述多头注意力机制，揭示其与 FFN 共有的"两层矩阵乘法"结构，据此提出 UMoE 统一架构——在注意力和 FFN 层使用相同设计的专家并支持参数共享，在 Base(134M) 和 Large(1.1B) 模型上均优于现有 FFN-MoE 和 Attention-MoE 基线。

**[Unmasking COVID-19 Vulnerability in Nigeria: Mapping Risks Beyond Urban Hotspots](unmasking_covid-19_vulnerability_in_nigeria_mapping_risks_beyond_urban_hotspots.md)**

:   本文针对尼日利亚各州构建了一个综合 COVID-19 脆弱性风险评分体系,整合人口密度、贫困、医疗可及性和年龄风险四个维度,并通过 GIS 地图可视化热点区域,为公共卫生资源分配提供数据驱动的决策工具。

**[Vocabulary Customization for Efficient Domain-Specific LLM Deployment](vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)**

:   提出一种保证编码效率单调不降的BPE tokenizer扩展算法，将领域高频token追加到Llama 3.1词表中（+30K token），在电商场景实现输入序列缩短20%、推理吞吐量提升20-30%，经10K步继续训练后模型质量不降，且约98%情况下模型主动生成新token。

**[Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding](yggdrasil_bridging_dynamic_speculation_and_static_runtime_for_latency-optimal_tr.md)**

:   提出 Yggdrasil，一个延迟最优的推测解码系统，通过 Equal-Growth Tree (EGT) 结构实现编译友好的动态草稿、延迟感知优化目标替代传统 AAL 指标、以及阶段调度运行时减少 CPU-GPU 协调开销，在 A100/A40 上实现了最高 3.98× 的端到端加速。

**[ZeroS: Zero-Sum Linear Attention for Efficient Transformers](zeros_zero-sum_linear_attention_for_efficient_transformers.md)**

:   通过移除 softmax 的零阶均匀项 $1/t$，构建零和权重的线性注意力机制 ZeroS，突破凸组合只能做加法混合的限制，支持单层内的差分/对比操作，在保持 $O(Nd^2)$ 线性复杂度的同时，在多个序列建模基准上匹配甚至超越标准 softmax 注意力。
