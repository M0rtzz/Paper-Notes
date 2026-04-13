---
title: >-
  NeurIPS2025 模型压缩方向 140篇论文解读
description: >-
  140篇NeurIPS2025 模型压缩方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**🧠 NeurIPS2025** · 共 **140** 篇

**[3Did Direct 3D Inverse Design For Aerodynamics With Physics-Aware Optimization](3did_direct_3d_inverse_design_for_aerodynamics_with_physics-aware_optimization.md)**

:   提出 3DID 框架，通过学习物理-几何统一的三平面隐空间表示 + 目标梯度引导扩散采样 + 拓扑保持精炼的两阶段策略，从随机噪声开始直接在完整 3D 空间中进行逆向设计，在车辆气动外形优化上，模拟阻力（Sim-Drag）相比最优基线降低 13.6%。

**[4Dgcpro Efficient Hierarchical 4D Gaussian Compression For P](4dgcpro_efficient_hierarchical_4d_gaussian_compression_for_p.md)**

:   提出层级化的4D高斯压缩框架4DGCPro，通过感知加权的层级高斯表示、运动感知自适应分组和端到端熵优化训练，在单一模型内实现多码率渐进式体积视频流媒体，可在移动设备上实时解码和渲染，RD性能超越现有SOTA。

**[A-Thought Efficient Reasoning Via Bidirectional Compression For Low-Resource Set](a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)**

:   提出 A*-Thought——基于 A* 搜索算法的 CoT 压缩框架，通过双向重要性评分（BIS）衡量每个推理步骤对问题和答案的相关性，结合路径级 A* 搜索在指数级搜索空间中高效找到最紧凑的推理路径，在 512 token 预算下将 QwQ-32B 准确率提升 2.39 倍，在 4096 token 预算下减少约 50% 输出 token 且几乎不损失准确率。

**[A Granular Study Of Safety Pretraining Under Model Abliteration](a_granular_study_of_safety_pretraining_under_model_abliteration.md)**

:   本文系统地研究了 model abliteration（一种推理时激活空间编辑攻击）对不同数据驱动安全预训练阶段的影响，发现仅依赖 refusal 训练的安全机制极易被攻破，而 **组合多种安全信号**（safe-only 过滤 + 改写 + metatag + refusal）可使安全行为分散到更广泛的表征空间、从而更难被单一方向投影移除。

**[A Partition Cover Approach To Tokenization](a_partition_cover_approach_to_tokenization.md)**

:   将分词（tokenization）问题重新建模为**分区覆盖（partition cover）**优化问题，证明其为NP-hard，并提出多项式时间的贪心算法GreedTok，在压缩率和1B参数LLM预训练下游任务上均优于BPE。

**[A Token Is Worth Over 1000 Tokens Efficient Knowledge Distillation Through Low-R](a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)**

:   提出 Low-Rank Clone (LRC)，通过可学习低秩投影矩阵将 teacher 权重压缩为 student 权重（软剪枝），同时对齐 attention 和 FFN 的中间激活（激活克隆），仅用 20B tokens 训练的 1.7B 模型即超过用 36T tokens 训练的 Qwen3-1.7B（64.98 vs 63.17），实现 **1000 倍训练效率提升**。

**[Accurate And Efficient Low-Rank Model Merging In Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)**

:   提出 Core Space Merging 框架——通过在低秩 LoRA 矩阵的公共参考基空间中进行模型合并，**无信息损失**地将合并操作从 $m \times n$ 全尺寸空间压缩到 $Tr \times Tr$ 紧凑空间（$T$ 为任务数，$r$ 为 LoRA 秩），在 Llama 3 8B 上达到 SOTA 合并精度同时计算成本降低数个数量级。

**[Adaptive Kernel Design For Bayesian Optimization Is A Piece Of Cake With Llms](adaptive_kernel_design_for_bayesian_optimization_is_a_piece_of_cake_with_llms.md)**

:   提出 CAKE (Context-Aware Kernel Evolution)，利用 LLM 作为遗传算法的交叉和变异算子，在贝叶斯优化过程中自适应地生成和进化 GP 核函数表达式，结合 BAKER 排序机制平衡模型拟合（BIC）与期望改进（EI），在超参数优化、控制器调参和光子芯片设计等任务上持续超越固定核和自适应核基线。

**[Adaptive Predictionpowered Autoeval With Reliability And Eff](adaptive_predictionpowered_autoeval_with_reliability_and_eff.md)**

:   提出R-AutoEval+，通过e-value赌注算法自适应调整对合成数据（LLM评判器）的依赖权重，首次同时提供有限样本可靠性保证和可证明的采样效率改善，在GSM8K上比纯真实数据方法节省87个token。

**[Adaptive Stochastic Coefficients For Accelerating Diffusion Sampling](adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)**

:   通过理论分析 ODE 和 SDE 求解器的互补弱点（ODE 积累不可消除的梯度误差，SDE 在少步时离散化误差放大），提出 AdaSDE——在每个去噪步引入可学习随机系数 $\gamma_i$ 控制噪声注入强度，通过轻量蒸馏优化，在 5 NFE 下实现 CIFAR-10 FID 4.18、FFHQ FID 8.05 的 SOTA。

**[Admtree Compressing Lengthy Context With Adaptive Semantic Trees](admtree_compressing_lengthy_context_with_adaptive_semantic_trees.md)**

:   提出 AdmTree——一种自适应层次化上下文压缩框架,通过信息密度驱动的动态分段构建叶 gist token，再用二叉语义树底向上聚合实现多粒度语义保留，解决了显式方法丢失局部细节和隐式方法位置偏差的双重问题,在 LongBench 上比 SOTA 基线 Activation Beacon 高 10%+。

**[Ai-Generated Video Detection Via Perceptual Straightening](ai-generated_video_detection_via_perceptual_straightening.md)**

:   提出 ReStraV 方法，基于"感知拉直"假说（真实视频在神经表示空间形成更直的轨迹），利用 DINOv2 特征空间中的时间曲率和步距统计量训练轻量分类器检测 AI 生成视频，在 VidProM 上达到 97.17% 准确率和 98.63% AUROC，推理仅需 ~48ms。

**[Atlas Autoformalizing Theorems Through Lifting Augmentation And Synthesis Of Dat](atlas_autoformalizing_theorems_through_lifting_augmentation_and_synthesis_of_dat.md)**

:   ATLAS 提出了一个基于概念仓库、专家迭代+知识蒸馏、以及两种新颖增强策略的数据生成框架，构建了117K定理陈述的平行语料库，微调 Llama3.1-8B-Instruct 后在所有自动形式化基准上达到 SOTA。

**[Autodiscovery Open-Ended Scientific Discovery Via Bayesian Surprise](autodiscovery_open-ended_scientific_discovery_via_bayesian_surprise.md)**

:   AutoDiscovery 提出用贝叶斯惊奇度（Bayesian Surprise）作为开放式科学发现的客观奖励信号——通过 LLM 采样估计先验/后验信念分布的 KL 散度，配合 MCTS+渐进展宽在假设空间中探索，在 21 个真实数据集上比贪心/束搜索产生 5-29% 更多的惊奇发现，人类评估确认贝叶斯惊奇度与专家"惊讶感"的一致性（0.67）远超 LLM 自身评估的"新颖性"和"有用性"。

**[Autojudge Judge Decoding Without Manual Annotation](autojudge_judge_decoding_without_manual_annotation.md)**

:   AutoJudge 自动化了 Judge Decoding 中"重要 token"的标注——通过半贪心搜索替换不匹配 token 并检查答案是否改变来标注重要性，训练逻辑回归分类器预测 token 重要性，使投机解码每轮接受 40+ token（vs 标准 ~20），在 GSM8K 上加速 1.5× 且准确率损失 <1%。

**[Barista Brain Scale Informed Spatiotemporal Representation Of Human Intracranial](barista_brain_scale_informed_spatiotemporal_representation_of_human_intracranial.md)**

:   BaRISTA 系统探索 iEEG Transformer 的空间编码尺度（电极/脑区/脑叶），发现脑区级编码 + 空间掩码重建在语言任务解码上达 86.2% AUC（vs PopT 79.5%），编码尺度选择的影响 > 掩码策略选择，且跨被试泛化性好。

**[Benfords Curse Tracing Digit Bias To Numerical Hallucination In Llms](benfords_curse_tracing_digit_bias_to_numerical_hallucination_in_llms.md)**

:   本文发现 LLM 的数值幻觉根源于预训练语料中符合 Benford 定律的数字频率分布——数字 1 出现概率 ~30% 而数字 9 仅 ~5%，这种偏差被 FFN 后期层的特定"数字选择性神经元"内化，提出数字选择性分数（DSC）定位偏差神经元并通过剪枝 0.01% 的神经元修正 1.36-3.49% 的错误预测。

**[Beyond Higher Rank Token-Wise Input-Output Projections For Efficient Low-Rank Ad](beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)**

:   TopLoRA 从输入-输出投影角度分析 LoRA 的表达能力，发现所有 token 共享同一投影矩阵是关键瓶颈，提出通过可学习的 token 级对角矩阵 $\Sigma_X$ 动态调整 LoRA 权重（$\Delta W_X = B\Sigma_X A$），在不增加秩的前提下实现细粒度适配，跨任务一致优于 LoRA 2-3%。

**[Beyond Random Automatic Inner-Loop Optimization In Dataset Distillation](beyond_random_automatic_inner-loop_optimization_in_dataset_distillation.md)**

:   提出 AT-BPTT（自适应截断 BPTT），将 DNN 训练分为早/中/晚三阶段并自适应调整截断策略和窗口大小，在 CIFAR-10/100/Tiny-ImageNet/ImageNet-1K 上平均提升 3-17%，同时实现 3.9× 加速和 63% 内存节省。

**[Bezier Splatting For Fast And Differentiable Vector Graphics Rendering](bezier_splatting_for_fast_and_differentiable_vector_graphics_rendering.md)**

:   Bézier Splatting 将 Gaussian Splatting 框架与 Bézier 曲线结合，沿曲线均匀采样 2D Gaussian 点，通过 α-blending 渲染实现可微矢量图形，前向 30× / 反向 150× 加速（相比 DiffVG），同时保持或超越 LIVE 等方法的图像质量。

**[Binary Quadratic Quantization Beyond First-Order Quantization For Real-Valued Ma](binary_quadratic_quantization_beyond_first-order_quantization_for_real-valued_ma.md)**

:   BQQ 提出二次二值量化——用二值矩阵的乘积（而非线性组合）表示权重矩阵，突破传统一阶量化的表达能力限制，通过扩展 AMFD（退火均场下降）到 PUBO 问题求解混合整数优化，在 2-bit 无数据 ViT 量化上实现从 10.83% 到 58.25% 的准确率飞跃。

**[Biobench A Blueprint To Move Beyond Imagenet For Scientific Ml Benchmarks](biobench_a_blueprint_to_move_beyond_imagenet_for_scientific_ml_benchmarks.md)**

:   提出 BioBench——一个统一 9 个生态视觉任务、4 个分类界、6 种图像模态、310 万张图像的基准，证明 ImageNet top-1 准确率仅解释 34% 的生态任务方差，在 >75% 精度的前沿模型中 30% 的排名是错误的。

**[C-Lora Contextual Low-Rank Adaptation For Uncertainty Estimation In Large Langua](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)**

:   提出 C-LoRA，通过引入轻量级上下文模块使 LoRA 低秩矩阵的分布依赖于输入数据，实现样本级的异方差不确定性估计，在少样本微调场景中显著改善校准质量。

**[Casspec Cascade Adaptive Selfspeculative Decoding For Onthef](casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)**

:   CAS-Spec 通过 Dynamically Switchable Inference Acceleration (DSIA) 策略（如不同程度的 layer sparsity）从目标模型自身构建多级 draft 模型层级，配合 Dynamic Tree Cascade (DyTC) 算法基于在线 acceptance rate 和延迟预测自适应路由 draft 模型和分配 draft 长度，在完全 training-free 的条件下实现 1.1×-2.3× 的无损推理加速，DyTC 比 cascade 和 tree baseline 分别提升 47% 和 48%。

**[Chunkkv Semanticpreserving Kv Cache Compression For Efficien](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)**

:   ChunkKV 将 KV cache 压缩的基本单元从离散 token 提升为语义 chunk（连续 token 组），通过 chunk 级 attention score 聚合来选择保留哪些语义完整的片段，并利用 chunk 带来的高跨层索引相似性实现 layer-wise index reuse，在 10% 压缩率下比 SnapKV/PyramidKV 提升最高 8.7%，吞吐量提升 26.5%。

**[Codegemm A Codebook-Centric Approach To Efficient Gemm In Quantized Llms](codegemm_a_codebook-centric_approach_to_efficient_gemm_in_quantized_llms.md)**

:   提出 CodeGEMM，一种以 codebook 为中心的 GEMM kernel，通过预计算 centroid 与 activation 的内积并缓存为 Psumbook，替代传统反量化流程，在 2-bit 量化 LLM 上实现 1.83×（8B）到 8.93×（70B）的端到端加速。

**[Correlation Dimension Of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)**

:   引入源于分形几何的**相关维数（correlation dimension）**作为衡量自回归语言模型感知文本复杂度的指标，揭示了传统 perplexity 无法捕捉的长程结构特性，可检测幻觉和退化文本。

**[Curiosity-Driven Rl For Symbolic Equation Solving](curiosity-driven_rl_for_symbolic_equation_solving.md)**

:   将好奇心驱动探索（RND、ICM 等）与基于表达式树的图动作空间结合，使 PPO 智能体能够求解包含根号、指数和三角函数的非线性方程，超越了此前仅限于线性方程的 RL 方法。

**[Data Efficient Adaptation In Large Language Models Via Continuous Low-Rank Fine-](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)**

:   提出 DEAL 框架，通过小波核特征过滤保留 LoRA 低秩矩阵中的历史知识核心特征，结合受控知识更新模块和非对称正则化，实现 LLM 在小样本持续微调中学新不忘旧。

**[Deltaflow An Efficient Multi-Frame Scene Flow Estimation Method](deltaflow_an_efficient_multi-frame_scene_flow_estimation_method.md)**

:   提出 DeltaFlow (ΔFlow)，通过体素帧间差分（Δ scheme）提取运动线索，实现特征尺寸不随帧数增长的多帧场景流估计，在 Argoverse 2/Waymo/nuScenes 上达到 SOTA 且比次优多帧方法快 2 倍。

**[Dense Backpropagation Improves Training For Sparse Mixture-Of-Experts](dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)**

:   提出 Default MoE 方法，用指数移动平均（EMA）近似非激活 expert 的输出，使 MoE router 获得稠密梯度更新，在不显著增加计算开销的情况下提升稀疏 MoE 的训练性能。

**[Dependency Parsing Is More Parameter-Efficient With Normalization](dependency_parsing_is_more_parameter-efficient_with_normalization.md)**

:   揭示依存句法/语义分析中 biaffine scoring 缺乏归一化导致模型过参数化，通过简单的 $1/\sqrt{d}$ 缩放即可在减少高达 85% BiLSTM 参数的同时匹配甚至超越原始性能。

**[Deterministic Continuous Replacement Fast And Stable Module Replacement In Pretr](deterministic_continuous_replacement_fast_and_stable_module_replacement_in_pretr.md)**

:   DCR 通过确定性退火权重 α(t) 混合 teacher 和 student 模块输出，消除了随机门控（如 BERT-of-Theseus）带来的梯度方差，在冷启动模块替换场景下实现更快收敛和更强的特征对齐。

**[Disentangling Latent Shifts Of In-Context Learning With Weak Supervision](disentangling_latent_shifts_of_in-context_learning_with_weak_supervision.md)**

:   WILDA 将 ICL 视为弱监督信号，用 teacher-student 框架将示例引发的潜在偏移编码进轻量 LoRA 适配器，实现无需重复 prompting 的高效推理，且 student 通过伪标签修正和覆盖扩展超越 teacher（弱到强泛化）。

**[Dismo Disentangled Motion Representations For Openworld Moti](dismo_disentangled_motion_representations_for_openworld_moti.md)**

:   DisMo 通过双流架构（运动提取器 + 帧生成器）和图像空间重建目标，从原始视频中学习与外观、姿态、类别无关的抽象运动表征，实现跨类别/跨视角的开放世界运动迁移，并在零样本动作分类上大幅超越 V-JEPA 等视频表征模型。

**[Dp-Llm Runtime Model Adaptation With Dynamic Layer-Wise Precision Assignment](dp-llm_runtime_model_adaptation_with_dynamic_layer-wise_precision_assignment.md)**

:   DP-LLM 发现每层的量化敏感度在解码步间动态变化，提出基于 relative error 的动态逐层精度选择机制，在运行时根据输入为每层分配精度（h-bit 或 l-bit），实现了优于静态混合精度的性能-延迟权衡。

**[Dragon Guard Llm Unlearning In Context Via Negative Detection And Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)**

:   DRAGON 提出无需微调基座模型的系统性 LLM 遗忘框架：通过双层检测模块识别需遗忘的 prompt，再由专门微调的 guard 模型生成 CoT 推理指令实现上下文干预，在保持模型通用能力的同时有效删除隐私/有害知识。

**[Duogpt Training-Free Dual Sparsity Through Activation-Aware Pruning In Llms](duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)**

:   提出 DuoGPT，一种将激活稀疏（activation sparsity）重新解释为动态结构化权重稀疏、并与非结构化权重剪枝相结合的双稀疏（dual-sparse）框架，通过扩展 OBC 框架引入激活感知校准和稠密模型输出残差修正项，在不需要重训练的情况下实现 LLM 解码阶段的显著加速与内存节省。

**[Elastic Vits From Pretrained Models Without Retraining](elastic_vits_from_pretrained_models_without_retraining.md)**

:   SnapViT 提出一种后训练结构化剪枝方法：结合自监督梯度的局部 Hessian 和进化算法估计的全局跨模块相关性，无需重训练或标签即可在一次运行中生成连续稀疏度的弹性 ViT 子网络，在 A100 上仅需不到 5 分钟。

**[Emloc Emulator-Based Memory-Efficient Fine-Tuning With Lora Correction](emloc_emulator-based_memory-efficient_fine-tuning_with_lora_correction.md)**

:   EMLoC 通过对原始模型做 activation-aware SVD 构建轻量级 emulator 进行 LoRA 微调，并提出 LoRA 校正算法弥补 emulator 与原模型的不对齐，使得微调内存开销降至与推理持平，在单张 24GB GPU 上即可微调 38B 模型。

**[Enhancing Semi-Supervised Learning With Zero-Shot Pseudolabels](enhancing_semi-supervised_learning_with_zero-shot_pseudolabels.md)**

:   ZeroMatch 提出两阶段框架将基础模型的零样本伪标签与半监督学习相结合：先用知识蒸馏初始化学生模型，再以辅助 KD loss 防止灾难性遗忘的方式执行 SSL 训练，在 6 个视觉/NLP 基准上一致超越标准 SSL 和零样本增强方法。

**[Exact Expressive Power Of Transformers With Padding](exact_expressive_power_of_transformers_with_padding.md)**

:   本文精确刻画了带 padding 的 Transformer 的表达能力：固定深度 + 多项式 padding 恰好等于 $\mathsf{FO}$-uniform $\mathsf{TC}^0$，进一步结合 $O(\log^d n)$ looping 恰好等于 $\mathsf{FO}$-uniform $\mathsf{TC}^d$，polylog looping 收敛到 $\mathsf{NC}$，为 padding/looping 作为可并行推理时计算提供了完整理论基础。

**[Expo Unlocking Hard Reasoning With Self-Explanation-Guided Reinforcement Learnin](expo_unlocking_hard_reasoning_with_self-explanation-guided_reinforcement_learnin.md)**

:   提出 Self-Explanation Policy Optimization (ExPO)，一种通过让模型在给定正确答案条件下自主生成推理链（self-explanation）作为正样本的模块化框架，解决 GRPO 等 RL 后训练方法在困难推理任务上因缺乏有效正样本而无法学习（分布锐化）的根本问题——ExPO 生成的自解释样本既在当前策略分布内（in-distribution），又能提供正向学习信号，可无缝集成到 DPO 和 GRPO 中。

**[Eyes Wide Open Ego Proactive Videollm For Streaming Video](eyes_wide_open_ego_proactive_videollm_for_streaming_video.md)**

:   定义"第一视角流式视频主动理解"新任务——给定ego-streaming视频，AI助手在恰当时机主动回答多样化、随事件演变的问题，同时保持感知与推理的同步。提出ESTP-Bench评估框架、ESTP-F1指标，以及含数据引擎、多阶段训练和主动动态压缩的完整技术pipeline（VideoLLM-EyeWO），在ESTP-Bench上比最强baseline MiniCPM-V高11.8%。

**[Falqon Accelerating Lora Fine-Tuning With Low-Bit Floating-Point Arithmetic](falqon_accelerating_lora_fine-tuning_with_low-bit_floating-point_arithmetic.md)**

:   FALQON 通过将 LoRA 适配器直接融合 (meld) 到 FP8 量化的骨干权重中，消除了单独 LoRA 路径引入的小矩阵量化开销，结合高效梯度计算和行级代理更新机制，实现了相比现有量化 LoRA 方法约 3 倍的训练加速。

**[Fastlongspeech Enhancing Large Speech-Language Models For Efficient Long-Speech ](fastlongspeech_enhancing_large_speech-language_models_for_efficient_long-speech_.md)**

:   提出 FastLongSpeech，通过迭代融合策略压缩冗余语音表征和动态压缩训练转移短语音能力到长语音场景，使 LSLM 无需长语音训练数据即可高效处理长语音，在长语音 QA 上实现最优性能且推理效率提升 70%。

**[Find Your Needle Small Object Image Retrieval Via Multi-Object Attention Optimiz](find_your_needle_small_object_image_retrieval_via_multi-object_attention_optimiz.md)**

:   MaO 提出了一种针对小目标图像检索（SoIR）的新方法，通过多目标预训练和基于注意力的特征优化，将多个目标的表示融合为单一全局描述符，在多个基准上大幅超越现有检索方法。

**[Fira Can We Achieve Full-Rank Training Of Llms Under Low-Rank Constraint](fira_can_we_achieve_full-rank_training_of_llms_under_low-rank_constraint.md)**

:   提出 Fira，首个在低秩约束下实现全秩训练（全秩梯度+全秩权重）的 LLM 训练框架，通过观察到低秩与全秩训练中优化器的缩放因子高度相似，用低秩缩放因子近似校正子空间外梯度，配合 norm-growth limiter 防止 loss spike，在预训练和微调中均超越 LoRA 和 GaLore。

**[Gated Integration Of Low-Rank Adaptation For Continual Learning Of Large Languag](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)**

:   提出 GainLoRA，为持续学习中每个新任务的 LoRA 分支引入**门控模块**生成自适应集成系数，通过正交约束使新分支对旧任务的输出趋近于零，从而有效缓解灾难性遗忘。

**[Geometric Data Valuation Via Leverage Scores](geometric_data_valuation_via_leverage_scores.md)**

:   提出基于**统计杠杆分数（leverage scores）**的几何数据估值方法，作为 Data Shapley 值的高效代理，满足对称性、效率性和虚拟玩家等公理，并通过 ridge leverage 扩展解决维度饱和问题，提供 $O(\varepsilon)$ 近似最优的理论保证。

**[Geometry Of Decision Making In Language Models](geometry_of_decision_making_in_language_models.md)**

:   通过在 28 个开源 Transformer 模型上大规模测量各层隐藏表示的**内在维度（Intrinsic Dimension, ID）**，揭示了一致的"低-高-低"维度变化模式：早期层在低维流形上操作，中间层扩展空间，后期层再压缩至与决策相关的低维表示。

**[Global Minimizers Of Ellp-Regularized Objectives Yield The Sparsest Relu Neural ](global_minimizers_of_ellp-regularized_objectives_yield_the_sparsest_relu_neural_.md)**

:   证明了对于单隐层 ReLU 网络，最小化 $\ell^p$（$0 < p < 1$）路径范数的全局最优解恰好对应于**最稀疏**的数据插值网络，从而将组合优化的稀疏插值问题重新表述为连续可微的优化任务。

**[Gora Gradient-Driven Adaptive Low Rank Adaptation](gora_gradient-driven_adaptive_low_rank_adaptation.md)**

:   提出 GoRA，利用**预计算梯度信息**在训练前同时完成自适应秩分配和权重初始化——基于参数敏感度分配各层 rank，用梯度伪逆初始化 $B$ 矩阵使初始输出近似一步梯度下降，统一解决 LoRA 的两大瓶颈。

**[Graph Your Own Prompt](graph_your_own_prompt.md)**

:   提出图一致性正则化（GCR）框架，通过在网络任意深度插入无参数的图一致性层（GCL），将中间特征的关系图与基于预测的类感知语义图对齐，以自我提示的方式促进语义一致的特征学习，在不修改架构和不增加参数的前提下提升分类泛化性能。

**[Grass Scalable Data Attribution With Gradient Sparsification And Sparse Projecti](grass_scalable_data_attribution_with_gradient_sparsification_and_sparse_projecti.md)**

:   提出 GraSS 与 FactGraSS 两阶段梯度压缩算法，利用逐样本梯度的固有稀疏性实现**亚线性**时间与空间复杂度（$O(k')$），在十亿参数模型上比 SOTA 基线 LoGra 快 **165%**，同时保持数据归因质量。

**[Graver Generative Graph Vocabularies For Robust Graph Foundation Models Fine-Tun](graver_generative_graph_vocabularies_for_robust_graph_foundation_models_fine-tun.md)**

:   提出 Graver 框架，通过 ego-graph 解耦提取可迁移子图词汇、graphon 专家建模词汇分布、MoE-CoE 路由选择性增强 support 样本，解决 GFM 少样本微调中因结构不匹配导致的不稳定性问题。

**[Hankel Singular Value Regularization For Highly Compressible State Space Models](hankel_singular_value_regularization_for_highly_compressible_state_space_models.md)**

:   通过在训练中正则化 SSM 层的 **Hankel 奇异值核范数**促使其快速衰减，使训练后模型可用平衡截断压缩至原始阶数的 **10%** 而保持精度，并利用旋转矩阵块对角参数化将 Gramian 计算从 $\mathcal{O}(n^3)$ 降至 $\mathcal{O}(n^2)$。

**[Heterogeneous Adversarial Play In Interactive Environments](heterogeneous_adversarial_play_in_interactive_environments.md)**

:   提出 **HAP（Heterogeneous Adversarial Play）**，将教师-学生交互形式化为极小极大博弈：教师网络自动生成针对学生弱点的挑战任务，学生策略不断适应进化，形成无需手工设计的自适应课程——在多任务 RL 环境中超越 SOTA 基线，且生成的课程对人类学习者同样有效。

**[Homogeneous Keys Heterogeneous Values Exploiting Local Kv Cache Asymmetry For Lo](homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)**

:   发现 LLM 注意力机制中一个被忽视的**局部 Key-Value 不对称性**——相邻 Key 具有同质性（相似注意力权重），而相邻 Value 呈异质分布——据此提出 **AsymKV** 无训练压缩框架：基于同质性合并 Key + 基于基数归一化的**无损** Value 表示，在 LongBench 上超越 H2O 达 **5 分**。

**[Hyperbolic Dataset Distillation](hyperbolic_dataset_distillation.md)**

:   提出 HDD 方法，首次将双曲空间引入数据集蒸馏，通过在 Lorentz 双曲空间中匹配原始和合成数据的 Riemannian 质心来替代欧氏空间的分布匹配，利用双曲几何的层级加权特性让"更具代表性"的底层样本获得更高权重，在多个数据集上持续提升 DM/IDM 基线准确率。

**[Hyperbolic Fine-Tuning For Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)**

:   发现 LLM token 嵌入具有幂律分布和树状双曲结构，据此提出 HypLoRA——在 Lorentz 双曲流形上直接执行低秩适配（避免切空间映射的相消效应），在算术推理和常识推理任务上相比标准 LoRA 取得显著提升（如 Qwen2.5-7B 上 M.AVG +7.5%）。

**[Inference-Time Hyper-Scaling With Kv Cache Compression](inference-time_hyper-scaling_with_kv_cache_compression.md)**

:   提出"推理时超缩放"（Inference-Time Hyper-Scaling）范式：通过高效压缩KV缓存，在相同计算/内存预算下生成更长或更多并行推理序列，显著提升推理模型在数学、代码、科学推理等任务上的准确率。

**[Infrequent Exploration In Linear Bandits](infrequent_exploration_in_linear_bandits.md)**

:   提出 INFEX 框架，按给定调度表在探索步执行基线算法（如 LinUCB/LinTS）、其余时刻贪心选臂，证明只要探索次数超过 $\omega(\log T)$ 即可达到与全时刻探索相同的多项对数 regret，同时大幅降低计算开销（80%-99% 时间步为贪心）。

**[Jet-Nemotron Efficient Language Model With Post Neural Architecture Search](jet-nemotron_efficient_language_model_with_post_neural_architecture_search.md)**

:   NVIDIA 提出 PostNAS 流水线——从预训练全注意力模型出发，冻结 MLP 权重，通过四步搜索（全注意力层放置→线性注意力块选择→新注意力块 JetBlock 设计→硬件感知超参搜索）得到混合架构 Jet-Nemotron，2B 模型在 MMLU-Pro 上超越 Qwen3-1.7B 同时生成吞吐提升 47×。

**[Keydiff Key Similarity-Based Kv Cache Eviction For Long-Context Llm Inference In](keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)**

:   提出 KeyDiff——一种无需注意力分数的 KV cache 驱逐策略，通过保留与其他 key 余弦相似度最低（即几何上最独特）的 key 来维护 cache，在严格内存约束的逐块推理场景下以 8K cache 在 LongBench 上仅损失 ≤0.04% 精度，同时端到端推理延迟减少最高 30%。

**[Kindle Knowledge-Guided Distillation For Prior-Free Gene Regulatory Network Infe](kindle_knowledge-guided_distillation_for_prior-free_gene_regulatory_network_infe.md)**

:   提出 KINDLE 三阶段框架，通过知识蒸馏将先验引导的教师模型中学到的基因调控知识迁移到无先验的学生模型，在不依赖任何外部先验知识的情况下实现了基因调控网络（GRN）推断的 SOTA 性能。

**[Ktae A Model-Free Algorithm To Key-Tokens Advantage Estimation In Mathematical R](ktae_a_model-free_algorithm_to_key-tokens_advantage_estimation_in_mathematical_r.md)**

:   KTAE 提出了一种不依赖额外模型的 token 级优势估计算法，通过 Fisher 精确检验和信息增益量化每个 token 与正确推理结果的统计关联，将细粒度 token 重要性叠加到 GRPO/DAPO 的 rollout 级优势上，在5个数学推理基准上超越基线并显著缩短生成长度。

**[Kvzip Query-Agnostic Kv Cache Compression With Context Reconstruction](kvzip_query-agnostic_kv_cache_compression_with_context_reconstruction.md)**

:   提出 KVzip，一种查询无关的 KV Cache 驱逐方法，通过利用 LLM 自身从缓存的 KV 对中重建原始上下文来量化每个 KV 对的重要性，实现 3-4× 的 KV Cache 压缩率和约 2× 的 FlashAttention 解码延迟降低，同时在多查询场景下显著优于现有查询感知方法。

**[Layerif Estimating Layer Quality For Large Language Models Using Influence Funct](layerif_estimating_layer_quality_for_large_language_models_using_influence_funct.md)**

:   LayerIF 提出用影响函数（Influence Functions）逐层量化 LLM 的训练质量，通过聚合各层的正向影响分数得到数据驱动的层重要性估计，并将其应用于 LoRA-MoE 专家分配和层级稀疏剪枝两个下游任务，在 Mistral-7B 和 Gemma-7B 上分别获得 1.61% 和 0.90% 的准确率提升。

**[Learning Grouped Lattice Vector Quantizers For Low-Bit Llm Compression](learning_grouped_lattice_vector_quantizers_for_low-bit_llm_compression.md)**

:   GLVQ 提出为 LLM 权重的每个分组学习专属的格（lattice）码本（由可学习生成矩阵定义），配合分组特异的 μ-law companding 变换适应重尾分布，在 2-bit 量化下 Llama-2-70B 的 Wikitext-2 困惑度达到 3.36，大幅领先 QuIP#（3.91）和 QTIP（3.78）。

**[Learning To Better Search With Language Models Via Guided Reinforced Self-Traini](learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)**

:   提出 Guided-ReST，通过将最优解作为子目标逐步融入模型自生成的搜索轨迹中，生成高质量训练数据并蒸馏更高效的搜索策略，在Countdown和代码自修复任务上显著提升搜索效率和准确率。

**[Learning To Factorize And Adapt A Versatile Approach Toward Universal Spatio-Tem](learning_to_factorize_and_adapt_a_versatile_approach_toward_universal_spatio-tem.md)**

:   提出 FactoST-v2，一个因式分解的时空基础模型框架，将通用时间预训练与领域特定空间适配解耦，以线性复杂度实现跨领域零样本/少样本/全样本时空预测。

**[Less Is More But Where Dynamic Token Compression Via Llm-Guided Keyframe Prior](less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)**

:   提出 DyToK，一种无需训练的视频 token 动态压缩方法，利用 VLLM 深层注意力中固有的 query 条件关键帧先验，为不同帧自适应分配 token 预算，实现即插即用式的效率-精度最优权衡。

**[Linear Attention For Efficient Bidirectional Sequence Modeling](linear_attention_for_efficient_bidirectional_sequence_modeling.md)**

:   提出 Lion 框架，首次系统地将线性 Transformer 扩展到双向序列建模，统一了全线性注意力、双向 RNN 和分块并行三种等价表示，在图像分类和 MLM 任务上训练速度比 SSM 快达 10 倍且性能可比 softmax Transformer。

**[Littlebit Ultra Low-Bit Quantization Via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)**

:   提出 LittleBit 框架，通过低秩潜空间矩阵分解 + 二值化 + 多尺度补偿机制，实现低至 0.1 BPW（每权重比特）的极端 LLM 压缩，将 Llama2-13B 压缩到不足 0.9GB，在子1比特领域大幅超越 STBLLM。

**[Loquetier A Virtualized Multi-Lora Framework For Unified Llm Fine-Tuning And Ser](loquetier_a_virtualized_multi-lora_framework_for_unified_llm_fine-tuning_and_ser.md)**

:   提出Loquetier框架，通过虚拟化模块（Virtualized Module）和分段多LoRA乘法内核（SMLM），将多个LoRA适配器的微调和推理统一到单一运行时中，实现推理任务3.0×吞吐率提升和统一任务46.4×更高的SLO达成率。

**[Lt-Soups Bridging Head And Tail Classes Via Subsampled Model Soups](lt-soups_bridging_head_and_tail_classes_via_subsampled_model_soups.md)**

:   提出 LT-Soups，一个两阶段模型融合框架，通过在不同不平衡比例的子采样数据上训练多个模型并进行权重平均，在长尾分布的全频谱上实现头部类和尾部类的均衡性能。

**[Matryoshka Pilot Learning To Drive Black-Box Llms With Llms](matryoshka_pilot_learning_to_drive_black-box_llms_with_llms.md)**

:   提出 Matryoshka Pilot (M-Pilot)，用轻量级白盒 LLM 作为控制器，通过生成中间引导（任务分解、高层计划、用户画像）来驱动黑盒 LLM 在推理、规划和个性化等复杂长程任务上的性能，并通过迭代 DPO 实现自我改进。

**[Memory-Efficient Training With In-Place Fft Implementation](memory-efficient_training_with_in-place_fft_implementation.md)**

:   提出 rdFFT——首个真正原地（in-place）的实数域快速傅里叶变换框架，通过隐式复数编码方案消除中间缓冲区，实现训练时零额外内存开销的 FFT/IFFT 计算，内存效率最高提升 1500 倍以上。

**[Mitigating Semantic Collapse In Partially Relevant Video Retrieval](mitigating_semantic_collapse_in_partially_relevant_video_retrieval.md)**

:   针对部分相关视频检索（PRVR）中的语义坍塌问题，提出文本相关性保持学习和跨分支视频对齐（CBVA）方法，在文本和视频嵌入空间中分别解决坍塌现象，显著提升检索准确率。

**[Mixture Of Noise For Pre-Trained Model-Based Class-Incremental Learning](mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)**

:   提出学习有益的"混合噪声"来抑制预训练模型在增量学习中的参数漂移，通过在任务间进行动态权重混合噪声实现 SOTA 性能，特别在 50 步增量设置下表现突出。

**[Modhifi Identifying High Fidelity Predictive Components For Model Modification](modhifi_identifying_high_fidelity_predictive_components_for_model_modification.md)**

:   提出 Subset Fidelity 度量和 ModHiFi 框架，通过理论证明 Lipschitz 连续网络的局部重构误差线性上界全局误差，无需训练数据、损失函数或梯度，仅用合成数据即可识别模型中的高保真 (HiFi) 组件，统一实现结构化剪枝和类别遗忘两大任务。

**[Multi-Task Vehicle Routing Solver Via Mixture Of Specialized Experts Under State](multi-task_vehicle_routing_solver_via_mixture_of_specialized_experts_under_state.md)**

:   提出 **State-Decomposable MDP (SDMDP)** 框架将多种 VRP 变体重新表述为基础状态空间的笛卡尔积，再通过 **Mixture-of-Specialized-Experts Solver (MoSES)** 用专用 LoRA 专家实现基础策略的潜在空间复用，高效处理 16 种 VRP 变体。

**[Mustafar Promoting Unstructured Sparsity For Kv Cache Pruning In Llm Inference](mustafar_promoting_unstructured_sparsity_for_kv_cache_pruning_in_llm_inference.md)**

:   提出 MUSTAFAR 框架，系统性地证明了非结构化稀疏性在 KV 缓存剪枝中的优越性（Key 和 Value 均可达 70% 稀疏度且不损精度），并设计了基于 bitmap 的稀疏格式和自定义注意力内核，实现了端到端推理吞吐量 2.23 倍加速。

**[Navigating Simply Aligning Deeply Winning Solutions For Mouse Vs Ai 2025](navigating_simply_aligning_deeply_winning_solutions_for_mouse_vs_ai_2025.md)**

:   在NeurIPS 2025 Mouse vs. AI竞赛中，本文展示了轻量级两层CNN在视觉鲁棒性任务上大幅超越深度网络的反直觉发现，同时证明深层ResNet架构在神经对齐任务上更具优势，揭示了行为鲁棒性与生物合理性之间的根本张力。

**[On The Creation Of Narrow Ai Hierarchy And Nonlocality Of Neural Network Skills](on_the_creation_of_narrow_ai_hierarchy_and_nonlocality_of_neural_network_skills.md)**

:   研究创建窄域（narrow）AI 系统面临的两大挑战：任务的层级依赖使得某些窄域技能必须在宽分布上训练才能学会；技能的非局部性使得剪枝无法精确分离想要保留和舍弃的能力——但剪枝+恢复训练仍优于蒸馏和从头训练。

**[On The Hardness Of Approximating Distributions With Tractable Probabilistic Mode](on_the_hardness_of_approximating_distributions_with_tractable_probabilistic_mode.md)**

:   本文证明了用可处理概率模型（如分解概率电路）在有界f-散度下近似任意分布是NP-hard的，并证明了在近似建模条件下分解PC和（确定性+分解）PC之间存在指数级大小差距，揭示了近似放宽并不能缓解精确建模中的复杂度瓶颈。

**[One-Step Diffusion-Based Image Compression With Semantic Distillation](one-step_diffusion-based_image_compression_with_semantic_distillation.md)**

:   提出OneDC——首个一步扩散生成式图像编解码器，将超先验（hyperprior）替代文本作为扩散模型的语义引导并通过语义蒸馏增强其表示能力，实现了比多步扩散编解码器节省39%码率、解码加速20倍的SOTA感知质量。

**[Online Mixture Of Experts No-Regret Learning For Optimal Collective Decision-Mak](online_mixture_of_experts_no-regret_learning_for_optimal_collective_decision-mak.md)**

:   提出在线专家混合（OMoE）框架，包含 UCB 逐步消除和在线加权多数投票两种算法，理论上保证无遗憾（no-regret），并应用于 LLM 专家的在线动态聚合。

**[Optimizing Distributional Geometry Alignment With Optimal Transport For Generati](optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)**

:   将数据集蒸馏重新表述为最优传输（OT）距离最小化问题，通过三阶段（OT引导扩散采样、标签-图像对齐软重标注、OT logit匹配）实现细粒度分布几何对齐，在ImageNet-1K IPC=10上比之前SOTA提升至少4%。

**[Order-Level Attention Similarity Across Language Models A Latent Commonality](order-level_attention_similarity_across_language_models_a_latent_commonality.md)**

:   提出 Order-Level Attention (OLA)——对 Attention Rollout 的阶次分解，发现不同语言模型在同阶 OLA 上存在显著相似性 (OLAS)，并且 OLA 隐式编码了句法知识，基于此提出 TOA 实现首个无需训练的跨LM适配器迁移。

**[Paretoq Improving Scaling Laws In Extremely Low-Bit Llm Quantization](paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)**

:   提出 ParetoQ——首个统一 1/1.58/2/3/4 比特量化的框架，通过系统研究训练策略（全精度预训练 vs. QAT 分配）和量化函数设计（提出 SEQ 量化器），发现 2-bit 和 1.58-bit 量化在精度-模型大小折中上优于传统 4-bit，且各比特位宽均达到 SOTA。

**[Ppg-Distill Efficient Photoplethysmography Signals Analysis Via Foundation Model](ppg-distill_efficient_photoplethysmography_signals_analysis_via_foundation_model.md)**

:   PPG-Distill提出一种针对PPG信号的知识蒸馏框架，通过预测级、特征级和Patch级（形态+节律）蒸馏，将大型PPG基础模型的知识迁移到轻量学生模型，在保持性能（最高提升21.8%）的同时实现7倍推理加速和19倍内存压缩。

**[Q-Palette Fractional-Bit Quantizers Toward Optimal Bit Allocation For Efficient ](q-palette_fractional-bit_quantizers_toward_optimal_bit_allocation_for_efficient_.md)**

:   从信息论角度推导高斯化权重的最优比特分配,提出 Q-Palette 分数位量化器集合和混合方案量化框架,在 LLM 推理中实现近最优的量化性能和推理加速。

**[Qsvd Efficient Low-Rank Approximation For Unified Query-Key-Value Weight Compres](qsvd_efficient_low-rank_approximation_for_unified_query-key-value_weight_compres.md)**

:   提出QSVD方法，通过对QKV联合权重矩阵的SVD分解共享下投影矩阵来减少KV缓存和计算开销，结合基于重要性评分的自适应秩分配和量化技术，在VLM上实现超过10%的精度提升且硬件成本更低。

**[Quadenhancer Leveraging Quadratic Transformations To Enhance Deep Neural Network](quadenhancer_leveraging_quadratic_transformations_to_enhance_deep_neural_network.md)**

:   提出一种轻量级的二次增强器（QuadEnhancer），通过在每个线性层引入稀疏化的二次交互项，以极少的额外参数和计算开销显著提升现有神经网络架构的性能。

**[Quantization Error Propagation Revisiting Layer-Wise Post-Training Quantization](quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)**

:   识别现有逐层 PTQ 方法忽略量化误差跨层累积和增长的关键瓶颈，提出 QEP 框架通过误差传播和补偿显式纠正累积误差，在极低比特（INT2/INT3）下实现大幅性能提升。

**[Rat Bridging Rnn Efficiency And Attention Accuracy Via Chunk-Based Sequence Mode](rat_bridging_rnn_efficiency_and_attention_accuracy_via_chunk-based_sequence_mode.md)**

:   提出 RAT（Recurrence And aTtention），一种基于 Chunk 的中间架构——在 Chunk 内使用线性 RNN 建模局部依赖、Chunk 间使用 softmax 注意力实现全局访问。L=16 时单层解码速度提升 9 倍、最大吞吐量提升 10 倍，且性能与标准注意力持平；与滑动窗口注意力交替使用的混合变体在几乎所有 benchmark 上最优。

**[Rccda Adaptive Model Updates In The Presence Of Concept Drift Under A Constraine](rccda_adaptive_model_updates_in_the_presence_of_concept_drift_under_a_constraine.md)**

:   提出 RCCDA，一种基于 Lyapunov 漂移惩罚框架的轻量级模型更新策略，在数据分布随时间漂移（concept drift）场景下，仅利用历史推理损失信息和可调阈值，就能贪心最优地决定何时重训模型，同时可证明地满足严格资源预算约束。

**[Rectifying Soft-Label Entangled Bias In Long-Tailed Dataset Distillation](rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)**

:   揭示了长尾数据集蒸馏中软标签存在来自蒸馏模型和蒸馏图像的双重纠缠偏差，提出 ADSA 自适应软标签对齐模块，通过logit空间的后处理校准消除偏差，作为即插即用模块可无缝集成到现有蒸馏方法中，在 ImageNet-1k-LT 上将尾部类准确率提升高达11.8%。

**[Reflora Refactored Low-Rank Adaptation For Efficient Fine-Tuning Of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)**

:   RefLoRA 通过在每次迭代中选择最优的低秩分解形式（最小化损失上界），解决了 LoRA 因分解不唯一性导致的权重更新不一致和不平衡问题，在几乎不增加计算开销的前提下加速收敛并提升微调性能。

**[Reject Only Critical Tokens Pivot-Aware Speculative Decoding](reject_only_critical_tokens_pivot-aware_speculative_decoding.md)**

:   PAD 提出了基于效用匹配（而非分布匹配）的推测解码新范式：训练一个轻量分类器识别"关键 token"（pivot token），仅拒绝会导致最终输出效用下降的 draft token，从而在 GSM8K 上实现 2.46× 加速且几乎不损失准确率。

**[Reordering Patches Improves Vision Models](reordering_patches_improves_vision_models.md)**

:   揭示了视觉模型中 patch 排列顺序对长序列模型性能有显著影响，并提出 REOrder 框架通过信息论先验和强化学习自动发现最优 patch 排列，在 ImageNet-1K 上提升高达 3.01%，在 FMoW 上提升 13.35%。

**[Rep Resource-Efficient Prompting For Rehearsal-Free Continual Learning](rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)**

:   REP 通过轻量代理模型的快速提示选择、自适应 Token 合并（AToM）和自适应层丢弃（ALD）三种互补技术，将基于提示的无排练持续学习方法的训练时间减少最高 51%、内存降低最高 41%，精度损失微乎其微。

**[Replaceme Network Simplification Via Depth Pruning And Transformer Block Lineari](replaceme_network_simplification_via_depth_pruning_and_transformer_block_lineari.md)**

:   提出 ReplaceMe，一种无训练的深度剪枝方法：用少量校准数据估计线性变换来近似被剪枝的 Transformer 块组，该变换可融合到相邻层权重中不增加参数，在 LLaMA-2-7B 上实现 25% 剪枝率并保留约 90% 性能。

**[Representation Consistency For Accurate And Coherent Llm Answer Aggregation](representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)**

:   提出 Representation Consistency (RC)，通过分析 LLM 生成多个候选答案时内部激活的一致性来改进答案聚合：同一答案的多条推理路径如果内部表示高度一致则更可能正确，结合稀疏自编码器的稀疏变体 RC-S 效果最优，在 4 个 LLM 和 4 个推理数据集上一致优于 Self-Consistency。

**[Restoring Pruned Large Language Models Via Lost Component Compensation](restoring_pruned_large_language_models_via_lost_component_compensation.md)**

:   RestoreLCC 提出了一种面向剪枝 LLM 的定向恢复策略：通过对比探测定位关键注意力头，利用 SVD 分解提取剪枝丢失的激活成分，将其作为可优化的偏置向量注入回剪枝模型，在不影响稀疏性和推理速度的前提下显著恢复性能。

**[Revisiting Semi-Supervised Learning In The Era Of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)**

:   系统性研究发现传统 SSL 方法在 VFM 时代效益有限——仅用有标签数据的 PEFT 即可匹敌 SSL——由此提出 V-PET：集成多种 PEFT 方法和多种 VFM 的伪标签来实现简洁高效的半监督学习。

**[Robust Federated Finetuning Of Llms Via Alternating Optimization Of Lora](robust_federated_finetuning_of_llms_via_alternating_optimization_of_lora.md)**

:   提出 RoLoRA，通过交替优化 LoRA 的 down-projection (A) 和 up-projection (B) 矩阵，解决联邦学习中 LoRA 聚合不精确和表达力受限的问题，在 RoBERTa-Large 和 Llama-2-7B 上显著优于 FedAVG of LoRA 和 FFA-LoRA。

**[Robustifying Learning-Augmented Caching Efficiently Without Compromising 1-Consi](robustifying_learning-augmented_caching_efficiently_without_compromising_1-consi.md)**

:   提出 Guard 框架，一种轻量级鲁棒化方法，将广泛的学习增强缓存算法的鲁棒性提升至 $2H_{k-1}+2$，同时保持 1-一致性和 O(1) 的额外请求开销。

**[S2M-Former Spiking Symmetric Mixing Branchformer For Brain Auditory Attention De](s2m-former_spiking_symmetric_mixing_branchformer_for_brain_auditory_attention_de.md)**

:   提出 S2M-Former，一种脉冲驱动的对称混合 Branchformer 框架，通过空间-频率双分支的互补学习和轻量化 1D token 表示，在 EEG 听觉注意力检测任务上以仅 0.06M 参数实现了 SOTA 级精度，同时将能耗降低至双分支 ANN 模型的 1/5.8。

**[S2Q-Vdit Accurate Quantized Video Diffusion Transformer With Salient Data And Sp](s2q-vdit_accurate_quantized_video_diffusion_transformer_with_salient_data_and_sp.md)**

:   针对视频扩散 Transformer 的超长 token 序列导致的量化校准高方差和学习困难问题，提出 S²Q-VDiT 框架，利用 Hessian 感知的显著数据选择和注意力引导的稀疏 token 蒸馏两项技术，首次在 W4A6 设置下实现无损量化，带来 3.9× 模型压缩和 1.3× 推理加速。

**[Shap Meets Tensor Networks Provably Tractable Explanations With Parallelism](shap_meets_tensor_networks_provably_tractable_explanations_with_parallelism.md)**

:   本文首次为张量网络（Tensor Networks）提供可证明精确的 SHAP 解释计算框架，证明张量列车（Tensor Train）结构下 SHAP 可在多对数时间内并行计算（NC² 复杂度），并通过归约揭示二值化神经网络中**宽度而非深度**才是 SHAP 计算的核心瓶颈。

**[Single-Teacher View Augmentation Boosting Knowledge Distillation Via Angular Div](single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)**

:   提出Angular-KD，通过在单个教师模型上附加多个轻量线性分支并引入两种角度多样性损失（约束型视角间角度多样性损失和视角内角度多样性损失），从单教师生成多样化监督信号，以低成本替代多教师蒸馏方案，在多个KD基准上取得SOTA表现。

**[Skrull Towards Efficient Long Context Fine-Tuning Through Dynamic Data Schedulin](skrull_towards_efficient_long_context_fine-tuning_through_dynamic_data_schedulin.md)**

:   针对长上下文监督微调（Long-SFT）中长短序列混合导致的训练效率低下问题，提出动态数据调度器Skrull，通过分布感知的上下文并行（DACP）和全局数据调度（GDS）两个组件，在真实Long-SFT场景中实现平均3.76倍（最高7.54倍）的训练加速。

**[Smooth Regularization For Efficient Video Recognition](smooth_regularization_for_efficient_video_recognition.md)**

:   提出一种基于高斯随机游走（GRW）的平滑正则化技术，通过对视频识别模型中间层嵌入施加时序平滑约束（惩罚高加速度变化），在轻量级模型上实现3.8%–6.4%的准确率提升，在相应FLOP约束下刷新Kinetics-600 SOTA。

**[Spark Transformer Reactivating Sparsity In Ffn And Attention](spark_transformer_reactivating_sparsity_in_ffn_and_attention.md)**

:   提出 Spark Transformer 架构，通过 Statistical Top-k 算子在 FFN 和注意力机制中同时实现高水平激活稀疏性（FFN 仅 8% 神经元激活、每个 token 最多关注 256 个 token），在保持与 Gemma-2 相当质量的同时实现 2.5× FLOPs 降低和高达 1.79× 的推理加速。

**[Specialization After Generalization Towards Understanding Test-Time Training In ](specialization_after_generalization_towards_understanding_test-time_training_in_.md)**

:   提出"泛化之后特化"框架，基于线性表示假设（LRH）从理论和实验两方面解释了测试时训练（TTT）在分布内数据上的有效性：基础模型全局欠参数化导致概念叠加干扰，TTT通过局部特化将模型容量重新分配给与测试任务相关的少数概念，从而在不增加模型规模的情况下提升预测性能。

**[Spiking Brain Compression Post-Training Second-Order Compression For Spiking Neu](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)**

:   提出 Spiking Brain Compression（SBC），一种基于 Van Rossum 距离的二阶后训练一次性压缩框架，专为脉冲神经网络（SNN）设计，通过替代膜电位（SMP）Hessian 实现高效的模块级剪枝和量化，在 ImageNet 规模下首次压缩 SEW-ResNet152 和 Spike-Driven Transformer。

**[Synergy Between The Strong And The Weak Spiking Neural Networks Are Inherently S](synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)**

:   本文发现 SNN 可以在时间维度上天然解构为多个子模型，通过对比各时间步子模型的输出置信度识别"强"与"弱"，提出 Strong2Weak 和 Weak2Strong 两种自蒸馏方案，无需额外教师模型即可显著提升 SNN 性能，尤其在神经形态数据集上提升高达 5.36%。

**[The Graphon Limit Hypothesis Understanding Neural Network Pruning Via Infinite W](the_graphon_limit_hypothesis_understanding_neural_network_pruning_via_infinite_w.md)**

:   提出"Graphon极限假说"：当网络宽度趋于无穷时，不同剪枝方法产生的二值掩码序列在cut距离下收敛到各自独特的graphon极限，并在此基础上推导出Graphon NTK来分析稀疏网络训练动态，从理论层面解释了为什么不同剪枝方法在相同稀疏度下表现迥异。

**[The Structure Of Relation Decoding Linear Operators In Large Language Models](the_structure_of_relation_decoding_linear_operators_in_large_language_models.md)**

:   揭示 Transformer 语言模型中的线性关系解码器（LRE）并非编码细粒度关系，而是提取共享的粗粒度语义属性（如"国家"、"性别"），并利用阶-3 张量网络将大量关系解码矩阵压缩数个数量级。

**[Tighter Cmi-Based Generalization Bounds Via Stochastic Projection And Quantizati](tighter_cmi-based_generalization_bounds_via_stochastic_projection_and_quantizati.md)**

:   通过在 CMI（条件互信息）框架中引入**随机投影**和**有损压缩**，推导出更紧的泛化界，解决了经典 CMI 界在 SCO 反例上失效的问题，并证明记忆化对良好泛化并非必要。

**[Tokensqueeze Performance-Preserving Compression For Reasoning Llms](tokensqueeze_performance-preserving_compression_for_reasoning_llms.md)**

:   提出TokenSqueeze方法，通过自适应推理深度选择、步内语言精炼（基于KL散度约束）和长度感知的偏好优化三阶段流程，仅用模型自生成数据实现推理链50%的token压缩而不损失准确率。

**[Toward Efficient Inference Attacks Shadow Model Sharing Via Mixture-Of-Experts](toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)**

:   提出基于 Mixture-of-Experts 的影子模型共享方案，通过在多种推理攻击任务间共享影子模型的特征提取层、仅训练任务特定的轻量专家模块来降低影子模型的整体训练成本，同时保持或提升攻击性能。

**[Towards Implicit Aggregation Robust Image Representation For Place Recognition I](towards_implicit_aggregation_robust_image_representation_for_place_recognition_i.md)**

:   提出 ImAge（Implicit Aggregation），在 Transformer 骨干网络的特定层插入可学习聚合 Token，利用内在自注意力机制将 patch 特征隐式聚合为全局描述符，完全消除了额外聚合器的需要。以最小的描述符维度（6144）和最快推理速度，在多个 VPR 数据集上超越 SALAD、BoQ 等 SOTA，并在 MSLS Challenge 排行榜排名第 1。

**[Towards Unsupervised Open-Set Graph Domain Adaptation Via Dual Reprogramming](towards_unsupervised_open-set_graph_domain_adaptation_via_dual_reprogramming.md)**

:   提出 GraphRTA 框架，通过模型重编程（基于梯度的权重剪枝）和图重编程（目标图结构与特征优化）双重机制，解决无监督开放集图域适应中已知类分类与未知类识别难题，无需人工设定阈值。

**[Train With Perturbation Infer After Merging A Two-Stage Framework For Continual ](train_with_perturbation_infer_after_merging_a_two-stage_framework_for_continual_.md)**

:   提出Perturb-and-Merge (P&M)框架，将模型合并机制引入持续学习范式：训练时沿任务向量方向添加随机扰动以平滑损失面，推理时通过闭式最优系数对历史模型和当前任务模型做凸组合合并，结合LoRA实现内存高效的SOTA持续学习性能。

**[Trajectory Balance With Asynchrony Decoupling Exploration And Learning For Fast ](trajectory_balance_with_asynchrony_decoupling_exploration_and_learning_for_fast_.md)**

:   提出 TBA（Trajectory Balance with Asynchrony），将 GFlowNet 的轨迹平衡（TB）目标与异步分布式 RL 架构结合，实现 LLM 后训练中探索与学习的解耦，在数学推理、偏好微调和自动红队测试任务上获得 4-50 倍加速且性能不降反升。

**[Traversal Verification For Speculative Tree Decoding](traversal_verification_for_speculative_tree_decoding.md)**

:   提出 Traversal Verification，一种从叶节点到根节点的自底向上验证算法，通过考虑整条路径的序列级概率而非单 token 概率来决定接受/拒绝，理论证明无损性和单链最优性，在多种树结构和任务上一致提升接受长度 2.2%-5.7%。

**[Twilight Adaptive Attention Sparsity With Hierarchical Top-P Pruning](twilight_adaptive_attention_sparsity_with_hierarchical_top-p_pruning.md)**

:   提出 Twilight，借鉴 top-p 采样（nucleus sampling）的思想替代固定预算 top-k 做注意力稀疏——动态选择注意力权重累积达 p% 的最少 Token，自适应不同注意力头的分布特征，在保持精度的同时比 SOTA 稀疏注意力再提速 1.4x。

**[Ultrametric Cluster Hierarchies I Want Em All](ultrametric_cluster_hierarchies_i_want_em_all.md)**

:   证明了对于任意合理的聚类层次树，都可以快速找到任意中心型聚类目标（如 k-means）的最优解，且这些解本身也是层次化的，从而从一棵树中解锁大量等价有意义的层次结构。

**[Understanding Differential Transformer Unchains Pretrained Self-Attentions](understanding_differential_transformer_unchains_pretrained_self-attentions.md)**

:   深入分析 Differential Transformer（差分注意力）的内部机制，揭示差分操作等效于一种鲁棒的注意力去噪过程——它"解放"了受 softmax 归一化约束的预训练自注意力，使注意力权重更自由地分配到真正重要的 Token 上。

**[Uni-Lora One Vector Is All You Need](uni-lora_one_vector_is_all_you_need.md)**

:   提出 Uni-LoRA 统一框架，证明各种 LoRA 变体（Tied-LoRA、VeRA、VB-LoRA 等）的参数缩减策略本质上是对全参数空间 $\mathbb{R}^D$ 到低维子空间 $\mathbb{R}^d$ 的投影差异，并设计了一种等距随机分组投影矩阵——只需训练一个向量即可重建整个 LLM 的 LoRA 参数，实现极致参数效率。

**[Universal Cross-Tokenizer Distillation Via Approximate Likelihood Matching](universal_cross-tokenizer_distillation_via_approximate_likelihood_matching.md)**

:   本文提出 Approximate Likelihood Matching (ALM)，一种基于二值化 f-散度的原则性跨分词器蒸馏方法，首次实现了跨根本不同分词器（如子词→字节级）的有效蒸馏和纯蒸馏。

**[Vessa Video-Based Object-Centric Self-Supervised Adaptation For Visual Foundatio](vessa_video-based_object-centric_self-supervised_adaptation_for_visual_foundatio.md)**

:   提出 VESSA，一种利用短物体中心视频进行无监督微调的方法，通过自蒸馏框架配合 LoRA 和不确定性加权损失，在不需要标注数据的情况下将视觉基础模型适配到目标域，在 33 个 VFM × 22 个数据集上持续提升下游分类性能。

**[Vision-Centric Token Compression In Large Language Model](vision-centric_token_compression_in_large_language_model.md)**

:   Vist 提出了一种以视觉为核心的慢-快双路径 token 压缩框架，将远端长文本渲染为图像后用轻量视觉编码器压缩，配合概率引导的视觉增强（PVE）训练目标，在 11 个 ICL 基准上以 2.3× 更少的 token 实现同等精度，FLOPs 降低 16%、显存减少 50%。

**[Vqtoken Neural Discrete Token Representation Learning For Extreme Token Reductio](vqtoken_neural_discrete_token_representation_learning_for_extreme_token_reductio.md)**

:   VQToken 提出了首个基于向量量化的视频 token 极限压缩框架，通过自适应离散化将连续 ViT embedding 聚类为紧凑码本，并用 token hash 函数保留时空位置信息，在 NextQA-MC 上仅用原始 0.07% 的 token（约 13 个）实现了仅 0.66% 的精度损失。

**[When Worse Is Better Navigating The Compression-Generation Tradeoff In Visual To](when_worse_is_better_navigating_the_compression-generation_tradeoff_in_visual_to.md)**

:   系统研究视觉 Tokenizer 的压缩-生成权衡，发现更激进的压缩反而有利于小模型生成，并提出因果正则化 Tokenization（CRT）方法，通过嵌入自回归归纳偏置使 token 更易建模，实现 2-3 倍计算效率提升。

**[Zip2Zip Inference-Time Adaptive Tokenization Via Online Compression](zip2zip_inference-time_adaptive_tokenization_via_online_compression.md)**

:   提出 zip2zip，通过将 LZW 在线压缩算法集成到 LLM 推理流程中，实现推理时动态扩展词表生成"超级Token"（hypertokens），将输入输出序列长度缩减 15-40%，端到端延迟降低最高 40%。
