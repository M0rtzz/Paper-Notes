---
title: >-
  ICML2026 模型压缩方向21篇论文解读
description: >-
  21篇ICML2026的模型压缩方向论文解读，涵盖模型压缩、压缩/编码、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "模型压缩"
  - "论文解读"
  - "论文笔记"
  - "压缩/编码"
  - "LLM"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📦 模型压缩

**🧪 ICML2026** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (49)](../../ACL2026/model_compression/index.md) · [📷 CVPR2026 (50)](../../CVPR2026/model_compression/index.md) · [🔬 ICLR2026 (90)](../../ICLR2026/model_compression/index.md) · [🤖 AAAI2026 (53)](../../AAAI2026/model_compression/index.md) · [🧠 NeurIPS2025 (134)](../../NeurIPS2025/model_compression/index.md) · [📹 ICCV2025 (48)](../../ICCV2025/model_compression/index.md)

🔥 **高频主题：** 模型压缩 ×5 · 压缩/编码 ×5 · LLM ×3

**[ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin](arcvq-vae_a_spherical_vector_quantization_framework_with_arccosine_additive_marg.md)**

:   作者诊断出 VQ-VAE 的码本坍塌根源是"码本向量 ℓ2 范数失衡 + 几何聚集"，于是提出 SAMP：Ball-Bounded Norm Regularization 把所有码本向量约束在时变 Euclidean 球内、ArcCosine Additive Margin Loss 借鉴 ArcFace 在球面上推开 latent 向量，从而让码本均匀分散、利用率大幅上升，在 ImageNet 重建和生成 FID 上都击败主流 VQ-VAE 变体。

**[Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](breaking_the_moe_llm_trilemma_dynamic_expert_clustering_with_structured_compress.md)**

:   针对 MoE LLM 的"负载不均–参数冗余–通信开销"三难，本文提出一个统一框架：用"参数 + 激活"双相似度在线聚类把专家分组，组内用"共享基矩阵 + 低秩残差"做结构化压缩 (~5×)，再做"先选组后选 expert"的两级分层路由 + FP16/INT4 异构精度 + 闲置组离线卸载，在 GLUE/WikiText-103 上以约 80% 参数缩减、10–20% 吞吐提升、专家负载方差降 3× 的代价匹配标准 MoE 性能。

**[Demystifying When Pruning Works via Representation Hierarchies](demystifying_when_pruning_works_via_representation_hierarchies.md)**

:   论文从"嵌入 → logit → 概率"三段表征层次出发，用 Taylor 局部展开理论证明：剪枝对嵌入空间和 logit 空间的扰动天生很小，但 softmax 这一非线性步骤会按 $\mathrm{Var}_r(\Delta z)/(2T^2)$ 把扰动放大到概率空间，再经过自回归解码的步间累积，最终导致生成任务崩溃；而非生成任务因为只依赖候选 token 子空间，对剪枝天然鲁棒——这统一解释了为什么剪枝在 MMLU、retrieval 上几乎无损但在 GSM8K、HumanEval 上骤降到 0。

**[Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](dispersion_loss_counteracts_embedding_condensation_and_improves_generalization_i.md)**

:   本文系统观测到 "小语言模型的 token 嵌入会随深度坍缩到一个窄锥体"（embedding condensation）这个普遍现象——大模型反而不会——并设计了一个角度分散损失 $\mathcal{L}_{\text{disp}}$ 直接逼嵌入散开，无须加参数就让 Qwen3 / GPT2 在 10 个 benchmark 上平均提升 3.3%。

**[Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)**

:   本文提出 TAD（Tail-Aware Distillation）：在标准 KD 的 KL 散度中显式把教师 top-$K$ 概率与"尾部"概率拆开并放大尾部贡献，从而在学术级算力（单卡 H100 + 1 周）内完成 LLM 预训练蒸馏，平均效果优于 MiniPLM 等数据中心方法。

**[FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA](fedrot-lora_mitigating_rotational_misalignment_in_federated_lora.md)**

:   本文指出联邦 LoRA 中朴素 factor-wise 平均的真正"敌人"是旋转不变性导致的潜在子空间错位，提出在客户端用正交 Procrustes 求解出旋转矩阵 $R_i^t$ 对齐 $A,B$ 因子后再聚合，理论与实验都证明能显著降低聚合误差且不增加通信开销。

**[FlattenGPT: Depth Compression for Transformer with Layer Flattening](flattengpt_depth_compression_for_transformer_with_layer_flattening.md)**

:   本文提出 FlattenGPT，先把 LLM 中输入相似度高的相邻 transformer 层"扁平化"合并为一个 2× 宽度的层 (保留所有参数知识)，再对合并层做通道剪枝把宽度恢复到原始规模——既享受深度压缩的推理加速，又避免传统层剪枝直接丢知识的性能塌方。

**[From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers](from_per-image_low-rank_to_encoding_mismatch_rethinking_feature_distillation_in_.md)**

:   作者用 sample-wise SVD + dataset-level PCA + token-level Spectral Energy Pattern (SEP) 三视角揭示了一个看似矛盾的 ViT 表征几何："每张图的特征矩阵都是低秩的，但跨图共享的子空间却几乎要满秩 + 单 token 的频谱带宽接近 100%"，进而提出 Lift（推理时保留 lifting projector）和 WideLast（只把最后一个 block 加宽到 teacher 宽度）两个极简补丁，让普通 MSE 特征蒸馏在 DeiT-Tiny ← CaiT-S24 上从 74.86% 一路涨到 78.23%。

**[Linearizing Vision Transformer with Test-Time Training](linearizing_vision_transformer_with_test-time_training.md)**

:   作者发现两层 TTT 内模型在结构上等价于 Softmax 注意力（Softmax 可看作两层动态 MLP），由此实现 Q/K/V/MLP 的全权重直接继承，再通过 key Instance Normalization 处理 shift-invariance、depthwise conv on Q/K 补齐 locality，仅 1 小时微调就把 Stable Diffusion 3.5 线性化并加速 1.32×–1.47×。

**[OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization](osaq_outlier_self-absorption_for_accurate_low-bit_llm_quantization.md)**

:   OSAQ 利用 LLM 各层 Hessian 在不同输入下保持一致的低秩零空间，将零空间向量线性组合成一个加性权重扰动 $\Delta W$，在不改变二阶任务损失的前提下把离群权重「自吸收」掉，使 2 比特仅权重量化的困惑度比朴素 GPTQ 降低 40% 以上。

**[Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs](preserve-then-quantize_balancing_rank_budgets_for_quantization_error_reconstruct.md)**

:   作者提出 SRR（Structured Residual Reconstruction），把 QER（Quantization Error Reconstruction）中固定用于补偿量化残差的低秩预算 $r$ 显式地拆成"先保留 $k$ 个主奇异方向再量化"和"用 $r-k$ 个秩去拟合残差"两部分，并给出一个只需一次随机探针的闭式准则来逐层选 $k^\star$，在 2/3 bit PTQ 和 QPEFT 上一致优于 LQER/QERA。

**[Proxy Compression for Language Modeling](proxy_compression_for_language_modeling.md)**

:   作者提出「proxy compression」——训练时把 90% 数据喂成 tokenizer / 神经压缩器产出的短序列、10% 喂原始 UTF-8 字节，配合 sentinel token 与短暂的 in-context translation warm-up；推理时丢掉所有压缩器，模型只看原始字节，却能在固定 compute 下显著超过纯字节模型，且在大规模下追平甚至超过 tokenizer baseline。

**[Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models](resting_neurons_active_insights_robustify_activation_sparsity_for_large_language.md)**

:   本文把激活稀疏导致 LLM 掉点的本质归因为"表示漂移"，并仿照生物自发放电向每层注入一个输入无关、训练后可吸收进 bias 的小向量（SPON），以接近零推理开销显著缩小稀疏模型与稠密模型的差距。

**[RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)**

:   RQ-MoE 用「两级 MoE + 双流量化」的设计，让残差向量量化（RQ）的码本随输入动态生成，又通过把指令流与重建流解耦实现 6–14× 解码加速，在四个 retrieval benchmark 上 MSE/Recall 持平或超越 QINCo。

**[ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)**

:   作者证明 LoRA 累加更新被困在固定低秩子空间，提出 ScaLoRA：每步把旧 $AB^\top$ 合并到 $W^{pt}$ 后，**用一个可解析求得的最优"列缩放"** 重启 adapter，使 AdamW 一阶/二阶动量可以 $O((m+n)r)$ 等变传递 (不需要重置/warm-up)、累加更新自然变高秩——在 DeBERTaV3、LLaMA2-7B、LLaMA3-8B、Gemma3-12B 上一致打过 LoRA / MoRA / HiRA / ReLoRA / LoRA-GA。

**[Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in.md)**

:   本文先用新基准 KVFundaBench 系统揭示「检索类长上下文压得动、推理类压不动」的关键不对称，并把原因归结到 KV 压缩破坏了少样本示例这一「语义单元」的完整性；据此提出 ShotKV——在 prefill 阶段保留整个 shot 作为不可分割单元、在 decoding 阶段做动态 token 级压缩，让 LG-GSM8K 在 40% 压缩率下从 baseline 46.0 提升到 47.33，并在长输入设置下端到端延迟降低 11.3%。

**[Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)**

:   SANTA 把 attention 的 value 聚合 $AV$ 看作 "按 softmax 概率 $A$ 对值行 $V$ 做加权求和", 改成 "从 $A$ 中无放回采样 $S\ll n_k$ 个索引、直接平均对应 $V$ 行"的无偏估计, 用 stratified / systematic 采样降方差, 再写成 GPU kernel 与 FlashDecoding 对齐——在 32k context 下端到端比 FlashInfer / FlashDecoding 快 1.5× 且精度不掉。

**[SURGE: Surrogate Gradient Adaptation in Binary Neural Networks](surge_surrogate_gradient_adaptation_in_binary_neural_networks.md)**

:   SURGE 给每个二值化层并联一个"全精度辅助分支"，前向输出不变但反向能从全精度分支额外回传一份"非 STE 截断"的高阶梯度，并用 AGS 按梯度范数比动态平衡两路贡献，让 BNN 在 ResNet-18/ImageNet 上做到 62.0% top-1，比 ReCU 高 1.0%、比 IR-Net 高 3.9%。

**[Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](task-driven_subspace_decomposition_for_knowledge_sharing_and_isolation_in_lora-b.md)**

:   LoDA 把 LoRA 的下投影矩阵按「投影能量」拆成一个跨任务共享的通用子空间和一个真正只激活新任务的隔离子空间，再用梯度对齐训练上投影、并在融合时给通用分支闭式重标定，从而在多个持续学习 benchmark 上稳定刷过现有 LoRA-CL 方法。

**[Test-Time Training with KV Binding Is Secretly Linear Attention](test-time_training_with_kv_binding_is_secretly_linear_attention.md)**

:   本文用四个「记忆悖论」反例 + 一套严格的展开定理，证明带 KV-binding 内循环的 TTT（如 LaCT、ViTTT）即便用多层 MLP + 动量也只是「学到的线性注意力算子」，并据此把它简化、并行化为标准线性注意力，吞吐提升 4× 而性能几乎不掉。

**[Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)**

:   作者发现 token 的"重要性"在层间和头间剧烈变化，传统 token eviction 一次性删除是不可逆的早期决策错误；他们提出 Token Sparse Attention，每层每个 attention head 独立选 $L' \ll L$ 个 token 做密集 attention，输出再 scatter 回原始序列长度，配上残差路径让被略过的 token 在下一层重新有机会被选中——既保留头/层级动态选择，又能直接调用 FlashAttention 等密集 kernel，在 128K 上下文上叠加 FlexPrefill 后达到 ×3.23 注意力加速、精度损失 <1%。
