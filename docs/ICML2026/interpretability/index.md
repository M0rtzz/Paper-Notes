---
title: >-
  ICML2026 可解释性方向21篇论文解读
description: >-
  21篇ICML2026的可解释性方向论文解读，涵盖 LLM、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "可解释性"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**🧪 ICML2026** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (51)](../../ACL2026/interpretability/index.md) · [📷 CVPR2026 (28)](../../CVPR2026/interpretability/index.md) · [🔬 ICLR2026 (55)](../../ICLR2026/interpretability/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/interpretability/index.md) · [🧠 NeurIPS2025 (84)](../../NeurIPS2025/interpretability/index.md) · [📹 ICCV2025 (10)](../../ICCV2025/interpretability/index.md)

🔥 **高频主题：** LLM ×3

**[All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)**

:   这篇论文用 Overlap-Aware Sheaf Repulsion (OASR) 算法系统性地证伪了机理可解释性领域的隐含假设——"一个 LLM 能力对应一个独特的电路"——发现同一任务可被多个几乎不重叠 (IoU ~4–11%) 但都满足 faithful/sparse/complete 的电路或 sheaf 支撑，并给出"分布式稠密电路假设"作为理论解释。

**[Barriers to Counterfactual Credit Attribution for Autoregressive Models](barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)**

:   本文形式化研究生成式模型在 RAG/in-context 部署时的"反事实信用归因（CCA）"问题，证明两条令人惊讶的负面结果：(1) 即便底层 next-token 预测器是 (0,0)-CCA，自回归 rollout 也并非 CCA——CCA 不像 DP 那样在自回归下天然 compose；(2) 对一个已部署的非归因模型做 black-box "CCA retrofitting" 至少需要在输出长度 $\ell$ 上指数级查询次数。

**[Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path](circuit_fingerprints_how_answer_tokens_encode_their_geometrical_path.md)**

:   本文提出 Circuit Fingerprint 假说——单独把答案 token 喂进 Transformer，它在隐空间留下的方向恰好就是产生该答案所要走的电路路径——并据此用纯几何对齐（无需梯度/干预）完成 circuit discovery，同时同一组方向反过来可以做 activation steering，证明"读"和"写"是同一个几何对象的两面。

**[CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)**

:   通过把生成时 token 上的 SAE 激活与任务正确性做 Pearson 相关来挑选可解释的引导特征, 用正样本均值激活直接当系数, 不需对比数据集也不需反向传播, 就能在 Gemma-2 2B / LLaMA-3.1 8B 上把 MMLU 提 +3.3%、HarmBench 提 +27.1%, 且副作用率比微调更低。

**[Disentangling Direction and Magnitude in Transformer Representations: A Double Dissociation Through L2-Matched Perturbation Analysis](disentangling_direction_and_magnitude_in_transformer_representations_a_double_di.md)**

:   本文用 L2 匹配扰动协议，证明 Pythia 系列里方向（角度）扰动对语言建模 loss 的破坏力是同等位移幅值扰动的 42.9 倍，而幅值扰动对句法（主谓一致）的破坏远高于角度——这是一对认知神经科学意义上的 "双重分离"，对应方向走 attention 路径、幅值走 LayerNorm 路径。

**[Do Activation Verbalization Methods Convey Privileged Information?](do_activation_verbalization_methods_convey_privileged_information.md)**

:   本文系统证明：当前流行的激活语言化方法（Patchscopes / LIT / SelfIE）在被用作 LLM 可解释性工具时，其性能完全可以由 "verbalizer 模型自己的知识" 解释，不需要任何 target 模型的内部激活——意味着这些工具在现有 benchmark 上看起来 work 是因为基准本身设计有缺陷，且当 verbalizer 知识超过 target 时会编造出 target 根本不具备的 "解释"。

**[SemGrad: Gradients w.r.t. Semantics-Preserving Embeddings Tell LLM Uncertainty](gradients_with_respect_to_semantics_preserving_embeddings_tell_the_uncertainty_o.md)**

:   SemGrad 首次把"基于梯度"的不确定性量化搬到 LLM 自由生成场景——用语义保留分 (SPS) 找到能编码输入语义的隐藏态，把对它们求出的对数似然梯度范数当作 LLM 自信度的度量，无需采样、单次反向即可在 3 个 QA 数据集上击败 11 个 SOTA baseline，特别在多有效答案的 TruthfulQA 上比 SAR 高 3.27 AUROC。

**[Grokking: From Abstraction to Intelligence](grokking_from_abstraction_to_intelligence.md)**

:   本文从结构简化（奥卡姆剃刀）的视角统一解释 grokking 现象：训练过程中模型经历因果中介度退化、流形坍缩到 $\mathbb{Z}_{97}$ 圆环、谱能量向稀疏 Fourier 模集中、BDM 算法复杂度急剧下降这四种同步发生的"内部凝聚"，并用一个可解析的奇异特征机（SFM）证明这等价于自由能驱动的相变。

**[Interpretability Can Be Actionable](interpretability_can_be_actionable.md)**

:   这是一篇立场论文，主张「可解释性研究缺的不是新方法、而是评估准则」：研究该以 actionability（insight 能否驱动可解释性领域之外的具体决策/干预）为核心评估维度，作者沿 concreteness + validation 两个维度定义 actionability、分析阻碍、列出 5 个有杠杆的应用域、给出研究者 6 步 checklist。

**[Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models](is_one_layer_enough_understanding_inference_dynamics_in_tabular_foundation_model.md)**

:   作者对 6 个主流表格基础模型 (TFM) 做了首个大规模分层机理分析，发现中后层主要在做"迭代精化"且存在大量冗余，并据此设计了一个只用 20% 参数的单层循环 TFM，性能几乎追平六层原版。

**[Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](manifold-aligned_guided_integrated_gradients_for_reliable_feature_attribution.md)**

:   本文提出 MA-GIG：把 Guided IG 的“按低梯度幅值选特征再走一步”策略从像素空间搬到预训练 VAE 的潜在空间，借助 decoder Jacobian 把潜空间内的轴对齐更新映射成数据流形切空间上的更新，从而既避开高梯度噪声区域，又让积分路径上的样本始终贴近真实数据流形，归因更可靠。

**[Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping](memory_as_a_markov_matrix_sample_efficient_knowledge_expansion_via_token-to-dict.md)**

:   把自回归 LLM 的下一个 token 分布解释成一条 Markov 链的状态转移矩阵，于是「学新词」就变成「在状态空间里加新状态、并把它表示为已有状态的稀疏组合」，理论上只需 $O(s)$ 样本（$s$ 为映射到的旧 token 数），实践中只 finetune 新 token 的 embedding 即可在严格零遗忘下完成跨语种/新概念扩展。

**[Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)**

:   本文在高维线性回归 ICL 框架下，用一种保留 softmax 归一化与温度选择性、又解析可解的"近似 softmax 注意力"，**给出 ICL 泛化误差的闭式解和最优 attention temperature 的显式表达式** $\tau_{\text{opt}}$，证明只要调对推理时温度就能恢复近 Bayes 最优表现；在 GPT-2、Llama2-7B 的真实 QA 中也验证了这把"轻量旋钮"的有效性。

**[Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks](probabilistic_modeling_of_latent_agentic_substructures_in_deep_neural_networks.md)**

:   作者把神经网络（特别是 LLM）形式化为多个隐式子代理（每个是 outcome 上的概率分布）通过对数加权池化合成的复合代理，并在认知效用 $W_i(o)=\log P_i(o)$ 框架下证明了 "严格一致受益（strict unanimity）" 在线性池化或二元 outcome 下不可能、但 $|\mathcal O|\ge 3$ 下可行，进而推出"显式让 Waluigi 先显形再压制"严格优于"只强化 Luigi"的对齐原则。

**[Provably Learning Attention with Queries](provably_learning_attention_with_queries.md)**

:   作者证明单头 softmax attention 在 value-query 访问下可以惊人简洁地被精确恢复 —— 只需 $O(d^2)$ 次查询，比同等结构的 ReLU MLP 容易得多；当头维 $r\ll d$ 时还能借压缩感知降到 $O(rd)$，并把结论扩展到带噪 oracle、membership query 以及多头不可识别性。

**[Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)**

:   本文把 "prompt steering"重新解释为 LLM 自己实现的一种 activation steering, 然后用一个**逐 token 的 ReLU 探针**来蒸馏 prompt 注入的激活差, 训练出 PSR (Prompt Steering Replacement) 模块, 既能在三个 steering 基准上超过现有激活引导方法 (CAA, ReFT-R1, Stolfo 等), 又在 AxBench 与人格引导上和 prompting 打成平甚至反超。

**[The Cylindrical Representation Hypothesis for Language Model Steering](the_cylindrical_representation_hypothesis_for_language_model_steering.md)**

:   本文提出 Cylindrical Representation Hypothesis（CRH），在保留"概念线性"的前提下放弃 LRH 的正交性，证明概念向量的叠加会自然诱导出"轴 + 法平面 + 敏感扇区"的圆柱几何，从而首次几何化地解释了 activation steering 为什么在样本层面不可预测但在群体层面可观测。

**[The Structural Origin of Attention Sink: Variance Discrepancy, Super Neurons, and Dimension Disparity](the_structural_origin_of_attention_sink_variance_discrepancy_super_neurons_and_d.md)**

:   本文揭示 LLM 中"注意力汇聚到第一个 token"的结构性根源 —— 因果掩码下首 token 缺乏 value 聚合导致维度方差差异,被 FFN 中的 super neurons 选择性放大形成维度极度悬殊,最终锁死 QK 投影迫使形成 attention sink;并据此提出 head-wise RMSNorm 在预训练阶段从根上抑制 sink。

**[Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions](towards_steering_without_sacrifice_principled_training_of_steering_vectors_for_p.md)**

:   作者用神经网络无穷宽缩放理论推出 steering vector 的 factor / direction 联合训练应满足 $\eta_{\mathbf{v}}\eta_{\alpha}=\Theta(1)$ 这一缩放约束，从而消掉推理时人工选 $\alpha$ 的环节；同时受 ReFT 启发只在前 4 个 prompt token 上做加性干预（PrOSV），在 AxBench 上既能维持模型实用性，又能在三档 Gemma2/Qwen2.5 模型上一致超过全序列 FSSV。

**[Understanding LoRA as Knowledge Memory: An Empirical Analysis](understanding_lora_as_knowledge_memory_an_empirical_analysis.md)**

:   作者用 PhoneBook 与新构造的 PaperQA 基准做系统实证审计，把 LoRA 看作可独立训练 / 加载 / 组合的知识记忆单元，定量给出"秩 → 容量 → 效率 → 多模块组合 → 与 RAG/ICL 互补"全链路的设计准则。

**[Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints](why_linear_interpretability_works_invariant_subspaces_as_a_result_of_architectur.md)**

:   本文给出"为什么 transformer 的内部表征可以被简单线性方法（probe、SAE、activation steering）反复成功解码"的架构级解释：只要语义特征是通过 OV 电路或 unembedding 这类**线性接口**被读出的，它就必须落在一个跨上下文不变的线性子空间里（Invariant Subspace Necessity 定理）；并推出一个零样本应用——Self-Reference Property，即 token 本身的嵌入方向就是其概念方向，从而可以无监督地用 class token 的几何位置直接做分类。
