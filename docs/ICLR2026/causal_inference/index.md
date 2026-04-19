---
title: >-
  ICLR2026 因果推理方向 18篇论文解读
description: >-
  18篇ICLR2026 因果推理方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🔬 ICLR2026** · **18** 篇论文解读

**[Action-Guided Attention for Video Action Anticipation](action-guided_attention_for_video_action_anticipation.md)**

:   提出动作引导注意力 (AGA) 机制，用模型自身的动作预测序列作为注意力的 Query 和 Key（而非像素特征），结合自适应门控融合历史上下文和当前帧特征，在 EPIC-Kitchens-100 上实现从验证集到测试集的良好泛化，同时支持训练后的可解释性分析。

**[AgentTrace: Causal Graph Tracing for Root Cause Analysis in Deployed Multi-Agent Systems](agenttrace_causal_graph_tracing_for_root_cause_analysis_in_deployed_multi-agent_.md)**

:   提出AgentTrace框架，从多智能体系统的执行日志中构建因果图，通过反向追踪+轻量级特征排序（五组特征的加权线性组合）定位根因节点，在550个合成故障场景上Hit@1达94.9%，延迟0.12秒，比LLM分析快69倍。

**[Copy-Paste to Mitigate Large Language Model Hallucinations](copy-paste_to_mitigate_large_language_model_hallucinations.md)**

:   提出 Copy-Paste 生成范式，通过训练 LLM 优先直接复制检索上下文中的片段来生成回答，而非自由改写，配合高复制偏好的 DPO 训练，在反事实 RAG 基准上将忠实度从 80.2% 提升到 92.8%。

**[Counterfactual Explanations on Robust Perceptual Geodesics](counterfactual_explanations_on_robust_perceptual_geodesics.md)**

:   提出 PCG（Perceptual Counterfactual Geodesic）方法，在鲁棒感知流形上通过测地线优化生成语义忠实的反事实解释，两阶段优化确保路径既感知自然又达到目标类别，在 AFHQ 上 FID=8.3 远优于 RSGD 的 12.9。

**[Direct Doubly Robust Estimation of Conditional Quantile Contrasts](direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)**

:   提出首个对条件分位数比较器 (CQC) 的**直接估计方法**，通过显式参数化 CQC 并结合双重鲁棒梯度下降，在理论上保持双重鲁棒性的同时，实验中在估计精度、可解释性和计算效率上全面优于现有的间接反演方法。

**[Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)**

:   首次在线性非高斯设定下、不依赖任何结构假设，给出了含潜变量和环的因果图之间分布等价性的完整图准则，核心工具是新提出的**边秩约束**（edge rank constraints），据此开发了遍历等价类和从数据恢复因果模型的算法——这是参数化因果模型中首个无结构假设的等价性刻画和发现方法。

**[Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)**

:   提出 E-CIT（集成条件独立性检验）框架，通过将数据分割为子集后独立执行检验并基于**稳定分布**的 p 值聚合方法合并结果，将任意条件独立性检验的计算复杂度降至关于样本量线性，同时在重尾噪声和真实数据等复杂场景下保持甚至提升检验功效。

**[Flattery, Fluff, and Fog: Diagnosing and Mitigating Idiosyncratic Biases in Preference Models](flattery_fluff_and_fog_diagnosing_and_mitigating_idiosyncratic_biases_in_prefere.md)**

:   系统研究偏好模型对五种表面特征（冗长、结构化、术语、谄媚、模糊）的过度依赖——通过因果反事实对量化偏差来源于训练数据的分布不平衡，并提出基于**反事实数据增强 (CDA)** 的后训练方法，将模型与人类判断的平均失校准率从 39.4% 降至 32.5%。

**[Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)**

:   通过 off-by-one addition（如 1+1=3, 2+2=5）这一反事实任务，利用 path patching 发现大语言模型内部存在 **function induction** 机制——一种超越 token 级别 pattern matching、在函数级别进行归纳推理的注意力头电路，并证明该机制可跨任务复用。

**[Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference](journey_to_the_centre_of_cluster_harnessing_interior_nodes_for_ab_testing_under_.md)**

:   针对网络干扰下 A/B 测试中 GATE 估计的高方差问题，提出 Mean-in-Interior (MII) 估计器——仅对 cluster 内部节点取均值，大幅降低方差；再通过反事实预测器进行协变量偏移校正，得到增广版 AMII 估计器，同时实现低偏差和低方差。

**[Learning Robust Intervention Representations with Delta Embeddings](learning_robust_intervention_representations_with_delta_embeddings.md)**

:   提出因果 Delta 嵌入（CDE）框架，将干预/动作表示为预干预和后干预状态在潜空间中的向量差，通过独立性、稀疏性和不变性三种约束学习鲁棒的干预表示，在 Causal Triplet 挑战中显著超越基线的 OOD 泛化性能，且能自动发现反义动作的反平行语义结构。

**[On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)**

:   提出基于结构因果模型（SCM）的分解式评估框架，将 LLM 的反事实推理拆分为四个阶段（因果变量识别→因果图构建→干预识别→结果推理），在 11 个多模态数据集上系统诊断 LLM 在各阶段的能力瓶颈，并提出工具增强和高级 elicitation 策略来改善性能。

**[PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits](personax_multimodal_datasets_with_llm-inferred_behavior_traits.md)**

:   构建了 PersonaX 多模态数据集（含 LLM 推断的 Big Five 行为特质、面部嵌入和传记元数据），并提出两层分析框架：结构化独立性检验 + 非结构化因果表示学习（带可识别性理论保证），揭示跨模态因果结构。

**[Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement](resisting_contextual_interference_in_rag_via_parametric-knowledge_reinforcement.md)**

:   提出 Knowledgeable-R1，一个基于强化学习的框架，通过联合采样参数知识（PK）和上下文知识（CK）的轨迹，结合局部/全局优势计算和自适应不对称优势变换，使 LLM 在 RAG 场景中能够抵抗误导性检索上下文的干扰，同时保留对可靠上下文的利用能力。

**[Self-Supervised Learning from Structural Invariance](self-supervised_learning_from_structural_invariance.md)**

:   提出 AdaSSL，通过引入潜变量建模正样本对之间的条件不确定性，推导出互信息的变分下界，使 SSL 能够处理自然配对数据中的复杂（多模态、异方差）条件分布，在因果表征学习、细粒度图像理解和视频世界模型上均优于基线。

**[SelfReflect: Can LLMs Communicate Their Internal Answer Distribution?](selfreflect_can_llms_communicate_their_internal_answer_distribution.md)**

:   提出SelfReflect度量指标——一个衡量LLM自述不确定性摘要与其真实内部答案分布之间差异的信息论距离，发现现代LLM普遍无法自主反映内部不确定性，但通过采样多个输出并反馈到上下文中可以生成忠实的不确定性摘要。

**[Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders](synthesising_counterfactual_explanations_via_label-conditional_gaussian_mixture_.md)**

:   提出 L-GMVAE（标签条件高斯混合 VAE）和 LAPACE 算法，通过在潜空间中学习每个类别的多个高斯聚类中心，然后从输入潜表征到目标类别中心进行线性插值，生成路径式反事实解释，同时保证有效性、似合性、多样性和对输入扰动的完美鲁棒性。

**[Validating Interpretability in siRNA Efficacy Prediction: A Perturbation-Based, Dataset-Aware Protocol](validating_interpretability_in_sirna_efficacy_prediction_a_perturbation-based_da.md)**

:   提出一个标准化的扰动式显著性忠实性验证协议用于 siRNA 效能预测，作为"合成前关卡"检验显著性图是否可信；同时提出 BioPrior 生物信息正则化提升解释忠实性，发现 19/20 折instances 通过验证，但跨数据集迁移暴露两种失败模式。
