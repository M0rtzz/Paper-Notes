---
title: >-
  NeurIPS2025 因果推理方向 24篇论文解读
description: >-
  24篇NeurIPS2025 因果推理方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**🧠 NeurIPS2025** · **24** 篇论文解读

**[A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)**

:   提出基于多智能体影响图（MAIDs）的**目标干预范式（Targeted Intervention）**，通过仅对单个目标智能体施加**预策略干预（Pre-Strategy Intervention, PSI）**，引导整个多智能体系统收敛到满足额外期望结果的优选Nash均衡，无需对所有智能体进行全局干预。

**[Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)**

:   将LLM对齐建模为约束优化问题——最大化主要奖励（如有用性）同时满足次要效用约束（如安全性），提出基于拉格朗日对偶的迭代方法交替更新LLM策略和对偶变量，理论上刻画了分布空间与参数空间之间的原对偶间隙和最优性间隙，证明可以找到近最优约束LLM策略（误差仅来自参数化间隙）。

**[An Analysis of Causal Effect Estimation Using Outcome Invariant Data Augmentation](an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)**

:   首次系统分析"结果不变数据增强"（outcome invariant DA）在因果效应估计中的作用，证明当 DA 操作保持结果变量的不变性时等价于对处理变量的软干预，可减少混杂偏差；进一步提出 IV-like（IVL）回归框架，将 DA 参数用作"类工具变量"，通过对抗性 DA 组合进一步降低偏差。

**[Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)**

:   提出 Bi-DFCL，通过双层优化框架联合利用观测数据和 RCT 实验数据来训练营销资源分配模型：上层用 RCT 数据的无偏决策损失端到端训练 Bridge Network 来动态纠正下层在观测数据上的偏差，同时设计了基于原始问题的可微代理决策损失（PPL/PIFD）和隐式微分算法，解决了传统两阶段方法的预测-决策不一致和偏差-方差困境。已在美团大规模在线部署。

**[Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](causality-induced_positional_encoding_for_transformer-based_representation_learn.md)**

:   CAPE 通过从表格数据中学习特征间的因果DAG结构，将其嵌入双曲空间生成因果感知的旋转位置编码（RoPE），使 Transformer 能处理非序列但因果相关的特征数据，在多组学数据的下游任务上显著提升性能。

**[Characterization and Learning of Causal Graphs from Hard Interventions](characterization_and_learning_of_causal_graphs_from_hard_interventions.md)**

:   首次系统分析硬干预（hard interventions）在含隐变量因果发现中的理论优势，提出广义do-演算（4条规则）和孪生增强MAG图表示，给出 $\mathcal{I}$-Markov 等价类的充要图条件，并设计可证明正确的FCI变体学习算法；实验表明硬干预比软干预将等价类缩小37-57%。

**[Conformal Prediction for Causal Effects of Continuous Treatments](conformal_prediction_for_causal_effects_of_continuous_treatments.md)**

:   首次为连续处理变量（如药物剂量）的因果效应构建共形预测区间，通过倾向性偏移参数化和分位数回归，在已知/未知倾向性两种场景下均提供有限样本 $1-\alpha$ 覆盖保证。

**[Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)**

:   提出COUPLE框架，通过构建结构因果模型（SCM）建模多维价值观的依赖关系与优先级，并利用反事实推理实现LLM对任意细粒度多元价值目标的可控对齐。

**[Cyclic Counterfactuals under Shift–Scale Interventions](cyclic_counterfactuals_under_shift-scale_interventions.md)**

:   本文在循环（非DAG）结构因果模型中建立了shift-scale软干预下反事实推理的理论框架，证明了全局收缩条件保证循环SCM的唯一可解性，并推导出反事实分布的sub-Gaussian集中不等式。

**[Demystifying Spectral Feature Learning for Instrumental Variable Regression](demystifying_spectral_feature_learning_for_instrumental_variable_regression.md)**

:   为基于谱特征的非参数工具变量（NPIV）回归建立严格的泛化误差界，揭示性能由结构函数与条件期望算子的**谱对齐**（近似误差）和**奇异值衰减速度**（估计误差）两因素共同决定，提出 Good-Bad-Ugly 三分类法并设计数据驱动诊断工具。

**[Differentiable Structure Learning and Causal Discovery for General Binary Data](differentiable_structure_learning_and_causal_discovery_for_general_binary_data.md)**

:   提出基于多元伯努利分布（MVB）的通用可微结构学习框架，不假设特定数据生成过程，能捕获二值离散变量间的任意高阶依赖关系，并证明在一般设定下DAG不可识别但可恢复最小等价类（Markov等价类）。

**[Do-PFN: In-Context Learning for Causal Effect Estimation](do-pfn_in-context_learning_for_causal_effect_estimation.md)**

:   提出 Do-PFN，将 Prior-data Fitted Networks (PFN) 扩展到因果效应估计，在大量合成 SCM 数据上预训练 Transformer 进行 in-context 因果推理，仅需观测数据即可预测干预分布（CID）和 CATE，无需因果图知识或不混杂假设，在合成和半合成实验中表现出色。

**[Domain-Adapted Granger Causality for Real-Time Cross-Slice Attack Attribution in 6G Networks](domain-adapted_granger_causality_for_real-time_cross-slice_attack_attribution_in.md)**

:   提出一种面向6G网络切片的域适应Granger因果框架，将增强型Granger因果检验与网络资源争用建模相结合，实现实时跨切片攻击归因，在1100个攻击场景上达到89.2%准确率和87ms响应时间，显著超越现有统计、深度学习和因果发现方法。

**[Dynamic Causal Discovery in Alzheimer's Disease through Latent Pseudotime Modelling](dynamic_causal_discovery_in_alzheimers_disease_through_latent_pseudotime_modelli.md)**

:   将 BN-LTE（贝叶斯网络+潜在时间嵌入）应用于 ADNI 真实 AD 数据，推断随疾病伪时间演变的动态因果图，伪时间预测诊断 AUC 0.82 远超年龄 0.59，并揭示了新型生物标志物 NfL/GFAP 与传统 AD 标志物之间的动态因果关系。

**[Few-Shot Knowledge Distillation of LLMs With Counterfactual Explanations](few-shot_knowledge_distillation_of_llms_with_counterfactual_explanations.md)**

:   提出 CoD（Counterfactual-explanation-infused Distillation），通过将反事实解释注入少样本训练集来精确映射 teacher 决策边界，在 6 个数据集上仅用 8–512 样本即显著超越标准蒸馏方法。

**[From Black-box to Causal-box: Towards Building More Interpretable Models](from_black-box_to_causal-box_towards_building_more_interpretable_models.md)**

:   提出"因果可解释性"（causal interpretability）的形式化定义，证明黑盒模型和概念瓶颈模型均不满足该性质，给出完整的图判据确定哪些模型架构能一致地回答反事实问题，揭示了因果可解释性与预测精度之间的根本性权衡。

**[GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)**

:   提出 GST-UNet，将 U-Net 时空编码器与迭代 G-computation 相结合，从**单条时空观测轨迹**中估计位置特异性的条件平均潜在结果 (CAPO)，可同时处理干扰（interference）、空间混杂、时间延续和时变混杂，并在加州山火烟雾对呼吸系统住院率的因果分析中验证了实用价值。

**[It's Hard to Be Normal: The Impact of Noise on Structure-agnostic Estimation](its_hard_to_be_normal_the_impact_of_noise_on_structure-agnostic_estimation.md)**

:   证明 Double Machine Learning (DML) 在高斯处理噪声下是极小极大最优的（$O(\epsilon^2 + n^{-1/2})$），但在非高斯噪声下变得次优；提出 Agnostic Cumulant-based Estimation (ACE) 利用高阶累积量达到 $r$ 阶不敏感性 $O(\epsilon^r + n^{-1/2})$。

**[LLM Interpretability with Identifiable Temporal-Instantaneous Representation](llm_interpretability_with_identifiable_temporal-instantaneous_representation.md)**

:   本文提出了一种面向 LLM 高维激活空间的可辨识时序因果表示学习框架，通过线性化公式同时建模时间延迟和瞬时因果关系，在保留理论可辨识性保证的同时解决了现有 CRL 方法无法扩展到 LLM 维度的计算瓶颈。

**[Performative Validity of Recourse Explanations](performative_validity_of_recourse_explanations.md)**

:   本文形式化分析了追索权解释（recourse explanations）的"表演性"效应——当大量被拒申请者按照追索建议行动时，集体行为会引发数据分布偏移并使模型更新后追索失效，并证明了只有基于因果变量的改进型追索（ICR）才能在广泛条件下保持"表演性有效性"。

**[Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)**

:   提出 Estimand-Agnostic（EA）方法和 Frontier-Reducibility Algorithm（FRA）来高效计算因果 Shapley 值（do-SV），通过训练单个 SCM 学习观测分布即可回答任意可辨识的因果查询，并通过联盟约减将计算量降低约 90%。

**[Revealing Multimodal Causality with Large Language Models](revealing_multimodal_causality_with_large_language_models.md)**

:   提出 MLLM-CD，首个面向多模态非结构化数据的因果发现框架，通过对比因子发现识别跨模态因果变量，结合统计因果结构推断，并利用 MLLM 的世界知识生成多模态反事实样本来迭代消除结构歧义，在合成和真实数据集上均显著优于现有方法。

**[Root Cause Analysis of Outliers with Missing Structural Knowledge](root_cause_analysis_of_outliers_with_missing_structural_knowledge.md)**

:   提出仅用**边际异常分数**即可做根因分析的两个简单高效算法——已知因果图时用 SMOOTH TRAVERSAL（沿因果路径找分数跳变最大的节点），未知因果图时用 SCORE ORDERING（按分数排序取 top-k），在 polytree 结构下给出非参数概率保证，仅需单个异常样本即可工作。

**[Transferring Causal Effects using Proxies](transferring_causal_effects_using_proxies.md)**

:   提出基于代理变量（proxy）的多域因果效应迁移方法，在目标域仅观测到代理变量 W 的条件下，利用多源域数据识别并估计目标域中含未观测混淆因子的干预分布，给出两种一致性估计器及渐近置信区间。
