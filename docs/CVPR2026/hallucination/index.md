---
title: >-
  CVPR2026 幻觉检测论文汇总 · 19篇论文解读
description: >-
  19篇CVPR2026的幻觉检测方向论文解读，涵盖多模态、推理、对抗鲁棒、扩散模型、LLM、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "幻觉检测"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "推理"
  - "对抗鲁棒"
  - "扩散模型"
  - "LLM"
  - "对齐/RLHF"
item_list:
  - u: "beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations/"
    t: "Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations"
  - u: "fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_/"
    t: "Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression"
  - u: "finer_mllms_hallucinate_under_fine-grained_negative_queries/"
    t: "FINER: MLLMs Hallucinate under Fine-grained Negative Queries"
  - u: "hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc/"
    t: "HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models"
  - u: "hulluedit_subspace_editing_hallucination/"
    t: "HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs"
  - u: "kvsmooth_mitigating_hallucination_in_multi-modal_large_language_models_through_k/"
    t: "KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing"
  - u: "locate-then-sparsify_attribution_guided_sparse_strategy_for_visual_hallucination/"
    t: "Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation"
  - u: "mitigating_multimodal_hallucinations_via_gradient-based_self-reflection/"
    t: "Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection"
  - u: "mitigating_object_hallucinations_in_lvlms_via_attention_imbalance_rectification/"
    t: "Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification"
  - u: "mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit/"
    t: "MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization"
  - u: "overthinking_causes_hallucination_tracing_confounder_propagation_in_vision_langu/"
    t: "Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models"
  - u: "prefill-time_intervention_for_mitigating_hallucination_in_large_vision-language_/"
    t: "Prefill-Time Intervention for Mitigating Hallucination in Large Vision-Language Models"
  - u: "reallocating_attention_across_layers_to_reduce_multimodal_hallucination/"
    t: "Reallocating Attention Across Layers to Reduce Multimodal Hallucination"
  - u: "residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_/"
    t: "Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance"
  - u: "tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att/"
    t: "Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention"
  - u: "tridf_evaluating_perception_detection_and_hallucination_for_interpretable_deepfa/"
    t: "TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection"
  - u: "understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model/"
    t: "Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models"
  - u: "understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim/"
    t: "Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models"
  - u: "zina_multimodal_fine-grained_hallucination_detection_and_editing/"
    t: "Zina: Multimodal Fine-grained Hallucination Detection and Editing"
item_total: 19
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👻 幻觉检测

**📷 CVPR2026** · **19** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (19)](../../ICML2026/hallucination/index.md) · [💬 ACL2026 (27)](../../ACL2026/hallucination/index.md) · [🔬 ICLR2026 (9)](../../ICLR2026/hallucination/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/hallucination/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/hallucination/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/hallucination/index.md)

🔥 **高频主题：** 多模态 ×9 · 推理 ×2

**[Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations.md)**

:   提出基于 patch 级别的 LVLM 幻觉检测框架，发现幻觉 token 表现出弥散注意力模式和低语义对齐两个特征标志，据此设计注意力弥散分数（ADS）和跨模态接地一致性（CGC）两个轻量指标，检测准确率达 90%。

**[Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)**

:   提出 CIPHER，一种无需训练的测试时幻觉抑制方法：离线阶段用扩散模型生成反事实图像构建 OHC-25K 数据集，通过 SVD 提取视觉幻觉子空间；推理阶段将隐状态投影到该子空间的正交补空间，在不修改模型参数、不增加推理开销的前提下显著降低 LVLM 的视觉幻觉。

**[FINER: MLLMs Hallucinate under Fine-grained Negative Queries](finer_mllms_hallucinate_under_fine-grained_negative_queries.md)**

:   发现 MLLM 在细粒度负查询（涉及多个对象/属性/关系的查询中仅有一个细微错误）下幻觉率急剧上升，提出 FINER 基准和 FINER-Tuning 方法（基于 DPO），在 InternVL3.5-14B 上最高提升 24.2%。

**[HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)**

:   提出HulluEdit，一种单次前向、无参考模型的子空间编辑框架，通过将隐藏状态分解为正交的视觉证据子空间、冲突先验子空间和残差不确定性子空间，选择性抑制幻觉模式而不干扰视觉定位，在POPE和CHAIR基准上达到SOTA幻觉缓解效果。

**[HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](hulluedit_subspace_editing_hallucination.md)**

:   提出HulluEdit，一个单次推理、无参考模型的幻觉缓解框架，通过将隐藏状态正交分解为视觉证据子空间、冲突先验子空间和残差不确定性子空间，选择性抑制幻觉模式而不干扰视觉接地，在POPE和CHAIR上达到SOTA。

**[KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing](kvsmooth_mitigating_hallucination_in_multi-modal_large_language_models_through_k.md)**

:   提出KVSmooth，一种免训练的即插即用方法，通过注意力行熵引导的自适应指数移动平均（EMA）对KV-Cache进行平滑，有效抑制多模态大语言模型（MLLM）在解码过程中因sink token引发的语义漂移与幻觉生成，在LLaVA-1.5上将CHAIR_S从41.8降至18.2（降幅56%），同时F1从77.5提升至79.2。

**[Locate-then-Sparsify: Attribution Guided Sparse Strategy for Visual Hallucination Mitigation](locate-then-sparsify_attribution_guided_sparse_strategy_for_visual_hallucination.md)**

:   提出 LTS-FS（Locate-Then-Sparsify for Feature Steering）框架，通过因果干预归因方法定位幻觉相关层，并根据归因分数逐层稀疏地控制特征引导强度，在有效缓解 LVLM 幻觉的同时保持模型泛化能力。

**[Mitigating Multimodal Hallucinations via Gradient-based Self-Reflection](mitigating_multimodal_hallucinations_via_gradient-based_self-reflection.md)**

:   提出 GACD（Gradient-based Influence-Aware Constrained Decoding），利用一阶 Taylor 梯度估计每个 token 对输出的影响力，在推理阶段同时缓解文本-视觉偏差和共现偏差导致的多模态幻觉，无需辅助模型或微调。

**[Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification](mitigating_object_hallucinations_in_lvlms_via_attention_imbalance_rectification.md)**

:   提出注意力失衡（Attention Imbalance）概念来解释 LVLM 中的对象幻觉现象，并设计轻量级解码时干预方法 AIR，通过跨模态注意力重新分配和方差约束投影正则化矫正注意力失衡，在四个 LVLM 上将幻觉率最高降低 35.1%，同时提升通用能力最高达 15.9%。

**[MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization](mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit.md)**

:   提出 MoD-DPO（Modality-Decoupled DPO），通过不变性正则化、敏感性正则化和语言先验去偏三个机制解耦多模态 LLM 中各模态的贡献，有效缓解跨模态幻觉（如用听觉信息回答视觉问题），并推导出闭式最优策略。

**[Overthinking Causes Hallucination: Tracing Confounder Propagation in Vision Language Models](overthinking_causes_hallucination_tracing_confounder_propagation_in_vision_langu.md)**

:   揭示VLM幻觉的新机制——"过度思考"(overthinking)：模型在中间解码层产生过多竞争性物体假设，混杂因子沿层传播至最终预测引发幻觉；提出Overthinking Score量化层间假设多样性×不确定性，在MSCOCO上F1达78.9%，OOD AMBER上71.58%。

**[Prefill-Time Intervention for Mitigating Hallucination in Large Vision-Language Models](prefill-time_intervention_for_mitigating_hallucination_in_large_vision-language_.md)**

:   PTI 把缓解 LVLM 幻觉的 steering 干预从「逐 token 的解码阶段」前移到「只做一次的 prefill 阶段」，对初始 KV cache 施加模态感知、key/value 解耦的方向向量，从源头修正易致幻表征，在三个 LVLM、五个 benchmark 上超过现有解码期方法，且能与它们即插即用叠加。

**[Reallocating Attention Across Layers to Reduce Multimodal Hallucination](reallocating_attention_across_layers_to_reduce_multimodal_hallucination.md)**

:   提出一种轻量级、无需训练的插件方法，通过识别感知型和推理型注意力头并进行类别条件缩放（Class-Conditioned Rescaling），重新平衡跨层注意力分配，从而缓解多模态大推理模型（MLRM）中的幻觉问题，在5个基准上平均提升4.2%，几乎无额外推理开销。

**[Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)**

:   提出 Residual Decoding (ResDec)——一种训练免的即插即用解码策略，通过分析历史 token 的 logit 分布中的 U 型 JSD 模式发现语义锚定阶段，聚合该阶段的历史 logits 作为残差引导融入当前解码，以近乎零的额外推理开销有效抑制 LVLM 中的语言先验幻觉。

**[Tell Model Where to Look: Mitigating Hallucinations in MLLMs by Vision-Guided Attention](tell_model_where_to_look_mitigating_hallucinations_in_mllms_by_vision-guided_att.md)**

:   提出Vision-Guided Attention (VGA)，一种免训练的方法，通过利用视觉token的语义特征构建精确的视觉定位，引导模型注意力聚焦于相关视觉区域，有效缓解MLLM幻觉，且兼容FlashAttention。

**[TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection](tridf_evaluating_perception_detection_and_hallucination_for_interpretable_deepfa.md)**

:   提出TriDF——首个从感知 (Perception)、检测 (Detection) 和幻觉 (Hallucination) 三个维度综合评估可解释深度伪造检测的基准，包含55K高质量样本覆盖16种DeepFake类型和3种模态，揭示了准确感知是可靠检测的基础但幻觉会严重破坏决策的三方耦合关系。

**[Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)**

:   本文系统分析了多模态 CoT 模型中幻觉的成因，发现模型在凭联想自由发挥的推理步骤（论文称之为"发散思维" divergent thinking）中最易产生幻觉，并提出基于视觉熵的免训练检测+解码干预策略，在 Object HalBench 上将 CHAIRS 降低超过 30%，同时保持甚至提升通用推理能力。

**[Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)**

:   本文提出 Hallucination-as-Cue 分析框架，通过三种模态特定腐蚀策略（空白图像、随机图像、文本移除）系统研究 RL 后训练对多模态推理模型的真实作用机制，发现即使在 100% 腐蚀视觉输入下 GRPO 训练仍能显著提升推理性能，挑战了"RL 训练能有效利用视觉信息"的主流假设。

**[Zina: Multimodal Fine-grained Hallucination Detection and Editing](zina_multimodal_fine-grained_hallucination_detection_and_editing.md)**

:   Zina 提出了多模态细粒度幻觉检测与编辑任务，设计了两阶段系统（detector MLLM + reviewer MLLM）将 token 复制委托给确定性函数以简化模型负担，同时构建了 VisionHall 数据集（6.9K 人工标注 + 20K 图结构合成数据），在检测 F1 上超过 GPT-4o 达 15.8 个点。
