---
title: >-
  CVPR2026 可解释性方向26篇论文解读
description: >-
  26篇CVPR2026的可解释性方向论文解读，涵盖多模态、布局/合成、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "可解释性"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "布局/合成"
  - "推理"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**📷 CVPR2026** · **26** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (72)](../../ICML2026/interpretability/index.md) · [💬 ACL2026 (63)](../../ACL2026/interpretability/index.md) · [🔬 ICLR2026 (60)](../../ICLR2026/interpretability/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/interpretability/index.md) · [🧠 NeurIPS2025 (81)](../../NeurIPS2025/interpretability/index.md) · [📹 ICCV2025 (10)](../../ICCV2025/interpretability/index.md)

🔥 **高频主题：** 多模态 ×3 · 布局/合成 ×2 · 推理 ×2

**[Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)**

:   提出"信息范围"（information scope）作为SAE特征可解释性的新维度，通过Contextual Dependency Score（CDS）将CLIP的SAE特征分为局部特征（低CDS）和全局特征（高CDS），揭示两类特征在分类、分割、深度估计中的差异化功能角色。

**[CI-ICE: Intrinsic Concept Extraction Based on Compositional Interpretability](ciice_intrinsic_concept_extraction_compositional.md)**

:   提出CI-ICE新任务和HyperExpress方法：在双曲空间(Poincaré球)中利用层次建模能力提取可组合的物体级/属性级内在概念，通过Horosphere投影保证概念嵌入空间的可组合性，在UCEBench上概念解耦ACC₁达0.504(较ICE的0.325提升55%)。

**[U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models](clipfree_label_free_unsupervised_concept_bottlenec.md)**

:   提出TextUnlock方法，通过训练轻量MLP将任意冻结视觉分类器的特征投射到文本嵌入空间（同时保持原分类器分布不变），无需CLIP、无需标注、无需训练线性探针，即可将任何legacy分类器转化为可解释的概念瓶颈模型——在40+架构上测试，超越甚至有监督的CLIP基CBM。

**[Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events](cut_to_the_chase_training-free_multimodal_summarization_via_chain-of-events.md)**

:   提出 CoE，一个免训练的多模态摘要框架，通过构建层次事件图（HEG）引导链式事件推理，在8个数据集上超越SOTA视频CoT基线，平均提升 +3.04 ROUGE、+9.51 CIDEr、+1.88 BERTScore。

**[DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)**

:   提出 DINO-QPM，一种轻量级可解释性适配器，将冻结的 DINOv2 骨干网络的复杂高维特征转换为对比性的、类无关的可解释表示，通过二次规划进行稀疏特征选择和类级特征分配，在 CUB-2011 和 Stanford Cars 上同时超越了 DINOv2 线性探测的准确率和所有可比方法的可解释性。

**[Draft and Refine with Visual Experts](draft_and_refine_with_visual_experts.md)**

:   提出 DnR（Draft and Refine），一个基于问题条件视觉利用度（Visual Utilization）指标的 Agent 框架，量化 LVLM 对视觉证据的实际依赖程度，并通过外部视觉专家（检测/分割/OCR等）的渲染反馈迭代改善视觉定位，减少幻觉。

**[Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing](edit-as-act_goal-regressive_planning_for_open-vocabulary_3d_indoor_scene_editing.md)**

:   将开放词汇的3D室内场景编辑重新定义为目标回归规划问题，设计PDDL风格的EditLang符号语言，通过LLM驱动的Planner-Validator循环从目标状态逆向推导最小编辑序列，在63个编辑任务上同时实现指令忠实度（69.1%）、语义一致性（86.6%）和物理合理性（91.7%）三个指标的最佳平衡。

**[ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](ermoe_eigen-reparameterized_mixture-of-experts_for_stable_routing.md)**

:   ERMoE 提出在正交特征基（eigenbasis）中重参数化MoE专家权重，并用特征基分数（cosine similarity）替代传统路由logits，无需辅助负载均衡损失即可实现稳定路由和可解释的专家特化。

**[Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)**

:   提出 FASS 基准，通过强制预测不变性过滤、三轴稳定性分解（空间/排序/显著区域）和多类型扰动（几何/光度/压缩），系统评估后验特征归因方法的稳定性，揭示了现有评估体系的根本性缺陷。

**[From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition](from_weights_to_concepts_data-free_interpretability_of_clip_via_singular_vector_.md)**

:   本文提出 SITH（Semantic Inspection of Transformer Heads），一个完全无需数据和训练的 CLIP 可解释性框架：直接对注意力头的 Value-Output 权重矩阵做 SVD 分解，然后用自研的 COMP 算法将每个奇异向量解释为语义一致的概念稀疏组合，实现了比现有方法更细粒度的 intra-head 级别可解释性，并支持精准的权重编辑来改善下游性能。

**[Geometry-Guided Camera Motion Understanding in VideoLLMs](geometry-guided_camera_motion_understanding_in_videollms.md)**

:   本文揭示了 VideoLLM 在细粒度相机运动原语（pan/tilt/dolly等）识别上几乎等于随机猜测，构建了 CameraMotionDataset（12K 段 × 15 种原子运动）和 CameraMotionVQA benchmark，并提出通过冻结 3DFM（VGGT）提取几何相机线索 + 轻量时序分类器 + structured prompting 注入的 model-agnostic 方案来弥补这一能力缺口。

**[Inside-Out: Measuring Generalization in Vision Transformers Through Inner Workings](inside-out_measuring_generalization_in_vision_transformers_through_inner_working.md)**

:   提出基于模型内部电路（circuits）的泛化性能预测指标，包括部署前模型选择的Dependency Depth Bias（DDB）和部署后性能监控的Circuit Shift Score（CSS），分别比现有代理指标的相关性平均提升13.4%和34.1%。

**[Language Models Can Explain Visual Features via Steering](language_models_can_explain_visual_features_via_steering.md)**

:   提出通过对VLM视觉编码器进行SAE特征因果干预（steering），在输入空白图像后让语言模型描述其"看到"的视觉概念，从而实现无需评估图像集的可扩展视觉特征自动解释，并提出混合方法Steering-informed Top-k达到SOTA。

**[Measuring the (Un)Faithfulness of Concept-Based Explanations](measuring_the_unfaithfulness_of_concept-based_explanations.md)**

:   本文揭示了现有无监督概念解释方法 (U-CBEMs) 的忠实度被高估——原因是使用了过于复杂的代理模型和有缺陷的删除式评估。作者提出 SURF（Surrogate Faithfulness），一个简单的线性代理 + 双空间度量框架，通过"随机概念应该更不忠实"的 sanity check 验证了其正确性，并首次系统地揭示了多个 SOTA U-CBEMs 实际上并不忠实。

**[Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared](missing_no_more_dictionary-guided_cross-modal_image_fusion_under_missing_infrare.md)**

:   提出首个在系数域（而非像素域）进行红外缺失条件下跨模态融合的框架：通过共享卷积字典建立 IR-VIS 统一原子空间，在系数域完成 VIS→IR 推理和自适应融合，配合冻结 LLM 提供弱语义先验进行热信息补全，在仅输入可见光图像的条件下达到接近双模态融合方法的性能。

**[Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion](neurodynamics-driven_coupled_neural_p_systems_for_multi-focus_image_fusion.md)**

:   提出 ND-CNPFuse，通过对耦合神经 P (CNP) 系统进行神经动力学分析，建立网络参数与输入信号的约束关系以避免神经元异常持续放电，从而在多焦点图像融合 (MFIF) 任务上无需训练即可生成高质量、可解释的决策图。

**[On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)**

:   揭示主流 image-in-image 深度隐写方案的根本安全缺陷：嵌入过程本质上是一个混合过程，可被独立成分分析 (ICA) 轻易分离，并提出基于小波域独立成分统计矩的可解释隐写分析方法（仅 8 维特征即达 84.6% 准确率），同时证明经典 SRM+SVM 方法可达 99% 以上检测率。

**[Pixel2Phys: Distilling Governing Laws from Visual Dynamics](pixel2phys_distilling_governing_laws_from_visual_dynamics.md)**

:   提出 Pixel2Phys，一个基于 MLLM 的多智能体协作框架，通过 Plan-Variable-Equation-Experiment 四个 Agent 的迭代假设-验证-精化循环，从原始视频中自动发现可解释的物理控制方程，外推精度比基线提升 45.35%。

**[Reallocating Attention Across Layers to Reduce Multimodal Hallucination](reallocating_attention_across_layers_to_reduce_multimodal_hallucination.md)**

:   提出一种轻量级、无需训练的插件方法，通过识别感知型和推理型注意力头并进行类别条件缩放（Class-Conditioned Rescaling），重新平衡跨层注意力分配，从而缓解多模态大推理模型（MLRM）中的幻觉问题，在5个基准上平均提升4.2%，几乎无额外推理开销。

**[Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)**

:   提出 CBM-Suite 框架，系统性解决概念瓶颈模型的四大缺陷——缺乏概念相关性预评估指标、线性问题导致概念瓶颈被绕过、与黑盒模型的精度差距、以及不同视觉骨干/VLM 影响的研究空白——通过熵度量、非线性层和蒸馏损失显著提升 CBM 的精度与可解释性。

**[RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation](riskprop_collision-anchored_self-supervised_risk_propagation_for_early_accident_.md)**

:   提出 RiskProp，一种以碰撞帧为锚点的自监督风险传播范式，通过未来帧正则化损失和自适应单调约束损失，仅依赖碰撞帧标注即可学习时序连贯的风险演化曲线，在 CAP 和 Nexar 数据集上达到 SOTA。

**[SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World](safedrive_fine-grained_safety_reasoning_for_end-to-end_driving_in_a_sparse_world.md)**

:   提出 SafeDrive 端到端规划框架，通过轨迹条件化的稀疏世界模型（SWNet）模拟关键实体的未来行为，再由细粒度推理网络（FRNet）进行逐实例碰撞评估和逐时刻可行驶区域合规评估，在 NAVSIM 上 PDMS 达 91.6、仅 0.5% 碰撞率，Bench2Drive 驾驶分 66.8%。

**[TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment](tdatr_improving_end-to-end_table_recognition_via_table_detail-aware_learning_and.md)**

:   提出TDATR框架，通过"先感知后融合"策略和结构引导的单元格定位模块，在有限标注数据下实现端到端表格识别，在7个基准上无需数据集特定微调即达到SOTA。

**[Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)**

:   提出f-CBM——首个忠实的多模态概念瓶颈模型框架，通过可微分泄漏损失减少概念表示中的非预期信息泄漏，同时用Kolmogorov-Arnold Network (KAN) 预测头提升概念检测精度，在任务准确率、概念检测和泄漏减少间取得最优Pareto前沿。

**[VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension](viro_robust_and_efficient_neuro-symbolic_reasoning_with_verification_for_referri.md)**

:   VIRO在神经符号REC管道中嵌入轻量算子级验证机制（CLIP不确定性验证+空间逻辑验证），使每个推理步骤能自我验证并在无目标时提前终止，在零样本设置下以61.1%平衡准确率大幅超越组合推理baselines，同时保持0.3%以下的程序失败率和高效推理速度。

**[Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)**

:   提出 I2X 框架，通过在训练检查点追踪从 GradCAM 提取的原型强度与模型置信度的协同演化，将非结构化的可解释性（显著性图）转化为结构化的可解释性，揭示模型"为什么关注那里"的推理结构，并利用这种理解指导微调提升性能。
