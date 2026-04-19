---
title: >-
  CVPR2026 可解释性方向 34篇论文解读
description: >-
  34篇CVPR2026 可解释性方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**📷 CVPR2026** · **34** 篇论文解读

**[Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)**

:   提出"信息范围"（information scope）作为SAE特征可解释性的新维度，通过Contextual Dependency Score（CDS）将CLIP的SAE特征分为局部特征（低CDS）和全局特征（高CDS），揭示两类特征在分类、分割、深度估计中的差异化功能角色。

**[Beyond the Fold: Quantifying Split-Level Noise and the Case for Leave-One-Dataset-Out AU Evaluation](beyond_the_fold_quantifying_split-level_noise_and_the_case_for_leave-one-dataset.md)**

:   揭示面部AU检测中被试独立交叉验证本身引入±0.065 F1的随机噪声（noise floor），许多声称的SOTA提升落入此噪声带内不可区分，并提出Leave-One-Dataset-Out（LODO）协议作为更稳定可靠的替代评估方案。

**[CI-ICE: Intrinsic Concept Extraction Based on Compositional Interpretability](ciice_intrinsic_concept_extraction_compositional.md)**

:   提出CI-ICE新任务和HyperExpress方法：在双曲空间(Poincaré球)中利用层次建模能力提取可组合的物体级/属性级内在概念，通过Horosphere投影保证概念嵌入空间的可组合性，在UCEBench上概念解耦ACC₁达0.504(较ICE的0.325提升55%)。

**[Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events](cut_to_the_chase_training-free_multimodal_summarization_via_chain-of-events.md)**

:   提出 CoE，一个免训练的多模态摘要框架，通过构建层次事件图（HEG）引导链式事件推理，在8个数据集上超越SOTA视频CoT基线，平均提升 +3.04 ROUGE、+9.51 CIDEr、+1.88 BERTScore。

**[DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)**

:   提出 DINO-QPM，一种轻量级可解释性适配器，将冻结的 DINOv2 骨干网络的复杂高维特征转换为对比性的、类无关的可解释表示，通过二次规划进行稀疏特征选择和类级特征分配，在 CUB-2011 和 Stanford Cars 上同时超越了 DINOv2 线性探测的准确率和所有可比方法的可解释性。

**[Draft and Refine with Visual Experts](draft_and_refine_with_visual_experts.md)**

:   提出 DnR（Draft and Refine），一个基于问题条件视觉利用度（Visual Utilization）指标的 Agent 框架，量化 LVLM 对视觉证据的实际依赖程度，并通过外部视觉专家（检测/分割/OCR等）的渲染反馈迭代改善视觉定位，减少幻觉。

**[Edit-As-Act: Goal-Regressive Planning for Open-Vocabulary 3D Indoor Scene Editing](edit-as-act_goal-regressive_planning_for_open-vocabulary_3d_indoor_scene_editing.md)**

:   将开放词汇的3D室内场景编辑重新定义为目标回归规划问题，设计PDDL风格的EditLang符号语言，通过LLM驱动的Planner-Validator循环从目标状态逆向推导最小编辑序列，在63个编辑任务上同时实现指令忠实度（69.1%）、语义一致性（86.6%）和物理合理性（91.7%）三个指标的最佳平衡。

**[EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](emoverse_a_mllms-driven_emotion_representation_dataset_for_interpretable_visual_.md)**

:   构建 EmoVerse——首个同时覆盖 CES（Mikels 8 类离散情感）和 DES（1024 维连续情感空间）的大规模可解释视觉情感数据集（219K+ 图像），提出 B-A-S（Background-Attribute-Subject）三元组知识图谱标注体系和 Annotation & Verification Pipeline（Gemini/GPT-4o + EmoViT + CoT Critic Agent），并基于 Qwen2.5-VL-3B 微调实现 1024 维 DES 投射与情感归因解释。

**[EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](emoverse_mllm_emotion_representation_dataset.md)**

:   提出 EmoVerse，一个219K规模的视觉情感数据集，通过知识图谱启发的Background-Attribute-Subject三元组实现词级和主体级情感归因，同时提供离散CES和连续1024维DES双情感标注，配合多阶段标注验证流水线和基于Qwen2.5-VL的可解释情感模型。

**[Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)**

:   提出 FASS 基准，通过强制预测不变性过滤、三轴稳定性分解（空间/排序/显著区域）和多类型扰动（几何/光度/压缩），系统评估后验特征归因方法的稳定性，揭示了现有评估体系的根本性缺陷。

**[FINER: MLLMs Hallucinate under Fine-grained Negative Queries](finer_mllms_hallucinate_under_fine-grained_negative_queries.md)**

:   发现 MLLM 在细粒度负查询（涉及多个对象/属性/关系的查询中仅有一个细微错误）下幻觉率急剧上升，提出 FINER 基准和 FINER-Tuning 方法（基于 DPO），在 InternVL3.5-14B 上最高提升 24.2%。

**[From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition](from_weights_to_concepts_data-free_interpretability_of_clip_via_singular_vector_.md)**

:   本文提出 SITH（Semantic Inspection of Transformer Heads），一个完全无需数据和训练的 CLIP 可解释性框架：直接对注意力头的 Value-Output 权重矩阵做 SVD 分解，然后用自研的 COMP 算法将每个奇异向量解释为语义一致的概念稀疏组合，实现了比现有方法更细粒度的 intra-head 级别可解释性，并支持精准的权重编辑来改善下游性能。

**[Geometry-Guided Camera Motion Understanding in VideoLLMs](geometry-guided_camera_motion_understanding_in_videollms.md)**

:   本文揭示了 VideoLLM 在细粒度相机运动原语（pan/tilt/dolly等）识别上几乎等于随机猜测，构建了 CameraMotionDataset（12K 段 × 15 种原子运动）和 CameraMotionVQA benchmark，并提出通过冻结 3DFM（VGGT）提取几何相机线索 + 轻量时序分类器 + structured prompting 注入的 model-agnostic 方案来弥补这一能力缺口。

**[Geometry-Guided Camera Motion Understanding in VideoLLMs](geometryguided_camera_motion_understanding_in_vide.md)**

:   通过 benchmarking-diagnosis-injection 框架系统揭示 VideoLLM 的相机运动盲区，并利用冻结 3DFM (VGGT) 提取几何线索 + 轻量时序分类器 + 结构化提示注入，无需微调即可显著提升 VideoLLM 的细粒度相机运动理解。

**[How to Take a Memorable Picture? Empowering Users with Actionable Feedback](how_to_take_a_memorable_picture_empowering_users_with_actionable_feedback.md)**

:   定义了记忆性反馈（MemFeed）新任务，提出 MemCoach——一种 training-free 的 MLLM 激活导向方法，通过教师-学生策略将记忆性感知知识注入模型激活空间，使 MLLM 能生成提升照片记忆性的自然语言可操作建议。

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

**[On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_imageinimage_steg.md)**

:   揭示基于可逆神经网络（INN）的"图像中藏图像"隐写方案存在根本性安全漏洞：嵌入过程本质上是可通过独立成分分析（ICA）识别的混合过程，仅用8维统计特征+SVM即可达84.6%检测率，经典SRM+SVM更是达到99%以上。

**[Pixel2Phys: Distilling Governing Laws from Visual Dynamics](pixel2phys_distilling_governing_laws_from_visual_dynamics.md)**

:   提出 Pixel2Phys，一个基于 MLLM 的多智能体协作框架，通过 Plan-Variable-Equation-Experiment 四个 Agent 的迭代假设-验证-精化循环，从原始视频中自动发现可解释的物理控制方程，外推精度比基线提升 45.35%。

**[Reallocating Attention Across Layers to Reduce Multimodal Hallucination](reallocating_attention_across_layers_to_reduce_multimodal_hallucination.md)**

:   提出一种轻量级、无需训练的插件方法，通过识别感知型和推理型注意力头并进行类别条件缩放（Class-Conditioned Rescaling），重新平衡跨层注意力分配，从而缓解多模态大推理模型（MLRM）中的幻觉问题，在5个基准上平均提升4.2%，几乎无额外推理开销。

**[Reallocating Attention Across Layers to Reduce Multimodal Hallucination](reallocating_attention_reduce_hallucination.md)**

:   将多模态推理模型幻觉分解为浅层的感知偏差和深层的推理漂移两种失效模式，通过识别感知/推理功能头并选择性放大其贡献，以即插即用、无需训练的方式平均提升4.2%准确率，仅增加约1%计算开销。

**[Recursive Think-Answer Process for LLMs and VLMs](recursive_think-answer_process_for_llms_and_vlms.md)**

:   R-TAP 提出一种递归思考-回答过程，通过置信度生成器评估模型回答确定性并引导迭代推理修正，配合递归置信度增长奖励和最终答案置信度奖励的双重强化信号，在 LLM 和 VLM 上均一致超越单次推理方法，同时显著减少推理中的"Oops!"式自我反思表达。

**[SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World](safedrive_fine-grained_safety_reasoning_for_end-to-end_driving_in_a_sparse_world.md)**

:   提出 SafeDrive 端到端规划框架，通过轨迹条件化的稀疏世界模型（SWNet）模拟关键实体的未来行为，再由细粒度推理网络（FRNet）进行逐实例碰撞评估和逐时刻可行驶区域合规评估，在 NAVSIM 上 PDMS 达 91.6、仅 0.5% 碰撞率，Bench2Drive 驾驶分 66.8%。

**[SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)**

:   SubspaceAD 证明了在强视觉基础模型（DINOv2-G）特征上做一次 PCA 拟合就足以超越所有需要训练/记忆库/提示调优的少样本异常检测方法，1-shot 下在 MVTec-AD 上达 98.0% 图像级 AUROC 和 97.6% 像素级 AUROC。

**[TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment](tdatr_improving_end-to-end_table_recognition_via_table_detail-aware_learning_and.md)**

:   提出TDATR框架，通过"先感知后融合"策略和结构引导的单元格定位模块，在有限标注数据下实现端到端表格识别，在7个基准上无需数据集特定微调即达到SOTA。

**[Text-guided Fine-Grained Video Anomaly Understanding](text-guided_fine-grained_video_anomaly_understanding.md)**

:   提出T-VAU框架，通过异常热力图解码器(AHD)实现像素级时空异常定位，并设计区域感知异常编码器(RAE)将热力图证据注入LVLM进行异常判断、定位和语义解释的统一推理。

**[Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)**

:   提出f-CBM——首个忠实的多模态概念瓶颈模型框架，通过可微分泄漏损失减少概念表示中的非预期信息泄漏，同时用Kolmogorov-Arnold Network (KAN) 预测头提升概念检测精度，在任务准确率、概念检测和泄漏减少间取得最优Pareto前沿。

**[VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension](viro_robust_and_efficient_neuro-symbolic_reasoning_with_verification_for_referri.md)**

:   VIRO在神经符号REC管道中嵌入轻量算子级验证机制（CLIP不确定性验证+空间逻辑验证），使每个推理步骤能自我验证并在无目标时提前终止，在零样本设置下以61.1%平衡准确率大幅超越组合推理baselines，同时保持0.3%以下的程序失败率和高效推理速度。

**[Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](where_mllms_attend_and_what_they_rely_on_explaining_autoregressive_token_generat.md)**

:   提出Eagle，一个轻量级黑盒归因框架，通过insight score（充分性）和necessity score（不可或缺性）的统一目标函数对MLLM的自回归token生成进行空间归因，并量化每个token依赖语言先验还是感知证据，在忠实度/定位/幻觉诊断上全面超越现有方法且GPU显存需求大幅降低。

**[Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)**

:   提出 I2X 框架，通过在训练检查点追踪从 GradCAM 提取的原型强度与模型置信度的协同演化，将非结构化的可解释性（显著性图）转化为结构化的可解释性，揭示模型"为什么关注那里"的推理结构，并利用这种理解指导微调提升性能。
