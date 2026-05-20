---
title: >-
  ICML2026 医学图像方向17篇论文解读
description: >-
  17篇ICML2026的医学图像方向论文解读，涵盖生物分子、多模态、对齐/RLHF、医学影像、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "医学图像"
  - "论文解读"
  - "论文笔记"
  - "生物分子"
  - "多模态"
  - "对齐/RLHF"
  - "医学影像"
  - "推理"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🏥 医学图像

**🧪 ICML2026** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (51)](../../ACL2026/medical_imaging/index.md) · [📷 CVPR2026 (119)](../../CVPR2026/medical_imaging/index.md) · [🔬 ICLR2026 (79)](../../ICLR2026/medical_imaging/index.md) · [🤖 AAAI2026 (106)](../../AAAI2026/medical_imaging/index.md) · [🧠 NeurIPS2025 (147)](../../NeurIPS2025/medical_imaging/index.md) · [📹 ICCV2025 (38)](../../ICCV2025/medical_imaging/index.md)

🔥 **高频主题：** 生物分子 ×5 · 多模态 ×3 · 对齐/RLHF ×3 · 医学影像 ×3 · 推理 ×2

**[Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions](auditing_sybil_explaining_deep_lung_cancer_risk_prediction_through_generative_in.md)**

:   本文提出 S(H)NAP——基于 3D 扩散桥的「移除 + 插入」生成式干预框架，把 Sybil 这一前沿肺癌风险预测模型的决策反向拆解为「肺结节主效应 + 两两交互 + 背景」的 LMPI（线性+二阶交互模型），首次以因果而非相关的方式审计出它对 ECG 电极、衣物金属扣等院内伪影的依赖以及对外周肺结节的「径向不敏感」严重失败模式。

**[EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](eeg-based_multimodal_learning_via_hyperbolic_mixture-of-curvature_experts.md)**

:   EEG-MoCE 给 EEG-based 多模态学习（情绪/睡眠/认知）每个模态分配一个**可学习曲率**的 Lorentz 流形 expert，再用"曲率大→层级结构更丰富→在 fusion 中权重更高"的 curvature-aware attention 做跨模态融合，在 EAV/ISRUC/Cognitive 三个数据集上 cross-subject 准确率分别 +14.14%、+3.34%、+7.98%。

**[Evidential Reasoning Advances Interpretable Real-World Disease Screening](evidential_reasoning_advances_interpretable_real-world_disease_screening.md)**

:   EviScreen 用「正常 + 病理」双知识库做区域级证据检索，再以 cross-attention + self-attention 在当前病例和证据间做循证推理，既给出**回溯式可解释性**（哪几个历史病例支持当前判断）又给出**定位可解释性**（对比检索得到的异常图），在 4 个真实外部测试集上把高召回处的特异性提升到 SOTA。

**[Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration](federated_distillation_for_whole_slide_image_via_gaussian-mixture_feature_alignm.md)**

:   本文提出 FedHD：在异构联邦病理学场景下，用 Gaussian-mixture 特征对齐做「一对一」WSI 特征级蒸馏，再通过课程学习把跨机构合成特征逐步注入本地训练，使各机构能在不共享原始数据、不交换模型参数的前提下协作，且兼容异构 MIL 架构与特征提取器，在 TCGA-IDH / CAMELYON16 / CAMELYON17 上全面超越现有联邦与蒸馏基线。

**[From Holo Pockets to Electron Density: GPT-style Drug Design with Density](from_holo_pockets_to_electron_density_gpt-style_drug_design_with_density.md)**

:   本文把结构药物设计的 condition 从"刚性 empty pocket"换成"包含配体与溶剂的 filler 低分辨率电子云"，并提出第一个 decoder-only autoregressive 的 EDMolGPT，在 DUD-E 101 个靶点上 bioactive recovery 达 41%、远超先前 ED-based 方法。

**[OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport](geometrically_constrained_stenosis_editing_in_coronary_angiography_via_entropic_.md)**

:   OT-Bridge Editor 把"在冠脉造影上编辑一段血管狭窄"重写为"在血管-结构复合域里的约束熵 OT 问题"，用 Schrödinger Bridge 沿路径加几何投影监督，做到像素级形状/位置可控的合成造影，在 ARCADE 公开集上把下游狭窄检测 mAP@0.5 相对提升 27.8%。

**[Layer-Specific Fine-Tuning for Improved Negation Handling in Medical Vision-Language Models](layer-specific_fine-tuning_for_improved_negation_handling_in_medical_vision-lang.md)**

:   NAST 用因果追踪 (causal tracing) 算出 CLIP 文本编码器各层对否定理解的因果贡献度 (CTE)，再以这些 CTE 做层级化梯度缩放微调 LoRA，让医学 VLM 在区分"有 / 没有某症状"时的语义敏感度大幅提升，并把肯定-否定准确率差距从 21.6% 缩到 4.2%。

**[Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach](learning_the_interaction_prior_for_protein-protein_interaction_prediction_a_mode.md)**

:   L3-PPI 把生物学里的 "L3 规则"（蛋白质对之间的 length-3 路径越多越可能相互作用）变成可学习的 graph prompt：用预训练 GNN 识别 L3 模式，再用门控网络生成虚拟 L3 路径并按 PPI 标签正则路径数量，做成一个即插即用的分类头，把任意 PPI 表征模型平均涨 2-4 个点。

**[Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning](marrying_generative_model_of_healthcare_events_with_digital_twin_of_social_deter.md)**

:   本文提出 DiffDT：用一个条件 Latent Diffusion 框架把电子病历（ICD-coded 事件序列）与多器官生物标记数字孪生（脑/心/肝/肾的影像衍生表格特征与脑功能连接 SPD 矩阵）连起来，关键创新是一个基于 Cholesky 分解的 SPD-VQVAE 把 $\mathcal{O}(N^3)$ 的 SPD 流形扩散降到流形保形且高效的潜空间，再让 AR 模型借“生成数字孪生 → 预测下一个 ICD”这条中介路径完成多通路疾病推理；在 UKB 上对 1944 类疾病的下一次预测 AUC 提到 0.91，刷新 SOTA。

**[MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](meg-xl_data-efficient_brain-to-text_via_long-context_pre-training.md)**

:   MEG-XL 用 2.5 分钟（191k token）的 MEG 上下文做 mask token 预训练（比此前长 5–300×），再微调到 50 词的脑到文本任务上，仅用 1 小时数据就达到 SOTA 监督方法 50 小时的解码精度，并显著超过所有 brain foundation models。

**[Protein Circuit Tracing via Cross-layer Transcoders](protein_circuit_tracing_via_cross-layer_transcoders.md)**

:   作者把 NLP 中的 cross-layer transcoder 搬到蛋白质语言模型 ESM2 上,提出 ProtoMech 框架以 < 1% 的稀疏潜变量电路恢复 79% 的下游性能,并能沿电路 steering 设计出高 fitness 的蛋白变体,在 70%+ 案例中击败基线。

**[Scaling Vision Transformers for Functional MRI with Flat Maps](scaling_vision_transformers_for_functional_mri_with_flat_maps.md)**

:   把 3D fMRI 体积按"皮层展平图"投影成 2D 视频后直接喂给标准 spacetime MAE-ViT，得到一个在 2.1K 小时 HCP 数据上训练的 CortexMAE：在认知状态解码上大幅超 SOTA，验证 flat map 是体素 (volume) 和脑区平均 (parcellation) 之间的"goldilocks zone"；同时发布首个开源 fMRI 基础模型基准 Brainmarks，给出 fMRI 模型的第一份系统 scaling law 与一个"个体特质预测仍打不过简单功能连接 baseline"的诚实 null result。

**[SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)**

:   SIGMA 用 token 级对比损失把同一分子不同 SMILES 排列的隐状态强制对齐到同一条轨迹，并配套提出 IsoBeam 在解码阶段剪掉同构冗余路径，让序列模型在化学空间中真正"按图而非按字符串"思考。

**[SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment](synermedgen_synergizing_medical_multimodal_understanding_with_generation_via_tas.md)**

:   SynerMedGen 提出"生成对齐理解（generation-aligned understanding）"原则——把理解任务直接从同一份配对合成数据里派生出来（CTS / MI / TIA 三个任务），先两阶段训练让理解分支学到对合成有用的表征，再迁移到 latent flow matching 生成分支，在 22 个医学合成任务上同时碾压专用合成模型和已有统一 MLLM。

**[TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation](td3b_transition-directed_discrete_diffusion_for_allosteric_binder_generation.md)**

:   TD3B 把激动剂/拮抗剂的设计当作「方向性转移算子」生成任务，用一个方向 Oracle + 亲和力门控 + 树搜索摊销微调的掩码离散扩散框架，让预训练肽段生成器学会写出能定向偏移蛋白质活/失活构象转移的多肽序列。

**[Towards A Generative Protein Evolution Machine with DPLM-Evo](towards_a_generative_protein_evolution_machine_with_dplm-evo.md)**

:   本文提出 DPLM-Evo，把蛋白质语言模型的离散扩散从"只支持掩码替换"扩展为"显式建模替换+插入+删除三种进化编辑"，通过把变长观测序列解耦到上采样长度的隐对齐空间 + 上下文化进化噪声核，实现变长进化生成、进化轨迹式的蛋白质后编辑/优化，并在 ProteinGym 单序列变体效应预测上取得 SOTA。

**[Towards Universal Gene Regulatory Network Inference: Unlocking Generalizable Regulatory Knowledge in Single-cell Foundation Models](towards_universal_gene_regulatory_network_inference_unlocking_generalizable_regu.md)**

:   本文指出单细胞基础模型 (scFM) 蕴含丰富但被"重建式预训练"遮蔽的基因调控知识，并提出 Virtual Value Perturbation 与 Gradient Trajectory 两种探针，从冻结的 scFM 中蒸馏出可跨基因/跨数据集泛化的成对基因特征，在 BEELINE 基准上把 AUPRC 从 ~0.5 推到 0.8–0.97，开启了"通用 GRN 推断 (UGRN)"这一新范式。
