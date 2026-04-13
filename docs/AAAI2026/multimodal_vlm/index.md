---
title: >-
  AAAI2026 多模态VLM方向 74篇论文解读
description: >-
  74篇AAAI2026 多模态VLM方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧩 多模态VLM

**🤖 AAAI2026** · 共 **74** 篇

**[Aligning The True Semantics Constrained Decoupling And Distr](aligning_the_true_semantics_constrained_decoupling_and_distr.md)**

:   提出 CDDS 算法，通过双路径 UNet 将嵌入解耦为语义和模态分量，并利用分布采样方法间接实现跨模态语义对齐，避免直接调整嵌入导致的分布扭曲，在 Flickr30K 和 MS-COCO 上超越 SOTA 6.6%~14.2%。

**[Anyecg-Chat A Generalist Ecg-Mllm For Flexible Ecg Input And](anyecg-chat_a_generalist_ecg-mllm_for_flexible_ecg_input_and.md)**

:   构建anyECG数据集（含报告生成、波形定位、多ECG比较三大任务）并提出anyECG-chat模型，通过动态ECG输入机制支持变长/少导联/多ECG输入，采用三阶段课程学习训练，在报告生成的OOD泛化、秒级异常波形定位和多ECG对比分析上全面超越现有ECG-MLLM。

**[Are We Done Yet A Vision-Based Judge For Autonomous Task Completion Of Computer ](are_we_done_yet_a_vision-based_judge_for_autonomous_task_completion_of_computer_.md)**

:   提出基于 VLM 的自主任务完成评估框架，通过截图+任务描述判断 CUA 是否完成任务，并将评估反馈回传给 Agent 实现自我纠正，在 macOS 环境上达到 73% 评估准确率和 27% 的任务成功率相对提升。

**[Astar Boosting Multimodal Reasoning With Automated Structure](astar_boosting_multimodal_reasoning_with_automated_structure.md)**

:   提出AStar，一种training-free的多模态推理范式，通过从500个种子样本中构建高层"thought cards"推理模板库，在推理时自适应检索最优模板引导MLLM结构化推理，7B模型在MathVerse上达53.9%准确率（超越GPT-4o的50.2%），仅需50分钟预处理时间且无需训练。

**[Biprompt Bilateral Prompt Optimization For Visual And Textual Debiasing In Visio](biprompt_bilateral_prompt_optimization_for_visual_and_textual_debiasing_in_visio.md)**

:   提出 BiPrompt，一种双边 prompt 优化框架，在测试时同时缓解 CLIP 等 VLM 中视觉侧（结构化注意力擦除）和文本侧（平衡 prompt 归一化）的虚假偏差，无需重训练即可提升 OOD 鲁棒性。

**[Bofa Bridge-Layer Orthogonal Low-Rank Fusion For Clip-Based ](bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)**

:   提出BOFA框架，仅微调CLIP已有的跨模态投影层（bridge-layer），通过正交低秩融合（Orthogonal Low-Rank Fusion）将参数更新约束在与旧任务特征正交的低秩"安全子空间"中，配合跨模态混合原型分类器，在不增加任何额外参数和推理开销的前提下实现了SOTA的无样本存储类增量学习。

**[Branch Or Layer Zeroth-Order Optimization For Continual Lear](branch_or_layer_zeroth-order_optimization_for_continual_lear.md)**

:   本文系统探索了零阶（ZO）优化在基于PEFT的视觉-语言持续学习（VLCL）中的应用，发现全ZO替换会导致训练不稳定，提出从分支级（branch-wise）到层级（layer-wise）的渐进式ZO-FO混合策略，并基于视觉模态方差更大的理论发现提出MoZO策略（梯度符号归一化+视觉扰动约束），在四个benchmark上达到SOTA。

**[Bridging Modalities Via Progressive Re-Alignment For Multimo](bridging_modalities_via_progressive_re-alignment_for_multimo.md)**

:   提出 BriMPR 框架，通过"分而治之"策略将多模态测试时自适应(MMTTA)分解为多个单模态特征对齐子问题，先用 prompt tuning 校准各模态全局特征分布实现初始跨模态语义对齐，再通过跨模态掩码嵌入重组和实例级对比学习精细化对齐。

**[Bridging The Copyright Gap Do Large Vision-Language Models R](bridging_the_copyright_gap_do_large_vision-language_models_r.md)**

:   首次系统评估 LVLM 在多模态上下文中对版权内容的识别和遵守能力，构建了 50,000 对多模态查询-内容的大规模 benchmark，发现 11/12 个 SOTA LVLM 即使面对明确版权声明也无法有效拒绝侵权请求，并提出 CopyGuard 工具增强框架将侵权拒绝率从 ~3% 提升至 ~62%。

**[Concept-Rulenet Grounded Multi-Agent Neurosymbolic Reasoning](concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)**

:   提出Concept-RuleNet——一个三智能体协作的神经符号推理框架，通过从训练图像中提取视觉概念来条件化符号生成和规则构建，解决了现有方法（如Symbol-LLM）仅依赖标签导致的符号幻觉和不代表性问题，在5个OOD基准上平均提升~5%准确率，幻觉符号减少达50%。

**[Conditional Information Bottleneck For Multimodal Fusion Overcoming Shortcut Lea](conditional_information_bottleneck_for_multimodal_fusion_overcoming_shortcut_lea.md)**

:   揭示多模态讽刺检测中三类捷径学习问题（角色标签偏见、罐头笑声标签泄漏、情感不一致捷径）并重构了无捷径的 MUStARD++R 数据集，提出基于条件信息瓶颈的多模态融合框架 MCIB，通过压缩主模态冗余同时保留辅助模态的互补信息来实现有效融合。

**[Crebench Human-Aligned Creativity Evaluation From Idea To Process To Product](crebench_human-aligned_creativity_evaluation_from_idea_to_process_to_product.md)**

:   提出 CreBench，一个覆盖创意想法→创作过程→创意产品三个维度、12个细粒度指标的多模态创造力评估基准，配套构建 CreMIT（2.2K样本、79.2K人工评价、4.7M指令）并微调出 CreExpert，在创造力评估上显著优于 GPT-4V 和 Gemini-Pro-Vision。

**[Cross-Modal Proxy Evolving For Ood Detection With Vision-Lan](cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)**

:   提出 CoEvo，一个 training-free 和 annotation-free 的 test-time 框架，通过双向 sample-conditioned 的文本/视觉 proxy 协同演化机制动态更新正负代理缓存，在 ImageNet-1K 上比最强负标签基线 AUROC 提升 1.33%、FPR95 降低 45.98%（从 18.92% 降至 10.22%），实现 SOTA 的 zero-shot OOD 检测。

**[Cross-Modal Unlearning Via Influential Neuron Path Editing I](cross-modal_unlearning_via_influential_neuron_path_editing_i.md)**

:   提出 MIP-Editor，通过跨层梯度积分（文本）和 Fisher 积分（视觉）定位多模态大语言模型中编码待遗忘知识的**影响力神经元路径**，再用基于路径的表示误导（RMisU）编辑这些神经元，在 MLLMU-Bench 上实现最高 87.75% 的遗忘率和 54.26% 的通用知识保留提升。

**[Crossvid A Comprehensive Benchmark For Evaluating Cross-Vide](crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)**

:   提出首个系统评估多模态大语言模型（MLLM）跨视频推理（Cross-Video Reasoning, CVR）能力的综合基准CrossVid，涵盖4个维度10个任务、5,331个视频和9,015个QA对，实验揭示当前最佳模型Gemini-2.5-Pro仅达50.4%准确率，远低于人类89.2%。

**[Difference Vector Equalization For Robust Fine-Tuning Of Vis](difference_vector_equalization_for_robust_fine-tuning_of_vis.md)**

:   提出DiVE方法，通过约束预训练和微调模型嵌入之间的"差异向量"在各样本间保持相等，从而在CLIP微调过程中保持嵌入空间的几何结构，同时在ID、OOD、零样本三个指标上取得全面优于现有方法的结果（零样本平均提升8+点）。

**[Discode Distribution-Aware Score Decoder For Robust Automatic Evaluation Of Imag](discode_distribution-aware_score_decoder_for_robust_automatic_evaluation_of_imag.md)**

:   提出 DISCODE，一种免微调的测试时自适应解码器，通过引入高斯先验分布最小化 ATT 损失，使 LVLM 生成的图像描述评估分数更鲁棒地对齐人类判断，并构建了覆盖六个视觉域的 MCEval 基准。

**[Em-Kd Distilling Efficient Multimodal Large Language Model W](em-kd_distilling_efficient_multimodal_large_language_model_w.md)**

:   提出EM-KD框架，通过Hungarian算法解决teacher-student间视觉token数量不平衡问题，结合视觉语义蒸馏(VSD)和视觉-语言亲和力蒸馏(VLAD)将vanilla teacher的知识迁移到高效student MLLM，在11个benchmark上以144 token/patch达到50.4均分，超越576 token的LLaVA-NeXT(49.4)同时推理速度提升近2倍。

**[Exo2Ego Exocentric Knowledge Guided Mllm For Egocentric Vide](exo2ego_exocentric_knowledge_guided_mllm_for_egocentric_vide.md)**

:   提出 Exo2Ego 框架，通过学习外中心(第三人称)与自中心(第一人称)域之间的映射关系，将 MLLM 中丰富的外中心知识迁移到自中心视频理解，结合新构建的 110万同步 ego-exo clip-text 对数据集 Ego-ExoClip 和 60万指令微调数据集 EgoIT，在 8 个自中心视频基准上取得了领先的开源模型性能。

**[Exploring Llms For Scientific Information Extraction Using The Sciex Framework](exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)**

:   本文提出SciEx，一个模块化、可组合的科学信息抽取框架，将PDF解析、多模态检索、Schema引导的抽取和跨文档聚合解耦为独立组件，在医学和环境科学的143篇论文数据集上评估了GPT-4o和Gemini-2.5-Flash的抽取能力，揭示了当前LLM在跨模态推理、数值精度和领域泛化方面的系统性不足。

**[Filter Correlate Compress Training-Free Token Reduction For ](filter_correlate_compress_training-free_token_reduction_for_.md)**

:   提出FiCoCo三阶段框架（Filter-Correlate-Compress），通过集成视觉感知+语义感知冗余度量筛选丢弃token，利用token间相关性自适应回收信息，实现training-free的MLLM加速。在LLaVA-NeXT上达14.7×FLOPs压缩同时保留93.6%性能，在5种MLLM架构上全面超越FastV、SparseVLM等SOTA。

**[Finmmdocr Benchmarking Financial Multimodal Reasoning With Scenario Awareness Do](finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)**

:   本文提出FinMMDocR，一个面向真实金融场景的双语多模态推理基准，包含1200道专家标注的数值推理题目，涵盖12类隐式金融情景、9类长文档（平均50.8页）和平均11步推理链，最强MLLM (o4-mini-high) 仅达58%准确率，揭示现有模型在复杂金融推理中的严重不足。

**[Format Matters The Robustness Of Multimodal Llms In Reviewing Evidence From Tabl](format_matters_the_robustness_of_multimodal_llms_in_reviewing_evidence_from_tabl.md)**

:   本文系统研究了多模态LLM在使用表格和图表作为证据验证科学声明时的鲁棒性，通过扩展SciTabAlign和ChartMimic两个数据集构建了表格-图表对齐的评估基准，发现12个多模态LLM在表格证据上的表现一致优于图表证据，而人类在两种格式上表现一致，揭示了当前模型在图表理解方面的关键短板。

**[Ft-Ncfm An Influence-Aware Data Distillation Framework For Efficient Vla Models](ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)**

:   提出 FT-NCFM 框架，通过因果归因（Fact-Tracing）评估样本价值并引导对抗式 NCFM 过程合成高信息密度核心集，仅用 5% 合成数据即可达到全量训练 85-90% 的性能，训练时间减少 80% 以上。

**[Global Compression Commander Plug-And-Play Inference Acceler](global_compression_commander_plug-and-play_inference_acceler.md)**

:   提出GlobalCom²，一个**即插即用、无需训练**的token压缩框架，专为动态裁剪（dynamic cropping）结构的高分辨率VLM设计：利用全局缩略图（thumbnail）作为"指挥官"引导局部裁剪区域（crop）的差异化压缩，在压缩90%视觉token的同时保持>90%原始性能。

**[Graph-Of-Mark Promote Spatial Reasoning In Multimodal Langua](graph-of-mark_promote_spatial_reasoning_in_multimodal_langua.md)**

:   提出 Graph-of-Mark (GoM)，一种无需训练的像素级视觉提示方法，通过在输入图像上直接叠加深度感知的场景图（包含节点和有向边），显式编码物体间的空间关系，使多模态语言模型在 VQA 和定位任务中的零样本空间推理准确率最高提升 11 个百分点。

**[Harnessing Textual Semantic Priors For Knowledge Transfer And Refinement In Clip](harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)**

:   本文提出SECA框架，利用CLIP文本分支的稳定语义先验来指导骨干网络中语义相关的历史知识迁移（SG-AKT模块），并通过文本嵌入的类间语义关系精炼视觉原型构建混合分类器（SE-VPR模块），在ImageNetR/A和CIFAR100上超越现有SOTA。

**[Harnessing Vision-Language Models For Time Series Anomaly Detection](harnessing_vision-language_models_for_time_series_anomaly_detection.md)**

:   提出两阶段零样本时序异常检测框架：ViT4TS 用轻量 ViT 对时序折线图做多尺度 cross-patch 匹配定位候选异常区间，VLM4TS 用 GPT-4o 结合全局时序上下文验证和精炼检测结果，在 11 个 benchmark 上 F1-max 超最优 baseline 24.6%，token 用量仅为现有 LLM 方法的 1/36。

**[Headhunt-Vad Hunting Robust Anomaly-Sensitive Heads In Mllm ](headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)**

:   提出HeadHunt-VAD，不用MLLM的文本输出，而是直接从冻结MLLM中"猎取"一小批对异常敏感且跨prompt鲁棒的注意力头，配合轻量逻辑回归scorer，在仅用1%数据、零微调的条件下，在UCF-Crime(87.03% AUC)和XD-Violence(82.63% AP)上达到tuning-free方法SOTA。

**[Heterogeneous Uncertainty-Guided Composed Image Retrieval With Fine-Grained Prob](heterogeneous_uncertainty-guided_composed_image_retrieval_with_fine-grained_prob.md)**

:   本文提出了HUG范式，通过细粒度高斯概率嵌入和异构不确定性估计（区分查询侧多模态协调不确定性与目标侧内容质量不确定性），结合动态加权融合和不确定性引导的对比学习，在Fashion-IQ和CIRR两个CIR基准上取得SOTA。

**[Imagebinddc Compressing Multi-Modal Data With Imagebind-Based Condensation](imagebinddc_compressing_multi-modal_data_with_imagebind-based_condensation.md)**

:   本文提出ImageBindDC，首个在ImageBind统一特征空间中进行多模态数据压缩的框架，利用特征函数距离（CFD）替代传统MMD，并设计单模态/跨模态/联合模态三级分布对齐损失，在NYU-v2上仅用5个合成样本/类即实现与全数据训练相当的性能（97.30%），比前SOTA绝对提升8.2%，且压缩时间削减4.6倍。

**[Inex Hallucination Mitigation Via Introspection And Cross-Mo](inex_hallucination_mitigation_via_introspection_and_cross-mo.md)**

:   提出 InEx 框架，通过内部自省推理（TVER 驱动的不确定性感知视觉增强）和外部跨模态多智能体协作（文本自反思 + 图像编辑验证 + 视觉自反思）迭代验证和修正 MLLM 输出，在 POPE 上提升 8.9%，在多个幻觉和通用 benchmark 上持续超越 OPERA/VCD/ICD。

**[Information Theoretic Optimal Surveillance For Epidemic Prevalence In Networks](information_theoretic_optimal_surveillance_for_epidemic_prevalence_in_networks.md)**

:   本文首次提出以互信息作为优化准则的流行病监测框架 TestPrev，旨在选择网络中的最优节点子集以最大化与疾病流行度分布的互信息，从而提供传统方法无法给出的暴发规模分布级别洞察，并证明了其 NP-hard 性质，设计了贪心算法 GreedyMI 在合成与真实网络上优于基线方法。

**[Learning To Tell Apart Weakly Supervised Video Anomaly Detection Via Disentangle](learning_to_tell_apart_weakly_supervised_video_anomaly_detection_via_disentangle.md)**

:   本文提出DSANet，通过自引导正常模式建模（SG-NM，粗粒度）和解耦对比语义对齐（DCSA，细粒度）从两个层面增强弱监督视频异常检测中正常与异常特征的可区分性，在XD-Violence上AP达86.95%（+1.14%），在UCF-Crime细粒度mAP达13.01%（+3.39%），均为SOTA。

**[Llm-Cas Dynamic Neuron Perturbation For Real-Time Hallucinat](llm-cas_dynamic_neuron_perturbation_for_real-time_hallucinat.md)**

:   LLM-CAS 首次将 LLM 实时幻觉纠正建模为层次强化学习（HRL）问题，训练 RL Agent 在推理时动态选择最优的神经元扰动策略（高层选择功能网络类别，低层选择扰动类型和幅度），结合自适应掩码+因果追踪精确定位目标神经元，在 StoryCloze 上提升 10.98%，超越 ITI/CAA/SADI 等静态/动态基线。

**[Macvqa Adaptive Memory Allocation And Global Noise Filtering For Continual Visua](macvqa_adaptive_memory_allocation_and_global_noise_filtering_for_continual_visua.md)**

:   本文提出MacVQA框架，通过全局噪声过滤（GonF）增强视觉特征的鲁棒性，并通过自适应记忆分配（AMA）基于原型检索和记忆衰减优化知识保留与更新，在VQA v2的10个持续学习任务上实现43.38%平均准确率（+3.57%）和2.32%遗忘率。

**[Mcmoe Completing Missing Modalities With Mixture Of Experts For Incomplete Multi](mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)**

:   本文首次探索不完整多模态动作质量评估问题，提出 MCMoE 框架，利用自适应门控模态生成器（AGMG）补全缺失模态，并通过混合专家（MoE）动态融合单模态和跨模态联合表示，在单阶段训练中统一学习，在三个公开 AQA 基准上的完整和不完整场景中均达到 SOTA，且参数量仅 4.90M。

**[Multi-Agent Vlms Guided Self-Training With Pnu Loss For Low-Resource Offensive C](multi-agent_vlms_guided_self-training_with_pnu_loss_for_low-resource_offensive_c.md)**

:   本文提出了一种多智能体视觉语言模型（MA-VLMs）引导的自训练框架，结合新颖的PNU损失函数，在仅有少量标注数据（如50个）的低资源场景下实现高质量攻击性内容检测，性能接近大规模模型。

**[Multi-Faceted Attack Exposing Cross-Model Vulnerabilities In Defense-Equipped Vi](multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)**

:   提出多面攻击框架MFA，通过注意力转移攻击(ATA)突破对齐、对抗签名绕过内容审核、视觉编码器攻击覆写系统提示三个维度，系统性暴露配备多层防御的VLM（含GPT-4o/Gemini等商业模型）的安全漏洞，总体攻击成功率达58.5%。

**[O3Slm Open Weight Open Data And Open Vocabulary Sketch-Language Model](o3slm_open_weight_open_data_and_open_vocabulary_sketch-language_model.md)**

:   本文构建了大规模草图-图像-指令三元组数据集SketchVCL（包含600K预训练 + 215K微调数据），并训练了O3SLM——首个能够流畅理解手绘草图并完成检测、计数、检索和VQA四大任务的开源大视觉语言模型，在所有任务上大幅超越现有LVLM。

**[Oida-Qa A Multimodal Benchmark For Analyzing The Opioid Industry Documents Archi](oida-qa_a_multimodal_benchmark_for_analyzing_the_opioid_industry_documents_archi.md)**

:   本文基于UCSF-JHU阿片类药物行业文档档案（OIDA），构建了包含400K训练文档和370K多跳QA对的多模态文档问答基准OIDA-QA，并开发了结合内容重述和页面查找器的领域特化LLM系统，有效处理超长文档的多轮问答和答案页面定位。

**[Omnipt Unleashing The Potential Of Large Vision Language Models For Pedestrian T](omnipt_unleashing_the_potential_of_large_vision_language_models_for_pedestrian_t.md)**

:   本文提出OmniPT，一个基于大视觉语言模型（LVLM）的统一行人跟踪框架，通过RL-Mid Training-SFT-RL四阶段训练策略，同时支持传统MOT、基于语言引用的跟踪（RMOT/CRMOT）和语义理解（SMOT），在多个基准上取得SOTA结果，尤其在BenSMOT上HOTA达75.04，较前SOTA提升3.06。

**[Panda Test-Time Adaptation With Negative Data Augmentation](panda_test-time_adaptation_with_negative_data_augmentation.md)**

:   提出 Panda，通过负数据增强（patch 打乱重组）生成保留 corruption 但破坏语义的图像，用其特征偏移原始嵌入以抑制 corruption 引起的预测偏差，以极低开销（<10%）即插即用提升各类 TTA 方法的鲁棒性。

**[Patientvlm Meets Docvlm Pre-Consultation Dialogue Between Vision-Language Models](patientvlm_meets_docvlm_pre-consultation_dialogue_between_vision-language_models.md)**

:   提出Pre-Consultation Dialogue Framework (PCDF)，通过两个VLM（DocVLM和PatientVLM）模拟医生-患者多轮对话，生成image-dialogue-diagnosis三元组用于微调DocVLM，在四个医学影像基准上平均F1提升11.48。

**[Patientvlm Meets Docvlm Pre-Consultation Dialogue Between Vision Language Models](patientvlm_meets_docvlm_pre-consultation_dialogue_between_vision_language_models.md)**

:   本文提出PCDF（Pre-Consultation Dialogue Framework），通过两个VLM角色扮演——DocVLM提问、PatientVLM回答——模拟真实医患对话，生成image-dialogue-diagnosis三元组用于微调DocVLM，在四个医学影像基准上平均F1提升11.48个百分点，且不依赖真实临床对话数据。

**[Pet2Rep Towards Vision-Language Model-Drived Automated Radiology Report Generati](pet2rep_towards_vision-language_model-drived_automated_radiology_report_generati.md)**

:   本文提出 PET2Rep，首个专用于正电子发射断层扫描（PET）放射报告生成的大规模基准数据集（565例全身 PET/CT 图像-报告对），并设计了 PET 临床效能（CE）评估指标，对 30 个前沿通用和医疗专用 VLM 进行系统评估，发现当前 SOTA VLM 在 PET 报告生成任务上表现不佳，甚至无法超越简单的模板基线。

**[Phantom Menace Exploring And Enhancing The Robustness Of Vla Models Against Phys](phantom_menace_exploring_and_enhancing_the_robustness_of_vla_models_against_phys.md)**

:   本文首次系统研究Vision-Language-Action（VLA）模型面对物理传感器攻击的安全性，提出"Real-Sim-Real"框架评估六种摄像头攻击和两种麦克风攻击对四个VLA模型的影响，发现所有VLA模型均存在严重脆弱性，并提出基于对抗训练的防御方法将中等强度攻击下的性能提升高达60%。

**[Pharos-Esg A Framework For Multimodal Parsing Contextual Narration And Hierarchi](pharos-esg_a_framework_for_multimodal_parsing_contextual_narration_and_hierarchi.md)**

:   本文提出Pharos-ESG框架，通过基于版面流的阅读顺序建模、目录锚点引导的层次结构重建、上下文感知的多模态图像描述转换、以及多级金融标签预测四个核心模块，实现对ESG报告的结构化解析，在全面评估中F1达93.59、ROKT达0.92、TBTA达92.46%，显著超越MinerU、GPT-4o、Gemini 2.5 Pro等基线，并发布了首个大规模公开ESG报告数据集Aurora-ESG（24K+报告）。

**[Planttraitnet An Uncertainty-Aware Multimodal Framework For Global-Scale Plant T](planttraitnet_an_uncertainty-aware_multimodal_framework_for_global-scale_plant_t.md)**

:   本文提出 PlantTraitNet，一个多模态、多任务、不确定性感知的深度学习框架，利用公民科学平台（iNaturalist、Pl@ntNet）的弱监督植物照片，结合图像特征（DINOv2）、深度先验（Depth-Anything-V2）和地理空间先验（Climplicit），同时预测四种关键植物性状（株高、叶面积、比叶面积、叶氮含量），生成的全球性状图在与 sPlotOpen 植被调查数据的基准测试中一致优于现有全球性状产品。

**[Recad Reinforcement Learning Enhanced Parametric Cad Model Generation With Visio](recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)**

:   提出 ReCAD 框架，通过将 CAD 脚本重写为参数化代码进行 SFT，再利用 GRPO 强化学习与分层基元课程学习策略，使 VLM 能从文本或图像输入生成高精度、可编辑的参数化 CAD 模型，在分布内和分布外设置上均大幅超越现有方法。

**[Rethinking Visual Token Reduction In Lvlms Under Cross-Modal Misalignment](rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)**

:   揭示了 LVLM 中文本引导视觉token重要性评估的三种跨模态失配问题（因果、语义、空间），提出 VisionDrop——一个仅依赖视觉自注意力的免训练渐进式token剪枝框架，跨视觉编码器和 LLM 解码器多阶段压缩，在保留 5.6% token 时仍能维持 91%+ 原始性能。

**[Revisiting The Data Sampling In Multimodal Post-Training From A Difficulty-Disti](revisiting_the_data_sampling_in_multimodal_post-training_from_a_difficulty-disti.md)**

:   提出两种多模态数据难度评估策略——PISM（渐进图像语义遮蔽）和CMAB（跨模态注意力平衡），发现在难度分层数据上仅用GRPO训练即可一致超越传统SFT+GRPO流水线，证明了战略性数据筛选比复杂训练范式更重要。

**[Rmadapter Reconstructionbased Multimodal Adapter For Visionlanguage](rmadapter_reconstructionbased_multimodal_adapter_for_visionlanguage.md)**

:   提出 RMAdapter，一种双分支适配器架构：在标准 adapter 的适应分支旁增加重建分支（类 AutoEncoder），通过共享下投影层和逐层本地重建损失，在 CLIP 少样本微调中实现任务特定适应与通用知识保持的最佳平衡，在 Base-to-Novel 泛化、跨数据集和领域泛化三个任务上全面超越 SOTA（含 Prompt-based 方法）。

**[Safer-Clip Mitigating Nsfw Content In Vision-Language Models While Preserving Pr](safer-clip_mitigating_nsfw_content_in_vision-language_models_while_preserving_pr.md)**

:   提出SafeR-CLIP框架，通过近邻感知重定向（将不安全嵌入重定向到语义最近的安全目标而非固定配对）和相对跨模态重定向损失（仅以不安全表示作为负样本而非随机批内负样本），在保持安全性的同时将零样本分类精度比Safe-CLIP恢复8.0%。

**[Sage Spuriousness-Aware Guided Prompt Exploration For Mitigating Multimodal Bias](sage_spuriousness-aware_guided_prompt_exploration_for_mitigating_multimodal_bias.md)**

:   提出SAGE，一种无需训练、微调或外部标注的提示选择方法，通过计算提示模板在类别间的分离度得分来缓解CLIP模型中的多模态虚假偏差，在四个基准+五个骨干模型上一致提升最差组准确率（WGA）和调和均值（HM）。

**[Satiredecoder Visual Cascaded Decoupling For Enhancing Satirical Image Comprehen](satiredecoder_visual_cascaded_decoupling_for_enhancing_satirical_image_comprehen.md)**

:   提出SatireDecoder，一种无需训练的框架，通过多智能体视觉级联解耦和不确定性引导的CoT推理来增强MLLM对讽刺图像的深层语义理解，在YesBut数据集上正确性、完整性和忠实性三项指标分别提升10%-40%。

**[Sdeval Safety Dynamic Evaluation For Multimodal Large Language Models](sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)**

:   提出 SDEval，首个面向 MLLM 安全性的动态评估框架，通过文本动态（同义替换/多语混合/CoT 注入等）、图像动态（空间/颜色变换/扩散生成）和文本-图像交互动态三类策略，从静态 benchmark 可控生成不同复杂度的新样本，缓解数据泄露并暴露模型安全短板。

**[See Symbolize Act Grounding Vlms With Spatial Representations For Better Gamepla](see_symbolize_act_grounding_vlms_with_spatial_representations_for_better_gamepla.md)**

:   系统性评估了符号化空间表示（物体坐标）对VLM游戏能力的影响，发现符号信息仅在检测准确时有益，当VLM自提取符号时效果取决于模型能力和场景复杂度，视觉帧始终不可或缺。

**[Seeing Justice Clearly Handwritten Legal Document Translation With Ocr And Visio](seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)**

:   本文系统性对比了传统 OCR+机器翻译（OCR-MT）流水线与视觉大语言模型（vLLM）在手写马拉地语法律文档翻译为英语任务上的表现，发现两类方法均未达到法律级部署要求，OCR-MT 受级联错误影响严重，vLLM 存在严重的幻觉问题，但 vLLM 展现出统一端到端处理的发展潜力。

**[Speakerlm End-To-End Versatile Speaker Diarization And Recognition With Multimod](speakerlm_end-to-end_versatile_speaker_diarization_and_recognition_with_multimod.md)**

:   SpeakerLM 是首个专为端到端说话人分离与识别（SDR）设计的多模态大语言模型，通过音频编码器-投影器-LLM 架构和灵活的说话人注册机制，在多个公开基准上大幅超越级联基线系统（cpCER 绝对降低最高达 13.82%），并在域外测试集上展现强鲁棒性。

**[Stola Self-Adaptive Touch-Language Framework With Tactile Commonsense Reasoning ](stola_self-adaptive_touch-language_framework_with_tactile_commonsense_reasoning_.md)**

:   SToLa 提出首个基于混合专家（MoE）的触觉-语言框架，通过动态路由机制管理触觉和语言两种模态的差异，并构建了覆盖8种物理属性、4种交互特征的开放式触觉常识推理数据集 TactileBench，在 PhysiCLeAR 基准上以 7B 参数量超越 13B 的 Octopi 取得 SOTA。

**[Tabflash Efficient Table Understanding With Progressive Question Conditioning An](tabflash_efficient_table_understanding_with_progressive_question_conditioning_an.md)**

:   TabFlash 提出渐进式问题条件化（Progressive Question Conditioning）和 Token 聚焦（Token Focusing）两大技术，在 ViT 中注入问题信息生成问题感知的视觉特征，并基于 L2 范数剪枝背景 token 同时通过对比训练将关键信息集中到保留 token 中，在7个表格理解基准上超越 GPT-4o 和 Gemini 2.5 Pro，同时减少 27% FLOPs 和 30% 显存。

**[The Triangle Of Similarity A Multi-Faceted Framework For Comparing Neural Networ](the_triangle_of_similarity_a_multi-faceted_framework_for_comparing_neural_networ.md)**

:   本文提出"相似性三角"（Triangle of Similarity）框架，整合静态表征相似性（CKA/Procrustes）、功能相似性（线性模式连接/预测分布相似性）和稀疏性相似性（剪枝鲁棒性）三个互补视角来全面比较神经网络，发现架构家族是表征相似性的主要决定因素，且模型的表征结构比任务准确率在剪枝下更为鲁棒。

**[Tinychemvl Advancing Chemical Vision-Language Models Via Efficient Visual Token ](tinychemvl_advancing_chemical_vision-language_models_via_efficient_visual_token_.md)**

:   TinyChemVL 是一个仅4B参数的化学领域VLM，通过自适应token合并与剪枝策略将视觉token压缩至原来的1/16，并引入反应级别任务和基准ChemRxn-V，在分子和反应级别的视觉化学任务上达到SOTA性能，同时显著提升推理和训练速度。

**[Tofa Training-Free One-Shot Federated Adaptation For Vision-Language Models](tofa_training-free_one-shot_federated_adaptation_for_vision-language_models.md)**

:   提出TOFA框架，在联邦学习场景下通过层次贝叶斯模型学习个性化视觉prototype分布 + 全局对齐的LLM文本增强 + 自适应模态融合，实现无需训练、仅一轮通信的CLIP高效适配，在9个数据集上超越one-shot基线甚至部分多轮训练方法。

**[Towards A Rigorous Understanding Of The Population Dynamics Of The Nsga-Iii Tigh](towards_a_rigorous_understanding_of_the_population_dynamics_of_the_nsga-iii_tigh.md)**

:   本文首次为 NSGA-III 在经典双目标 OneMinMax 基准上建立了紧致运行时界 $\Theta(n^2 \ln n / \mu)$，揭示了 NSGA-III 的种群动态特性，并证明其在适当种群规模下优于 NSGA-II。

**[Towards Authentic Movie Dubbing With Retrieve-Augmented Director-Actor Interacti](towards_authentic_movie_dubbing_with_retrieve-augmented_director-actor_interacti.md)**

:   Authentic-Dubber 模拟真实配音工作流程中导演与演员的交互过程，通过构建多模态参考素材库、基于情感相似度的检索增强策略和渐进式图语音生成方法，显著提升了自动电影配音的情感表现力，在V2C-Animation数据集上的情感准确率和MOS评分均达到SOTA。

**[Towards Human-Ai Accessibility Mapping In India Vlm-Guided Annotations And Poi-C](towards_human-ai_accessibility_mapping_in_india_vlm-guided_annotations_and_poi-c.md)**

:   本文将Project Sidewalk无障碍标注平台适配到印度昌迪加尔，通过定制化界面标签、VLM驱动的任务指导（Gemini 2.5 Flash），以及以POI为中心的分析框架，在三个不同土地用途的区域中审计了约40公里人行道，识别出1,644处可改善的无障碍设施位置。

**[Towards Long-Window Anchoring In Vision-Language Model Distillation](towards_long-window_anchoring_in_vision-language_model_distillation.md)**

:   LAid（Long-window Anchoring distillation）提出了一种位置感知的知识蒸馏框架，通过头部级别的傅里叶增强位置知识传递，将小型VLM（3B/7B）的有效上下文窗口扩展至原来的3.2倍，接近大型教师模型（32B）的水平，同时保持标准VL基准上的性能。

**[Tri-Bench Stress-Testing Vlm Reliability On Spatial Reasoning Under Camera Tilt ](tri-bench_stress-testing_vlm_reliability_on_spatial_reasoning_under_camera_tilt_.md)**

:   Tri-Bench 是一个包含400张实拍三角形图像的紧凑基准，通过控制相机姿态（平面/倾斜）和物体干扰两个因素，系统测试了四个领先VLM的空间几何推理能力，发现模型默认依赖2D图像平面线索而非3D真实几何（即使提供了明确的参考框架提示），在非多数类形状上准确率降至接近0%。

**[Urag Unified Retrieval And Generation In Multimodal Llms For](urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)**

:   URaG 发现 MLLM 处理长文档时存在类人的"粗到细"推理模式（浅层注意力均匀分散、深层集中于证据页），基于此洞察在第 6 层插入轻量跨模态检索模块（仅占参数 0.05%），选取 Top-5 相关页面丢弃其余内容，实现 SOTA 性能的同时减少 44-56% 计算量。

**[Verb Mirage Unveiling And Assessing Verb Concept Hallucinations In Multimodal La](verb_mirage_unveiling_and_assessing_verb_concept_hallucinations_in_multimodal_la.md)**

:   首次系统研究多模态大语言模型（MLLM）中的动词概念幻觉问题，构建了多维度基准测试，发现现有幻觉缓解方法对动词幻觉无效，并提出基于丰富动词知识微调的基线方法，显著缓解动词幻觉。

**[Vipact Visual-Perception Enhancement Via Specialized Vlm Age](vipact_visual-perception_enhancement_via_specialized_vlm_age.md)**

:   VipAct 提出了一个多Agent协作框架，通过编排器Agent（任务分析+规划+协调）、专用Agent（描述/比较/视觉提示解读）和视觉专家模型（深度估计/目标检测/分割等）三层协作，显著提升 VLM 在细粒度视觉感知任务上的表现，在 Blink 上从 63.74% (zero-shot GPT-4o) 提升到 73.79%。

**[Vp-Bench A Comprehensive Benchmark For Visual Prompting In M](vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)**

:   VP-Bench 提出了首个系统评估 MLLM 视觉提示（Visual Prompt）理解能力的两阶段 Benchmark：Stage 1 用 30K+ 图像覆盖 8 种 VP 形状×355 种属性组合评测 VP 感知能力，Stage 2 评测 VP 对 6 个下游任务的实际效果。在 28 个 MLLM 上的评测揭示了 VP 形状选择对性能的关键影响。
