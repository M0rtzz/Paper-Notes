---
title: >-
  ACL2025 LLM 安全方向65篇论文解读
description: >-
  65篇ACL2025的 LLM 安全方向论文解读，涵盖 LLM、对抗鲁棒、水印/隐写、对齐/RLHF、问答、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ACL2025"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
  - "水印/隐写"
  - "对齐/RLHF"
  - "问答"
  - "多模态"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**💬 ACL2025** · **65** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (21)](../../ACL2026/llm_safety/index.md) · [📷 CVPR2026 (16)](../../CVPR2026/llm_safety/index.md) · [🔬 ICLR2026 (36)](../../ICLR2026/llm_safety/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (59)](../../NeurIPS2025/llm_safety/index.md) · [📹 ICCV2025 (8)](../../ICCV2025/llm_safety/index.md)

🔥 **高频主题：** LLM ×28 · 对抗鲁棒 ×17 · 水印/隐写 ×6 · 对齐/RLHF ×3 · 问答 ×2

**[AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection](agrail_a_lifelong_agent_guardrail_with_effective_and_adaptive_safety_detection.md)**

:   提出 AGrail，一个终身学习的 LLM Agent 安全护栏框架，通过双 LLM 协作（Analyzer + Executor）和记忆模块，在测试时自适应地生成和优化安全检查策略，有效防御任务特定风险和系统性风险。

**[Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering](aligning_large_language_models_to_follow_instructions_and_hallucinate_less_via_e.md)**

:   提出NOVA框架，通过内部一致性探测(ICP)衡量LLM对指令的熟悉度+语义等价识别(SEI)衡量LLM对目标回复的熟悉度，筛选出知识对齐的高质量指令数据，仅用5%数据微调LLaMA-3-8B即可在BioGEN上提升8.6分、FollowRAG上提升7.2分，同时保持指令遵循能力。

**[Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning](alleviating_hallucinations_from_knowledge_misalignment_in_large_language_models_.md)**

:   本文针对LLM中因知识错位（模型参数知识与事实不一致）导致的幻觉问题，提出选择性弃权学习（Selective Abstention Learning）方法，让模型在遇到知识边界外的问题时学会主动拒绝回答而非编造内容，从而减少幻觉。

**[Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning](answer_when_needed_forget_when_not_language_models_pretend_to_forget_via_in-cont.md)**

:   提出"上下文知识遗忘"方法，通过引入特殊的遗忘 token `<<UNL>>...<</UNL>>` 使 LLM 在推理时根据上下文选择性遗忘特定知识，在 TOFU/AGE/RWKU 上达到 95% 遗忘准确率且保留 80% 无关知识，深入的内部分析发现 LLM 并未真正删除知识而是在最后一层"假装遗忘"。

**[Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs](are_the_hidden_states_hiding_something_testing_the_limits_of_factuality-encoding.md)**

:   本文挑战了"LLM隐藏状态能编码事实真假信息"这一先前结论，通过构建更真实、更具挑战性的数据集（基于困惑度的负样本采样和基于QA的LLM生成数据集），发现先前方法在更贴近实际场景的数据上泛化能力有限，为LLM事实性评估研究提供了更严格的评估基准和实践指导。

**[ArgHiTZ at ArchEHR-QA 2025: A Two-Step Divide and Conquer Approach to Patient Question Answering for Top Factuality](arghitz_at_archehr-qa_2025_a_two-step_divide_and_conquer_approach_to_patient_que.md)**

:   在 ArchEHR-QA 2025 共享任务中提出两阶段"分治"方法：先用重排序模型从电子健康记录中提取关键句子，再用小型医学 LLM 生成回复，在不使用外部知识的情况下取得事实性排名第一、总分第 8/30 的成绩。

**[Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval](automated_explanation_generation_and_hallucination_detection_for_heritage_image_.md)**

:   本文针对文化遗产图像检索任务，提出了一个结合自动解释生成和幻觉检测的框架，利用视觉语言模型为检索结果生成可解释的文本描述，同时通过领域知识约束的幻觉检测机制确保描述的事实准确性，在多个文化遗产数据集上验证了方法的有效性。

**[Bias in the Mirror: Are LLMs' Opinions Robust to Their Own Adversarial Attacks](bias_in_the_mirror_are_llms_opinions_robust_to_their_own_adversarial_attacks.md)**

:   本文提出一种新颖的"自辩论"范式，让同一个LLM的两个实例分别扮演正方和反方进行辩论，试图说服一个中立版本的模型，以此评估LLM内在偏见的鲁棒性——偏见是否容易被动摇，以及模型是否容易被自身的对抗性论证带偏。

**[CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks](cavgan_unifying_jailbreak_and_defense_of_llms_via_generative_adversarial_attacks.md)**

:   提出 CAVGAN 框架，利用生成对抗网络在 LLM 内部表示空间中同时学习越狱攻击（生成器）和安全防御（判别器），首次将攻防统一到同一框架中实现"攻防共进"，攻击成功率平均 88.85%，防御成功率平均 84.17%。

**[Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models](chinese_simpleqa_a_chinese_factuality_evaluation_for_large_language_models.md)**

:   提出 Chinese SimpleQA——首个全面的中文事实性评估基准，包含 3000 个高质量短问答（覆盖 6 大主题、99 个子主题），评估 41 个 LLM 后发现仅 o1-preview（63.8%）和 Doubao-pro-32k（61.9%）能通过，并系统揭示了"大模型更好"、"RAG缩小差距"、"对齐降低事实性"等关键洞察。

**[CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP](cliperase_efficient_unlearning_of_visual-textual_associations_in_clip.md)**

:   提出 CLIPErase，一种专为 CLIP 多模态模型设计的机器遗忘框架，通过遗忘模块、保留模块和一致性模块三部分协同，选择性地移除特定视觉-文本关联，同时保持模型在保留数据上的性能。

**[ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)**

:   构建 ComparisonQA 基准（283K 配对问题），通过让高频和低频实体共享同一抽象问题实现受控对比，结合正确性和不确定性的两轮评估方法发现 LLM（包括 GPT-4o）对低频知识的鲁棒性极差。

**[Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities](con_instruction_universal_jailbreaking_of_multimodal_large_language_models_via_n.md)**

:   本文提出 Con Instruction 方法，通过优化对抗性图像或音频使其在嵌入空间中与目标恶意指令对齐，实现无需文本输入即可越狱多模态大语言模型（MLLM），在 LLaVA-v1.5 上达到 86.6% 的攻击成功率，并提出了 ARC 评估框架来同时衡量攻击响应的质量和相关性。

**[Core: Robust Factual Precision with Informative Sub-Claim Identification](core_robust_factual_precision_with_informative_sub-claim_identification.md)**

:   本文提出 Core 框架，通过识别和过滤信息性子声明（informative sub-claims）来实现鲁棒的事实精度（factual precision）评估，解决了现有方法因无信息声明的稀释效应而导致评估不准确的问题。

**[Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence](cracking_hallucination_vhd.md)**

:   提出 VHD 指标量化每个注意力头输出对视觉输入的敏感程度，发现仅少数注意力头对视觉信息高度敏感而模型过度依赖语言先验是导致幻觉的关键因素，进而设计 VHR 免训练方法逐层自适应增强视觉感知头的贡献（$\alpha=2$），在 CHAIR 上将 LLaVA-1.5 的 CHAIR$_S$ 从 49.68 降至 33.32，且几乎无额外推理开销。

**[Defense Against Prompt Injection Attack by Leveraging Attack Techniques](defense_prompt_injection.md)**

:   本文提出一种"以攻为防"的 prompt injection 防御策略：将已有的攻击技术（ignore、escape、fake completion）反转用于防御，在被注入的数据内容后追加 shield prompt + 原始指令，使 LLM 忽略注入指令而执行原始指令，在多种攻击场景下将 ASR 降至接近零。

**[ReDial: Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](dialect_fairness_robustness.md)**

:   本文构建了首个高质量人工标注的标准英语-AAVE平行推理基准ReDial（1216对），系统评估LLM在方言输入下的公平性与鲁棒性，发现几乎所有主流模型在AAVE查询上性能显著下降超过10%。

**[ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)**

:   建立了 ELBA-Bench——一个涵盖 12 种攻击方法、18 个数据集和 12 个 LLM 的综合后门攻击基准，系统评估 PEFT 和无微调两种范式下 LLM 后门攻击的有效性和隐蔽性。

**[Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)**

:   提出集成水印方法，将文体特征（藏头词 acrostic + 感觉运动词 sensorimotor norms）与已有红绿水印组合，在 paraphrasing 攻击后三特征集成检测率达 95%，而单独红绿水印仅 49%。

**[Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models](estimating_privacy_leakage_of_augmented_contextual_knowledge_in_language_models.md)**

:   本文提出context influence指标，基于差分隐私框架量化语言模型在解码时对增强上下文知识的隐私泄露程度，并系统分析了模型大小、上下文大小、生成位置等因素对隐私泄露的影响。

**[Exploring Forgetting in Large Language Model Pre-Training](exploring_forgetting_in_large_language_model_pre-training.md)**

:   系统性地探索了 LLM 预训练阶段的灾难性遗忘问题，提出了基于实体记忆的新指标（M_ex、M_in）替代传统 PPL 来检测遗忘，并验证了周期性高强度 memory replay 策略在缓解预训练遗忘中的有效性。

**[Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs](fairness_difference_awareness.md)**

:   挑战当前LLM公平性评估中"差异无视"(difference unawareness)的主导范式，提出DiffAware和CtxtAware两个指标和包含8个场景16K问题的基准套件，证明在法律、文化、伤害评估等场景中模型应当区分群体差异，而现有去偏方法反而损害了这种必要的差异感知能力。

**[Faithful and Robust LLM-Driven Theorem Proving for NLI Explanations](faithful_and_robust_llm-driven_theorem_proving_for_nli_explanations.md)**

:   本文研究 LLM 与定理证明器（TP）的交互架构，提出四种策略来缓解自动形式化中的语义信息损失、语法错误、证明构造不足和反馈解读困难等问题，在 e-SNLI、QASC 和 WorldTree 三个数据集上分别实现了形式化精度 +18.46%/+34.2%/+39.77% 和解释质量 +29.5%/+51.5%/+41.25% 的显著提升。

**[From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs](from_misleading_queries_to_accurate_answers_a_three-stage_fine-tuning_method_for.md)**

:   提出三阶段微调方法（误导检测->查询纠正->准确回答）增强 LLM 处理含误导信息输入的能力，在误导检测和 QA 任务上显著提升准确率，同时减少幻觉生成。

**[From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models](from_tradeoff_to_synergy_a_versatile.md)**

:   提出SymMark共生水印框架，融合logits-based和sampling-based两类水印方法（串行/并行/混合三种策略），通过token熵和语义熵自适应选择水印策略，在可检测性、鲁棒性、文本质量和安全性上实现SOTA。

**[Hallucination Detox: Sensitivity Dropout (SenD) for Large Language Model Training](hallucination_detox_send.md)**

:   提出Sensitivity Dropout (SenD)训练协议，通过识别并确定性丢弃训练过程中波动最大的嵌入索引（Sensitive Embedding Indices），减少LLM训练中幻觉的振荡行为，同时提出高效EigenScore近似方法(EES)实现2倍加速。

**[HALoGEN: Fantastic LLM Hallucinations and Where to Find Them](halogen_hallucinations.md)**

:   提出 HALoGEN——覆盖 9 个领域（含编程、科学引用、摘要等）的 10,923 条 prompt 的大规模幻觉评测框架，配套原子级自动验证器，在 14 个 LLM 的约 150,000 条生成上系统性评估幻觉，发现即使最佳模型也可能有高达 86% 的原子事实存在幻觉，并提出 Type A/B/C 三类错误分类法。

**[HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs](hd-ndes_neural_differential_equations_for_hallucination_detection_in_llms.md)**

:   本文首次将神经微分方程（Neural DEs）应用于LLM幻觉检测，通过对隐空间中token激活的连续轨迹建模来系统评估陈述的真实性，在True-False数据集上AUC-ROC超过SOTA 14%以上。

**[How Does Response Length Affect Long-Form Factuality](how_does_response_length_affect_long-form_factuality.md)**

:   本文系统研究了LLM响应长度与事实精确度的关系，提出高效的双层事实性评估框架Bafe（与人类注释89.31%一致），确认了长度偏差的存在，并通过排除错误传播和长上下文假说，证明"事实耗竭"是事实性下降的主要原因。

**[Improved Unbiased Watermark for Large Language Models](improved_unbiased_watermark_for_large_language.md)**

:   提出 MCmark，一族基于多通道（Multi-Channel）的无偏水印算法，通过将词表分割为 $l$ 个段并在选中段内提升 token 概率来嵌入统计信号，在保持 LLM 原始输出分布的同时，可检测性比现有无偏水印提升超 10%。

**[Ewe: Improving Factuality with Explicit Working Memory](improving_factuality_with_explicit_working_memory.md)**

:   提出 Ewe（Explicit Working mEmory），在 LLM 解码过程中引入由多个 KV cache 单元组成的显式工作记忆，实时接收检索知识反馈和事实核查反馈，检测到错误时删除错误句子并用更新后的记忆重新生成，在 4 个事实性长文本生成基准上将 VeriScore F1 提升 2–6 分且不损失回答有用性。

**[Improving Fairness of Large Language Models in Multi-document Summarization](improving_fairness_of_large_language_models_in_multi-document_summarization.md)**

:   提出 FairPO（Fair Preference Optimization），通过扰动式偏好对生成和公平感知偏好调优，同时优化多文档摘要中的摘要级和语料级公平性。

**[Improving Model Factuality with Fine-grained Critique-based Evaluator](improving_model_factuality_with_fine-grained_critique-based_evaluator.md)**

:   训练细粒度的事实性评估器 FenCE，通过在公开数据集上增强文本批评（critique）和多工具获取的多样化源文档来提升评估准确率，并利用 FenCE 对生成器响应进行修订和评分以构建偏好训练数据，使 Llama2-7B/Llama3-8B 在 FActScore 上分别提升 16.86%/14.45%。

**[Can Indirect Prompt Injection Attacks Be Detected and Removed?](indirect_prompt_injection_detection.md)**

:   本文系统研究间接 prompt injection 攻击的检测与移除：构建评估基准，发现现有检测模型对间接攻击表现不佳但专门训练的模型可达 99% 准确率，提出分割移除和抽取移除两种方法，并将检测+移除组合为过滤管道，有效降低间接 prompt injection 的攻击成功率。

**[Beyond Facts: Evaluating Intent Hallucination in Large Language Models](intent_hallucination_eval.md)**

:   本文提出"意图幻觉"（Intent Hallucination）概念——LLM 在处理复杂多条件查询时遗漏或误解部分意图约束导致的偏离用户意图的生成，构建 FaithQA 基准（20,068 题）和 Constraint Score 评估指标，实验表明意图幻觉在 SOTA 模型中普遍存在且随查询复杂度增加而加剧。

**[Jailbreak Large Vision-Language Models Through Multi-Modal Linkage](jailbreak_large_vision-language_models_through_multi-modal_linkage.md)**

:   提出多模态链接（MML）攻击框架，通过跨模态加密-解密机制和"邪恶对齐"策略，以极高成功率（GPT-4o上达99%+）越狱当前最先进的视觉语言模型。

**[Language Models Can Subtly Deceive Without Lying: A Case Study on Strategic Phrasing](language_models_can_subtly_deceive_without_lying_a_case_study_on_strategic_phras.md)**

:   构建了一个立法环境测试平台（LobbyLens），研究 LLM 是否能通过策略性措辞（strategic phrasing）——即不说谎但有意操纵表达方式——来隐藏修正案中对特定公司的利益导向，发现 LLM 经过 re-planning 可使欺骗率提升最多 40 个百分点。

**[Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation](learning_auxiliary_tasks_improves_reference-free_hallucination_detection_in_open.md)**

:   系统性地研究了开放域长文本生成中的无参考幻觉检测问题，发现 LLM 内部状态（概率/熵）不足以可靠区分事实与幻觉内容，并提出 RATE-FT（Rationale and Auxiliary Task Enhanced Fine-Tuning），通过引入推理解释和辅助 QA 任务增强微调，在 LongFact 上比普通微调提升 3% 以上。

**[Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?](llm_watermark_distillation_robustness.md)**

:   本文首次系统研究 LLM 水印在防止未授权知识蒸馏中的鲁棒性，提出三种水印去除攻击（无目标/有目标释义 + 推理时水印中和），发现有目标释义和水印中和可以彻底去除继承的水印，其中水印中和在保持知识迁移效率的同时实现零额外训练开销的水印去除。

**[Fine-grained Hallucination Detection and Mitigation in Long-form Question Answering](localizing_and_mitigating_errors_in_long-form_question_answering.md)**

:   本文构建了首个包含 span 级别错误标注的 LFQA 幻觉数据集 HaluQuestQA（698 QA 对，4.7k 错误标注，5 种错误类型），训练了一个自动反馈模型来检测不完整信息的错误 span 并生成解释，最后提出 Error-informed Refinement 方法利用反馈信号精炼答案，将幻觉减少约 3%，且人类评估中 84% 的用户更偏好精炼后的答案。

**[Mamba Knockout for Unraveling Factual Information Flow](mamba_knockout_for_unraveling_factual_information_flow.md)**

:   将 Transformer 上的 Attention Knockout 可解释性方法迁移至 Mamba-1 和 Mamba-2，揭示了 SSM 模型中事实信息的流动模式——发现 Mamba 与 Transformer 共享"主语 token 在中后层向最后 token 传递关键信息"的普遍模式，但在首 token 偏置和关系 token 依赖等方面存在架构特异性差异。

**[Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](merge_hijacking_backdoor_attacks_to_model_merging_of_large_language_models.md)**

:   提出 Merge Hijacking——首个针对 LLM 模型合并的后门攻击方法，攻击者仅需上传一个恶意模型，当受害者将其与任意干净模型合并时，生成的合并模型继承后门并在所有任务上保持攻击有效性和正常性能，且对现有防御方法具有鲁棒性。

**[Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation](monitoring_decoding_mitigating_hallucination_via_evaluating_the_factuality_of_pa.md)**

:   提出 Monitoring Decoding (MD) 框架，在生成过程中动态监控部分响应的事实性，通过监控函数识别易产生幻觉的 token 并利用树搜索策略选择性地修正这些关键 token，从而在保持效率的同时显著提升事实准确性。

**[MorphMark: Flexible Adaptive Watermarking for Large Language Models](morphmark_adaptive_watermarking.md)**

:   MorphMark 通过多目标权衡分析框架揭示了绿表概率 P_G 在水印效果与文本质量之间的关键作用，并据此提出自适应调整水印强度 r 的方法——当 P_G 高时增强水印、P_G 低时减弱水印，实现了在不依赖额外模型训练的前提下同时提升水印可检测性和文本质量。

**[On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation](on-policy_self-alignment_with_fine-grained_knowledge_feedback_for_hallucination_.md)**

:   提出 RLFH（Reinforcement Learning for Hallucination），一种在策略（on-policy）自对齐方法，让 LLM 自己作为评判者，将回复分解为原子事实并进行真实性和信息量评估，生成 token 级别的密集奖励信号，通过在线 PPO 优化来有效缓解幻觉问题。

**[Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport](opt-out_investigating_entity-level_unlearning_for_large_language_models_via_opti.md)**

:   提出 Opt-Out，一种基于最优传输理论的实体级 LLM 遗忘方法，利用 Sliced Wasserstein Distance 正则化参数偏移实现精细遗忘；同时构建首个实体级遗忘数据集 ELUDe（20 目标实体 + 144 邻居实体，15K+ forget / 90K+ retain QA 对），在 Llama-3.1-8B 和 Phi-3.5 上全面超越现有方法。

**[Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models](private_memorization_editing_turning_memorization_into_a_defense_to_strengthen_d.md)**

:   提出 PME（Private Memorization Editing），将 LLM 的记忆化特性从安全弱点转化为防御手段，通过编辑 Feed Forward 层参数来移除已记忆的个人身份信息（PII），实现无需重训的隐私保护。

**[Real-time Factuality Assessment from Adversarial Feedback](real-time_factuality_assessment_from_adversarial_feedback.md)**

:   本文揭示了现有事实性评估数据集存在"数据泄漏"问题（LLM 因预训练记忆而轻松识别旧虚假信息），提出了一个基于 RAG 检测器对抗反馈的迭代改写流水线来生成真正具有挑战性的实时虚假新闻变体，使 GPT-4o RAG 检测器的 ROC-AUC 绝对下降 17.5%。

**[ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)**

:   ReLearn提出用"正向学习"替代传统的"逆向优化"来实现LLM知识遗忘，通过数据增强和微调pipeline使模型在遗忘目标知识的同时保持语言生成质量和流畅性，并设计了包含KFR、KRR和LS三个指标的综合评估框架。

**[REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space](revs_unlearning_sensitive_information_in_language_models_via_rank_editing_in_the.md)**

:   提出 REVS，一种无梯度的模型编辑方法，通过在 FF2 层中定位与敏感 token 关联最强的神经元，将其投影到词汇空间后迭代降低目标 token 排名，在 SSN/Email/URL 三类敏感数据上 Unlearning Score 显著超越 6 种基线（89.58 vs 36.98），同时通用能力几乎零损（MMLU 61.05→60.87），且对 Logit-Lens 和 Delta 提取攻击高度鲁棒。

**[Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](robust_data_watermarking_in_language_models_by_injecting_fictitious_knowledge.md)**

:   提出一种基于虚构知识（Fictitious Knowledge）的数据水印方法，通过在训练数据中注入虚构但合理的实体及其属性描述，实现对 LLM 训练数据所有权的可追溯验证，水印抗数据预处理过滤且支持黑盒 QA 验证。

**[SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models](saferoute_adaptive_model_selection_for_efficient_and_accurate_safety_guardrails_.md)**

:   提出 SafeRoute，一个二分类路由器，根据输入难度自适应地在小型和大型安全护栏模型之间选择，仅对约5%的"困难"样本使用大模型，在保持安全检测精度的同时大幅降低计算开销。

**[SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?](seuf_is_unlearning_one_expert_enough_for_mixture-of-experts_llms.md)**

:   SEUF 首次揭示现有 LLM 遗忘方法在 MoE 模型上严重失效（效用下降 35%+），根因是遗忘过程导致路由器的专家选择漂移形成"捷径"——本该遗忘的目标专家被绕过而无辜专家被破坏，并提出通过专家归因定位目标专家+路由器锚定损失固定选择的框架，仅更新 0.06% 参数即可同时提升遗忘质量和模型效用。

**[Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs](stochastic_chameleons_irrelevant_context_hallucinations_reveal_class-based_misge.md)**

:   通过行为分析和机械可解释性实验揭示 LLM 无关上下文幻觉的内部机制：模型在底层构建抽象类别表示（如"语言"），然后两条竞争电路（query-based vs context-based）争夺特征选择权，相对激活强度决定正确泛化还是产生幻觉。

**[TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](tip_iceberg_adversarial_attacks.md)**

:   本文提出 Task-in-Prompt (TIP) 攻击——一类通过在 prompt 中嵌入序列到序列任务（如密码解码、谜语、代码执行）来间接生成违禁内容的新型越狱攻击类别，并构建 PHRYGE benchmark 系统评估，证明该攻击可成功绕过 GPT-4o、LLaMA 3.2 等六种 SOTA LLM 的安全防护。

**[Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach](towards_context-robust_llms_a_gated_representation_fine-tuning_approach.md)**

:   提出 Grft（Gated Representation Fine-Tuning），一种轻量级即插即用的门控表示微调方法，仅需不到 200 个训练样本和模型 0.0004% 的参数，即可让 LLM 在面对矛盾、无用的外部上下文时表现出类似人类的鲁棒认知行为。

**[Towards Effective Extraction and Evaluation of Factual Claims](towards_effective_extraction_and_evaluation_of_factual_claims.md)**

:   提出了一个用于评估事实声明抽取质量的标准化框架（包含覆盖率和去语境化等指标），并开发了Claimify——一个能在高置信度下处理歧义并抽取声明的LLM方法，在该框架下显著优于已有方法。

**[TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation](treecut_a_synthetic_unanswerable_math_word_problem_dataset_for_llm_hallucination.md)**

:   提出 TreeCut，一种基于树结构的合成数据集生成方法，通过在树路径上移除必要条件边来系统性生成无穷多的不可回答数学应用题，用以评估 LLM 在面对不可解问题时的幻觉行为。

**[Truth Knows No Language: Evaluating Truthfulness Beyond English](truth_knows_no_language_evaluating_truthfulness_beyond_english.md)**

:   构建首个专业翻译的多语言 TruthfulQA 基准（巴斯克语、加泰罗尼亚语、加利西亚语、西班牙语），发现 LLM 的跨语言真实性差异小于预期，且 LLM-as-a-Judge 比多选题指标更贴合人类判断。

**[The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](tug_of_war_fairness_privacy.md)**

:   发现 LLM 通过 SFT 增强隐私意识会显著降低公平性意识（trade-off），提出无训练方法 SPIN（抑制公平-隐私耦合神经元），基于信息论解耦两种意识，在 Qwen2-7B 上同时提升公平性 12.2% 和隐私意识 14.0%。

**[UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models](ualign_leveraging_uncertainty_estimations_for_factuality_alignment_on_large_lang.md)**

:   提出 UAlign 框架，利用置信度分数和语义熵两种不确定性估计来显式建模 LLM 知识边界，并将其作为输入特征融入 PPO 对齐训练，使模型自信回答已知问题、坚定拒绝未知问题，在多个知识 QA 数据集上显著提升可靠性与泛化性。

**[Unveiling and Addressing Pseudo Forgetting in Large Language Models](unveiling_and_addressing_pseudo_forgetting_in_large_language_models.md)**

:   揭示 LLM 持续学习中的"伪遗忘"现象：性能下降并非因为模型丧失了旧任务能力，而是指令无法正确激活已有能力。通过归因分析证明遗忘模型的指令依赖度降低，并提出基于 Rationale-Guidance Difficulty（RGD）的动态数据回放框架 RGD-R 来缓解伪遗忘。

**[When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations](when_backdoors_speak_understanding_llm_backdoor_attacks_through_model-generated_.md)**

:   本文首次从自然语言解释的角度研究 LLM 后门攻击，发现后门模型对干净输入生成逻辑连贯的解释，但对中毒输入生成多样且逻辑有缺陷的解释；进一步通过 token 级和句子级分析揭示中毒样本的预测语义仅在最后几层才出现，且注意力从输入上下文转移到新生成的 token。

**[Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning](which_retain_set_matters_for_llm_unlearning_a_case_study_on_entity_unlearning.md)**

:   系统研究实体遗忘中 retain set 的选择问题，提出 Syntactically Similar Neighbor Set，发现句法相似性（而非领域/实体相似性）才是遗忘过程中知识退化的主要驱动因素，用句法相似的 retain set 做正则化可同时最优保护所有类型的邻居知识。

**[ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)**

:   在 SemEval-2025 Task 4（LLM 敏感内容遗忘）中获得第二名，核心思路是训练两个互补模型（一个过度遗忘、一个遗忘不足），通过 TIES-Merging 合并得到平衡遗忘的模型，本地实验达到近乎完美的 MIA 分数 0.501。
