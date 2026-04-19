---
title: >-
  ACL2025 AI安全方向 28篇论文解读
description: >-
  28篇ACL2025 AI安全方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI安全

**💬 ACL2025** · **28** 篇论文解读

**[CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks](cavgan_unifying_jailbreak_and_defense_of_llms_via_generative_adversarial_attacks.md)**

:   提出 CAVGAN 框架，利用生成对抗网络在 LLM 内部表示空间中同时学习越狱攻击（生成器）和安全防御（判别器），首次将攻防统一到同一框架中实现"攻防共进"，攻击成功率平均 88.85%，防御成功率平均 84.17%。

**[CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference](centaur_bridging_the_impossible_trinity_of.md)**

:   提出 Centaur 框架，融合随机置换矩阵和安全多方计算（SMPC）来打破隐私保护 Transformer 推理（PPTI）中的"不可能三角"——同时实现强隐私保护、5-30x 加速和明文级别推理精度。

**[ReDial: Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](dialect_fairness_robustness.md)**

:   本文构建了首个高质量人工标注的标准英语-AAVE平行推理基准ReDial（1216对），系统评估LLM在方言输入下的公平性与鲁棒性，发现几乎所有主流模型在AAVE查询上性能显著下降超过10%。

**[ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)**

:   建立了 ELBA-Bench——一个涵盖 12 种攻击方法、18 个数据集和 12 个 LLM 的综合后门攻击基准，系统评估 PEFT 和无微调两种范式下 LLM 后门攻击的有效性和隐蔽性。

**[Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)**

:   提出集成水印方法，将文体特征（藏头词 acrostic + 感觉运动词 sensorimotor norms）与已有红绿水印组合，在 paraphrasing 攻击后三特征集成检测率达 95%，而单独红绿水印仅 49%。

**[Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models](estimating_privacy_leakage_of_augmented_contextual_knowledge_in_language_models.md)**

:   本文提出context influence指标，基于差分隐私框架量化语言模型在解码时对增强上下文知识的隐私泄露程度，并系统分析了模型大小、上下文大小、生成位置等因素对隐私泄露的影响。

**[FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes](fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s.md)**

:   本文提出 Indic-Bias，首个面向印度多元社会的大规模 LLM 公平性基准，通过 20,000 个人工验证的场景模板在三大评估任务上测试 14 个 LLM，揭示模型对达利特等边缘化群体存在严重负面偏见，且超过 70% 的情况下会强化刻板印象。

**[Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs](fairness_difference_awareness.md)**

:   挑战当前LLM公平性评估中"差异无视"(difference unawareness)的主导范式，提出DiffAware和CtxtAware两个指标和包含8个场景16K问题的基准套件，证明在法律、文化、伤害评估等场景中模型应当区分群体差异，而现有去偏方法反而损害了这种必要的差异感知能力。

**[From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models](from_tradeoff_to_synergy_a_versatile.md)**

:   提出SymMark共生水印框架，融合logits-based和sampling-based两类水印方法（串行/并行/混合三种策略），通过token熵和语义熵自适应选择水印策略，在可检测性、鲁棒性、文本质量和安全性上实现SOTA。

**[Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework](gifi_gender_fairness.md)**

:   提出 GIFI（Gender Inclusivity Fairness Index），一个涵盖代词识别、情感中立性、毒性、反事实公平性、刻板印象关联、职业公平性和数学推理一致性七个维度的多层次评估框架，在 22 个主流 LLM 上系统量化二元与非二元性别的公平性，揭示新代词在无提示时完全缺席、"she" 过度矫正等深层偏见模式。

**[Improved Unbiased Watermark for Large Language Models](improved_unbiased_watermark_for_large_language.md)**

:   提出 MCmark，一族基于多通道（Multi-Channel）的无偏水印算法，通过将词表分割为 $l$ 个段并在选中段内提升 token 概率来嵌入统计信号，在保持 LLM 原始输出分布的同时，可检测性比现有无偏水印提升超 10%。

**[Improving Fairness of Large Language Models in Multi-document Summarization](improving_fairness_of_large_language_models_in_multi-document_summarization.md)**

:   提出 FairPO（Fair Preference Optimization），通过扰动式偏好对生成和公平感知偏好调优，同时优化多文档摘要中的摘要级和语料级公平性。

**[Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?](llm_watermark_distillation_robustness.md)**

:   本文首次系统研究 LLM 水印在防止未授权知识蒸馏中的鲁棒性，提出三种水印去除攻击（无目标/有目标释义 + 推理时水印中和），发现有目标释义和水印中和可以彻底去除继承的水印，其中水印中和在保持知识迁移效率的同时实现零额外训练开销的水印去除。

**[MorphMark: Flexible Adaptive Watermarking for Large Language Models](morphmark_adaptive_watermarking.md)**

:   MorphMark 通过多目标权衡分析框架揭示了绿表概率 P_G 在水印效果与文本质量之间的关键作用，并据此提出自适应调整水印强度 r 的方法——当 P_G 高时增强水印、P_G 低时减弱水印，实现了在不依赖额外模型训练的前提下同时提升水印可检测性和文本质量。

**[Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries](multi-task_adversarial_attacks_against_black-box_model_with_few-shot_queries.md)**

:   提出 CEMA（Cluster and Ensemble Multi-task Text Adversarial Attack）方法，通过训练"深层替代模型"将复杂的多任务黑盒攻击转化为单任务文本分类攻击，仅需约 100 次查询即可同时攻击分类、翻译、摘要、文生图等多种任务，并在 ChatGPT-4o、百度翻译、Stable Diffusion 等商用模型上验证了有效性。

**[PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance](privacibench_evaluating_privacy_with_contextual_integrity.md)**

:   提出 PrivaCI-Bench，基于 Contextual Integrity 理论构建了目前最大的上下文隐私评估基准（154K 实例），涵盖真实法院案例、隐私政策和 EU AI Act 合规检查器合成数据，评估 LLM 在 HIPAA/GDPR/AI Act 下的法律合规能力。

**[Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models](private_memorization_editing_turning_memorization_into_a_defense_to_strengthen_d.md)**

:   提出 PME（Private Memorization Editing），将 LLM 的记忆化特性从安全弱点转化为防御手段，通过编辑 Feed Forward 层参数来移除已记忆的个人身份信息（PII），实现无需重训的隐私保护。

**[Quantifying Misattribution Unfairness in Authorship Attribution](quantifying_misattribution_unfairness_in_authorship_attribution.md)**

:   本文提出MAUI_k指标量化作者归因系统中"错误归因不公平性"——某些作者系统性地更容易被误判为可疑作者，并发现这种不公平与作者嵌入在向量空间中距质心的距离高度相关。

**[Robust and Minimally Invasive Watermarking for EaaS](robust_and_minimally_invasive_watermarking_for_eaas.md)**

:   提出 ESpeW（Embedding-Specific Watermark），一种嵌入特异性水印方法，通过在每个嵌入向量的不同位置注入独特水印，实现对 Embeddings as a Service (EaaS) 的鲁棒版权保护，抵抗各种水印移除攻击且对嵌入质量的影响小于 1%。

**[Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](robust_data_watermarking_in_language_models_by_injecting_fictitious_knowledge.md)**

:   提出一种基于虚构知识（Fictitious Knowledge）的数据水印方法，通过在训练数据中注入虚构但合理的实体及其属性描述，实现对 LLM 训练数据所有权的可追溯验证，水印抗数据预处理过滤且支持黑盒 QA 验证。

**[Sandcastles in the Storm: Revisiting Watermarking Impossibility](sandcastles_watermarking_impossibility.md)**

:   本文通过大规模实验和人类评估挑战了 "Watermarks in the Sand" (WITS) 的理论不可能性结论：证明随机游走攻击的两个关键假设在实践中不成立——混合(mixing)速度极慢（100% 的攻击文本仍可追溯原始来源）且质量预言机(quality oracle)不可靠（仅 77% 准确率），自动攻击仅 26% 成功率，人类质量审核后降至 10%。

**[SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods](speechfake_a_largescale_multilingual_speech_deepfake.md)**

:   构建 SpeechFake 大规模语音深伪数据集，包含 300 万+深伪样本、3000+ 小时音频、40 种生成工具和 46 种语言，并通过基线实验系统分析了生成方法、语言多样性和说话人变化对检测性能的影响。

**[TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](tip_iceberg_adversarial_attacks.md)**

:   本文提出 Task-in-Prompt (TIP) 攻击——一类通过在 prompt 中嵌入序列到序列任务（如密码解码、谜语、代码执行）来间接生成违禁内容的新型越狱攻击类别，并构建 PHRYGE benchmark 系统评估，证明该攻击可成功绕过 GPT-4o、LLaMA 3.2 等六种 SOTA LLM 的安全防护。

**[Towards Fairness Assessment of Dutch Hate Speech Detection](towards_fairness_assessment_of_dutch_hate_speech_detection.md)**

:   本文系统评估了荷兰语仇恨言论检测模型的反事实公平性，提出四种反事实数据生成方法（LLMdef、LLMlist、SLL、MGS），并通过在 BERTje 模型上微调验证了反事实数据增强对模型性能和公平性的改进效果。

**[The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](tug_of_war_fairness_privacy.md)**

:   发现 LLM 通过 SFT 增强隐私意识会显著降低公平性意识（trade-off），提出无训练方法 SPIN（抑制公平-隐私耦合神经元），基于信息论解耦两种意识，在 Qwen2-7B 上同时提升公平性 12.2% 和隐私意识 14.0%。

**[Efficiently Identifying Watermarked Segments in Mixed-Source Texts](watermark_segment_detection.md)**

:   提出两种高效方法（Geometric Cover Detector 和 Adaptive Online Locator）用于在长文混合来源文本中检测和精确定位水印片段，时间复杂度从 O(n²) 降至 O(n log n)，在三种主流水印技术上均显著优于 baseline。

**[WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermark](wet_eaas_watermark.md)**

:   揭示了现有 EaaS 嵌入水印（EmbMarker/WARDEN）可被改写攻击绕过，提出 WET（线性变换水印），通过秘密循环矩阵对嵌入做线性变换注入水印，理论和实验证明其对改写攻击具有鲁棒性，验证 AUC 接近 100%。

**[When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations](when_backdoors_speak_understanding_llm_backdoor_attacks_through_model-generated_.md)**

:   本文首次从自然语言解释的角度研究 LLM 后门攻击，发现后门模型对干净输入生成逻辑连贯的解释，但对中毒输入生成多样且逻辑有缺陷的解释；进一步通过 token 级和句子级分析揭示中毒样本的预测语义仅在最后几层才出现，且注意力从输入上下文转移到新生成的 token。
