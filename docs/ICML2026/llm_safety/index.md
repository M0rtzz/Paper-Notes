---
title: >-
  ICML2026 LLM 安全方向18篇论文解读
description: >-
  18篇ICML2026的 LLM 安全方向论文解读，涵盖 LLM、对抗鲁棒、推理、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "LLM 安全"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "对抗鲁棒"
  - "推理"
  - "Agent"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔒 LLM 安全

**🧪 ICML2026** · **18** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (115)](../../ACL2026/llm_safety/index.md) · [📷 CVPR2026 (29)](../../CVPR2026/llm_safety/index.md) · [🔬 ICLR2026 (54)](../../ICLR2026/llm_safety/index.md) · [🤖 AAAI2026 (43)](../../AAAI2026/llm_safety/index.md) · [🧠 NeurIPS2025 (84)](../../NeurIPS2025/llm_safety/index.md) · [📹 ICCV2025 (13)](../../ICCV2025/llm_safety/index.md)

🔥 **高频主题：** LLM ×10 · 对抗鲁棒 ×5 · 推理 ×4 · Agent ×2

**[From Flat Facts to Sharp Hallucinations: Detecting Stubborn Errors via Gradient Sensitivity](from_flat_facts_to_sharp_hallucinations_detecting_stubborn_errors_via_gradient_s.md)**

:   本文把 LLM 幻觉检测从"看输出概率"切到"看 loss landscape 曲率"——在 embedding 加 Gaussian 噪声测量梯度方向与幅度的扰动，作为 Hessian 谱半径的廉价代理，在 12 个 model-dataset 组合上 AUROC 全面超越 entropy / Semantic Entropy / EigenScore 等基线。

**[From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)**

:   作者通过追踪 LoRA 微调过程中参数沿"危险/安全方向"的累积漂移，发现善意数据破坏对齐的根本机制是参数在 fine-tuning 中向危险方向单调漂移；进而提出 SQSD——用单步梯度沿两方向的投影差对每个样本打连续风险分，在 3 个模型 × 2 数据集上保持单调 ASR 排名，且能跨架构、跨规模、跨 LoRA→Full 迁移。

**[Harnessing Reasoning Trajectories for Hallucination Detection via Answer-agreement Representation Shaping](harnessing_reasoning_trajectories_for_hallucination_detection_via_answer-agreeme.md)**

:   本文针对大推理模型（LRM）的幻觉检测提出 ARS：不在文本层扰动 reasoning trace，而是**直接在 trace 末端的潜表示上施加小扰动并续解码**得到反事实答案，再用"答案是否一致"作为标签训一个轻量 contrastive 头来塑形 trace-conditioned answer embedding，使后续 embedding-based detector 把幻觉与真实回答分得更开（TruthfulQA 上 AUROC $66.85\to 86.64$）。

**[Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models](inducing_overthink_hierarchical_genetic_algorithm-based_dos_attack_on_black-box_.md)**

:   本文针对大型推理模型 (LRM) 易被"逻辑残缺输入"激发过度思考的弱点，提出一个层级化遗传算法 (HGA)，在纯黑盒条件下把结构化分解后的题目当成基因，通过句子级/问题级交叉和增删变异搜索逻辑断裂的对抗样本，最高可在 MATH 上把响应长度放大 26.1 倍，制造低成本 DoS 攻击。

**[Internalizing Safety Understanding in Large Reasoning Models via Verification](internalizing_safety_understanding_in_large_reasoning_models_via_verification.md)**

:   本文论证「会生成安全答案」≠「懂安全」，提出 SInternal 框架：只训练大型推理模型去 verify 自己生成答案的安全性，由此涌现的内在安全理解大幅压制 jailbreak 攻击（StrongREJECT ASR 从 41% 降到 0.6%）并成为后续 RL 的更好起点。

**[Jailbreaking Vision-Language Models Through the Visual Modality](jailbreaking_vision-language_models_through_the_visual_modality.md)**

:   作者提出 4 种只通过视觉输入就能越狱前沿 VLM 的攻击（视觉密码 / 物体替换 / 文本替换 / 视觉类比谜题），在 6 个前沿 VLM 上系统验证了"文本端的安全对齐不会自动迁移到视觉端"，并用 mechanistic 分析揭示了背后的层级机理。

**[Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)**

:   论文揭示了 Test-Time Scaling (TTS) 一个被忽视的失效模式——只要把候选回复的多样性压低，TTS 反而比直接喂高对抗性 prompt 更容易输出不安全内容；并提出 RefDiv，一个用 Shannon 熵 + 参考引导双信号驱动的遗传算法，能在 MCTS 和 Best-of-N 上跨模型、跨闭源、跨 guardrail 地高效越狱。

**[Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)**

:   把多轮 jailbreak 重新形式化为推理时的策略优化问题——在 adversarial POMDP 框架下，Attacker 与 Metacognitive Evaluator 构成闭环：Evaluator 输出的密集分析反馈被当作「语义梯度」来引导 Attacker 的 belief 更新与策略改进，从而在不重新训练任何权重的情况下，对包括 O1 / GPT-5-chat / Claude-3.7 在内的 10 个前沿模型平均 ASR 89.2%，token 消耗较强 baseline 平均降低 8.2 倍。

**[MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)**

:   MultiBreak 用"主动学习 + 不确定性引导改写"的迭代框架把多轮越狱数据集扩到 10,389 条对话、2,665 个独立有害意图，多样性 0.942 全面碾压前作，并在 DeepSeek-R1-7B / GPT-4.1-mini 上把 ASR 相比次优数据集分别提升 54% / 34.6%。

**[OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](otora_a_unified_red_teaming_framework_for_reasoning-level_denial-of-service_in_l.md)**

:   OTora 提出一种全新的攻击范式 Reasoning-Level Denial-of-Service（R-DoS）：不破坏任务正确性，而是通过两阶段红队管线（先用插入感知优化诱导 agent 主动访问攻击者控制的外部资源，再在该资源里投放经 ICL 遗传搜索优化的「思考型 payload」）让 LLM agent 进入持续多轮的过度推理状态，在 WebShop / Email / OS 三类 agent 上实现 10× 推理 token 膨胀和数量级延迟攻击，且最终任务准确率几乎不变。

**[REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations.md)**

:   REALISTA 在 LLM 隐空间里构造"输入相关的编辑方向字典"，把对抗 prompt 优化变成一个 simplex 约束下的连续问题，既保住了 SECA 这类离散方法的语义等价/连贯，又有 LARGO 那种连续方法的搜索灵活度，首次在 GPT-5 这类闭源推理模型 free-form 输出上诱发幻觉成功。

**[SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)**

:   SafeHarbor 把 LLM Agent 的安全防御从「静态粗粒度分类器」升级为「动态分层记忆树 + 双分数门控」，通过对抗规则生成 + 信息熵自演化让 GPT-4o 在保持 93%+ 拒绝率的同时把 benign 工具调用成功率拉到 63.6%，显著缓解 over-refusal 问题。

**[Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)**

:   本文证明所有现有「在参数空间设约束」的 HFT 防御都会因参数冗余而被绕过，提出 Safety Bottleneck Regularization (SBR) 把防御战场搬到 unembedding 层这一几何瓶颈上：仅锚定 1 个高危 prompt 的最后一层隐状态，就能在 50 epoch 持续 HFT 攻击下把 Harmful Score 压到 < 10，同时不损 benign 任务精度。

**[Self-Debias: Self-correcting for Debiasing Large Language Models](self-debias_self-correcting_for_debiasing_large_language_models.md)**

:   Self-Debias 把 LLM 的去偏问题重塑为「在自回归推理链上对概率质量做公平资源分配」：用轨迹级后缀边际作为资源单位，套 Jain 公平指数防止资源在易样本上塌缩，再配 cold-start SFT 与基于一致性过滤的在线自训练，仅用 20k 标注种子就让 Qwen3-8B 在 8 个 fairness/utility 基准上的平均分从 77.5 拉到 81.7，并把基础模型「自我纠错越纠越歪」的塌缩翻转成稳定 +0.4。

**[Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](stable-gflownet_toward_diverse_and_robust_llm_red-teaming_via_contrastive_trajec.md)**

:   本文指出现有 GFlowNet 红队的两大不稳定来源——partition function $Z_\theta$ 估计带来的高方差，与 toxicity classifier 给 OOD gibberish 文本的噪声 reward 引发的 mode collapse——并用三件简单组件（pairwise 对比目标 CTB 消除 $Z$、Noisy Gradient Pruning 过滤无信息 pair、Min-K Fluency Stabilizer 卡掉 gibberish）让红队攻击在 Qwen2.5-1.5B 上独特攻击数从 17 飙到 134（约 7×），ASR 维持 92%，且跨模型/跨防御迁移性全面碾压 baseline。

**[STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)**

:   本文把 T2I 模型的整个去噪轨迹本身当成 VLM 红队攻击的"攻击面"，用一个 high-level prompt editor + low-level GRPO 微调 rectified-flow 模型的分层 RL 框架（STARE），不仅把 attack success rate 比 SOTA 提升 68%，更揭示了一个全新现象——Optimization-Induced Phase Alignment：对抗优化会自动把"概念性毒性"绑到去噪早期、"细节性毒性"绑到后期，从而把混沌的毒性形成过程变成几个可预测的"漏洞时间窗"。

**[Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)**

:   本文用 Causal Tracing 在 LLM 内部发现"拒绝"不是终端 token 的静态向量、而是横跨上游中间层与 token 的"拒绝轨迹"(Refusal Trajectory)，并据此设计 SALO——一个只在常规对齐数据上训练、却能利用 Transformer 因果掩码不可逆性识别 GCG / AutoDAN / Prefilling 等对抗攻击的 <20M 参数检测器，把 GCG/Prefilling 上 0% 的检测率拉到 >85%。

**[Watermarking LLM Agent Trajectories (ACTHOOK)](watermarking_llm_agent_trajectories.md)**

:   ACTHOOK 把"软件 hook"思想搬进 agent 轨迹：在 action 边界处插入一个由秘密 key 触发的额外动作作为水印，被它训练过的 LLM 会在带 key 的 prompt 上以显著更高频率执行 hook，从而支持只通过黑盒查询就完成版权检测，平均 AUC 达 94.3 而几乎不影响下游任务表现。
