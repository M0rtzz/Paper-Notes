---
title: >-
  ICML2026 LLM 推理方向20篇论文解读
description: >-
  20篇ICML2026的 LLM 推理方向论文解读，涵盖推理、LLM、扩散模型、强化学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "LLM 推理"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "LLM"
  - "扩散模型"
  - "强化学习"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💡 LLM 推理

**🧪 ICML2026** · **20** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (64)](../../ACL2026/llm_reasoning/index.md) · [📷 CVPR2026 (12)](../../CVPR2026/llm_reasoning/index.md) · [🔬 ICLR2026 (63)](../../ICLR2026/llm_reasoning/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/llm_reasoning/index.md) · [🧠 NeurIPS2025 (66)](../../NeurIPS2025/llm_reasoning/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/llm_reasoning/index.md)

🔥 **高频主题：** 推理 ×10 · LLM ×4 · 扩散模型 ×2 · 强化学习 ×2

**[A Formal Comparison Between Chain of Thought and Latent Thought](a_formal_comparison_between_chain_of_thought_and_latent_thought.md)**

:   本文从计算复杂度理论出发，形式化比较 CoT（链式思维）与隐式思维（Looped Transformer / Coconut）的表达能力，证明隐式思维在多对数深度下严格达到 $\mathsf{TC}^k$，而 CoT 最多到 $\mathsf{TC}^{k-1}$；同时在概率设置下首次揭示 CoT 通过随机解码可支持 FPRAS 计数，反过来超越确定论隐式思维。

**[ANCHOR: Abductive Network Construction with Hierarchical Orchestration for Reliable Probability Inference in Large Language Models](anchor_abductive_network_construction_with_hierarchical_orchestration_for_reliab.md)**

:   ANCHOR 用"自底向上溯因 + 层级聚类" 构造稠密因子空间，对下游条件做粗到细检索得到稀疏相关因子集，再联合 Naïve Bayes 与一个 LLM 现场构造的潜变量因果贝叶斯网络做后验聚合，在 LLM 高风险决策场景中显著减少 "unknown" 预测并提升概率校准。

**[Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs](automated_formal_proofs_of_combinatorial_identities_via_wilf-zeilberger_guidance.md)**

:   WZ-LLM 把经典的 Wilf–Zeilberger 符号证明流程编译成 Lean 4 中可执行的证明骨架（递推 + 边界条件 + 侧条件），交给专门用 SFT + expert-iteration + DAPO 训练出的 WZ-Prover 逐项 discharge，在 100 个经典组合恒等式上把 pass@32 从 Goedel-Prover-V2 的 9% 提升到 34%。

**[Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)**

:   针对扩散语言模型 (dLLM) 半自回归生成时"块大小固定"破坏推理逻辑链的问题，本文提出 b1：用 RL 学一个块结束指示 token 来生成动态长度块，并用一个"块级熵单调下降 (Monotonic Entropy Descent, MED) 奖励"驱动连贯推理，作为即插即用的奖励项接入现有 dLLM RL 框架（Diffu-GRPO/GDPO/d1/wd1），在 Countdown 上将 wd1 从 39.45 推到 58.98。

**[Conformal Thinking: Risk Control for Reasoning on a Compute Budget](conformal_thinking_risk_control_for_reasoning_on_a_compute_budget.md)**

:   本文把"reasoning LLM 何时停止思考"从一个不可解释的阈值调参问题，重构为一个**用户可指定 risk 容忍度**的 conformal 风险控制问题：用两个阈值——上阈值在模型自信时停（控 false positive），新提出的**参数化下阈值**在模型在不可解题上"想不动"时强行停（控 false negative）——并通过 UCB 算法从校准集自动求出满足风险约束的阈值，在 AIME / GPQA / MathVision 上实现"准确率几乎不掉、token 大幅省"。

**[Efficient Reasoning with Hidden Thinking](efficient_reasoning_with_hidden_thinking.md)**

:   Heima 把多模态 LLM 的冗长 CoT 每个阶段（summary / caption / reasoning）蒸馏成**一个特殊 thinking token**，让模型在隐空间里"想"，token 数从 100-200 量级降到 13-16 个的同时 zero-shot 准确率反而比 LLaVA-CoT 更稳；配套训练一个 LLM "interpreter"用 thinking token 的 hidden state 重建出文字推理链，从而验证压缩损失的信息论上界。

**[Entropy-informed Decoding: Adaptive Information-Driven Branching](entropy-informed_decoding_adaptive_information-driven_branching.md)**

:   EDEN（Entropy-informed DEcodiNg）把每一步的束宽 $B_t$ 设成与归一化熵 $\bar H_t$ 单调正比——高熵 fork 多分支、低熵步骤近贪心——用更少的总扩展近似更宽的 beam search；理论上证明熵单调的分支因子在期望累计 regret 上严格优于任何固定束宽，且能给出 $\mathbb{E}[R_T] \leq G P_\max \sum_t \exp(-c m_t \Delta_\min^2)$ 的显式 regret 率。

**[ETS: Energy-Guided Test-Time Scaling for Training-Free RL Alignment](ets_energy-guided_test-time_scaling_for_training-free_rl_alignment.md)**

:   ETS 直接从 KL 正则化 RLHF 目标的**闭式最优解**采样，把它写成「参考策略 × 指数 reward 的条件期望（能量项）」，再用 Monte Carlo + 自归一化重要性采样在测试时近似这个能量项，从而**不训练**就达到甚至超过经过 RL 后训练的策略，并通过 lightweight proposal + Fast-dLLM 把延迟控制在可用范围。

**[Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs](express_your_doubts_--_probabilistic_world_modeling_should_not_be_based_on_token.md)**

:   这是一篇 position paper，主张：**用 LLM 的 token softmax 概率（logprob）当成"世界事件概率"是理论上错的**——因为 distribution estimation、response prediction 和 target distribution estimation 是三个不同任务，对应不同 ideal 输出分布；获取世界概率的正确做法是**二阶预测**——让 LLM 在输出里**显式写出**它对事件的概率（数值或语言修饰词），而不是去算"它说 X 的概率"。

**[Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)**

:   本文把 LLM 主动提问场景（20 Questions / 医疗诊断 / 故障排查）建模成两人零和扩展式博弈 (EFG)，提出 Game of Thought (GoT)：用深度有限的子博弈构造 + CFR 求 Nash 均衡来产生“随机化提问策略”，在所有数据集上把 worst-case 交互轮数显著降低，且 weighted 变体下相对 UoT 提升 15–40%。

**[GRPO is Secretly a Process Reward Model](grpo_is_secretly_a_process_reward_model.md)**

:   本文从理论上证明 GRPO + ORM 在"组内轨迹共享前缀"的温和条件下**等价于**一个带有 Monte-Carlo PRM 的过程奖励 RL 目标，从而揭示出 vanilla GRPO 隐藏的一个 bug——前缀长度不均会让高奖励轨迹的大部分 token 拿到负 advantage——并提出 $\lambda$-GRPO 做一个 PRM-aware 归一化，在推理 benchmark 上稳定超过 GRPO 且训练快约 2 倍。

**[Hidden Error Awareness in Chain-of-Thought Reasoning: The Signal Is Diagnostic, Not Causal](hidden_error_awareness_in_chain-of-thought_reasoning_the_signal_is_diagnostic_no.md)**

:   用一个简单的逻辑回归探针在 LLM 思维链生成时的隐藏状态上能以 0.95 AUROC 预测整条推理是否会出错（从第 1 步就有 0.79），但文本表面同样训出来的分类器只有 0.59；可惜 4 种干预手段（激活引导、探针引导 best-of-N、自我修正、激活补丁）全部失败——这个错误信号是"诊断性"的而非"因果性"的。

**[Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks](lifting_traces_to_logic_programmatic_skill_induction_with_neuro-symbolic_learnin.md)**

:   NSI 把 LLM agent 的交互轨迹 "提升" 为带显式条件分支和动态变量绑定的神经符号工作流图，使技能从无状态脚本进化成可状态感知的逻辑程序，在 ALFWorld / WebShop / TextCraft 上分别拿到 98.0 / 76.5 / 95.2 的成功率，全面碾压 ASI 和 AWM 等编程式技能基线。

**[Many-Shot CoT-ICL: Making In-Context Learning Truly Learn](many-shot_cot-icl_making_in-context_learning_truly_learn.md)**

:   本文系统揭示了非推理任务的 many-shot ICL “经验法则”在 CoT 推理任务上**全部失效**——相似度检索反而有害、顺序敏感性随 shot 数增长——并把成功的 many-shot CoT 重新解读为“in-context 测试时学习”，由此提出按 embedding 轨迹曲率排序 demonstration 的 CDS 方法，在 64-shot 几何题上提升 5.42 pp。

**[Multimodal Fact-Level Attribution for Verifiable Reasoning](multimodal_fact-level_attribution_for_verifiable_reasoning.md)**

:   MURGAT 是首个评测 MLLM 在多模态推理输出中"按事实粒度精确引用模态+时间段"能力的基准，搭配一个三步评估协议（可验证句识别 → 原子事实分解 → 归因质量）和高度与人工对齐的自动评测器 MURGAT-SCORE（Pearson 0.84），揭示了强模型即使答案对也常常胡乱引用，且强推理常以牺牲可验证引用为代价。

**[Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)**

:   作者把"为离散扩散语言模型（dLLM）做高效 test-time scaling"这一问题拆成三件事——按"探索→渐进剪枝→精修"的层级时间表分配计算（HTS）、用部分 remask 做局部分支保住高置信"逻辑骨架"、把 dLLM 自己当 Yes/No 验证器（SVF），最终在 4 个数学/代码基准、3 个 dLLM 上以远少于 best-of-$N$ 的 NFE 达到相近甚至更好的精度。

**[Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training](provable_benefit_of_curriculum_in_transformer_tree-reasoning_post-training.md)**

:   本文为「先易后难」的课程式 RL 后训练给出第一份严格的样本复杂度证明：在 transformer 的状态条件自回归推理树上，若课程能让相邻阶段的难度比保持在目标难度的 $L/p$ 次根级别，则总样本数可从直接训练的指数级 $(C^\star)^L$ 降到课程版的多项式级 $L\cdot (C^\star)^{p_\max}$。

**[ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)**

:   ResRL 从理论上把 RLVR 中 "负样本梯度污染正样本"现象 (Lazy Likelihood Displacement) 分解成"logit × 表征"两个分量,然后在表征层用正样本的 SVD 低秩子空间做投影残差,根据每个负 token 的"正交分量能量"给它一个 [ξ,1] 区间的梯度权重——表征越像正样本(残差越小)就罚得越轻,纯错误成分才被重罚,既保住 Pass@1 又不丢 Pass@k 多样性;在 Qwen3-4B 数学任务上 Avg@16 比 NSR 提升 9.4%,Pass@128 提升 7.0%。

**[ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)**

:   作者把 MATH 数据集的人工标注解题步骤逐步翻译成"带描述与类型签名的可复用 Python 工具"，构造出含 8K 题 + 12K 工具的 ToolMATH 基准；它同时覆盖长程多工具组合（hop 1-8+）、可控的干扰工具相似度（5 级 × 4 种密度）、以及"金标工具被全部移除"的工具缺失场景，验证显示模型失败的主导因素不是工具选择而是推理本身——thought error 占 90%+，而干扰工具会把早期的小偏差放大成不可逆的执行漂移。

**[Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards](unlocking_zero-shot_geospatial_reasoning_via_indirect_rewards.md)**

:   作者把"地面街景与卫星图能否定位为同一坐标"作为可验证间接奖励，用 GRPO 对 Qwen2.5-VL-7B 做两阶段后训练（CoT scaffolding + RL self-exploring），让模型仅凭 GPS metadata 学到可零样本迁移到 25+ 地理空间任务的通用推理能力。
