---
title: >-
  ACL2026 LLM 效率方向8篇论文解读
description: >-
  8篇ACL2026的 LLM 效率方向论文解读，涵盖对齐/RLHF、LLM、推理、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# ⚡ LLM 效率

**💬 ACL2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (4)](../../CVPR2026/llm_efficiency/index.md) · [🔬 ICLR2026 (18)](../../ICLR2026/llm_efficiency/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/llm_efficiency/index.md) · [🧠 NeurIPS2025 (35)](../../NeurIPS2025/llm_efficiency/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/llm_efficiency/index.md) · [🧪 ICML2025 (13)](../../ICML2025/llm_efficiency/index.md)

🔥 **高频主题：** 对齐/RLHF ×2

**[BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs](bosch_black-box_binary_optimization_for_short-context_attention-head_selection_i.md)**

:   提出 BOSCH，一种免训练的注意力头级别 SWA 混合方法，将 SWA 头选择建模为大邻域搜索问题并分解为三阶段优化（层重要性探测→自适应比例分配→分组头选择），在 4 个模型 4 种比例设置下系统性超越层级启发式和 6 种静态头级别方法。

**[HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns](humanllm_benchmarking_and_improving_llm_anthropomorphism_via_human_cognitive_pat.md)**

:   本文提出 HumanLLM 框架，将 244 个心理学模式（100 个人格特质 + 144 个社会认知模式）建模为相互作用的因果力而非孤立标签，构建了 11,359 个包含 2-5 个模式交互的场景和多轮对话数据集，通过双层 checklist 评估实现与人类判断的高对齐（$r=0.90$），HumanLLM-8B 在多模式动态上以 4 倍小的参数量超越 Qwen3-32B。

**[Multi-Drafter Speculative Decoding with Alignment Feedback](multi-drafter_speculative_decoding_with_alignment_feedback.md)**

:   本文提出 MetaSD，一个将多个异构草稿器整合到推测解码中的统一框架，将草稿器选择建模为多臂赌博机问题，通过块散度（Block Divergence）奖励信号动态选择与目标 LLM 最对齐的草稿器，在黑盒和白盒配置下一致优于单草稿器方法。

**[Native Hybrid Attention for Efficient Sequence Modeling](native_hybrid_attention_for_efficient_sequence_modeling.md)**

:   本文提出 Native Hybrid Attention (NHA)，将线性 RNN 的长期记忆槽与滑动窗口的短期精确 token 拼接后通过单次 softmax 注意力统一处理，实现层内和层间混合的原生统一——无需额外融合参数即可动态分配长短期注意力权重，在 recall 密集和常识推理任务上超越 Transformer 和其他混合基线。

**[Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)**

:   本文提出 PLOTTER 框架，首次将叙事规划从文本表示转移到图结构表示（事件图+角色图），通过多 agent 的 Evaluate-Plan-Revise 迭代循环在图拓扑上诊断和修复叙事缺陷，在叙事性、角色塑造、戏剧张力等维度上显著优于现有方法。

**[SciCoQA: Quality Assurance for Scientific Paper–Code Alignment](scicoqa_quality_assurance_for_scientific_paper--code_alignment.md)**

:   本文提出 SciCoQA，首个用于检测科学论文与其代码实现之间差异的基准数据集，包含 635 个差异实例（92 个真实 + 543 个合成），评测 22 个 LLM 后发现最强模型仅能检测 46.7% 的真实差异，揭示了自动化科学质量保证中的关键能力缺口。

**[SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration](specbound_adaptive_bounded_self-speculation_with_layer-wise_confidence_calibrati.md)**

:   提出 SpecBound 自草稿推测解码框架，通过逐层温度退火抑制浅层虚假高置信度预测，并设计有界推测算法自适应控制草稿的深度和宽度，在保持输出无损的同时实现最高 2.33× 的推理加速。

**[Speculative Verification: Exploiting Information Gain to Refine Speculative Decoding](speculative_verification_exploiting_information_gain_to_refine_speculative_decod.md)**

:   提出推测验证（Speculative Verification, SV），通过引入与草稿模型同等规模的伴随模型（companion model），利用草稿-伴随分布的相似性预测推测准确率，动态调整验证长度以最大化有效吞吐量，在大批量推理中实现相对标准推测解码平均1.4×、最高1.9×的加速。
