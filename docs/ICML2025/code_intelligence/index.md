---
title: >-
  ICML2025 代码智能方向11篇论文解读
description: >-
  11篇ICML2025的代码智能方向论文解读，涵盖推理、LLM、代码智能、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**🧪 ICML2025** · **11** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (20)](../../ACL2026/code_intelligence/index.md) · [📷 CVPR2026 (2)](../../CVPR2026/code_intelligence/index.md) · [🔬 ICLR2026 (21)](../../ICLR2026/code_intelligence/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/code_intelligence/index.md) · [🧠 NeurIPS2025 (22)](../../NeurIPS2025/code_intelligence/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/code_intelligence/index.md)

🔥 **高频主题：** 推理 ×3 · LLM ×3 · 代码智能 ×3 · 对抗鲁棒 ×2

**[AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence](adaptivestep_automatically_dividing_reasoning_step_through_model_confidence.md)**

:   提出 AdaptiveStep，基于模型预测下一个 token 的置信度自动划分推理步骤，替代传统基于规则（如换行符）的粗粒度划分方式，训练出的 PRM (ASPRM) 在数学推理和代码生成任务上达到 SOTA 的 Best-of-N 性能，且数据构建成本降低超 30%。

**[DyCodeEval: Dynamic Benchmarking of Reasoning Capabilities in Code Large Language Models Under Data Contamination](dynamic_benchmarking_of_reasoning_capabilities_in_code_large_language_models_und.md)**

:   基于蜕变测试思想，将编程问题分解为复杂度相关的算法抽象和复杂度无关的上下文描述，通过四个 LLM Agent 协作自动生成语义等价但文本不同的编程问题变体，有效规避数据污染并评估 Code LLM 的真实推理能力，在 18 个模型上验证了框架的有效性。

**[EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning](efficoder_enhancing_code_generation_in_large_language_models_through_efficiency-.md)**

:   EffiCoder 通过构建“正确且高效”的指令微调数据集 EffiInstruct，让代码大模型在提升 pass@1 的同时显著降低执行时间和总内存开销，证明“效率可以通过数据配方学习出来”。

**[EpiCoder: Encompassing Diversity and Complexity in Code Generation](epicoder_encompassing_diversity_and_complexity_in_code_generation.md)**

:   提出基于特征树（Feature Tree）的代码数据合成框架，通过从代码中提取层次化语义特征并迭代进化，实现对合成数据复杂度和多样性的精确控制，训练得到的 EpiCoder 系列模型在函数级和文件级代码生成基准上达到同规模 SOTA。

**[Function-to-Style Guidance of LLMs for Code Translation](function-to-style_guidance_of_llms_for_code_translation.md)**

:   提出 F2STrans，通过功能学习（正确性）和风格学习（可读性）两阶段渐进式微调 LLM，使 Qwen-1.5B 在 20 种代码翻译场景中平均超越 prompt 增强的 Qwen-32B 和 GPT-4。

**[Mind the Gap: A Practical Attack on GGUF Quantization](mind_the_gap_a_practical_attack_on_gguf_quantization.md)**

:   首次提出针对 GGUF 量化格式的攻击：利用量化误差作为"自由度"训练恶意量化模型，全精度下正常但量化后注入后门，在不安全代码生成（Δ=88.7%）、定向内容注入（Δ=85.0%）和良性拒绝（Δ=30.1%）上有效。

**[Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation](reasoning_through_execution_unifying_process_and_outcome_rewards_for_code_genera.md)**

:   提出 ORPS（Outcome-Refining Process Supervision），通过将代码执行反馈与 LLM 自我批评结合，在树状搜索框架中统一过程奖励与结果奖励，无需训练 PRM 即可在代码生成中实现 26.9% 的正确率提升和 42.2% 的效率提升。

**[Robust Learning of Diverse Code Edits (NextCoder)](robust_learning_of_diverse_code_edits.md)**

:   提出合成代码编辑数据生成流水线 + 鲁棒自适应算法 SeleKT（Selective Knowledge Transfer），通过在微调过程中周期性地对任务向量做 top-k 稀疏投影，使模型在获得强代码编辑能力的同时保留原始代码生成与通用推理能力，得到的 NextCoder 系列模型在五个代码编辑基准上超越同规模甚至更大模型。

**[SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity](sparselora_accelerating_llm_fine-tuning_with_contextual_sparsity.md)**

:   提出 SparseLoRA，通过**上下文稀疏性 (contextual sparsity)** 动态选择权重子集进行前向/梯度计算，首次将推理时的稀疏加速思路迁移到 LLM 微调阶段，实现最高 2.2× FLOPs 降低和 1.6× 实测加速，同时保持精度。

**[Towards Practical Defect-Focused Automated Code Review](towards_practical_defect-focused_automated_code_review.md)**

:   提出面向真实生产环境的端到端自动代码审查框架，通过AST代码切片提取上下文、多角色LLM协作审查、三层冗余评论过滤和内联行号定位四大模块，在近4亿日活公司的工业级C++代码库历史故障数据上实现KBI(关键缺陷包含率)2倍于标准LLM、10倍于先前基线的显著提升。

**[Training Software Engineering Agents and Verifiers with SWE-Gym](training_software_engineering_agents_and_verifiers_with_swe-gym.md)**

:   本文提出 SWE-Gym——首个用于训练软件工程 Agent 的环境，包含来自 11 个开源 Python 仓库的 2438 个真实任务实例，通过在 SWE-Gym 上进行拒绝采样微调训练 SWE Agent 和 Verifier，在 SWE-Bench Verified/Lite 上最终达到 32.0%/26.0% 的解决率，创造了开源权重 SWE Agent 的新 SOTA。
