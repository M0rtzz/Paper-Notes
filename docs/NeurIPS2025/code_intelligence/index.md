---
title: >-
  NeurIPS2025 代码智能方向22篇论文解读
description: >-
  22篇NeurIPS2025的代码智能方向论文解读，涵盖代码智能、LLM、推理、Agent、布局/合成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "代码智能"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "推理"
  - "Agent"
  - "布局/合成"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💻 代码智能

**🧠 NeurIPS2025** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [🧪 ICML2026 (2)](../../ICML2026/code_intelligence/index.md) · [💬 ACL2026 (48)](../../ACL2026/code_intelligence/index.md) · [📷 CVPR2026 (2)](../../CVPR2026/code_intelligence/index.md) · [🔬 ICLR2026 (20)](../../ICLR2026/code_intelligence/index.md) · [🤖 AAAI2026 (9)](../../AAAI2026/code_intelligence/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/code_intelligence/index.md)

🔥 **高频主题：** 代码智能 ×5 · LLM ×3 · 推理 ×3 · Agent ×2 · 布局/合成 ×2

**[A Self-Improving Coding Agent](a_selfimproving_coding_agent.md)**

:   提出SICA（Self-Improving Coding Agent），一个能自主编辑自身代码库来提升性能的编程Agent——消除了meta-agent和target-agent的区分，通过迭代式自我改进在SWE-Bench Verified子集上从17%提升到53%。

**[A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions](a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)**

:   将 LLM 迭代交互中的多目标优化建模为 SDE（漂移-扩散过程），通过干扰矩阵量化目标间的耦合模式，通过特征值谱分析策略收敛行为，在代码生成（安全性、效率、功能性三目标）上验证了不同策略的收敛率（0.33-1.29）和可预测性（$R^2$ 达 0.74）。

**[AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy](astrovisbench_a_code_benchmark_for_scientific_computing_and_visualization_in_ast.md)**

:   AstroVisBench 构建了首个评估 LLM 天文科学计算和可视化能力的代码基准——从 110 个 Jupyter Notebook 提取 864 个任务（处理+可视化），设计双重评估管线（执行式变量检查 + VLM-as-Judge 可视化评分，与专家 Spearman ρ=0.822），评测 8 个 SOTA 模型后发现 Gemini 2.5 Pro 最佳但无错误率仅 15.7%，FileNotFoundError 占 43% 错误。

**[VeriMaAS: Automated Multi-Agent Workflows for RTL Design](automated_multi-agent_workflows_for_rtl_design.md)**

:   VeriMaAS 提出自动组合多 Agent 工作流的框架用于 RTL 代码生成，核心创新是将 HDL 工具的形式化验证反馈（Yosys 综合 + OpenSTA 时序分析）直接整合到工作流编排中，在 VeriThoughts 上 pass@1 提升 2-12%，且仅需数百样本做控制器调优，比全量微调训练数据少一个量级。

**[Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)**

:   提出 CURE 框架，让同一个 LLM 同时扮演代码生成器和单元测试生成器两个角色，通过生成代码与生成测试的交叉执行构建成对奖励矩阵，用基于理论推导的奖励信号进行强化学习，在完全不需要 ground-truth 代码标注的情况下实现代码生成能力和单元测试生成能力的共同进化，在五个编程基准上大幅超过同规模的专用 Coder 模型。

**[CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks](core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)**

:   提出 CoRe，一个包含 12,553 个人工验证任务实例的高质量 benchmark，通过数据依赖、控制依赖和信息流三类静态分析基础任务，直接评估 LLM 的代码语义推理能力，揭示模型在 trace 生成和源枚举等需要多步推理的任务上仍严重不足。

**[Embedding Alignment in Code Generation for Audio](embedding_alignment_in_code_generation_for_audio.md)**

:   提出双 MLP + InfoNCE 对比学习框架，将代码嵌入（distilroberta-base）和音频嵌入（wav2vec2）对齐到共享空间，使 LLM 代码生成流程无需编译执行即可从代码推断音乐相似性，CKA 从 0.090 提升至 0.590。

**[FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts](flylora_boosting_task_decoupling_and_parameter_efficiency_via_implicit_rank-wise.md)**

:   FlyLoRA 受飞蝇嗅觉回路启发，将 LoRA 的下投影矩阵 $A$ 替换为冻结的稀疏随机投影，通过 top-$k$ 激活值选择实现隐式 rank-wise MoE 路由，在消除路由参数的同时减少任务内干扰，并利用随机投影的近正交性天然支持多任务模型合并。

**[FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis](fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)**

:   提出 FractalBench，一个通过分形图像程序合成诊断 MLLM 视觉-数学推理能力的 benchmark：12 种经典分形、610 张测试图、4 个 MLLM，揭示 76% 的代码能执行但仅 4% 视觉正确，暴露了模型在递归抽象能力上的根本缺陷。

**[Learning From Design Procedure To Generate CAD Programs for Data Augmentation](learning_from_design_procedure_to_generate_cad_programs_for_data_augmentation.md)**

:   提出一种受工业设计流程启发的CAD程序数据增强范式，通过向LLM提供参考曲面程序和设计流程描述来引导生成包含B-Spline有机形状的CAD程序，显著缩小了公开CAD数据集与工业级设计在几何复杂度上的差距。

**[Learning to Solve Complex Problems via Dataset Decomposition](learning_to_solve_complex_problems_via_dataset_decomposition.md)**

:   提出Decomp方法，利用教师模型将复杂数学题按推理步骤递归分解为更简单的子问题，构建概念依赖图量化难度，再按从易到难的课程顺序训练学生模型——Qwen2.5-1.5B在MATH-500上达51.6%（超MuggleMath用147K数据的50.4%），Qwen3-4B在AIME2025仅用385样本达16.7%（超Qwen2.5-72B的15%）。

**[MaintainCoder: Maintainable Code Generation Under Dynamic Requirements](maintaincoder_maintainable_code_generation_under_dynamic_requirements.md)**

:   首次系统定义并解决 LLM 代码生成的**可维护性**问题，同时贡献基准和方法：MaintainBench 通过 4 种需求变化模式 + 动态指标评测代码在需求演化下的可维护性；MaintainCoder 将 Waterfall 模型、设计模式与 6 个专业化 Agent 结合，动态可维护性指标提升 60%+，且初始代码正确性也一并提高。

**[MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)**

:   提出 MLR-Bench，一个包含 201 个开放式 ML 研究任务的综合基准，配套 MLR-Judge（LLM 评审框架）和 MLR-Agent（模块化研究代理），发现当前最先进的编码代理在约 80% 的情况下会生成伪造或未验证的实验结果，揭示了 AI 自动化科学研究的核心瓶颈。

**[Once Upon an Input: Reasoning via Per-Instance Program Synthesis](once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)**

:   提出 PIPS（Per-Instance Program Synthesis），通过实例级别的程序合成与结构化反馈迭代改进，结合置信度度量动态选择直接推理或程序合成，在30个基准上将调和平均准确率提升8.6%。

**[Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization](preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o.md)**

:   系统研究了校准数据的组成特性（序列长度/样本量/来源/格式）和领域对应关系对LLM压缩后能力保持的影响，发现激活空间中的代表性和多样性是数据质量的本质决定因素，并据此提出三阶段校准数据策展框架COLA。

**[Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward](principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)**

:   系统研究如何利用用户编辑数据微调 LLM，将偏好、监督标签和代价三种反馈类型统一起来，并提出一种简单的集成方法，在不同用户分布下实现鲁棒适应。

**[Program Synthesis via Test-Time Transduction](program_synthesis_via_test-time_transduction.md)**

:   提出 SYNTRA 框架，将程序合成重新定义为转导式学习——在测试时利用可见的 test inputs 和 LLM 的判断来迭代消除不一致的候选程序假设，通过 greedy maximin 算法最小化 LLM 查询次数，在 4 个 benchmark 上准确率提升最高达 196%。

**[QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)**

:   提出信号级感知学习方法 QiMeng-SALV，通过从部分错误的 Verilog 模块中提取信号级功能正确的代码片段作为 DPO 训练的奖励信号，将优化粒度从模块级提升到信号级，在 VerilogEval 和 RTLLM 上达到 SOTA。

**[Searching Latent Program Spaces](searching_latent_program_spaces.md)**

:   提出 Latent Program Network（LPN），通过编码器将输入-输出示例映射为潜在程序表示，在测试时通过梯度搜索潜在空间来适应新任务，在 ARC-AGI 基准上显著优于 in-context learning 和 test-time training 方法。

**[SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)**

:   构建全自动化流水线从 GitHub 持续挖掘真实软件工程交互任务，生成 21,000+ 可执行 Python 任务的 SWE-rebench 数据集和去污染 benchmark，揭示部分模型在 SWE-bench Verified 上的性能存在污染膨胀问题（如 DeepSeek-V3 在 SWE-bench 上 39.7% vs SWE-rebench 上 21.3%）。

**[Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models](table2latex-rl_high-fidelity_latex_code_generation_from_table_images_via_reinfor.md)**

:   提出VSGRPO——基于GRPO的双奖励强化学习策略，联合优化结构级奖励（TEDS-Structure）和视觉保真度奖励（CW-SSIM渲染图比较），使微调后的MLLM（仅3B参数）在表格图像到LaTeX代码生成任务上超越GPT-4o和72B+规模模型，尤其在复杂表格上提升显著。

**[Text-to-Code Generation for Modular Building Layouts in Building Information Modeling](text-to-code_generation_for_modular_building_layouts_in_building_information_mod.md)**

:   提出 Text2MBL 框架，将自然语言描述转化为可执行的 BIM 代码（而非坐标序列），通过面向对象的代码架构和 LLM 微调实现模块化建筑布局的自动生成，在几何一致性上比坐标驱动方法提升 10%+ IoU。
