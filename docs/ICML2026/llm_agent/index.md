---
title: >-
  ICML2026 LLM Agent方向16篇论文解读
description: >-
  16篇ICML2026的 LLM Agent 方向论文解读，涵盖 Agent、LLM、对齐/RLHF、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICML2026"
  - "LLM Agent"
  - "论文解读"
  - "论文笔记"
  - "Agent"
  - "LLM"
  - "对齐/RLHF"
  - "对抗鲁棒"
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🦾 LLM Agent

**🧪 ICML2026** · **16** 篇论文解读

📌 **同领域跨会议浏览：** [💬 ACL2026 (69)](../../ACL2026/llm_agent/index.md) · [📷 CVPR2026 (15)](../../CVPR2026/llm_agent/index.md) · [🔬 ICLR2026 (39)](../../ICLR2026/llm_agent/index.md) · [🤖 AAAI2026 (43)](../../AAAI2026/llm_agent/index.md) · [🧠 NeurIPS2025 (47)](../../NeurIPS2025/llm_agent/index.md) · [📹 ICCV2025 (4)](../../ICCV2025/llm_agent/index.md)

🔥 **高频主题：** Agent ×5 · LLM ×3

**[A Minimal Agent for Automated Theorem Proving](a_minimal_agent_for_automated_theorem_proving.md)**

:   本文提出 AxProverBase——一个极简的 Lean 4 定理证明智能体，仅靠"编译器反馈 + 自管理笔记本 + 轻量工具搜索"三个组件，在不微调的前沿 LLM（Claude Opus）上达到甚至超越 Hilbert/Seed-Prover 等专用系统，成本却低出 100 倍。

**[Adaptive Querying with AI Persona Priors](adaptive_querying_with_ai_persona_priors.md)**

:   作者把"LLM 在 persona 条件下产生的回答分布"打包成一个有限混合的贝叶斯先验，让用户在仅被问几道题的情况下，通过对 persona 后验做闭式更新来高效预测其他回答，性能上压过经典 CAT/IRT 基线。

**[Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)**

:   通过 Monte-Carlo rollout 量化"哪些回合的 thought / observation 可以省"，再用冷启动 SFT + 双采样 omit-aware GRPO 训出能自适应跳过冗余思考和观测的 8B agent，五个基准上 token 用量大降而准确率与七大前沿模型持平。

**[AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction](agentxray_white-boxing_agentic_systems_via_workflow_reconstruction.md)**

:   作者把"对黑盒 agent 系统反推一个等价白盒 workflow"作为新任务 AWR，用 MCTS 在 agent 原语序列空间中搜索，再配上一种基于评分动态着色的 Red-Black 剪枝来平衡深度与宽度，在五个真实领域上实现可解释的白盒重建。

**[BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)**

:   BioAgent Bench 给"用 LLM agent 跑生物信息学 pipeline"这件事造了一个端到端的评测套件——10 个真实 bioinformatics 任务 × 10 个 frontier/open-weight 模型 × 3 个 agent harness，配合 LLM 判官评分和 corrupted/decoy/prompt-bloat 三类扰动测试，发现前沿模型能完成 90%+ pipeline 但鲁棒性堪忧。

**[DiscoverLLM: From Executing Intents to Discovering Them](discoverllm_from_executing_intents_to_discovering_them.md)**

:   DiscoverLLM 把 "用户没想清楚自己要什么" 形式化为意图层级树的渐进发现过程，用可奖励的层级化用户模拟器训练模型在不清晰时主动发散探索、在清晰时收敛执行，在创意写作 / 技术写作 / SVG 三任务上比 CollabLLM 等 baseline 满意度 +10%、对话长度 -40%。

**[EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)**

:   EvolveR 给 LLM agent 套一个「在线交互 → 离线自蒸馏成原则库 → GRPO 策略进化」的闭环生命周期：agent 不再丢弃过去轨迹，而是把自己的成功失败抽象成可检索的「策略原则」，再用 RL 学会**如何用自己的原则**去解新问题，在 7 个多跳 QA benchmark 上明显跑赢 Search-R1 等 RL agent baseline。

**[ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation](excytin-bench_evaluating_llm_agents_on_cyber_threat_investigation.md)**

:   本文构建了首个评测 LLM Agent 端到端做"网络威胁调查"的 benchmark ExCyTIn-Bench：从真实 Azure 租户的 57 张安全日志表里，用 alert-entity 二部图自动生成 7542 道带证据链的 SQL 问答题，并提供 MySQL 环境让 Agent 通过查询日志、多跳追踪证据来回答，目前最强模型 Claude-Opus-4.5 也只能拿 0.606 的 reward。

**[Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration](group_cognition_learning_making_everything_better_through_governed_two-stage_age.md)**

:   针对集中式多模态融合带来的"模态主导"和"虚假模态耦合"两个痼疾，GCL 把多模态学习重写为**两阶段四 agent 的协议化协作**：第一阶段由 Routing/Auditing agent 用边际预测增益逐样本决定哪些跨模态交流被允许，第二阶段由 Public-Factor/Aggregation agent 把共享语义与私有特化解耦后再聚合，在 MOSI/MOSEI/MIntRec 上拿到 SOTA。

**[Internalizing Agency from Reflective Experience](internalizing_agency_from_reflective_experience.md)**

:   本文提出 LEAFE 框架，让 LLM agent 通过反思失败轨迹生成「失败→回滚→修正→成功」的经验数据，再用 SFT 蒸馏出 feedback-grounded 的恢复能力，在 CodeContests、WebShop、ALFWorld 等长程任务上把 Pass@128 拉高最多 14%，远胜 GRPO 等 outcome-driven RL。

**[Position: Agentic AI Orchestration Should Be Bayes-Consistent](position_agentic_ai_orchestration_should_be_bayes-consistent.md)**

:   这篇 position paper 主张：不要再尝试让 LLM 本身 "Bayesian"（那条路在工程上和理论上都跳不过去），而是把贝叶斯结构搬到 agentic AI 的**编排控制层**——让控制器维护一个低维任务级隐变量的信念，按 Bayes 规则在 agent/工具返回的"消息观测"上更新，并用期望效用或 value-of-information 做路由、停止、升级和预算分配。

**[Position: Assistive Agents Need Accessibility Alignment](position_assistive_agents_need_accessibility_alignment.md)**

:   这是一篇 position paper，作者通过对 417 篇文献中 778 个盲人辅助任务实例做系统综述，论证 "accessibility alignment" 应当被视为与 helpful/harmless/honest 并列的 Agent 一级对齐目标，并提出覆盖目标-交互-风险-生命周期四维度的设计 pipeline。

**[PragLocker: Protecting Agent Intellectual Property in Untrusted Deployments via Non-Portable Prompts](praglocker_protecting_agent_intellectual_property_in_untrusted_deployments_via_n.md)**

:   PragLocker 用 "代码符号初始化 + 黑盒目标模型反馈下的噪声注入" 两阶段策略，把 agent system prompt 编码成一段只能在 target LLM 上 work、迁移到其它任意 LLM 都会失效的 obfuscated text，从而在 prompt 被部署侧窃取时让攻击者无法在自己的 LLM 上复用。

**[ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)**

:   ReSeek 给 RL-trained 搜索 agent 增加一个 JUDGE 动作 + 用 BGE-reranker 计算"理想判断"作为过程奖励,使 agent 能在每次检索后软性"屏蔽"无效信息并重新查询;同时提出 FictionalHot 这一基于虚构实体的抗污染评测,Qwen2.5-7B 上平均 EM 达到 0.377,比 ZeroSearch 高 +3.1。

**[Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](video2gui_synthesizing_large-scale_interaction_trajectories_for_generalized_gui_.md)**

:   Video2GUI 用「元数据粗筛 → 视频质量精筛 → Gemini-3-Pro 提任务/动作 → 高分辨率三帧精确空间 grounding」四段流水线把 5 亿条 YouTube 视频元数据炼成 WildGUI（12.7M 轨迹、124.5M 截图、1500+ 应用），并把 Qwen2.5-VL/Mimo-VL 在多个 GUI grounding 与 agent benchmark 上提升 5–20%。

**[When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)**

:   CAIA 用 17 个前沿大模型在 178 个时间锚定的加密货币真实任务上构建首个"对抗性高风险"agent 基准，发现：无工具时所有模型只有 12–28% 准确率（接近随机猜测），有工具时最强 GPT-5 也只到 67.4% vs. 人类入门分析师 80%；更致命的是模型 55.5% 的工具调用偏向"不可靠的网页搜索"而绕过权威链上数据，导致 Pass@k 指标系统性掩盖了"靠试错碰运气"的危险行为。
