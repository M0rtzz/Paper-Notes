---
title: >-
  [论文解读] Near-Optimal Online Deployment and Routing for Streaming LLMs
description: >-
  [ICLR 2026][LLM/NLP][LLM路由] 首次形式化 LLM 流式在线部署+路由联合问题：新模型持续出现、旧模型可能过时，在并发部署上限 $M_{\max}$ 和成本预算约束下，提出 StageRoute 分层算法，证明 $\tilde{\mathcal{O}}(T^{2/3})$ 遗憾界并给出匹配下界，达到近最优。
tags:
  - ICLR 2026
  - LLM/NLP
  - LLM路由
  - 在线部署
  - streaming bandits
  - 并发上限
  - 预算约束
---

# Near-Optimal Online Deployment and Routing for Streaming LLMs

**会议**: ICLR 2026  
**arXiv**: [2506.17254](https://arxiv.org/abs/2506.17254)  
**代码**: 无  
**领域**: LLM NLP / 系统优化  
**关键词**: LLM路由, 在线部署, streaming bandits, 并发上限, 预算约束

## 一句话总结
首次形式化 LLM 流式在线部署+路由联合问题：新模型持续出现、旧模型可能过时，在并发部署上限 $M_{\max}$ 和成本预算约束下，提出 StageRoute 分层算法，证明 $\tilde{\mathcal{O}}(T^{2/3})$ 遗憾界并给出匹配下界，达到近最优。

## 研究背景与动机

**领域现状**：LLM 路由（按查询选模型）已有大量工作（RouteLLM、Hybrid-LLM、Zooter 等），但都假设模型集合固定不变。然而实际中新模型持续发布，旧模型逐渐过时——例如 Azure OpenAI 每个资源限 32 个标准部署+5 个微调部署，GPT-4.1 有 1000 RPM 和 1M TPM 的速率上限。

**两个时间尺度的决策耦合**：实际 LLM 服务面临两个根本不同时间尺度的决策：
   - **宏观（stage-wise）**：决定哪些模型保持部署（受并发上限 $M_{\max}$ 约束，整个 stage 不可更改）。这是一个**高风险决策**——激活不确定的新模型可能需要驱逐已知可靠模型一整个 stage
   - **微观（per-query）**：每个查询路由到哪个已部署模型（受预算+吞吐量约束）
   - 部署决策**决定了**路由的整个动作空间——这是此前路由工作忽略的根本前置问题

**现有方法的缺口**：静态池路由（RouteLLM）无动态部署；预算感知路由（TensorOpera）无并发上限；流式 bandits（UniRoute/CSCR）无 stage-level 承诺；BwK 建模消耗性预算但无主动集替换。没有现有框架同时处理**流式到达+动态部署+并发上限+预算+吞吐量**这五个维度

**核心idea**：将 LLM 部署和路由建模为耦合两个时间尺度的在线决策问题，提出 StageRoute——宏观用乐观估计选部署集，微观用 LP 做实时路由——达到近最优 $\tilde{\mathcal{O}}(T^{2/3})$ 遗憾界

## 方法详解

### 整体框架
StageRoute 是一个两层分层算法，镜像问题的层级结构：
- **战略层（部署阶段）**：在每个更新点 $\tau_k$，从包含新到达模型的候选池中选择 $\leq M_{\max}$ 个模型部署，整个 stage 不变
- **战术层（路由阶段）**：每个查询到达时，在已部署模型中求解 LP 得到路由分布，采样后分派

### 关键设计
1. **部署优化（DeployOPT）**：将其建模为混合整数规划（MIP），用二值变量 $z_m$ 选模型、连续变量 $d_m$ 表示权重。用 UCB 估计性能上界 $\mu_m^U$、LCB 估计成本下界 $c_m^L$，在预算和并发约束下最大化乐观性能。解 $d^*$ 仅用于选模型集合，**不作为路由概率**。
2. **路由优化（RouteOPT）**：每个查询到达时，在已部署集上求解线性规划，最大化 UCB 估计的期望奖励，受预算和吞吐量约束。使用当前统计量实时更新 UCB/LCB。
3. **置信界设计**：$\mu_m^U = \text{proj}_{[0,1]}(\bar{\mu}_m + 2f_{rad})$，$c_m^L = \text{proj}_{[c_1,c_2]}(\bar{c}_m - 2f_{rad})$，其中 $f_{rad}(v,n) = \sqrt{\gamma v/n} + \gamma/n$。

### 理论保证
- **遗憾上界（Theorem 1）**：$\text{Regret} \leq \mathcal{O}(\sqrt{M_{\max}KT\log(NT/\delta)} + NT/(M_{\max}K))$
    - 第一项：已部署集内路由的统计学习代价
    - 第二项：结构性模型发现瓶颈（新模型发现的难度）
- 取 $K = \Theta(T^{1/3})$，$M_{\max} = \Omega(N^{2/3})$ 时得 $\tilde{\mathcal{O}}(N^{1/3}T^{2/3})$
- **匹配下界（Theorem 2）**：$\Omega(T^{2/3})$，证明近最优性

## 实验关键数据

### 与现有 LLM 路由框架的对比

| 方法 | 流式模型 | 动态部署($M_{\max}$限制) | 预算感知 | 吞吐量限制 |
|------|---------|----------------------|---------|----------|
| RouteLLM | ✗ | ✗ | ✓ | ✗ |
| UniRoute | ✓ | ✗ | ✓ | ✗ |
| CSCR | ✓ | ✗ | ✓ | ✗ |
| **StageRoute** | **✓** | **✓** | **✓** | **✓** |

### 仿真实验（RouterBench 真实成本/分数数据）

| 实验设置 | StageRoute vs Oracle 差距 | 说明 |
|---------|------------------------|------|
| 紧预算 $(b=0.3)$ | <5% sub-optimality | 在严格预算下仍紧跟 oracle |
| 宽预算 $(b=0.7)$ | <2% sub-optimality | 预算宽裕时几乎最优 |
| $M_{\max}$ 变化 | 性能随 $M_{\max}$ 单调提升 | 验证理论中并发上限的作用 |
| $K$ 变化 | $K=T^{1/3}$ 附近最优 | 验证理论最优 stage 数 |
| 多任务/多语言查询 | 一致有效 | 跨多样负载鲁棒 |

### 关键发现
- 两项遗憾分解清晰揭示了"路由学习"vs"模型发现"的根本权衡
- StageRoute 在紧预算下尤其优于静态部署策略——动态替换过时模型带来显著收益
- 吞吐量约束自然节流高负载模型，缓解延迟尖峰

## 亮点与洞察
- **问题形式化创新**：首次将 LLM 部署+路由建模为耦合两个时间尺度的在线决策问题，涵盖流式到达+并发上限+预算+吞吐量约束的完整实际场景
- **遗憾界的双项分解**深具洞察——统计学习代价 vs 结构发现瓶颈的权衡关系为系统设计提供指导
- **匹配下界**证实了 $\tilde{\mathcal{O}}(T^{2/3})$ 的不可改进性——这是问题本身的固有难度，非算法缺陷
- DeployOPT 的解仅用于选模型而非做路由概率，将部署与路由执行解耦——允许 stage 内的快速查询级适应

## 局限与展望
- 假设每个模型的性能分布是固定的（均值 $\mu_m$ 不变），未建模模型退化或查询分布漂移
- 路由是无上下文的（非 contextual bandit），未利用查询特征信息——加入 contextual estimator 可能进一步提升
- 吞吐量约束通过概率 $p_t(m) \leq \alpha_m$ 建模，假设瞬时负载共享，实际中可能需考虑排队延迟
- 仿真实验基于 RouterBench 的离线数据，未在真实部署环境中验证

## 相关工作与启发
- **vs RouteLLM/Hybrid-LLM**：固定模型池+无并发限制→StageRoute 处理完整动态场景
- **vs BwK (Bandits with Knapsacks)**：BwK 建模消耗性预算但无 stage-level 承诺和主动集替换
- **vs CMAB (Combinatorial MAB)**：CMAB 从固定基集选超臂而无流式到达
- **vs Streaming Bandits**：允许流式到达但不耦合 stage 部署与查询路由
- **启发**：两层决策分离（战略部署+战术路由）的架构思想可推广到其他资源受限的在线服务系统

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次完整形式化+近最优算法+匹配下界
- 实验充分度: ⭐⭐⭐ 仿真为主，缺乏真实部署验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，问题动机清晰
- 价值: ⭐⭐⭐⭐ 对 LLM 服务系统设计有理论指导意义

<!-- RELATED:START -->

## 相关论文

- [BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](../../ICML2025/llm_nlp/best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)
- [ICL-Router: In-Context Learned Model Representations for LLM Routing](../../AAAI2026/llm_nlp/icl-router_in-context_learned_model_representations_for_llm_routing.md)
- [LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding](../../ACL2025/llm_nlp/llm_as_effective_streaming_processor_bridging_streaming-batch_mismatches_with_gr.md)
- [WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality](webdevjudge_mllm_web_development.md)
- [d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)

<!-- RELATED:END -->
