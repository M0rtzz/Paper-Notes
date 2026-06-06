---
title: >-
  [论文解读] Weasel: 通过重要性-多样性数据选择实现 Web Agent 的域外泛化
description: >-
  [ICML 2026][LLM Agent][数据选择] 通过结合目标相关性和多样性的轨迹步骤选择方法，Weasel 在减少训练数据到原始 20% 的同时实现 9.7-12.5 倍训练加速，并显著提升 Web Agent 在未见域上的泛化能力。
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "数据选择"
  - "Web Agent"
  - "域外泛化"
  - "轨迹精选"
  - "训练效率"
---

# Weasel: 通过重要性-多样性数据选择实现 Web Agent 的域外泛化

**会议**: ICML 2026  
**arXiv**: [2605.20291](https://arxiv.org/abs/2605.20291)  
**代码**: https://github.com/fatemehpesaran310/weasel  
**领域**: LLM Agent / Web Agent  
**关键词**: 数据选择, Web Agent, 域外泛化, 轨迹精选, 训练效率

## 一句话总结
通过结合目标相关性和多样性的轨迹步骤选择方法，Weasel 在减少训练数据到原始 20% 的同时实现 9.7-12.5 倍训练加速，并显著提升 Web Agent 在未见域上的泛化能力。

## 研究背景与动机

**领域现状**：LLM 驱动的 Web Agent 已通过大规模指令数据和强基础模型取得进展，但多数研究在基准内评估，无法测试真正泛化能力。

**现有痛点**：（1）Agent 在训练分布外的网站或交互模式上性能大幅下降；（2）离线 Web 交互数据通常冗长嘈杂，专家轨迹包含大量冗余步骤；（3）AgentTrek 中单条轨迹可达 45 个状态-动作对，每个 Web 状态的可访问树（AXTree）最长可达 180K tokens。

**核心矛盾**：如何在有限预算内选择既相关又多样的数据子集——这是 NP 困难问题。

**本文目标**：设计轨迹选择方法同时优化（1）改善域外泛化；（2）降低计算成本。

**切入角度**：将轨迹精选建模为约束优化问题，结合目标条件的重要性和成对多样性，用贪心算法高效求解。

**核心 idea**：平衡单项重要性得分与成对多样性距离，从长轨迹中选出固定预算的信息密集子集，实现少数据、高效率、强泛化。

## 方法详解

### 整体框架
三组件——（1）**轨迹选择**：根据目标相关性和多样性给步骤打分，贪心选子集；（2）**目标中心剪枝**：保留 AXTree 中目标动作周围局部上下文；（3）**推理风格合成**：对推理原生模型用自生成风格一致的推理过程替换异质推理痕迹。

### 关键设计

1. **重要性-多样性轨迹子集选择**:

    - 功能：从长且冗余的轨迹中选高价值步骤子集。
    - 核心思路：单项重要性 $\Phi(t) = \text{BERTScore}(g, s_t)$；成对多样性 $D(i,j) = \max(\delta(s_i, s_j), \delta(y_i, y_j))$。目标 $\max_J \sum_{j\in J} \Phi(j) + \lambda \sum_{i<j, i,j\in J} D(i,j)$ 约束 $|J| = T_0 \ll T$。贪心算法先选最优对，再迭代加入最大边际增益元素 $i_m = \arg\max_{k \notin J_{m-1}} \Phi(k) + \lambda \sum_{i \in J_{m-1}} D(k,i)$。
    - 设计动机：单独最大化重要性易选冗余相似状态；加入多样性能覆盖异质网页/状态/交互模式。问题为 NP 困难 max-sum 多元化，贪心在 99.7% 轨迹上落入前 1% 最优解，近似比 0.9999±0.0005。

2. **目标中心 AXTree 剪枝**:

    - 功能：保留 Web 状态中与目标动作相关局部上下文，去除外围冗余。
    - 核心思路：给定线性化 AXTree 节点序列 $V_t$ 和目标位置 $k_t^*$，保留大小为 $2w+1$ 的连续窗口；对非节点动作（如 goto）用固定长度前缀。
    - 设计动机：AgentTrek 单条状态可达 180K tokens；保留目标附近内容显著减少计算（2× 速度提升），实验验证偏移距离增加时成功率线性下降。

3. **推理风格合成**:

    - 功能：为推理原生模型生成与其预训练风格一致的推理过程。
    - 核心思路：对每个选中步骤 $t \in J^*$，用目标模型根据 $g, h_t, \tilde{s}_t, a_t$ 生成 $\hat{r}_t$；训练目标 $\max_\theta \sum_{\tau \in \mathcal{D}} \sum_{t \in J^*(\tau)} \log \pi_\theta(a_t, \hat{r}_t | g, h_t, \tilde{s}_t)$。
    - 设计动机：Qwen3 等推理原生模型在预训练时学会特定推理方式；异质模型推理痕迹训练导致风格不匹配，反而破坏泛化。表 4 显示仅加风格合成就能从 17.0% 提升到 21.2% SR。

## 实验关键数据

### 主实验

| 数据集 | 模型 | 训练配置 | WebArena-Lite | WebArena | MiniWob | 训练加速 |
|--------|------|----------|---------------|----------|---------|----------|
| AgentTrek | Qwen2.5-7B | Full (52K) | 10.9 | 8.7 | 44.6 | 1.0× |
| AgentTrek | Qwen2.5-7B | Weasel (10K) | 14.5 | 9.5 | 48.0 | 11.3× |
| AgentTrek | Gemma3-4B | Full (52K) | 9.1 | 4.3 | 28.6 | 1.0× |
| AgentTrek | Gemma3-4B | Weasel (10K) | 11.5 | 5.5 | 30.6 | 12.5× |
| AgentTrek | Qwen3-8B | Full (52K) | 17.7 | 18.2 | 59.4 | 1.0× |
| AgentTrek | Qwen3-8B | Weasel (10K) | 21.2 | 19.2 | 61.9 | 10.7× |

### 消融

| 方法 | 数据选择 | 推理合成 | WebArena-Lite |
|------|----------|----------|---------------|
| 基础 Qwen3-8B | ✗ | ✗ | 16.4 |
| SFT (Random) | ✗ | ✗ | 16.5 |
| SFT + 推理合成 | ✗ | ✓ | 18.2 |
| Weasel w/o 推理合成 | ✓ | ✗ | 17.0 |
| **Weasel (完整)** | **✓** | **✓** | **21.2** |

### 关键发现
- 三个 LLM 上一致超越全数据 SFT，9.7-12.5× 训练加速，仅用 20% 数据达到或超过完整微调性能。
- 多样性必要性：仅状态多样性 9.7%，仅动作多样性 13.9%，结合 14.5%。
- 重要性-多样性平衡：仅重要性 10.9%，仅多样性 7.9%，结合 14.5%。
- 跨域迁移：移植到 AITW 安卓 GUI 设置，3.1K 子集超过随机采样（5.8% → 6.6%）。

## 亮点与洞察
- **精妙问题建模**：抽象为 max-sum 多元化，既理论有据（贪心近似保证）又实践高效。
- **多维多样性设计**：同时考虑状态空间和动作空间多样性。
- **推理风格匹配的关键洞察**：发现推理原生模型对训练数据推理风格敏感性，从 17% 直接跳到 21%。
- **端到端完整方案**：数据选择 + 状态剪枝 + 推理适配形成完整流程。

## 局限与展望
- 贪心算法理论保证不适用于伪距离（非度量），仅是启发式有效。
- 多模态 GUI 实验改进较小（5.8% → 6.6%），视觉主导场景需进一步优化。
- BERTScore 评分本身可能有偏。
- 改进：探索学习 importance/diversity 权重；结合上下文学习；推理模型侧研究更灵活风格适配。

## 相关工作与启发
- **vs WebRL / WebAgent-R1**：在线交互 RL vs 本文离线数据精选，无需 environment rollout 成本。
- **vs 通用数据选择**：通用方法关注模型独立的样本代表性；Weasel 针对 web agent 特殊性设计 goal-conditioned 重要性。
- **vs 其他 state pruning**：Lee 等通过学习模块或检索引入额外参数；Weasel 是轻量级无参设计。
- **vs 指令微调优化**：偏好优化面向对齐；Weasel 同时优化泛化和效率。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  首次将 max-sum 多元化引入 web agent 数据选择。
- 实验充分度: ⭐⭐⭐⭐⭐  3 LLM + 多 benchmark + 跨域验证 + 完整消融。
- 写作质量: ⭐⭐⭐⭐⭐  结构清晰，formalization 严谨。
- 价值: ⭐⭐⭐⭐⭐  对 web agent 实际部署有直接帮助。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Agent JIT Compilation for Latency-Optimizing Web Agent Planning and Scheduling](agent_jit_compilation_for_latency-optimizing_web_agent_planning_and_scheduling.md)
- [\[CVPR 2026\] Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](../../CVPR2026/llm_agent/ego2web_a_web_agent_benchmark_grounded_in_egocentric_videos.md)
- [\[AAAI 2026\] Prune4Web: DOM Tree Pruning Programming for Web Agent](../../AAAI2026/llm_agent/prune4web_dom_tree_pruning_programming_for_web_agent.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[NeurIPS 2025\] Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](../../NeurIPS2025/llm_agent/web-shepherd_advancing_prms_for_reinforcing_web_agents.md)

</div>

<!-- RELATED:END -->
