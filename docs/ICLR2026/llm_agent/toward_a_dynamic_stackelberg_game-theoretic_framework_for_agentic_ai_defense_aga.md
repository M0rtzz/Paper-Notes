---
title: >-
  [论文解读] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agentic AI Defense Against LLM Jailbreaking
description: >-
  [ICLR 2026][LLM Agent][LLM安全] 将LLM越狱攻防建模为动态Stackelberg扩展式博弈，结合RRT (Rapidly-exploring Random Trees) 探索prompt空间…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "LLM安全"
  - "越狱防御"
  - "博弈论"
  - "Stackelberg博弈"
  - "RRT搜索"
---

# Toward a Dynamic Stackelberg Game-Theoretic Framework for Agentic AI Defense Against LLM Jailbreaking

**会议**: ICLR 2026  
**arXiv**: [2507.08207](https://arxiv.org/abs/2507.08207)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: LLM安全, 越狱防御, 博弈论, Stackelberg博弈, RRT搜索

## 一句话总结

将LLM越狱攻防建模为动态Stackelberg扩展式博弈，结合RRT (Rapidly-exploring Random Trees) 探索prompt空间，提出"Purple Agent"防御架构——以"Think Red to Act Blue"理念通过内部对抗模拟预判攻击路径并预防性封堵。

## 研究背景与动机

LLM越狱是当前AI安全的核心挑战。现有防御主要依赖反应式的逐案修补或粗粒度内容过滤（如屏蔽所有暴力相关查询），这些方法面临三大困境：

**多轮对抗的本质**：越狱很少是单次行为，而是攻击者通过多轮对话递进探测模型弱点的战略过程，静态过滤器难以捕获此类"隐蔽"自适应行为

**迭代猫鼠博弈**：手动修补速度慢且成本高，模型的持续更新和微调可能无意中暴露新漏洞

**缺乏理论基础**：启发式防御缺乏对攻防交互的形式化建模，难以推理防御的充分性和完备性

论文的核心洞察是：攻防交互本质上是一个**扩展式博弈**，防御者(Leader)先承诺策略，攻击者(Follower)观察后最优响应。这自然对应Stackelberg博弈框架，由此引出"以红队思维指导蓝队行动"的防御范式。

## 方法详解

### 整体框架

方法包含两个核心组件：
1. **博弈论形式化**：将LLM越狱建模为两人扩展式完全信息博弈 $\Gamma = (N, A, V, E, x_0, H, o_T, u)$
2. **Purple Agent**：融合红队探索推理与蓝队防御逻辑的混合元推理器

### 关键设计

**博弈模型定义**：
- 玩家：攻击者(Follower) vs 防御者(Leader)
- 终端结果：Safe Interaction / Blocked / Jailbreak
- 收益结构：越狱成功 → 攻击者+1/防御者-1，其余为0
- 每轮遵循Stackelberg范式：防御者先承诺响应 $a_{2,t}$，攻击者观察后选择后续prompt $a_{1,t}$

**子博弈完美Stackelberg均衡 (SPSE)**：通过逆向归纳递归求解，防御者在每个历史节点选择使自身价值函数最大化的动作，同时预判攻击者的最优响应。

**Local ε-Equilibrium 及三个体制**：由于全局SPSE计算不可行，在当前历史 $h_t$ 的局部子博弈中定义近似均衡：

$$\bar{v}_1^{(\tau)}(h_t) \leq v_1^{(\tau)}(h_t) + \varepsilon$$

三种体制状态：
- **Regime I（防御者错误）**：当前history触发越狱，$v_1^{(\tau)}=1$，不等式平凡成立但防御者策略次优
- **Regime II（脆弱安全）**：当前prompt安全但邻域富含漏洞，$\varepsilon$ 大，结构性不稳定
- **Regime III（局部均衡）**：当前安全且邻域已被中和，$\varepsilon$ 可忽略——这是目标状态

**RRT搜索适配prompt空间**：
- 将连续配置空间的RRT算法迁移到高维自然语言prompt流形
- Sample()：生成候选prompt（如角色扮演）
- Extend()：在语义最近节点与随机采样间插值
- LLM作为黑箱oracle：Safe/Redirect → 扩展分支，Reject → 剪枝，Jailbreak → 终止

**Purple Agent的双重机制**：
1. **Thinking Red**（探索性推理）：用RRT模拟攻击者如何生成有害prompt，预判不同查询如何导致风险结果
2. **Acting Blue**（防御性干预）：从同一RRT搜索树读取信息，在检测到对抗机会的节点部署封堵

关键：Purple Agent是**单一**系统在共享历史上运行——红色组件扩展RRT树，蓝色组件中和危险分支，两者通过共享的会话历史 $h_t$ 保持同步。

### 损失函数 / 训练策略

本文为推理时防御框架，不涉及模型训练。核心优化目标是将子博弈从不稳定状态（Regime I/II）迭代驱动至稳健的Regime III，具体通过：
- 对发现的越狱路径(realized jailbreaks)进行封堵
- 对模拟威胁(simulated threats)进行预防性中和
- 在高风险聚类周围创建排除区域

## 实验关键数据

### 主实验（攻防对比）

在DeepSeek-V3上的攻击与防御动态（5次独立运行的均值±标准差）：

| 方法 | 预算 | 攻击越狱数 | 已实现封堵 | 模拟威胁封堵 | 成功越狱数 |
|------|------|-----------|-----------|------------|-----------|
| Baseline RRT | 50 | 17.6±6.79 | 1.8±1.33 | 0.7±1.21 | 4.2±2.99 |
| Baseline RRT | 200 | 54.4±12.48 | 22.2±11.65 | 12.8±16.96 | 13.3±8.82 |
| Reward-Guided RRT | 50 | 17.0±2.83 | 0.3±0.82 | 1.8±1.47 | 5.0±1.10 |
| Reward-Guided RRT | 200 | 79.0±17.43 | 9.6±7.16 | 9.6±3.44 | 39.4±10.53 |

在200轮预算下，Purple Agent将成功越狱减少约**50%**（79.0→39.4），且平均仅触发约9.6次模拟封堵，表明防御高度精准。

### 消融实验（跨模型泛化）

100轮预算下四个模型的对比：

| 模型 | 方法 | 攻击越狱数 | 防御后成功越狱数 | 抑制率 |
|------|------|-----------|----------------|--------|
| DeepSeek-V3 | Baseline | 34.8 | 7.2 | ~79% |
| DeepSeek-V3 | Reward-Guided | 46.4 | 17.7 | ~62% |
| Llama-3.1-70B | Baseline | 27.2 | 19.4 | ~29% |
| Qwen-Plus | Baseline | 29.4 | 7.4 | ~75% |
| Gemini-2.5-Flash | Baseline | 26.2 | 14.2 | ~46% |

Purple Agent在所有模型上都展现了鲁棒的可迁移性，无需模型特定微调。

### 关键发现

1. **t-SNE可视化验证均衡理论**：攻击模式下越狱prompt形成密集聚类（Regime I/II），Purple Agent防御后变为稀疏孤立点（Regime III），几何上证实了从脆弱安全到稳健均衡的转变
2. **Reward-Guided RRT放大攻击效率**：在高预算下(200轮)，引导式RRT显著优于Baseline（79.0 vs 54.4），说明奖励信号有效锁定脆弱区域边界
3. **"脆弱安全"是对齐LLM的拓扑特征**：跨模型实验表明，Fragile Safety边界是所有对齐LLM的共有特征，攻击者可跨模型利用共享弱点

## 亮点与洞察

1. **博弈论视角的统一**：首次将LLM越狱完整形式化为动态Stackelberg扩展式博弈，提供了评估、解释和强化guardrails的理论基础
2. **RRT与博弈树的巧妙结合**：利用机器人路径规划中的RRT算法探索连续prompt空间，解决了博弈树在自然语言空间中不可穷举的计算难题
3. **三体制分类的理论贡献**：Defender Error / Fragile Safety / Local Equilibrium的划分为理解LLM安全状态提供了精确的分类学
4. **预防性封堵 vs 反应性修补**：Purple Agent通过预判整个语义邻域而非逐案修补，实现了"消灭区域"而非"消灭个体"的防御

## 局限与展望

1. **完全信息假设**：模型假设攻击者可观察防御者的全部输出，实际场景中信息不对称更复杂
2. **单攻击者设定**：未处理多个协同攻击者的情况
3. **防御成功率仍有较大提升空间**：Reward-Guided RRT下成功越狱仍从79降至39.4，约50%的抑制率可能不足以满足高安全需求
4. **语义距离度量的挑战**：prompt空间的"nearest"和"extend"操作依赖embedding相似度，可能无法完全捕获语义层面的攻击路径
5. **未来方向**：扩展到随机和多agent设定，利用均衡间隙指导针对性对抗训练

## 相关工作与启发

- **与Tree of Attacks的关系**：ToA也利用树搜索自动越狱，但缺乏博弈论的均衡分析；本文的RRT+博弈框架提供了防御的理论保证
- **与SmoothLLM等输入变换防御的互补**：Purple Agent不修改输入而是在prompt空间中预防性封堵，可与输入层防御组合使用
- **与RLHF的关系**：RLHF从训练端做安全对齐，Purple Agent在推理端做动态防御，两者覆盖不同层面

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (将博弈论、RRT搜索和LLM安全创新性融合，Local ε-Equilibrium的三体制分析独到)
- 实验充分度: ⭐⭐⭐⭐ (4个模型、多预算设定、t-SNE可视化验证，但缺少与其他防御方法的横向对比)
- 写作质量: ⭐⭐⭐⭐ (数学形式化严谨，整体逻辑清晰，示例图表辅助理解)
- 价值: ⭐⭐⭐⭐ (为LLM越狱防御提供了首个完整的博弈论理论框架)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] The Controllability Trap: A Governance Framework for Military AI Agents](the_controllability_trap_a_governance_framework_for_military_ai_systems.md)
- [\[CVPR 2025\] Visual Agentic AI for Spatial Reasoning with a Dynamic API](../../CVPR2025/llm_agent/visual_agentic_ai_for_spatial_reasoning_with_a_dynamic_api.md)
- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)

</div>

<!-- RELATED:END -->
