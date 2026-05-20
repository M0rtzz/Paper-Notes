---
title: >-
  [论文解读] MARS²: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation
description: >-
  [ACL 2026][代码智能][多智能体强化学习] 本文提出 MARS²，将多智能体协作直接嵌入树结构搜索中进行强化学习训练，通过路径级分组优势和树一致性奖励塑形解决复杂搜索轨迹的信用分配问题，在代码生成基准上一致性地超越单智能体方法。
tags:
  - "ACL 2026"
  - "代码智能"
  - "多智能体强化学习"
  - "树搜索"
  - "代码生成"
  - "GRPO"
  - "信用分配"
---

# MARS²: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation

**会议**: ACL 2026  
**arXiv**: [2604.14564](https://arxiv.org/abs/2604.14564)  
**代码**: [GitHub](https://github.com/TsinghuaC3I/MARTI)  
**领域**: 代码智能 / 强化学习  
**关键词**: 多智能体强化学习, 树搜索, 代码生成, GRPO, 信用分配

## 一句话总结

本文提出 MARS²，将多智能体协作直接嵌入树结构搜索中进行强化学习训练，通过路径级分组优势和树一致性奖励塑形解决复杂搜索轨迹的信用分配问题，在代码生成基准上一致性地超越单智能体方法。

## 研究背景与动机

**领域现状**：以 GRPO 为代表的强化学习范式在代码生成等推理密集型任务上取得显著进展。然而，单智能体 RL 的探索行为受限于模型自身先验分布，导致轨迹多样性不足和过早收敛到局部最优。搜索增强 RL（如 TreeRL）通过引入结构化搜索缓解了这一问题，但搜索树仍由单一策略驱动。

**现有痛点**：存在两个核心挑战：(1) 单策略先验下的探索收益递减——随着训练推进，搜索行为集中在少数高概率分支上，难以持续扩展探索前沿；(2) 多智能体协作缺乏结构化搜索集成——现有多智能体 LLM 推理框架（辩论、投票等）将智能体交互视为轻量级协调而非结构化探索过程，缺乏分支、回溯和预算分配机制。

**核心矛盾**：单智能体树搜索的探索空间受限于共享先验，而多智能体协作又脱离了结构化搜索动态，两者各自的优势未能统一。

**本文目标**：构建统一框架，让多个独立优化的策略在共享的树结构搜索环境中协作，同时解决多智能体树搜索中的信用分配和训练稳定性问题。

**切入角度**：将搜索树建模为可学习的多智能体交互环境，而非静态采样过程。

**核心 idea**：多异构智能体在共享搜索拓扑中协作生成和精炼候选解，通过 Thompson 采样选择智能体-节点对，路径级分组优势 + 树一致性奖励塑形实现有效信用分配。

## 方法详解

### 整体框架

多智能体共享树搜索环境 → Thompson 采样选择智能体和待扩展节点 → 区分生成节点（水平扩展新解）和精炼节点（垂直优化现有解） → 树一致性奖励塑形（结合父节点和兄弟节点信号） → 路径级分组优势计算 → 各智能体独立 GRPO 优化。

### 关键设计

1. **多智能体树搜索交互环境**:

    - 功能：将搜索树转化为多智能体可学习的协作环境
    - 核心思路：在每个扩展步，使用 Thompson 采样先选择最有前景的智能体，再选择关联节点进行扩展。区分两类可扩展节点：生成节点（提出新候选解，水平扩展）和精炼节点（优化已有解，垂直精炼），动态平衡开发与探索
    - 设计动机：不同策略先验产生多样化探索信号，突破单策略搜索的隐式边界。Thompson 采样的随机性保证了智能体选择的探索-利用平衡

2. **树一致性奖励塑形 (Tree-Consistent Reward Shaping)**:

    - 功能：在树结构约束下实现稳定的奖励分配
    - 核心思路：对每个非根节点 $v$，定义混合基线 $b(v) = (1-\lambda)r_{p(v)} + \lambda \cdot \mu_{C(p(v))\setminus v}$，结合父节点奖励和兄弟节点均值奖励。结构一致性增益 $\Delta(v) = r_v - b(v)$，塑形后奖励 $\hat{r}_v = r_v + \gamma \cdot \Delta(v)$
    - 设计动机：纯全局树级优势无法捕获层级依赖关系。子节点不仅应获得高全局奖励，还应相对父节点有改进并超越同级兄弟。$\lambda$ 平衡纵向改进和横向竞争

3. **路径级分组优势与独立优化**:

    - 功能：将 GRPO 扩展到树搜索场景，支持多智能体独立更新
    - 核心思路：搜索树中所有节点源自同一输入问题，自然构成语义分组。树级相对优势在塑形后奖励上计算 $\hat{A}_{v,j} = (\hat{r}_{v,j} - \text{mean}) / \text{std}$。每个智能体仅用自己生成的节点进行参数更新，采用缓冲异步更新避免同步等待
    - 设计动机：多智能体并发更新共享搜索树引入非平稳性，独立优化 + 树一致性奖励塑形共同缓解这一问题

### 损失函数 / 训练策略

采用 GRPO 风格策略优化，扩展为多智能体版本（公式 7），结合 GSPO（几何均值替代 token 级聚合）、过长惩罚（DAPO 风格长度奖励塑形）和 TIS（token 级重要性采样，对齐 vLLM 推理与 FSDP 训练分布）等稳定化技术。所有稳定化技术同等应用于基线以保证公平比较。训练数据使用 DeepCoder 的 7,992 个代码生成 prompt。

## 实验关键数据

### 主实验

| 模型/系统 | 方法 | Pass@1 | Pass@1(MCTS) | Pass@N |
|--------|------|------|----------|------|
| Qwen3-8B | Base→GRPO→RS2→MARS² | 50.3→52.5→55.4→**58.3** | 54.3→57.1→60.6→**60.8** | 68.6→73.1→71.4→**72.3** |
| AReaL-14B | Base→GRPO→RS2→MARS² | 58.4→58.9→62.3→**64.6** | 62.9→60.7→68.0→**68.1** | 74.3→75.4→81.1→**80.2** |
| Q&A-8B System | Base→GRPO→RS2→MARS² | —  | 57.2→56.0→57.2→**61.7** | 69.7→72.0→72.6→**75.4** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 引入弱智能体 DeepCoder-14B | 个体 Pass@1 下降，系统 Pass@1(MCTS) 仍提升 | 系统级优势不依赖智能体能力均衡 |
| 无奖励塑形 | 训练曲线不稳定，收敛延迟 | 奖励塑形提供更密集的中间监督 |
| 多样性指标对比 | MARS² 在 AEC、DA@K 上最优 | 多智能体提升解空间覆盖 |
| MATH 数据集泛化 | Pass@1(MCTS) 0.756→0.804 | 树级信用分配不限于代码任务 |

### 关键发现
- MARS² 在所有模型组合和设置上一致性改进：Qwen3-8B 的 Pass@1 从 50.3% 提升到 58.3%（+8.0）
- AReaL-14B 在 MARS² 训练后达到 64.6%，超越 O4-Mini (Low) 的 63.7%
- 单智能体 RS² 已优于 GRPO，但多智能体 MARS² 进一步突破性能上限
- 多样性分析显示 MARS² 的优势主要来自更丰富的候选解池而非重复利用高奖励轨迹

## 亮点与洞察
- 将搜索树建模为多智能体学习环境的视角新颖，统一了结构化搜索和多智能体协作两个方向
- 树一致性奖励塑形优雅地解决了树结构中的信用分配问题，同时鼓励纵向改进和横向竞争
- 弱智能体实验表明系统级收益不依赖能力均衡，增强了实际部署的灵活性
- 从代码到数学推理的泛化验证了框架的通用性

## 局限与展望
- 多智能体树搜索的顺序交互降低了训练效率（rollout 并行度下降），需要开发更高效的搜索机制
- 实验主要集中在 8B 和 14B 模型，更大规模模型上的效果待验证
- 训练预算固定条件下的公平比较虽严格，但未充分探索增加预算时的收益递减曲线
- 智能体间的协作机制（Thompson 采样）较为简单，可能存在更优的选择策略

## 相关工作与启发
- **vs TreeRL**: TreeRL 将 MCTS 引入 RL 训练但使用单策略，MARS² 通过多策略突破单策略先验限制
- **vs MAPoRL**: MAPoRL 通过多轮对话实现多智能体协作但脱离结构化搜索，MARS² 将协作直接嵌入搜索树
- **vs Vanilla GRPO**: GRPO 的 i.i.d. 采样限制探索空间，MARS² 通过树结构和多智能体打破这一限制

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 多智能体树搜索 RL 框架是全新的统一范式
- 实验充分度: ⭐⭐⭐⭐ 多模型、多规模、多指标、消融全面，数学泛化验证加分
- 写作质量: ⭐⭐⭐⭐ 技术细节完整，公式清晰，实验组织合理
- 价值: ⭐⭐⭐⭐ 为搜索增强 RL 提供了有效的多智能体扩展路径，对推理能力提升有实际推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](recode_reinforcing_code_generation_with_reasoning-process_rewards.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[ACL 2026\] ChipSeek: Optimizing Verilog Generation via EDA-Integrated Reinforcement Learning](chipseek_optimizing_verilog_generation_via_eda-integrated_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
