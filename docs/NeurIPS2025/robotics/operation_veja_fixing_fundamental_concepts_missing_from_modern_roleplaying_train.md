---
title: >-
  [论文解读] Operation Veja: Fixing Fundamental Concepts Missing from Modern Roleplaying Training Paradigms
description: >-
  [NeurIPS 2025][机器人][角色扮演] 本文系统批判了现有角色扮演模型训练的四大范式（RAG、事实值设定、文学数据、合成数据）为何都无法产生有深度的角色，提出VEJA框架（Values-Experiences-Judgments-Abilities）作为角色定义和数据策化的结构化基础，在LLM评判A/B测试中VEJA指导的人工策化数据以43:28:29（胜:负:平）显著优于Gemini Pro 2.5生成的合成基线。
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "角色扮演"
  - "VEJA框架"
  - "价值冲突推理"
  - "数据策化"
  - "角色一致性"
---

# Operation Veja: Fixing Fundamental Concepts Missing from Modern Roleplaying Training Paradigms

**会议**: NeurIPS 2025  
**arXiv**: [2601.06039](https://arxiv.org/abs/2601.06039)  
**代码**: [GitHub](https://github.com/HyouinKyoumaIRL/Operation-Veja)  
**领域**: 机器人  
**关键词**: 角色扮演, VEJA框架, 价值冲突推理, 数据策化, 角色一致性

## 一句话总结

本文系统批判了现有角色扮演模型训练的四大范式（RAG、事实值设定、文学数据、合成数据）为何都无法产生有深度的角色，提出VEJA框架（Values-Experiences-Judgments-Abilities）作为角色定义和数据策化的结构化基础，在LLM评判A/B测试中VEJA指导的人工策化数据以43:28:29（胜:负:平）显著优于Gemini Pro 2.5生成的合成基线。

## 研究背景与动机

**领域现状**：角色扮演（RP）模型越来越复杂，但始终无法捕捉可信、引人入胜的角色本质。用户无论在游戏、AI伴侣还是交互式叙事中，都期望角色有深度、一致性和内在冲突。

**现有痛点**：作者以《命运石之门》中的牧濑红莉栖为例，指出没有现有技术能复现她"智识好奇心与社交防卫性的冲突"这一核心特质。模型的回复仅仅是反应性的，从不具有真正的探究性或内在冲突驱动性。核心问题在于当前模型缺乏**价值冲突推理**（value-conflicted reasoning）。

**核心矛盾**：人类的互动不是检索"正确"回复的过程，而是不断在冲突价值观之间协商——礼貌vs效率、好奇心vs实用性、脆弱vs专业。当前所有训练范式都忽略了这个内在审慎过程。

**本文目标** 四个主流RP训练范式的系统性缺陷分别是什么？需要什么样的角色定义框架才能让数据具备足够的角色深度？

**切入角度**：从戏剧艺术（斯坦尼斯拉夫斯基体系）的角色构建原则出发，认为角色通过目标、动机和过往经历的交互来展现，将此转化为可操作的AI训练数据策化框架。

**核心 idea**：角色深度来自价值观、经历、判断和能力四个维度的因果链式交互，只有按VEJA结构策化数据才能突破合成数据的质量天花板。

## 方法详解

### 整体框架

分为两部分：(1)对现有四大范式的系统性批判，识别各自的根本性缺陷；(2)提出VEJA框架，定义角色深度的四个核心支柱并形成因果链：**经验(E) → 价值观(V) → 判断(J) → 能力表达(A)**。框架用于指导人工写手创作高质量训练数据。

### 关键设计

1. **四大范式的系统性批判**:

    - 功能：解释为什么现有方法都不能产生有深度的角色
    - 核心思路：**RAG的扩展性问题**——人类价值体系组合爆炸式增长，无法穷举为可检索的事实，情境修饰符使复杂度指数增长。**事实值设定的去语境化问题**——将价值观提炼为孤立公式（如"对陌生人开放"），模型会过度索引单一特质。**文学数据的隐含上下文诅咒**——角色内在推理过程在对话中仅被暗示而非显式提供，经验通过叙述而非对话传达。**合成数据的鸡和蛋问题**——用GPT-4生成有价值冲突的对话时，GPT-4自身就不具备平衡冲突价值观的能力
    - 设计动机：通过排除法论证新框架的必要性

2. **VEJA框架四支柱**:

    - 功能：为角色定义提供结构化、可操作的概念框架
    - 核心思路：**Values(价值观)**——驱动角色目标和欲望的高层动机（如"我想要更多时间因为有很多事想做"→野心/生产力）；**Experiences(经历)**——塑造价值观和判断的具体过往事件（如"因合同纠纷上过法庭，所以深度怀疑口头承诺"）；**Judgments(判断)**——角色对世界的具体观点和启发式规则，是价值观经由经历过滤后的输出（如"认为婚姻是过时的制度"）；**Abilities(能力)**——角色的技能、知识和能力范围，定义权威领域并影响词汇和兴趣
    - 设计动机：四支柱形成因果链（经验锻造价值观→价值观和经验产生判断→能力提供表达工具），创造连贯的内在逻辑

3. **数据策化实验设计**:

    - 功能：初步验证VEJA框架能否提升数据质量
    - 核心思路：构建以牧濑红莉栖为中心的两个小规模数据集。基线组用Gemini Pro 2.5基于10天时间线和20个用户生成合成对话；VEJA组给15名人工写手提供同一时间线+VEJA结构化角色Profile。100次随机配对A/B测试，Gemini 2.5 Flash盲审
    - 设计动机：明确承认存在人类vs机器的混淆变量，但论证核心观点是"当前模型无法合成此质量的数据，人工策化+正确框架是必要组件"

### 损失函数 / 训练策略

本文不涉及模型训练，而是关注数据策化框架。评估采用LLM-as-judge的A/B偏好测试，评判标准是"哪个对话更像角色本人"。

## 实验关键数据

### 主实验

LLM评判A/B偏好测试（N=100，Gemini 2.5 Flash盲审）：

| 结果 | 数量 | 占比 |
|------|------|------|
| VEJA人工数据胜出 | 43 | 43% |
| 合成基线胜出 | 28 | 28% |
| 平局 | 29 | 29% |

排除平局后VEJA胜率：43/(43+28) = **60.6%**

### 消融实验

论文未进行VEJA各组件的消融实验（作者承认这是因时间限制导致的局限）。

| 分析维度 | 说明 |
|---------|------|
| V/E/J/A各自贡献 | 未消融，无法判断哪个支柱最关键 |
| 人工 vs 框架效应 | 存在混淆：无法区分是VEJA框架还是人工写作本身带来的质量提升 |
| 评判模型偏差 | 仅用Gemini 2.5 Flash评判，未交叉验证 |

### 关键发现

- **VEJA数据在叙事连续性和角色一致性上显著优于合成数据**：评审频繁指出VEJA对话展示了"优越的叙事连续性、更细腻的反应、角色人设与对话之间更清晰的联系"
- **合成数据的典型缺陷**：被标记为"通用、脱离角色、过度依赖单一特质"。例如一个对话竟然让红莉栖说中文（她不说中文），另一个出现了完全不符合她性格的性暗示场景
- **VEJA赢的原因不仅是人工写作**：作者论证VEJA框架是使人工写手保持高一致性和质量的关键工具——没有框架指导，人工写手也会产生不一致的数据

## 亮点与洞察

- **系统性问题诊断比解决方案更有价值**：四大范式的批判分析直击痛点——RAG的组合爆炸、事实值的去语境化、文学数据的隐含上下文、合成数据的递归质量天花板——每个都精准且有说服力
- **"价值冲突推理"概念的提出**：将角色深度的核心定义为在冲突价值观间的协商过程，这是一个非常有洞察力的观察。它解释了为什么RP模型总是"浅薄可预测"——它们学的是"对应打出正确回复"而非"在内在冲突中做出选择"
- **因果链设计可迁移**：E→V→J→A的因果链不仅适用于虚拟角色，还可以应用于用户画像建模、对话系统人设设计、NPC行为树设计等场景

## 局限与展望

- **数据集规模极小**：仅围绕一个角色（牧濑红莉栖），15个人工写手+20个合成用户，统计效力有限
- **人类vs机器混淆**：无法区分VEJA框架的贡献和人工写作本身的贡献。理想实验应加入"无VEJA指导的人工写作"组和"VEJA指导的合成生成"组
- **评估仅用LLM-as-judge**：作者自己也承认用LLM来评估关于LLM能力缺陷的论文有讽刺性。需要大规模人类评估
- **可扩展性存疑**：人工策化成本高昂，论文虽提及半自动化方向但未具体探索
- **缺乏下游模型训练验证**：没有在VEJA数据上训练模型并评估其RP能力，仅停留在数据质量比较
- 改进方向：多角色验证、加入消融实验、探索VEJA引导的半自动生成pipeline、用训练后的模型做端到端评估

## 相关工作与启发

- **vs RoleLLM (Wang et al., 2024)**：RoleLLM的评估框架聚焦可测量的"人格保真度"，但其数据构建过程直接展示价值观而非嵌入多面场景，正是"事实值设定"的典型代表
- **vs COSER (Wang et al., 2025)**：COSER使用文学对话，虽然角色有深度但训练数据只能暗示推理过程而不能显式提供，是"文学数据诅咒"的例证
- **vs Persona Hub (Chan et al., 2024)**：大规模合成数据代表，但受限于生成器自身缺乏冲突推理能力的递归天花板

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统性批判视角和VEJA框架概念具有原创性和启发性
- 实验充分度: ⭐⭐ 单角色小规模试验、缺少消融、缺少下游训练验证
- 写作质量: ⭐⭐⭐⭐⭐ 论述逻辑严密，问题分析深刻，用牧濑红莉栖的例子贯穿全文极具说服力
- 价值: ⭐⭐⭐⭐ 问题诊断对RP研究社区有显著价值，但实验验证不足限制了影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data](himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)
- [\[NeurIPS 2025\] The Impact of Scaling Training Data on Adversarial Robustness](the_impact_of_scaling_training_data_on_adversarial_robustness.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)

</div>

<!-- RELATED:END -->
