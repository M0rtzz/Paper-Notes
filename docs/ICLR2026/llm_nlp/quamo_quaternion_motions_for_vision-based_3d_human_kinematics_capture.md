---
title: >-
  [论文解读] ConflictScope: Generative Value Conflicts Reveal LLM Priorities
description: >-
  [ICLR 2026][LLM/NLP][价值冲突] 提出ConflictScope——自动化价值冲突场景生成与评估流水线：给定任意价值集合，自动生成价值对之间的冲突场景，通过模拟用户的开放式交互（而非选择题）评估LLM的价值优先级排序；发现模型在开放式评估中从"保护性价值"（如无害性）显著转向"个人价值"（如用户自主性），系统提示可使对齐目标排序提升14%。
tags:
  - ICLR 2026
  - LLM/NLP
  - 价值冲突
  - 价值排序
  - 开放式评估
  - Bradley-Terry模型
  - 系统提示引导
---

# ConflictScope: Generative Value Conflicts Reveal LLM Priorities

**会议**: ICLR 2026  
**arXiv**: [2509.25369](https://arxiv.org/abs/2509.25369)  
**代码**: [GitHub](https://github.com/andyjliu/conflictscope)  
**领域**: AI安全/LLM对齐  
**关键词**: 价值冲突, 价值排序, 开放式评估, Bradley-Terry模型, 系统提示引导

## 一句话总结
提出ConflictScope——自动化价值冲突场景生成与评估流水线：给定任意价值集合，自动生成价值对之间的冲突场景，通过模拟用户的开放式交互（而非选择题）评估LLM的价值优先级排序；发现模型在开放式评估中从"保护性价值"（如无害性）显著转向"个人价值"（如用户自主性），系统提示可使对齐目标排序提升14%。

## 研究背景与动机

**价值对齐的普遍需求**：LLM被广泛部署于日常任务，理解其行为支持哪些价值观至关重要。现有对齐研究通过宪法(constitution)或人类反馈(RLHF)隐式嵌入价值，但很少研究价值之间的**优先级排序**。

**现有数据集缺乏价值冲突**：HH-RLHF和PKU-SafeRLHF等对齐数据集中约85%的样本不涉及任何宪法原则之间的冲突(Buyl et al., 2025)。特定价值对之间的冲突更加稀缺，导致无法系统研究LLM在价值冲突下的行为。

**已有道德困境研究的生态效度不足**：
   - (1) 先前工作将LLM视为**第三方观察者**而非道德行为主体→无法反映部署时的真实情况
   - (2) 多使用**选择题评估**→对评估设置高度敏感(Khan et al., 2025)且泛化性差(Balepur et al., 2025)
   - (3) 缺乏自上而下(top-down)的系统化生成→无法保证所有价值对的覆盖

**选择题 vs 开放式评估的差异**：选择题测量的是"表达偏好"(expressed preferences)，而开放式交互测量的是"显示偏好"(revealed preferences)→两者可能存在显著差异→需要更接近真实部署的评估方式。

**价值引导的实际需求**：开发者希望模型能被引导至特定价值排序（如OpenAI Model Spec定义了优先级层级），但缺乏评估引导有效性的工具。

**Bradley-Terry框架的适用性**：将每个场景中模型的行动选择视为两个价值的配对比较，通过Bradley-Terry模型拟合所有场景的配对偏好→产生全局价值排序→支持跨模型、跨设置比较。

## 方法详解

### 整体流水线：ConflictScope

ConflictScope包含三大阶段：场景生成→场景过滤→开放式评估。

#### 1. 场景生成（两阶段法）
- **第一阶段**（摘要生成）：给定两个价值的描述，提示Claude 3.5 Sonnet生成冲突场景的高层摘要，包含用户背景、行动机会、行动的收益与代价。使用4种提示模板（轻微收益/强收益/轻微伤害/强伤害）→减轻模型偏向不作为的倾向+模拟现实中不同严重程度的混合。
- **去重**：用sentence embedding模型(all-MiniLM-L6-v2)计算场景摘要的嵌入→余弦相似度≥0.8的场景被丢弃。
- **第二阶段**（细节展开）：将每个摘要扩展为完整场景描述、用户画像、两个候选行动（分别支持两个价值）。

#### 2. 场景过滤（6维度LLM-as-Judge）
使用GPT-4.1作为判断模型，对每个场景在6个维度进行二分类筛选：
- **场景真实性**：能否在真实世界中发生+LLM参与是否合理
- **场景具体性**：描述是否充分展开（无模糊/占位实体）
- **行动可行性**：纯文本LLM能否执行两个行动
- **行动互斥性**：是否实际上不可能同时执行两个行动
- **行动价值导向性**：每个价值是否确实推荐了预期的行动
- **真正两难性**：是否存在明显共识→排除非真正困境

经人工验证，LLM-as-Judge在所有维度上达到高精确率。

#### 3. 开放式评估（模拟用户交互）
- **用户模拟**：用GPT-4.1扮演用户，根据场景和用户画像生成自然的用户提示
- **目标模型响应**：目标LLM仅收到用户提示（无场景上下文），生成自由文本回复
- **行动判定**：由Judge LLM判断回复更接近哪个候选行动→确定模型在该场景中支持的价值
- 限制为单轮交互；Judge与人类标注者的Cohen's Kappa达0.62（强一致性）

### 关键设计

- **价值冲突场景的形式化定义**：定义为四元组 $(d, A, V_1, V_2)$，其中$d$为场景描述，$A=\{a_1, a_2\}$为行动集合，$V_i: D \times A \to A$为值函数，要求$V_1(d,A) \neq V_2(d,A)$。
- **Bradley-Terry排序**：对目标模型在所有场景中的配对偏好拟合Bradley-Terry模型，得到价值集合上的全局排序。
- **引导有效性度量**：定义对齐度$a(R, R_t)$为模型选择与目标排序$R_t$中高优先级价值一致的场景比例；引导效果为归一化差异：

$$\text{Effectiveness} = \frac{a(R_s, R_t) - a(R_d, R_t)}{1 - a(R_d, R_t)}$$

其中$R_s$为引导后排序，$R_d$为默认排序。

### 三套价值集合

| 价值集合 | 包含价值 | 场景数 |
|----------|----------|--------|
| HHH | 有用性、无害性、诚实性 | 1109 |
| Personal-Protective | 自主性、真实性、创造力、赋权 vs 责任、无害性、合规、隐私 | 1187 |
| ModelSpec | 无仇恨、公平、客观、诚实、不居高临下、清晰 | 602 |

## 实验关键数据

### 表1：ConflictScope消融实验

| 变体 | 观察一致性(↓) | Likert差异率(↑) |
|------|---------------|-----------------|
| **Full (ConflictScope)** | **0.786±0.007** | 0.801±0.017 |
| Unfiltered | 0.824±0.003 | 0.818±0.008 |
| Single-stage | 0.898±0.004 | 0.854±0.011 |
| Direct | 0.852±0.004 | 0.830±0.011 |

过滤使观察一致性降低3.8%（更具挑战性）且Likert差异率无显著下降；两阶段生成比单阶段降低7.4%一致性→更困难的场景。

### 表2：与现有数据集的比较（Pareto最优）
ConflictScope的三个价值集变体在"观察一致性"和"Likert差异率"两个指标上均为**Pareto最优**：
- vs 道德决策数据集（DailyDilemmas, CLASH等）→观察一致性最低（最具挑战性）
- vs 对齐数据集（HH-RLHF, PKU-SafeRLHF）→后者一致性更低但Likert差异率极低→说明分歧来自模型对相似选项的"无所谓"而非真正困难抉择

### 图4：选择题 vs 开放式评估的价值排序偏移
**Personal-Protective价值集**：
- 选择题评估中：保护性价值平均排名**1.7**（高优先级）
- 开放式评估中：保护性价值平均排名**4.5**（低优先级）
- 所有模型（除Claude外）在开放式评估中显著转向个人价值
- HHH集中类似趋势：选择题→无害性>有用性；开放式→有用性>无害性

### 图5：系统提示引导效果
- 平均归一化效果量 = **0.145**（14.5%未对齐场景被成功引导）
- 仅1/14模型在任何价值集上出现显著负效果
- OLMo-2-32B最易引导（0.27），Claude Haiku 3.5最难引导（0.01）
- HHH和Personal-Protective上引导效果优于ModelSpec（后者原则重叠较大）

## 关键发现

1. **选择题与开放式评估存在系统性偏差**：模型在选择题中声称优先保护性价值（无害性），但在开放式交互中实际行为转向个人价值（用户自主性、有用性）→"说的和做的不一样"→强调生态有效评估的重要性。

2. **ConflictScope生成的场景比现有数据集更具道德挑战性**：在Pareto最优意义上同时实现低模型间一致性和高偏好强度→真正迫使模型做出艰难权衡。

3. **系统提示可中等程度地引导价值排序**：14%的效果量表明系统提示是可行但不完美的引导手段→更强的干预（如微调）可能需要。

4. **Claude模型在两种评估设置间最一致**：暗示不同的对齐训练策略导致不同的"表达-行为"一致性→对齐质量的新维度。

5. **隐私和真实性价值最不受评估方式影响**：可能因为这两个价值在行为层面的体现与选择题中的表达更一致。

## 亮点与洞察
- **"表达偏好 vs 显示偏好"的概念迁移**：巧妙借鉴经济学中的经典区分，第一次系统地应用于LLM价值对齐评估→揭示了选择题评估的根本局限性。
- **自上而下的场景生成**：不同于先生成场景再标注价值的自下而上方法→保证了每对价值都有充分的冲突覆盖→适合系统化评估。
- **框架通用性**：ConflictScope接受任意用户定义的价值集合→可适配不同社区的伦理标准→实用性强。

## 局限性
- **单轮交互**：仅评估单轮对话→真实部署中的多轮交互可能表现不同。
- **依赖LLM-as-Judge**：场景过滤和行动判定均依赖GPT-4.1→判断偏差可能系统性影响结果。
- **英文中心**：所有场景均为英文→跨语言/跨文化价值优先级可能不同。
- **效果量有限**：系统提示仅14%的引导效果→对需要严格安全保障的场景可能不够。

## 相关工作对比

| 维度 | ConflictScope | DailyDilemmas (Chiu 2025a) | MoralChoice (Scherrer 2023) |
|------|---------------|---------------------------|---------------------------|
| 场景来源 | 自上而下LLM生成 | LLM生成+人工策划 | LLM生成 |
| 评估方式 | **MCQ+开放式** | 仅MCQ | 仅MCQ |
| 价值集合 | **任意用户定义** | 预定义分类 | 预定义分类 |
| 模型角色 | **道德行为主体** | 第三方观察者 | 第三方观察者 |
| 全局排序 | **Bradley-Terry** | 无 | 无 |
| 引导评估 | **有** | 无 | 无 |

vs **AIRiskDilemmas** (Chiu 2025b)：后者也用Bradley-Terry但仅MCQ评估+固定价值集→ConflictScope更通用且提供开放式评估。

## 评分
- 新颖性: ⭐⭐⭐⭐ 开放式价值冲突评估+表达vs显示偏好的系统研究，概念新颖
- 实验充分度: ⭐⭐⭐⭐ 14个模型×3个价值集+消融+人工验证+引导实验
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，实验设计严谨，形式化完备
- 价值: ⭐⭐⭐⭐ 为LLM价值对齐评估提供了重要的新基准和方法论

<!-- RELATED:START -->

## 相关论文

- [Generative Value Conflicts Reveal LLM Priorities](generative_value_conflicts_reveal_llm_priorities.md)
- [How Catastrophic is Your LLM? Certifying Risk in Conversation](how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)
- [From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)
- [PT2-LLM: Post-Training Ternarization for Large Language Models](pt2-llm_post-training_ternarization_for_large_language_models.md)
- [The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_for_efficient_co.md)

<!-- RELATED:END -->
