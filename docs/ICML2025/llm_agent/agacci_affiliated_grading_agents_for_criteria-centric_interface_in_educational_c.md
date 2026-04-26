---
title: >-
  [论文解读] AGACCI: Affiliated Grading Agents for Criteria-Centric Interface in Educational Coding Contexts
description: >-
  [ICML 2025 (Workshop on Multi-Agent Systems)][LLM Agent][多Agent系统] AGACCI 提出一个由 9 个专门化 Agent 组成的多 Agent 评估框架，将教育编程作业的评估任务分解为 rubric 解析、代码执行验证、可视化评估、解释性推理评估等角色，通过协作实现比单模型 baseline 更准确、一致且可解释的 rubric 对齐反馈。
tags:
  - ICML 2025 (Workshop on Multi-Agent Systems)
  - LLM Agent
  - 多Agent系统
  - 教育评估
  - 代码评估
  - rubric对齐
  - 自动反馈
---

# AGACCI: Affiliated Grading Agents for Criteria-Centric Interface in Educational Coding Contexts

**会议**: ICML 2025 (Workshop on Multi-Agent Systems)  
**arXiv**: [2507.05321](https://arxiv.org/abs/2507.05321)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: 多Agent系统, 教育评估, 代码评估, rubric对齐, 自动反馈

## 一句话总结

AGACCI 提出一个由 9 个专门化 Agent 组成的多 Agent 评估框架，将教育编程作业的评估任务分解为 rubric 解析、代码执行验证、可视化评估、解释性推理评估等角色，通过协作实现比单模型 baseline 更准确、一致且可解释的 rubric 对齐反馈。

## 研究背景与动机

### 现有问题

在 AI 辅助教育评估领域，现有基于 LLM 的自动评估系统存在三个核心问题：

**反馈质量低**：单一 LLM 常常产生过于正面的评价（即使学生答案错误），或生成缺乏证据基础的幻觉推理（Jansen et al., 2024）。反馈往往是肤浅的表扬或模糊的建议，无法真正反映学生的表现或误解。

**Rubric 对齐不足**：现有方法常忽视教学 rubric（评估量规）中定义的细粒度标准，仅关注表面级的代码正确性或语法（Phung et al., 2023），导致反馈与教师的评估意图产生偏差。

**评估不一致**：LLM 对相似或相同的提交可能产生截然不同的评价，即使使用集成策略（Pathak et al., 2025）也难以解决单模型系统的结构性局限。

### 现有方案的不足

- **G-Eval**（Liu et al., 2023）：使用 LLM 作为评估器，但仍是单模型系统，缺乏结构化的角色分工
- **生成-评估-再生成流水线**（Guo et al., 2024; Seo et al., 2025）：虽然增加了迭代优化，但错误检测不完善，发现的问题不一定反映到最终反馈中
- **Agent-as-a-Judge**（Zhuge et al., 2024）：专门设置评估 Agent 可以提升与人类评分者的对齐度，但仍受限于单一评估角色的结构约束

### 动机

教育编程作业（尤其是 Jupyter Notebook 形式）涉及多维度的评估需求：代码执行正确性、可视化输出质量、解释性推理深度等。单一模型难以同时兼顾这些维度。作者认为，通过系统化的角色分配和结构化的评估流水线，可以在每个维度上实现更精准的判断，同时保持整体评估的一致性。

## 方法详解

### 整体框架

AGACCI 基于 **AutoGen** 框架构建，使用 **GPT-4o mini** 作为所有 Agent 的 backbone 模型。系统将评估过程分解为一个模块化的 Agent 流水线，共包含 **9 个专门化 Agent**。

整体工作流程分为三个阶段：

1. **输入解析**：Rubric Interpreter 和 Submission Analyzer 将任务分解为结构化的评估目标
2. **并行评估**：三条并行评估流——执行评估（Execution + Result Evaluator）、可视化评估（Visualization Evaluator）、推理评估（Interpretation Evaluator）
3. **聚合输出**：Meta Evaluator 检查跨流一致性 → Final Judge 综合裁决 → Summarizer 格式化输出

### 关键设计

#### 1. Rubric Interpreter（量规解析器）

- **功能**：将高层级的 rubric 描述重构为可操作的评估标准
- **设计要点**：不是将 rubric 视为静态的检查清单，而是将其转化为可执行的评估目标，识别隐含依赖关系、顺序约束和最低性能期望
- **输出格式**：结构化 JSON，包含 `final_objective`、`prerequisite_items`、`subgoals`、`evidence_types`

#### 2. Submission Analyzer（提交分析器）

- **功能**：整体审视学生提交内容，识别主要目标、逻辑结构以及与 rubric 标准的对齐情况
- **设计要点**：检测代码块的序列和目的、注释内容和输出，充当人类可读目标与机器级分析之间的桥梁
- **作用**：确保下游评估发生在正确的教学上下文中

#### 3. Execution Evaluator（执行评估器）

- **功能**：关注代码的功能有效性
- **检查内容**：代码是否无错运行、核心计算步骤是否存在、预期输出（如图表、打印指标）是否生成
- **定位**：在定性评估开始前确保技术性能的可靠性

#### 4. Result Evaluator（结果评估器）

- **功能**：判断执行结果是否满足 rubric 定义的量化性能标准
- **评估方式**：解析打印输出、日志或数值结果，产出二值判断（pass/fail）
- **特殊处理**：如果执行失败或无可测量结果，该 Agent 等待指令并放弃判断

#### 5. Visualization Evaluator（可视化评估器）

- **功能**：检查视觉输出（图表、图形）的清晰度和适当性
- **评估维度**：可视化方法是否匹配数据性质，视觉组件（坐标轴、标签、图例）是否支持可解释性

#### 6. Interpretation Evaluator（解释评估器）

- **功能**：评估学生超越观察进行推理的能力
- **关注点**：因果或推断性解释，从数据模式、异常或趋势中提取意义
- **惩罚**：过于描述性或论证不足的评论会被标记

#### 7. Meta Evaluator（元评估器）

- **功能**：作为内部一致性检查器，交叉验证各 Agent 的输出
- **操作**：标记矛盾或缺乏支持的评估，执行观察证据与声明的 rubric 满足之间的对齐检查
- **权限**：可以建议覆盖或调整置信度

#### 8. Final Judge（最终裁判）

- **功能**：将所有评估汇总为最终决定
- **输出**：解决跨 Agent 输出中的歧义，确定二值 rubric 满足分数（pass/fail），生成人类可读反馈

#### 9. Summarizer（总结器）

- **功能**：将系统的裁决浓缩为紧凑的面向学习者的摘要
- **输出格式**：结构化 JSON，包含关键发现、建议和 rubric 分数

### 架构设计选择

- **并行 + 层级控制**：三条评估流并行执行提高效率，Meta Evaluator 和 Final Judge 提供层级控制
- **Backbone 选择**：使用 GPT-4o mini 而非更大的模型，考虑到教育场景中有限的计算资源、预算约束和响应速度需求
- **框架选择**：基于 AutoGen 实现模块化 Agent 编排和灵活的交互模式

### 损失函数 / 训练策略

本文不涉及模型训练。所有 Agent 均基于 GPT-4o mini 的 prompt engineering 实现，每个 Agent 有精心设计的系统 prompt 来定义其角色和评估逻辑。论文附录中提供了所有 9 个 Agent 的完整 prompt。

## 实验关键数据

### 数据集

- **来源**：真实大学课程中收集的学生提交
- **规模**：60 名参与者 × 6 个编程任务 = **360 份提交**
- **任务领域**：机器学习（ML）、计算机视觉（CV1: 人脸检测, CV2: 分割）、自然语言处理（NLP1: 文本分类, NLP2: 摘要, NLP3: 对话机器人）
- **标注**：领域专家标注 3 个二值 rubric 分数 + 定性反馈
- **语言**：韩语

### 评估策略

- **定量**：Rubric 分类准确率（多标签二分类问题）
- **定性**：基于 G-Eval 的 4 维度评估（每条反馈重复 20 次取平均，使用 GPT-4o 评分）
- **Baseline**：SLI（Single-model baseline），同样使用 GPT-4o mini
- **重复**：每个系统独立运行 6 轮

### 主实验

#### Rubric 准确率总体对比

| 系统 | 平均 Rubric 准确率 |
|------|-------------------|
| SLI (单模型) | ~48% |
| **AGACCI** | **~60%** |

AGACCI 在整体 rubric 准确率上比 baseline **高出约 12 个百分点**。

#### 各任务域细粒度 Rubric 准确率（Table 4 / Table 2 精选）

| 任务 | Rubric 项 | AGACCI (mean±std) | SLI (mean±std) |
|------|----------|-------------------|----------------|
| ML | 预处理、训练和可视化 | **0.734±0.098** | 0.174±0.018 |
| ML | Kaggle 提交状态 | 0.473±0.011 | **0.587±0.059** |
| ML | 排行榜准确率阈值 | 0.239±0.000 | **0.685±0.042** |
| CV1 | 贴纸自然对齐在面部 | **0.746±0.027** | 0.179±0.048 |
| CV2 | 竖屏模式错误解决 | **0.680±0.044** | 0.386±0.052 |
| CV2 | 竖屏错误清晰定位 | **0.654±0.009** | 0.434±0.046 |
| NLP1 | Word2Vec 改善准确率 | **0.651±0.019** | 0.406±0.011 |
| NLP2 | 抽取式 vs 生成式对比 | **0.867±0.020** | 0.454±0.051 |
| NLP3 | 稳定 Transformer 收敛 | **0.969±0.020** | 0.577±0.092 |
| NLP3 | 韩语响应生成模型 | **0.959±0.000** | 0.510±0.096 |

### 定性评估结果（G-Eval 4 维度，5 分制）

| 维度 | AGACCI | SLI |
|------|--------|-----|
| Feedback Accuracy | **更高** | 较低 |
| Consistency | **更高，方差更低** | 较低，方差较高 |
| Coherence | **更高** | 较低 |
| Relevance | 相当（方差略高） | 相当 |

### 消融实验

论文未设置严格的消融实验（如逐个移除 Agent），但通过对不同任务域和不同 rubric 项的细粒度分析，间接揭示了各模块的贡献：

| 分析维度 | 发现 |
|---------|------|
| 高复杂度 rubric 项（需多步推理） | AGACCI 平均准确率 >0.73，显著优于 SLI |
| 低复杂度/外部验证 rubric 项（如 Kaggle 状态） | SLI 反而表现更好 |
| NLP 任务（需解释性推理） | AGACCI 优势最大 |
| ML 任务（需外部行为验证） | 两者接近或 SLI 略胜 |
| Meta Evaluator 的作用 | Consistency 提升 + 方差降低归功于该模块 |
| Rubric Interpreter 的作用 | Relevance 分数稳定归功于该模块的 rubric 结构化 |

### 关键发现

1. **AGACCI 在需要多步推理和结构化理解的高复杂度 rubric 项上优势最显著**：包括视觉一致性、错误诊断、比较性摘要策略、深度学习模型稳定实现等
2. **外部验证标准是短板**：涉及代码之外行为（如 Kaggle 提交状态、排行榜验证）的 rubric 项，AGACCI 无法推断，表现不如 baseline
3. **Consistency 提升来自 Meta Evaluator**：通过在最终输出前协调各 Agent 间的矛盾评估，AGACCI 的反馈保持更稳定的评价立场
4. **Relevance 方差较高的原因**：AGACCI 倾向于提供超出 rubric 的前瞻性建议和反思性评论，虽然对教育有益，但在严格的 rubric 对齐评估下可能被扣分

## 亮点与洞察

1. **角色分解思路值得借鉴**：将评估任务分解为解析-分析-执行验证-可视化评估-推理评估-元检查-裁决-总结的流水线，每个角色边界清晰、职责单一。这种分解方式对其他需要多维度判断的 Agent 系统也有参考价值
2. **Meta Evaluator 的设计精妙**：引入一致性检查器来交叉验证多个 Agent 的输出，有效减少矛盾和无根据的判断，是多 Agent 系统中值得推广的设计模式
3. **实际教育场景验证**：使用真实课程的 360 份学生作业进行评估，而非合成数据，增强了结果的实际意义
4. **反馈质量的多维度评估设计**：采用 G-Eval 的 4 维度（准确性、相关性、一致性、连贯性），每条反馈重复评估 20 次取平均，评估方法论本身也值得学习
5. **GPT-4o mini 的务实选择**：考虑教育场景的预算和延迟约束选择轻量模型，而非追求最强模型，体现了实际部署的设计思维

## 局限与展望

1. **无法处理外部验证标准**：系统仅基于代码和 rubric 上下文评估，无法验证 Kaggle 提交状态、排行榜分数等需要外部信息的标准。可通过接入外部 API 或添加截图解析 Agent 改进
2. **缺少严格消融实验**：未逐个移除 Agent 验证其贡献，无法定量衡量各模块的独立价值。9 个 Agent 是否存在冗余尚不明确
3. **数据规模有限**：360 份提交来自单一大学课程，60 名参与者，6 个任务。泛化性存疑——换一个课程或编程语言效果如何？
4. **仅限 Workshop paper**：论文发表于 ICML 2025 的 Multi-Agent Systems Workshop，非主会。篇幅和实验深度受限
5. **韩语语境限制**：所有实验材料为韩语，评估使用 GPT-4o 的韩语能力，跨语言泛化性未验证
6. **Rubric 模糊或矛盾时表现下降**：作者在 Discussion 中承认，当 rubric 标准不够明确或存在矛盾时，Agent 间的一致性会降低
7. **缺少成本分析**：9 个 Agent 串/并联调用带来的 API 成本和延迟开销未量化。相比单模型 baseline，总 token 消耗可能增加数倍
8. **未与更强 baseline 对比**：仅与单一 GPT-4o mini 对比，未与 GPT-4o 单模型、Claude、或其他多 Agent 评估系统比较
9. **学生反馈体验未验证**：是否学生真的从 AGACCI 的反馈中获益更多？缺少用户研究

## 相关工作与启发

- **G-Eval**（Liu et al., 2023）：LLM-as-Judge 的基础工作，AGACCI 的定性评估就采用了这一范式
- **Agent-as-a-Judge**（Zhuge et al., 2024）：将评估功能 dedicated 给一个单独 Agent，AGACCI 进一步将其扩展为多 Agent
- **AutoGen**（Wu et al., 2023）：AGACCI 的基础框架，提供多 Agent 对话和编排能力
- **VISTA**（Lee et al., 2024）：分离任务特定 LLM 组件可以改善教育内容生成，为 AGACCI 的模块化设计提供了启示
- **Rubric is All You Need**（Pathak et al., 2025）：基于 rubric 的 LLM 代码评估增强，与 AGACCI 的 rubric-centric 理念一致

**启发**：多 Agent 系统在需要多维度评判的任务中有天然优势。AGACCI 的角色分解思路可以迁移到代码 review、论文审稿、面试评估等场景。Meta Evaluator 的一致性检查模式值得在其他多 Agent 系统中推广。

## 评分

- 新颖性: ⭐⭐⭐ 多 Agent 评估框架的角色分工设计合理，Meta Evaluator 有亮点，但多 Agent 协作用于评估并非全新概念
- 实验充分度: ⭐⭐ 数据规模有限（360份），缺少消融实验和成本分析，仅与单模型 baseline 对比，Workshop 论文的实验深度
- 写作质量: ⭐⭐⭐⭐ 结构清晰，每个 Agent 的角色定义明确，附录提供完整 prompt 和运行示例，方便复现
- 价值: ⭐⭐⭐ 对教育 AI 领域有直接应用价值，角色分解模式可迁移到其他多维度评估场景，但受限于 Workshop 论文的深度

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](../../ACL2026/llm_agent/agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)
- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](../../ACL2026/llm_agent/securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)
- [\[CVPR 2026\] EpiAgent: An Agent-Centric System for Ancient Inscription Restoration](../../CVPR2026/llm_agent/epiagent_agent_centric_system_for_ancient_inscription_restoration.md)
- [\[ICLR 2026\] FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](../../ICLR2026/llm_agent/featurebench_benchmarking_agentic_coding_for_complex_feature_development.md)
- [\[NeurIPS 2025\] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](../../NeurIPS2025/llm_agent/rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)

<!-- RELATED:END -->
