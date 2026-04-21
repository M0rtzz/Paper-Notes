---
title: >-
  [论文解读] ScEdit: Script-based Assessment of Knowledge Editing
description: >-
  [ACL 2025][知识编辑] 提出 ScEdit，一个基于脚本（Script）的知识编辑评估基准，将传统的"What"类事实回忆评估扩展为"How"类程序性推理评估，同时引入 token 级和文本级双层评估体系，揭示了现有知识编辑方法在实际应用场景中的显著不足。
tags:
  - ACL 2025
  - 知识编辑
  - 脚本生成
  - 评估基准
  - 程序性推理
  - 文本级评估
---

# ScEdit: Script-based Assessment of Knowledge Editing

**会议**: ACL 2025  
**arXiv**: [2505.23291](https://arxiv.org/abs/2505.23291)  
**代码**: 有 (https://github.com/asdfo123/ScEdit)  
**领域**: NLP / 知识编辑评估  
**关键词**: 知识编辑, 脚本生成, 评估基准, 程序性推理, 文本级评估

## 一句话总结

提出 ScEdit，一个基于脚本（Script）的知识编辑评估基准，将传统的"What"类事实回忆评估扩展为"How"类程序性推理评估，同时引入 token 级和文本级双层评估体系，揭示了现有知识编辑方法在实际应用场景中的显著不足。

## 研究背景与动机

知识编辑（KE）领域近年来发展迅速，但**现有评估框架存在根本性不足**：

**评估过于简单**：当前指标（Efficacy、Generalization、Locality）主要基于"What"类问题的下几个 token 预测，很多方法已接近满分

**脱离实际应用**：LLM 越来越多地被部署为 Agent，用户提出的是"How"类程序性问题（如"如何从北京到新加坡旅行？"），而非简单事实查询

**缺乏文本级评估**：现有评估仅关注 token 概率，忽略了完整文本生成的质量

以论文中的例子说明：新加坡对中国游客实施了免签政策，但未更新的 LLM 仍会建议用户申请签证。这种**错误不是简单的事实错误，而是会导致整个行动计划（脚本）出错**。

ScEdit 的动机是：将知识编辑评估从"能正确回答事实问题"提升到"能正确指导多步骤程序性任务"，更贴近 LLM 在实际部署中面临的挑战。

## 方法详解

### 整体框架

ScEdit 包含三个核心要素：
1. **事实（Facts）**：标准的 (s, r, o^c) → (s, r, o) 知识三元组编辑
2. **脚本问题（Script Questions）**：基于编辑事实的"How"类多步骤问题
3. **脚本（Scripts）**：模型对脚本问题的多步骤回答

评估体系分为两层：
- **Token 级**：通过填空提示评估 S-ES、S-NS、S-BO 等指标
- **文本级**：通过自动评估（GPT-4）和人工评估，评估 Executability、Coherence、Consistency、Completeness

### 关键设计

1. **脚本问题生成**

    - 功能：从每个知识编辑三元组出发，生成会受编辑影响的"How"类问题
    - 核心思路：
        - 使用 GPT-4 + few-shot prompt 生成候选脚本问题
        - 人工过滤确保问题确实依赖于被编辑的事实
        - 例如：(Panamera, 制造商, Porsche → Ford) → "如何预约 Panamera 的维修服务？"
    - 设计动机：模拟真实场景中 LLM 作为 Agent 需要回答的程序性问题

2. **Token 级评估指标**

    - **S-ES（Script-based Efficacy Success）**：编辑后模型在脚本填空提示下是否偏好新答案
        - 将原始脚本截断到首次出现旧答案的位置，拼接脚本问题构成填空提示
        - 检查 P(o|Q̃) > P(o^c|Q̃)
    - **S-NS（Script-based Neighborhood Success）**：编辑是否影响了无关邻居事实
        - 构建基于邻居事实的脚本填空提示
        - 检查模型是否仍保留邻居事实
    - **S-BO（Script-based Bleedover）**：编辑对语义相近事实的溢出影响
        - 度量编辑前后邻居事实准确度的下降

3. **文本级评估指标（7 分 Likert 量表）**

    - **Executability（可执行性）**：脚本步骤是否在逻辑上可执行（不考虑知识更新）
    - **Coherence（连贯性）**：脚本是否与更新后的事实一致
    - **Consistency（一致性）**：脚本内部是否无矛盾（不混淆新旧事实）
    - **Completeness（完整性）**：脚本是否充分回答了问题的所有方面
    - 设计动机：token 级指标无法捕捉生成文本的整体质量和编辑效果

4. **两个子任务**

    - **ScEdit-CF（反事实编辑）**：基于 CounterFact 数据集，测试反事实知识编辑在脚本场景中的表现
    - **ScEdit-T（时间编辑）**：基于 WikiFactDiff，测试时间更新类编辑（更接近真实场景）

### 数据集统计

| 子任务 | 编辑案例数 | S-Eff. 样本 | S-Spec. 样本 |
|--------|-----------|------------|-------------|
| ScEdit-CF | 1,830 | 7,342 | 13,672 |
| ScEdit-T | 1,762 | 7,038 | 6,597 |

## 实验关键数据

### 主实验（Token 级，ScEdit-CF 和 ScEdit-T）

| 方法 | 模型 | ES↑ | S-ES↑ | S-NS↑ | ES↑(T) | S-ES↑(T) | S-BO↓(T) |
|------|------|-----|-------|-------|--------|----------|----------|
| Base | GPT2-XL | 20.55 | 21.18 | 81.52 | 44.27 | 41.72 | 0.00 |
| FT | GPT2-XL | **100.00** | 71.27 | 65.08 | 87.17 | 52.80 | 1.15 |
| ROME | GPT2-XL | 99.95 | **74.76** | 80.24 | 99.15 | 68.00 | 0.13 |
| MEMIT | GPT2-XL | 93.72 | 58.11 | **81.16** | 81.44 | 52.13 | **0.03** |
| PROMPT | GPT2-XL | 96.28 | 69.63 | 42.88 | **99.49** | **84.39** | 0.54 |
| FT | GPT-J | **100.00** | 83.94 | 25.81 | 99.60 | **97.90** | 5.47 |
| ROME | GPT-J | 99.95 | **86.50** | 83.35 | 99.60 | 74.29 | 0.28 |
| MEMIT | GPT-J | 99.95 | 74.59 | **85.07** | 99.09 | 64.66 | **0.08** |

关键：S-ES 相比传统 PS 指标平均下降 27%，证明脚本场景更具挑战性。

### 文本级评估（ScEdit-CF, LLAMA3-8B）

| 方法 | Exec.↑ | Coh.↑ | Cons.↑ | Comp.↑ |
|------|--------|-------|--------|--------|
| Base Model | 6.74 | 2.48 | **6.86** | 5.40 |
| FT | 2.94 | 2.97 | 6.17 | 2.17 |
| ROME | 6.41 | **4.32** | 6.57 | 4.67 |
| MEMIT | **6.54** | 3.67 | 6.70 | 4.98 |
| PROMPT | 6.36 | 4.35 | 6.05 | **5.49** |

### 关键发现

1. **S-ES 比 ES 难得多**：S-ES 平均比传统 PS 低 27%，脚本场景显著增加了编辑难度
2. **FT 在文本级几乎失效**：Executability 仅 2.94/7，几乎摧毁了模型的基本能力
3. **ROME 综合最优**：在 token 级和文本级都表现稳健，locate-then-edit 策略在脚本场景中仍然有效
4. **MEMIT 保局部性最强**：S-NS 和 S-BO 最优，但牺牲了编辑有效性
5. **PROMPT 编辑效果好但局部性差**：S-NS 仅 42.88%，上下文注入会严重干扰邻居事实
6. **Token 级与文本级评估维度不同**：Coherence 与 S-ES 仅弱相关，Consistency 与 token 级指标几乎无关，说明两层评估各有不可替代的价值
7. **编辑效果-局部性权衡在脚本场景中更加突出**

## 亮点与洞察

- **问题定义有价值**：首次将知识编辑评估从事实回忆扩展到程序性推理，更贴近实际部署场景
- **双层评估体系互补**：token 级和文本级评估捕捉不同维度，相关性分析证明了这一点
- **ScEdit-T 子任务更有实际意义**：基于真实 Wikipedia 时间更新，而非人工构造的反事实
- **发现了 token 级高分但文本级失败的案例**：揭示了现有评估框架的盲区

## 局限与展望

1. **模型规模有限**：仅评测了 <10B 的模型，更大模型可能有不同表现
2. **仅评估单次编辑**：未考虑大规模/序列编辑场景
3. **数据由 GPT-4 生成**：可能对因果语言模型有偏向，部分样本可能不够自然
4. **未涉及具身 AI 场景**：脚本执行仅限于人类级别，机器人执行是重要的未来方向
5. **编辑方法本身仍基于三元组范式**：如何开发适合脚本场景的编辑方法是开放问题
6. **文本级自动评估可能存在高估/低估**：虽然有人工验证，但覆盖面有限

## 相关工作与启发

- 与 ROME/MEMIT 的评估体系互补：ScEdit 提供了更具挑战性和实际意义的评估场景
- 脚本生成工作（Schank & Abelson, 1975; CoScript）提供了脚本的理论基础
- 可启发开发"脚本感知"的知识编辑方法：不仅编辑事实，还要确保依赖该事实的推理链正确更新
- 多跳推理评估（Ripple Effect, MeLLo）是相关方向，但 ScEdit 更进一步关注程序性推理

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将知识编辑评估提升到脚本/程序性推理层面是有价值的贡献，"How"类评估视角独特
- **实验充分度**: ⭐⭐⭐⭐ — 三个模型、六种编辑方法、token+文本双层评估、人工+自动评估、指标相关性分析，覆盖面全面
- **写作质量**: ⭐⭐⭐⭐ — 问题动机阐述清晰，评估体系设计逻辑严密，案例图示直观
- **价值**: ⭐⭐⭐⭐ — 对知识编辑领域的评估范式提出了重要的补充，揭示了现有方法在实际场景中的差距

<!-- RELATED:START -->

## 相关论文

- [SAKE: Steering Activations for Knowledge Editing](sake_steering_activations_for_knowledge_editing.md)
- [Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing](revealing_the_deceptiveness_of_knowledge_editing_a_mechanistic_analysis_of_super.md)
- [Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)
- [CompKe: Complex Question Answering under Knowledge Editing](compke_complex_question_answering_under_knowledge_editing.md)
- [Efficient Knowledge Editing via Minimal Precomputation](efficient_knowledge_editing.md)

<!-- RELATED:END -->
