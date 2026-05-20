---
title: >-
  [论文解读] Which of These Best Describes Multiple Choice Evaluation with LLMs?
description: >-
  [ACL 2025][LLM/NLP][多选题评估] 系统论证 MCQA 作为 LLM 标准评估格式存在三大类问题：(1) 格式缺陷——无法测试生成/主观性、不匹配 LLM 真实使用场景、不能充分测试知识深度；(2) 数据集缺陷——泄露、不可回答、捷径和饱和…
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "多选题评估"
  - "MCQA 缺陷"
  - "教育测试学"
  - "Constructed Response"
  - "Item Response Theory"
---

# Which of These Best Describes Multiple Choice Evaluation with LLMs?

**会议**: ACL 2025  
**arXiv**: [2502.14127](https://arxiv.org/abs/2502.14127)  
**领域**: LLM 评估方法论  
**关键词**: 多选题评估, MCQA 缺陷, 教育测试学, Constructed Response, Item Response Theory

## 一句话总结

系统论证 MCQA 作为 LLM 标准评估格式存在三大类问题：(1) 格式缺陷——无法测试生成/主观性、不匹配 LLM 真实使用场景、不能充分测试知识深度；(2) 数据集缺陷——泄露、不可回答、捷径和饱和；(3) 模型行为问题——鲁棒性差、选项偏置和不忠实解释。借鉴教育测试学提出 Constructed Response、Explanation MCQA、IRT 分析等系统化修复方案。

## 研究背景与动机

**领域现状**：MCQA 是 LLM 评估的标准范式，因其简单易评分且类似人类测试而广受欢迎。HELM 中 32%、GPT-4 技术报告中 71%、OpenLLM 排行榜中 79% 的任务为 MCQA。

**现有痛点**：(1) LLM 在实际使用中约 90%+ 的查询为开放式生成（代码/写作/解释），与 MCQA 评估严重脱节；(2) MCQA 数据集普遍存在泄露、标注错误和捷径问题（MMLU 已被发现大量错误）；(3) LLM 对选项符号、顺序和措辞高度敏感，不同评估设置产生矛盾排名；(4) 教育测试学已积累数十年 MCQA 最佳实践，但 NLP 社区几乎未借鉴。

**核心矛盾**：MCQA 之所以流行是因为"简单"，但评估最重要的属性不是简单，而是能预测系统在部署时的实际表现。

**本文目标** 论证 MCQA 不是 LLM 评估的金标准，并从教育测试学引入系统化改进方案。

**核心 idea**：MCQA 是有缺陷但可修复的——借鉴教育测试学可从格式、数据集和评分三个层面系统改进。

## 方法详解

### 整体框架

论文按三个层面组织论证：格式问题（MCQA 本身的局限）→数据集问题（现有 MCQA 数据集的质量）→模型行为问题（LLM 在 MCQA 上的异常行为），每个层面都对应教育测试学的改进方案。

### 关键设计

1. **格式层面：MCQA 的固有局限与生成式替代**
    - **"选最佳答案"过于刚性**：单一金标准答案无法评估主观性任务（常识/道德/文化），用户在常识 MCQ 中 20% 的情况下认为干扰项比金标准更合理；选择 ≠ 生成，LLM 的验证和生成能力不一致
    - **Constructed Response (CR)**：去掉选项让 LLM 生成短答案，对应教育学的构造式回答，更能暴露知识空缺。现有 MCQ 可通过去掉选项直接转化
    - **Explanation MCQA (E-MCQA)**：除选答案外还需提供解释，可检验推理过程的忠实性，支持主观任务的部分赋分。类似教育学中的"展示你的工作"
    - **设计动机**：LLM 的实际用例（代码/写作/解释）均为生成任务，评估应匹配

2. **数据集层面：四大质量问题与修复方案**
    - **泄露**：GPT-3 已见过 RACE 测试集的 45%；解决方案——持续更新的"活题库"（类似教育考试每年换题）
    - **不可回答**：标签错误、多个正确选项、歧义等使题目无法正确回答（MMLU 已被发现大量此类错误）；解决方案——使用教育学的 MCQ 编写清单和规范验证每道题
    - **捷径**：LLM 仅通过选项（不看题目）就能答对，说明存在分布差异泄漏；解决方案——统一设计（同一来源/方法生成各部分）、对比集（Contrast Sets）确保模型关注所有输入
    - **饱和**：模型持续提分使数据集区分度消失；解决方案——Item Response Theory (IRT) 过滤保留高难度高区分度题目，对抗式数据收集创建"对模型难但对人类简单"的题目

3. **模型行为层面：三大异常与评估启示**
    - **鲁棒性差**：选项顺序、符号、措辞改变即导致答案翻转，反映泄露或偏置而非真实能力
    - **选项偏置**：LLM 基于位置/符号/短语（如"以上都不是"）选答案而非内容
    - **不忠实解释**：LLM 即使选对答案也可能给出逻辑不一致的解释，且解释质量常高于众包标注者，易误导用户
    - **设计动机**：这些问题本质上源于 MCQA 格式和数据集的缺陷，修复前两层可更好地诊断这些问题

## 实验关键数据

### MCQA 在 LLM 评估中的过度代表

| 评估平台 | MCQA 任务占比 | 用户实际 MCQA 需求 |
|---------|:-----------:|:---------------:|
| HELM | 32% | ~7% (ShareGPT) |
| GPT-4 技术报告 | 71% | <6.3% (WildChat) |
| OpenLLM 排行榜 | 79% | - |

### MCQA 数据集质量问题案例

| 问题类型 | 代表案例 | 影响 |
|---------|--------|-----|
| 泄露 | RACE 测试集 45% 被 GPT-3 见过 | 混淆记忆与泛化 |
| 不可回答 | MMLU 发现大量标签错误和歧义 | 高估模型准确率 |
| 捷径 | HellaSwag 仅用选项即可高分 | 不测试真实理解 |
| 饱和 | 多个模型在经典基准上超 90% | 无法区分模型能力 |

### 关键发现

- Constructed Response 下学生（和模型）的表现比 MCQA 更差，说明 MCQA 确实高估了知识水平
- IRT 可同时识别困难题目和有缺陷题目（负区分度→可能有标注错误）
- 对抗式数据收集（游戏化）可产生长期保持难度的数据集
- 校准评分（置信度/负分/排除法）可抑制模型猜测行为

## 亮点与洞察

1. **教育测试学 × NLP 的跨领域视角**：从 1914 年 MCQA 起源到 IRT、Bloom 分类学、对抗式考试等，系统地将百年教育测试研究引入 NLP 社区
2. **"MCQA 测的是验证而非生成"**：验证和生成是独立技能，MCQA 排行榜无法反映 LLM 帮助用户的真实能力
3. **实用的改进路线图**：不主张废弃 MCQA 而是分层修复——格式（CR/E-MCQA）、数据（清单/对比集/IRT）、评分（校准/部分赋分），每个改进都可独立采用
4. **"Benchmarking 101" 设计指南**：论文最后给出了从头设计评估基准的完整流程建议

## 局限与展望

1. 立场论文性质，未提供新的实验数据或基准实现
2. 生成式替代方案（CR/E-MCQA）的评分可靠性仍需大规模验证
3. IRT 需要大量模型评估数据来拟合参数，小规模场景适用性有限
4. 某些领域（如法律/医学执照考试）中 MCQA 有不可替代的制度性价值，论文未充分讨论
5. 活题库方案实际操作难度大——需要持续的题目开发资源

## 相关工作与启发

- Bloom 分类学的知识层次（记忆→理解→应用→分析→评估→创造）为评估设计提供了比"准确率"更有指导性的框架
- IRT 在 NLP 中仍属新兴应用，多维 IRT (MIRT) 可识别模型在不同推理类型上的优劣势
- 对抗式数据收集的游戏化设计（如 Quiz Bowl 模式）值得更广泛推广
- 跨模态/跨语言 MCQA 面临的问题（翻译误差传播、文化偏置）同样适用于其他评估格式

## 评分

⭐⭐⭐⭐

- **新颖性** ⭐⭐⭐⭐：跨学科视角独到，将教育测试学系统引入 NLP 评估讨论
- **论证充分度** ⭐⭐⭐⭐⭐：三层结构清晰，每个问题都有数据支持和对应解法
- **写作质量** ⭐⭐⭐⭐⭐：行文流畅、结构精巧、标题出色
- **价值** ⭐⭐⭐⭐：对 LLM 评估社区具有重要的方向性指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options](llm_robustness_incorrect_mcq.md)
- [\[ACL 2025\] Which Demographics Do LLMs Default to During Annotation?](which_demographics_do_llms_default_to_during_annotation.md)
- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)
- [\[ACL 2025\] ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation](atrie_legal_interpretation.md)
- [\[ACL 2025\] How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs](how_llms_comprehend_temporal_meaning_in_narratives_a_case_study_in_cognitive_eva.md)

</div>

<!-- RELATED:END -->
