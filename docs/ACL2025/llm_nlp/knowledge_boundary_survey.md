---
title: >-
  [论文解读] Knowledge Boundary of Large Language Models: A Survey
description: >-
  [ACL 2025][LLM/NLP][知识边界] 提出LLM知识边界的形式化定义框架——三层嵌套边界（Outward⊂Parametric⊂Universal）和四类知识分类（PAK/PSK/MSU/MAU），围绕"为何/如何识别/如何缓解"三个问题系统综述相关研究。
tags:
  - ACL 2025
  - LLM/NLP
  - 知识边界
  - 幻觉
  - 不确定性估计
  - 校准
  - 知识分类
---

# Knowledge Boundary of Large Language Models: A Survey

**会议**: ACL 2025  
**arXiv**: [2412.12472](https://arxiv.org/abs/2412.12472)  
**代码**: [GitHub](https://github.com/li-moxin/knowledge_boundary_survey)  
**领域**: LLM NLP  
**关键词**: 知识边界, 幻觉, 不确定性估计, 校准, 知识分类

## 一句话总结

提出LLM知识边界的形式化定义框架——三层嵌套边界（Outward⊂Parametric⊂Universal）和四类知识分类（PAK/PSK/MSU/MAU），围绕"为何/如何识别/如何缓解"三个问题系统综述相关研究。

## 研究背景与动机

### 领域现状

**领域现状**：**领域现状**: LLM在参数中存储大量知识，但在记忆和利用某些知识方面仍存在局限，导致不真实、不准确的回复。**现有痛点**: Know-Unknown Quadrant概念性强但缺乏形式化；已有形式化定义仅关注特定LLM。**核心矛盾**: 对LLM知识边界缺乏清晰统一的定义，阻碍了系统性的识别和缓解策略。**本文目标**: 提供全面的形式化知识边界定义并系统综述相关研究。**切入角度**: 从知识是否被人类已知、是否嵌入参数、是否可实证验证三个维度定义边界。**核心idea**: 将知识分为PAK（无论怎么问都会）、PSK（看怎么问）、MSU（模型不会但人类知道）、MAU（人类也不知道）四类。

## 方法详解

### 整体框架

从三个维度定义三层嵌套知识边界：(1)Outward（实证可验证）；(2)Parametric（参数中存在）；(3)Universal（人类已知全集）。基于此划分四类知识，围绕三个RQ展开综述。

### 关键设计

1. **四类知识形式化定义**:
    - 功能：将LLM知识形式化分为PAK/PSK/MSU/MAU四类
    - 核心思路：PAK: $K_{PAK}=\{k \in \mathcal{K} | \forall (q,a) \in \hat{Q}_k, P_\theta(a|q)>\epsilon\}$——无论哪种prompt表述都能正确回答。PSK: 参数中存在但对prompt敏感。MSU: 模型不具备但人类已知。MAU: 人类也未知
    - 设计动机：PAK/PSK的区分特别有洞见——同一知识因prompt不同可能"会"或"不会"，直接指导prompt工程

2. **不良行为与知识类型的映射**:
    - 功能：将LLM三类不良行为映射到知识类型
    - 核心思路：PSK→被上下文误导的不真实回复；MSU→事实性幻觉（领域知识不足/知识过时/过度自信）；MAU→对模糊知识随机回答/对争议知识偏见回复
    - 设计动机：为针对性改进提供路线图——知道是哪种知识问题才能选择正确的缓解策略

3. **识别与缓解方法分类**:
    - 功能：对识别方法（不确定性估计/校准/探测）和缓解方法（prompt优化/RAG/知识编辑/弃权）进行系统分类
    - 核心思路：识别方法按是否需要内部状态访问分类；缓解方法按知识类型和参数修改程度分类
    - 设计动机：不同知识类型需要不同策略——PSK用prompt优化，MSU用RAG/编辑，MAU用弃权/澄清

### 损失函数 / 训练策略

作为综述论文不涉及训练，但系统梳理了各类缓解方法的训练策略：SFT弃权训练、RL诚实性对齐、RAG检索增强等。

## 实验关键数据

### 主实验

综述论文不含自身实验，但系统梳理了关键映射：

| 知识类型 | 不良行为 | 识别方法 | 缓解策略 |
|---------|---------|---------|---------|
| PAK | 无 | 高概率阈值验证 | — |
| PSK | 上下文误导 | Prompt扰动/不确定性分解 | Prompt优化/ICL/推理/解码 |
| MSU | 事实幻觉 | 语义一致性/校准/探测 | RAG/知识编辑/弃权 |
| MAU | 偏见/随机 | 未充分探索 | 对齐训练/澄清提问 |

### 消融实验

综述梳理的关键对比：

| 对比维度 | 发现 |
|---------|------|
| 不确定性估计 vs 校准 | 前者关注整体分布，后者关注特定预测——概念相近但本质不同 |
| 认知不确定性 vs 偶然不确定性 | 分别对应参数边界和外显边界间的差距 |
| 弃权 vs 澄清 | 前者对MAU有效但可能过度弃权，后者更友好但成本更高 |

### 关键发现

1. **多数识别方法仅关注外显边界**——参数边界的识别仍是开放问题
2. **LLM过度自信问题严重**——在不熟悉主题上保持高置信度但输出错误
3. **弃权策略未区分MSU和MAU**——导致可回答问题也被拒绝，用户体验差
4. **知识边界随训练数据时间截止而动态变化**——LLaMA2虽训练至2022但倾向使用2019数据

## 亮点与洞察

- **形式化知识分类框架**是核心贡献——将概念性的Known-Unknown Quadrant转化为可操作的数学定义
- **三层嵌套结构**（Outward⊂Parametric⊂Universal）提供清晰问题定位框架
- **PAK/PSK区分**直接指导prompt工程——很多"不会"的问题只是问法不对
- **Summary Box设计**便于读者快速把握每节核心

## 局限与展望

- MAU（人类未知知识）讨论较少，因该领域本身研究匮乏
- 知识边界的形式化依赖阈值ε，最优选择缺乏指导
- 综述截至2024年底，快速发展的LLM领域可能已有更新
- 未深入讨论多模态场景下的知识边界

## 相关工作与启发

- **vs Know-Unknown Quadrant（Yin et al. 2023）**: 概念性框架——本文提供形式化定义
- **vs 幻觉综述（Ji et al. 2023）**: 未从知识边界角度分析——本文建立了映射关系
- **vs Semantic Entropy（Kuhn et al. 2023）**: 具体方法——本文提供方法分类框架
- **启发**: 识别知识边界应为LLM可靠部署的第一步——知道"模型不知道什么"比让它"知道更多"更重要

## 评分

- 新颖性: ⭐⭐⭐⭐ 形式化定义框架有原创贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 系统全面，从动机到识别到缓解
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，Summary Box便于把握
- 价值: ⭐⭐⭐⭐⭐ 对LLM可靠性改进有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [\[ACL 2025\] Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [\[ACL 2025\] Acquisition and Application of Novel Knowledge in Large Language Models](acquisition_and_application_of_novel_knowledge_in_large_language_models.md)
- [\[ACL 2025\] A Survey of Large Language Models in Psychotherapy: Current Landscape and Future Directions](a_survey_of_large_language_models_in_psychotherapy_current_landscape_and_future_.md)
- [\[ACL 2025\] When Large Language Models Meet Speech: A Survey on Integration Approaches](when_large_language_models_meet_speech_a_survey_on_integration_approaches.md)

<!-- RELATED:END -->
