---
title: >-
  [论文解读] Among Us: Language of Conspiracy Theorists on Mainstream Reddit
description: >-
  [ACL 2026][阴谋论] 分析5亿条Reddit评论的10年纵向数据，发现活跃于阴谋论社区的用户在主流社区中也展现出可检测的独特语言模式（平均87%分类准确率），但这些模式高度依赖社区上下文，社区特定模型比全局模型高出最多17个百分点。
tags:
  - ACL 2026
  - 阴谋论
  - 语言特征
  - Reddit分析
  - 心理语言学
  - 社区适应性
---

# Among Us: Language of Conspiracy Theorists on Mainstream Reddit

**会议**: ACL 2026  
**arXiv**: [2506.05086](https://arxiv.org/abs/2506.05086)  
**代码**: 无  
**领域**: Social Computing / Computational Linguistics  
**关键词**: 阴谋论, 语言特征, Reddit分析, 心理语言学, 社区适应性

## 一句话总结

分析5亿条Reddit评论的10年纵向数据，发现活跃于阴谋论社区的用户在主流社区中也展现出可检测的独特语言模式（平均87%分类准确率），但这些模式高度依赖社区上下文，社区特定模型比全局模型高出最多17个百分点。

## 研究背景与动机

**领域现状**：阴谋论不仅是边缘信仰——它们与疫苗犹豫、公共健康风险甚至对民主制度的威胁（如2021年国会山事件）相关。现有研究主要关注阴谋论内容的检测，但忽略了阴谋论信仰者在主流空间中的语言表现。

**现有痛点**：(1) 已知阴谋论者使用特定修辞风格和词汇，但不清楚这些模式是仅限于阴谋论空间还是渗透到主流交流中；(2) 现有检测方法多关注内容层面（如主题词），忽略了与讨论话题无关的语言风格特征。

**核心矛盾**：阴谋论者是否具有一种渗透到所有交流中的"阴谋论心态"（monological worldview），还是他们能够完全适应不同社区的语言规范？

**本文目标**：利用大规模纵向数据系统检验阴谋论社区用户在主流空间中的语言可区分性。

**切入角度**：使用LIWC-22心理语言学特征（而非话题词）构建用户语言画像，在22个主流社区上分别训练分类器。

**核心 idea**：阴谋论用户的语言确实可区分，但区分模式高度依赖社区——没有单一的全局模型能捕捉这些模式，需要社区特定的分析。

## 方法详解

### 整体框架

数据收集（r/conspiracy全部评论 + 22个主流社区） → LIWC-22特征提取（110维） → 用户级别特征聚合 → 每个社区训练Random Forest分类器 → 特征重要性分析（SHAP值） → 跨社区相似性分析。

### 关键设计

1. **大规模纵向数据构建**:

    - 功能：提供可靠的长期语言行为数据
    - 核心思路：从Pushshift Reddit数据集提取2013-2023年约5.1亿条评论，覆盖r/conspiracy的98万用户和22个主流社区。排除bot和低活跃度用户（<20条评论）
    - 设计动机：需要足够的评论量来构建稳定的用户语言画像，短期或少量数据可能被噪声主导

2. **社区特定分类实验**:

    - 功能：检验语言可区分性是否跨社区一致
    - 核心思路：对每个主流社区独立训练Random Forest，正类=在r/conspiracy有过评论的用户，负类=随机采样等量的普通用户。重复5次随机采样以减少方差。使用置换检验验证统计显著性
    - 设计动机：分类器不是目的本身，而是用来量化语言可区分性的代理工具

3. **SHAP特征重要性分析和跨社区聚类**:

    - 功能：揭示哪些语言特征最具区分力，以及不同社区的区分模式是否相似
    - 核心思路：对每个社区模型计算SHAP值，得到110维特征重要性向量，然后用余弦相似度+层级聚类分析跨社区的模式相似性
    - 设计动机：如果所有社区用相同的特征区分，说明有全局的"阴谋论语言"；如果特征因社区而异，说明语言表达是上下文适应的

### 损失函数 / 训练策略

Random Forest使用网格搜索和5折交叉验证调参，80/20训练-测试划分。特征归一化仅在训练数据上进行。100次置换检验评估统计显著性。

## 实验关键数据

### 主实验

| 指标 | 数值 | 说明 |
|------|------|------|
| 平均分类准确率 | 87% | 跨20+个社区的二分类 |
| 社区特定 vs 全局 | 最多+17pp | 社区特定模型显著优于全局模型 |
| 统计显著性 | p<0.01 | 所有社区的置换检验均显著 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 活跃度阈值 | 高活跃度效果更好 | 更多评论→更稳定的语言画像 |
| r/AskReddit正类 | 准确率~随机 | 通用社区用户无法区分（负对照） |
| r/MensRights正类 | 中等准确率 | 意识形态社区也有部分区分力 |

### 关键发现
- 阴谋论用户的语言确实在主流空间中可检测——平均87%准确率远超随机
- 但没有单一全局模型能捕捉这些模式——社区特定模型比全局模型高出最多17个百分点
- 这表明阴谋论用户的语言表达是动态适应的——虽然有独特特征，但会根据社区规范调整
- r/AskReddit用户（负对照）无法被区分，验证了效果的特异性

## 亮点与洞察
- "可区分但上下文依赖"是一个精妙的发现——既支持"阴谋论心态"的存在，又说明它不是简单的全局标签
- 对内容审核策略有直接启示——统一的检测模型不够，需要社区定制的方法
- 使用LIWC心理语言学特征（而非话题词）确保分析的是语言风格而非讨论内容

## 局限与展望
- 将"在r/conspiracy评论过"等同于"阴谋论信仰者"可能过于宽泛
- LIWC的词典方法可能错过新兴的语言模式
- 仅分析Reddit一个平台，其他社交媒体上的模式可能不同
- 未来可结合内容分析和风格分析进行更细粒度的研究

## 相关工作与启发
- **vs 内容检测方法**: 关注语言风格而非内容，揭示了更深层的认知特征
- **vs 用户pathway研究**: 不是追踪用户如何进入阴谋论社区，而是分析他们在主流空间的行为
- **vs 社区检测**: 揭示了跨社区的行为适应性，补充了社区边界研究

## 评分
- 新颖性: ⭐⭐⭐⭐ 从语言风格角度研究阴谋论用户的跨社区行为
- 实验充分度: ⭐⭐⭐⭐⭐ 5亿条评论、10年纵向、22个社区、统计检验
- 写作质量: ⭐⭐⭐⭐ 研究设计严谨，负对照完善
- 价值: ⭐⭐⭐⭐ 对社交媒体治理和阴谋论研究有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Conspiracy Theories and Where to Find Them on TikTok](../../ACL2025/social_computing/conspiracy_theories_and_where_to_find_them_on_tiktok.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[ACL 2026\] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)

<!-- RELATED:END -->
