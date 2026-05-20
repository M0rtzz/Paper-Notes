---
title: >-
  [论文解读] Enabling LLM Knowledge Analysis via Extensive Materialization
description: >-
  [ACL 2025][LLM/NLP][知识库构建] 本文提出通过递归查询和结果整合将 LLM 的事实知识大规模物化为知识库的方法论，构建了包含 1.01 亿三元组、290 万实体的 GPTKB，首次全面分析了 GPT-4o-mini 知识的规模、准确性、偏见、时效性和一致性。
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "知识库构建"
  - "LLM知识物化"
  - "知识图谱"
  - "递归查询"
  - "事实知识分析"
---

# Enabling LLM Knowledge Analysis via Extensive Materialization

**会议**: ACL 2025  
**arXiv**: [2411.04920](https://arxiv.org/abs/2411.04920)  
**代码**: [gptkb.org](https://gptkb.org)  
**领域**: LLM/NLP  
**关键词**: 知识库构建, LLM知识物化, 知识图谱, 递归查询, 事实知识分析

## 一句话总结

本文提出通过递归查询和结果整合将 LLM 的事实知识大规模物化为知识库的方法论，构建了包含 1.01 亿三元组、290 万实体的 GPTKB，首次全面分析了 GPT-4o-mini 知识的规模、准确性、偏见、时效性和一致性。

## 研究背景与动机

LLM 内化了大量事实知识，这是其成功的重要因素。自 Petroni et al. (2019) 以来，分析 LLM 知识已成为一个独立子领域。然而，现有方法存在根本性问题——**可用性偏差（Availability Bias）**：

**瞬时单点探索**：每次只用一个问题探测，结果用完即丢

**受限于预定义样本**：只能发现实验者预设范围内的知识

**浅尝辄止**：通常只用几百到几十万样本，未触及 LLM 知识的广度和深度

例如，作者发现 GPT-4o-mini 拥有关于艺术流派、兴趣爱好等大量知识，但这些在现有知识库中根本没有覆盖。因此，作者提出将 LLM 知识**持久化物化为知识库**——不再是一次性探测后丢弃，而是构建可重复使用的结构化资源。

这一任务面临三大挑战：(1) 运行时间和成本——推理慢且贵；(2) 方差、幻觉和范围界定——需要高产出但不鼓励幻觉；(3) 全局不一致性——连续提示可能产生重复关系和实体。

## 方法详解

### 整体框架

GPTKB 构建分为两个阶段：

**阶段一：知识抽取（Knowledge Elicitation）**
- 从种子实体（Vannevar Bush）出发
- 提示 LLM 返回关于该实体的三元组形式知识
- 通过 NER 识别三元组对象中的新命名实体
- 将新实体加入队列进行广度优先搜索（BFS），迭代扩展知识图

**阶段二：知识整合（Knowledge Consolidation）**
- 关系聚类：合并重复关系名称
- 类别聚类：合并重复类别名称
- 分类体系构建：为类别构建连贯的层级分类
- 实体去重：消除重复实体

### 关键设计

**知识提示设计**：
- 不固定返回三元组数量，而是根据实体知名度给出弹性指引，使爱因斯坦返回的三元组远多于普通实体
- 要求至少返回一个 `instance_of` 三元组以进行结构化
- 使用 OpenAI 的结构化输出功能减少解析错误和幻觉

**命名实体识别（NER）处理**：
- 早期实验因语言知识、翻译等内容导致主题跑偏
- 现有 NER 框架难以处理无上下文的短实体标签
- 最终采用 LLM 自身进行命名实体识别，批量处理多个候选

**关系聚类算法**（Algorithm 1）：
- 基于贪心策略，将低频关系合并到与其最相似的高频关系
- 使用 SentenceTransformers 计算余弦相似度
- 采用自适应阈值：低频关系阈值更低（更积极合并），高频关系阈值更高（更保守）
- 参数：α=1.4，最高阈值 H=0.95，最低阈值 L=0.75

**分类体系构建算法**（Algorithm 2）：
- 先让 LLM 生成高层分类骨架
- 为每个现有类别计算通用性分数并排序
- 深度优先搜索找到最低匹配节点后，让 LLM 更新子分类体系
- 可能自动生成中间类别节点

**实体去重**：
- 采用基于分块键（blocking key）的标准去重方法
- 以人物类别为重点，使用出生日期作为分块键
- 同一块内条件：标签嵌入余弦相似度 > 0.85 且 30% 三元组完全匹配

### 损失函数 / 训练策略

本文不涉及模型训练。核心策略是**批量 API 调用 + 后处理整合**：
- 使用 GPT-4o-mini 的批量请求功能，启动后可并行发送 100 个批次（每批 10000 实体）
- BFS 深度 10 层，总共 2200 个批次，580 万实体被提示
- 总耗时 27 小时，API 总成本 3500 美元

## 实验关键数据

### 主实验

**GPTKB 规模统计**：
- 2.9M 实体，101M 三元组
- 567K 关系（聚类前 788K），4715 个类别（聚类前 103K）
- 平均每个实体 35 个三元组
- 37M 三元组对象为实体，64M 为字面量

**精度评估（基于 1000 样本）**：
- 实体级：74% 可验证，9% 合理，17% 不可验证
- 三元组级：31% 真实，61% 合理，1% 不合理，7% 错误
- 分类体系：64% 的子类-超类边被判为正确，70% 选择的超类被认为是最佳选项

**与 Wikidata 对比**：
- 37% 的 GPTKB 实体存在于 Wikidata，63% 是新发现的
- Vannevar Bush 的 41 个三元组中超过 10 个不在 Wikidata 中
- GPTKB 拥有 Wikidata 未建模的关系：historical_significance (342K), art_style (84K), hobbies (24K)

### 消融实验

**不同 LLM 精度对比**：

| 模型 | 可验证三元组 | Wikidata实体 | 可验证实体 |
|------|-------------|-------------|-----------|
| GPT-4o-mini | 0.38 | 0.78 | 0.80 |
| Llama 3.1 70B | 0.69 | 0.83 | 0.95 |
| GPT-4o | 0.78 | 0.88 | 0.98 |

**知识一致性测试（Vannevar Bush 重复 100 次）**：
- 两个明显聚类：52 次运行平均 21 个三元组，32 次运行平均 38 个三元组
- 第一聚类内 1116 个总三元组中 79 个唯一，平均每个三元组被 14 次运行共享
- 精确匹配集合交集平均重叠率 0.67

### 关键发现

1. **LLM 事实知识的规模远超预期**：GPT-4o-mini (~8B参数) 可提取 1.01 亿三元组，约 79 参数/三元组
2. **与传统 KB 高度互补**：63% 的实体在 Wikidata 中不存在，涵盖数字媒体、艺术流派、个人爱好等新领域
3. **存在显著的地理和文化偏见**：美国人 119K vs 中国人仅 3K，反映训练语料的英语中心倾向
4. **逆向关系不一致**：318K 配偶三元组中仅 8K 是对称的，61K 母公司三元组中仅 6K 有对应子公司三元组
5. **知识时效性清晰**：2023 年后频率急剧下降，与已知知识截止时间一致
6. **性别偏见有所改善**：性别属性中女性 15K vs 男性 8K，反映 LLM 去偏见的努力

## 亮点与洞察

- **范式创新**：从"即查即丢"转向"物化持久"，为 LLM 知识分析开辟了新范式
- **成本效率惊人**：每正确三元组 API 成本仅 $0.0001，比传统自动化 KB 构建低 100 倍以上
- **一资源多分析**：GPTKB 作为持久资源可同时支持规模、准确性、偏见、时效性、一致性等多维分析
- **递归图扩展避免可用性偏差**：不依赖预设问题集，能发现研究者未预料到的知识

## 局限与展望

- **提示依赖性**：不同提示会产生不同的 KB，当前结果只是 LLM 知识的下界
- **可复现性风险**：基于闭源 LLM，服务可能被中断
- **幻觉问题未完全解决**：特别是虚构角色类别中出现大量编造（如 Officer K.I.T.T. XV）
- **去重和规范化仍有大量空间**：如实体规范化、字面量类型化、关系子关系组织等
- **精度-召回率权衡**：长尾知识中幻觉难以界定
- **GPT-4o 完整运行预算约 82.5 万美元**，超出学术预算

## 相关工作与启发

- 继承了 DIPRE (Brin 1998) 和 Snowball 等经典迭代信息抽取的思想，但首次将其应用于 LLM 内部知识
- 与 Cohen et al. (2023) 的小规模提案相比，本文解决了实际工程挑战并实现了大规模构建
- 为 LLM 认识论提供了实证基础：关于 LLM "知道"什么的争论可以通过物化资源进行实证分析
- 传统 KB（Wikidata/YAGO/DBpedia）创新停滞，本文提供了全新的构建范式

## 评分

- **创新性**：★★★★★（提出全新的 LLM 知识物化范式）
- **实验充分性**：★★★★★（1.01 亿三元组的大规模构建与多维分析）
- **实用价值**：★★★★☆（GPTKB 可直接使用，但精度仍需提升）
- **写作质量**：★★★★★（结构清晰，贡献明确，讨论深入）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Pre³: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation](pre3_deterministic_pda_structured_gen.md)
- [\[ACL 2025\] Concreteness Versus Abstractness: A Selectivity Analysis in LLMs](concreteness_versus_abstractness_a_selectivity_analysis_in_llms.md)
- [\[ACL 2025\] Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](condor_enhance_llm_alignment_with_knowledge-driven_data_synthesis_and_refinement.md)
- [\[ACL 2025\] INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models](interact_enabling_interactive_question-driven_learning_in_large_language_models.md)
- [\[ACL 2025\] Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives](re-task_revisiting_llm_tasks_from_capability_skill_and_knowledge_perspectives.md)

</div>

<!-- RELATED:END -->
