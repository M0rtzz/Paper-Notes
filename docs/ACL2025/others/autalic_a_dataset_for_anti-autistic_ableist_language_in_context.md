---
title: >-
  [论文解读] Autalic: A Dataset for Anti-Autistic Ableist Language In Context
description: >-
  [ACL 2025][反自闭症歧视] 提出 Autalic——首个专注于上下文中反自闭症残障歧视语言检测的数据集，包含 2,400 条 Reddit 句子及上下文标注，由神经多样性背景的专家标注，实验揭示当前 LLM（包括 DeepSeek、Llama3、Gemma2、Mistral）在识别反自闭症歧视语言时与人类判断严重不一致（平均 Cohen's Kappa 仅 0.091），凸显该任务的困难性。
tags:
  - "ACL 2025"
  - "反自闭症歧视"
  - "残障歧视语言"
  - "上下文标注"
  - "神经多样性"
  - "Reddit数据集"
  - "LLM偏见"
---

# Autalic: A Dataset for Anti-Autistic Ableist Language In Context

**会议**: ACL 2025  
**arXiv**: [2410.16520](https://arxiv.org/abs/2410.16520)  
**代码**: [https://nrizvi.github.io/AUTALIC.html](https://nrizvi.github.io/AUTALIC.html)  
**作者**: Naba Rizvi, Harper Strickland, Daniel Gitelman, Tristan Cooper, Alexis Morales-Flores, Michael Golden, Aekta Kallepalli, Akshat Alurkar, Haaset Owens, Saleha Ahmedi, Isha Khirwadkar, Imani Munyaka, Nedjma Ousidhoum
**机构**: UC San Diego, Cardiff University
**领域**: 社会偏见与公平性 / 仇恨言论检测  
**关键词**: 反自闭症歧视, 残障歧视语言, 上下文标注, 神经多样性, Reddit数据集, LLM偏见

## 一句话总结

提出 Autalic——首个专注于上下文中反自闭症残障歧视语言检测的数据集，包含 2,400 条 Reddit 句子及上下文标注，由神经多样性背景的专家标注，实验揭示当前 LLM（包括 DeepSeek、Llama3、Gemma2、Mistral）在识别反自闭症歧视语言时与人类判断严重不一致（平均 Cohen's Kappa 仅 0.091），凸显该任务的困难性。

## 研究背景与动机

**领域现状**：自闭症的医学模型将其定义为"疾病"和"缺陷"，这种框架在技术研究中被广泛使用，但与推崇所有神经类型均为人类多样性有效形式的"神经多样性"理念相冲突。反自闭症歧视语言因其微妙性和上下文依赖性，对 NLP 研究构成重大挑战。

**现有方法的不足**：
   - 现有毒性检测数据集多关注仇恨言论和冒犯语言，但几乎没有专门针对自闭症群体的数据集
   - 23 个 LLM 偏见评估数据集中仅 3 个涉及残障，且没有针对自闭症
   - 毒性分类器存在强烈的残障负面偏见——倾向于将任何与残障相关的文本标记为有毒
   - LLM 被发现会隐含地传播残障歧视刻板印象

**核心动机**：构建以自闭症人群视角为中心的标注数据集，并评估当前 NLP 工具在该任务上的能力

## 方法详解

### 整体框架

数据收集（Reddit）→ 标注员培训 → 专家标注 → 基线评估（传统模型 + LLM）

### 数据收集

1. **数据源**：Reddit（文本导向、API 限制较少）
2. **搜索关键词**：包括 "autis*", "ASD", "aspergers", "disabilit*" 等
3. **收集策略**：对每个目标句子，同时收集前后上下文句子
4. **最终规模**：2,400 条目标句子 + 2,014 条前文 + 2,400 条后文
5. **来源**：192 个不同子版块，主要来源包括 r/Aspergers (116)、r/Autism (88)、r/AmITheAsshole (39) 等
6. **数据清洗**：精确词搜索过滤缩写歧义、排除非英文帖子和含媒体帖子

### 标注设计

#### 标注员选拔与培训
- 招募 9 名高年级本科生志愿者，分 3 组（每组 3 人），每组标注 800 条
- 标注员背景多元：至少 3 名自我认同为神经发散，4 名性别少数群体
- 提供全面培训：自闭症歧视的历史（包括纳粹优生学）→ 医学模型的局限 → 神经多样性理念 → 当代歧视实例 → 标注示例讲解
- 提供术语表作为动态参考资源

#### 标注标签

| 标签 | 含义 | 数量 |
|------|------|------|
| 1 (Ableist) | 包含反自闭症歧视情感 | 1,023 |
| 0 (Not Ableist) | 正面/中性/社区内讨论 | 5,582 |
| -1 (Needs More Context) | 无法确定分类 | 595 |

**注**：标注员标注目标句子时可参考上下文来判断意图（如是否为社区内讨论、反讽等）

#### 标注结果
- 多数投票确定最终标签：242 条 (10%) 歧视, 2,160 条 (90%) 不歧视
- Fleiss' Kappa = 0.25（低一致性凸显任务困难性）
- 完成时间与一致性呈显著负相关 (R=-0.644, p=0.0096)——培训后立即标注的人一致性更高

### 上下文的重要性

论文通过详细案例说明上下文对判断的关键作用：
- 例如 "it's good that at least there's no link between the two" 单独看模糊不清
- 加入上下文后发现作者在讨论疫苗-自闭症关联的虚假说法（反自闭症污名化）
- 标注员可随着认知更新修改之前的标注

## 实验

### 实验设置

- **传统基线**：Logistic Regression (BoW)、BERT (预训练 + 微调)
- **LLM**：Gemma2, Mistral, Llama3, DeepSeek（均 < 10B 参数）
- **提示类型**：三种用语——PFL (person-first: "people with autism")、IFL (identity-first: "autistic people")、AA (概念性: "anti-autistic")
- **提示方式**：简单零样本 vs. ICL（从标注培训中提取示例）

### 主实验结果

| 模型 | 模式 | PFL F1 | IFL F1 | AA F1 |
|------|------|--------|--------|-------|
| LR (BoW) | 预训练 | 0.20 | — | — |
| BERT | 预训练 | 0.43 | — | — |
| BERT | 微调 | **0.90** | — | — |
| Gemma2 | 零样本 | 0.23 | 0.19 | 0.33 |
| Mistral | 零样本 | 0.28 | 0.27 | 0.34 |
| Llama3 | 零样本 | 0.09 | 0.10 | 0.15 |
| DeepSeek | 零样本 | **0.58** | **0.57** | **0.59** |
| Gemma2 | ICL | 0.25 | 0.24 | 0.34 |
| Mistral | ICL | 0.31 | 0.24 | 0.34 |
| Llama3 | ICL | 0.14 | 0.14 | 0.11 |
| DeepSeek | ICL | 0.55 | 0.56 | 0.55 |

### 关键发现

1. **LLM 与人类判断严重不一致**：所有 LLM 的 Cohen's Kappa 平均仅 0.091 (SD=0.110)，远低于可靠水平
2. **DeepSeek 最佳但仍不可靠**：DeepSeek 表现最好且最一致（不受用语变化影响），但与人类一致性仍仅约 0.11
3. **用语敏感性**：
    - Llama3 从 PFL→AA 的 F1 变化高达 **67.49%**，说明模型不理解不同描述指向同一现象
    - ICL 后一致性改善（如 Llama3 从 67.49% 降至 17.40%），但绝对性能仍低
4. **ICL 效果混合**：Llama3 (+22.96%) 和 Gemma2 (+12.68%) 提升较大，但 DeepSeek 略微下降
5. **微调 BERT 显著优于所有 LLM**（F1=0.90），但初始存在高假阳性率问题

### 错误分析

分析人类标注一致但 LLM 不一致的 top 10% 句子，发现 LLM 严重**过度分类**：
- Llama3 将 42 条句子标记为歧视，而人类标注全部为 0
- 这 42 条中 **29 条是社区内讨论**——如果用 LLM 做内容审核，会导致严重的社区审查
- 34 条包含负面含义词（"burden", "threat"等），但并非在反自闭症语境中使用
- 例：某句子引用了一个组织的观点，作者明确表示不同意该观点——标注员正确判断为"不歧视"，但 LLM 仅因负面词汇存在而错误分类

### 不一致性分析

100 个高不一致句子中观察到：
1. 48 条使用了医学模型术语或刻板印象（术语本身存争议）
2. 其余需要超出所提供上下文的额外信息

## 亮点与洞察

1. **首个数据集**：Autalic 是首个专门针对反自闭症歧视语言的标注数据集，填补了 NLP 公平性研究的重要空白
2. **以自闭症群体视角为中心**：标注员包含神经发散者，培训中包含医学模型批判和神经多样性教育——与主流数据集采用的"距离化"标注范式不同
3. **上下文的关键性被量化**：论文通过案例和统计分析清楚表明，脱离上下文的分类必然导致大量误判
4. **LLM 用于内容审核的风险**：LLM 的过度分类倾向会压制自闭症社区内部讨论——这对内容审核策略有重要警示
5. **保留个体标注**：公开全部标注（而非仅聚合标签），支持分歧分析研究
6. **标注培训与完成时间的关系**：量化证据表明及时标注（培训后立即执行）对一致性有重要影响

## 局限性

1. **数据选择偏差**：依赖关键词搜索和特定社交媒体线程，可能遗漏隐性歧视表达
2. **西方中心视角**：仅反映西方英语语境下的反自闭症歧视，不同文化中的歧视表现可能完全不同
3. **数据规模相对较小**（2,400 条），限制了深度学习模型的训练
4. **计算资源限制**：无法微调 LLM，仅使用 < 10B 参数的开源模型，无法评估更大模型
5. **搜索词"r*tard"适用范围更广**，可能引入与自闭症不直接相关的内容
6. 数据主要来自 2023 年，可能不反映更早或更新的语言演变

## 相关工作

- **仇恨言论检测**：Waseem & Hovy (2016) → Founta (2018) → 但大多忽视残障维度
- **残障偏见**：毒性检测器对残障内容存在负面偏见（Narayanan Venkit et al., 2023）→ LLM 传播隐性歧视（Gadiraju et al., 2023）→ 仅 3/23 偏见数据集涉及残障
- **标注分歧研究**：Plank et al. (2014) → Pavlick & Kwiatkowski (2019) → Leonardelli et al. (2021) → Autalic 保留全部标注支持分歧研究
- **自闭症与 AI**：主流研究多采用医学模型（诊断/治疗）→ Bottema-Beutel et al. (2021) 批判缺陷框架 → Autalic 转向神经多样性视角

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — 首个反自闭症歧视语言数据集，研究空白的重要填补
- **实验充分性**: ⭐⭐⭐⭐ — 4 个 LLM + 传统基线 × 3 种提示 × 零样本/ICL 的全面组合
- **写作质量**: ⭐⭐⭐⭐⭐ — 伦理考量详尽，背景阐述深入，案例分析直观
- **实用价值**: ⭐⭐⭐⭐ — 数据集公开，对内容审核策略有重要警示
- **局限**: 数据规模小、仅英语、无法微调 LLM 进行更深入对比

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Inducing Lexicons of In-Group Language with Socio-Temporal Context](inducing_lexicons_of_in-group_language_with_socio-temporal_context.md)
- [\[ACL 2025\] ConECT Dataset: Overcoming Data Scarcity in Context-Aware E-Commerce MT](conect_dataset_overcoming_data_scarcity_in_context-aware_e-commerce_mt.md)
- [\[ACL 2025\] QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning](qualispeech_a_speech_quality_assessment_dataset_with_natural_language_reasoning_.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [\[ACL 2025\] Task-Informed Anti-Curriculum by Masking Improves Downstream Performance on Text](task-informed_anti-curriculum_by_masking_improves_downstream_performance_on_text.md)

</div>

<!-- RELATED:END -->
