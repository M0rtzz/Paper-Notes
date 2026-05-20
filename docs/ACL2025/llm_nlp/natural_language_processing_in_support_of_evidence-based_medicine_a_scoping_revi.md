---
title: >-
  [论文解读] Natural Language Processing in Support of Evidence-based Medicine: A Scoping Review
description: >-
  [ACL2025][LLM/NLP][evidence-based medicine] 基于 PRISMA 指南对 129 篇研究（2019-2024）进行范围综述，以 EBM 五步流程（Ask-Acquire-Appraise-Apply-Assess）为组织框架…
tags:
  - "ACL2025"
  - "LLM/NLP"
  - "evidence-based medicine"
  - "NLP survey"
  - "clinical NLP"
  - "PICO extraction"
  - "clinical trial matching"
  - "systematic review"
  - "LLM"
---

# Natural Language Processing in Support of Evidence-based Medicine: A Scoping Review

**会议**: ACL2025  
**arXiv**: [2505.22280](https://arxiv.org/abs/2505.22280)  
**代码**: [GitHub](https://github.com/bionlplab/awesome-nlp-in-ebm)  
**机构**: Weill Cornell Medicine, Columbia University, University of Sydney
**领域**: LLM/NLP  
**关键词**: evidence-based medicine, NLP survey, clinical NLP, PICO extraction, clinical trial matching, systematic review, LLM

## 一句话总结

基于 PRISMA 指南对 129 篇研究（2019-2024）进行范围综述，以 EBM 五步流程（Ask-Acquire-Appraise-Apply-Assess）为组织框架，全面梳理了 NLP 技术在循证医学中的应用现状、技术演进路径与未来方向。

## 研究背景与动机

**循证医学的核心地位**：EBM 是现代医疗的基石，通过整合最佳科学证据、临床专业知识和患者价值观来指导临床决策，服务于临床医生、患者、指南制定者和政策制定者。

**文献爆炸与人工瓶颈**：医学文献数量庞大且增长迅速，人工完成文献筛选、数据提取、质量评估和证据合成的成本极高，一项系统综述往往需要数月甚至数年，亟需 NLP 自动化工具辅助。

**NLP 技术演进驱动**：从早期统计机器学习和 RNN/LSTM 到 Transformer 预训练模型（BioBERT、PubMedBERT），再到 GPT-4 等大语言模型，NLP 在医学文本处理中的能力持续跃升，但现有综述未能全面反映这一技术演进对 EBM 的系统性影响。

**LLM 时代的双面性**：大模型在复杂任务（证据评估与合成、临床问答、患者-试验匹配）上展现出强大能力，但幻觉生成、缺乏来源归因和对提示词的敏感性等问题限制了其在高风险临床场景中的可靠应用。

**缺乏统一框架**：现有综述通常按 NLP 技术类型或疾病领域组织，缺乏以 EBM 实践流程为主线、跨任务统一映射的系统框架，使临床工作者难以找到针对特定 EBM 步骤的最佳工具。

**基准数据集缺口**：证据合成与评估、医学问答等关键领域缺乏专用基准数据集，现有摘要数据集（如 CNN-DailyMail）为通用领域而非医学领域，严重制约了方法的可比较性和进步。

## 方法详解

### 文献检索与筛选流程

遵循 PRISMA 指南，从 PubMed、IEEE Xplore、ACM Digital Library 和 ACL Anthology 四个数据库中检索 2019-2024 年的文献。检索策略采用 NLP 领域关键词（natural language processing, language model, LLM, information extraction 等）与 EBM 领域关键词（Evidence-Based Medicine, Clinical Trial 等）的布尔 AND 组合。筛选流程如下：

| 阶段 | 操作 | 数量变化 |
|------|------|----------|
| 初始检索 | 数据库检索 + 额外来源 | 601 + 9 = 610 篇 |
| 去重 | 移除重复文献 | -8 篇 |
| 标题/摘要筛选 | 依据排除标准初筛 | -386 篇 |
| 全文筛选 | 目标不匹配或与 EBM 无关 | -88 篇 |
| 最终纳入 | 符合纳入标准 | **129 篇** |

纳入标准：英文发表、明确应用 NLP 技术于 EBM、关注人类应用。排除标准：与 NLP+EBM 无关、非英文、系统综述/案例报告等二次文献或无实验结果的描述性论文。两位标注者交叉验证，分歧由第三人裁决。

### EBM 五步流程与 NLP 任务映射

本文的核心贡献在于将 NLP 任务系统映射到 EBM 的五步实践流程（5A 框架）。每个步骤对应的 NLP 任务、典型方法和代表系统如下：

| EBM 步骤 | 核心功能 | 对应 NLP 任务 | 代表性方法/系统 |
|----------|----------|--------------|----------------|
| **Ask** | 搜索与筛选研究 | 信息检索 | BioBERT 嵌入检索、Bing AI/ChatGPT 辅助搜索 |
| **Acquire** | 收集与结构化数据 | 实体提取(PICO)、关系提取 | SciBERT/PubMedBERT 实体识别、知识图谱关系编码 |
| **Appraise** | 评估证据质量 | 质量评估、证据排序筛选、证据合成、证据摘要 | BERT 微调质量分类、BM25+BERT 重排、RAG+GPT-4 |
| **Apply** | 临床应用 | 临床试验设计与匹配、专科应用 | AutoTrial、TrialGPT、Watson Oncology |
| **Assess** | 评估与改进 | 问答、药物再利用 | BioBERT/BioGPT 问答、知识图谱药物预测 |

### Ask 阶段：信息检索技术演进

该阶段目标是从海量文献库中检索与临床问题相关的研究。技术路线经历了三代演进：

- **第一代（关键词匹配）**：基于 MEDLINE/PubMed 的结构化关键词查询和规则过滤（如 SR[pt]、CQrs），精确但维护成本高、领域敏感。
- **第二代（上下文表征）**：统计 ML 和上下文感知模型改善了文本表征，但扩展性不足。Lokker 等（2023）使用 BioBERT 的嵌入和注意力机制提升了生物医学文献检索质量。
- **第三代（生成式 AI）**：Gwon 等（2024）对比了 Bing AI 和 ChatGPT 在系统性文献搜索中的加速效果，发现两者均能加速搜索流程，但存在幻觉问题。

### Acquire 阶段：实体提取与关系抽取

**PICO 实体提取**是 EBM 数据采集的核心任务：

- **规则方法**：利用预定义的词汇、句法和上下文规则提取临床试验数据中的实体，适合高精度结构化场景，但难以处理复杂或歧义数据。
- **RNN/LSTM**：用于序列句子分类，改善了非结构化摘要中的上下文利用，但长程依赖能力不足。
- **Transformer 主导**：SciBERT 和 PubMedBERT 专门用于干预（Intervention）提取；SrBERT 用于基于预定义标准的文献纳入/排除分类。

**关系提取**将已识别的 PICO 元素关联起来：从规则和传统 ML → 深度学习（BERT + Argument Mining 识别支持/反对关系）→ 模式化关系提取（C-TrO 本体指导）→ 知识图谱（节点表示实体，边表示关系，如 Pan 等 2021 用于 COVID-19 临床实体组织）。

### Appraise 阶段：质量评估、排序与合成

**质量评估**：Brassey 等（2021）开发全自动工具，结合 ML 和规则方法，通过情感分析、偏倚指示和样本量计算评估 RCT 和系统综述的证据质量。BioBERT、BlueBERT、BERTBASE 通过微调用于基于方法学质量标准的文献分类。

**证据排序与筛选**：Norman 等（2019b）开发优先级排序方法，使技术辅助筛选显著减少需筛选的文献数量。A2A 平台采用 BM25（基于词频和文档长度的评分）和 DFR（基于随机性偏离的信息量化）进行文档排序。SciBERT/BioBERT/BlueBERT 重排进一步提升鲁棒性。

**证据合成**：Mutinda 等（2022b）自动提取和归一化 PICO 元素，复现乳腺癌 RCT 的 meta 分析（计算风险比、生成森林图），但仅支持二元结局。EvidenceMap 通过实体-命题-地图三层结构增强证据的可解释性和可检索性。

**证据摘要**分为提取式（TextRank 优于 LexRank 和 LSA；KeBioSum 融合 PICO 知识到 PLM 中降低计算成本）和生成式（T5、BART、PEGASUS；TrialsSummarizer 支持用户交互式修正；RAG+GPT-4 回答临床试验资格问题；TriSum 通过结构化理由蒸馏到小模型）。

### Apply & Assess 阶段：临床应用与持续评估

**专科应用**：Watson Oncology（WOLI）自动识别和提取相关肿瘤学研究；CTM 系统在澳大利亚癌症中心以 92% 的总体准确率筛选肺癌患者。心脏病学中，混合模型在 4 万患者中达到 87.3% 的资格评估准确率。

**临床试验匹配**技术路线：混合方法（模式匹配+知识驱动+特征加权）→ 深度学习（CNN/RNN 多标签分类）→ C2Q 将自由文本标准转为结构化查询 → LLM 增强（AutoTrial 利用多步推理生成试验资格标准；TrialGPT 实现大规模患者-试验匹配）。

**药物再利用**：CovidX 网络算法利用 NLP 分析 COVID-19 文献排序候选药物；知识图谱嵌入用于预测阿尔茨海默病候选药物。

## 关键发现

### 代表性系统性能对比

| 系统/方法 | 任务 | 疾病/场景 | 关键性能指标 |
|-----------|------|-----------|-------------|
| CTM (Alexander 2020) | 临床试验匹配 | 肺癌 | 总体准确率 92% |
| 混合NLP+规则 (Tun 2023) | 患者资格评估 | 心衰/房颤 | 准确率 87.3%（40,000 患者） |
| BioBERT (Lokker 2023) | 高质量文献识别 | 通用 | 深度学习显著优于传统方法 |
| BM25+BERT重排 (Rybinski 2020b) | 临床试验搜索 | 通用 | 鲁棒性显著提升 |
| TrialGPT (Jin 2024) | 大规模患者-试验匹配 | 通用 | 真实部署，显著节省时间 |
| AutoTrial (Wang 2023b) | 试验资格标准生成 | 通用 | 多步推理+混合提示 |
| TextRank (Gulden 2019) | 提取式摘要 | 通用 | 优于 LexRank 和 LSA |
| TriSum (Jiang 2024) | 生成式摘要 | 通用 | 结构化理由蒸馏+课程学习 |
| RAG+GPT-4 (Unlu 2024) | 临床试验资格问答 | 心衰 | 基于检索增强生成 |

### 基准数据集现状

| 数据集 | 规模 | 标注类型 | 主要任务 |
|--------|------|----------|----------|
| EBM-NLP (Nye 2018) | 4,993 篇摘要 | P, I, O 手动标注 | PICO 提取 |
| Chia (Kury 2020) | 1,000 项试验/12,409 条标准 | 12 类手动标注 | 资格标准分析 |
| MS^2 (DeYoung 2021) | 470K 文档/20K 摘要 | 多类手动标注 | 多文档摘要 |
| PICO-Corpus (Mutinda 2022a) | 1,011 篇乳腺癌 RCT | P, I, C, O 手动标注 | PICO 提取 |
| NICTA-PIBOSO (Kim 2024) | 1,000 篇摘要 | P, I, O + 背景/设计 | 句子分类 |
| Trialstreamer (Marshall 2020) | 191 篇 RCT | P, I, O + RCT 分类器 | 证据流式处理 |
| LCT (Dobbins 2022) | 1,000+ 资格声明 | 6 类手动标注 | 临床试验解析 |

**关键缺口**：证据合成与评估、医学问答领域严重缺乏专用基准。现有摘要数据集（如 CNN-DailyMail）为通用领域而非医学内容。

## 亮点与洞察

1. **EBM 5A 框架映射**：首次以 EBM 五步实践流程为主线组织 NLP 综述，提供了从提出临床问题到评估实践效果的完整技术全景图，使临床工作者能快速定位各环节的最佳工具
2. **四代技术演进路径**：清晰呈现了规则方法 → 统计 ML → RNN/LSTM → Transformer → LLM 的技术迭代路径，每代的优势、局限和代表工作一目了然
3. **开源资源库**：公开了 awesome-nlp-in-ebm 资源库，包含论文列表、数据集索引和详细元数据，为后续研究提供了便利的入口
4. **129 篇论文的系统元数据**：附录提供了每篇论文的模型类型、疾病领域、任务分类的完整标注，构成了领域知识库

## 局限性

1. **语言偏差**：仅纳入英文论文，可能遗漏其他语言（尤其是中文、日文）的重要研究
2. **时间窗口限制**：覆盖 2019-2024，遗漏了奠基性早期工作
3. **数据库覆盖不全**：仅检索 4 个数据库，可能未捕获其他平台的相关研究
4. **综述性质局限**：无法对不同方法在统一条件下进行直接实验对比
5. **专科覆盖不均**：泌尿科、肝病科等领域的 NLP 应用尚未充分覆盖
6. **LLM 研究的时滞**：2023-2024 年 LLM 论文爆发，但受发表周期影响，综述可能未完全覆盖最新进展

## 相关工作与启发

- **与其他综述的差异化**：已有综述多以 NLP 技术类型（如 NER、QA）或疾病领域（如肿瘤学）为组织维度，本文创新性地以 EBM 5A 实践流程为主线，更贴近临床工作者的使用场景
- **未来研究方向**：（1）整合真实世界数据（移动设备、社交媒体、基因组学），捕获传统临床记录之外的健康指标；（2）少样本学习应对罕见病数据稀缺问题；（3）构建证据合成与评估专用基准数据集；（4）提升 NLP 模型的可解释性和跨人群公平性；（5）发展能跨多种临床问题和异质数据源推理的全面自适应框架
- **对 NLP 研究的启发**：EBM 是一个天然的 NLP 技术集成应用场景，从信息抽取、信息检索、问答到文本生成和推理涵盖 NLP 全栈能力，可作为评估通用 NLP 系统的综合测试床

## 评分

- 新颖性: ⭐⭐⭐ (综述类论文，5A框架映射有组织创新但无方法创新)
- 实验充分度: ⭐⭐⭐⭐ (129篇论文系统纳入、严格PRISMA流程、4数据库检索、双标注者验证)
- 写作质量: ⭐⭐⭐⭐⭐ (5A框架组织清晰、覆盖全面、附录元数据丰富、图表质量高)
- 价值: ⭐⭐⭐⭐ (为NLP+EBM交叉领域提供了最全面的全景索引和资源入口)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Internal and External Impacts of Natural Language Processing Papers](internal_and_external_impacts_of_natural_language_processing_papers.md)
- [\[ACL 2025\] What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)
- [\[ACL 2025\] LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models](llm_mapreduce_simplified_long_sequence_processing.md)
- [\[ACL 2025\] On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs](on_the_risk_of_evidence_pollution_for_malicious_social_text_detection_in_the_era.md)

</div>

<!-- RELATED:END -->
