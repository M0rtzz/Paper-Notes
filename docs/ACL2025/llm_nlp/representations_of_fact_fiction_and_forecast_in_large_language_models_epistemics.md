---
title: >-
  [论文解读] Representations of Fact, Fiction and Forecast in Large Language Models: Epistemics and Attitudes
description: >-
  [ACL 2025][LLM/NLP][epistemic modality] 通过受控故事任务评估 8 个开源 LLM 对认识情态（may/must、know/believe/doubt）的语义知识，发现 LLM 在生成恰当认知表达方面表现有限且不鲁棒——必然性（must）优于可能性（may），事实陈述优于信念陈述。
tags:
  - ACL 2025
  - LLM/NLP
  - epistemic modality
  - modal auxiliaries
  - attitude verbs
  - uncertainty expression
  - Theory of Mind
  - linguistic knowledge
---

# Representations of Fact, Fiction and Forecast in Large Language Models: Epistemics and Attitudes

**会议**: ACL 2025  
**arXiv**: [2506.01512](https://arxiv.org/abs/2506.01512)  
**代码**: https://github.com/limengnlp/llm-fff  
**领域**: LLM/NLP  
**关键词**: epistemic modality, modal auxiliaries, attitude verbs, uncertainty expression, Theory of Mind, linguistic knowledge

## 一句话总结
通过受控故事任务评估 8 个开源 LLM 对认识情态（may/must、know/believe/doubt）的语义知识，发现 LLM 在生成恰当认知表达方面表现有限且不鲁棒——必然性（must）优于可能性（may），事实陈述优于信念陈述。

## 研究背景与动机

**领域现状**：LLM 越来越多地部署在需要表达不确定性的真实场景中，大量工作致力于 LLM 的置信度估计和校准（verbalized uncertainty、logit-based 等方法）。

**现有痛点**：现有工作默认 LLM 已掌握不确定性表达的语言知识，只需解决"何时触发"的对齐问题；但这一假设**从未被系统验证**——LLM 是否真正理解 may/must/know/believe/doubt 的语义差异？

**核心矛盾**：如果 LLM 连认识情态词的语义知识都不充分，那么基于 verbalized uncertainty 的校准方法的可靠性就存在根本质疑。

**本文目标** 从语言类型学（typological）框架出发，系统评估 LLM 对认识情态表达的语义知识。

**切入角度**：受儿童语言发展研究（hidden object task）和语言田野调查（targeted storyboards）启发，设计受控故事任务——通过简化推理复杂度、控制证据类型和确定性程度，观察 LLM 能否选择恰当的情态表达。

**核心 idea**：LLM 对认识情态的语义知识不足是其无法可靠表达不确定性的潜在根源，建构不确定性感知的 LLM 需要增强情态语义表示。

## 方法详解

### 整体框架
基于 Boye (2012) 的认识情态类型学框架（evidence / commitment），设计两个实验：实验1 测试情态助动词（may/might vs must/have to），实验2 测试态度动词（know/believe/doubt）。使用模板+手工编写生成受控故事，在 8 个开源指令微调 LLM 上用贪婪解码测试，用逻辑回归分析参数量、情态条件、提示格式和故事类型对准确率的影响。

### 关键设计 1：情态助动词实验（Experiment 1）
- **功能**：生成 150 个受控故事（5 个模板 × 3 种故事类型），每个故事包含必然性条件（N-modal: must/have to）和可能性条件（P-modal: may/might），测试 LLM 能否根据证据充分性选择正确的情态动词。
- **核心思路**：Hidden object 范式——玩具藏在三个盒子之一，排除两个盒子后"must be in the blue box"（必然），排除一个后"may be in the blue box"（可能）。设计 3 种 QA 格式（direct slot / indirect slot / indirect sentence）检验鲁棒性，3 种故事类型（base / 1-shot / counterfactual）测试对额外信息的敏感度。1-shot 通过添加解题示例帮助模型；counterfactual 通过改变已验证条件引入反事实推理。
- **设计动机**：借鉴发展心理学实验范式，将推理复杂度降到最低，只测语义知识本身。
  如果知识足够鲁棒，不同 QA 格式和故事类型间性能应一致。
  paired accuracy 要求同一故事的 N/P 两个条件都答对，比单项准确率更严格。
  为防止 LLM 简单复制 1-shot 答案，使用了词汇变体（不同颜色、人名等）。

### 关键设计 2：态度动词实验（Experiment 2）
- **功能**：从 ToMi 数据集选取 30 个 Theory-of-Mind 故事，
  构造 8 个陈述（2×前事实 + 2×当前事实 + 2×agent0 的信念 + 2×agent1 的信念），
  每对陈述在低/非低确定性间对比，测试 LLM 能否为事实选 know、为信念选 believe/doubt。
- **核心思路**：利用 ToM 场景中的信息不对称——agent0 不知道 agent1 移动了物体，因此 agent0 对位置的陈述属于"信念"而非"事实"。LLM 需要区分事实陈述（用 know）和信念陈述（用 believe/doubt），并根据确定性程度选择合适动词。
- **设计动机**：态度动词是人类认知元能力（metacognition）的语言表达，其正确使用要求理解不同主体的心理状态差异。Joint accuracy（同一故事 8 个陈述全对）提供了最严格的评测标准。

### 关键设计 3：统计分析框架
- **功能**：对每类 LLM（Qwen2/2.5、Llama3/3.1）分别拟合逻辑回归模型，系统化评估参数量、情态条件、陈述类型、QA 格式等因素对准确率的调节效应。
- **核心思路**：不止看平均准确率，而是通过回归系数量化各因素的效应方向和显著性，揭示系统性模式（如"必然性一致优于可能性"不是随机波动，而是跨模型的显著效应）。
- **设计动机**：行为实验范式要求严格的统计推断，通过拟合回归模型避免被描述性统计量误导，同时能发现因素间的交互效应。

## 实验关键数据

### 主实验 — 实验1 情态助动词准确率

| 模型 | 参数量 | 准确率 | Paired 准确率 |
|------|--------|--------|-------------|
| Qwen2-7B | 7B | 51.2% | 2.4% |
| Qwen2.5-7B | 7B | 72.3% | 45.1% |
| Llama3-8B | 8B | 55.1% | 11.8% |
| Llama3.1-8B | 8B | 55.2% | 11.8% |
| Qwen2-72B | 72B | 82.3% | 64.7% |
| Qwen2.5-72B | 72B | **95.3%** | **90.7%** |
| Llama3-70B | 70B | 79.6% | 59.1% |
| Llama3.1-70B | 70B | 91.1% | 82.2% |

### 实验2 态度动词准确率

| 模型 | 参数量 | 准确率 | Paired 准确率 | Joint 准确率 |
|------|--------|--------|-------------|-------------|
| Qwen2-7B | 7B | 56.9% | 38.6% | 0% |
| Qwen2.5-72B | 72B | 60.8% | 46.4% | 5.6% |
| Llama3.1-70B | 70B | **72.8%** | **54.2%** | 0% |
| Llama3-70B | 70B | 62.1% | 40.3% | 0% |

### 关键发现
- **必然性 > 可能性**：跨所有模型一致，LLM 在"必须"（唯一结论）场景下的准确率显著高于"可能"（多种可能）场景——模型更擅长处理无歧义情境
- **参数量有显著作用**：70B+ 模型（79.6-95.3%）大幅优于 7-8B 模型（51.2-72.3%），回归系数均显著（p<.001）
- **事实陈述 >> 信念陈述**：实验2中，LLM 对事实性陈述的准确率系统性高于信念性陈述（回归效应 b=2.74~4.79，p<.001）
- **Joint 准确率极低**：即使最好的 72B 模型，在 ToM 任务上也只有 5.6% 的 joint 准确率——说明 LLM 对同一故事中不同态度动词的选择极不一致
- **QA 格式影响小但不一致**：不同提示格式的效应在不同模型间方向不一致，说明情态知识不够鲁棒
- **Llama3/3.1-70B 出现反转现象**：中等参数量的 Llama 模型在低确定性（doubt）上反而准确率更高，与 Qwen 系列趋势相反，可能与指令微调数据分布有关
- **1-shot 示例未带来一致改善**：与预期不同，增加 in-context 示例并未显著提升所有模型的表现，进一步证明问题在于语义知识本身而非任务理解
- **反事实故事增加难度**：包含反事实条件的故事并未像预期那样显著降低准确率，说明模型对反事实语境的处理能力超出预期，但效应在不同模型间不一致
- **小模型的 paired 准确率极低**：Qwen2-7B 仅 2.4% 的 paired 准确率，说明 7B 模型几乎无法同时正确处理必然性和可能性场景
- **Qwen2.5 系列进步显著**：Qwen2.5-72B (95.3% 准确率) 相比 Qwen2-72B (82.3%) 提升明显，说明训练数据和策略的迭代对情态知识有正向作用

## 亮点与洞察
- **首次从语言类型学角度系统评估 LLM 的情态语义知识**——将"LLM 能否可靠表达不确定性"的问题从对齐层面下沉到语义知识层面，提出了一个被忽视的根本瓶颈。
- **受控实验设计精巧**：借鉴儿童语言发展心理学的 hidden object task 和 ToM 测试范式，推理复杂度降到最低，隔离出纯粹的语义知识维度。
- **Paired/Joint 准确率指标**揭示了单项准确率掩盖的真实问题——看似不错的 72.8% 准确率，在要求"全对"时降到 0%。
- **人机对比视角有启发性**：4-5 岁儿童已具备基本的认识情态理解，成人达到 97-100%，而 LLM 与人类成人仍有显著差距。
- **将语言不确定性与认知不确定性（aleatoric/epistemic）解耦**：明确指出 verbalized uncertainty 的可靠性取决于语言知识本身，为该领域提供了清晰的问题分层。
- **跨模型一致性分析**：通过逻辑回归系统化地展示了哪些效应在所有模型间一致（参数量、必然性>可能性），哪些不一致（QA格式），为后续研究提供了可操作的基准线。

## 局限性
- 仅评估了英语，未涉及低资源语言中情态的形态学编码（如词缀、格标记等）
- 未直接测试人类参与者作为对照组，依赖文献中的儿童/成人数据
- 未利用 logit 概率设计内在不确定性指标
- 实验排除了否定对情态语义的影响（如 "not certain" vs "certain that not"），简化了问题
- 仅测试 8 个开源模型，未涵盖闭源 LLM（GPT-4、Claude 等）
- 故事模板数量有限（5 个模板 × 30 个故事），结果的泛化性有待更大规模验证
- 未探讨微调或 RLHF 能否改善情态知识，仅评估现有模型的零样本能力
- 未考察情态表达在多轮对话中的一致性（实际场景中 LLM 需要在多次交互中保持语义一致性）
- 只使用贪婪解码，未探索采样策略对情态词选择的影响
- 仅关注英语中三类认知情态形式，未涉及其他语言中的情态标记（如日语的推量助词、土耳其语的情态后缀等）

## 相关工作与启发
- **vs Xiong et al. (2024) verbalized uncertainty**：Xiong 等假设 LLM 已掌握不确定性表达能力，只需校准"何时说"；本文质疑这一前提，证明 LLM 对情态表达的语义知识本身就不充分——校准之前先要解决"会不会说"
- **vs Holliday et al. (2024) 条件句+情态推理**：Holliday 测试 LLM 在条件句和情态推理中的逻辑谬误，聚焦推理能力；本文刻意最小化推理复杂度，聚焦语义知识本身——互补视角
- **vs Zhou et al. (2024) LLM 不确定性表达行为**：Zhou 发现 LLM 在给出错误答案时不愿表达不确定性；本文揭示了更底层的原因——可能不是"不愿说"，而是"不知道该说什么"
- **vs Sileo & Moens (2023) 概率词理解**：Sileo 发现神经语言模型在理解概率副词/形容词方面有困难，本文扩展到情态助动词和态度动词，覆盖更完整的认知情态表达谱系
- **启发**：本文提出的儿童语义发展视角（从互动+语用信息丰富的环境中学习态度动词）为 LLM 训练数据构建提供了新思路——当前预训练语料中情态语义的标注可能不够充分，可考虑构造针对性 SFT 数据增强情态表达能力。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从类型学角度评估 LLM 情态语义知识，实验设计创新性强，填补了研究空白
- 实验充分度: ⭐⭐⭐⭐ 8 个模型、严格统计分析、多条件控制，但缺少人类对照和闭源模型
- 写作质量: ⭐⭐⭐⭐⭐ 语言学理论与 NLP 实验结合紧密，论述清晰有深度，跨学科视角精彩
- 价值: ⭐⭐⭐⭐ 揭示 LLM 不确定性表达的底层瓶颈，对 uncertainty-aware LLM 研究有方向性启发，为服化不确定性校准提供了新视角

<!-- RELATED:START -->

## 相关论文

- [Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts](cross_model_transferability_sv.md)
- [Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](language_codec_bridging_discrete_codec_speech_language_models.md)
- [Exploring Graph Representations of Logical Forms for Language Modeling](exploring_graph_representations_of_logical_forms_for_language_modeling.md)
- [Theory of Mind in Large Language Models: Assessment and Enhancement](theory_of_mind_llm.md)
- [Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)

<!-- RELATED:END -->
