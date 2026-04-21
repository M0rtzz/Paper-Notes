---
title: >-
  [论文解读] CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming
description: >-
  [NeurIPS 2025][AI安全][competitive programming retrieval] 针对竞赛编程中重复/相似题目泛滥导致比赛不公平及 LLM 评测分数虚高的问题，构建了包含四种检索任务的大规模基准 CPRet，并提出 Group-InfoNCE 损失训练的专用检索模型 CPRetriever，在所有任务上超越 20+ 现有嵌入模型，同时揭示了题目相似性对 LiveCodeBench 评测的系统性偏差。
tags:
  - NeurIPS 2025
  - AI安全
  - competitive programming retrieval
  - duplicate problem detection
  - embedding model
  - Group-InfoNCE
  - data contamination
  - benchmark
---

# CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming

**会议**: NeurIPS 2025  
**arXiv**: [2505.12925](https://arxiv.org/abs/2505.12925)  
**代码**: https://github.com/coldchair/CPRet  
**领域**: ai_safety  
**关键词**: competitive programming retrieval, duplicate problem detection, embedding model, Group-InfoNCE, data contamination, benchmark

## 一句话总结

针对竞赛编程中重复/相似题目泛滥导致比赛不公平及 LLM 评测分数虚高的问题，构建了包含四种检索任务的大规模基准 CPRet，并提出 Group-InfoNCE 损失训练的专用检索模型 CPRetriever，在所有任务上超越 20+ 现有嵌入模型，同时揭示了题目相似性对 LiveCodeBench 评测的系统性偏差。

## 研究背景与动机

**领域现状**：竞赛编程（ICPC、IOI 等）已成为评测 LLM 代码生成与推理能力的标准基准（LiveCodeBench、HumanEval 等），每年新增数千道题目。

**重复题目泛滥**：大量重复或高度相似的题目在各 OJ 平台间累积，社区讨论中关于"重题"的帖子近年显著增加。这带来双重危害：(a) 让见过原题的参赛者获得不公平优势；(b) 导致 LLM 评测分数虚高——模型可能依靠记忆而非真正推理。

**检索任务缺口**：现有代码检索基准（如 CoIR）仅覆盖 Text-to-Code 和 Code-to-Code 两个维度，缺乏**题目级别**的相似检索任务，也缺少相应的训练数据和评测标准。

**时间泄露问题**：对 APPS 数据集按时间分组分析发现，2022 年前的老题上模型表现远高于新题，说明训练数据泄露严重影响了评测公正性。

**本文目标**：(a) 定义四种检索任务并构建大规模基准；(b) 训练竞赛编程专用检索模型；(c) 量化题目相似性对 LLM 评测的系统性影响。

**核心 idea**：用 Group-InfoNCE 损失对齐题目与其多种正确解法，再微调为题目级检索器，同时为竞赛编程检索建立首个全面基准。

## 方法详解

### 四种检索任务定义

作者定义了涵盖代码和题目两个维度的四种检索任务：

- **Text-to-Code (T2C)**：给定题目自然语言描述，检索正确的代码解法。评估模型对题目语义与代码实现的对齐能力。训练集 38.8K 题目 / 2.93M 代码对，测试集 4.9K query / 41.6K corpus。
- **Code-to-Code (C2C)**：给定一个正确解法，检索同一题目的其他正确解法。评估模型理解代码功能和识别语法不同但语义等价实现的能力。测试集 4.8K query / 39.8K corpus。
- **Problem-to-Duplicate (P2Dup)**（新提出）：给定题目描述，在 10,900+ 题目的语料库中检索重复或高度相似的题目。训练 491 对，测试 168 query / 202 相关对。
- **Simplified-to-Full (S2Full)**（新提出）：给定题目的简化/改写版本，检索对应的原始完整描述。训练 7.6K 对，测试 10K query / 10K corpus。

### 数据集构建（CPRet-PCPCD）

数据集规模达 42.2K 题目 + 2.9M 代码，来自 12 个 OJ 平台，覆盖 3 种语言题面（英/中/日）和 20+ 编程语言，同时包含 ICPC 风格（全分制）和 OI 风格（部分分），时间截止 2024.12。相比现有数据集大幅领先：

| 数据集 | 题目数 | 代码数 | 截止时间 |
|---|---|---|---|
| Description2Code | 7.8K | 309K | 2016/08 |
| APPS | 10K | 232K | 2020/10 |
| CodeContests | 13.6K | 4.5M | 2021/07 |
| TACO | 26.4K | 1.55M | 2023/02 |
| **CPRet-PCPCD** | **42.2K** | **2.9M** | **2024/12** |

关键数据采集流程：
- **重复题目对**：爬取 Codeforces 和 Luogu 全部公开讨论帖和博客，通过关键词启发式 + LLM 分类筛选出 ~5000 条候选，由多名经验丰富的竞赛选手人工标注。定义三级重复标准：Exact Match（代码直接互过）、Near Match（小修改即可互过）、Method Match（核心方法相同但实现细节不同）。最终标注 ~700 对，聚类后随机选 30% 的簇构建测试集。
- **简化题目对**：爬取 Luogu 上用户贡献的中文翻译/简化版本（原题来自 Codeforces、AtCoder 等），过滤低质量机翻后得到 ~17K 对。
- **时间分割**：代码检索任务的测试集严格使用 2023 年之后的题目，避免训练数据泄露。

### Group-InfoNCE 损失

**动机**：竞赛编程中同一题目通常有多种正确解法（不同算法、不同编程语言），标准 InfoNCE 只处理单正样本对，Multi-Pos 扩展简单平均正样本相似度但忽略组内一致性。

**核心设计**：将同一题目的所有正确解法视为一个"正样本组" $G_i = \{x_i^{1+}, \ldots, x_i^{m+}\}$，定义组相似度：

$$\mathrm{sim}_G(x_i, G_j) = \frac{1}{m}\sum_{k=1}^{m}\mathrm{sim}(x_i, x_j^{k+})$$

完整损失函数包含对比项、组对比项和方差正则化三部分：

$$\mathcal{L}_{\text{Group}} = -\log\frac{\exp(\mathrm{sim}_G(x_i,G_i)/\tau)}{\exp(\mathrm{sim}_G(x_i,G_i)/\tau) + \sum_{j\neq i}[\exp(\mathrm{sim}(x_i,x_j)/\tau) + \exp(\mathrm{sim}_G(x_i,G_j)/\tau)]} + \frac{\text{Penalty}_G(x_i,G_i)}{\tau^2}$$

其中方差正则化 $\text{Penalty}_G = \mathrm{Var}_{k=1}^{m}(\mathrm{sim}(x_i, x_i^{k+}))$ 确保同题各解法与题目嵌入保持一致距离。关键洞察：方差惩罚约束的是相似度的一致性而非解法嵌入本身——这允许不同算法策略在嵌入空间中保持多样性，同时都"环绕"在题目嵌入周围。

### Format Masking

训练时随机遮盖题目中的 I/O 格式说明、样例输入输出、数据范围约束等非核心部分，迫使模型关注算法语义而非格式线索（如样例输入的数值模式），提升对不同题目格式的泛化能力。

### CPRetriever-Prob 微调

在 CPRetriever-Code 基础上，使用三元组损失在 P2Dup 和 S2Full 训练数据上联合微调：

$$\mathcal{L}_{\text{triplet}} = \max(0, \mathrm{sim}(x, x^-) - \mathrm{sim}(x, x^+) + \alpha)$$

硬负样本挖掘策略：先用 SFR-Embedding-Code-2B 预检索每个 query 的 top-10 相似题目作为候选负样本，再用 Qwen-2.5-Max 自动验证排除假负例（实际上是重复题目的样本），确保负样本可靠性，无需额外人工标注。

## 实验关键数据

### 主实验（NDCG@10，覆盖 Tiny/Small/Medium/Large 四个规模档）

| 模型 (规模) | T2C | C2C | P2Dup | S2Full | Avg |
|---|---|---|---|---|---|
| gte-modernbert-base (149M) | 14.99 | 36.22 | 21.12 | 77.45 | 37.44 |
| SFR-Emb-Code-400M (400M) | 9.43 | 43.59 | 19.40 | 75.31 | 36.93 |
| SFR-Emb-Code-2B (2B) | 39.60 | 68.05 | 45.26 | 86.43 | 59.84 |
| Qodo-Embed-1-7B (7B) | 36.47 | 51.91 | 47.15 | 91.17 | 56.68 |
| Qwen3-Embedding-0.6B (600M) | 48.96 | 60.49 | 36.26 | 81.63 | 56.83 |
| Qwen3-Embedding-4B (4B) | 66.62 | 71.97 | 56.59 | 89.39 | 71.15 |
| Qwen3-Embedding-8B (8B) | 60.54 | 72.97 | 53.23 | 87.95 | 68.67 |
| **CPRetriever-Code (2B)** | **70.40** | 70.59 | 38.68 | 81.45 | 65.28 |
| **CPRetriever-Prob (2B)** | 56.50 | 70.68 | 60.06 | 90.74 | 69.50 |
| **CPRetriever-Code-Qwen3 (4B)** | **86.22** | **86.70** | 41.14 | 88.10 | 75.54 |
| **CPRetriever-Prob-Qwen3 (4B)** | 80.84 | 87.10 | **74.33** | **96.15** | **84.60** |

### 消融实验（Group-InfoNCE vs 其他损失）

| 损失函数 | T2C | C2C | Avg |
|---|---|---|---|
| InfoNCE 单正样本 | 基线 | 基线 | 基线 |
| Multi-Pos | 略优 | 略优 | +1~2% |
| **Group-InfoNCE** | **最优** | **最优** | 在两种 base model 上一致提升 |

### 关键发现
- CPRetriever-Code 在代码检索（T2C 70.4、C2C 70.6）上大幅超越同规模最强基线 SFR-Emb-Code-2B（T2C 39.6），提升约 78%
- 微调为 Prob 后在 P2Dup 上从 38.7 跃升至 60.1（+55%），S2Full 从 81.4 到 90.7（+11%），同时 T2C 下降到 56.5——两类任务存在内在张力
- 代码领域专用模型持续优于通用嵌入模型（如 7B 的 Qodo 在 Avg 上不如 2B 的 CPRetriever-Prob）
- Qwen3-4B 作为 backbone 比 SFR-2B 更强，CPRetriever-Prob-Qwen3 在四任务均值达 84.6

### 题目相似性对 LLM 评测的影响

对 388 道 2024 年 9 月后发布的 LiveCodeBench 题目进行分析：

- **通过率随相似度上升**：与历史题目最大余弦相似度越高的题目，所有模型的平均通过率越高。中等难度题目正相关性最强；困难题目通过率普遍低但高相似区间仍有上升趋势。
- **高相似度压缩模型差异**：O3-Mini 和 O4-Mini 的 Low/Medium/High 三个变体在高相似度区间（0.80-0.90）性能差异几乎消失，而低相似度区间差异显著。这说明高相似题目使弱模型也能靠记忆获得好成绩，掩盖了真实推理能力差距。
- **难度与相似度部分独立**：简单题目的历史相似度略高于困难题目，但差异有限，说明相似度应作为独立因素在基准构建中控制。

## 亮点与洞察
- **Group-InfoNCE** 是处理"一对多"正样本关系的通用方案，核心思想（组相似度 + 方差正则化）可迁移到同义句多表达、同概念多图片等多正样本对比学习场景
- **对 LiveCodeBench 的相似性分析**揭示了社区基准中一个隐藏的系统性偏差：高相似度题目膨胀了模型评测分数并掩盖了模型间真实能力差距，未来基准应按相似度分层或过滤
- **竞赛编程检索从"代码级"扩展到"题目级"**是一个有远见的问题定义——从实际需求（赛事查重、教育搜索）出发，填补了现有基准空白
- **两阶段训练策略**（先 Code 再 Prob）揭示了有趣的 trade-off：T2C 依赖实现细节，P2Dup/S2Full 依赖高层语义，两者存在内在张力，因此发布两个专用模型是合理的工程选择

## 局限与展望
- 重复题目对标注量有限（~700 对），P2Dup 测试集仅 168 query，评估鲁棒性和统计显著性受限
- 仅基于文本嵌入做相似度，未利用题目的结构化信息（如约束范围、算法标签、时间/空间限制）
- 测试集虽做了时间分割，但随着新模型训练数据更新，这些题目未来仍可能泄露，作者计划每 6-12 个月更新
- 仅分析了 LiveCodeBench，未扩展到 HumanEval、APPS 或数学竞赛（IMO、AIMO）等场景
- 可结合 RAG 技术，用检索到的相似题目辅助 LLM 解题，形成检索-生成闭环

## 相关工作对比
- **vs CoIR/APPS 基准**：仅有 T2C 和 C2C 两个任务，无时间分割，数据截止 2020-2023。CPRet 新增两个题目级任务，严格时间分离，数据更新至 2024.12
- **vs SFR-Embedding-Code-2B**：代码领域强嵌入模型，在竞赛编程 T2C 上仅 39.6。CPRetriever-Code 同规模达 70.4，提升 78%
- **vs Qwen3-Embedding-4B/8B**：通用大模型在 P2Dup 上仅 56.6/53.2，专用微调后 CPRetriever-Prob-Qwen3 (4B) 达 74.3，提升 31%
- **vs Qodo-Embed-1-7B**：7B 代码嵌入模型在 S2Full 上达 91.2 但 T2C 仅 36.5，说明通用高性能与专项优化间存在 trade-off

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统定义竞赛编程题目级检索任务，Group-InfoNCE 有理论启发性
- 实验充分度: ⭐⭐⭐⭐⭐ 20+ 模型对比，四任务全覆盖，LiveCodeBench 相似性分析有说服力
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机充分，图表丰富，实验设计严谨
- 价值: ⭐⭐⭐⭐ 数据集/模型/在线 Demo 全部开源，对竞赛编程社区和 LLM 评测社区有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [Differentially Private High-dimensional Variable Selection via Integer Programming](differentially_private_high-dimensional_variable_selection_via_integer_programmi.md)
- [ORBIT -- Open Recommendation Benchmark for Reproducible Research with Hidden Tests](orbit_--_open_recommendation_benchmark_for_reproducible_research_with_hidden_tes.md)
- [Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](impact_of_dataset_properties_on_membership_inference.md)
- [Differential Privacy for Euclidean Jordan Algebra with Applications to Private Symmetric Cone Programming](differential_privacy_for_euclidean_jordan_algebra_with_applications_to_private_s.md)
- [ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization](forensichub_a_unified_benchmark_codebase_for_all-domain_fake_image_detection_and.md)

<!-- RELATED:END -->
