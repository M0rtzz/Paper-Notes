---
title: >-
  [论文解读] Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation
description: >-
  [ICML 2025][数学推理] 利用 Art of Problem Solving (AoPS) 论坛的社区内容，构建了 652K 奥赛级数学 QA 对的训练集 AoPS-Instruct 和带时间戳的抗污染评估集 LiveAoPSBench，揭示了 LLM 在旧数据上的高表现可能源于预训练数据泄露而非真正推理能力。
tags:
  - ICML 2025
  - 数学推理
  - 数据污染
  - 奥赛数学
  - 指令微调
  - 评估基准
---

# Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation

**会议**: ICML 2025  
**arXiv**: [2501.14275](https://arxiv.org/abs/2501.14275)  
**代码**: [livemathbench.github.io](https://livemathbench.github.io/leaderboard.html)  
**领域**: LLM/NLP  
**关键词**: 数学推理, 数据污染, 奥赛数学, 指令微调, 评估基准

## 一句话总结

利用 Art of Problem Solving (AoPS) 论坛的社区内容，构建了 652K 奥赛级数学 QA 对的训练集 AoPS-Instruct 和带时间戳的抗污染评估集 LiveAoPSBench，揭示了 LLM 在旧数据上的高表现可能源于预训练数据泄露而非真正推理能力。

## 研究背景与动机

奥赛级数学推理是 LLM 最具挑战性的任务之一，但面临两大瓶颈：

**训练数据稀缺**：现有 SFT 数据集（GSM8K、MATH、Orca-Math 等）主要覆盖中小学到高中水平，缺乏大规模的竞赛级数学数据。人工创建奥赛级题目和解答成本极高，即使是专家也需大量时间。

**评估不可靠**：GSM8K 和 MATH 等经典基准已被 SOTA 模型刷到 90%+ 的准确率，趋于饱和。更严重的是，公开测试集容易被训练数据污染——n-gram 匹配等去污染方法无法捕获改写后的重复题目。

AoPS 论坛拥有超过 100 万条数学讨论帖，内容以竞赛级问题为主（AMC、AIME、IMO 等），是天然的高质量数学数据源。然而论坛数据非结构化、包含无关评论和不完整解答，直接使用困难。本文的核心目标是：**设计一条自动化流水线，将非结构化的论坛内容转化为高质量的训练集和抗污染评估集**。

## 方法详解

### 整体框架

整个系统分为两条并行流水线：

- **训练流水线** → AoPS-Instruct（652K QA 对，2023年12月前的帖子）
- **评估流水线** → LiveAoPSBench（3,863 条样本，2024年1-8月的帖子）

两条流水线共享前端数据采集和问题检测步骤，在后端的质量控制策略上有所不同。

### 关键设计

#### 训练集构建（AoPS-Instruct）五步流水线

**Step 0 — 原始论坛数据采集**：从 AoPS 论坛收集 1,076,712 条讨论帖（topic），每条帖子包含问题描述和后续回复。

**Step 1 — 数学问题检测**：使用 Qwen 2.5 14B 模型对每条帖子的首条消息进行分类，判断是否为数学问题。通过手工设计的 few-shot prompt 实现。此步骤筛除 598,375 条无关帖子，保留 478,337 条数学问题。

**Step 2 — 问答对提取**：使用 Llama 3.1 70B 模型从帖子的后续讨论中识别和提取问题及其对应解答。选用 70B 大模型是因为该任务需要理解整段对话上下文并判断哪些回复包含有效解答。

**Step 3 — 解答重写（Solution Rewriting）**：这是流水线中最关键的一步。论坛用户的解答通常非常简洁，会跳过"显然"的推理步骤（如直接写出 AM-GM 不等式的结果而不提及定理名称）。实验表明，直接用简洁解答微调会**显著降低**模型在标准基准上的表现。因此使用 Qwen 2.5 72B 将所有解答重写为详细的逐步推理格式（step-by-step），补充中间推理步骤，统一格式，并将最终答案置于 `\boxed{}` 中。

**Step 4 — 数据去污染**：使用 10-gram 精确匹配去污染方法，确保训练集与常用数学基准（MATH、GSM8K 等）的测试集无重叠。

#### 评估集构建（LiveAoPSBench）

LiveAoPSBench 的核心设计理念是**利用时间戳实现抗污染**：

1. **时间戳排序**：只使用最近的论坛帖子（2024年1-8月），确保数据出现在大多数 LLM 的训练截止日期之后。
2. **启发式过滤**：排除证明类问题，只保留有具体数值答案（boxed answer）的题目。
3. **更严格的去污染**：使用 8-gram 匹配（比训练集的 10-gram 更严格），进一步排除可能与训练语料重叠的问题。
4. **双模型交叉验证**：分别用 Llama 3.1 70B 和 Qwen 2.5 72B 对每个问题独立重写解答，得到三元组 $(A_{\text{qwen}}, A_{\text{llama}}, A_{\text{original}})$。只有两个模型答案一致的 QA 对才被保留，使用字符串匹配、数值匹配和 SymPy 符号等价判断来进行一致性检查。
5. **持续更新**：流水线全自动化，可以持续获取最新论坛数据更新评估集，使基准始终保持"未见过"的状态。

### 损失函数 / 训练策略

- **微调策略**：标准指令微调（SFT），以问题为 instruction、重写后的解答为 response
- **训练轮数**：3 个 epoch（消融实验表明更多轮次无额外收益）
- **数据混合**：探索了三种配置——单独用 AoPS-Instruct、单独用 Numina、两者混合
- **模板格式**：使用各模型原生的 chat template（如 Mathstral 使用 `<s>[INST] question [/INST] solution`）

## 实验关键数据

### 主实验

在 4 个模型 × 4 个基准上评估了不同训练数据的效果：

| 模型 | 微调数据 | LiveAoPS'24 | MATH | OlympiadBench | Omni-Math |
|------|---------|-------------|------|---------------|-----------|
| DeepSeek-Math-7B | 无 SFT | 11.7 | 47.1 | 14.5 | 12.3 |
| DeepSeek-Math-7B | Numina | 16.3 | 55.5 | 22.7 | 17.0 |
| DeepSeek-Math-7B | **AoPS-Ins** | **19.0** | **58.8** | **24.3** | **17.8** |
| DeepSeek-Math-7B | Numina+AoPS | **19.7** | 58.8 | **25.6** | **18.0** |
| Mathstral-7B | 无 SFT | 15.4 | 56.3 | 21.2 | 15.9 |
| Mathstral-7B | Numina | 16.6 | 54.6 | 23.4 | 17.1 |
| Mathstral-7B | **AoPS-Ins** | **23.6** | **60.8** | **27.1** | **19.9** |
| Mathstral-7B | Numina+AoPS | **24.9** | 59.6 | **29.6** | **21.1** |
| Llama-3.2-3B | 无 SFT | 12.0 | 47.4 | 16.1 | 12.9 |
| Llama-3.2-3B | AoPS-Ins | 16.7 | 54.6 | 19.6 | 16.4 |
| Llama-3.2-3B | Numina+AoPS | **17.4** | **55.6** | **22.8** | **17.2** |
| Llama-3.2-1B | 无 SFT | 5.3 | 28.8 | 4.7 | 7.0 |
| Llama-3.2-1B | AoPS-Ins | 10.0 | 34.7 | 11.1 | 11.0 |
| Llama-3.2-1B | Numina+AoPS | **11.2** | **36.6** | **12.0** | **11.7** |

**关键结论**：AoPS-Instruct 在所有模型和基准上都优于 Numina 单独微调；两者混合训练实现最佳效果。

### 消融实验

#### 解答重写的影响

| 配置 | 效果 | 说明 |
|------|------|------|
| 原始论坛解答（无重写） | 性能**显著下降** | 简洁解答破坏了 chain-of-thought 能力 |
| Llama 3.1 70B 重写 | 明显提升 | 但在竞赛级基准上略逊于 Qwen |
| Qwen 2.5 72B 重写 | **最优** | 细节更丰富、更简洁（less verbose） |

#### 评估集质量验证

| 验证维度 | 结果 | 说明 |
|----------|------|------|
| 人工标注正确率 | 92% | 10名研究生标注 386 条（10%），5% 错误，3% 无答案 |
| 与 OlympiadBench 相关性 | 高度相关 | 自动构建的基准与人工构建的基准质量一致 |
| 标注者间一致性 | 高 | 每题两人独立标注 |

#### 时间戳与污染率的关系

| 时间窗口 | 23/01-04 | 23/05-08 | 23/09-12 | 24/01-04 | 24/05-08 |
|----------|----------|----------|----------|----------|----------|
| 10-gram 重叠率 | 13.24% | 11.65% | 12.82% | 9.92% | 6.88% |

随着时间推移，与 Numina 训练集的重叠率持续下降，验证了时间戳分割的有效性。

### 关键发现

1. **性能随时间下降**：所有 17 个被评估的 LLM 在 2024 年的题目上表现都低于 2023 年，准确率下降幅度从 2.4% 到 23.6%，表明旧基准上的高分可能来自数据泄露。
2. **小模型受污染影响更大**：Llama-3.2-1B 的性能下降最为剧烈（23.6%），暗示小模型更依赖记忆而非推理。
3. **数学专用模型更鲁棒**：Qwen2.5-Math 系列的性能下降仅 4-5%，远低于通用模型。
4. **数据集互补性**：AoPS-Instruct 与 Numina 的重叠率低于 14.1%，混合训练带来额外增益。

## 亮点与洞察

1. **时间戳 = 天然去污染器**：这是一个简单但深刻的洞察——只要评估数据严格新于训练数据的截止时间，就能有效避免污染。比复杂的 n-gram 匹配或 LLM 检测更可靠。
2. **解答重写是关键**：不仅仅是"有数据"就够了，数据的表述形式直接影响模型能力。简洁的专家解答反而有害，详细的 step-by-step 格式才能增强 chain-of-thought 推理。
3. **社区数据的价值被低估**：AoPS 论坛每月产生 1,000+ 道新数学题，这种持续生长的数据源既能训练又能评估，远比一次性的人工标注数据集有价值。
4. **流水线的可迁移性**：该方法不限于数学领域，可推广到物理、计算机科学等其他知识密集型论坛。

## 局限与展望

1. **缺乏视觉内容**：当前仅处理纯文本题目，几何等严重依赖图示的领域覆盖不足。
2. **无法评估证明题**：评估集仅包含有明确数值答案的问题，排除了大量需要逻辑推理和多步证明的奥赛问题。
3. **社区内容质量不一**：论坛解答质量参差不齐，虽然有过滤但仍可能引入噪声。
4. **重写模型的天花板**：解答质量受限于重写模型（Qwen 2.5 72B）的能力，未来可用更强模型提升。
5. **未探索 RL/RLHF**：仅使用了 SFT，未结合强化学习方法（如 DPO、PPO），可能限制了训练效果的上限。

## 相关工作与启发

- **LiveCodeBench**（Jain et al., 2024）：代码领域的抗污染基准，启发了本文的时间戳分割思想
- **Numina**（Li et al., 2024）：最相关的竞品，包含 190K 奥赛级 QA 对 + 其他 SFT 数据集的混合，也使用了 GPT-4o 进行解答重写
- **OpenMathInstruct**（Toshniwal et al., 2024）：1.8M QA 对，但均由 Mixtral 生成，非奥赛级别
- **DeepSeek-R1**（2025）：最新的推理模型，在 LiveAoPSBench 上也有评估

**对后续工作的启发**：可以将此流水线应用到其他领域的在线社区（如 Stack Overflow、Physics Forums），构建持续更新的训练和评估资源。

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐ | 时间戳抗污染思想不算全新（借鉴 LiveCodeBench），但在数学领域的系统实现值得认可 |
| 技术深度 | ⭐⭐⭐ | 流水线工程完善，但核心技术（LLM 过滤 + 重写）相对直接 |
| 实验充分性 | ⭐⭐⭐⭐ | 4 个模型 × 多基准，有消融、人工验证、时间趋势分析 |
| 实用价值 | ⭐⭐⭐⭐⭐ | 数据集和基准持续更新，对社区有直接贡献 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，图表丰富 |
| **综合** | **⭐⭐⭐⭐** | **扎实的数据集工作，核心贡献在抗污染评估和开源高质量奥赛数据** |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [How Much Can We Forget about Data Contamination?](how_much_can_we_forget_about_data_contamination.md)
- [Improved and Oracle-Efficient Online $\ell_1$-Multicalibration](improved_and_oracle-efficient_online_ell_1-multicalibration.md)
- [Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](../../ICLR2026/llm_evaluation/truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)
- [PhantomWiki: On-Demand Datasets for Reasoning and Retrieval Evaluation](phantomwiki_on-demand_datasets_for_reasoning_and_retrieval_evaluation.md)
- [MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](../../ACL2025/llm_evaluation/mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)

<!-- RELATED:END -->
