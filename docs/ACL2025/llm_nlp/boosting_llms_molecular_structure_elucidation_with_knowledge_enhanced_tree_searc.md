---
title: >-
  [论文解读] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning
description: >-
  [LLM/NLP] 提出 K-MSE（Knowledge-enhanced Molecular Structure Elucidation）框架，构建分子子结构知识库扩展 LLM 的化学结构空间覆盖，设计专用分子-光谱打分器替代 LLM 自身评估，结合蒙特卡洛树搜索（MCTS）实现测试时推理缩放，在 MolPuzzle 基准上分别将 GPT-4o-mini 和 GPT-4o 的准确率从 3.7% 和 27.8% 提升至 27.3% 和 39.8%。
tags:
  - LLM/NLP
---

# Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning

| 项目 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| 作者 | Xiang Zhuang, Bin Wu, Jiyu Cui, Kehua Feng, Xiaotong Li, Huabin Xing, Keyan Ding, Qiang Zhang, Huajun Chen (浙江大学, UCL) |
| arXiv | [2506.23056](https://arxiv.org/abs/2506.23056) |
| 代码 | [GitHub](https://github.com/HICAI-ZJU/K-MSE) |
| 领域 | LLM/NLP, 化学推理, 分子结构解析 |
| 关键词 | molecular structure elucidation, MCTS, knowledge base, reward model, spectral data, test-time scaling |

## 一句话总结

提出 K-MSE（Knowledge-enhanced Molecular Structure Elucidation）框架，构建分子子结构知识库扩展 LLM 的化学结构空间覆盖，设计专用分子-光谱打分器替代 LLM 自身评估，结合蒙特卡洛树搜索（MCTS）实现测试时推理缩放，在 MolPuzzle 基准上分别将 GPT-4o-mini 和 GPT-4o 的准确率从 3.7% 和 27.8% 提升至 27.3% 和 39.8%。

## 研究背景与动机

- **核心问题:** 分子结构解析是化学实验分析的基础任务——从 NMR、IR 等光谱数据推断分子结构。即使专家也需 10-15 分钟处理一个分子。LLM 有潜力自动化这一过程，但面临两大挑战。
- **现有不足:** (1) LLM 缺乏对化学分子结构空间的全面覆盖——对噻吩等非常见结构常误判为苯环（最常见芳香结构）；(2) LLM 无法准确评估自身推理结果，缺乏领域知识来判断预测分子与光谱数据的匹配度，导致树搜索推理缺乏有效奖励信号。
- **研究动机:** 通过外部知识增强化学结构覆盖 + 专用打分器提供准确奖励 → 结合 MCTS 实现 LLM 在分子结构解析中的测试时推理缩放。

## 方法详解

### 整体框架

K-MSE 由三个组件构成：
1. **分子子结构知识库** $\mathcal{KB} = \{(s_i, d_i)\}$：包含子结构 SMILES 表示和文本描述，从 MOSES 分子数据库提取环状和链状子结构
2. **分子-光谱打分器**：由分子编码器 $g_m$（GIN + MLP 处理分子图和指纹）和光谱编码器 $g_s$（Transformer 处理 C-NMR/H-NMR 的化学位移、裂分模式、耦合常数）组成
3. **MCTS 推理框架**：先从知识库检索相关子结构 → 迭代执行选择(UCT)→扩展(Critique+Rewrite)→评估(打分器)→反向传播

### 关键设计

1. **知识库构建**：从 MOSES 数据库自动提取分子子结构，利用 LLM 结合外部工具生成的结构信息自动生成可靠描述。兼顾多样性和通用性。
2. **专用打分器**：分子编码器使用 GIN 编码分子图 + MLP 编码 Morgan 指纹，光谱编码器将 NMR 化学位移和耦合常数离散化为 token ID 后输入 Transformer。训练采用 NT-Xent 对比学习损失，使匹配的分子-光谱对嵌入相似度最大化。
3. **打分器双重角色**：既作为 MCTS 奖励模型评估候选分子（$R(a') = \text{sim}(g_m(m_{a'}), g_s(n))$），又作为知识库检索桥梁——用光谱编码器编码查询光谱，用分子编码器编码子结构，进行 Top-k 检索。

### 损失函数

打分器训练使用 NT-Xent 对比学习损失：最大化正确分子-光谱对的余弦相似度，最小化批内负样本对的相似度，温度参数 $\tau$ 控制分布锐度。MCTS 反向传播采用 $Q(a) = 0.5 \times Q(a') + 0.5 \times Q(a)$ 的加权更新。

## 实验

### 主实验——MolPuzzle 基准（216 个分子，zero-shot）

| 模型 | 方法 | Morgan FTS | MACCS FTS | ACC |
|------|------|-----------|-----------|-----|
| GPT-4o-mini | baseline | 0.260 | 0.512 | 0.037 |
| GPT-4o-mini | + Self-Refine | 0.287 | 0.523 | 0.069 |
| GPT-4o-mini | + MCTSr | 0.281 | 0.530 | 0.069 |
| GPT-4o-mini | **+ K-MSE** | **0.470** | **0.651** | **0.273** |
| GPT-4o | baseline | 0.493 | 0.690 | 0.278 |
| GPT-4o | + Self-Consistency | 0.551 | 0.732 | 0.347 |
| GPT-4o | **+ K-MSE** | — | — | **0.398** |
| Llama-3.2-11B | baseline | 0.163 | 0.349 | 0.014 |
| Llama-3.2-11B | **+ K-MSE** | **0.298** | **0.465** | **0.111** |

### 消融实验

| 消融组件 | 对 GPT-4o-mini ACC 的影响 |
|---------|-------------------------|
| 完整 K-MSE | 0.273 |
| 移除知识库 | 下降明显——LLM 无法识别非常见子结构 |
| 用 LLM 替代专用打分器 | 下降显著——LLM 无法准确评估分子-光谱匹配 |
| 移除 Critique 中的分子图像 | 下降——纯文本 critique 难以发现结构错误 |
| 移除 Critique 中的化学式 | 下降——缺乏化学约束信息 |

### 关键发现

- K-MSE 在所有基座模型上带来大幅提升：GPT-4o-mini ACC +23.6%，GPT-4o ACC +12.0%，Llama-3.2-11B ACC +9.7%
- 现有通用推理增强方法（Self-Refine, MCTSr, MAD）在分子结构解析上效果有限——缺乏领域知识是核心瓶颈
- 专用打分器远优于 LLM 自评估——LLM 缺乏判断分子-光谱匹配度的领域知识
- 知识库的子结构信息对处理非常见分子结构至关重要
- 作为即插即用框架，K-MSE 可与任何 LLM 组合使用

## 亮点

- 首次将测试时推理缩放（test-time scaling）+ 外部知识增强应用于分子结构解析任务
- 打分器同时充当奖励模型和检索桥梁的双重角色设计精巧
- 框架的即插即用特性使其具有很强的实用价值
- MolPuzzle 上 20%+ 的绝对准确率提升非常显著

## 局限性

- 仅在 MolPuzzle 基准上评估，该基准规模较小（216 个分子）
- 打分器的训练数据覆盖范围可能限制其对罕见分子类型的泛化能力
- MCTS 的迭代次数增加带来显著的推理时间成本（API 调用+打分器推理）
- 知识库是静态的，未探索在线扩展或自适应更新机制
- 仅考虑 NMR 和 IR 光谱，未处理质谱（MS）等其他常用分析数据

## 相关工作

- **LLM 化学推理:** ChemCrow (M. Bran et al., 2024) 集成外部工具、ChatDrug (Liu et al., 2024) 分子编辑、STRUCTCHEM (Ouyang et al., 2024) 预定义推理模板
- **树搜索推理:** Tree-of-Thought (Yao et al., 2023)、MCTSr (Zhang et al., 2024a)，但缺乏领域特定的准确奖励模型
- **分子结构解析:** MolPuzzle (Guo et al., 2024) 首次提出该任务的 LLM 基准

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验完整度 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes](mathneuro_math_reasoning_isolation.md)
- [PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services](disco_device-server_collaborative_llm-based_text_streaming_services.md)

<!-- RELATED:END -->
