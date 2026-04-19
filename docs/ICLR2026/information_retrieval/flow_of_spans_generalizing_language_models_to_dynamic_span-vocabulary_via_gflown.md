---
title: >-
  [论文解读] Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets
description: >-
  [ICLR 2026][GFlowNets] 提出 FoSS，首次将 GFlowNets 引入 span 级别语言模型，通过构建 DAG 结构的状态空间代替传统 token-by-token 的树形结构，实现更灵活多样的文本生成，MAUVE 分数最高提升 12.5%。
tags:
  - ICLR 2026
  - GFlowNets
  - 动态词表
  - span生成
  - DAG状态空间
  - 文本生成
---

# Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets

**会议**: ICLR 2026  
**arXiv**: [2602.10583](https://arxiv.org/abs/2602.10583)  
**代码**: [GitHub](https://github.com/sappho-x/Flow-of-Spans)  
**领域**: 模型压缩 / LLM效率  
**关键词**: GFlowNets, 动态词表, span生成, DAG状态空间, 文本生成

## 一句话总结

提出 FoSS，首次将 GFlowNets 引入 span 级别语言模型，通过构建 DAG 结构的状态空间代替传统 token-by-token 的树形结构，实现更灵活多样的文本生成，MAUVE 分数最高提升 12.5%。

## 研究背景与动机

标准自回归语言模型逐 token 生成文本，使用固定有限词表。这种方式有两个本质局限：（1）固定词表限制了生成粒度；（2）token 级生成形成的是树状状态空间（每个状态只有唯一前驱），限制了模型探索替代生成路径的能力。

近年来有工作引入动态词表和变长生成单元（如 CoG、kNN-LM），允许模型采样检索到的 text span。但这些方法忽略了一个关键事实：同一个句子可以由不同长度的 span 组合而成（如 "ABCDEFGH" 可以分解为 "AB-CD-EFGH" 或 "AB-C-DE-FG-H"），本质上构成了有向无环图（DAG）结构，而非树结构。现有方法没有显式建模这一 DAG 空间，导致对组合路径的探索受限。

GFlowNets 是一种强大的生成模型，擅长在 DAG 结构的状态空间上高效探索和泛化。但此前将 GFlowNets 应用于语言模型的工作仍停留在 token 级别的树形空间，未能发挥 GFlowNets 的全部潜力。

## 方法详解

### 整体框架

FoSS 将文本生成建模为在 DAG 结构状态空间上的 GFlowNets 采样过程。模型从动态 span 词表中选择变长文本片段进行拼接。由于同一文本可以通过不同 span 组合到达，自然形成 DAG 结构。

FoSS 定义了完整的 MDP 框架：

- **状态**：由前缀 c 和已生成文本 t_i 拼接而成
- **动作**：从动态词表中选择一个 span（单词或多词短语）
- **转移**：确定性地将选中的 span 追加到当前状态
- **奖励**：由专门的奖励模型评估终端状态的文本质量

与传统 token 级生成不同，span 级动作使得同一状态可通过多条不同轨迹到达。例如状态 "ABCD" 可通过 "AB-ABCD" 或 "AB-ABC-ABCD" 两条路径到达，这种 DAG 结构使 GFlowNets 能全面探索多条组合路径。

### 关键设计

**1. DAG 诱导的 span 分割算法**

为从训练数据中构建 DAG 结构的训练轨迹，FoSS 提出概率化 span 分割算法。在标准前向最大匹配基础上，引入受控的随机提前停止机制：对每个候选 span，以概率 p_r 决定是否将其作为短语提取。通过使用不同阈值集合 P = {p_0=0, p_1, ..., p_n}，同一文档产生多条不同的分割轨迹，共享公共子序列，自然诱导 DAG 结构。

**2. 策略网络架构**

策略网络包含两个组件：

- **前缀编码器**：Transformer 架构（初始化自 GPT-2），通过因果注意力对当前状态编码，得到上下文向量 h_i
- **Span 编码器**：双向 Transformer（初始化自 BERT），对候选短语编码。对检索文档中的短语（位置 s 到 e），通过 MLP 变换起止位置嵌入后拼接得到完整短语表示 v_a

前向策略分布为 P_SLM(a|s_i; theta) 正比于 exp(h_i^T v_a)，通过内积计算上下文与候选 span 的匹配分数。

动态词表由三部分组成：（1）token 级固定词表 V；（2）从外部语料检索到的短语 T（长度 2-8 token 的子串）；（3）终止动作。

**3. 后向策略与混合训练**

后向策略采用均匀分布，对动态词表中当前状态的所有可能后缀赋予相等概率。这与树形空间中后向策略退化为确定性（P_B=1）形成对比。

训练使用三种来源的轨迹组成 mini-batch：前向策略在线采样轨迹、奖励优先的 replay buffer 高奖励轨迹、训练集构造轨迹。第一个 epoch 完全使用离线数据，此后以 0.2/0.8 概率混合离线/在线轨迹。

### 损失函数 / 训练策略

采用**子轨迹平衡（SubTB）**学习目标。对一条完整轨迹，损失函数对所有合法子轨迹对求和，其中指示函数确保只在完整句子间计算，将学习信号集中在有意义的句子过渡上。

**奖励函数**由两个互补部分组成：

- **语言模型（LM）**：衡量流畅度，给定前缀下序列的似然
- **偏好模型（PM）**：Bradley-Terry 目标训练的判别器，区分人类文本和模型生成文本，使用 score-centering 正则化

最终奖励为两者的加权几何平均，alpha 控制流畅度与偏好对齐的权衡。LM 和 PM 均基于 GPT-2 全参微调。

## 实验关键数据

### 主实验

**表1：开放域文本生成 MAUVE 分数**

| 方法 | In-Domain Greedy | In-Domain Nucleus | OOD Nucleus | Scaling Nucleus |
|------|:---:|:---:|:---:|:---:|
| Transformer w/ FT | 19.87 | 23.43 | 26.85 | 21.31 |
| kNN-LM | 19.92 | 22.50 | 24.75 | 23.39 |
| CoG | 26.01 | 26.14 | 28.14 | 26.97 |
| GFlowNets-FT | 26.58 | 29.61 | 28.62 | - |
| **FoSS** | **30.78** | **31.65** | **32.17** | **33.79** |

FoSS 全面领先，In-Domain Nucleus 下比 CoG 高 5.51%，比 GFlowNets-FT 高 2.04%。

**表2：GPT-4 成对评估（FoSS vs. baselines）**

| 对比方法 | FoSS更好 | 持平 | FoSS更差 |
|----------|:---:|:---:|:---:|
| Transformer | 53% | 19% | 28% |
| kNN-LM | 67% | 15% | 18% |
| CoG | 42% | 31% | 27% |
| GFlowNets-FT | 55% | 29% | 16% |

**表3：知识密集型任务准确率**

| 方法 | TruthfulQA | OpenBookQA | ARC-Challenge |
|------|:---:|:---:|:---:|
| Transformer w/ FT | 28.76 | 22.71 | 24.00 |
| CoG | 29.38 | 24.29 | 24.34 |
| **FoSS** | **30.45** | **26.20** | **24.63** |

### 消融实验

| 变体 | In-Domain MAUVE | Diversity | OOD MAUVE |
|------|:---:|:---:|:---:|
| FoSS w/o DAG（退化为树） | 29.61 | 65.72 | 28.62 |
| FoSS w/o PM | 28.25 | 89.91 | 30.49 |
| FoSS w/o LM | 29.09 | 92.77 | 29.79 |
| **FoSS（完整）** | **31.65** | 92.48 | **32.17** |

### 关键发现

1. **DAG 结构至关重要**：移除 span 使状态空间退化为树后，MAUVE 显著下降（31.65 到 29.61），Diversity 从 92.48 暴跌至 65.72
2. **LM 和 PM 奖励互补**：单独移除任一组件都导致 MAUVE 下降；PM-only 有最高 Diversity 但 MAUVE 明显下降
3. **优秀的扩展性**：在模型规模（GPT-2 base 到 XL）、训练数据量和检索库大小三个维度上保持稳定提升
4. **域适应能力**：仅更新检索库就能超越域内微调的 Transformer；0.47% 数据下就能超越完整训练的 CoG

## 亮点与洞察

- 首次将 span 级生成与 GFlowNets 结合，构建了真正的 DAG 状态空间
- 概率化 span 分割算法通过随机提前停止为同一文本产生多条重叠分割路径
- 推理效率与标准 Transformer 相当，因为 span 级生成减少了解码步数
- 在极少量训练数据（0.47%）下即可超越完整训练的基线方法

## 局限与展望

1. 实验主要基于 GPT-2 规模，尚不清楚在更大规模 LLM 上的表现
2. 依赖外部检索库，需要预编码源文本集合，增加部署复杂度
3. 训练涉及多个组件（策略网络、奖励模型、检索器），流程较为复杂
4. 仅在英文任务上评估，跨语言泛化能力未知

## 相关工作与启发

本文将 GFlowNets 的 DAG 探索能力与动态词表语言模型结合，是新颖的交叉方向。对于研究 LLM 生成多样性、检索增强生成的工作有参考价值。span 级生成的思路也可启发投机解码中的 draft 策略设计。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（首次在 span 级 LM 中构建 DAG + GFlowNets）
- 技术深度: ⭐⭐⭐⭐⭐（完整 MDP 形式化 + SubTB 训练 + 双组件奖励）
- 实验充分度: ⭐⭐⭐⭐（多维评估 + GPT-4 评判 + 消融 + 扩展性）
- 实用性: ⭐⭐⭐（依赖检索库，部署门槛较高）
- 写作质量: ⭐⭐⭐⭐（结构清晰，图示直观）

<!-- RELATED:START -->

## 相关论文

- [TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)
- [FutureMind: Equipping Small Language Models with Strategic Thinking-Pattern Priors via Adaptive Knowledge Distillation](futuremind_equipping_small_language_models_with_strategic_thinking-pattern_prior.md)
- [Hierarchical Concept-based Interpretable Models](hierarchical_concept-based_interpretable_models.md)
- [Grounding Language Models for Visual Entity Recognition](../../ECCV2024/information_retrieval/grounding_language_models_for_visual_entity_recognition.md)

<!-- RELATED:END -->
