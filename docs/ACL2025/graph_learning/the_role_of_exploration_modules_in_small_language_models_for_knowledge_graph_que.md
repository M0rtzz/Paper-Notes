---
title: >-
  [论文解读] The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering
description: >-
  [ACL 2025][图学习][知识图谱问答] 本文系统性地诊断了小语言模型（SLM，0.5B–8B）在 Think-on-Graph 知识图谱问答框架中失效的根因——**探索阶段（exploration）而非推理阶段是性能瓶颈**…
tags:
  - "ACL 2025"
  - "图学习"
  - "知识图谱问答"
  - "小语言模型"
  - "Think-on-Graph"
  - "段落检索"
  - "探索模块"
---

# The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering

**会议**: ACL 2025  
**arXiv**: [2509.07399](https://arxiv.org/abs/2509.07399)  
**代码**: [yijie-cheng/SLM-ToG](https://github.com/yijie-cheng/SLM-ToG/)  
**作者**: Yi-Jie Cheng, Oscar Chew, Yun-Nung Chen (ASUS / 台湾大学)
**领域**: Graph Learning / KGQA  
**关键词**: 知识图谱问答, 小语言模型, Think-on-Graph, 段落检索, 探索模块

## 一句话总结

本文系统性地诊断了小语言模型（SLM，0.5B–8B）在 Think-on-Graph 知识图谱问答框架中失效的根因——**探索阶段（exploration）而非推理阶段是性能瓶颈**，并证明用零样本、即插即用的轻量级段落检索模块（SentenceBERT/GTR，仅~110M参数）替代 SLM 进行 KG 遍历，即可在 CWQ 和 WebQSP 两个基准上带来一致且显著的 EM 提升。

## 研究背景与动机

**领域现状**: LLM×KG 范式（如 Think-on-Graph）通过让语言模型作为 agent 在知识图谱上动态交互、检索外部知识，被广泛认为是缓解大模型幻觉、提升可解释性的有效路径。ToG 框架在 GPT-4 等大模型上展现了优异效果（CWQ 上 EM 从 0.457 提升到 0.540，WebQSP 上从 0.710 提升到 0.813）。

**现有痛点**: 现有 LLM×KG 方法几乎全部依赖超大规模私有模型（如 GPT-4）或需要任务特定的训练/微调，在低资源环境下不可用。当终端用户仅能访问 0.5B–8B 参数的小语言模型时，这些方法是否仍然有效，缺乏系统性研究。

**核心矛盾**: 将 ToG 直接应用于 SLM 后，效果不升反降——SLM 使用 ToG 的平均 EM 在 CWQ 上仅为 0.217，甚至低于简单 CoT 基线的 0.219。ToG 为 LLM 设计的交互范式在小模型上完全失效，但失效的具体环节（初始化/探索/推理）和原因不清楚。

**本文目标**: 精确诊断 SLM 在 ToG 框架中哪个阶段失效，并提出一种无需训练、即插即用的修复方案，使 SLM 也能有效地从知识图谱中获益。

**切入角度**: 通过可控实验将"探索"和"推理"两个能力解耦——用 GPT-4.1 仅替换探索阶段，推理阶段仍由 SLM 完成——定量确认探索是 SLM 的唯一瓶颈。然后用在段落检索领域已经成熟的轻量模型来专门处理 KG 探索这一子任务。

**核心 idea**: 小模型不是不能推理，而是不会在知识图谱上"找路"——用专业的检索小模型帮它找路就行了。

## 方法详解

### 整体框架

本文基于 Think-on-Graph（ToG）框架，保留其三阶段流程但修改了关键的探索阶段。ToG 原始工作流为：

1. **初始化（Initialization）**: SLM 从输入问题中提取主题实体（topic entities），在知识图谱中定位这些实体，形成初始推理路径
2. **探索（Exploration）**: 采用 beam search 策略迭代扩展推理路径，每一步从邻居关系和实体中选择候选并剪枝。**本文的核心修改在这一步**——将 SLM 的排序/剪枝替换为轻量级检索模块
3. **推理（Reasoning）**: 收集到足够证据后，SLM 基于维护的推理路径生成最终答案

修改后的框架实现了"探索"与"推理"的解耦：探索由专业检索模块负责，推理由 SLM 负责，各司其职。整个过程无需任何训练或微调，检索模块以零样本方式即插即用。此外，将 SLM 调用次数从 $2ND+D+1$ 降至 $D+1$（$D$ 为迭代次数，$N$ 为推理路径数），大幅节约推理成本。

### 关键设计一：探索瓶颈的精确诊断

本文最关键的贡献在于通过精心设计的可控实验确认了探索阶段是 SLM 的性能瓶颈：

- **GPT-4.1 辅助探索实验**: 让 GPT-4.1 仅负责 KG 探索，将获得的知识三元组提供给 SLM 进行推理。结果显示，SLM 获得高质量上下文后性能大幅提升（SLM 平均 CWQ +0.159，WebQSP +0.238），证明 SLM 的推理能力本身足够，问题在探索环节
- **典型失败案例**: 问题"Northern District 所在国家的政府类型"——SLM 自主探索仅获得 `(Northern District, country, Israel)` 等两个三元组，缺少关键的 `(Israel, form_of_government, Parliamentary system)`，因此无法作答。而同一 SLM 拿到 GPT-4.1 探索到的完整三元组后可正确回答
- **交叉熵对齐度量**: 以 GPT-4.1 的探索输出为伪标签，SLM 探索输出与之的交叉熵随模型规模一致降低，定量支持"探索质量是限制因素"
- **Constrained Decoding 排除格式干扰**: 强制所有模型以 JSON 格式输出，消除 SLM 因格式不规范导致的性能虚低。结果显示格式因素被控制后，探索瓶颈的结论依然成立

### 关键设计二：轻量级段落检索替代 SLM 探索

确认瓶颈后，本文用段落检索模型替代 SLM 进行 KG 探索。每步给定问题 $q$ 和候选集 $P_{cand}$，检索模块计算相关性分数并选出 top-k：

$$P_q = \text{Top}_k(\text{score}(p, q)), \quad \forall p \in P_{cand}$$

三种检索方案各有特点：

- **BM25**: 基于词频和逆文档频率的经典稀疏排序函数，将问题分词后与候选段落的 TF-IDF 匹配。优点是无需 GPU，缺点是纯词汇匹配无法捕获语义
- **SentenceBERT**（~110M 参数）: 基于 BERT 的 Siamese 网络，通过对比学习生成语义丰富的句子向量，相似度为编码向量的点积 $\text{score}(p,q) = \langle \mathcal{T}(p), \mathcal{T}(q) \rangle$
- **GTR**（~110M 参数）: 基于 T5 的大规模双编码器，在多种检索任务上联合训练，泛化能力更强

SentenceBERT 和 GTR 仅约 110M 参数，远小于最小的 SLM（0.5B），引入的计算开销极小，且无需微调、零样本直接使用。

### 关键设计三：LLM vs. SLM 的不对称效应

一个重要发现：同一技术在 LLM 和 SLM 上效果相反。Sun et al. (2024) 报告在 GPT-4 上使用段落检索替代其自身探索会导致 EM 显著下降（如 CWQ 从 0.575 降至 0.505）。本文解释了背后原因：LLM 的 KG 探索能力已经很强，额外检索模块成了降级替代；而 SLM 的探索本就是短板，检索模块反而是升级。这一发现提醒社区不能简单将大模型结论迁移至小模型。

## 实验关键数据

### Table 1: ToG 在 LLM vs. SLM 上的效果对比

| 模型 | 方法 | CWQ (EM) | WebQSP (EM) |
|------|------|----------|-------------|
| GPT-4.1 | CoT | 0.457 | 0.710 |
| GPT-4.1 | ToG | **0.540** (+0.083) | **0.813** (+0.103) |
| Qwen2-0.5b | CoT | 0.129 | 0.272 |
| Qwen2-0.5b | ToG | 0.081 (-0.048) | 0.210 (-0.062) |
| Gemma2-2b | CoT | 0.127 | 0.373 |
| Gemma2-2b | ToG | 0.140 (+0.013) | 0.382 (+0.009) |
| Phi-3-mini-3.8b | CoT | 0.273 | 0.522 |
| Phi-3-mini-3.8b | ToG | 0.270 (-0.003) | 0.520 (-0.002) |
| Qwen2-7b | CoT | 0.275 | 0.544 |
| Qwen2-7b | ToG | 0.300 (+0.025) | 0.637 (+0.093) |
| Llama-3-8b | CoT | 0.291 | 0.603 |
| Llama-3-8b | ToG | 0.296 (+0.005) | 0.569 (-0.034) |
| **SLM 均值** | CoT | 0.219 | 0.456 |
| **SLM 均值** | ToG | 0.217 (-0.002) | 0.464 (+0.008) |

GPT-4.1 从 ToG 获得显著提升，而 SLM 均值几乎无变化甚至微降，ToG 对 SLM 基本无效。Qwen2-0.5b 使用 ToG 后 CWQ EM 反而从 0.129 暴跌至 0.081。

### Table 2: 轻量检索模块替代探索的效果

| 模型 | ToG 原始 | +BM25 | +SentenceBERT | +GTR |
|------|----------|-------|---------------|------|
| Qwen2-0.5b (CWQ) | 0.081 | 0.106 | 0.061 | **0.123** |
| Qwen2-0.5b (WebQSP) | 0.210 | 0.236 | 0.174 | **0.262** |
| Gemma2-2b (CWQ) | 0.140 | 0.127 | 0.136 | **0.152** |
| Gemma2-2b (WebQSP) | 0.382 | 0.236 | **0.520** | 0.511 |
| Phi-3-mini (CWQ) | 0.270 | 0.254 | 0.284 | **0.291** |
| Phi-3-mini (WebQSP) | 0.520 | 0.416 | 0.577 | **0.605** |
| Qwen2-7b (CWQ) | 0.300 | 0.251 | 0.318 | **0.331** |
| Qwen2-7b (WebQSP) | 0.637 | 0.513 | 0.665 | **0.671** |
| Llama-3-8b (CWQ) | 0.296 | 0.274 | 0.313 | **0.325** |
| Llama-3-8b (WebQSP) | 0.569 | 0.456 | 0.640 | **0.642** |

GTR 在 CWQ 上对所有 SLM 一致最优；SentenceBERT 和 GTR 在 WebQSP 上提升幅度更大（Gemma2-2b: 0.382→0.520）。BM25 的稀疏词频匹配在 KG 关系选择上表现不稳定，部分场景反而有害。

### Table 3: GPT-4.1 辅助探索的上界

| 模型 | CoT 基线 | GPT-4.1 辅助 ToG | 提升 |
|------|----------|-------------------|------|
| Qwen2-0.5b (CWQ/WebQSP) | 0.129 / 0.272 | 0.301 / 0.578 | +0.172 / +0.306 |
| Gemma2-2b (CWQ/WebQSP) | 0.127 / 0.373 | 0.306 / 0.672 | +0.179 / +0.299 |
| Phi-3-mini (CWQ/WebQSP) | 0.273 / 0.522 | 0.421 / 0.736 | +0.148 / +0.214 |
| Qwen2-7b (CWQ/WebQSP) | 0.275 / 0.544 | 0.409 / 0.746 | +0.134 / +0.202 |
| Llama-3-8b (CWQ/WebQSP) | 0.291 / 0.603 | 0.451 / 0.772 | +0.160 / +0.169 |

即使是 0.5B 的 Qwen2，在获得高质量探索上下文后也能大幅超越 CoT，证明推理能力不是瓶颈。

## 亮点与洞察

- **诊断价值大于方法创新**: 本文最核心的贡献不是提出新架构，而是通过严谨的实验设计精确定位了"SLM 在 KGQA 中失效"的根因。这种诊断式研究帮助社区避免在错误方向（如提升 SLM 推理能力）上投入
- **"解耦+专业化"的工程智慧**: 将复杂任务的不同阶段交给最擅长的组件——仅用 110M 的检索模型就能让 8B 模型显著受益，比端到端训练或 scaling up 更实际
- **LLM 和 SLM 的结论不可简单迁移**: 同一技术（段落检索替代探索）在 GPT-4.1 上导致性能下降（CWQ: 0.575→0.505），在 SLM 上却带来一致提升——这是一个深刻且反直觉的发现
- **零样本、零训练、即插即用**: 所有检索方案无需任务特定训练，对实际部署场景极为友好

## 局限与展望

- **单次运行无方差**: 所有结果基于单次运行，受限于计算资源无法报告多次运行的标准差，统计显著性有待确认
- **知识图谱单一**: 仅在 Freebase 上验证，Wikidata、YAGO 等其他主流 KG 上的泛化性未知
- **检索模型探索不充分**: 仅测试了 BM25/SentenceBERT/GTR，更先进的检索模型（如 ColBERT、BGE 系列）或 reranking 策略可能效果更好
- **其他阶段的瓶颈未分析**: 聚焦于探索阶段，但初始化阶段（主题实体提取）在 SLM 上是否也有瓶颈，未做对等分析
- **缺乏与训练式方案的对比**: 未与需要训练的 SLM-KG 方案（如 Reasoning on Graphs、G-Retriever）进行直接 EM 比较

## 相关工作与启发

- **Think-on-Graph (Sun et al., 2024)**: 本文的核心基础框架，提出 LLM 作为 agent 动态遍历 KG 的范式。本文揭示了该框架在 SLM 上的失效模式及其根因
- **Reasoning on Graphs (Luo et al., 2024)** / **G-Retriever (He et al., 2024)**: 提出额外推理/探索模块改善 LLM-KG 集成，但需要任务特定训练，不适用于零样本设定
- **LightProf (Ao et al., 2025)**: 同样关注降低 LLM-KG 集成的推理开销，提出轻量级推理框架
- **CuriousLLM (Yang et al., 2025)**: 用 LLM 增强的 KG 推理提升多文档 QA，代表了另一个将 KG 用于 NLP 下游任务的方向
- **SentenceBERT (Reimers & Gurevych, 2019)** / **GTR (Ni et al., 2022)**: 段落检索领域的成熟模型，本文展示了其在 KG 探索这一全新场景下的跨领域通用性
- **启发**: 在资源受限场景下，模块化设计——让不同规模的模型各负其责——比一味 scaling up 更务实，也更容易部署

## 评分

- 新颖性: ⭐⭐⭐ 方法本身无创新（直接使用已有检索模块），但诊断视角独特，揭示了社区忽视的关键问题
- 实验充分度: ⭐⭐⭐⭐ 5 个 SLM + 1 个 LLM、2 个数据集、3 种检索模块、constrained decoding 控制变量、交叉熵对齐分析、案例研究，实验设计严谨
- 写作质量: ⭐⭐⭐⭐ 三个 RQ 层层递进逻辑清晰，定性案例与定量分析结合
- 价值: ⭐⭐⭐⭐ 对 SLM+KG 的实践社区有直接指导意义，揭示了小模型使用知识图谱的正确姿势

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering](ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled.md)
- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)
- [\[ACL 2025\] Agent Steerable Search for Knowledge Graph Question Answering](agent_steerable_search_for_knowledge_graph_question_answering.md)
- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](../../ICML2026/graph_learning/kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)

</div>

<!-- RELATED:END -->
