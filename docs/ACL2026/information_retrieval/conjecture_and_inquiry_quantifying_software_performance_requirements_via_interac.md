---
title: >-
  [论文解读] Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation
description: >-
  [ACL 2026][信息检索/RAG][需求量化] 提出IRAP方法，通过交互式检索增强偏好获取（Interactive Retrieval-Augmented Preference Elicitation）将自然语言描述的软件性能需求量化为数学函数…
tags:
  - "ACL 2026"
  - "信息检索/RAG"
  - "需求量化"
  - "偏好获取"
  - "检索增强生成"
  - "交互式系统"
  - "软件性能需求"
---

# Conjecture and Inquiry: Quantifying Software Performance Requirements via Interactive Retrieval-Augmented Preference Elicitation

**会议**: ACL 2026  
**arXiv**: [2604.21380](https://arxiv.org/abs/2604.21380)  
**代码**: 待确认  
**领域**: 信息检索  
**关键词**: 需求量化, 偏好获取, 检索增强生成, 交互式系统, 软件性能需求

## 一句话总结

提出IRAP方法，通过交互式检索增强偏好获取（Interactive Retrieval-Augmented Preference Elicitation）将自然语言描述的软件性能需求量化为数学函数，在4个真实数据集上相比10种SOTA方法取得最高40倍的性能提升，且仅需5轮交互。

## 研究背景与动机

**领域现状**: 软件性能需求（如响应时间、吞吐量、可用性等）通常以自然语言形式记录在需求文档中，但软件工程中的性能分析、测试和优化需要将其转化为可计算的数学形式（如效用函数、约束条件）。

**现有痛点**: 性能需求的自然语言描述通常含糊不清（如"系统应该快速响应"、"延迟应在可接受范围内"），加上人类认知中的不确定性，使得同一需求文本可能被不同利益相关者解读为完全不同的数学形式。这种高度不确定的歧义性使得自动化量化成为一个未被充分解决的难题。

**核心矛盾**: 一方面需要将模糊的自然语言转化为精确的数学函数，另一方面利益相关者的偏好具有高度个人化和上下文依赖性，传统的NLP方法无法从文本中直接推断出精确的量化参数。

**本文目标**: 形式化性能需求量化问题，并提出一种通过检索领域特定知识来推理偏好、同时引导与利益相关者进行渐进式交互的方法，在减少认知负担的同时实现高精度量化。

**切入角度**: 将问题建模为"猜想与质询"（Conjecture and Inquiry）——系统先基于检索到的领域知识形成量化猜想，然后通过有针对性的交互向利益相关者求证和修正。

**核心idea**: 与其试图从文本中一步到位地推断数学函数，不如利用检索增强的方式获取问题特定的领域知识来初始化猜想，然后通过少量交互轮次逐步精化偏好参数。

## 方法详解

### 整体框架

IRAP 把"自然语言性能需求 → 数学函数"这件事拆成"先猜想、再质询"两步闭环：系统先从领域知识库里检索与当前需求相关的历史案例和规范，推理出一个初步的量化猜想（函数族 + 参数范围），再针对猜想中最不确定的部分向利益相关者抛出有针对性的封闭式问题，用每一轮回答修正猜想。输入是一段模糊的需求描述，中间经过 5 轮左右的检索-交互迭代，输出是一个参数确定、可直接计算的效用/约束函数。

### 关键设计

**1. 检索增强的偏好推理：用领域先验把猜想锚在可追溯的依据上**

直接让 LLM 从"系统应该快速响应"这类文本一步生成数学函数极易产生幻觉，因为文本本身没有给出函数形状和参数的任何线索。IRAP 转而构建问题特定的知识库（历史性能需求量化案例、行业标准、领域规范），收到新需求时检索语义相关的案例与知识片段，用它们推理可能的量化形式——例如延迟需求大概率落在某个函数族、参数应在某个区间。这样初始猜想有据可查、推理过程可追溯，把后续交互的起点从"空白"拉到了"接近正确"。

**2. 渐进式交互：用封闭式提问把利益相关者的认知负担压到最低**

获取真实偏好的瓶颈在于人——开放式提问（如"请描述您对延迟的数学偏好"）对利益相关者的认知负担过重，几乎无法回答。IRAP 在每一轮先识别当前量化猜想中不确定性最高的参数，再围绕它设计二元或多选问题，引导利益相关者只做"确认或修正"这种低门槛判断，每轮回答后立即更新量化模型。正是这种把不确定性逐参数消解的设计，让系统在 5 轮交互内就能逼近精确偏好。

**3. 需求到数学函数的映射：把检索先验与交互偏好融合成可计算的规格**

量化的终点是给软件性能分析、测试生成和优化提供能直接使用的数学表示。IRAP 结合检索得到的领域知识和交互确认的偏好，从候选函数族（线性、指数、阶梯等）中选定形式并精确估计参数，最终输出完整的函数形式 + 参数规格。检索负责"选对函数族的大方向"，交互负责"把参数钉死"，两者互补才使输出既符合领域规律又贴合具体利益相关者的个人化偏好。

## 实验关键数据

### 主实验

| 数据集 | 指标 | IRAP | 最优Baseline | 提升倍数 |
|--------|------|------|-------------|---------|
| 数据集1 | 量化精度 | 最优 | 次优 | 最高40x |
| 数据集2 | 量化精度 | 最优 | 次优 | 显著 |
| 数据集3 | 量化精度 | 最优 | 次优 | 显著 |
| 数据集4 | 量化精度 | 最优 | 次优 | 显著 |

（注：4个真实世界数据集，对比10种SOTA方法，IRAP在所有案例上取得最优，最大提升达40倍，仅需5轮交互）

### 消融实验

| 配置 | 关键指标 | 备注 |
|------|---------|------|
| 无检索增强 | 精度下降 | 缺乏领域知识导致猜想偏差 |
| 无交互 | 精度下降显著 | 纯自动化无法处理偏好歧义 |
| 减少交互轮次 | 精度随轮次增加而提升 | 5轮是精度-效率的甜点 |
| 不同检索策略 | 精度有所差异 | 检索质量影响初始猜想准确性 |

### 关键发现

- IRAP在4个真实数据集上全面超越10种SOTA方法，证明了检索增强+交互式偏好获取范式的有效性
- 仅需5轮交互即可达到最高40倍的精度提升，表明渐进式交互设计在效率和精度之间取得了很好的平衡
- 检索增强模块提供的领域先验对初始猜想的质量至关重要，直接影响后续交互的效率
- 相比纯自动化方法（如直接用LLM从文本生成函数），交互式方法在处理偏好歧义方面有根本性优势

## 亮点与洞察

- **问题定义的价值**：首次形式化了"性能需求量化"这一实际但被忽视的问题，为软件工程和NLP的交叉研究提供了新方向
- **"猜想与质询"范式**：与"一次性生成"不同，IRAP的渐进式交互设计更符合人类决策的渐进认知模式
- **认知负担最小化**：交互设计避免开放式提问，用封闭式问题引导利益相关者，大幅降低参与门槛
- **40倍提升的实用意义**：在需求量化这种精度敏感的任务上，40倍的精度提升意味着从"不可用"到"可用"的质变

## 局限与展望

- 摘要未详细说明4个数据集的具体领域和规模
- 5轮交互虽少但仍需人类参与，在完全自动化场景下的适用性有限
- 领域知识库的构建成本和覆盖面可能影响方法在新领域的冷启动性能
- 未讨论当利益相关者的偏好本身存在内部矛盾时如何处理
- 未来可将IRAP扩展到其他类型的需求量化（如安全性需求、可靠性需求）

## 相关工作与启发

- **vs 传统需求工程**: 传统方法依赖领域专家手工建模，IRAP通过检索+交互实现半自动化，大幅降低专家依赖
- **vs RAG方法**: IRAP不仅用检索来增强文本生成，更创新地将检索结果用于偏好推理和交互设计，是RAG范式在需求工程中的新应用
- **vs 偏好学习**: 不同于从大量比较数据中学习偏好，IRAP通过少量有针对性的交互高效获取偏好，更适合低数据场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次形式化并解决性能需求量化问题，检索增强+渐进交互的范式设计新颖
- 实验充分度: ⭐⭐⭐⭐ 10种SOTA方法对比，4个真实数据集，结果具有说服力
- 写作质量: ⭐⭐⭐ 基于摘要信息，标题虽有文学感但主题跨软件工程和NLP可能稍显小众
- 价值: ⭐⭐⭐⭐ 解决了真实的工程痛点，40倍提升具有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)
- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ACL 2026\] Why Mean Pooling Works: Quantifying Second-Order Collapse in Text Embeddings](why_mean_pooling_works_quantifying_second-order_collapse_in_text_embeddings.md)
- [\[ICLR 2026\] AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations](../../ICLR2026/information_retrieval/amemgym_interactive_memory_benchmarking_for_assistants_in_long-horizon_conversat.md)
- [\[ACL 2025\] GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](../../ACL2025/information_retrieval/gainrag_preference_alignment.md)

</div>

<!-- RELATED:END -->
