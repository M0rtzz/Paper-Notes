---
title: >-
  [论文解读] NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation
description: >-
  [AAAI 2026][图学习][NOTAM解析] 提出 NOTAM-Evolve，一个自进化框架，通过知识图谱增强的表格检索（KG-TableRAG）进行动态知识接地，结合迭代SFT+DPO偏好优化及多视角投票推理机制，使7B参数LLM自主掌握复杂航空NOTAM的深层解析，准确率较基础LLM提升30.4%。
tags:
  - AAAI 2026
  - 图学习
  - NOTAM解析
  - 知识图谱增强检索
  - 大语言模型
  - 自进化优化
  - 航空安全
---

# NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation

**会议**: AAAI 2026  
**arXiv**: [2511.07982](https://arxiv.org/abs/2511.07982)  
**代码**: [https://github.com/Estrellajer/NOTAM-Evolve](https://github.com/Estrellajer/NOTAM-Evolve)  
**领域**: 图学习 / NLP应用  
**关键词**: NOTAM解析, 知识图谱增强检索, 大语言模型, 自进化优化, 航空安全

## 一句话总结

提出 NOTAM-Evolve，一个自进化框架，通过知识图谱增强的表格检索（KG-TableRAG）进行动态知识接地，结合迭代SFT+DPO偏好优化及多视角投票推理机制，使7B参数LLM自主掌握复杂航空NOTAM的深层解析，准确率较基础LLM提升30.4%。

## 研究背景与动机

### 问题背景

**NOTAM（Notice to Airmen）** 是航空当局发布的官方公告，通知飞行员和空管人员有关空域结构、机场设施或飞行程序的时效性变化。全球每年发布超过100万条NOTAM。NOTAM使用高度压缩的电报式语言，含大量专用缩写和非标准语法，准确解读对飞行安全至关重要。

### 浅层解析 vs 深层解析

**现有系统局限**：当前自动化系统（基于正则表达式、传统NER等）仅能进行**浅层解析（Shallow Parsing）**——提取表面信息，无法获取决策所需的可操作情报。

作者提出**深层解析（Deep Parsing）** 概念，这是一个双重推理挑战：

**动态知识接地（Dynamic Knowledge Grounding）**：NOTAM不是自包含的，需要将文本引用关联到外部的、不断更新的航空基础设施知识库。例如，机场代码ZBAA需要检索该机场的跑道配置（如RWY 09L）

**基于模式的推理（Schema-Based Inference）**：推导NOTAM的真实含义需要超越原始文本提取的模式推理。例如，"...REDUCED LENGTH OF 300M" 提供的是原始参数，需要应用ICAO规则推断 300 米系统构成"基本进近灯光系统（BALS）"

### 核心动机
- 浅层解析无法满足运营安全需求
- 深层解析需要同时具备知识接地和规则推理能力
- LLM具有强大的语义理解能力，但需要领域知识增强和迭代优化才能胜任
- 航空领域的安全和成本约束通常不允许使用闭源商用API，需要可部署的开源方案

## 方法详解

### 整体框架

NOTAM-Evolve 包含三个核心阶段：
1. **知识接地检索（Knowledge-Grounded Retrieval）**：KG-TableRAG 将预测接地于航空领域知识
2. **自优化模型精炼（Self-Optimizing Model Refinement）**：通过迭代SFT+DPO实现自进化
3. **多视角推理（Multi-View Inference）**：通过改写+投票机制确保稳健解析

### 关键设计

#### 1. **KG-TableRAG 知识接地检索**：解决动态知识接地挑战

**设计动机**：航空数据是动态更新的（机场设施状态、跑道可用性等存储在定期更新的表格中）。传统检索在航空领域不够，因为表格列和数据含义往往是隐式的——"跑道关闭"查询可能无法检索到相关的灯光系统或导航设备信息。

**工作流程**：
1. LLM 接收原始NOTAM，生成 Cypher 查询搜索知识图谱
2. 知识图谱返回结构化领域知识（如机场-跑道从属关系）
3. 将图谱查询结果与原始查询拼接，形成增强查询
4. 用增强查询从操作表格中检索最相关信息
5. 检索信息与原始NOTAM组合，作为LLM的最终输入

**关键优势**：知识图谱提供了传统表格检索缺少的结构化真实世界知识，弥补了隐式关系的缺失。

#### 2. **自优化模型精炼**：通过闭环学习实现自进化

这是框架的核心，包含交替进行的SFT和DPO两个阶段。

**初始化**：
- 数据集 $\mathcal{D}_0 = \{(x \circ K, Y^*)\}$ 按8:2划分训练/测试集
- 基础模型 $\pi_0$（DeepSeek-R1-Distill-Qwen-7B）
- 初始化空的响应池 $\mathcal{R}$

**迭代优化循环**（每轮迭代 $e$）：

**Step 1: 生成与评估**
- 用当前模型 $\pi_e$ 对训练集生成响应 $\hat{Y}^{(e)}$
- 与真实标注 $Y^*$ 比较，标记正确/错误
- 计算每个输入的错误率（回溯窗口 $K'$）：

$$\xi(x) = \frac{\sum_{k=1}^{K'} \mathbb{I}(\hat{Y}^{(k)} \neq Y^*)}{K'}$$

**Step 2: SFT阶段**
- 从响应池 $\mathcal{R}$ 提取正确的输入-输出对构成 SFT 数据集
- 标准负对数似然损失微调：

$$\mathcal{L}_{\text{SFT}}^{(e)} = -\mathbb{E}_{(x,Y^*) \sim \mathcal{D}_{\text{SFT}}^{(e)}} \left[\sum_{i=1}^{m} \log \pi_\theta(Y_i^* \mid x \circ K, Y_{<i}^*)\right]$$

**Step 3: DPO阶段**
- 构建偏好数据集：正样本 $y^*$（正确响应）+ 负样本 $y^-$（错误响应）
- **动态数据增强**：高错误率（$\xi(x) \geq \tau$）的输入生成语义保持的变体 $\mathcal{V}_x$
- **加权课程学习**：自适应采样权重从均匀过渡到错误加权：

$$w_e(x) = (1-\alpha_e) \frac{1}{N} + \alpha_e \frac{\exp(\beta_{\text{weight}} \xi(x))}{\sum_{j=1}^{N} \exp(\beta_{\text{weight}} \xi(x_j))}$$

其中 $\alpha_e = \min(e/E, 1)$ 是课程进度

- DPO损失优化：

$$\mathcal{L}_{\text{DPO}}^{(e)} = -\mathbb{E} \left[\log \sigma\left(\beta_{\text{DPO}} \log \frac{\pi_\theta(y^*|x)}{\pi_{\text{ref}}(y^*|x)} - \beta_{\text{DPO}} \log \frac{\pi_\theta(y^-|x)}{\pi_{\text{ref}}(y^-|x)}\right)\right]$$

**收敛条件**：测试集准确率达到目标 $\eta$。实验中3-5轮迭代即可达到商用SOTA水平。

#### 3. **多视角推理（改写+投票）**：提升推理稳定性

**设计动机**：标准解析范式在边缘案例中产生不一致预测。基线模型虽展现出部分理解能力，但推理路径的微小变化可能决定正确与否。

**实现**：
1. 通过控制改写生成 $N=5$ 个语义等价的NOTAM变体（保持航空术语、时空约束和安全关键数值不变）
2. 每个变体独立处理得到候选结构化输出
3. 多数投票确定最终预测：

$$\hat{Y}_{\text{final}} = \arg\max_{Y} \sum_{k=1}^{N} \mathbb{I}(Y = \hat{Y}^{(k)})$$

改写机制包括：词汇替换（如"CTAM" ↔ "Controller Advisory Message"）、句法重构（语态转换）、上下文扩展（ICAO术语澄清）。

### 损失函数 / 训练策略

- 使用 Unsloth 框架进行高效微调
- 基于 DeepSeek-R1-Distill-Qwen-7B（7B参数）
- 单卡 NVIDIA A800-80GB 进行所有实验
- 迭代优化3轮，总计算时间0.58h → 1.5h → 3.2h
- 三种机制抑制理论 $O(t^2)$ 扩展到实际2.3×平均增长

## 实验关键数据

### 数据集
新构建的NOTAM基准数据集：10,000条专家标注样本，覆盖全球分布，分4个子集：Light(1,000)、Area(4,000)、Runway(2,500)、Taxiway(2,500)。标注者间一致性 Krippendorff's Alpha = 0.96。

### 主实验

| 模型 | Light | Area | Runway | Taxiway | AVG |
|------|-------|------|--------|---------|-----|
| Regex模板匹配 | 0.370 | 0.491 | 0.443 | 0.396 | 0.425 |
| UIE | 0.270 | 0.380 | 0.320 | 0.430 | 0.350 |
| Qwen2.5-7B | 0.560 | 0.777 | 0.412 | 0.748 | 0.624 |
| DeepSeek-R1-7B (base) | 0.410 | 0.484 | 0.446 | 0.492 | 0.458 |
| Qwen2.5-7B (SFT) | 0.590 | 0.793 | 0.730 | 0.864 | 0.744 |
| **NOTAM-Evolve** | **0.620** | 0.725 | **0.836** | **0.868** | **0.762** |
| GPT-4o | 0.605 | **0.851** | 0.770 | 0.914 | 0.785 |
| DeepSeek-R1 (full) | **0.725** | **0.871** | 0.792 | **0.924** | **0.828** |

NOTAM-Evolve 相比基础模型提升 **30.4%**（0.458 → 0.762），AVG接近GPT-4o水平。

### 消融实验

| KG-TableRAG | Multi-View | AVG |
|-------------|-----------|-----|
| ✓ | ✓ | **0.762** |
| ✓ | ✗ | 0.721 |
| ✗ | ✓ | 0.740 |
| ✗ | ✗ | 0.690 |

迭代自优化各类别性能变化：

| 类别 | Iter 1 | Iter 2 | Iter 3 |
|------|--------|--------|--------|
| Light | 45.0% | 57.5% | 62.0% |
| Taxiway | 64.6% | 80.4% | 86.8% |

### 关键发现

1. **自进化效果显著**：3轮迭代从45%→62%（Light）、64.6%→86.8%（Taxiway），证明闭环学习的有效性
2. **SFT单独使用可能有害**：DeepSeek-R1-Distill-Qwen-7B单纯SFT后性能从0.458**骤降至0.212**，因为缺乏CoT推理链的微调会损害推理能力
3. **多视角推理贡献最大**：移除Multi-View导致4.1%性能下降，大于KG-TableRAG的2.2%
4. **7B模型逼近商用大模型**：NOTAM-Evolve性能接近GPT-4o（0.762 vs 0.785），这对航空领域的安全和成本约束下的实际部署至关重要
5. **案例验证**：机场AGGC关闭 → 正确推理出跑道RWY 07R也关闭（因为属于该机场），普通基线无法完成此推理

## 亮点与洞察

1. **问题定义清晰**：将NOTAM解析明确界定为"深层解析"双重推理挑战，区别于浅层信息提取，问题定义本身就是贡献
2. **自进化范式实用**：不需要大量人工标注推理链，模型从自身输出中学习，显著降低了领域适配成本
3. **课程学习的巧妙应用**：采样权重从均匀到错误加权的渐进过渡，让模型先学简单样本再学困难样本
4. **航空安全的实际价值**：7B模型可本地部署，解决了航空领域对闭源API的安全和成本顾虑
5. **数据集贡献**：10,000条专家标注、高一致性（α=0.96）的NOTAM数据集本身对社区有独立价值

## 局限与展望

1. **计算成本随迭代增长**：尽管有三种机制抑制，迭代优化的偏好对数仍近似 $O(t^2)$ 增长
2. **NOTAM标注的固有困难**：即使专家标注也难以保证完美准确，可能限制性能上限
3. **仅支持英文NOTAM**：未涉及多语言场景（实际NOTAM存在地区变体）
4. **知识图谱需要人工维护**：航空基础设施数据的更新频率和覆盖度未详细讨论
5. **改进方向**：LLM辅助标注+专家验证、更高效的优化策略、多语言扩展、实时运行场景

## 相关工作与启发

- **TableRAG** 提供基础表格检索能力，NOTAM-Evolve用知识图谱增强它处理隐式航空关系
- **DPO偏好优化** 结合课程学习，在领域适配中展现出比单纯SFT更好的效果
- 自进化框架的思路可迁移到**医疗报告解析、法律文件理解**等需要领域知识的高精度NLP任务
- 改写+投票的多视角推理是提升LLM在结构化输出任务稳定性的通用策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 自进化框架设计巧妙，但各组件（KG检索、DPO、投票）本身并非全新
- **实验充分度**: ⭐⭐⭐⭐ — 新数据集+完整消融+迭代分析+案例+复杂度分析
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题定义清晰，Deep vs Shallow Parsing的框架很好
- **实用价值**: ⭐⭐⭐⭐⭐ — 航空安全领域真实需求，7B可部署方案，开源数据集

<!-- RELATED:START -->

## 相关论文

- [Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing](assessing_llms_for_serendipity_discovery_in_knowledge_graphs_a_case_for_drug_rep.md)
- [Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [Self-Adaptive Graph Mixture of Models](self-adaptive_graph_mixture_of_models.md)
- [Explore-on-Graph: Incentivizing Autonomous Exploration of LLMs on Knowledge Graphs](../../ICLR2026/graph_learning/explore-on-graph_incentivizing_autonomous_exploration_of_large_language_models_o.md)
- [Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data](format_as_a_prior_quantifying_and_analyzing_bias_in_llms_for_heterogeneous_data.md)

<!-- RELATED:END -->
