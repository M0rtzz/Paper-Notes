---
title: >-
  [论文解读] An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling
description: >-
  [ACL 2025][课程学习] 提出面向序列标注任务的双阶段课程学习（DCL）框架，通过数据级与模型级两阶段由易到难训练策略，配合基于贝叶斯不确定性的 token 级动态难度度量和 Root 函数训练调度器，在 CWS、POS、NER 三类任务上实现性能提升与训练加速超 27% 的双重收益。
tags:
  - ACL 2025
  - 课程学习
  - 序列标注
  - 贝叶斯不确定性
  - 中文分词
  - 命名实体识别
---

# An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling

**会议**: ACL 2025  
**arXiv**: [2402.13534](https://arxiv.org/abs/2402.13534)  
**代码**: [GitHub](https://github.com/tangxuemei1995/DCL4SeqLabeling)  
**领域**: 序列标注 / 课程学习  
**关键词**: 课程学习, 序列标注, 贝叶斯不确定性, 中文分词, 命名实体识别

## 一句话总结

提出面向序列标注任务的双阶段课程学习（DCL）框架，通过数据级与模型级两阶段由易到难训练策略，配合基于贝叶斯不确定性的 token 级动态难度度量和 Root 函数训练调度器，在 CWS、POS、NER 三类任务上实现性能提升与训练加速超 27% 的双重收益。

## 研究背景与动机

**核心问题**：序列标注模型引入外部知识（词典、n-gram、句法）后，数据异质性增加、模型复杂度飙升，训练成本高昂，如何在保持甚至提升性能的同时加速复杂模型的训练？

**外部知识的双刃剑**：近年来序列标注性能的提升大量依赖整合词典（lexicons）、n-gram、句法树等外部知识，但这带来了输入数据异构化和额外编码模块（注意力机制、GNN）的参数膨胀问题，使得训练一个高性能模型的计算代价急剧增加。

**课程学习的潜力与不足**：课程学习（CL）通过模拟人类"由易到难"的学习过程，已在机器翻译、对话生成、文本分类等任务中取得成功。然而现有 CL 的难度度量主要集中于句子级（如句子长度、句子级置信度），缺乏针对序列标注任务的 token 级和词级难度度量，无法精确捕捉标注任务中"哪些 token 最难标"的信号。

**静态课程的局限**：传统 CL 在训练前确定样本难度排序后不再更新，但模型能力随训练不断变化——初始"困难"的样本在模型学会一定能力后可能变"简单"，静态排序无法适应模型的学习轨迹。

**冷启动困境**：直接对学生模型做课程学习存在冷启动问题——初始能力弱的模型无法可靠评估样本难度，导致课程排序失准。

## 方法详解

### 整体框架

DCL 框架由三个核心组件构成：教师序列标注模型（RoBERTa + Softmax）、学生序列标注模型（McASP 或 SynSemGCN 等复杂模型）和双阶段课程学习训练策略。DCL 独立于具体序列标注模型架构，可即插即用地与任意编码器-解码器组合使用。

| 阶段 | 目标 | 输入 | 输出 |
|------|------|------|------|
| 数据级 CL | 初始化样本难度排序，缓解冷启动 | 全部训练数据 $\mathcal{D}$ | 按难度升序排列的 $\mathcal{D}_r$ |
| 模型级 CL | 动态扩展训练集，自适应学习轨迹 | 排序后子集 $\mathcal{D}_s$ + 剩余 $\mathcal{D}_o$ | 训练至收敛的学生模型 $\theta$ |

**数据级课程学习（第一阶段）**：在全部训练数据 $\mathcal{D}$ 上训练一个基础教师模型 $\theta_0$，训练 $E_0$ 个 epoch（少于收敛所需），用 $\theta_0$ 对所有样本计算难度分数 $S(\theta_0)$，按难度升序排序形成 $\mathcal{D}_r$。教师模型的作用是提供一个合理的初始难度排序，避免学生模型冷启动时面对随机顺序的数据。

**模型级课程学习（第二阶段）**：用 $\mathcal{D}_r$ 中最简单的 $\lambda_0$ 比例样本初始化学生训练集 $\mathcal{D}_s$。每个 epoch 后用当前学生模型 $\theta_*$ 重新评估剩余数据 $\mathcal{D}_o$ 的难度并重排序，根据 Root 函数调度器递增 $\lambda$ 扩展训练集。当 $\lambda = 1$ 时使用全部数据训练至收敛。

### 关键设计

**1. 贝叶斯不确定性（BU）token 级难度度量**

基于 Monte Carlo dropout 近似贝叶斯推断，将难度度量从句子级细化到 token 级。对每个 token $x_i$ 执行 $K$ 次随机前向传播（每次随机关闭不同的 dropout 神经元），得到 $K$ 个预测分布 $P(y_i|x_i)_1, \ldots, P(y_i|x_i)_K$。计算每个 token 在标签集 $T$ 上的概率方差：

$$var(x_i, \theta) = \sum_{y_i \in T} \left( \frac{1}{K} \sum_{k=1}^K P(y_i|x_i)_k^2 - \mathbb{E}[P(y_i|x_i)]^2 \right)$$

最终句子级难度分数同时考虑序列中最不确定的位置和整体不确定性水平：$S(\theta)^{BU} = var(\theta)_{max} + var(\theta)_{aver.}$，其中 $var(\theta)_{max}$ 捕获最难标注的 token（如罕见实体边界），$var(\theta)_{aver.}$ 反映整句标注的平均难度。消融实验表明两者对性能贡献相当，缺一不可。

**2. Root 函数训练调度器**

控制每个 epoch 加入的新样本比例 $\lambda$ 的增长节奏：

$$\lambda = \min\left(1, \sqrt{\frac{1-\lambda_0^2}{E_{grow}} \cdot t + \lambda_0^2}\right)$$

其中 $\lambda_0$ 为初始训练集比例，$E_{grow}$ 为 $\lambda$ 增长到 1 所需的 epoch 数，$t$ 为当前 epoch。Root 函数的特性是初期增长较快（快速让模型接触中等难度样本），后期增长放缓（给模型充分消化最难样本的时间），相比线性调度器更符合学习由快到慢的规律。

**3. 模型级动态难度重评估机制**

与静态课程学习的关键区别在于：DCL 在模型级阶段的每个 epoch 都用当前学生模型 $\theta_*$ 对剩余数据 $\mathcal{D}_o$ 重新计算难度并重排序。随着学生模型能力逐步增强，样本的相对难度会发生变化——之前被判定为"困难"的样本可能因为模型已掌握相关模式而变得"简单"。动态重评估使得每轮新加入的样本能够精确匹配模型当前的学习前沿（learning frontier），实现真正的自适应课程学习。

## 实验关键数据

### 主实验：CWS 与 POS 联合标注

在 CTB5、CTB6、PKU 三个中文数据集上，以 SynSemGCN 为学生模型，对比无 CL 基线和不同难度度量的 DCL 设置：

| 模型 / CL 设置 | CTB5 CWS | CTB5 POS | CTB6 CWS | CTB6 POS | PKU CWS | PKU POS |
|----------------|----------|----------|----------|----------|---------|---------|
| SynSemGCN（无 CL） | 98.83 | 96.77 | 97.86 | 94.98 | 98.05 | 95.50 |
| + DCL (Random) | 98.84 | 97.86 | 97.99 | 95.05 | 98.48 | 96.40 |
| + DCL (Length) | 98.80 | 96.84 | 97.40 | 94.94 | 98.53 | 96.48 |
| + DCL (TLC) | 98.83 | 97.81 | 97.98 | 95.02 | 98.61 | 96.55 |
| + DCL (MNLP) | 98.78 | 97.72 | 98.04 | 95.13 | 98.56 | 96.48 |
| **+ DCL (BU)** | **98.90** | **97.95** | **98.05** | **95.14** | **98.59** | **96.54** |

### 消融实验与训练效率

| 设置 | CTB5 CWS | CTB5 POS | 训练时间 |
|------|----------|----------|----------|
| SynSemGCN + DCL (BU) 完整 | 98.90 | 97.95 | 287 min |
| 去除数据级 CL | 98.90 | 97.88 | — |
| 去除模型级 CL | 98.85 | 97.51 | — |
| 去除 DCL（原始 SynSemGCN） | 98.75 | 96.73 | 393 min |

BU 难度度量组件消融（McASP 模型，CTB5）：

| 设置 | CWS F1 | POS F1 |
|------|--------|--------|
| McASP + DCL (BU) 完整 | 98.91 | 96.87 |
| 去除 $var(\theta)_{max}$ | 98.78 | 96.78 |
| 去除 $var(\theta)_{aver.}$ | 98.86 | 96.74 |
| McASP（无 CL） | 98.73 | 96.60 |

### NER 泛化实验

| 模型 | Weibo（中文） | OntoNotes4（中文） | CoNLL-2003（英语） |
|------|-------------|-------------------|-------------------|
| BERT（无 CL） | 66.22 | 79.15 | 90.94 |
| BERT + CL (Length) | 66.81 | 79.63 | 90.79 |
| BERT + DCL (BU) | **66.74** | **80.02** | **91.77** |

### 关键发现

1. **BU 度量一致最优**：在绝大多数数据集上 BU 优于 TLC、MNLP、句子长度和随机排序，表明基于不确定性的度量比启发式度量更能捕捉序列标注的真实难度
2. **训练加速 27%**：DCL 将 SynSemGCN 的训练时间从 393 分钟缩短至 287 分钟，性能反而更优
3. **模型级 CL 贡献更大**：去除模型级 CL 导致 POS F1 下降 0.44，而去除数据级 CL 仅下降 0.07，因为模型级 CL 影响全训练过程而数据级仅影响初期排序
4. **BU 两组件互补**：去除 $var(\theta)_{max}$ 使 CWS F1 从 98.91 降至 98.78，去除 $var(\theta)_{aver.}$ 降至 98.86，两者贡献相当
5. **跨任务泛化**：DCL 在 CWS、POS、NER（中英文）三类六个数据集上均有效，验证了框架的通用性

## 亮点与洞察

1. **双阶段分工明确**：数据级 CL 用轻量教师模型提供粗粒度初始排序解决冷启动，模型级 CL 用学生模型自身进行细粒度动态重排序实现自适应——两阶段互补而非重复
2. **token 级度量填补空白**：既往 CL 难度度量停留在句子级，而序列标注的核心挑战在于少数难标注 token，BU 度量通过 max + avg 的组合精准捕捉了这一特征
3. **模型无关的即插即用设计**：DCL 不修改模型架构，仅改变训练数据的呈现顺序，可与任意序列标注模型搭配使用
4. **性能与效率的双赢**：课程学习的早期阶段在小子集上训练减少了计算量，同时由易到难的渐进式学习又带来了更好的收敛质量

## 局限性

- 难度度量涉及多个超参数（Monte Carlo dropout 次数 $K$、初始比例 $\lambda_0$、增长 epoch 数 $E_{grow}$），调优成本较高
- 仅探索了由易到难的课程策略，未研究 anti-curriculum（由难到易）的效果对比
- 训练调度器仅对比了 Root 函数与线性函数，未探索阶梯式、余弦等更多调度策略
- 实验数据集规模较小（CTB5 约 19K 句），在大规模序列标注场景下的可扩展性有待验证
- NER 泛化实验中仅使用 BERT 作为学生模型，未在复杂 NER 模型上验证 DCL 效果

## 相关工作

- **Bengio et al. (2009)**：课程学习开创性工作，提出由易到难的训练范式
- **Tian et al. (2020b) McASP**：基于多注意力机制融合词典和 n-gram 的序列标注模型，本文基线之一
- **Tang et al. (2024) SynSemGCN**：通过 GCN 融合句法和语义知识的序列标注模型，本文另一基线
- **Gal & Ghahramani (2016)**：Monte Carlo dropout 方法，本文 BU 度量的理论基础
- **Wang et al. (2021)**：课程学习在 NLP 中的综述，梳理了数据选择策略和训练调度器的分类体系

## 评分

- **创新性**: ⭐⭐⭐ — 双阶段 CL 框架和 BU token 级度量有一定新意，但整体属于已有技术的组合与细化
- **实用性**: ⭐⭐⭐⭐ — 性能提升 + 训练加速 27% 的双重收益具有实际工程价值，且模型无关设计易于落地
- **实验充分度**: ⭐⭐⭐⭐ — 多数据集、多模型、完整消融实验和跨任务泛化，但数据集规模偏小
- **写作质量**: ⭐⭐⭐ — 结构清晰但符号较繁琐，部分公式排版可更精简

<!-- RELATED:START -->

## 相关论文

- [Incorporating Domain Knowledge into Materials Tokenization](incorporating_domain_knowledge_into_materials_tokenization.md)
- [Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning](making_llms_better_many-to-many_speech-to-text_translators_with_curriculum_learn.md)
- [Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)
- [Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)

<!-- RELATED:END -->
