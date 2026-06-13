---
title: >-
  [论文解读] SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis
description: >-
  [ACL 2025][NLP理解][流式情感分析] 提出SynGraph框架，在连续时间动态图上将稀疏用户分为mid-tail/long-tail/extreme三类，针对不同稀疏程度利用LLM合成增强数据（结合局部-全局图理解、高阶关系和画像生成），有效缓解流式评论情感分析中的数据稀疏问题。 问题背景 电商平台用户评论蕴含…
tags:
  - "ACL 2025"
  - "NLP理解"
  - "流式情感分析"
  - "数据稀疏"
  - "动态图"
  - "LLM数据增强"
  - "用户建模"
---

# SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis

**会议**: ACL 2025  
**arXiv**: [2503.04619](https://arxiv.org/abs/2503.04619)  
**作者**: Xin Zhang (Manchester), Qiyu Wei (Manchester), Yingjie Zhu (HIT), Linhai Zhang (KCL), Deyu Zhou (SEU), Sophia Ananiadou (Manchester)
**代码**: 未公开  
**领域**: NLP理解  
**关键词**: 流式情感分析, 数据稀疏, 动态图, LLM数据增强, 用户建模

## 一句话总结

提出SynGraph框架，在连续时间动态图上将稀疏用户分为mid-tail/long-tail/extreme三类，针对不同稀疏程度利用LLM合成增强数据（结合局部-全局图理解、高阶关系和画像生成），有效缓解流式评论情感分析中的数据稀疏问题。

## 研究背景与动机

### 问题背景
电商平台用户评论蕴含丰富的情感动态变化信息——用户对某类商品的情感评分可能随时间从正面逐渐转为中性。传统情感分析方法将每条评论视为独立样本进行静态处理，忽略了评论序列中的时序依赖。流式评论情感分析任务正是为了建模这种时序演变并预测未来情感。

### 已有工作的不足
- **静态方法忽视时序**：传统情感分类模型不考虑用户评论间的时间依赖关系
- **数据稀疏问题严重**：在流式场景中，稀疏性表现为三种形式——时序稀疏（用户评论间隔大）、空间稀疏（用户社交连接少）、以及两者兼有的复合稀疏
- **图结构引入后稀疏更复杂**：已有工作用图结构补充稀疏用户信息，但在流式场景下，稀疏用户的分类和处理变得更加困难
- **现有LLM增强方法有限**：已有LLM推荐系统增强方法多只利用一阶邻居信息，未考虑高阶关系和时序动态

### 核心动机
设计一个统一框架，对不同稀疏程度的用户采取差异化策略，联合利用动态图结构和LLM生成能力来合成高质量补充数据，从而提升流式评论情感预测效果。

## 方法详解

### 整体框架
SynGraph是一个三阶段框架：
1. **稀疏用户分类（Categorization）**：根据用户的时序和结构特征将其分为mid-tail、long-tail和extreme三类
2. **LLM增强数据合成（Synthesis）**：针对不同稀疏类别，利用LLM结合图结构信息生成合成数据
3. **类别插值（Interpolation）**：将合成数据插入用户的空白时间段，确保每个用户至少拥有10条交互记录

底层建模基于连续时间动态图（CTDG），用户和商品为节点，评论为带时间戳的有向边，每条边附带评论文本和情感评分。

### 关键设计1：稀疏用户三级分类

根据用户的时序活跃度和图结构连接性，将用户分为三类：

- **Mid-tail用户**：在特定时间段内有一定活跃度，但评论频率波动大。通过计算交互方差 $\sigma^2_{I(u_m)}$ 捕捉这种不稳定性，方差高说明用户参与度随时间起伏
- **Long-tail用户**：评论总量少且一阶邻居（直接连接）稀少，但通常拥有较多二阶连接。定义一阶连接强度 $C_1(u_l)$ 为直接邻居的权重和，该值小但二阶连接 $C_2(u_l)$ 较大
- **Extreme用户**：在评论活动和图连接两个维度上都极度稀疏，几乎是孤立节点。总连接度 $C(u_e) \approx 0$

这种分类使得后续数据合成可以为不同类型用户量身定制策略——mid-tail用户有足够历史评论可供LLM生成画像，long-tail用户需要借助二阶邻居的间接信息，extreme用户则只能依赖全局高评分商品作为代理。

### 关键设计2：LLM增强数据合成的三组件

框架通过三个核心组件支撑数据合成：

**(1) 局部-全局图理解**：从局部邻居获取近期直接交互信息，从全局邻居捕获长期趋势和整体网络上下文。LLM基于两个层面的信息丰富用户画像。

**(2) 高阶关系理解**：不止依赖一阶连接，还提取二阶邻居关系 $\mathcal{N}^2(u) = (\bigcup_{v \in \mathcal{N}(u)} \mathcal{N}(v)) \setminus \{u\}$，通过间接信息传播弥补直接连接的不足。

**(3) 画像生成**：通过LLM和特定prompt从评论中为用户和商品生成合成画像：$\mathcal{P}(x) = \text{LLM}(\mathcal{R}(x), P_x)$，分别使用用户prompt $P_u$、商品prompt $P_{\text{prod}}$ 和数据合成prompt $P_{\text{data}}$。

三组件在不同稀疏类别中差异化组合：
- **Mid-tail合成**：直接从历史评论生成用户画像 + 二阶商品画像 → LLM合成新评论
- **Long-tail合成**：结合局部和全局二阶邻居信息 + 精选评论生成用户画像 + 二阶商品画像 → LLM合成
- **Extreme合成**：使用预定义用户画像 + 高评分热门商品作代理 → LLM合成伪评论

### 关键设计3：插值机制

通过插值因子 $I_f(u) = \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}_{\{I_t(u)=0\}}$ 量化用户在各时间窗口的缺失程度，在缺失位置插入合成数据，确保每个用户维持最低10条记录的数据可用性阈值。

## 实验关键数据

### 实验设置
- **数据集**：Amazon数据集的三个小规模子集——Magazine_Subscriptions、Appliances、Gift_Cards
- **评估指标**：Accuracy、Precision、Recall、F1、MSE、RMSE、MAE
- **基线模型**：BiLSTM+Att、BERT-Sequence、CHIM、IUPC、NGSAM、DC-DGNN

### 表1：主实验RMSE下降率（加*表示使用SynGraph增强）

| 模型 | Magazine | Appliances | Gift_Cards |
|------|----------|------------|------------|
| BiLSTM+Att → +SynGraph | ↓3.93% | ↓48.61% | ↓0.12% |
| BERT-Sequence → +SynGraph | ↓36.80% | ↓4.36% | ↓2.57% |
| NGSAM → +SynGraph | ↓14.38% | ↓14.94% | ↓21.73% |
| CHIM → +SynGraph | ↑10.52% | ↓23.92% | ↑40.89% |
| IUPC → +SynGraph | ↓22.16% | ↓4.36% | — |
| DC-DGNN → +SynGraph | **↓26.42%** | **↓20.22%** | **↓38.88%** |

18组实验中，15组在SynGraph增强后取得显著提升。DC-DGNN在增强前后始终表现最优，突出了动态用户-商品建模的重要性。

### 表2：消融实验（DC-DGNN在三个数据集上的RMSE）

| 变体 | Magazine | Appliances | Gift_Cards |
|------|----------|------------|------------|
| DC-DGNN* (三类联合) | **0.6485** | **0.8165** | **0.3383** |
| DC-DGNN-M (仅mid-tail) | 0.6551 | 1.0235 | 0.3805 |
| DC-DGNN-L (仅long-tail) | 0.7324 | 0.7868 | 0.5535 |
| DC-DGNN-E (仅extreme) | 0.6385 | 1.0235 | — |

联合三类合成数据通常效果最佳；单独使用某一类有时会引入噪声或加剧标签不平衡。

### GPT-4合成数据质量评估
通过语言风格相似度(LSS)、评分习惯相似度(RHS)、情感相似度(SS)、方面相似度(AS)四个维度评估：mid-tail用户合成质量最高，extreme用户的LSS和RHS较低但SS和AS保持稳定，说明LLM能有效维持产品级别的语义连贯性。

## 关键发现

1. **15/18配置有效**：SynGraph增强在绝大多数模型-数据集组合上带来性能提升，验证了框架的通用性
2. **DC-DGNN最佳搭档**：专为连续动态图设计的DC-DGNN与SynGraph配合效果最突出（三个数据集RMSE分别降低26%、20%、39%），说明动态图建模是流式情感分析的关键
3. **三类联合优于单类**：联合使用所有稀疏类别的合成数据通常效果最佳，但在极小规模数据上可能引入额外噪声
4. **LLM合成的词汇多样性不足**：合成文本的词汇丰富度一致低于原始数据，表明LLM存在词汇同质化倾向

## 亮点与洞察

- **分类-合成-插值的三阶段设计**很实用：将稀疏问题分解为可操作的子问题，每类用户有定制化的增强方案，而非一刀切
- **图结构+LLM的融合思路**有启发性：不是简单地让LLM生成数据，而是让LLM理解图的局部-全局结构和高阶关系，这为图增强学习提供了新范式
- **极端稀疏用户的处理策略**值得借鉴：对几乎无数据的用户，通过热门商品做代理来实现"冷启动"，思路简洁有效

## 局限性

- **数据集规模过小**：三个Amazon子数据集（总用户不足500人）都是极小规模，在大规模真实场景中的可扩展性未知
- **随机邻居采样策略粗糙**：选取邻居时采用随机采样，在大规模图上采样方差可能显著影响效果
- **LLM幻觉风险**：合成数据可能不忠实反映真实分布，论文未设计专门的质量过滤机制
- **LLM对图结构的理解不透明**：未分析LLM如何区分和利用局部vs全局图信息
- **少数配置性能下降**：CHIM在Magazine和Gift_Cards上增强后性能反降（RMSE分别上升10.52%和40.89%），暴露了方法对特定模型-数据组合的脆弱性
- **标签严重不均衡**：Gift_Cards数据集5分评分占比超92%，合成数据可能加剧不平衡

## 相关工作与启发

- **数据稀疏缓解**：传统方法通过用户属性、社交网络、跨域迁移或元学习处理稀疏，但均为静态方案
- **LLM增强推荐**：LLMRec (Wei et al. 2023) 用一阶邻居信息增强，Sun et al. 2023 从文本推断用户画像，但都不考虑高阶关系和时序
- **流式情感分析**：SentiStream (Wu et al. 2023) 处理数据流但不处理稀疏问题；DC-DGNN (Zhang et al. 2023) 用连续动态图建模但无数据增强
- **启发**：将LLM作为数据增强引擎嵌入图学习流程是有前景的方向，特别是在用户冷启动和长尾分布场景下

## 评分

- 新颖性: ⭐⭐⭐⭐ — 稀疏用户三级分类+LLM增强动态图是新组合，但各组件并非首创
- 实验充分度: ⭐⭐⭐ — 6个基线、消融、质量评估较全面，但数据集太小（用户<500），可信度受限
- 写作质量: ⭐⭐⭐⭐ — 框架描述清晰，公式化表达完整，图示直观
- 价值: ⭐⭐⭐ — 框架思路有启发性，但小数据集实验和部分配置的性能下降削弱了说服力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis](dot_absa_template.md)
- [\[ACL 2025\] A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment](a_comprehensive_graph_framework_for_question_answering_with_mode-seeking_prefere.md)
- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [\[ACL 2025\] Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](embqa_embedding_odqa.md)
- [\[ACL 2025\] Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)

</div>

<!-- RELATED:END -->
