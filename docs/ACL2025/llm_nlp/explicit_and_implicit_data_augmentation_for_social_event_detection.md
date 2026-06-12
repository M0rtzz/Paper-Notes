---
title: >-
  [论文解读] Explicit and Implicit Data Augmentation for Social Event Detection
description: >-
  本文提出SED-Aug，一个结合显式（LLM文本增强）和隐式（特征空间扰动）的双重数据增强框架用于社交事件检测，在Twitter2012和Twitter2018上分别超越最优基线17.67%和15.57%的平均F1。
tags:

---

# Explicit and Implicit Data Augmentation for Social Event Detection

| 属性 | 内容 |
|------|------|
| 标题 | Explicit and Implicit Data Augmentation for Social Event Detection |
| 会议 | ACL2025 |
| arXiv | [2509.04202](https://arxiv.org/abs/2509.04202) |
| 代码 | [github.com/congboma/SED-Aug](https://github.com/congboma/SED-Aug) |
| 领域 | Others / Social Event Detection |
| 关键词 | Social Event Detection, Data Augmentation, LLM, Feature Perturbation, Graph Neural Network |

## 一句话总结

本文提出SED-Aug，一个结合显式（LLM文本增强）和隐式（特征空间扰动）的双重数据增强框架用于社交事件检测，在Twitter2012和Twitter2018上分别超越最优基线17.67%和15.57%的平均F1。

## 研究背景与动机

- **社交事件检测（SED）**：从社交媒体中识别和分类重要事件，是危机管理、舆情分析和金融市场分析的重要工具
- **核心挑战**：SED高度依赖标注数据，但标注成本高且费力，限制了模型在多样事件上下文中的泛化能力
- **现有方法局限**：
    - 图方法（如GraphHAM、ETGNN）利用了文本和结构信息，但未充分利用数据增强来提升数据多样性
    - LLM擅长文本增强但不擅处理图数据
    - 纯文本增强忽视了用户和事件交互的结构信息
- **研究空白**：LLM从未被应用于SED任务的数据增强；缺乏同时覆盖文本和结构信息的双重增强框架

## 方法详解

### 整体框架

SED-Aug是一个即插即用的双重数据增强框架，分两阶段：
1. 显式增强：用LLM增强社交消息的文本内容
2. 隐式增强：在特征空间对结构融合嵌入进行扰动

处理流程：原始消息 → LLM文本增强 → 增强消息+原始消息+结构数据 → 预训练语言模型提取嵌入 → 构建社交图 → 图聚合得到结构融合嵌入 → 特征空间扰动 → 分类

### 显式数据增强（5种策略）

通过LLM生成 $m_i^k = LLM^k(m_i)$，其中 $k$ 代表不同增强类型：

| 策略 | 说明 |
|------|------|
| **Paraphrasing** | 在保持语义不变的前提下改写措辞和结构 |
| **Adding Context** | 增加相关上下文信息，丰富消息的理解和清晰度 |
| **Style Transfer** | 修改写作风格（语气、正式程度等）但保持核心含义 |
| **Keep Entity Unchanged** | 修改文本但保持关键实体（人名、地点、日期等）不变 |
| **Extract & Rewrite**（两阶段） | 先用LLM提取关键信息（关键词/实体/知识图谱），再生成新版本消息 |

### 隐式数据增强（5种扰动方法）

对结构融合嵌入 $g^i$ 进行扰动，训练时以概率 $\alpha$ 决定使用增强版本还是原始版本：

**1. Gaussian Perturbation (GP)**：
$$g^i_{GP} = g^i + n_{GP}, \quad n_{GP} \sim \mathcal{N}(0, \sigma^2)$$
直接添加固定标准差的高斯噪声。

**2. Proportional Gaussian Perturbation (PGP)**：
$$g^i_{PGP} = g^i + n_{PGP}, \quad n_{PGP} \sim \mathcal{N}(0, \sigma^2) \cdot G$$
噪声与特征值成比例，避免噪声相对数据规模过大或过小。

**3. In-Distribution Gaussian Perturbation (IDGP)**：
$$g^i_{IDGP} = g^i + n_{IDGP}, \quad n_{IDGP} \sim \mathcal{N}(0, \alpha \cdot \text{std}(G)^2)$$
噪声标准差基于输入数据自身的统计特性，自适应地匹配数据分布。

**4. Clipped Gaussian Perturbation (CGP)**：
$$g^i_{CGP} = g^i + \text{Clip}(n_{CGP}, c), \quad n_{CGP} \sim \mathcal{N}(0, \sigma^2)$$
将噪声截断在 $[-c, c]$ 范围内，防止极端扰动。

**5. Frequency-Domain Perturbation (FDP)**：
$$F^i = \mathcal{F}(g^i) \rightarrow F^i_{filtered} \rightarrow F^i_{FDP} = F^i_{filtered} + n \rightarrow g^i_{FDP} = \mathcal{F}^{-1}(F^i_{FDP})$$
傅里叶变换到频域 → 选择性保留频率分量 → 添加噪声 → 逆变换回时域。可以有针对性地增强特定频率分量。

## 实验

### 主实验：整体性能比较

| 方法 | Kawarith6 Avg | Twitter2012 Avg | Twitter2018 Avg |
|------|-------------|-----------------|-----------------|
| TF-IDF | 92.68 | 50.97 | 31.30 |
| BERT | 75.99 | 60.24 | 43.73 |
| GraphMSE | 94.35 | 73.87 | 71.46 |
| GraphHAM | 94.84 | 77.57 | 76.16 |
| **SED-Aug** | **98.35** | **91.28** | **88.02** |
| 提升 | +3.70↑ | +17.67↑ | +15.57↑ |

SED-Aug在所有数据集上全面超越基线，尤其在Twitter2012和Twitter2018上提升巨大。

### 消融实验

**显式增强方法对比**：
- "Keep Entity Unchanged"在所有数据集上一致表现最好（Twitter2012 Micro F1: 92.76）
- "Style Transfer"效果最弱但仍有明显提升
- 保持实体信息对SED至关重要，因为实体是区分和理解事件的关键

**隐式增强方法对比**：
- PGP在Kawarith6和Twitter2018表现最好
- CGP在Twitter2012的Micro F1最高
- 所有5种隐式方法都持续提升Macro F1，尤其在类别不平衡的Twitter2018（从82.14%→86.43%）
- 隐式增强对少数类别（稀有事件）的帮助最大

**显式+隐式组合**：
- 隐式增强与显式增强组合始终带来额外收益，无性能下降
- 即使与最弱的显式方法组合也有正向效果

### Extract & Rewrite的信息类型对比

| 信息类型 | Kawarith6 | Twitter2012 | Twitter2018 |
|----------|-----------|-------------|-------------|
| Keywords | **最优** | **最优** | 第二 |
| Entities | 第二 | 第二 | **最优** |
| Knowledge Graph | 第三 | 第三 | 第三 |

关键词最有效，其次是实体，知识图谱效果最弱。

### 频域扰动模式分析

保留高频分量（即衰减低频）效果最好，因为低频分量包含语义结构的关键信息，保留它们并对高频噪声进行操作更合理。

### 低资源场景

| 训练数据比例 | 无增强 | 有增强 | 提升 |
|-------------|--------|--------|------|
| 10% | 68.86 | 75.94 | +10.29 |
| 20% | 75.80 | 82.71 | +9.12 |
| 50% | 82.07 | 88.66 | +8.03 |
| 70% | 85.71 | 91.28 | +6.50 |

在数据极度稀缺（10%）时增强效果最显著；无增强在30%后性能停滞，双重增强则持续提升。

### 可视化分析

- 直方图显示增强前后特征分布形状基本一致，仅方差略增（0.3284→0.3302），符合"不改变均值只增加多样性"的设计目标
- PCA可视化显示增强点与原始点高度重叠但有细微差异

## 亮点与洞察

1. **首次将LLM应用于SED数据增强**：开辟了社交事件检测利用LLM的新方向
2. **双重增强互补设计**：显式增强丰富文本多样性，隐式增强在特征空间引入结构感知的变化，两者互补无冲突
3. **即插即用**：框架与底层SED模型解耦，可以无缝集成到任何基于图的SED方法中
4. **对类别不平衡的有效缓解**：隐式增强在不平衡数据集上效果尤为突出，显著提升少数类的识别能力
5. **训练时增强、推理时无需LLM**：LLM仅在数据准备阶段使用，推理时不产生额外开销

## 局限性

1. 缺乏确定最优增强数据量的明确标准——增强过多可能引入噪声和冗余，过少则效果不足
2. LLM添加上下文信息可能引入幻觉（实验发现约6%的虚假声明）
3. 5种显式方法 × 5种隐式方法 × 多个数据集的组合空间巨大，未能穷尽所有组合
4. 增强效果依赖于底层LLM的质量，不同LLM可能产生不同结果

## 相关工作

- **社交事件检测**：从基于内容的方法（TF-IDF、Word2Vec）到图方法（GCN、GAT、KPGNN、GraphHAM）
- **数据增强**：文本级（插入、删除、替换、回译）和特征空间级（高斯过程、类协方差矩阵）
- **LLM用于数据增强**：在NLP其他任务中已有应用，但SED领域首次

## 评分 ⭐⭐⭐⭐

**优点**：框架设计清晰，显式和隐式增强的设计都有充分的动机和理论支撑；实验非常全面（6个研究问题逐一回答）；提升幅度巨大（15-17%平均F1）。

**不足**：增强方法本身并非特别新颖（高斯扰动等较为基础）；频域扰动的设计缺乏更深入的理论分析；缺少与其他数据增强方法（如Mixup, CutMix等）的对比。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Is Linguistically-Motivated Data Augmentation Worth It?](is_linguistically-motivated_data_augmentation_worth_it.md)
- [\[ICML 2025\] Curvature Enhanced Data Augmentation for Regression](../../ICML2025/others/curvature_enhanced_data_augmentation_for_regression.md)
- [\[ACL 2025\] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](seoe_semantic_eval.md)
- [\[ICCV 2025\] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents](../../ICCV2025/others/adversarial_data_augmentation_for_single_domain_generalization_via_lyapunov_expo.md)
- [\[ACL 2025\] SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)

</div>

<!-- RELATED:END -->
