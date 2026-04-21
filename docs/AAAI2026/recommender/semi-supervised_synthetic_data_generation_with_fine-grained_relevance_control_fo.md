---
title: >-
  [论文解读] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling
description: >-
  [AAAI 2026][合成数据] 提出SSRA（半监督相关性感知合成数据管道），通过两阶段流程生成具有可控细粒度相关性标签（4级）的领域自适应短视频数据，增强embedding模型的语义相关性建模能力，在抖音双列场景线上A/B测试中CTR提升1.45%。
tags:
  - AAAI 2026
  - 合成数据
  - 细粒度相关性
  - 短视频搜索
  - 半监督学习
  - Embedding模型
---

# Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling

**会议**: AAAI 2026  
**arXiv**: [2509.16717](https://arxiv.org/abs/2509.16717)  
**代码**: 无  
**领域**: 推荐系统 / 搜索相关性  
**关键词**: 合成数据, 细粒度相关性, 短视频搜索, 半监督学习, Embedding模型

## 一句话总结

提出SSRA（半监督相关性感知合成数据管道），通过两阶段流程生成具有可控细粒度相关性标签（4级）的领域自适应短视频数据，增强embedding模型的语义相关性建模能力，在抖音双列场景线上A/B测试中CTR提升1.45%。

## 研究背景与动机

### 现有合成数据方法的局限

Embedding模型是搜索、推荐的基础组件。近期大量工作利用LLM合成多样化训练数据来提升embedding质量（如Gemini Embedding、Qwen3-Embedding等）。但存在两个关键问题：

**领域差距（Domain Gap）**：基于prompt的合成方法受限于LLM的生成能力，合成数据与特定领域的真实数据分布存在差距。FinMTEB的实验表明，MTEB上的SOTA模型在金融垂直领域性能显著下降。

**相关性粒度不足（Relevance Granularity）**：绝大多数合成方法仅使用**二元相关性**（相关/不相关），但实际检索任务需要**细粒度相关性排序**——二元标签与下游任务需求存在错位。

### 短视频领域的特殊挑战

- 短视频天然是**多模态**的，缺乏显式文本表示
- 不同相关性等级的分布极度不均衡（中间等级1/2严重不足）
- 搜索列表排序受个性化和热度等非语义因素影响

### 核心思路

构建一个**可控的query生成模型**$f:(d,s)\mapsto\hat{q}$，给定文档$d$和目标相关性标签$s\in\{0,1,2,3\}$，生成满足领域分布且语义相关性匹配目标标签的query。

## 方法详解

### 整体框架

SSRA是一个两阶段半监督管道：
- **Stage 1**：通过基于分数的重标注增强query多样性
- **Stage 2**：通过迭代精炼增强生成query与目标相关性标签的对齐

两个核心模型协同训练：
- **Query模型**：给定document和目标相关性标签，生成query
- **Score模型**：给定query-document对，预测相关性标签

### 关键设计

#### 1. 短视频相关性数据集构建

**功能**：构建首个带4级细粒度相关性标注的中文短视频搜索数据集。

**4级相关性定义**：

| 标签 | 定义 |
|------|------|
| 3 | 完全精确满足用户意图 |
| 2 | 大体满足但部分非关键要素缺失 |
| 1 | 部分满足，关键实体/概念相关但重要方面不匹配 |
| 0 | 完全不满足用户意图 |

**数据构建流程**：
1. **Query-Item收集**：从抖音搜索点击日志中收集query-item对（query驱动检索 + 点击采样两种策略）
2. **Document生成**：用豆包大模型将视频的OCR+ASR改写为连贯文本描述，结合标题作为document
3. **双注标协议**：两名标注者独立标注，不一致时由专家裁决

**数据规模**：训练集207,439对 | 检索测试集10,866对 | 分类测试集3,390对。中间相关性标签（1/2）严重缺乏（仅占~4.5%训练集）。

#### 2. Stage 1: 基于分数的重标注增强多样性

**功能**：解决原始标注数据中"一个query对应多个document"导致的query生成单一化问题。

**核心思路**：
1. 在标注数据上训练score模型（预测4级相关性标签）
2. 对无标签数据按document分组，用score模型为每个document关联的多个query标注相关性分数
3. 将重标注数据与去重原始数据合并，训练初版query模型

**设计动机**：原始数据中多个document共享同一高频query → query模型学会将不同document映射到相同query → 生成多样性差。重标注后，每个document关联多个不同相关性等级的query，形成document-to-queries (D2Q)结构。

**实测效果**：重复query率从6.57%降至5.20%（相对减少20.85%）。

#### 3. Stage 2: 迭代精炼增强相关性对齐

**功能**：提高query模型生成的query与目标相关性标签的匹配度。

**核心步骤**：
1. **初始合成**：用Stage 1训练的query模型在无标签document集上生成条件化于不同相关性标签的query
2. **Score模型过滤**：用score模型预测每个合成query-document对的相关性标签，仅保留预测标签与目标标签一致的样本
3. **LLM成对一致性过滤**：用LLM比较同一document下不同相关性标签的query对，过滤掉相对排序与标签不一致的样本
4. 将高质量样本与Stage 1训练数据合并，进行第二轮query模型训练

**设计动机**：query模型初期生成的query与目标标签的对齐度不够。通过score模型+LLM双重过滤，确保训练数据中的相关性标签是可靠的。

**实测效果（人工标注验证）**：

| 相关性标签 | Stage 1 Only | Stage 1+2 |
|-----------|-------------|-----------|
| 标签1 | 81/200 | 130/200 |
| 标签2 | 80/200 | 131/200 |
| 标签3 | 189/200 | 178/200 |

Stage 2使相关性匹配的一致性提升了25.43%（标签1和2显著改善）。

### 损失函数 / 训练策略

**Embedding模型训练**：使用带标签权重的InfoNCE损失：

$$\mathcal{L}=\frac{1}{B}\sum_{i=1}^{B}s_i\cdot\mathcal{L}_{\text{infoNCE}}$$

其中$s_i\in\{0,1,2,3\}$是相关性标签。这使得更高相关性的正样本对获得更大权重。

**实现细节**：
- Query/Score模型骨干：豆包-1.5-Pro-32K
- Embedding模型骨干：Qwen3-Embedding (0.6B/4B两个规模)
- LoRA微调，rank=32，batch size=512
- 合成100万文档的多级别query，经过滤后与标注数据合并训练

## 实验关键数据

### 主实验

| 方法 | 0.6B nDCG@10 | 0.6B Avg AP | 4B nDCG@10 | 4B Avg AP |
|------|-------------|------------|------------|----------|
| Base Model | 71.36 | 69.50 | 73.20 | 69.57 |
| SyCL Modified (prompt合成) | 71.50 | 67.33 | 73.56 | 67.95 |
| Vanilla SFT | 71.88 | 70.39 | 74.32 | 70.87 |
| **SSRA** | **71.97** | **70.79** | **74.47** | **71.52** |

**关键对比**：
- SSRA vs Base：4B模型nDCG@10 +1.73%，AP +2.80%
- SyCL Modified（prompt方法）**反而降低**了分类性能（AP降2.17/1.62%），说明prompt合成在领域特定场景不可靠
- SSRA一致超越Vanilla SFT，证明两阶段精炼的价值

### 消融实验

| 方法 | 0.6B nDCG@10 | 0.6B AP | 4B nDCG@10 | 4B AP |
|------|-------------|---------|------------|-------|
| w/o Stage 1 & 2 | 71.44 | 70.70 | 74.02 | 70.34 |
| w/o Stage 2 | 71.79 | 70.77 | 74.23 | 71.30 |
| **SSRA (full)** | **71.97** | **74.13** | **74.47** | **74.92** |

**任务特异性分析**：
- **检索任务**同时受益于Stage 1（多样性）和Stage 2（相关性对齐）
- **分类任务**主要受益于Stage 2（不依赖query多样性，依赖相关性精度）

### 二元 vs 多级相关性

| 配置 | 0.6B nDCG@10 | 4B nDCG@10 | 说明 |
|------|-------------|------------|------|
| 二元标签(0,1) | 71.66 | 73.73 | 仅正/负 |
| 多级标签(0,1,2,3) | 71.78 | 74.23 | 4级 |

多级相关性标签在检索任务上一致更好，在分类的中间阈值（AP@≥1, AP@≥2）上也更好，验证了细粒度相关性的价值。

### 线上A/B测试

| 指标 | 提升 | 说明 |
|------|------|------|
| CTR | +1.45% | 点击通过率 |
| SRR | +4.9% | 强相关内容占比 |
| IUPR | +0.1054% | 图文用户渗透率 |

在抖音双列场景中，1.9亿用户/天的随机实验，运行10天。

### 关键发现

1. **Prompt合成方法不适合垂直领域**：SyCL Modified在分类任务上反而降低性能，表明LLM的prompt合成在捕捉领域特定分布方面存在根本局限
2. **中间相关性标签的增强有显著价值**：原始数据中标签1/2仅占~4.5%，SSRA通过合成补充这些缺失等级
3. **两阶段各有分工**：Stage 1→多样性（降重复率20.85%），Stage 2→精度（提一致性25.43%）
4. **Scale matters**：4B模型的改善幅度大于0.6B，说明SSRA的合成数据在大模型上更好地被利用

## 亮点与洞察

1. **数据资源贡献**：首个4级相关性标注的中文短视频搜索数据集，填补了该领域基准缺失
2. **半监督闭环设计**：Score模型标注 → Query模型生成 → Score模型验证 → LLM过滤，形成了质量逐步提升的闭环
3. **新视角**：相关性多样性（而不仅是query/document多样性）是embedding模型训练的关键维度
4. **工业验证**：在抖音级别的线上实验中获得显著提升，具有很强的实践说服力
5. **带标签权重的InfoNCE**：简单而有效地将多级相关性融入对比学习损失

## 局限与展望

1. **数据集未公开**：虽然声称contribution是数据集，但受限于商业场景可能无法完全开源
2. **依赖于领域标注数据**：SSRA仍需要一定量的高质量标注数据来训练Score模型，冷启动成本不低
3. **LLM过滤的成本**：Stage 2的LLM成对一致性检查对大规模应用来说开销可观
4. **标签数量固定**：仅验证了4级相关性，更细粒度（如连续分数）的效果未探索
5. 可以探索**主动学习**来选择最需要标注的样本，降低人工标注成本
6. **多模态信息未充分利用**：短视频有视觉信息，仅通过OCR/ASR转文本可能损失信息

## 相关工作与启发

- **SyCL (Esfandiarpoor et al. 2025)**：通过prompt生成4级相关性文档，但本文证明prompt方法在垂直领域效果差
- **Gecko (Lee et al. 2024)**：用生成的query检索候选文档再LLM打分选正例，启发了"合成+验证"的范式
- **Qwen3-Embedding (Zhang et al. 2025)**：利用Persona Hub指导合成，侧重persona多样性而非相关性多样性
- **Hard Negative Mining**：广泛用于增强模型对细微差异的敏感性，本文提供了更系统的方案
- 启发：**半监督框架**在数据合成中的价值被低估——让两个模型协同迭代比单一prompt策略更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ （两阶段半监督合成管道+相关性多样性视角）
- 实验充分度: ⭐⭐⭐⭐⭐ （离线多指标+线上1.9亿用户A/B+多规模模型+消融+人工验证）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，但部分段落偏冗长）
- 价值: ⭐⭐⭐⭐⭐ （实际部署验证，解决真实的工业问题）

<!-- RELATED:START -->

## 相关论文

- [Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information](generalization_bounds_for_semi-supervised_matrix_completion_with_distributional_.md)
- [CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search](crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)
- [Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)
- [FineVQ: Fine-Grained User Generated Content Video Quality Assessment](../../CVPR2025/recommender/finevq_fine-grained_user_generated_content_video_quality_assessment.md)
- [When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation](when_top-ranked_recommendations_fail_modeling_multi-granular_negative_feedback_f.md)

<!-- RELATED:END -->
