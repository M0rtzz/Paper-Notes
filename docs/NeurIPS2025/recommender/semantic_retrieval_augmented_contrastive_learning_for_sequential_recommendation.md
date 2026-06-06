---
title: >-
  [论文解读] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation
description: >-
  [NeurIPS 2025][推荐系统][序列推荐] 提出SRA-CL框架，利用LLM的语义理解能力构建高质量对比样本对，通过语义检索+可学习样本合成器增强序列推荐的对比学习，以即插即用的方式在4个数据集上取得SOTA。
tags:
  - "NeurIPS 2025"
  - "推荐系统"
  - "序列推荐"
  - "对比学习"
  - "大语言模型"
  - "语义检索"
  - "数据增强"
---

# Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation

**会议**: NeurIPS 2025  
**arXiv**: [2503.04162](https://arxiv.org/abs/2503.04162)  
**代码**: [GitHub](https://github.com/ziqiangcui/SRA-CL)  
**领域**: 推荐系统  
**关键词**: 序列推荐, 对比学习, 大语言模型, 语义检索, 数据增强

## 一句话总结

提出SRA-CL框架，利用LLM的语义理解能力构建高质量对比样本对，通过语义检索+可学习样本合成器增强序列推荐的对比学习，以即插即用的方式在4个数据集上取得SOTA。

## 研究背景与动机

对比学习在序列推荐中被广泛使用以缓解数据稀疏问题，但现有方法在构造正样本对时存在两大缺陷：

**语义偏差**: 随机增强方法（masking、dropout）可能彻底改变序列的用户偏好语义；基于协同信号的聚类方法（如K-means）受限于稀疏ID信号，聚类不精确

**不可学习性**: 现有方法依赖预定义硬规则（同聚类配对、共享下一物品配对），无法让模型自主学习最优的对比样本构造方式

作者的关键洞察是：文本语义信息（类别、品牌、描述）天然稳定且不受数据量和训练动态影响，可作为更可靠的对比样本来源。而LLM具备强大的语义理解和推理能力，适合生成捕捉用户偏好和物品特征的语义嵌入。

## 方法详解

### 整体框架

SRA-CL是一个模型无关的插件框架，包含三个模块：(1) 基于用户语义检索的跨用户对比学习；(2) 基于物品语义检索的用户内对比学习；(3) 推荐主任务。三者通过加权损失 $\mathcal{L} = \mathcal{L}_{\text{Rec}} + \alpha \mathcal{L}_{\text{CS}} + \beta \mathcal{L}_{\text{IS}}$ 联合训练。推理时仅使用推荐骨干，无额外LLM开销。

### 关键设计

1. **基于LLM的用户偏好语义嵌入**: 将用户交互序列（按时间排序的物品属性和描述）构造为prompt $\mathcal{P}_u$，送入LLM（DeepSeek-V3）推理用户偏好 $\mathcal{A}_u = \text{LLM}(\mathcal{P}_u)$，再用预训练文本嵌入模型（SimCSE-RoBERTa）编码为固定语义向量 $\tilde{\mathbf{h}}_u$。基于余弦相似度检索Top-k相似用户构成候选池 $\mathcal{N}_u$。语义嵌入全程固定，不参与训练。

2. **可学习对比样本合成器**: 不直接选择候选池中的某个用户作为正样本（这种硬规则效果次优），而是通过注意力机制计算每个候选适合度 $p_{u,u'} = \text{softmax}(\text{LeakyReLU}(\mathbf{a}^\top[\mathbf{W}\tilde{\mathbf{h}}_u \| \mathbf{W}\tilde{\mathbf{h}}_{u'}]))$，然后加权组合候选的推荐模型表示 $\mathbf{h}_u^+ = \sum_{u' \in \mathcal{N}_u} p_{u,u'} \mathbf{h}_{u'}$。合成器参数与推荐模型联合训练。

3. **物品语义检索的用户内对比学习**: 同样用LLM理解物品（提供物品属性+上下文序列信息），编码为语义嵌入 $\tilde{\mathbf{e}}_v$，检索Top-k相似物品构建候选池 $\mathcal{N}_v$。通过随机选择20%物品并用候选池中的相似物品替换，生成两个语义一致的增强视图 $\mathcal{S}_u', \mathcal{S}_u''$ 作为正样本对。这里不使用可学习合成器（实验发现无额外收益），因为物品语义比用户偏好更可量化。

### 损失函数 / 训练策略

- 推荐损失: 标准交叉熵 $\mathcal{L}_{\text{Rec}} = -\hat{y}_{v^+} + \log(\sum_v \exp(\hat{y}_v))$
- 跨用户对比损失（InfoNCE）: $\mathcal{L}_{\text{CS}} = -\log \frac{\exp(\mathbf{h}_u \cdot \mathbf{h}_u^+)}{\exp(\mathbf{h}_u \cdot \mathbf{h}_u^+) + \sum_{\mathbf{h}_u^-} \exp(\mathbf{h}_u \cdot \mathbf{h}_u^-)}$
- 用户内对比损失: $\mathcal{L}_{\text{IS}}$ 类似，以两个增强视图表示为正样本对，batch内其余为负样本
- 所有语义嵌入训练前预计算并冻结，不引入推理延迟

## 实验关键数据

### 主实验

| 数据集 | 指标 | SRA-CL | MCLRec(次优CL) | ICSRec | DuoRec | 提升 |
|--------|------|--------|---------------|--------|--------|------|
| Yelp | HR@20 | **0.1282** | 0.1150 | 0.1165 | 0.1173 | +9.29% |
| Yelp | NDCG@20 | **0.0533** | 0.0486 | 0.0495 | 0.0493 | +7.68% |
| Sports | HR@20 | **0.0823** | 0.0736 | 0.0728 | 0.0706 | +11.82% |
| Sports | NDCG@20 | **0.0347** | 0.0318 | 0.0304 | 0.0302 | +9.12% |
| Beauty | HR@20 | **0.1314** | 0.1239 | 0.1205 | 0.1224 | +6.05% |
| Office | HR@20 | **0.1702** | 0.1629 | 0.1643 | 0.1549 | +3.59% |

### 消融实验

| 配置 | 说明 |
|------|------|
| w/o CL | 去除所有对比学习，性能显著下降 |
| w/o $\mathcal{L}_{\text{CS}}$ | 去除跨用户对比，性能下降明显 |
| w/o $\mathcal{L}_{\text{IS}}$ | 去除用户内对比，性能略降 |
| w/o learnable synthesizer | 用硬规则替代合成器，性能下降 |
| w/o semantic (用随机增强) | 随机增强替代语义检索，性能退化最大 |
| w/o LLM (用原始文本) | 不用LLM处理直接用原始文本嵌入，性能下降 |

### 关键发现

- Sports数据集提升最大（+11.82% HR@20），可能因为该数据集最稀疏，语义信息收益最大
- 模型无关性验证：SRA-CL分别提升GRU4Rec (+27.3% HR)、SASRec (+15.2%)、DuoRec (+8.3%)
- 可学习合成器比硬规则选择提升约2-4%，表明学习最优融合权重的重要性
- 推理阶段无额外开销，因为语义嵌入预计算且只在训练时使用

## 亮点与洞察

- 巧妙利用LLM作为离线语义编码器而非在线推理组件，避免推理延迟问题
- 双层检索设计（用户级+物品级）覆盖了对比学习的两大范式（inter和intra）
- 可学习合成器将"选择正样本"转化为"加权组合候选"，比硬规则更灵活
- 物品语义检索中提供上下文序列信息是重要创新——让LLM理解物品在推荐场景中的角色

## 局限与展望

- LLM调用成本高（DeepSeek-V3 API），大规模数据集上语义嵌入预计算开销大
- 物品上下文序列限制为10条，可能遗漏重要使用模式
- Top-k超参数对性能影响的sensitivity分析不够
- 语义嵌入完全固定，无法随训练动态更新——可探索轻量级微调
- 实验数据集规模中等，工业级场景效果待验证

## 相关工作与启发

- 与LRD/RLMRec等LLM增强推荐方法的区别：SRA-CL专注于改善对比学习而非直接增强推荐模型
- 语义检索增强的思路可推广到其他自监督学习任务（如图表示学习）
- 可学习样本合成器的注意力机制可看作一种soft sample selection策略

## 评分

- 新颖性: ⭐⭐⭐⭐ LLM语义+对比学习的结合新颖，可学习合成器设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 4个数据集、13个baseline、模型无关验证、详细消融
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，图表直观
- 价值: ⭐⭐⭐⭐ 为序列推荐的对比学习提供了新的语义增强范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation](tv-rec_time-variant_convolutional_filter_for_sequential_recommendation.md)
- [\[ACL 2025\] Laser: Bi-Tuning with Collaborative Information for Controllable LLM-Based Sequential Recommendation](../../ACL2025/recommender/bi-tuning_with_collaborative_information_for_controllable_llm-based_sequential_r.md)
- [\[ACL 2026\] Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders](../../ACL2026/recommender/bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_llms_a.md)
- [\[ICLR 2026\] C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](../../ICLR2026/recommender/c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)
- [\[ICML 2026\] Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy](../../ICML2026/recommender/rethinking_contrastive_learning_for_graph_collaborative_filtering_limitations_an.md)

</div>

<!-- RELATED:END -->
