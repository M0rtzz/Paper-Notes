---
title: >-
  [论文解读] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation
description: >-
  [AAAI 2026] 提出 UniTok，一个统一的商品 tokenization 框架，通过定制的 Mixture-of-Experts（TokenMoE）架构结合共享码本，实现跨多个领域的高效商品离散化表示，避免为每个领域单独训练 tokenizer，同时通过互信息校准机制保持跨域语义平衡。
tags:
  - AAAI 2026
---

# Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation

**会议**: AAAI 2026  
**arXiv**: [2511.12922](https://arxiv.org/abs/2511.12922)  
**代码**: [github.com/jackfrost168/UniTok](https://github.com/jackfrost168/UniTok)

## 一句话总结

提出 UniTok，一个统一的商品 tokenization 框架，通过定制的 Mixture-of-Experts（TokenMoE）架构结合共享码本，实现跨多个领域的高效商品离散化表示，避免为每个领域单独训练 tokenizer，同时通过互信息校准机制保持跨域语义平衡。

## 研究背景与动机

### 问题背景
基于大语言模型（LLM）的推荐系统通过 item tokenization 将商品空间映射到语言空间，使 LLM 能够将商品作为自然语言序列的一部分进行处理。然而，现有的 item tokenization 方法（如 TIGER、LC-Rec、LETTER）都是针对**单领域**设计的，需要为每个领域单独训练一个 tokenizer。

### 核心挑战

**训练开销（C1）**：当推荐系统要覆盖多个商品领域时，重复训练领域特定的 tokenizer 效率低下、资源消耗大。在 10 个领域上，现有方法的可训练参数量是 UniTok 的 9.63 倍。

**语义对齐（C2）**：不同领域的数据分布和语义差异大，简单地使用共享 token 空间会导致语义混淆和有偏的 token 分配。

### 动机
NLP 和 CV 领域已经出现了统一多领域学习的趋势，但推荐系统中的 item tokenization 仍停留在"一个领域一个模型"的范式。本文首次尝试构建一个跨领域统一的 item tokenization 框架。

## 方法详解

### 整体框架
UniTok 由四个核心组件构成：**共享自编码器**、**TokenMoE**、**码本标识符**和**互信息校准机制**。

### 1. 共享自编码器
- 使用预训练内容编码器获取各领域商品的语义嵌入 $\mathbf{X}^k \in \mathbb{R}^{|\mathcal{I}_k| \times d}$
- 共享编码器 $f_\theta$ 将所有领域的商品投射到统一的潜在空间：$\mathbf{z}_i^k = f_\theta(\mathbf{x}_i^k)$
- 共享解码器 $g_\phi$ 从量化后的潜在表示重建原始嵌入
- 优化重建损失 $\mathcal{L}_{\text{Rec}} = \sum_{k=1}^{K} \sum_{\mathbf{x}_i^k} \|\mathbf{x}_i^k - \hat{\mathbf{x}}_i^k\|^2$

### 2. TokenMoE：定制的 MoE 架构
**核心创新点**——将 MoE 引入 tokenization 模块（而非传统的 Transformer FFN 层）：
- **领域特定专家**：每个领域对应一个专家网络，捕捉该领域独有的语义模式
- **共享专家**：始终激活，编码跨领域的通用知识
- **路由器**：softmax 分布决定 top-N 个领域特定专家的选择
- 输出计算：$\hat{\mathbf{z}}_i^k = \sum_{k=1}^{K} G_k E_k(\mathbf{z}_i^k) + E_{\text{share}}(\mathbf{z}_i^k)$
- 专家初始化：用各领域的均值特征初始化，提供强归纳偏置

### 3. 码本标识符（Residual Quantization）
- 每个专家内部使用残差量化（RQ）将商品离散化为紧凑的 token 序列
- 使用 $L$ 级码本，每级包含 $T$ 个码向量
- 最终每个商品被表示为：$\mathbf{z}_i^k \mapsto \mathbf{c}_i^k = (z_1, \dots, z_L, e_1, \dots, e_N)$
- 其中 $z_\ell$ 是码本索引，$e_n$ 是被选中专家的 ID
- 训练 RQ 损失包含码本学习和 commitment 两项

### 4. 互信息（MI）校准机制
解决跨域语义不平衡问题：
- 使用 HSIC（Hilbert-Schmidt 独立性准则）作为互信息的代理指标
- 度量输入语义嵌入 $\mathbf{X}^k$ 与潜在嵌入 $\mathbf{Z}^k$ 之间的依赖关系
- MI 校准损失：$\mathcal{L}_{\text{MI}} = \text{Var}[\hat{I}^{(k)}] - \beta \mathbb{E}[\hat{I}^{(k)}]$
- 第一项惩罚跨域 MI 方差以缓解语义不平衡
- 第二项鼓励每个域保留足够的领域特定信息

### 总体优化目标
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Rec}} + \lambda_{\text{RQ}} \mathcal{L}_{\text{RQ}} + \lambda_{\text{MI}} \mathcal{L}_{\text{MI}}$$

## 实验

### 实验设置
- **数据集**：10 个真实世界数据集（Beauty、Cellphones、Grocery、Instruments、Office、Pet Supplies、Tools、Toys、Games、Yelp）
- **对比方法**：4 个协同过滤方法（MF、LightGCN、SASRec、Bert4Rec）+ 5 个 item tokenization 方法（P5-TID、P5-SemID、TIGER、LC-Rec、LETTER）
- **评价指标**：Recall@M、NDCG@M（M∈{5,10}），全排序协议
- **实现细节**：4 级码本，每级 256 个码向量，维度 32；$\lambda_{\text{RQ}}=1$，$\lambda_{\text{MI}}=0.03$

### 主要结果（Table 1: NDCG@10）

| 方法 | Beauty | Cellphones | Grocery | Tools | Toys | Yelp |
|------|--------|------------|---------|-------|------|------|
| LETTER（次优） | 0.0364 | 0.0473 | 0.0392 | 0.0298 | 0.0291 | 0.0231 |
| **UniTok** | **0.0478** | **0.0647** | **0.0533** | **0.0439** | **0.0442** | **0.0321** |
| 提升 | +25.5% | +36.8% | +36.0% | +43.0% | +51.9% | +39.0% |

- UniTok 在全部 10 个数据集上均取得最优，最高提升达 **51.89%**（Toys）
- 注意：UniTok 使用一个统一模型处理所有 10 个数据集，而对比方法需要为每个数据集单独训练

### 效率对比（Table 2: 可训练参数量）

| 模块 | 传统码本方法（10 数据集总和） | UniTok |
|------|-------------------------------|--------|
| 码本 | 0.33M | 0.36M |
| 自编码器 | 87.45M | 8.75M |
| 路由器 | — | 0.01M |
| **总计** | **87.78M** | **9.11M** |

参数量减少约 **9.63 倍**，主要得益于共享自编码器的设计。

### 零样本泛化（Table 4: 未见领域）
在 Clothing、Health、Sports 三个训练时未见过的领域上直接测试：
- UniTok 无需重训即可在新领域上取得最优表现
- Health 上 NDCG@10 提升 **17.87%**，Clothing 上 Recall@10 提升 **12.33%**

### 消融实验（Table 5）

| 变体 | 说明 | Beauty N@10 |
|------|------|-------------|
| UniTok-1 | 移除 TokenMoE + MI | 0.0304 |
| UniTok-2 | 保留 MoE，移除共享专家 + MI | 0.0436 |
| UniTok-3 | 移除 MI 校准 | 0.0457 |
| **UniTok** | 完整版 | **0.0478** |

TokenMoE 贡献最大，MI 校准提供进一步的稳定提升。

## 理论分析

论文提供了三个理论支撑：
1. **Theorem 1**：UniTok 的 token 空间具有严格更高的熵，意味着更大的 token 空间容量
2. **Theorem 2**：UniTok 的期望量化误差不超过标准码本方法，即量化更精确
3. **Theorem 3**：跨域性能差异的上界受 MI 方差控制，减小方差即可促进更稳定的跨域泛化

## 亮点

- **一次训练，处处推荐**：一个统一模型即可处理多个领域，显著降低训练和部署成本
- **TokenMoE 设计新颖**：将 MoE 引入 tokenization 模块而非 FFN，领域特定专家 + 共享专家的组合既保持领域特性又共享通用知识
- **MI 校准机制**：通过 HSIC 代理互信息并最小化跨域方差，优雅地解决了多领域语义不平衡问题
- **理论+实验双重验证**：三个定理从熵、量化误差、性能一致性三个角度提供理论保证
- **零样本泛化能力**：无需重训即可在新领域上取得竞争力表现

## 局限性

1. **仅利用内容语义**：为保持通用性，刻意不使用协同过滤信号（用户-商品交互），可能损失部分推荐精度
2. **专家数量与领域数绑定**：领域特定专家数 $K$ 通常等于领域数，新增领域可能需要调整架构
3. **码本容量固定**：所有领域共享相同的码本大小（256 个码向量，4 级），不同领域可能需要不同的粒度
4. **评估场景有限**：主要在 Amazon 和 Yelp 数据集上验证，暂未在工业级大规模系统中测试
5. **HSIC 计算开销**：互信息校准需要计算核矩阵，在大规模数据集上可能带来额外的计算负担

## 相关工作

- **Item Tokenization for LLM-Rec**：TIGER（Rajput et al. 2023）通过 RQ 生成码本标识符；LC-Rec（Zheng et al. 2024）在此基础上增强；LETTER（Wang et al. 2024）进一步改进。这些方法均为单领域设计。
- **Mixture-of-Experts**：从经典 MoE（Jacobs et al. 1991）到 Switch Transformers（Fedus et al. 2022）和 DeepSeekMoE（Dai et al. 2024），MoE 已在模型扩展中广泛应用。
- **多领域推荐**：ADIN（Jiang et al. 2022）和 MDRED（Ning et al. 2023）研究跨域推荐，但未涉及 item tokenization 的统一化。
- **P5 系列**：P5-TID 和 P5-SemID（Hua et al. 2023）探索了不同的 item ID 索引方式，但缺乏语义码本支撑。

## 评分

⭐⭐⭐⭐ (4/5)

**理由**：提出了推荐系统中一个被忽视但重要的问题——跨域 item tokenization，方法设计清晰（TokenMoE + MI 校准），理论分析扎实，实验覆盖广泛且提升显著（最高 51.89%）。扣分原因是未使用协同信号的设计决策使得方法更偏向于内容推荐而非完整推荐场景，且 HSIC 在大规模场景下的可扩展性存疑。

<!-- RELATED:START -->

## 相关论文

- [Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](align3gr_unified_multi-level_alignment_for_llm-based_generat.md)
- [When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation](when_top-ranked_recommendations_fail_modeling_multi-granular_negative_feedback_f.md)
- [Token-Efficient Item Representation via Images for LLM Recommender Systems](../../ICLR2026/recommender/token-efficient_item_representation_via_images_for_llm_recommender_systems.md)
- [Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents](bid_farewell_to_seesaw_towards_accurate_long-tail_session-based_recommendation_v.md)
- [FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)

<!-- RELATED:END -->
