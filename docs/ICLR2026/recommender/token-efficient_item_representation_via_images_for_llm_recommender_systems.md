---
title: >-
  [论文解读] Token-Efficient Item Representation via Images for LLM Recommender Systems
description: >-
  [ICLR 2026][LLM推荐系统] 提出 I-LLMRec，利用商品图像替代冗长文本描述来表示推荐系统中的物品语义，通过 RISA 对齐模块和 RERI 检索模块，在仅用单个token表示物品的同时保留丰富语义，推理速度提升约2.93倍且推荐性能超越文本描述方法。
tags:
  - ICLR 2026
  - LLM推荐系统
  - 图像表示
  - Token效率
  - 多模态对齐
  - 检索式推荐
---

# Token-Efficient Item Representation via Images for LLM Recommender Systems

**会议**: ICLR 2026  
**arXiv**: [2503.06238](https://arxiv.org/abs/2503.06238)  
**代码**: https://github.com/rlqja1107/torch-I-LLMRec  
**领域**: LLM NLP / 推荐系统  
**关键词**: LLM推荐系统, 图像表示, Token效率, 多模态对齐, 检索式推荐

## 一句话总结
提出 I-LLMRec，利用商品图像替代冗长文本描述来表示推荐系统中的物品语义，通过 RISA 对齐模块和 RERI 检索模块，在仅用单个token表示物品的同时保留丰富语义，推理速度提升约2.93倍且推荐性能超越文本描述方法。

## 研究背景与动机

**领域现状**：基于LLM的推荐系统需要将物品交互历史转换为自然语言输入。现有方法分两派：属性表示法（Attribute-Based，如品牌+类别，简短但语义有限）和描述表示法（Description-Based，如完整商品描述，语义丰富但token开销巨大）。

**现有痛点**：这两种方法存在**效率与效果的根本权衡**——属性表示法token少但丢失细粒度语义导致推荐性能下降13%+；描述表示法语义丰富但token量大（平均160 tokens/item），LLM推理时间增加2.5倍以上，且复杂度随用户交互序列长度二次增长。

**核心矛盾**：只要物品用自然语言表示，"更丰富的语义表示→更长的输入→更低的效率" 这一矛盾无法避免。

**本文目标** 如何在保留丰富商品语义的同时，大幅减少token占用？

**切入角度**：作者通过CLIP模型测量发现，电商数据集中**商品图像与文本描述之间存在显著的信息重叠**（Amazon Sport/Art数据集相似度约0.31，甚至高于COCO数据集中精心标注的图文对0.26）。这意味着图像可以用极少token承载描述的大部分语义。

**核心 idea**：用单个图像token替代冗长文本描述来表示推荐物品，通过专门的推荐导向对齐策略弥合视觉和语言空间的差距。

## 方法详解

### 整体框架
I-LLMRec 的输入是用户的物品交互历史（图像序列），输出是下一个推荐物品。整体pipeline：(1) 将每个物品的图像通过预训练视觉编码器（CLIP-ViT）提取特征 → (2) 通过可学习适配器映射到LLM空间（单个token/物品）→ (3) RISA模块对齐视觉与语言空间 → (4) RERI模块通过检索从物品池中推荐物品。LLM参数冻结，仅训练适配器和投影器。

### 关键设计

1. **图像到LLM的映射（Image-to-LLM Mapping）**:

    - 功能：将物品图像压缩为LLM可理解的单个token
    - 核心思路：对用户交互的每个物品 $i$，用CLIP视觉编码器提取特征 $v_i = V(\mathbf{I}_i) \in \mathbb{R}^{d_v}$，再通过适配器网络 $M: \mathbb{R}^{d_v} \rightarrow \mathbb{R}^d$ 映射到LLM维度。提示词格式为 `Title: ITEM_TITLE, Visual Representation: [VISUAL]`，其中 `[VISUAL]` 被替换为适配后的视觉特征
    - 设计动机：将物品描述从平均160 tokens压缩到仅1个token，使复杂度从 $O((|f(\mathbf{D}_i)||\mathcal{S}_u|)^2 d)$ 大幅降低。保留了标题（约10 tokens）提供基础文本锚点

2. **推荐导向的图像-LLM语义对齐（RISA模块）**:

    - 功能：训练适配器使视觉特征在推荐场景下与语言空间对齐
    - 核心思路：构建"输入-目标"格式的训练数据——输入是包含图像特征的用户交互提示 + 关于下一物品属性的问题（如"该用户下一个可能消费什么品牌？"），目标输出是对应答案。4种属性（品牌/类别/标题/描述） × 5种问题模板 = 20种模板，每步随机选一种。训练目标：$\mathcal{L}_{\text{RISA}} = \max_M \sum_{k=1}^{|y|} \log(P_{\theta,M}(y_k|x, y_{<k}))$
    - 设计动机：不同于通用视觉-语言对齐（如UniMP），RISA针对推荐场景定制，让LLM从图像中理解用户偏好。消融实验证明RISA带来显著提升（Hit@5: 0.395→0.432）

3. **基于图像特征的检索式推荐（RERI模块）**:

    - 功能：将推荐任务转化为检索任务，从物品池中直接检索相关物品
    - 核心思路：在用户交互提示后附加指令提示，引导LLM生成推荐token [REC]，利用其最后一层隐状态 $h(\text{[REC]})$ 作为用户偏好表示。通过投影器将用户表示和物品视觉特征映射到共享推荐空间，计算亲和度分数 $r_{u,i}^{\text{Img}} = o_u^{\text{Img}} \circledast o_i^{\text{Img}}$，使用二元交叉熵损失训练
    - 设计动机：解决两个问题——(a) 标题生成推荐无法保证推荐物品存在于物品池中；(b) 扩展词表做物品token预测不可扩展。检索式方法同时保证可靠性和效率

4. **多特征类型扩展**:

    - 功能：在图像特征之外整合ID嵌入（来自SASRec）和文本特征
    - 核心思路：为每种特征类型引入独立的投影器对 $(F_u^*, F_i^*)$，推理时通过分数求和聚合：$rec_u^k = \text{Top-k}(r_{u,i}^{\text{Img}} + r_{u,i}^{\text{CF}} + r_{u,i}^{\text{Text}})$
    - 设计动机：不同特征互补——图像特征在冷启动物品上更优，CF特征在热门物品上更强，文本特征提供额外语义

### 损失函数 / 训练策略
总训练目标：$\mathcal{L}_{final} = \mathcal{L}_{\text{RISA}} + \mathcal{L}_{\text{RERI}}^{\text{Img}} + \mathcal{L}_{\text{RERI}}^{\text{CF}} + \mathcal{L}_{\text{RERI}}^{\text{Text}}$

所有损失权重固定为1，LLM参数冻结，仅训练适配器 $M$ 和6个投影器。推理时计算三种特征的亲和度分数并求和取Top-k。

## 实验关键数据

### 主实验
在Amazon四个数据集（Sports, Grocery, Art, Phone）上对比CF模型和LLM模型：

| 方法 | 类型 | Sport Hit@5 | Sport NDCG@5 | Art Hit@5 | Phone Hit@5 |
|------|------|-------------|--------------|-----------|-------------|
| SASRec | CF | 0.3841 | 0.3129 | 0.5374 | 0.4366 |
| TALLRec | 属性LLM | 0.3801 | 0.2938 | 0.5663 | 0.4986 |
| A-LLMRec | CF+LLM | 0.4070 | 0.3352 | 0.5681 | 0.4502 |
| TRSR | 描述LLM | 0.4302 | 0.3375 | 0.5841 | 0.5148 |
| UniMP | 图像LLM | 0.4030 | 0.3364 | 0.5315 | 0.4427 |
| **I-LLMRec** | **图像LLM** | **0.4570** | **0.3711** | **0.5883** | **0.5156** |

I-LLMRec 在几乎所有数据集和指标上都优于TRSR（描述方法），同时推理速度快约2.93倍。

### 消融实验
| 配置 | Sport Hit@5 | Sport NDCG@5 | 说明 |
|------|-------------|--------------|------|
| RERI(Img only) | 0.3953 | 0.3043 | 仅图像检索，无对齐 |
| +RISA | 0.4316 | 0.3403 | 加入对齐后提升9.2% |
| RISA+RERI(Img+CF) | 0.4491 | 0.3630 | 多特征进一步提升 |
| Full model (Img+CF+Text) | 0.4570 | 0.3711 | 三特征完整模型 |

### 关键发现
- **RISA模块是核心贡献**：去掉RISA后Hit@5从0.432掉到0.395，说明推荐导向的对齐至关重要
- **图像+描述并不比单独图像更好**（I-LLMRec+D ≈ I-LLMRec），证实图像和描述之间信息高度重叠
- **冷热物品互补**：图像特征在冷启动物品上更优，CF在热门物品上更强，结合使用互补
- **上下文窗口鲁棒性**：当窗口缩小到256 tokens时，TRSR性能剧降但I-LLMRec几乎不受影响
- **噪声鲁棒性**：文本描述中常含HTML标签等噪声，图像天然避免这一问题

## 亮点与洞察
- **信息重叠的逆向利用**：以往多模态推荐将图文信息重叠视为障碍，本文反其道而行，利用这种重叠实现"以少代多"——用1个图像token替代160个文本token。这个视角转换非常巧妙
- **单token表示**：每个物品只用1个图像token就能承载丰富语义，使复杂度与用户序列长度的关系从二次降为近线性
- **检索式推荐框架**：RERI模块的设计可以轻松扩展到任意特征类型，只需添加投影器对。这种plug-and-play的设计可迁移到其他多模态检索场景
- **推荐导向对齐**：RISA不做通用对齐，而是专门为推荐场景设计问答模板。这种任务特定的对齐策略可推广到其他垂直领域（如医疗问答、金融分析）

## 局限与展望
- **图像质量依赖**：当商品图像缺失或质量差时（论文附录讨论了fallback方案），系统性能会受影响
- **数据集局限**：仅在Amazon电商数据集上验证，对于图像信息不丰富的领域（如书籍、音乐）是否有效未知
- **视觉编码器固定**：使用冻结的CLIP-ViT，未探索端到端微调视觉编码器的效果
- **简单的分数聚合**：多特征推理时简单求和，未探索更复杂的融合策略（如attention-based fusion）
- **可改进方向**：可以尝试用更强的视觉模型（如SigLIP-2）或引入时序感知的图像编码来建模用户兴趣演化

## 相关工作与启发
- **vs TALLRec**：TALLRec用属性+标题表示物品做LoRA微调，效率高但语义不足。I-LLMRec用图像弥补了语义缺陷同时保持效率
- **vs TRSR**：TRSR用大模型总结描述再喂给小模型，虽然语义丰富但token开销大且对噪声敏感。I-LLMRec通过图像完全绕过文本描述的问题
- **vs UniMP**：UniMP虽然也用图像但依赖预训练多模态LLM的通用对齐，在推荐场景下表现不如专门设计的RISA对齐
- 这篇论文对"多模态压缩表示"方向有启发：在任何需要长文本输入的LLM应用中，如果存在信息重叠的替代模态，都可以用类似思路压缩输入

## 评分
- 新颖性: ⭐⭐⭐⭐ 视角转换巧妙（信息重叠从障碍变优势），但图像替代文本的想法在多模态VLM中不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、效率/效果/鲁棒性多维分析、丰富的消融、冷热物品分组分析
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，motivation图示直观，trade-off分析到位
- 价值: ⭐⭐⭐⭐ 对LLM推荐系统的效率优化有实际意义，2.93倍加速+性能提升是实用的改进

<!-- RELATED:START -->

## 相关论文

- [Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)
- [CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems](../../ACL2025/recommender/cove_compressed_vocabulary_expansion_makes_better_llm-based_recommender_systems.md)
- [Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](../../AAAI2026/recommender/tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [Validating LLM-as-a-Judge Systems under Rating Indeterminacy](../../NeurIPS2025/recommender/validating_llm-as-a-judge_systems_under_rating_indeterminacy.md)
- [From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging](../../AAAI2026/recommender/from_parameter_to_representation_a_closed-form_approach_for_controllable_model_m.md)

<!-- RELATED:END -->
