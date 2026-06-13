---
title: >-
  [论文解读] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization
description: >-
  [AAAI 2026][推荐系统][跨域推荐] 提出 GenCDR 框架，通过领域自适应语义分词和跨域自回归推荐两大模块，首次将生成式语义 ID 范式引入 LLM 驱动的跨域推荐，有效解决传统方法中 item ID 不可迁移和领域个性化建模不足的问题。 跨域推荐（CDR）旨在利用用户在多个异质领域的交互行为来提升推荐精度和泛…
tags:
  - "AAAI 2026"
  - "推荐系统"
  - "跨域推荐"
  - "语义ID"
  - "生成式推荐"
  - "大语言模型"
  - "领域自适应"
---

# From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization

**会议**: AAAI 2026  
**arXiv**: [2511.08006](https://arxiv.org/abs/2511.08006)  
**代码**: [https://github.com/hupeiyu21/GenCDR](https://github.com/hupeiyu21/GenCDR)  
**领域**: 推荐系统  
**关键词**: 跨域推荐, 语义ID, 生成式推荐, 大语言模型, 领域自适应

## 一句话总结

提出 GenCDR 框架，通过领域自适应语义分词和跨域自回归推荐两大模块，首次将生成式语义 ID 范式引入 LLM 驱动的跨域推荐，有效解决传统方法中 item ID 不可迁移和领域个性化建模不足的问题。

## 研究背景与动机

跨域推荐（CDR）旨在利用用户在多个异质领域的交互行为来提升推荐精度和泛化能力。然而，现有方法面临两个核心挑战：

**Item 分词困境（Item Tokenization Gap）**：传统方法严重依赖共享用户/物品 ID 作为跨域知识迁移桥梁，但在真实场景中（如不同平台间），用户和物品 ID 往往不能对齐。直接使用原始 ID 会导致词汇表爆炸，且无法捕获高阶协作知识。

**领域个性化不足（Domain Personalization Gap）**：现有方法难以有效解耦和建模通用兴趣与领域特定表达之间的动态交互。例如，"Apple Watch"在科技领域和新鲜"Apple"在生活领域虽然共享语义概念，但具有截然不同的领域属性（健身 vs 甜味、维生素）。

现有 LLM-based CDR 方法要么将 LLM 作为特征增强器，要么将推荐任务重新定义为自然语言问题，但都没有从根本上解决上述两个挑战。本文的核心洞察是：**原始语义信息（如文本描述）天然可跨域迁移，而 item ID 则不行**。因此提出用离散语义 ID（SID）替代传统 ID。

## 方法详解

### 整体框架

GenCDR 框架包含三个核心模块：(a) 领域自适应分词模块（Domain-adaptive Tokenization），为物品生成解耦的语义 ID；(b) 跨域自回归推荐模块（Cross-Domain Autoregressive Recommendation），融合通用和领域特定兴趣建模用户偏好；(c) 领域感知前缀树（Domain-aware Prefix-tree），确保高效且准确的生成。整体流程为两阶段：先训分词器生成 SID，再训推荐模型。

### 关键设计

#### 1. **领域通用语义 Token 生成（RQ-VAE）**

基于残差量化变分自编码器（RQ-VAE）预训练，将所有物品的文本特征编码为离散语义码序列 $\mathbf{c} = (c_0, \dots, c_{M-1})$。编码器 $E$ 将特征嵌入 $\mathbf{x}$ 映射到隐含表示 $\mathbf{z}$，然后逐级在 $M$ 个码本中量化。训练目标包含三项：

- **重建损失** $\mathcal{L}_{\text{REC}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$
- **量化损失** $\mathcal{L}_Q$：使用 commitment 项对齐编码器输出与码本向量
- **掩码 Token 建模损失** $\mathcal{L}_{\text{MTM}}$：预测被掩码的语义码以确保上下文一致性

总预训练损失为 $\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{REC}} + \mu\mathcal{L}_Q + \lambda\mathcal{L}_{\text{MTM}}$。训练完成后冻结通用编码器和码本。

#### 2. **领域特定语义 Token 适配器（LoRA Adapters）**

通用编码器无法完全捕获领域差异性特征（如视频的视觉美学、书籍的叙事风格），因此为每个领域 $d$ 引入低秩适配（LoRA）模块：

$$h_{\text{out}} = W_0 h_{\text{in}} + B_d A_d h_{\text{in}}$$

冻结通用编码器权重 $W_0$，仅微调各领域的 $\theta_d = \{B_d, A_d\}$，使用自监督重建损失优化。

#### 3. **物品级动态语义路由网络（Dynamic Routing）**

为避免静态融合策略带来的负迁移风险，设计了逐物品的门控路由网络 $R_\phi$，动态平衡通用和领域特定表示：

$$\alpha = \sigma(R_\phi(\mathbf{x})), \quad \mathbf{z}_{\text{fused}} = (1-\alpha)\cdot\mathbf{z}_{\text{uni}} + \alpha\cdot\mathbf{z}_{\text{spec}}$$

并使用变分信息瓶颈（VIB）正则化路由器，通过 KL 散度约束路由器从输入中提取的信息量，促进解耦表示。

#### 4. **跨域自回归推荐模块**

对称地在用户侧进行通用-特定兴趣解耦：

- **通用兴趣建模**：在冻结 LLM 上添加混合多 LoRA 专家，用全领域数据训练，捕获可迁移的行为模式
- **领域特定兴趣适配**：冻结通用参数后，为每个领域训练专属 LoRA 适配器
- **用户级动态路由**：融合通用模型概率分布 $P_{\text{uni}}$ 和领域适配模型分布 $P_{\text{spec}}$：$P_{\text{final}} = (1-\gamma)\cdot P_{\text{uni}} + \gamma\cdot P_{\text{spec}}$

#### 5. **领域感知前缀树推理**

为每个领域构建离线前缀树，编码所有合法 SID 序列。推理时约束 LLM 仅在当前前缀的合法子集中生成，避免无效 ID 输出，同时大幅降低计算开销。

### 损失函数 / 训练策略

整体训练分为多阶段：
1. RQ-VAE 预训练（通用语义 Token）
2. 领域 LoRA 微调 + VIB 路由（领域自适应分词）
3. 通用兴趣 LoRA 混合微调（自回归推荐）
4. 领域特定 LoRA 微调 + 用户级路由

## 实验关键数据

### 主实验

数据集涵盖三个跨域场景对（均为真实数据）：Sports-Clothing（休闲）、Phones-Electronics（科技）、Books-Movies（娱乐）。

| 场景 | 领域 | 指标 | GenCDR（本文） | LLM4CDSR | TriCDR | TIGER |
|------|------|------|----------------|----------|--------|-------|
| 休闲 | Sports | R@5 | **0.0274** | 0.0263 | 0.0266 | 0.0267 |
| 休闲 | Clothing | R@5 | **0.0181** | 0.0176 | 0.0174 | 0.0173 |
| 科技 | Phones | N@5 | **0.0411** | 0.0401 | 0.0396 | 0.0315 |
| 科技 | Electronics | N@5 | **0.0235** | 0.0230 | 0.0231 | 0.0214 |
| 娱乐 | Books | R@5 | **0.0192** | 0.0161 | 0.0155 | 0.0172 |
| 娱乐 | Movies | R@5 | 见完整表格 | - | - | - |

GenCDR 在所有场景和指标上均显著优于现有 SOTA（包括单域 SDSR、生成式 GenRec 和跨域 CDSR 三类基线）。

### 消融实验

| 配置 | R@5 (Sports) | N@5 (Phones) | 说明 |
|------|-------------|--------------|------|
| GenCDR（完整） | **0.0274** | **0.0411** | 全部模块 |
| w/o 动态路由 | 下降 | 下降 | 验证路由必要性 |
| w/o 领域适配器 | 下降 | 下降 | 领域特定信息重要 |
| w/o VIB 正则 | 下降 | 下降 | 防止过拟合 |
| w/o 前缀树 | 有效性下降 | - | 确保生成合法性 |

### 关键发现

- GenCDR 在高稀疏度（>99.9%）的真实数据集上表现优异，验证了语义 ID 范式在跨域推荐中的有效性
- 动态路由机制比静态融合更稳健，能够根据每个物品/用户动态决定通用和领域特定知识的融合比例
- VIB 正则化有效防止路由器过拟合，促进解耦表示学习
- 领域感知前缀树在保证生成有效性的同时，显著提升推理效率

## 亮点与洞察

- **首次将生成式语义 ID 引入 LLM-based 跨域推荐**，优雅地解决了 item ID 不可迁移的长期难题
- **对称设计思想**：物品侧和用户侧采用对称的通用-特定解耦架构（LoRA + 动态路由），体现了框架的统一性
- **信息瓶颈正则化**的巧妙运用，从理论上保证了通用与领域特定表示的解耦
- 前缀树约束生成的设计简洁有效，解决了生成式推荐中无效 ID 的常见问题

## 局限与展望

- 实验数据集虽涵盖三个场景，但领域数量仅为 2，未验证大量领域（>5）时的可扩展性
- LoRA 适配器数量随领域线性增长，当领域非常多时参数管理成本增加
- 未探索冷启动场景（新领域或新用户），这在实际应用中非常重要
- 二阶段分词+推荐的训练流程较为复杂，端到端联合训练可能进一步提升性能

## 相关工作与启发

- 与 TIGER（单域生成式推荐）的关系：GenCDR 将其扩展到多域，增加了领域解耦机制
- 与 LLM4CDSR 的区别：后者直接用 LLM 做 CDR 但缺乏语义 ID 和领域适配
- LoRA 混合专家 + 路由网络的组合启发了一种通用的多域建模范式
- 对于多模态推荐系统，类似的通用-特定解耦思路同样适用

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将语义 ID 引入跨域推荐，对称解耦设计有新意
- 实验充分度: ⭐⭐⭐⭐ — 三个场景六个领域，多类基线对比和消融完备
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 为 LLM 时代的跨域推荐提供了新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](inductive_generative_recommendation_via_retrieval-based_speculation.md)
- [\[ACL 2025\] GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion](../../ACL2025/recommender/gram_generative_recommendation.md)
- [\[AAAI 2026\] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](align3gr_unified_multi-level_alignment_for_llm-based_generat.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)

</div>

<!-- RELATED:END -->
