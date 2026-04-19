---
title: >-
  [论文解读] Enhancing the Comprehensibility of Text Explanations via Unsupervised Concept Discovery
description: >-
  [ACL 2025 (Findings)][概念瓶颈模型] 提出 ECO-Concept 框架，通过 slot attention 机制自动提取文本概念，并利用 LLM 作为人类代理评估概念的可理解性，用可理解性反馈损失指导模型微调，在无概念标注的情况下实现了兼具高分类精度和人类可理解性的概念解释。
tags:
  - ACL 2025 (Findings)
  - 概念瓶颈模型
  - 注意力机制
  - LLM 作为人类代理
  - 自监督概念发现
  - 可理解性
---

# Enhancing the Comprehensibility of Text Explanations via Unsupervised Concept Discovery

**会议**: ACL 2025 (Findings)  
**arXiv**: [2505.20293](https://arxiv.org/abs/2505.20293)  
**代码**: [项目页面](https://vicki-sun.github.io/projects/ECO-Concept)  
**领域**: NLP / 可解释性 / 概念发现  
**关键词**: 概念瓶颈模型, Slot Attention, LLM 作为人类代理, 自监督概念发现, 可理解性

## 一句话总结

提出 ECO-Concept 框架，通过 slot attention 机制自动提取文本概念，并利用 LLM 作为人类代理评估概念的可理解性，用可理解性反馈损失指导模型微调，在无概念标注的情况下实现了兼具高分类精度和人类可理解性的概念解释。

## 研究背景与动机

基于概念的可解释方法（concept-based explanation）因为能将模型决策映射到人类可理解的属性上，受到越来越多的关注。但现有方法存在明显缺陷：

**监督方法依赖标注**：CBM 等方法需要大量预定义的概念标注，成本高且无法发现新概念。即使用 LLM 生成概念集合（如 TBM），LLM 本身的映射过程仍是黑箱

**无监督方法缺乏可理解性**：SelfExplain、PROTOTEX 等方法虽能自动提取概念，但提取的概念往往语义模糊、缺乏连贯性，甚至会高亮无关或误导性信息

**可理解性从未进入训练循环**：现有方法仅在实验阶段通过人工评估来衡量概念的可理解性，但从未将这种反馈融入训练过程来优化概念质量

核心洞察：如果能在训练过程中实时评估概念的可理解性并将其作为反馈信号，就能在保持分类性能的同时大幅提升概念的人类可理解性。

## 方法详解

### 整体框架

ECO-Concept 由三个模块组成：
- **概念提取器（Concept Extractor）**：基于 slot attention 的概念自动发现
- **分类器（Classifier）**：概念瓶颈层 + 全连接分类
- **概念评估器（Concept Evaluator）**：用 LLM 评估概念可理解性并生成反馈损失

训练分两阶段：先训练基础模型（概念提取 + 分类），再用可理解性损失微调。

### 关键设计

1. **基于 Slot Attention 的概念提取器**：

    - 核心思路：将 slot attention 从视觉目标识别迁移到文本概念发现
    - 输入文本经编码器得到 $\bm{X} \in \mathbb{R}^{L \times D}$，$M$ 个可学习的概念原型 $\bm{C} \in \mathbb{R}^{M \times D}$ 作为 query
    - 通过 dot-product attention 计算注意力矩阵 $\bm{A} \in [0,1)^{M \times L}$
    - 关键区别：softmax 沿概念轴（M 轴）归一化，引入**稀疏竞争**——每个 token 主要只与少数概念关联
    - 概念特征 $\bm{U} = \frac{\bm{A}}{\sum_l \bm{A}_{:,l}} (\bm{W}_v \bm{X})$
    - 设计动机：slot 之间的竞争机制自然促进了概念的多样性和专注性

2. **概念正则化**：

    - **一致性损失**（$\mathcal{L}_{con}$）：同一概念在不同样本上的特征应相似，选 mini-batch 中激活最高的 top-k 样本计算特征间距离
    - **区分性损失**（$\mathcal{L}_{dist}$）：不同概念的平均特征应尽量不同，最大化概念间距离
    - 这两个正则化器互相增强：一致性确保每个概念有清晰语义，区分性避免概念冗余

3. **LLM 驱动的可理解性评估与增强（核心创新）**：

    - **概念重要性**：$\beta_m = \frac{1}{|\mathcal{B}|} \sum_{\bm{t} \in \mathcal{B}} \bm{t}_m \sum_{\omega=1}^{\Omega} |\bm{W}_{m,\omega}|$，综合考虑激活值和分类器权重
    - **可理解性度量流程**：
        - (a) 从每个概念中选最高激活样本构建两个集合 $\mathcal{D}_{sum}$ 和 $\mathcal{D}_{high}$
        - (b) 将 $\mathcal{D}_{sum}$ 的样本和注意力值交给 GPT-4o，让其总结"这个概念关注什么"
        - (c) 如果 LLM 判断概念有语义意义，则用 GPT-4o-mini 在 $\mathcal{D}_{high}$ 的新样本上标注概念相关 token（0-1 标注矩阵 $\bm{S}$）
        - (d) 如果概念无语义意义，设 $\bm{S}$ 全零以压制该概念
    - **可理解性损失**：$\mathcal{L}_{com} = \frac{1}{M} \sum_{m=1}^{M} \frac{\beta_m \sum \|\bm{A}_{m,:} - \bm{S}_{m,:}\|_2^2}{|\mathcal{B}_m \cap \mathcal{D}_{high}|}$
    - 迭代微调直到所有概念的语义稳定

### 损失函数 / 训练策略

- **第一阶段**：$\mathcal{L} = \mathcal{L}_{ce} + \lambda_{con}\mathcal{L}_{con} + \lambda_{dist}\mathcal{L}_{dist}$
- **第二阶段（加入可理解性）**：$\mathcal{L} = \mathcal{L}_{ce} + \lambda_{con}\mathcal{L}_{con} + \lambda_{dist}\mathcal{L}_{dist} + \lambda_{com}\mathcal{L}_{com}$
- 超参设置：$\lambda_{con}=0.1$，$\lambda_{dist}=-0.01$，$\lambda_{com}=1$
- 可理解性增强阶段通常在 3 轮内收敛
- 编码器使用 RoBERTa，概念数量固定为 20

## 实验关键数据

### 主实验：分类性能（表格）

| 方法 | CEBaB | Beer | Hotel | IMDB | AGnews | Twitter | SciCite |
|------|-------|------|-------|------|--------|---------|---------|
| RoBERTa (黑箱) | .682/.797 | .882/.882 | .981/.981 | .937/.937 | .941/.960 | .828/.812 | .858/.879 |
| CBM (监督) | .669/.802 | .883/.885 | .979/.979 | - | - | - | - |
| SelfExplain (无监督) | .683/.799 | .873/.872 | .978/.979 | .936/.936 | .925/.949 | .817/.806 | .856/.873 |
| **ECO-Concept** | **.697/.808** | **.885/.885** | **.981/.981** | **.937/.937** | **.941/.961** | **.828/.813** | **.860/.881** |

> ECO-Concept 在所有数据集上都达到或超过黑箱模型，同时远超无监督基线。

### 概念可理解性评估（表格）

| 方法 | CEBaB (Sem/Dist/Con) | IMDB (Sem/Dist/Con) | AGnews (Sem/Dist/Con) |
|------|---------------------|---------------------|----------------------|
| Cockatiel | .50/.40/.47 | .35/.40/.41 | .65/.60/.41 |
| Concept-Shap | .25/.30/.42 | .40/.30/.32 | .35/.35/.35 |
| ProtoTEx | .45/.45/.35 | .25/.25/.36 | .40/.45/.41 |
| **ECO-Concept** | **.60/.60/.51** | **.65/.65/.52** | **.70/.65/.54** |

> ECO-Concept 在语义性、区分性、一致性三个维度上全面领先。

### 关键发现

1. **无需概念标注即可达到监督方法水准**：在 CEBaB 上甚至超过 CBM
2. **可理解性增强不损害分类性能**：消融实验显示加入 $\mathcal{L}_{com}$ 后精度基本不变
3. **人类前向模拟实验**：ECO-Concept 的解释帮助用户推断模型输出的准确率最高（Beer: 98.3%, AGnews: 86.7%），且信心评分也最高
4. **入侵者检测实验**：ECO-Concept 在所有任务上的概念入侵者检测准确率最高（至 90%）
5. **两个正则化器相互增强**：去掉一致性损失会同时降低区分性，反之亦然

## 亮点与洞察

- **LLM-in-the-loop 训练范式**：将 LLM 作为人类代理嵌入训练循环来优化可理解性，是非常新颖的思路。不同于后置评估，这是一种主动引导
- **Slot Attention 在 NLP 中的成功迁移**：证明了视觉领域的 object-centric 架构在文本概念发现中同样有效
- **可理解性的可操作化定义**：将"可理解性"操作化为"LLM 能否根据概念总结重建注意力分布"，这个定义既可量化又直觉合理
- **不损害性能的可解释性**：打破了"可解释性必然牺牲精度"的常见假设

## 局限与展望

1. **概念数量固定**：当前框架中概念数 M 不可在训练中自适应调整
2. **依赖 API LLM**：使用 GPT-4o/4o-mini 评估概念，成本较高且受限于 API
3. **编码器较小**：仅在 BERT/RoBERTa 级别模型上验证，未扩展到 LLaMA 等大模型
4. **子集采样评估**：出于成本限制，仅在样本子集上进行可理解性评估
5. **未探索生成式任务**：仅在分类任务上验证，生成/问答等任务适用性未知

## 相关工作与启发

- **BotCL (Wang et al., 2023)** 和 **CCTs (Hong et al., 2024)** 在视觉中使用 slot attention 做概念可解释性，是本文在文本领域的直接延伸
- **Bills et al. (2023)** 的神经元自动解释方法启发了本文的可理解性度量设计
- **CBM (Koh et al., 2020)** 是概念瓶颈模型的经典框架
- 启发：LLM 作为人类代理的"可理解性评估器"可以扩展到更多可解释性场景

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4.5 | LLM-in-the-loop 可理解性优化 + slot attention 文本迁移，非常新颖 |
| 实验充分度 | 4.5 | 7 个数据集 + 三种人类评估 + 充分消融 |
| 写作质量 | 4 | 结构清晰，方法阐述详细 |
| 价值 | 4 | 为无监督概念可解释性提供了实用且有原则的解决方案 |

<!-- RELATED:START -->

## 相关论文

- [Self-Foveate: Enhancing Diversity and Difficulty of Synthesized Instructions from Unsupervised Text via Multi-Level Foveation](self-foveate_enhancing_diversity_and_difficulty_of_synthesized_instructions_from.md)
- [U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models](../../CVPR2026/others/clipfree_label_free_unsupervised_concept_bottlenec.md)
- [Partial Colexifications Improve Concept Embeddings](partial_colexifications_improve_concept_embeddings.md)
- [Synthia: Novel Concept Design with Affordance Composition](synthia_novel_concept_design_with_affordance_composition.md)
- [Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)

<!-- RELATED:END -->
