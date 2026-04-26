---
title: >-
  [论文解读] Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification
description: >-
  [AAAI 2026][自动驾驶][行人重识别] 提出统一框架 HPL，通过任务路由 Transformer（双分类 token）解耦 I2I 和 T2I 任务，利用层次化提示学习（身份级 + 实例级伪文本 token）结合跨模态提示正则化，首次在单一模型中同时实现图像-图像和文本-图像行人重识别的 SOTA。
tags:
  - AAAI 2026
  - 自动驾驶
  - 行人重识别
  - 提示学习
  - 跨模态对齐
  - CLIP
  - 统一检索框架
---

# Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification

**会议**: AAAI 2026  
**arXiv**: [2511.13575](https://arxiv.org/abs/2511.13575)  
**代码**: [https://github.com/LH-Z-Ac/HPL-AAAI26](https://github.com/LH-Z-Ac/HPL-AAAI26)  
**领域**: 行人重识别 / 多模态检索  
**关键词**: 行人重识别, 提示学习, 跨模态对齐, CLIP, 统一检索框架

## 一句话总结

提出统一框架 HPL，通过任务路由 Transformer（双分类 token）解耦 I2I 和 T2I 任务，利用层次化提示学习（身份级 + 实例级伪文本 token）结合跨模态提示正则化，首次在单一模型中同时实现图像-图像和文本-图像行人重识别的 SOTA。

## 研究背景与动机

### 行人重识别的两大任务

行人重识别（ReID）根据查询模态分为两类：
- **I2I（Image-to-Image）**：给定查询图像，从大规模图库中检索同一人的其他图像。核心挑战在于提取对视角、背景变化鲁棒的身份判别性特征
- **T2I（Text-to-Image）**：给定自然语言描述，检索匹配的行人图像。核心挑战在于精确的跨模态语义对齐

### 联合训练的困境

现有方法通常将 I2I 和 T2I 作为独立任务处理。然而实际应用中，图像和文本查询可能共存，急需一个能同时处理两种任务的统一框架。

但直接在单一模型中联合训练两个任务，性能反而下降（如论文 Fig.1(a) 所示）。作者将此归因于**语义冲突**：
- I2I 关注**身份级语义**（衣着、性别等跨视角一致的特征）
- T2I 额外依赖**实例级属性**（如"手持桌子"等在文本中描述但在 I2I 中被忽略的细节）
- 这些不一致的监督信号形成冲突的优化方向，导致两个任务相互干扰

### 提示学习的启发

CLIP-ReID 等工作展示了提示学习在 I2I ReID 中的成功——通过注入身份级语义显著提升性能。但直接扩展到 T2I 任务并不简单：T2I 还需要实例级的细粒度语义（动作、手势、随身物品等），这些语义因样本和视角而异。

**核心思路**：设计层次化提示结构——"A photo of [id-tokens] and [inst-tokens] person"，其中身份级 token 提供稳定的身份锚点，实例级 token 通过模态特定的逆向网络动态生成，捕获样本特有的属性。

## 方法详解

### 整体框架

框架包含三大核心模块：
1. **Task-Routed Transformer (TRT)**：双分类 token 实现任务特定编码
2. **Hierarchical Prompt Learning (HPL)**：层次化提示生成与对齐
3. **Cross-Modal Prompt Regularization (CMPR)**：跨模态提示正则化

训练分两阶段：提示构建（Stage I）→ 表示学习（Stage II）。

### 关键设计

#### 1. **Task-Routed Transformer (TRT)**

在 CLIP 视觉编码器中增加一个额外的分类 token，形成双 token 设计：
- $v_{t2i}^i$：原始分类 token，针对 T2I 跨模态对齐优化
- $v_{i2i}^i$：新增分类 token，针对 I2I 身份判别优化

视觉特征提取：

$$F_i^v = [v_{t2i}^i, v_{i2i}^i, v_1^i, \ldots, v_N^i] = \mathcal{M}^v(V_i)$$

多目标监督策略：

$$\mathcal{L}_{base} = \mathcal{L}_{sdm} + \mathcal{L}_{id}^{t2i} + \mathcal{L}_{tri} + \mathcal{L}_{id}^{i2i}$$

其中 $v_{t2i}$ 由跨模态相似度分布匹配和跨模态身份分类损失优化，$v_{i2i}$ 由身份分类和三元组排序损失优化。

**设计动机**：ViT 中分类 token 通过自注意力自然聚合上下文信息，而聚合的语义受任务目标引导。双 token 设计以极轻量的方式（仅增加一个 token）实现了任务解耦，让共享骨干网络同时服务两个任务。

#### 2. **Hierarchical Prompt Learning (HPL)**

**层次化提示构建**：设计模板 "A photo of [id-tokens] and [inst-tokens] person"：
- **[id-tokens]**：固定数量的可学习 token，编码身份级语义
- **[inst-tokens]**：由模态特定逆向网络动态生成的伪文本 token

逆向网络从视觉/文本特征生成伪文本 token：

$$P_i^t = \mathcal{I}_t(F_i^t), \quad P_i^v = \mathcal{I}_v(F_i^v)$$

其中 $\mathcal{I}_v$ 和 $\mathcal{I}_t$ 由 4 层 Transformer 块组成。生成的伪 token 插入模板中：
- $T_i^v$: "A photo of [id-tokens] and $P_i^v$ person."
- $T_i^t$: "A photo of [id-tokens] and $P_i^t$ person."

反演一致性损失确保伪提示保留源模态语义：

$$\mathcal{L}_{ic} = \frac{1}{|B|}\sum_{i \in B}\|\tilde{v}_{eos}^i - v_{t2i}^i\|_2^2 + \frac{1}{|B_{t2i}|}\sum_{i \in B_{t2i}}\|\tilde{t}_{eos}^i - t_{eos}^i\|_2^2$$

**层次化提示对齐**：
- T2I 任务使用完整层次提示（身份+实例），通过 ILPA 损失对齐：

$$\mathcal{L}_{ILPA} = \mathcal{L}_{tgps} + \mathcal{L}_{vgps}$$

- I2I 任务使用简化的身份级提示，通过跨模态身份分类损失对齐：

$$\mathcal{L}_{cic} = -\sum_{i \in B}\log \frac{\exp[\text{sim}(v_{i2i}^i, \tilde{r}_{eos}^{y_i})]}{\sum_{j=1}^{N_{id}}\exp[\text{sim}(v_{i2i}^i, \tilde{r}_{eos}^j)]}$$

**设计动机**：HPL 的三大优势：(1) 实例级伪 token 捕获细粒度属性，超越类别身份；(2) 双向跨模态对齐弥合模态鸿沟；(3) 身份和实例提示拼接后共同优化，兼顾身份一致性和实例特异性。

#### 3. **Cross-Modal Prompt Regularization (CMPR)**

实例级提示 $P_i^v$ 和 $P_i^t$ 可能编码模态特定偏差。CMPR 直接在提示 token 空间对齐两者：

$$\mathcal{L}_{CMPR} = \frac{1}{|B_{t2i}|}\sum_{i \in B_{t2i}}\|P_i^t - P_i^v\|_F^2$$

**设计动机**：确保从图像和文本分别生成的伪提示在语义上一致，减少跨模态语义漂移，提升文本引导的视觉检索效果。

### 损失函数 / 训练策略

**Stage I（提示构建，10 epochs）**：

$$\mathcal{L}_{construct} = \mathcal{L}_{t2i} + \mathcal{L}_{i2t} + \mathcal{L}_{ic}$$

编码器冻结，仅更新逆向网络和可学习提示。

**Stage II（表示学习，60 epochs）**：

$$\mathcal{L}_{total} = \mathcal{L}_{base} + \mathcal{L}_{cic} + \lambda_1 \mathcal{L}_{ILPA} + \lambda_2 \mathcal{L}_{CMPR}$$

其中 $\lambda_1 = 0.4$, $\lambda_2 = 0.06$。

## 实验关键数据

### 主实验

**T2I ReID 性能**：

| 数据集 | 指标 | HPL (本文) | Propot (MM'24) | UMSA (AAAI'24) |
|--------|------|-----------|----------------|----------------|
| CUHK-PEDES | Rank-1 | **76.28** | 74.89 | 74.25 |
| CUHK-PEDES | mAP | **70.90** | 67.12 | 66.15 |
| ICFG-PEDES | Rank-1 | **66.61** | 65.12 | 65.62 |
| RSTPReID | Rank-1 | **64.00** | 61.87 | 63.40 |
| RSTPReID | mAP | **53.13** | 47.82 | 49.28 |

**I2I ReID 性能**：

| 数据集 | 指标 | HPL (本文) | CLIP-ReID (AAAI'23) | TransReID (ICCV'21) |
|--------|------|-----------|--------------------|--------------------|
| Market1501 | Rank-1 | **95.99** | 95.50 | 95.20 |
| Market1501 | mAP | **89.82** | 89.60 | 88.90 |
| MSMT17 | Rank-1 | **91.04** | 88.70 | 85.30 |
| MSMT17 | mAP | **79.01** | 73.40 | 67.40 |
| DukeMTMC | Rank-1 | **90.35** | 90.00 | 90.70 |

### 消融实验

各模块贡献（CUHK-PEDES + Market1501）：

| TRT | HPL | CMPR | T2I Rank-1 | T2I mAP | I2I Rank-1 | I2I mAP |
|-----|-----|------|-----------|---------|-----------|---------|
| ✗ | ✗ | ✗ | 74.22 | 70.45 | 94.50 | 86.91 |
| ✓ | ✗ | ✗ | 75.27 (+1.05) | 70.80 | 95.36 (+0.86) | 88.98 (+2.07) |
| ✓ | ✓ | ✗ | 75.60 (+0.33) | 70.88 | 95.57 (+0.21) | 89.72 (+0.74) |
| ✓ | ✓ | ✓ | **76.28 (+0.68)** | **70.89** | **95.99 (+0.42)** | **89.82 (+0.10)** |

实例级对齐消融：

| $\mathcal{L}_{tgps}$ | $\mathcal{L}_{vgps}$ | T2I Rank-1 | I2I mAP |
|-------|-------|-----------|---------|
| ✗ | ✗ | 75.27 | 88.98 |
| ✓ | ✗ | 75.58 | 89.35 |
| ✗ | ✓ | 75.60 | 89.59 |
| ✓ | ✓ | 75.60 | 89.72 |

### 关键发现

1. TRT 的双 token 设计贡献最大（I2I mAP +2.07%），验证了任务解耦的必要性
2. HPL 单独使用提升有限，需配合 CMPR 才能发挥最大效果（CMPR 提供 +0.68% T2I Rank-1）
3. 视觉引导提示（$\mathcal{L}_{vgps}$）对 I2I 贡献更大，文本引导提示（$\mathcal{L}_{tgps}$）对 T2I 贡献更大
4. Grad-CAM 可视化显示：I2I token 关注衣着、体型等跨视角一致特征；T2I token 关注文本描述的特定物品（如手机、挎包），证明任务感知的注意力解耦确实发生

## 亮点与洞察

1. **统一框架的实际意义**：首次在单一模型中同时达到 I2I 和 T2I 双 SOTA，避免了部署两套模型的开销
2. **双 token 设计的优雅**：仅增加一个分类 token 就实现了任务路由，改动极小但效果显著
3. **层次化提示设计合理**：身份级提供稳定锚点，实例级提供细粒度自适应，组合覆盖了 ReID 的核心需求
4. **跨模态正则化是关键胶水**：CMPR 将 HPL 从"并行的双任务"提升为"协同的统一框架"

## 局限与展望

1. 需要同时有 I2I 和 T2I 数据集进行联合训练，数据准备成本较高
2. 逆向网络使用 4 层 Transformer，在轻量级部署场景中可能需要精简
3. 未探索 T2I 生成式方法（如用 LLM 合成更多描述）来扩展训练数据
4. 在更大骨干（如 ViT-L/14）上的效果未验证
5. 跨数据集泛化能力（如在 CUHK 上训练，在 RSTPReID 上测试）未讨论

## 相关工作与启发

- **与 CLIP-ReID 的关系**：直接扩展其身份级提示方案，增加了实例级动态提示，是自然且有效的改进
- **与 GET 的关系**：借鉴了提示逆向（prompt inversion）思想，将视觉特征翻译为伪文本提示
- **启发**：双分类 token 的设计思路可泛化到其他多任务学习场景，如同时做检测和分割

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 各组件都有先例，但组合方式和统一框架设计新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — 6 个数据集、详细消融、可视化分析完整
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富，动机阐述到位
- **实用价值**: ⭐⭐⭐⭐ — 统一框架减少部署成本，但需要配对数据集

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Debiased Dual-Invariant Defense for Adversarially Robust Person Re-Identification](debiased_dual-invariant_defense_for_adversarially_robust_person_re-identificatio.md)
- [\[CVPR 2025\] Modeling Thousands of Human Annotators for Generalizable Text-to-Image Person Re-identification](../../CVPR2025/autonomous_driving/modeling_thousands_of_human_annotators_for_generalizable_text-to-image_person_re.md)
- [\[AAAI 2026\] When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework](when_person_re-identification_meets_event_camera_a_benchmark_dataset_and_an_attr.md)
- [\[CVPR 2026\] FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts](../../CVPR2026/autonomous_driving/fedbprompt_federated_domain_generalization_person.md)
- [\[AAAI 2026\] Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning](backdoor_attacks_on_open_vocabulary_object_detectors_via_multi-modal_prompt_tuni.md)

<!-- RELATED:END -->
