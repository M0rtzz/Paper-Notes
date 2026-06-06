---
title: >-
  [论文解读] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality
description: >-
  [CVPR 2026][医学图像][生存预测] 提出 MUST 框架，通过代数约束将多模态表征显式分解为模态特有和跨模态共享两部分，并用条件潜在扩散模型在模态缺失时生成特有信息，在五个 TCGA 癌症数据集上以 0.742 C-index 达到 SOTA，且在模态缺失场景下仅降约 0.4%-3.5%。
tags:
  - "CVPR 2026"
  - "医学图像"
  - "生存预测"
  - "缺失模态"
  - "代数分解"
  - "潜在扩散模型"
  - "多模态融合"
---

# MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality

**会议**: CVPR 2026  
**arXiv**: [2603.26071](https://arxiv.org/abs/2603.26071)  
**代码**: [项目主页](https://kylekwkim.github.io/MUST/)  
**领域**: 医学图像
**关键词**: 生存预测, 缺失模态, 代数分解, 潜在扩散模型, 多模态融合

## 一句话总结

提出 MUST 框架，通过代数约束将多模态表征显式分解为模态特有和跨模态共享两部分，并用条件潜在扩散模型在模态缺失时生成特有信息，在五个 TCGA 癌症数据集上以 0.742 C-index 达到 SOTA，且在模态缺失场景下仅降约 0.4%-3.5%。

## 研究背景与动机

1. **领域现状**：多模态生存预测（病理 WSI + 基因组）能显著提升预后评估精度，SurvPath、CMTA 等方法通过交叉注意力实现模态融合。
2. **现有痛点**：临床场景中模态频繁缺失——基因组检测昂贵且耗时、历史数据往往只有病理没有分子数据。现有多模态模型假设数据完整，缺失时性能骤降。
3. **核心矛盾**：现有缺失模态方法分三类——特征对齐（不知缺了什么）、插值（高维空间噪声大）、联合分布学习（未解耦模态特有 vs 共享信息）。根本问题是**没有显式建模每个模态的独特贡献**。
4. **本文目标** 在模态缺失时精确识别"丢了什么信息"，并针对性恢复。
5. **切入角度**：将模态表征做代数分解，在学到的低秩共享子空间中把每个模态拆成"特有分量"和"共享分量"，共享部分可从任一可用模态确定性恢复，特有部分用条件扩散模型生成。
6. **核心 idea**：通过代数可逆约束实现"缺什么补什么"的精确重建策略。

## 方法详解

### 整体框架

输入：病理 WSI 的 patch 特征集合 $P$ 和基因组 token 集合 $G$，经各自编码器得到全局表征 $g_P, g_G$。通过双向交叉注意力提取"对方包含的信息" $c_{P\leftarrow G}, c_{G\leftarrow P}$，再用自注意力提取模态特有分量 $u_P, u_G$。所有分量投影到低秩共享子空间后执行代数分解 $g_P = \hat{u}_P + \hat{c}_{G\leftarrow P}$。完整数据时，拼接三部分 $[\hat{u}_P; \hat{c}; \hat{u}_G]$ 送入预测头输出离散风险概率；模态缺失时，通过代数关系确定性恢复共享分量，再用 LDM 生成缺失的模态特有分量。

### 关键设计

1. **低秩共享子空间代数分解**:

    - 功能：将全局表征分解为模态特有和共享两部分
    - 核心思路：构造可学习低秩投影矩阵 $P_\cap = B_\cap B_\cap^T$（$B_\cap \in \mathbb{R}^{D\times r}$, $r\ll D$），满足幂等性。共享分量投影到子空间内，特有分量投影到正交补空间。三个约束：共享一致性（两方向交叉注意力结果一致）、模态间正交（$\hat{u}_P \perp \hat{u}_G$）、模态内正交（$\hat{u}_m \perp \hat{c}_m$）
    - 设计动机：不同于 ShaSpec 的隐式分布对齐，代数约束保证共享分量可从任一模态确定性恢复，为缺失模态重建提供"数学保证"

2. **条件潜在扩散模型 (LDM) 生成缺失特有分量**:

    - 功能：为真正不可从其他模态推断的特有信息提供高质量生成
    - 核心思路：冻结主网络参数后，训练 4 层 Transformer 去噪网络。以恢复的共享分量 $\hat{c}$ 和学到的模态特有 CLS token $[\text{CLS}_{u}]$ 作为条件，通过 DDIM 采样 50 步生成缺失的 $\hat{u}$。推理时生成 5 个样本取平均以降低随机性
    - 设计动机：将随机生成限制在"真正模态特有的残差"上，而非整个表征空间，大幅缩小生成难度

3. **渐进式两阶段训练**:

    - 功能：保证训练稳定收敛
    - 核心思路：第一阶段用生存损失 + 高斯噪声注入训练各模态编码器，让每个编码器先学到有意义的任务相关特征；第二阶段引入分解损失 $\mathcal{L}_{\text{decomp}}$、共享一致性损失 $\mathcal{L}_{\text{shared}}$、正交性损失 $\mathcal{L}_{\text{orth}}$
    - 设计动机：直接端到端训练分解框架容易陷入退化解，分阶段训练让编码器先有语义后再做结构化分解

### 损失函数 / 训练策略

- 第一阶段：$\mathcal{L}_{\text{warm}} = \mathcal{L}_{\text{surv}}(\phi([g_P; \epsilon_P])) + \mathcal{L}_{\text{surv}}(\phi([g_G; \epsilon_G]))$
- 第二阶段：$\mathcal{L}_{\text{main}} = \mathcal{L}_{\text{surv}} + \lambda_{\text{dec}}\mathcal{L}_{\text{decomp}} + \lambda_{\text{sh}}\mathcal{L}_{\text{shared}} + \lambda_{\text{orth}}\mathcal{L}_{\text{orth}}$
- LDM 阶段：标准扩散去噪损失 $\mathcal{L}_{\text{LDM}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t, \text{cond})\|^2]$
- 超参数：$\lambda_{\text{dec}}=1.0, \lambda_{\text{sh}}=1.0, \lambda_{\text{orth}}=0.5$，共享子空间秩 $r=64$，特征维度 $D=256$

## 实验关键数据

### 主实验

在 5 个 TCGA 癌症数据集（BLCA/BRCA/GBMLGG/LUAD/UCEC）上的 C-index 对比：

| 方法 | 设置 | BLCA | BRCA | GBMLGG | LUAD | UCEC | Overall |
|------|------|------|------|--------|------|------|---------|
| CMTA | 双模态完整 | 0.691 | 0.648 | 0.857 | 0.667 | 0.755 | 0.724 |
| **MUST** | **双模态完整** | **0.703** | **0.690** | **0.864** | **0.686** | **0.768** | **0.742** |
| LD-CVAE | 缺基因组 | 0.651 | 0.649 | 0.831 | 0.629 | 0.726 | 0.697 |
| **MUST** | **缺基因组** | **0.673** | **0.651** | **0.864** | **0.637** | **0.755** | **0.716** |
| ShaSpec | 缺病理 | 0.636 | 0.629 | 0.823 | 0.610 | 0.682 | 0.676 |
| **MUST** | **缺病理** | **0.702** | **0.692** | **0.865** | **0.690** | **0.748** | **0.739** |

### 消融实验

| 配置 | C-index (Overall) | 说明 |
|------|-------------------|------|
| 无热启动 | 降低 0.6-3.5% | 各数据集不等，UCEC 最明显 |
| LDM 仅用 $\hat{c}$ 条件 | 缺G: 0.712, 缺P: 0.732 | 缺少结构先验 |
| LDM 用 $[\hat{c}; \text{CLS}]$ | 缺G: 0.716, 缺P: 0.739 | CLS token 提供模态结构先验 |

### 关键发现

- 缺失病理时仅降 0.4%（0.742→0.739），缺失基因组降 3.5%（0.742→0.716）——说明 LDM 对高维噪声 patch 特征有"正则化去噪"效果
- BRCA/GBMLGG/LUAD 在缺病理时甚至性能微升，因为扩散生成过程滤除了 WSI 的高频噪声
- 分解保真度（cosine similarity）在 0.75-0.94 之间，验证代数分解的有效性
- 在 A6000 上完整数据推理 ≤70ms，缺失模态 879ms（50步 DDIM × 5样本），临床可接受

## 亮点与洞察

- **代数可逆性设计非常巧妙**：不同于 ShaSpec 的分布对齐，MUST 通过低秩投影 + 正交约束让共享分量可精确恢复，将不确定性严格限制在特有分量上。这使得缺失模态处理变成"确定性恢复 + 有限随机生成"
- **"缺失反而更好"的现象值得关注**：LDM 生成的病理特有分量因扩散去噪过程天然过滤了 WSI 的高维噪声，这为"数据增强式推理"提供了思路
- **渐进训练 + 噪声注入的组合**可迁移到其他多模态分解场景

## 局限与展望

- 仅处理两个模态（病理 + 基因组），扩展到 N 模态时两两交叉注意力的复杂度增长
- LDM 推理 879ms（5次采样取平均），在临床场景勉强可接受但仍较慢
- 分解保真度 0.75-0.94 说明代数分解并非完美，低保真情况下恢复的共享分量可能引入误差
- 可探索更轻量的生成模型（如 Flow Matching）替代 DDIM 降低采样步数

## 相关工作与启发

- **vs ShaSpec**: 同样尝试分离共享/特有信息，但 ShaSpec 用分布对齐（head distillation），缺乏代数可逆性保证，缺失时降幅更大（4.7% vs 3.5%）
- **vs LD-CVAE**: 联合分布学习但未解耦贡献，无法处理缺病理场景（单向架构），MUST 双向对称
- **vs CMTA**: 同样用交叉注意力但无缺失机制，MUST 证明"仅交叉注意力不够，需要代数框架防止模态坍缩"

## 评分

- 新颖性: ⭐⭐⭐⭐ 代数分解 + 条件扩散的组合很有创意，但整体框架仍是分解+生成的常见范式
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集、3种设置、完整消融、KM曲线分析、推理延迟分析
- 写作质量: ⭐⭐⭐⭐ 数学表述清晰，但符号较多，初读门槛高
- 价值: ⭐⭐⭐⭐ 临床场景模态缺失是真实痛点，方法实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modality_segmentation.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[AAAI 2026\] GROVER: Graph-guided Representation of Omics and Vision with Expert Regulation for Cancer Survival Prediction](../../AAAI2026/medical_imaging/grover_graph-guided_representation_of_omics_and_vision_with_expert_regulation_fo.md)
- [\[CVPR 2026\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)
- [\[CVPR 2025\] Knowledge Bridger: Towards Training-Free Missing Modality Completion](../../CVPR2025/medical_imaging/knowledge_bridger_towards_training-free_missing_modality_completion.md)

</div>

<!-- RELATED:END -->
