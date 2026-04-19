---
title: >-
  [论文解读] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition
description: >-
  [ICCV 2025][人体理解][全身生物特征识别] 本文提出 QME（Quality-guided Mixture of score-fusion Experts）框架，通过可学习的分数融合策略和基于模态质量的 MoE 路由机制，动态整合人脸、步态和行人重识别等多模态生物特征的相似度分数，在多个全身识别基准上实现了 SOTA 性能。
tags:
  - ICCV 2025
  - 人体理解
  - 全身生物特征识别
  - 分数融合
  - Mixture of Experts
  - 质量估计
  - 多模态识别
---

# A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition

**会议**: ICCV 2025  
**arXiv**: [2508.00053](https://arxiv.org/abs/2508.00053)  
**代码**: [Project Link](https://github.com/) (论文提到 "Code is available at the Project Link"，但未给出具体地址)  
**领域**: 人体识别与生物特征融合  
**关键词**: 全身生物特征识别, 分数融合, Mixture of Experts, 质量估计, 多模态识别

## 一句话总结
本文提出 QME（Quality-guided Mixture of score-fusion Experts）框架，通过可学习的分数融合策略和基于模态质量的 MoE 路由机制，动态整合人脸、步态和行人重识别等多模态生物特征的相似度分数，在多个全身识别基准上实现了 SOTA 性能。

## 研究背景与动机
全身生物特征识别（Whole-body biometric recognition）通过整合人脸识别（FR）、步态识别（GR）和行人重识别（ReID）等多种模态来克服单一模态的局限性。这在安防监控、执法等场景中至关重要——当人脸不清晰时可依赖步态信息，当衣着变化时可依赖人脸信息。

现有的多模态融合方法主要分为三类：决策级融合、特征级融合和分数级融合。特征级融合虽然理论上最优，但面临两大障碍：(1) 不同模态的特征空间维度不一致，对齐困难；(2) 缺少大规模配对多模态数据集——例如主流的人脸数据集不含全身信息，而行人数据集往往遮挡人脸且规模有限，无法支撑联合训练。

分数级融合更灵活、计算效率更高、对缺失模态更鲁棒，但传统方法（如 Z-score 归一化、加权平均）忽视了不同模态相似度分数分布的多样性，且难以自适应地为各模态分配最优权重（即使网格搜索也很困难）——不同质量的查询样本应该有不同的模态权重策略。

本文的核心切入点是：利用各模态预训练模型的中间特征来估计输入质量，然后让质量信息指导 MoE 路由器为不同"专家"分配权重，每个专家学习一种特定的分数融合策略。

核心 idea：用模态质量估计来动态路由多个分数融合专家，高质量模态的专家获得更高权重。

## 方法详解

### 整体框架
QME 包含三个阶段的 pipeline：(1) 各模态预训练骨干网络提取特征并计算相似度分数矩阵；(2) 质量估计器（QE）从骨干的中间特征中预测各模态质量权重；(3) MoE 分数融合层根据质量权重路由输入到不同专家，专家输出加权求和生成最终融合分数。整个训练分两阶段：先训练 QE，再冻结 QE 训练 MoE 融合层。

### 关键设计
1. **质量估计器（Quality Estimator, QE）**:

    - 功能：从预训练模型 $M_n$ 的中间激活特征 $\mathcal{I}_n \in \mathbb{R}^{L \times U \times P_n \times d_n}$ 中预测模态质量权重 $w_n \in \mathbb{R}$
    - 核心思路：提取多层 block 的均值和标准差特征，压缩为 $\mathbb{R}^{L \times 2d_n}$ 表示，输入编码器预测质量。
    - 训练使用伪质量损失 $\mathcal{L}_{rank}$：$$\mathcal{L}_{rank} = \sum_{i \in L} \text{MSELoss}(w_i, \text{ReLU}(\frac{\delta - r_i}{\delta - 1}))$$ 其中 $r_i$ 是查询特征 $q_i$ 在 gallery 中的排名结果，$\delta$ 是排名阈值超参数。排名越靠前意味着质量越高。
    - 设计动机：无需人工标注质量——排名结果作为代理标签。QE 可以泛化到任何预训练模型（不限于人脸），且可在域内或域外数据上训练。

2. **Mixture of Score-Fusion Experts (MoE)**:

    - 功能：将来自 $N$ 个模型的拼接分数矩阵 $\mathcal{S} \in \mathbb{R}^{T \times N}$ 经过多个融合专家 $\{\varepsilon_1, ..., \varepsilon_Z\}$（3 层 MLP），每个专家输出一个融合分数矩阵 $\mathcal{S}_z \in \mathbb{R}^{1 \times T}$
    - 核心思路：路由器 $\mathcal{N}_r$ 以 QE 预测的质量权重 $w_n$ 为输入，生成各专家的贡献权重 $\{p_1, ..., p_Z\}$。最终融合分数为加权和：$\mathcal{S}' = \sum_{z \in Z} p_z \mathcal{S}_z$。实验中 $Z=2$，$p_1 = w_n$，$p_2 = 1 - p_1$
    - 设计动机：与传统 MoE 从输入特征预测路由不同，相似度分数是高层语义特征，缺少细粒度质量线索。因此本文用独立的 QE 提供质量信息来指导路由，使得高质量模态对应的专家获得更大权重。

3. **分数三元组损失（Score Triplet Loss）**:

    - 功能：在分数域上施加约束，直接优化验证/开放集搜索等评估指标
    - 核心思路：$$\mathcal{L}_{score} = \text{ReLU}(\mathcal{S}'_{nm}) + \text{ReLU}(m - \mathcal{S}'_{mat})$$ 两项分别压低非匹配分数（使其趋向 0 以下）和保证匹配分数至少高于非匹配分数 $m$ 的间距
    - 设计动机：传统三元组损失只约束相对距离，不直接约束分数的绝对值。而 TAR@FAR 等指标依赖阈值判断，需要将非匹配分数的绝对值压低。本损失直接aligned训练目标与评估指标。

### 损失函数 / 训练策略
- 第一阶段训练 QE：使用 $\mathcal{L}_{rank}$ 损失
- 第二阶段训练 MoE：冻结 QE，使用 $\mathcal{L}_{score}$ 损失
- 所有预训练骨干网络冻结不动，仅训练轻量的 QE 编码器和 MoE 专家（3 层 MLP）
- Adam 优化器，学习率 $5 \times 10^{-5}$，余弦退火预热策略
- 使用 BatchNorm 层对拼接分数矩阵进行归一化
- 对使用欧氏距离的模型，通过 $1/(1+\text{Euc}(q,g))$ 转换为相似度分数

## 实验关键数据

### 主实验
| 数据集 | 指标 | 本文 QME | 最佳 baseline | 提升 |
|--------|------|---------|--------------|------|
| CCVID | Rank-1 | **94.1%** | 92.6% (SapiensID) | +1.5% |
| CCVID | TAR@1%FAR | **86.9%** (2模型) | 84.0% (GEFF) | +2.9% |
| MEVID | Rank-1 | **55.7%** | 54.1% (Z-score) | +1.6% |
| MEVID | TAR@1%FAR | **32.9%** | 30.7% (Passive MINT) | +2.2% |
| MEVID | FNIR@1%FPIR | **64.3%** | 65.9% (BSSF) | -1.6% |
| LTCC | Rank-1 | 提升显著 | - | - |
| BRIAR | 多指标 | SOTA | Farsight | 显著提升 |

### 消融实验
| 配置 | Rank-1 | TAR | 说明 |
|------|--------|-----|------|
| QME (完整) | **94.1** | **86.9** | 完整框架 |
| w/o QE（均匀权重） | 降低 | 降低 | 质量指导的路由是关键 |
| AdaFace-QE路由 | 92.6 | 75.0 | 以人脸质量路由 |
| CAL-QE路由 | 94.1 | 76.2 | 以 ReID 质量路由 |
| Weighted-sum | 91.7 | 73.6 | 固定权重基线 |
| Farsight | 92.0 | 73.9 | 学习型基线 |

### 关键发现
- 在人脸质量不佳的场景（MEVID、BRIAR）中，QME 的提升最为显著——因为动态路由能正确降低人脸模态权重
- CCVID 中人脸通常清晰可见，QME 的改进幅度相对较小，但依然达到SOTA
- 传统分数融合方法（Z-score、Min-max）在某些指标上表现不一致，而 QME 在所有指标上均保持最优或接近最优
- 使用不同模态的 QE 作为路由器产生不同效果，验证了质量估计确实影响了专家选择策略

## 亮点与洞察
- **无需重训骨干**：仅训练轻量的 QE 和 MoE（3 层 MLP），预训练模型完全冻结
- **伪质量标签**：利用排名结果作为质量代理标签的思路简洁实用，避免了人工标注
- **可扩展性强**：框架对模型组合、模态类型和相似度度量方式无特定要求，即插即用
- Score Triplet Loss 直接对齐训练目标和评估指标（TAR@FAR），是一个值得借鉴的技巧

## 局限与展望
- 专家数量固定为 $Z=2$（对应两个模态），扩展到更多模态时 MoE 设计可能需要调整
- QE 依赖预训练模型的中间特征，如果骨干模型更换则需要重新训练 QE
- 论文未探讨模态完全缺失时的处理策略（如某帧完全没有人脸检测结果）
- 在 CCVID 等人脸主导场景中提升有限，说明方法的优势集中在"困难融合"场景

## 相关工作与启发
- **vs Farsight**: 两者都是学习型分数融合方法，但 Farsight 使用固定的非对称聚合策略，而 QME 通过质量指导的动态路由更灵活
- **vs SapiensID**: SapiensID 是端到端多模态模型，需要大规模配对数据训练；QME 则在分数层面操作，无需修改任何骨干
- **vs GEFF**: GEFF 用超参数 $\alpha$ 硬编码两个模态的混合比例，无法扩展到三模态；QME 的 MoE 设计自然支持多模态

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 MoE 引入分数融合并用质量估计指导路由，思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、多种基线方法、多指标评估、消融分析全面
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，图表设计合理
- 价值: ⭐⭐⭐⭐ 对实际部署的多模态生物特征系统有直接应用价值

<!-- RELATED:START -->

## 相关论文

- [DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration](dynfacerestore_balancing_fidelity_and_quality_in_diffusion-guided_blind_face_res.md)
- [CarGait: Cross-Attention based Re-ranking for Gait Recognition](cargait_cross-attention_based_re-ranking_for_gait_recognition.md)
- [Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition](lvface_progressive_cluster_optimization_for_large_vision_models_in_face_recognit.md)
- [GENMO: A GENeralist Model for Human MOtion](genmo_a_generalist_model_for_human_motion.md)

<!-- RELATED:END -->
