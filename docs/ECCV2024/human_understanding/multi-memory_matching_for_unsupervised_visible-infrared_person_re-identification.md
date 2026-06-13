---
title: >-
  [论文解读] Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification
description: >-
  [ECCV 2024][人体理解][无监督行人重识别] 提出 Multi-Memory Matching（MMM）框架用于无监督可见光-红外行人重识别，通过跨模态聚类（CMC）、多记忆学习与匹配（MMLM）和软聚类级对齐损失（SCA）三个模块建立可靠的跨模态对应关系，在 SYSU-MM01 上 Rank-1 达到 61.6%，RegDB 上 Rank-1 达到 89.7%。
tags:
  - "ECCV 2024"
  - "人体理解"
  - "无监督行人重识别"
  - "可见光-红外跨模态"
  - "多记忆匹配"
  - "伪标签"
  - "聚类"
---

# Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification

**会议**: ECCV 2024  
**arXiv**: [2401.06825](https://arxiv.org/abs/2401.06825)  
**代码**: [GitHub](https://github.com/shijiangming1/MMM)  
**领域**: 人体理解  
**关键词**: 无监督行人重识别, 可见光-红外跨模态, 多记忆匹配, 伪标签, 聚类

## 一句话总结

提出 Multi-Memory Matching（MMM）框架用于无监督可见光-红外行人重识别，通过跨模态聚类（CMC）、多记忆学习与匹配（MMLM）和软聚类级对齐损失（SCA）三个模块建立可靠的跨模态对应关系，在 SYSU-MM01 上 Rank-1 达到 61.6%，RegDB 上 Rank-1 达到 89.7%。

## 研究背景与动机

无监督可见光-红外行人重识别（USL-VI-ReID）旨在不依赖标注的情况下在可见光和红外两种模态间进行行人检索，是实现 24 小时智能监控的关键技术。现有方法基于聚类生成伪标签并建立跨模态对应关系，但存在核心问题：

**跨模态对应关系不可靠**：作者引入 ARI（Adjusted Rand Index）指标评估发现，现有方法尽管在检索指标上表现不错，但其跨模态对应关系的质量很差（ARI 值低）

**单记忆表达不足**：现有方法用单一记忆（即单一聚类中心）代表一个身份，无法捕捉个体的多视角、多姿态等细微差异，导致跨模态匹配噪声大

**悖论现象**：具有重叠属性的不同行人因噪声对应关系而被进一步混淆，虽然可能在类间提高特征相似度带来指标提升，但实际精确检索能力受限

本文的核心洞察：**多记忆比单记忆更能完整表达一个身份的多样性特征（如正面、背面），从而建立更可靠的跨模态对应关系**。

## 方法详解

### 整体框架

MMM 采用 ResNet50（ImageNet 预训练）作为共享骨干网络提取 2048 维特征。整体流程为：(1) CMC 模块生成伪标签；(2) MMLM 模块通过多记忆匹配建立跨模态对应关系；(3) SCA 损失缩小模态差距并减轻噪声伪标签影响。

### 关键设计

1. **跨模态聚类（CMC）**：生成伪标签的基础模块

    - 使用 DBSCAN 算法分别对可见光样本、红外样本以及两者混合样本进行聚类：$Y^t = DBSCAN(F^t)$
    - 与现有方法不同，不仅进行模态内聚类（$t=v$ 或 $t=r$），还进行模态间联合聚类（$t=\{v,r\}$），间接建立跨模态对应
    - 为每个聚类计算三种记忆：可见光记忆 $C_{V^p}$、红外记忆 $C_{R^p}$、混合记忆 $C_{VR^p}$
    - 基于 ClusterNCE 对比损失优化：$L_{CMC} = L_V + L_R + L_{VR}$

2. **多记忆学习与匹配（MMLM）**：核心创新，建立可靠跨模态对应

    - **多记忆学习**：将单一聚类进一步细分为 $n$ 个子聚类（sub-cluster），每个子聚类的中心作为一个记忆。通过 K-Means 最小化子聚类内距离：
    $\min_{F_{C_{V_i^p}}} \sum_{i=1}^{n} \|f^v - K_{C_{V_i^p}}\|_2^2$
    - 例如 Memory 1 记录正面特征，Memory 2 记录背面特征，更完整地表达个体
    - **多记忆匹配**：将跨模态匹配问题建模为加权二部图匹配。设计代价矩阵为多记忆间最近邻距离之和：
    $M(K_{C_{V^p}}, K_{C_{R^{p'}}}) = \sum_{i=1}^{n} \min_{j \in \{1,...,n\}} \|K_{V_i^p} - K_{R_j^{p'}}\|_2$
    - 用匈牙利算法求解最优匹配 $Q$，将红外伪标签转移到可见光：$Y^v := QY^r$

3. **软聚类级对齐损失（SCA）**：缓解噪声伪标签影响并缩小模态差距

    - **置信度估计**：用双组分高斯混合模型（GMM）建模损失分布，通过后验概率计算每个样本的标签置信度 $W^v$
    - **置信度加权记忆更新**：用置信度加权更新各记忆，降低噪声样本的影响：$C_{V^p} := \frac{1}{N_p} \sum_i f(V_i^p) W_{V_i^p}$
    - **模态内对齐（Intra）**：将同一 ID 的样本向其置信度加权后的聚类中心对齐：$L_{Intra} = \sum_p \sum_{f^v} \|f^v - C_{V^p}\|_2^2 + \sum_p \sum_{f^r} \|f^r - C_{R^p}\|_2^2$
    - **模态间对齐（Inter）**：用 MMD²（Maximum Mean Discrepancy）度量同一 ID 在两个模态下的特征分布差异，最小化该差异实现软的多对多对齐：
    $L_{Inter} = \frac{1}{P} \sum_p \frac{1}{2}[D(F_p^v, sg(F_p^r)) + D(F_p^r, sg(F_p^v))]$
    - 使用 stop-gradient 操作防止两个模态相互坍塌
    - 总 SCA 损失：$L_{SCA} = \lambda_{Intra} L_{Intra} + \lambda_{Inter} L_{Inter}$

### 损失函数 / 训练策略

总损失：$L_{overall} = L_{CMC} + L_{SCA}$

- 骨干网络：ResNet50，ImageNet 预训练
- 训练 80 个 epoch，每步采样 8 个 ID，每个 ID 选 4 张可见光 + 4 张红外图像
- 图像尺寸 288×144，随机翻转和裁剪增强
- SGD 优化器，momentum=0.9，weight decay=5e-4
- Intra 损失从第 1 个 epoch 加入，Inter 损失从第 15 个 epoch 加入
- 温度系数 $\tau=0.05$，DBSCAN 参数 eps=0.6、min_samples=4
- 最优超参：$n=4$（记忆数），$\lambda_{Intra}=0.5$，$\lambda_{Inter}=0.05$

## 实验关键数据

### 主实验

**SYSU-MM01 All Search & RegDB Visible2Thermal**

| 方法 | 类型 | SYSU R-1 | SYSU mAP | RegDB R-1 | RegDB mAP |
|------|------|----------|----------|-----------|-----------|
| ADCA | USL | 45.5 | 42.7 | 67.2 | 64.1 |
| ADCA+MMM | USL | 49.7 | 44.7 | 77.8 | 70.9 |
| GUR* | USL | 61.0 | 57.0 | 73.9 | 70.2 |
| PCLHD | USL | 64.4 | 58.7 | 84.3 | 80.7 |
| **MMM** | **USL** | **61.6** | **57.9** | **89.7** | **80.5** |
| **MMM+PCLHD** | **USL** | **65.9** | **61.8** | **89.6** | **83.7** |
| DPIS | Semi | 58.4 | 55.6 | 62.3 | 53.2 |
| AGW | Sup. | 47.5 | 47.7 | 70.1 | 66.4 |

MMM 在无监督设置下超越多个半监督和有监督方法。在 RegDB 上相比 GUR 提升 Rank-1 +15.8%，mAP +10.3%。

### 消融实验

| 配置 | SYSU R-1 | SYSU mAP | Indoor R-1 | Indoor mAP |
|------|----------|----------|------------|------------|
| Baseline (CMC only) | 51.74 | 49.81 | 56.34 | 64.46 |
| + MMLM | 55.15 | 52.21 | 58.76 | 65.47 |
| + MMLM + Intra | 58.48 | 55.05 | 62.19 | 68.09 |
| + MMLM + Inter | 57.26 | 53.81 | 60.26 | 66.66 |
| + MMLM + Intra + Inter | **61.56** | **57.92** | **64.37** | **70.40** |

### 关键发现

- MMLM 模块带来 Rank-1 +3.41%，验证了多记忆比单记忆更有效地建立跨模态对应
- Intra 和 Inter 损失互补共同作用，完整 SCA 损失相比基线提升 Rank-1 +9.82%、mAP +8.11%
- 记忆数 $n=4$ 为最优，说明过少不够表达多样性，过多引入噪声
- 可视化分析表明 MMM 的模态内距离均值下降、模态间距离均值增大，特征分布更判别
- ARI 指标显示 MMM 的跨模态对应关系可靠性显著优于 GUR 等方法

## 亮点与洞察

1. **发现重要悖论**：揭示了现有 USL-VI-ReID 方法的跨模态对应关系虽然带来不错的检索结果，但实际上并不可靠的矛盾现象
2. **多记忆思想**：将单一聚类中心拆分为多个子聚类中心，更精细地描述身份的多样性，是对 Cluster-Contrast 范式的有效改进
3. **GMM 置信度估计**：利用损失分布建模来软化噪声伪标签的影响，比硬阈值过滤更优雅
4. **方法的通用性**：MMM 可作为插件增强其他方法（如 ADCA+MMM、MMM+PCLHD），验证了框架的广泛适用性

## 局限与展望

1. 作者坦言与有监督方法仍有差距（如 DEEN Rank-1 74.7% vs MMM 61.6%），主要受限于缺少跨模态数据标注
2. 多记忆的子聚类数 $n$ 需手动设定，不同数据集/身份可能需要不同值
3. DBSCAN 聚类参数对结果敏感，需精心调节
4. 可探索利用 CLIP 等视觉语言预训练模型引入语义先验来提升跨模态匹配质量
5. 计算多记忆的匈牙利匹配在大规模场景下可能成为瓶颈

## 相关工作与启发

- **Cluster-Contrast**：使用单一聚类中心的对比学习方法，MMM 是对其记忆表示的推广
- **PGM**：将跨模态匹配建模为二部图匹配，启发了 MMLM 的匹配策略
- **DivideMix**：使用 GMM 建模损失分布来处理噪声标签，启发了 SCA 的置信度估计
- **启发**：多记忆表示和软对齐策略可推广到其他无监督跨域/跨模态匹配任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 多记忆匹配思路新颖，ARI 指标揭示了现有方法的盲区
- **实验充分度**: ⭐⭐⭐⭐⭐ 两个标准数据集，三种设定（有监督/半监督/无监督）对比，消融到位
- **写作质量**: ⭐⭐⭐⭐ 问题分析深入，悖论现象的观察很有价值
- **价值**: ⭐⭐⭐⭐ 在无监督 VI-ReID 领域达到新 SOTA，框架可作为插件兼容其他方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification](../../AAAI2026/human_understanding/modality-aware_bias_mitigation_and_invariance_learning_for_unsupervised_visible-.md)
- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](../../ICCV2025/human_understanding/weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)
- [\[ECCV 2024\] Adaptive High-Frequency Transformer for Diverse Wildlife Re-Identification](adaptive_highfrequency_transformer_for_diverse_wildlife_reid.md)
- [\[ICML 2025\] LLaVA-ReID: Selective Multi-Image Questioner for Interactive Person Re-Identification](../../ICML2025/human_understanding/llava-reid_selective_multi-image_questioner_for_interactive_person_re-identifica.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](../../ICCV2025/human_understanding/openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)

</div>

<!-- RELATED:END -->
