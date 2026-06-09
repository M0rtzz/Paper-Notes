---
title: >-
  [论文解读] Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval
description: >-
  [AAAI2026][模型压缩][domain adaptive retrieval] 提出 PSCA 两阶段框架，通过正交 prototype 建立类级语义连接，结合几何-语义一致性对齐动态修正伪标签可靠性，并在重建特征上进行 hash 编码，显著提升跨域检索性能。
tags:
  - "AAAI2026"
  - "模型压缩"
  - "domain adaptive retrieval"
  - "hashing"
  - "prototype learning"
  - "伪标签"
  - "semantic alignment"
---

# Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval

**会议**: AAAI2026  
**arXiv**: [2512.04524](https://arxiv.org/abs/2512.04524)  
**代码**: 未公开  
**领域**: 模型压缩  
**关键词**: domain adaptive retrieval, hashing, prototype learning, pseudo-label correction, semantic alignment

## 一句话总结
提出 PSCA 两阶段框架，通过正交 prototype 建立类级语义连接，结合几何-语义一致性对齐动态修正伪标签可靠性，并在重建特征上进行 hash 编码，在多个跨域检索数据集上大幅超越现有方法。

## 研究背景与动机

### 领域现状

**领域现状**：哈希检索因其紧凑存储和高效计算优势在图像检索中广泛应用。Domain Adaptive Retrieval (DAR) 在此基础上进一步解决跨域场景问题——将有标签源域的知识迁移到无标签目标域，实现跨域和单域两种检索场景。已有方法如 PWCF 通过 focal-triplet 约束、DAPH 通过 MMD 分布对齐来缓解域差异，TSS、SGHL、DCS-LSG 则引入伪标签进行语义引导对齐。

### 现有痛点

**现有痛点**：现有 DAR 方法存在三个关键缺陷：(1) 过度追求 pair-wise 样本对齐——PWCF、TSS、SGHL 等方法逐对最小化语义一致样本间的分布差异，计算复杂度高（$O(n^2)$）且难以覆盖完整的数据分布；(2) 伪标签可靠性处理不足——目标域无标签时伪标签错误会导致偏差对齐和 hash code 质量下降，DCS-LSG 仅用语义共识评估，缺少几何知识的交叉验证；(3) 直接量化域偏移特征——将含域偏移的原始特征直接映射到 Hamming space 会产生大量量化误差。

### 核心矛盾

**核心矛盾**：伪标签的语义预测和特征空间的几何结构之间可能存在冲突——当两者一致时可以互相增强，但冲突时仍盲目信任语义预测会导致错误传播。现有方法要么忽略这种冲突，要么仅从单一视角评估标签可靠性。

### 解决思路

**本文目标**：设计一种同时利用几何近邻性和语义预测来自适应评估伪标签可靠性的框架。**切入角度**：通过正交 prototype 建立类级语义连接替代 pair-wise 对齐，在 prototype 空间中比较几何距离与语义预测的一致性来动态调整标签权重。**核心idea**：用几何-语义一致性对齐修正伪标签后，在重建特征（而非原始特征）上进行 hash 编码，从根本上提升编码质量。

## 方法详解

### 整体框架
PSCA 采用两阶段框架：阶段一通过正交 prototype 学习和语义一致性对齐获得可靠的软隶属矩阵；阶段二利用隶属矩阵和 prototype 重建语义增强特征，再在其上学习 hash codes。输入为源域有标签数据 $\mathcal{D}_s$ 和目标域无标签数据 $\mathcal{D}_t$，输出为统一的 binary hash codes $\mathbf{B} \in \{-1,1\}^{r \times n}$。

### 关键设计

1. **MMD 边际分布对齐 + 正交 Prototype 学习**:

    - 功能：建立域共享子空间并学习类级语义中心
    - 核心思路：投影矩阵 $\mathbf{P}$ 将两个域映射到共享 $q$ 维子空间并通过 MMD 缩小边际分布差异；同时学习 $c$ 个正交类原型 $\mathbf{O} \in \mathbb{R}^{q \times c}$（约束 $\mathbf{O}^\top \mathbf{O} = \mathbf{I}_c$），聚拢同类样本的同时最大化类间分离度。正交约束迫使不同 prototype 相互正交，保证最大可分性。
    - 设计动机：相比 pair-wise 对齐（$O(n^2)$ 复杂度），类级对齐仅需维护 $c$ 个 prototype，计算高效且覆盖完整的类分布

2. **语义一致性对齐（Semantic Consistency Alignment）**:

    - 功能：动态评估和修正伪标签可靠性，生成软隶属矩阵
    - 核心思路：设计软隶属矩阵 $\mathbf{R} \in \mathbb{R}^{n_t \times c}$，融合几何距离 $d_{ij} = \|\mathbf{P}^\top \mathbf{x}_{t_i} - \mathbf{o}_j\|_2^2$ 与语义伪标签信息。通过自适应权重 $\alpha_i$ 动态调控两种信号的贡献：当几何最近 prototype $k_\text{geo}$ 与语义预测 $k_\text{sem}$ 一致时，按语义/几何 margin 比值调整权重；冲突时按分歧程度 $|\pi_{i,k_\text{geo}} - \pi_{i,k_\text{sem}}|$ 降低语义贡献比重
    - 设计动机：解决伪标签错误传播问题——当语义和几何共识一致时互相增强信心，冲突时自动降低不可靠信号的影响，避免错误累积

3. **特征重建 + 域特定量化（Feature Reconstruction Hashing）**:

    - 功能：在语义增强的重建特征上学习高质量 hash codes
    - 核心思路：利用阶段一获得的 prototype $\mathbf{O}$ 和隶属矩阵 $\mathbf{R}$ 重建语义增强特征 $\widetilde{\mathbf{X}}$（目标域：$\widetilde{\mathbf{x}}_{t_i} = \sum_m r_{im} \mathbf{o}_m^\top$），拼接投影特征得到 $\mathbf{D} \in \mathbb{R}^{2q \times n}$。然后设计两个正交量化函数 $\mathbf{W}_s, \mathbf{W}_t$，在互相逼近约束 $\|\mathbf{W}_s - \mathbf{W}_t\|_F^2$ 下生成统一 binary hash codes
    - 设计动机：直接量化含域偏移的原始特征会引入大量量化误差；通过 prototype 重建后的特征具有更强的语义判别力，同时保留了投影特征的几何结构

### 优化与正则化
整体目标函数包含 MMD 对齐项、prototype 聚类项、$\ell_{2,1}$-norm 行稀疏约束（特征选择）和量化损失。采用交替优化策略分别更新 $\mathbf{P}$、$\mathbf{O}$、$\mathbf{R}$、$\mathbf{W}_s$、$\mathbf{W}_t$ 和 $\mathbf{B}$。Out-of-sample 阶段通过线性回归 $\mathbf{\Phi}$ 将新样本快速映射到 hash code。

## 实验关键数据

### 主实验

在 MNIST→USPS、COIL1→COIL2、Office-31 (A→D, A→W)、Office-Home (6 cases) 上评测，code length 从 16 到 128 bits。

| 方法 | MNIST→USPS | COIL1→COIL2 | A→D | A→W |
|---|---|---|---|---|
| DCS-LSG | 59.88% | 85.70% | 64.59% | 57.13% |
| TSS | 73.88% | 87.55% | 45.23% | 53.23% |
| SGHL | 71.46% | 83.00% | 59.91% | 55.64% |
| **PSCA** | **88.71%** | **90.76%** | **67.41%** | **65.78%** |

与深度方法对比（128-bit）：PSCA 在 MNIST→USPS 上较 COUPLE 高 15.89%，在 Office-Home 上平均高于 CPH 1.98%。

### 消融实验

| 变体 | MAP@128bit (MNIST→USPS) | 说明 |
|------|------------------------|------|
| PSCA-v1（去语义融合） | 61.38% | 仅靠几何结构，掉 27.33% |
| PSCA-v2（去一致性对齐） | 83.24% | 用硬伪标签，掉 5.47% |
| PSCA-v3（去 prototype） | 44.56% | 无类级结构，掉 44.15% |
| PSCA-v4（去特征重建） | 80.92% | 直接量化投影特征，掉 7.79% |
| **PSCA（完整模型）** | **88.71%** | — |

### 关键发现
- Prototype 学习贡献最大：去除后（v3）MAP 从 88.71% 暴跌至 44.56%，说明类级语义结构是框架的核心
- 语义融合信号不可或缺：v1 仅靠几何距离仅获得 61.38%，证明几何-语义双信号协同的必要性
- 特征重建效果显著：v4 直接量化投影特征损失 7.79%，验证了在重建特征上编码的优势
- 参数敏感性分析显示 $\lambda_1, \lambda_3$ 在 $[10^2, 10^4]$ 范围内性能稳定

## 亮点与洞察
- **类级对齐替代 pair-wise 对齐**：正交 prototype 高效建模语义结构，避免 $O(n^2)$ 计算开销，同时提供了比样本对更鲁棒的语义锚点
- **几何-语义双信号伪标签修正**：在一致和冲突两种情况下自适应调权，这种设计思路可推广到任何需要多源信号融合的场景
- **两阶段桥接设计**：特征重建巧妙连接 prototype 学习和 hash 编码，避免信息损失。这种"先纠正再编码"的思想对其他需要离散化的任务有借鉴价值

## 局限与展望
- 基于传统机器学习（非深度学习），在超大规模高维数据上可能受限于线性投影的表达能力
- 单域检索在 A→D 上提升有限（2.25%），域共享 prototype 可能过度平滑目标域的细粒度特征差异
- 参数 $\lambda_1, \lambda_2, \lambda_3$ 以及 $\sigma, \alpha$ 需要逐数据集调参，仅在 $[10^2, 10^4]$ 范围稳定
- 未与近期深度 DAR 方法（PEACE、CPH、COUPLE）在所有 code length 和数据集上全面对比

## 相关工作与启发
- **vs DCS-LSG**: 仅用语义共识评估伪标签，缺少几何校验；PSCA 的双信号一致性对齐在 MNIST→USPS 上提升 28.83%
- **vs TSS/SGHL**: 基于 pair-wise 语义对齐，计算代价高且对 outlier 敏感；PSCA 类级对齐更高效鲁棒
- **vs COUPLE (深度方法)**: 使用图 flow 扩散做跨域知识迁移；PSCA 以传统 ML 方法在部分数据集上超越深度方法，说明 prototype 机制的有效性
- 几何-语义一致性自适应权重的设计思路可以迁移到半监督学习中的伪标签选择

## 评分
- 新颖性: ⭐⭐⭐⭐ 几何-语义一致性自适应伪标签修正机制设计精巧
- 实验充分度: ⭐⭐⭐⭐ 4 数据集、多 code length、跨域+单域、消融+深度方法对比全面
- 写作质量: ⭐⭐⭐⭐ 公式推导严谨清晰，两阶段逻辑连贯
- 价值: ⭐⭐⭐ DAR 领域的扎实工作，但应用场景相对窄
---
title: >-
  [论文解读] Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval
description: >-
  [AAAI2026][模型压缩][domain adaptive retrieval] 提出 PSCA 两阶段框架，通过正交 prototype 建立类级语义连接，结合几何-语义一致性对齐动态修正伪标签可靠性，并在重建特征上进行 hash 编码，显著提升跨域检索性能。
tags:
  - AAAI2026
  - 模型压缩
  - domain adaptive retrieval
  - hashing
  - prototype learning
  - 伪标签
  - semantic alignment
---

# Prototype-Based Semantic Consistency Alignment for Domain Adaptive Retrieval

**会议**: AAAI2026  
**arXiv**: [2512.04524](https://arxiv.org/abs/2512.04524)  
**代码**: 未公开  
**领域**: model_compression  
**关键词**: domain adaptive retrieval, hashing, prototype learning, pseudo-label correction, semantic alignment

## 一句话总结
提出 PSCA 两阶段框架，通过正交 prototype 建立类级语义连接，结合几何-语义一致性对齐动态修正伪标签可靠性，并在重建特征上进行 hash 编码，显著提升跨域检索性能。

## 研究背景与动机
Domain Adaptive Retrieval (DAR) 旨在将有标签源域的知识迁移到无标签目标域，实现高效的跨域哈希检索。现有方法存在三个关键局限：

**过度关注 pair-wise 对齐**：PWCF、TSS、SGHL 等方法最小化语义一致样本对间的分布差异，计算代价高且覆盖有限

**伪标签可靠性不足**：目标域无标签，伪标签错误会导致偏差对齐和 hash code 质量下降；DCS-LSG 仅用语义共识评估，未利用几何知识

**直接量化对齐不完全的特征**：将含域偏移的原始特征直接映射到 Hamming space，量化误差大

## 方法详解

PSCA 采用两阶段框架：阶段一学习语义一致的 prototype 并生成可靠的软隶属矩阵；阶段二在重建特征上学习 hash codes。

### 阶段一：Prototype-Based Semantic Consistency Alignment
1. **MMD 边际对齐**：投影矩阵 $\mathbf{P}$ 将两个域映射到共享子空间，通过 MMD 缩小边际分布差异
2. **正交 Prototype 学习**：学习 $c$ 个正交类原型 $\mathbf{O} \in \mathbb{R}^{q \times c}$（$\mathbf{O}^\top \mathbf{O} = \mathbf{I}_c$），聚拢同类样本并最大化类间分离
3. **语义一致性对齐**：设计软隶属矩阵 $\mathbf{R}$，融合几何距离与语义伪标签，通过自适应权重 $\alpha_i$ 动态调控：
    - 几何与语义一致时（$k_\text{geo} = k_\text{sem}$）：语义 margin 大则增大 $\alpha_i$，几何 margin 大则减小
    - 冲突时：按分歧程度 $|\pi_{i,k_\text{geo}} - \pi_{i,k_\text{sem}}|$ 降低语义贡献
4. **$\ell_{2,1}$-norm** 约束投影矩阵行稀疏性，实现特征选择

### 阶段二：Feature Reconstruction Hashing
1. **特征重建**：利用 prototype $\mathbf{O}$ 和隶属矩阵 $\mathbf{R}$ 重建语义增强特征 $\widetilde{\mathbf{X}}$，拼接投影特征得到 $\mathbf{D} \in \mathbb{R}^{2q \times n}$
2. **域特定量化**：设计两个正交量化函数 $\mathbf{W}_s, \mathbf{W}_t$，在互相逼近约束下生成统一 binary hash codes
3. **Out-of-Sample**：线性回归 $\mathbf{\Phi}$ 将新样本映射到 hash code

## 实验关键数据

在 MNIST→USPS、COIL1→COIL2、Office-31 (A→D, A→W)、Office-Home (6 cases) 上评测，code length 从 16 到 128。

**跨域检索 MAP（128-bit）对比**：

| 方法 | MNIST→USPS | COIL1→COIL2 | A→D | A→W |
|---|---|---|---|---|
| DCS-LSG | 59.88 | 85.70 | 64.59 | 57.13 |
| TSS | 73.88 | 87.55 | 45.23 | 53.23 |
| SGHL | 71.46 | 83.00 | 59.91 | 55.64 |
| **PSCA** | **88.71** | **90.76** | **67.41** | **65.78** |

- MNIST→USPS 较次优高 **17.21%**，COIL 高 **3.94%**
- Office-Home 六个 case 平均 MAP 高 **8.82%**
- 单域检索同样全面领先

## 亮点与洞察
- **类级对齐替代 pair-wise 对齐**：正交 prototype 高效建模语义结构，避免 $O(n^2)$ 计算
- **几何-语义双信号伪标签修正**：在一致和冲突两种情况下自适应调权，有效减轻错误传播
- **特征重建桥接两阶段**：避免直接量化域偏移污染的原始特征，显著提升 hash code 质量
- 在小/中/大规模数据集上一致性优势，跨域和单域检索均大幅领先

## 局限与展望
- 基于传统机器学习（非深度学习），在超大规模数据上可能受限于特征维度和计算效率
- 单域检索在 A→D 上提升有限（2.25%），域共享 prototype 可能过度平滑目标域特征
- 参数 $\lambda_1, \lambda_2, \lambda_3$ 需要逐数据集调参，泛化性有待验证
- 未与近期深度 DAR 方法（PEACE、CPH、COUPLE）在所有设定下全面对比

## 评分
- 新颖性: ⭐⭐⭐⭐ — 几何-语义一致性自适应伪标签修正机制设计精巧
- 实验充分度: ⭐⭐⭐⭐ — 4 数据集、多 code length、跨域+单域、曲线分析全面
- 写作质量: ⭐⭐⭐⭐ — 公式推导严谨清晰，两阶段逻辑连贯
- 价值: ⭐⭐⭐ — DAR 领域的扎实工作，但应用场景相对窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Asymmetric Cross-Modal Knowledge Distillation: Bridging Modalities with Weak Semantic Consistency](asymmetric_cross-modal_knowledge_distillation_bridging_modalities_with_weak_sema.md)
- [\[NeurIPS 2025\] Mitigating Semantic Collapse in Partially Relevant Video Retrieval](../../NeurIPS2025/model_compression/mitigating_semantic_collapse_in_partially_relevant_video_retrieval.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](../../ACL2026/model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[ICML 2026\] MIC: Maximizing Informational Capacity in Adaptive Representations via Isotropic Subspace Alignment](../../ICML2026/model_compression/mic_maximizing_informational_capacity_in_adaptive_representations_via_isotropic_.md)
- [\[NeurIPS 2025\] AdmTree: Compressing Lengthy Context with Adaptive Semantic Trees](../../NeurIPS2025/model_compression/admtree_compressing_lengthy_context_with_adaptive_semantic_trees.md)

</div>

<!-- RELATED:END -->
