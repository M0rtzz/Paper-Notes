---
title: >-
  [论文解读] LLM Interpretability with Identifiable Temporal-Instantaneous Representation
description: >-
  [NeurIPS 2025][LLM可解释性] 本文提出了一种面向 LLM 高维激活空间的可辨识时序因果表示学习框架，通过线性化公式同时建模时间延迟和瞬时因果关系，在保留理论可辨识性保证的同时解决了现有 CRL 方法无法扩展到 LLM 维度的计算瓶颈。
tags:
  - NeurIPS 2025
  - LLM可解释性
  - 因果表示学习
  - 稀疏自编码器
  - 时序因果关系
  - 可辨识性
---

# LLM Interpretability with Identifiable Temporal-Instantaneous Representation

**会议**: NeurIPS 2025  
**arXiv**: [2509.23323](https://arxiv.org/abs/2509.23323)  
**代码**: 无  
**领域**: causal_inference  
**关键词**: LLM可解释性, 因果表示学习, 稀疏自编码器, 时序因果关系, 可辨识性  

## 一句话总结

本文提出了一种面向 LLM 高维激活空间的可辨识时序因果表示学习框架，通过线性化公式同时建模时间延迟和瞬时因果关系，在保留理论可辨识性保证的同时解决了现有 CRL 方法无法扩展到 LLM 维度的计算瓶颈。

## 研究背景与动机

### 1. 领域现状

机械可解释性（Mechanistic Interpretability, MI）是理解 LLM 内部表征的重要研究方向。稀疏自编码器（SAE）已成为从 LLM 激活中提取可解释特征的主流工具，能够将高维激活分解为稀疏的单义（monosemantic）特征。Anthropic 等团队已将 SAE 成功扩展到大规模模型，实现了百万级特征的自动解读。

### 2. 现有痛点

现有 SAE 存在三个关键限制：
- **缺乏时序依赖建模**：将每个特征视为孤立表征，无法捕捉序列中特征之间的因果影响（如前一个 token 的概念如何影响后续 token）
- **缺乏瞬时关系表示**：没有机制表达同一时间步内特征之间的逻辑关系（如互斥、共现约束）
- **缺乏理论保证**：没有关于恢复特征唯一性的理论保证，提取的特征可能是任意或不稳定的变换

### 3. 核心矛盾

因果表示学习（CRL）社区已提出具有理论保证的框架，但面临严重的**可扩展性瓶颈**：现有方法依赖 Jacobian 计算，其时间和内存复杂度随维度超线性增长。当维度超过 1000 时，单次 Jacobian 评估就需要约 10 秒，而 CRL 训练需要百万次调用——这使得现有方法只能处理几十到几百个潜变量，远不能满足 LLM 中数千到数万个概念特征的需求。

### 4. 本文目标

设计一个计算高效的时序因果表示学习框架，能够扩展到 LLM 的高维概念空间（千级到万级维度），同时保留理论可辨识性保证并捕捉时间延迟和瞬时两种因果关系。

### 5. 切入角度

利用线性表示假设（linear representation hypothesis）——LLM 激活是潜在因果概念的线性生成。在线性假设下，Jacobian 退化为模型参数本身，绕开了非参数 CRL 方法的计算瓶颈。通过自协方差结构和非高斯性建立可辨识性，而非依赖"充分变化"假设。

### 6. 核心 idea

用线性时序因果模型替代非参数 CRL，以参数矩阵的稀疏约束实现 Jacobian-free 的高效训练，实现从几十维到上千维的数量级扩展。

## 方法详解

### 整体框架

数据生成过程分为两层：

1. **线性混合层**：观测激活 x_t = A·z_t，其中 A 是线性混合矩阵
2. **线性潜变量时序 SEM**：z_{t,i} = Σ_τ Σ_j B_{i,j,τ}·z_{t-τ,j} + Σ_j M_{i,j}·z_{t,j} + ε_{t,i}

其中 B_{i,j,τ} 编码时间延迟因果关系（lag τ 时 z_j 对 z_i 的因果效应），M_{i,j} 编码瞬时因果关系（同一时间步内 z_j 对 z_i 的因果效应），ε_{t,i} 是时空独立噪声。M 被约束为严格下三角矩阵，保证瞬时因果图 G_e 是 DAG。

### 关键设计

#### 模块 1：理论可辨识性保证（Theorem 3）

- **功能**：证明在 4 个假设下，模型参数可辨识到签名置换（signed permutation）
- **核心思路**：利用观测的自协方差矩阵 R_x(k) 构建 Yule-Walker 型递推，求解得到混合参数 C_τ = A(I-M)^{-1}B_τA^{-1} 和 HH^T；再利用非高斯性将正交不确定性约束为签名置换
- **4 个假设**：A1（时序白噪声，分量独立）、A2（A 满列秩，B_L 满秩）、A3（过程稳定性）、A4（至多一个分量为高斯）
- **设计动机**：现有非参数证明（如 IDOL 的充分变化假设）在线性情况下不成立（系数矩阵秩至多为 n，存在瞬时关系时无法满秩），需要全新的证明路径

#### 模块 2：分量级和子空间级可辨识性推论

- **Corollary 1（分量级可辨识性）**：若 M 的每列具有空支撑或唯一支撑元素（假设 A5），则在稀疏约束下 z_t 可辨识到置换和缩放
- **Corollary 2（子空间可辨识性）**：若 M 具有块对角结构（假设 A6），则 z_t 在子空间级别可辨识
- **设计动机**：Theorem 3 中，瞬时关系 M≠0 将不确定性从正交矩阵扩大为一般可逆矩阵。通过施加稀疏结构约束恢复更强的可辨识性

#### 模块 3：观测重建（Section 4.1）

- **功能**：使用线性自编码器实现观测向量 x_t 与潜变量 z_t 之间的可逆线性变换
- **核心思路**：L_r = E[Σ(x_t - x̂_t)²]，其中 x̂_t = Decoder(Encoder(x_t))
- **设计动机**：线性混合假设直接对应 SAE 的编码-解码结构

#### 模块 4：独立噪声估计（Section 4.2）

- **功能**：通过反向数据生成过程估计独立噪声 ε̂_t = ẑ_t - M̂·ẑ_t - Σ_τ B̂_τ·ẑ_{t-τ}
- **核心思路**：对噪声建模为各向同性拉普拉斯分布（而非高斯），最小化 KL 散度
- **设计动机**：线性情况下各向同性高斯分布的密度函数具有旋转不变性（如线性 ICA 文献所述），无法区分旋转变换，因此必须使用拉普拉斯分布

### 损失函数 / 训练策略

总损失函数由三部分组成：

**L_total = L_r + α·L_n + β·L_s**

- **L_r**（重建损失）：x_t 与 x̂_t 的均方误差
- **L_n**（噪声独立性损失）：估计噪声 ε̂_t 的 L1 范数（对应拉普拉斯分布的负对数似然）
- **L_s**（稀疏正则化）：Σ_τ ||B̂_τ||_1 + ||M̂||_1，对时间延迟和瞬时关系矩阵施加 L1 稀疏约束

关键训练约束：M 被限制为严格下三角矩阵，以匹配左右两侧的置换不确定性。

## 实验关键数据

### 主实验

#### 合成数据——可扩展性对比

| 方法 | 最大可处理维度 | 1024 维训练时间 | 1024 维 MCC |
|------|-------------|---------------|------------|
| iCITRIS | 16 | OOM | N/A |
| IDOL | 200 | OOM | N/A |
| 本文方法 | **1024+** | ~50h | **≈0.9** |

#### 真实 LLM 激活——SAEBench 定量评估

| 模型 | Recon. Loss ↓ | Sparse Prob. ↑ | Absorp. ↓ | Autointerp ↑ |
|------|-------------|--------------|-----------|-------------|
| ReLU SAE | 0.0110 | 0.6555 | 0.0141 | 0.6791 |
| TopK SAE | 0.0097 | 0.7141 | 0.0280 | 0.6822 |
| 本文方法 | 0.0108 | 0.6736 | 0.0139 | **0.6883** |

#### 半合成数据——关系恢复得分

| 方法 | Legal | XML | Email |
|------|-------|-----|-------|
| SAE+regression | 0.54 | 0.94 | 0.74 |
| 本文方法 | **19.95** | **8.63** | **2.66** |

### 消融实验

#### Jacobian 计算瓶颈分析

| 输入维度 | IDOL 单步 Jacobian 时间 | IDOL 单步 Jacobian 显存 |
|----------|---------------------|---------------------|
| 100 | ~0.1s | 可接受 |
| 500 | ~2s | 接近上限 |
| 1000 | ~10s | 超出 GPU 容量 |

#### 线性模型扩展性

从 128 维到 1024 维，MCC 始终保持在约 0.9，计算时间线性增长。

### 关键发现

1. **现有 CRL 方法根本无法处理 LLM 级别的维度**：IDOL 在 200 维以上 OOM，iCITRIS 在 16 维以上就失效
2. **本文方法在 SAEBench 上与标准 SAE 表现相当**：在概念恢复的基础指标上不逊色，同时额外获得了因果关系结构
3. **关系恢复得分远超基线**：在 Legal 文本上高出 37 倍，证明方法能有效恢复概念间的时序因果关系
4. **案例研究印证两类关系均有意义**：时间延迟关系如"appeals→affirmed"（法律文本程序性流程），瞬时关系如两个地理位置概念同时激活

## 亮点与洞察

1. **精准定位了 CRL 与 MI 之间的 gap**：CRL 有理论但不能扩展，SAE 能扩展但无理论，本文用线性假设优雅地桥接了二者
2. **理论贡献扎实**：独立于现有非参数证明（如 IDOL），在线性情况下建立了全新的可辨识性定理，并通过结构化假设推导出分量级和子空间级可辨识性
3. **实验设计有策略性**：半合成实验（对比法律文本 vs 非结构化文本）巧妙展示了方法发现领域特异性时序模式的能力
4. **发现的关系直观可解释**：如"国籍形容词→修饰名词"、"日期月份⇔完整日期"，展示了 LLM 内部真实的概念组织方式

## 局限与展望

1. **线性假设的局限**：LLM 内部机制本质上非线性（attention 机制、激活函数），线性模型是近似。虽然论文引用了线性表示假设的经验支持，但非线性扩展是必要的下一步
2. **单层分析**：仅分析单层的激活，未建模跨层的概念转换，而 LLM 的信息处理是逐层深入的
3. **评估困难**：真实 LLM 上没有 ground truth 因果结构，只能通过 case study 和间接指标评估，缺乏直接的因果关系验证
4. **模型规模有限**：主要实验基于 Pythia-160M，在更大模型（如 7B、70B）上的效果待验证
5. **瞬时关系的 DAG 假设**：要求瞬时因果图为 DAG，但 LLM 内部可能存在循环依赖

## 相关工作与启发

- **SAE 系列 (Anthropic, Cunningham et al.)**：本文的编码器-解码器结构直接继承 SAE，核心改进是添加了时序 SEM 模块来捕捉特征间关系
- **IDOL (Li et al., 2025)**：本文在 Appendix A.1 中详细论证了为何 IDOL 的非参数证明在线性情况下失效（系数矩阵秩不够），从而有必要发展新的证明路径
- **线性 ICA / SSM (Zhang, 2011)**：Theorem 3 的证明思路受此启发，将自协方差方法推广到包含瞬时关系的时序过程
- **Attribution Graphs / Sparse Feature Circuits**：这些方法通过注意力分数推断时间延迟影响，但缺乏独立噪声假设下的可辨识性保证
- **启发**：线性假设下的 CRL 可作为非线性方法的基础——先用线性模型捕捉主要因果结构，再逐步引入非线性修正

## 评分

- 新颖性: ⭐⭐⭐⭐ — CRL 与 SAE 的结合思路新颖，线性化解决扩展性的Strategy简洁有效，但线性假设本身限制了方法的普适性
- 实验充分度: ⭐⭐⭐⭐ — 合成数据验证理论、半合成数据验证关系恢复、真实数据案例研究，覆盖面广；但缺少大规模模型和 ground truth 验证
- 写作质量: ⭐⭐⭐⭐ — 论文结构清晰，理论-算法-实验层次分明，但符号较多，部分推导需反复查阅
- 价值: ⭐⭐⭐⭐ — 为 LLM 可解释性提供了新的因果视角和理论保证，但实际应用价值取决于线性假设的适用范围

<!-- RELATED:START -->

## 相关论文

- [Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation](../../ICML2025/causal_inference/transformer-based_spatial-temporal_counterfactual_outcomes_estimation.md)
- [Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](causality-induced_positional_encoding_for_transformer-based_representation_learn.md)
- [Towards Fine-Grained Interpretability: Counterfactual Explanations for Misclassification](../../CVPR2025/causal_inference/towards_fine-grained_interpretability_counterfactual_explanations_for_misclassif.md)
- [Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](../../ICML2025/causal_inference/learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits](../../ICLR2026/causal_inference/personax_multimodal_datasets_with_llm-inferred_behavior_traits.md)

<!-- RELATED:END -->
