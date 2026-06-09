---
title: >-
  [论文解读] DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration
description: >-
  [ECCV 2024][可解释性][离线签名验证] 提出DetailSemNet用于离线签名验证，通过Detail-Semantics Integrator将特征解耦为细节和语义两个分支分别处理，并引入基于EMD的局部结构匹配，在多个多语言签名数据集上取得SOTA。
tags:
  - "ECCV 2024"
  - "可解释性"
  - "离线签名验证"
  - "局部结构匹配"
  - "特征解耦"
  - "Earth Mover距离"
  - "细节-语义集成"
---

# DetailSemNet: Elevating Signature Verification through Detail-Semantic Integration

**会议**: ECCV 2024  
**arXiv**: [2511.16364](https://arxiv.org/abs/2511.16364)  
**代码**: [https://github.com/nycu-acm/DetailSemNet_OSV](https://github.com/nycu-acm/DetailSemNet_OSV)  
**领域**: 可解释性  
**关键词**: 离线签名验证, 局部结构匹配, 特征解耦, Earth Mover距离, 细节-语义集成

## 一句话总结

提出DetailSemNet用于离线签名验证，通过Detail-Semantics Integrator将特征解耦为细节和语义两个分支分别处理，并引入基于EMD的局部结构匹配，在多个多语言签名数据集上取得SOTA。

## 研究背景与动机

**领域现状**：离线签名验证（Offline Signature Verification, OSV）是法证鉴定中的关键技术，广泛应用于银行和商业领域。近年来深度学习方法在OSV上取得了显著进展，但仍面临关键挑战。

**现有方法的痛点**：

**过度依赖全局特征**：现有方法通过全局特征进行相似度比较，忽略了签名中局部笔画结构和风格细节的差异。全局表征会破坏图像结构，丢失局部判别信息。如Fig.1所示，两个不同人的签名在整体上可能非常相似，但在patch级别存在明显差异。

**Transformer天然抑制高频信息**：多头自注意力模块会无差别地抑制高频信号，导致显著的细节信息丢失。实验表明，传统Transformer模型主要学习低频模式，即使输入富含高频细节也无法利用，EER在高频信息增加后几乎不降低。

**核心矛盾**：签名验证的本质在于比较隐藏在笔画中的细微风格特征，而非签名的整体内容。但现有方法缺乏对局部结构的精细比较能力，且backbone（尤其是Transformer）在特征提取中丢失了关键的高频细节。

**核心idea**：设计多分支网络分别提取和处理细节（高频）与语义（低频）特征，并通过局部patch级别的结构匹配替代单纯的全局距离度量。

## 方法详解

### 整体框架

模型处理参考签名R和查询签名Q的图像对。首先预处理为二值图像和前景掩码，经过Patch Embedding后进入四阶段特征提取backbone（每阶段包含Patch Embedding + 多层DSI模块）。输出的token特征 $f^\mathcal{R}$ 和 $f^\mathcal{Q}$ 用于计算全局距离 $dis_{global}$ 和结构距离 $dis_{struct}$，最终组合距离为：

$$dis = \lambda_0 \times dis_{global} + dis_{struct}$$

### 关键设计

1. **Detail-Semantics Integrator (DSI)**：核心特征增强模块，将输入特征 $X$ 解耦为语义和细节两部分：

    - **语义特征** $Sem[X]$：通过局部平均池化提取低频部分
    - **细节特征** $Det[X] = X - Sem[X]$：相减得到高频部分
    - **SemanticsAttend分支**：对 $Sem[X]_{proj}$ 使用注意力模块提取上下文语义特征 $Y_{Sem}$
    - **SalientConv分支**：使用最大值滤波器+卷积层处理一半细节特征，保留显著特征
    - **DetailConv分支**：使用两层连续卷积处理另一半细节特征，提取精细高频信息
   
   三个分支的输出沿通道维度拼接，通过残差卷积层融合。**设计动机**：卷积擅长检测高频细节（相比注意力），因此将细节部分交给卷积处理；注意力擅长全局上下文聚合，因此处理语义部分。这种分工使得模型同时具备细节感知和语义理解能力。

2. **局部结构匹配（Structural Matching）**：使用前景掩码过滤背景token后，计算局部embedding间的成对余弦距离：

$$d_{ij} = 1 - \frac{r_i^T q_j}{\|r_i\| \|q_j\|}$$

形成距离矩阵 $D$，然后通过Earth Mover's Distance (EMD)求解最优匹配流 $F^*$，得到结构距离：

$$dis_{struct} = \frac{\sum_{i}\sum_{j} d_{ij} f_{ij}^*}{\sum_{i}\sum_{j} f_{ij}^*}$$

使用Sinkhorn算法通过熵正则化高效求解EMD。**设计动机**：EMD允许一对多匹配且对离群点不敏感，比Hausdorff距离更鲁棒，比简单的Chamfer距离更能捕捉细微差异。

3. **前景掩码过滤**：签名图像通常有大量空白背景。将输入图像resize到特征图大小后进行全局阈值二值化，生成前景掩码 $Mask$，过滤掉无信息的背景token，只保留包含笔画的有意义token进行匹配。

### 损失函数

使用双边界对比损失（Double-Margin Contrastive Loss）：

$$Loss_{DM} = y \cdot \max(0, dis - m)^2 + (1-y) \cdot \max(0, n - dis)^2$$

其中 $y=1$ 对应正样本对（真-真），$y=0$ 对应负样本对（真-伪），$m < n$ 为边界参数。

## 实验关键数据

### 主实验

| 数据集 | 指标 | DetailSemNet | 之前SOTA | 提升 |
|--------|------|-------------|----------|------|
| BHSig-H (Hindi) | EER↓ | **2.07%** | 3.39% (TransOSV) | -1.32% |
| BHSig-H | Acc↑ | **98.24%** | 96.61% (TransOSV) | +1.63% |
| BHSig-B (Bengali) | EER↓ | **2.11%** | 3.96% (CaC) | -1.85% |
| BHSig-B | Acc↑ | **98.19%** | 96.04% (CaC) | +2.15% |
| CEDAR (English) | EER↓ | **0.58%** | 1.75% (SDINet) | -1.17% |
| CEDAR | Acc↑ | **99.53%** | 96.16% (AVN) | +3.37% |

### 跨数据集零样本迁移（EER%）

| 训练→测试 | SigNet | CaC | TransOSV | **Ours** |
|-----------|--------|-----|----------|---------|
| BHSig-H→BHSig-B | 39.35 | 14.66 | 18.66 | **7.46** |
| BHSig-H→CEDAR | 40.43 | 29.49 | - | **14.05** |
| BHSig-B→BHSig-H | 35.43 | 30.41 | 17.17 | **15.91** |
| CEDAR→BHSig-H | 44.39 | 39.08 | - | **16.35** |
| CEDAR→BHSig-B | 35.85 | 38.07 | - | **8.40** |

### 消融实验

| SM | DetailConv | SalientConv | BHSig-H EER↓ | CEDAR EER↓ | ChiSig EER↓ |
|----|-----------|-------------|-------------|-----------|-------------|
| ✗ | ✗ | ✗ | 4.70 | 3.41 | 12.47 |
| ✓ | ✗ | ✗ | 4.67 | 1.99 | 10.69 |
| ✗ | ✓ | ✓ | 2.62 | 1.74 | 7.00 |
| ✓ | ✓ | ✗ | 2.51 | 1.09 | 8.65 |
| ✓ | ✗ | ✓ | 2.72 | 2.10 | 6.36 |
| ✓ | ✓ | ✓ | **2.07** | **0.58** | **5.85** |

### Backbone对比

| Backbone | BHSig-H EER↓ | BHSig-B EER↓ |
|----------|-------------|-------------|
| PVT | 4.62 | 2.72 |
| Swin | 4.24 | 10.27 |
| DAT | 4.94 | 20.09 |
| BiFormer | 4.38 | 8.66 |
| **Ours (DSI)** | **2.07** | **2.11** |

### 关键发现

1. **三个模块的贡献逐步叠加**：DSI的双分支（DetailConv + SalientConv）贡献最大，将EER从4.70降至2.62；加上SM进一步降至2.07
2. **高频信息至关重要**：随着测试图像中高频信息增加，DetailSemNet的EER从6.52%持续降至0.58%，而传统Transformer在3.41%处饱和
3. **跨语言泛化能力强**：在跨数据集零样本迁移中，DetailSemNet大幅超越所有对比方法，说明局部结构匹配学到的是语言无关的笔画特征
4. **SM在最后阶段效果最佳**：将Structural Matching放在第4阶段（EER 2.09%）优于第3阶段（3.47%）

## 亮点与洞察

- 问题洞察精准：通过频率分析实验直观展示了Transformer抑制高频信息的问题，为DSI设计提供了充分依据
- 特征解耦思路优雅：用简单的平均池化+相减实现频率解耦，计算开销极小
- 局部结构匹配显著提升可解释性：可视化匹配流展示了模型如何对齐patch，便于人类理解决策依据
- 跨语言泛化能力验证了方法学到的是底层笔画结构特征而非特定语言模式

## 局限与展望

- EMD求解（即使用Sinkhorn加速）在token数量很大时仍有计算开销
- 前景掩码使用简单阈值二值化，在复杂背景下可能需要更精细的分割方法
- 仅在离线签名验证上评估，可否扩展到文档认证、笔迹鉴定等相关任务？
- 未探索在线（online）签名验证中利用时序信息的可能性
- 训练数据规模相对较小（每人24-30个样本），数据效率仍可提升

## 相关工作与启发

- TransOSV [Li et al.] 首次将Transformer引入OSV，但忽略了高频信息丢失问题
- CaC [Lu et al.] 的循环观察方法提供了新的比较策略，但仍基于全局特征
- Re-ID领域的局部匹配方法（BPB, PAT等）可以借鉴，但OSV对细粒度差异的要求更高
- DSI的频率解耦思路可推广到其他需要同时关注全局语义和局部细节的任务（如细粒度识别、缺陷检测）

## 评分

- 新颖性: ⭐⭐⭐⭐ DSI的频率解耦设计和局部结构匹配相结合是OSV领域的显著创新
- 实验充分度: ⭐⭐⭐⭐⭐ 四个多语言数据集+跨数据集迁移+详细消融+backbone对比+可视化分析
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，图示丰富直观
- 价值: ⭐⭐⭐⭐ 在OSV任务上取得显著SOTA提升，且具有良好的可解释性和跨语言泛化能力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning Visual Composition through Improved Semantic Guidance](../../CVPR2025/interpretability/learning_visual_composition_through_improved_semantic_guidance.md)
- [\[ICLR 2026\] Conjuring Semantic Similarity](../../ICLR2026/interpretability/conjuring_semantic_similarity.md)
- [\[CVPR 2026\] VIRO: Robust and Efficient Neuro-Symbolic Reasoning with Verification for Referring Expression Comprehension](../../CVPR2026/interpretability/viro_robust_and_efficient_neuro-symbolic_reasoning_with_verification_for_referri.md)
- [\[ACL 2026\] On Emergent Social World Models -- Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models](../../ACL2026/interpretability/on_emergent_social_world_models_--_evidence_for_functional_integration_of_theory.md)
- [\[CVPR 2026\] TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment](../../CVPR2026/interpretability/tdatr_improving_end-to-end_table_recognition_via_table_detail-aware_learning_and.md)

</div>

<!-- RELATED:END -->
