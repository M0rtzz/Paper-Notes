---
title: >-
  [论文解读] MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE
description: >-
  [ICLR 2026][模型压缩][MoE剪枝] 提出 MoNE（Mixture-of-Novices-and-Experts），通过联合评估专家的访问频率和输出方差来识别冗余专家，并用其输出均值（"新手"常量向量）替换之，在5种MoE模型上实现比现有剪枝方法更有效且更鲁棒的压缩…
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "MoE剪枝"
  - "专家冗余"
  - "新手替换"
  - "结构化压缩"
  - "访问频率"
---

# MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE

**会议**: ICLR 2026  
**arXiv**: [2507.00390](https://arxiv.org/abs/2507.00390)  
**代码**: [GitHub](https://github.com/zxgx/mode-pd)  
**领域**: 模型压缩  
**关键词**: MoE剪枝, 专家冗余, 新手替换, 结构化压缩, 访问频率

## 一句话总结
提出 MoNE（Mixture-of-Novices-and-Experts），通过联合评估专家的访问频率和输出方差来识别冗余专家，并用其输出均值（"新手"常量向量）替换之，在5种MoE模型上实现比现有剪枝方法更有效且更鲁棒的压缩，25%剪枝率下平均准确率下降仅0.14。

## 研究背景与动机
MoE架构通过稀疏激活扩展模型容量，但部署时需将所有专家保留在内存中，带来巨大开销。例如将所有64个专家（包括未被激活的）放在GPU上。结构化剪枝可直接减少专家数量来降低内存成本。

现有方法存在三维不稳定性：

**跨架构不稳定**: 层剪枝（Angular）和通道剪枝（FLAP）未考虑MoE的稀疏计算特性

**跨校准数据不稳定**: 不同校准数据源导致性能波动大

**跨样本数不稳定**: 100/500/1000样本的效果差异显著

核心问题是：现有专家剪枝主要依赖访问频率，但频率不能完全刻画冗余性。一个频率不高但输出高度变化的专家可能携带关键区分信息；反之，一个频率不低但输出极稳定的专家可以被常数替代。

核心idea：冗余 = 低频率 × 低方差。冗余专家用其输出均值（"新手"）替换，而非简单删除，最小化输出差异。

## 方法详解

### 整体框架
MoNE三步走：①在校准集上评估每个专家的访问频率和输出方差；②融合两指标得冗余分数，识别最冗余的专家子集；③用新手（输出均值向量）替换冗余专家。路由器不变，被替换专家仍可被选中但输出为常数。

### 关键设计
1. **方差冗余 $\phi^{var}$**:

    - 功能：衡量专家输出在校准集上的稳定程度
    - 核心思路：计算专家 $E_i$ 被选中时输出的无偏方差估计的L2范数 $\phi_i^{var} = \left\|\sqrt{\frac{\sum(E_i(\mathbf{x}) - \overline{E_i})^2 \cdot \mathbb{I}(E_i \in \mathcal{S}_k)}{\sum\mathbb{I} - 1}}\right\|_2$
    - 设计动机：高方差专家提供更多区分性信息，不可替换为常数；低方差专家输出稳定，可用均值替代且引入最小差异

2. **频率冗余 $\phi^{freq}$**:

    - 功能：衡量路由器对专家的依赖程度
    - 核心思路：$\phi_i^{freq} = \frac{\sum G_i(\mathbf{x}) \cdot \mathbb{I}(E_i \in \mathcal{S}_k)}{\sum \mathbb{I}(E_i \in \mathcal{S}_k)}$，即被选中时的平均路由分数
    - 设计动机：低频率专家对模型输出贡献小，但仅靠频率不够——有些高频专家输出也很稳定

3. **新手替换**:

    - 功能：用常量向量替换剪枝专家
    - 核心思路：冗余分数 $\phi = \phi^{var} \cdot \phi^{freq}$，低分专家被标记剪枝。新手 $N_i = \overline{E_i}$（校准集上的输出均值），是最小化L2差异的闭合形式最优解
    - 设计动机：新手是输出均值的无偏估计，$N_i$ 不涉及与输入token的计算，存储仅需一个 $d$ 维向量

### 损失函数 / 训练策略
MoNE是无需训练的后处理方法。仅需在校准集上做一次前向传播收集统计量即可完成剪枝。

## 实验关键数据

### 主实验 (25%剪枝, OLMoE-7B, 100样本Zyda2)

| 方法 | Arc-c | Arc-e | BoolQ | COPA | MMLU | OBQA | PIQA | RTE | WinoG | Avg |
|------|-------|-------|-------|------|------|------|------|-----|-------|-----|
| 原始模型 | 49.23 | 76.89 | 70.09 | 85.0 | 53.54 | 44.4 | 79.76 | 71.84 | 68.90 | 66.63 |
| Angular(层剪枝) | 32.76 | 61.91 | 61.71 | 74.0 | 23.13 | 37.6 | 71.65 | 53.07 | 55.09 | 52.33 |
| FLAP(通道剪枝) | 40.53 | 67.55 | 62.69 | 78.0 | 41.16 | 37.8 | 74.81 | 61.37 | 60.93 | 58.32 |
| MC-SMoE(专家合并) | 35.67 | 54.92 | 63.49 | 73.0 | 29.04 | 30.6 | 67.19 | 55.23 | 65.75 | 52.77 |
| RS(频率剪枝) | 25.85 | 43.01 | 59.08 | 74.0 | 29.63 | 36.2 | 66.16 | 56.68 | 59.98 | 50.07 |
| **MoNE** | **42.32** | **64.81** | **67.19** | **85.0** | **40.13** | **40.8** | **78.07** | **64.62** | **66.46** | **61.04** |

### 鲁棒性实验

| 维度 | 现有方法 | MoNE | 说明 |
|------|---------|------|------|
| 跨架构(5种模型) | 波动大 | 一致优于 | OLMoE/Moonlight/DS-V2/Qwen2-57B/Qwen3-30B |
| Zyda2 vs C4 | 差异显著 | 差异小 | 校准数据鲁棒 |
| 100/500/1000样本 | 波动大 | 稳定 | 样本量鲁棒 |
| Qwen2-57B-A14B@25% | 基线下降大 | 仅下降0.14 | 大模型上优势更明显 |

### 关键发现
- MoNE在Qwen2-57B上25%剪枝仅下降0.14准确率，比基线最高提升2.72
- 频率和方差信息互补：高频高方差(蓝)、仅高方差(红)、仅高频(绿)三类专家跨任务一致
- RS（仅频率）表现最差，验证了单一指标的不足
- 新手（常数向量）替换比完全删除更好：保留了知识估计，且减少了部分token的激活参数

## 亮点与洞察
- "新手"替换概念简洁而有效——用一个向量代替整个MLP，实现近零计算/内存开销
- 频率×方差的冗余度量设计有说服力，且跨任务一致性强
- 无需训练、无需权重更新的特性使其实际部署非常方便
- 跨3个维度的鲁棒性分析是亮点，填补了现有方法评测的不足

## 局限与展望
- 仅在零样本评估，复杂任务（代码、数学、推理）可能需要额外微调
- 新手是固定常数，不随输入变化——对于高度数据依赖的专家可能偏差较大
- 50%高剪枝率下性能下降仍然明显
- 未探索分层学习适应性的新手（如低秩新手）

## 相关工作与启发
- **vs MC-SMoE**: 新手替换比专家合并更简单且更有效
- **vs RS**: 增加方差维度后全面领先，单一频率指标不够
- **vs Angular/FLAP**: MoE专用剪枝比通用结构化剪枝更合适

## 评分
- 新颖性: ⭐⭐⭐⭐ "新手"替换概念新颖，双指标冗余度量有洞察力
- 实验充分度: ⭐⭐⭐⭐⭐ 5模型×2校准源×3样本量的全面鲁棒性分析
- 写作质量: ⭐⭐⭐⭐ 动机清晰，但符号较重
- 价值: ⭐⭐⭐⭐⭐ 对MoE部署有实际价值，方法简单高效

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] STUN: Structured-Then-Unstructured Pruning for Scalable MoE Pruning](../../ACL2025/model_compression/stun_moe_pruning.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[ICLR 2026\] KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](kbvq-moe_klt-guided_svd_with_bias-corrected_vector_quantization_for_moe_large_la.md)
- [\[ACL 2026\] Two-Stage Regularization-Based Structured Pruning for LLMs](../../ACL2026/model_compression/two-stage_regularization-based_structured_pruning_for_llms.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](steering_moe_llms_via_expert_deactivation.md)

</div>

<!-- RELATED:END -->
