---
title: >-
  [论文解读] Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts
description: >-
  [ICML 2025][概念瓶颈模型] 揭示概念模型（CBM）中的"泄漏中毒"现象——绕过概念瓶颈的信息泄漏在分布偏移下反而损害预测准确率，使概念干预失效，提出 MixCEM 通过置信度门控动态决定何时使用/丢弃泄漏信息，在分布内外均保持高准确率和有效干预。
tags:
  - ICML 2025
  - 概念瓶颈模型
  - 概念干预
  - 分布偏移
  - 泄漏中毒
  - 可解释性
---

# Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts

**会议**: ICML 2025  
**arXiv**: [2504.17921](https://arxiv.org/abs/2504.17921)  
**代码**: https://github.com/mateoespinosa/cem  
**领域**: 其他/可解释AI  
**关键词**: 概念瓶颈模型, 概念干预, 分布偏移, 泄漏中毒, 可解释性

## 一句话总结
揭示概念模型（CBM）中的"泄漏中毒"现象——绕过概念瓶颈的信息泄漏在分布偏移下反而损害预测准确率，使概念干预失效，提出 MixCEM 通过置信度门控动态决定何时使用/丢弃泄漏信息，在分布内外均保持高准确率和有效干预。

## 研究背景与动机

**领域现状**：概念瓶颈模型（CBM）先预测人类可理解的概念（如"条纹"、"黑色"），再从概念预测标签（如"斑马"），增强可解释性。关键优势是概念干预——专家在测试时纠正错误概念后模型自动更新预测。

**现有痛点**：
   - 训练概念标注通常不完整（部分概念缺少标注）→导致"不完整性鸿沟"（概念不足以准确预测标签）
   - 现有方法通过"旁路"机制（如动态概念嵌入/残差连接）让信息绕过概念瓶颈"泄漏"→在分布内场景下弥补了不完整性
   - 但本文发现：在分布偏移下，这种泄漏信息本身变得 OOD→"中毒"了模型的预测

**核心矛盾**：泄漏信息在 ID 场景下是有益的（弥补不完整概念），但在 OOD 场景下是有害的（泄漏本身变得不可靠）→需要在两者间动态切换。

**本文目标**：使概念模型在 ID 和 OOD 场景下都能有效响应概念干预。

**切入角度**：先准确诊断问题——"泄漏中毒"是一个此前未知的现象。然后设计 MixCEM 的置信度门控来自适应控制泄漏。

**核心 idea**：每个概念的嵌入 = 全局嵌入（无泄漏，安全但不完整） + 残差嵌入（有泄漏，完整但可能中毒），用置信度门控决定残差的权重——ID 样本信任残差，OOD 样本丢弃残差。

## 方法详解

### 整体框架
MixCEM 的概念嵌入由两部分混合：
1. **全局嵌入** $e_g$：与样本无关的固定概念嵌入（纯概念信息，无泄漏）
2. **残差嵌入** $e_r(x)$：依赖输入的动态嵌入（包含绕过瓶颈的额外信息）
3. **混合嵌入** $e(x) = (1-\gamma(x)) \cdot e_g + \gamma(x) \cdot e_r(x)$，其中 $\gamma(x)$ 是置信度门控

### 关键设计

1. **泄漏中毒的诊断**:

    - 功能：首次识别和分析概念模型在 OOD 下的泄漏中毒现象
    - 核心观察：bypass 机制（CEM、残差 CBM 等）在 ID 样本上概念干预后准确率显著提升，但在 OOD 样本上干预后准确率不升反降
    - 原因分析：残差嵌入编码了分布特定的捷径信息→OOD 下这些信息变得不可靠→即使概念被正确干预，残差中的"毒信息"仍然传递到分类器
    - 设计动机：准确诊断是有效治疗的前提——"泄漏中毒"是一个此前被忽视的设计考量
    - 实验证据：在 CUB 数据集上，CEM 在 ID 干预后准确率从 66%→85%，但 OOD 干预后从 60%→58%（不升反降！）

2. **MixCEM 的置信度门控**:

    - 功能：动态决定每个样本/每个概念使用多少残差嵌入
    - 核心思路：$\gamma(x) = \sigma(w^T \cdot [e_g; e_r(x); \text{conf}(x)])$，其中 $\text{conf}(x)$ 是输入的置信度估计
    - 行为模式：
        - ID 样本：$\gamma \to 1$（信任残差，利用泄漏弥补不完整性）
        - OOD 样本：$\gamma \to 0$（丢弃残差，回退到纯概念预测）
    - 设计动机：不是"有泄漏"或"无泄漏"的二选一→针对每个样本动态决定——ID 场景不牺牲性能，OOD 场景不受毒害

3. **干预时的行为**:

    - 功能：确保概念干预在 ID 和 OOD 下都有效
    - 核心思路：当概念被干预（修正为正确值）时，全局嵌入 $e_g$ 直接更新为正确概念对应的嵌入→不受残差污染
    - OOD 下 $\gamma \to 0$→干预效果直接通过全局嵌入传递→不被残差"稀释"
    - 设计动机：干预的价值在于"人类提供的外部知识"→不应该被内部泄漏信息覆盖

### 损失函数 / 训练策略
- 概念预测损失（交叉熵）
- 标签预测损失（交叉熵）
- 门控正则化：鼓励门控在 ID 训练分布上趋向 1（充分利用残差）
- 端到端训练
- 无需 OOD 数据——MixCEM 从 ID 数据学会在不确定时丢弃残差

## 实验关键数据

### 主实验
CUB-200（鸟类分类，不同类型OOD偏移）：

| 方法 | ID 无干预 | ID 干预后 | OOD 无干预 | OOD 干预后 |
|------|---------|---------|----------|----------|
| CBM（无泄漏） | 72.1% | 85.3% | 63.2% | 78.1% |
| CEM（有泄漏） | 80.2% | 88.5% | 60.5% | 58.2% ↓ |
| Residual CBM | 81.5% | 87.8% | 61.8% | 59.5% ↓ |
| **MixCEM** | **80.8%** | **88.2%** | **67.3%** | **80.5%** |

### 概念不完整场景

| 方法 | 完整概念 ID | 不完整概念 ID | 不完整概念 OOD |
|------|----------|-----------|-----------|
| CBM | 85.3% | 72.1% | 63.2% |
| CEM | 88.5% | 80.2% | 60.5% |
| **MixCEM** | **88.2%** | **80.8%** | **67.3%** |

### 消融实验

| 配置 | ID Acc | OOD 干预后 Acc | 说明 |
|------|--------|-------------|------|
| 无残差（纯 CBM） | 72.1% | 78.1% | 安全但 ID 不完整 |
| 残差固定权重 0.5 | 76.5% | 65.2% | 不自适应 |
| 残差固定权重 1.0（CEM） | 80.2% | 58.2% | 泄漏中毒 |
| **置信度门控 $\gamma(x)$** | **80.8%** | **80.5%** | 自适应最优 |
| 无置信度输入（仅嵌入） | 78.5% | 72.3% | 置信度提供关键信号 |

### 关键发现
- 泄漏中毒是真实存在的——CEM 在 OOD 下干预后准确率实际下降（88.5→58.2 OOD）
- MixCEM 在 ID 下不牺牲性能（80.8% vs CEM 80.2%），在 OOD 下大幅改善（67.3% vs 60.5%）
- OOD 干预后的改善最显著——MixCEM 80.5% vs CEM 58.2%（+22.3%）
- 门控学到了正确的行为——ID 样本 $\gamma \approx 0.85$，OOD 样本 $\gamma \approx 0.15$

## 亮点与洞察
- **"泄漏中毒"**命名精准——信息泄漏在 ID 下是"补品"，在 OOD 下是"毒药"
- MixCEM 的门控设计极其优雅——不是"要不要泄漏"的非此即彼，而是"要多少泄漏"的连续调节
- 对可解释 AI 的实际部署有重要警示——如果概念模型在 OOD 下干预反而变差，那它在关键场景中的可靠性就受质疑
- 仅从 ID 数据训练就能学会 OOD 感知——因为不确定性/置信度是 OOD 检测的天然信号
- 与 CUDA（同会议）互补——CUDA 处理域适应，MixCEM 处理干预下的 OOD 鲁棒性

## 局限与展望
- 置信度估计本身在 OOD 下可能不可靠（过度自信的 OOD 样本）
- 门控机制增加了模型复杂度
- 仅在分类任务验证
- 多概念同时干预的交互效应未分析

## 相关工作与启发
- **vs CBM**: 无泄漏→安全但不完整
- **vs CEM**: 有泄漏→完整但 OOD 下中毒
- **vs MixCEM**: 自适应泄漏→两全其美
- **vs CUDA**: CUDA 做域适应时的概念对齐，MixCEM 做干预时的泄漏控制——互补关系

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "泄漏中毒"是重要的新发现
- 实验充分度: ⭐⭐⭐⭐⭐ 多偏移类型、有/无概念完整、有/无干预
- 写作质量: ⭐⭐⭐⭐⭐ 问题诊断深入，方法设计优雅
- 价值: ⭐⭐⭐⭐⭐ 对可解释AI的可靠部署有直接影响

<!-- RELATED:START -->

## 相关论文

- [Concept-Based Unsupervised Domain Adaptation](concept-based_unsupervised_domain_adaptation.md)
- [Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [Conditional Distribution Compression via the Kernel Conditional Mean Embedding](../../NeurIPS2025/interpretability/conditional_distribution_compression_via_the_kernel_conditional_mean_embedding.md)
- [Flexible Concept Bottleneck Model](../../AAAI2026/interpretability/flexible_concept_bottleneck_model.md)
- [Towards Faithful Multimodal Concept Bottleneck Models](../../CVPR2026/interpretability/towards_faithful_multimodal_concept_bottleneck_models.md)

<!-- RELATED:END -->
