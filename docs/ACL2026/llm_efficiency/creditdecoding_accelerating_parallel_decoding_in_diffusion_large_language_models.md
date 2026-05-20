---
title: >-
  [论文解读] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit
description: >-
  [ACL 2026][LLM效率][扩散语言模型] 本文提出 CreditDecoding，一种无需训练的并行解码加速方法，通过累积 token 级历史证据（轨迹信用）来增强正确但置信度不足的 token，在 LLaDA-8B-Instruct 上实现最高 5.48 倍加速且准确率提升 0.48。
tags:
  - "ACL 2026"
  - "LLM效率"
  - "扩散语言模型"
  - "并行解码"
  - "轨迹信用"
  - "推理加速"
  - "置信度增强"
---

# CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit

**会议**: ACL 2026  
**arXiv**: [2510.06133](https://arxiv.org/abs/2510.06133)  
**代码**: 无  
**领域**: 图像复原  
**关键词**: 扩散语言模型, 并行解码, 轨迹信用, 推理加速, 置信度增强

## 一句话总结

本文提出 CreditDecoding，一种无需训练的并行解码加速方法，通过累积 token 级历史证据（轨迹信用）来增强正确但置信度不足的 token，在 LLaDA-8B-Instruct 上实现最高 5.48 倍加速且准确率提升 0.48。

## 研究背景与动机

**领域现状**：扩散大语言模型（dLLMs）通过迭代去噪生成文本，支持双向注意力和并行 token 预测。现有并行解码方案在每步仅确认高置信度位置，将其他位置重新遮盖等待后续细化。

**现有痛点**：(1) 计算冗余——模型往往在实际解码前很多步就已预测出正确 token，但因置信度不够而反复重新遮盖和预测；(2) 历史无关决策——每步解码独立于前几步预测，未利用 token 的历史一致性信号，暂时的误预测可能导致稳定 token 置信度波动。

**核心矛盾**：正确的 token 因为置信度暂时不足而被反复重新遮盖，造成大量冗余计算；但直接降低解码阈值又会引入错误解码。

**本文目标**：设计一种利用历史预测一致性的机制，安全地提前解码正确 token，减少冗余迭代。

**切入角度**：分析去噪轨迹发现 token 的置信度展现出时间一致性——正确 token 的置信度在多步中持续上升，这提供了可利用的先验信息。

**核心 idea**：轨迹信用 = 跨步骤累积的历史 logits，作为先验与当前 logits 融合，使正确但低置信度的 token 提前越过解码阈值。

## 方法详解

### 整体框架

CreditDecoding 在标准并行解码的基础上增加一个 token 级信用评分系统：(1) 在每步去噪中记录每个位置的预测 token 和置信度；(2) 跨步骤累积信用分数；(3) 将信用以 log 增益的形式融合到当前 logits 中，提升正确 token 的置信度使其更早被解码。

### 关键设计

1. **轨迹信用（Trace Credit）**:

    - 功能：量化 token 在历史步骤中被持续预测为正确的可信度
    - 核心思路：对每个位置 $i$ 和候选 token $v$，累积跨步骤的历史 logits 得到信用分数 $C_t^{i,v}$。信用反映了候选 token 收敛到高置信度的可能性，提供自适应增益
    - 设计动机：单步置信度不稳定且早期偏低，但时间一致性表明正确 token 的置信度趋势是可预测的

2. **信用融合解码**:

    - 功能：将历史信用与当前 logits 融合加速解码
    - 核心思路：对目标 token 的 logit 添加 $\log X$ 形式的增益：$\hat{l}_t^{i,v} = l_t^{i,v} + \log X$，其中 $X$ 由轨迹信用自适应确定。增益使得正确 token 的后验概率更早超过解码阈值 $\tau$
    - 设计动机：最小增益公式 $X_{\min} = \frac{\tau}{1-\tau} \cdot (\frac{1}{p_t^{i,v}} - 1)$ 表明直接使用瞬时概率的增益高度敏感，用历史累积的信用提供更稳健的增益

3. **无调参变体**:

    - 功能：提供开箱即用的加速方案
    - 核心思路：自动根据去噪进度和信用分布确定增益参数，无需手动调节超参数
    - 设计动机：降低使用门槛，使 CreditDecoding 可作为通用加速插件

### 损失函数 / 训练策略

CreditDecoding 是完全无训练的推理时方法，仅修改解码策略。与现有优化（如 KV 缓存、算子融合）正交，可叠加使用。

## 实验关键数据

### 主实验

**LLaDA-8B-Instruct 在 8 个基准上的表现**

| 方法 | 加速比 | 准确率变化 | 说明 |
|------|--------|-----------|------|
| 标准并行解码 | 1× | 基线 | 阈值控制 |
| Fast-dLLM | ~3× | 略降 | 自适应步数 |
| **CreditDecoding** | **5.48×** | **+0.48** | 历史信用增强 |
| CreditDecoding + KV缓存 | 更高 | +0.48 | 正交叠加 |

### 消融实验

| 组件 | 效果 | 说明 |
|------|------|------|
| 无信用（纯阈值） | 基线 | 标准并行解码 |
| 仅当前步信用 | 轻微加速 | 无累积效果 |
| 完整轨迹信用 | 最大加速 | 历史累积关键 |
| 不同 dLLM 架构 | 均有效 | 方法通用性强 |

### 关键发现

- CreditDecoding 在知识、推理和代码三类基准上均实现加速且不损害准确率
- 加速效果随去噪步数增加而更显著——步数越多冗余越大
- 方法在 LLaDA、Dream 等不同 dLLM 架构上均有效
- 与 KV 缓存、算子融合等优化正交，可叠加获得更大加速
- 可扩展到长上下文场景

## 亮点与洞察

- "早期预测、晚期解码"的冗余分析揭示了 dLLM 推理的核心瓶颈
- 轨迹信用是对 token 预测时间一致性的优雅利用——简单的历史累积就能显著加速
- 无训练+正交的特性使其成为即插即用的实用工具

## 局限与展望

- 信用累积在极短序列或极少步数场景中可能无法积累足够信号
- 信用融合的线性增益假设可能不是所有场景的最优选择
- 仅在离散 token 的扩散模型上验证，对连续扩散模型的适用性未探索

## 相关工作与启发

- **vs 标准阈值解码**: 阈值解码忽略历史信息，CreditDecoding 利用时间一致性加速
- **vs Fast-dLLM**: Fast-dLLM 调整步数调度，CreditDecoding 从 token 置信度层面优化
- **vs KV 缓存**: KV 缓存优化计算开销，CreditDecoding 减少冗余步数，两者正交

## 评分

- 新颖性: ⭐⭐⭐⭐ 轨迹信用概念直观有效，对 dLLM 推理有独特洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 四种模型、八个基准、多种消融、正交性验证
- 写作质量: ⭐⭐⭐⭐ 分析清晰，可视化直观
- 价值: ⭐⭐⭐⭐⭐ 为 dLLM 推理加速提供了实用且通用的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)
- [\[ACL 2026\] Lizard: An Efficient Linearization Framework for Large Language Models](lizard_an_efficient_linearization_framework_for_large_language_models.md)
- [\[ACL 2026\] Are Large Language Models Economically Viable for Industry Deployment?](are_large_language_models_economically_viable_for_industry_deployment.md)
- [\[ACL 2026\] Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](tandem_riding_together_with_large_and_small_language_models_for_efficient_reason.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](multi-drafter_speculative_decoding_with_alignment_feedback.md)

</div>

<!-- RELATED:END -->
