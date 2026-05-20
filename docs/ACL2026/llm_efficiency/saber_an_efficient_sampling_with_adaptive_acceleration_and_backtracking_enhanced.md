---
title: >-
  [论文解读] Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs
description: >-
  [ACL 2026][LLM效率][扩散语言模型] 本文提出 Saber，一个面向扩散语言模型（DLM）的免训练采样算法，通过自适应加速（根据已建立的上下文动态调整并行解码量）和回溯增强重遮蔽（撤销被新上下文证伪的 token）两种策略…
tags:
  - "ACL 2026"
  - "LLM效率"
  - "扩散语言模型"
  - "自适应采样"
  - "回溯重遮蔽"
  - "代码生成加速"
  - "速度-质量权衡"
---

# Saber: Efficient Sampling with Adaptive Acceleration and Backtracking Enhanced Remasking for DLMs

**会议**: ACL 2026  
**arXiv**: [2510.18165](https://arxiv.org/abs/2510.18165)  
**代码**: [GitHub](https://github.com/zhaoyMa/Saber)  
**领域**: 视频理解  
**关键词**: 扩散语言模型, 自适应采样, 回溯重遮蔽, 代码生成加速, 速度-质量权衡

## 一句话总结

本文提出 Saber，一个面向扩散语言模型（DLM）的免训练采样算法，通过自适应加速（根据已建立的上下文动态调整并行解码量）和回溯增强重遮蔽（撤销被新上下文证伪的 token）两种策略，在代码生成上平均提升 Pass@1 1.9% 的同时实现 251.4% 的推理加速。

## 研究背景与动机

**领域现状**：DLM（如 LLaDA、Dream）通过迭代去遮蔽实现并行生成，是自回归模型的有力替代。但在代码生成等结构约束强的任务上，减少采样步数会导致 Pass@1 灾难性暴跌（甚至超过 60%）。

**现有痛点**：(1) 静态加速策略（固定 token 数或置信度阈值）对简单阶段太保守、对复杂阶段太激进；(2) DLM 的解码是不可逆的——一旦 token 被解遮蔽就无法撤销，早期错误会永久锁定并传播。

**核心矛盾**：并行生成的速度优势 vs 错误传播导致的质量崩溃——需要同时解决非均匀难度和错误累积两个问题。

**本文目标**：设计一种能自适应调整并行度且允许自我修正的 DLM 采样方法。

**切入角度**：两个关键洞察——(1) 生成难度随上下文建立而递减（置信度单调上升）；(2) 已生成 token 的置信度会随新上下文变化（可能从高变低）。

**核心 idea**：自适应阈值 + 回溯重遮蔽——早期谨慎（少量解遮蔽）+后期激进（大量并行），同时允许撤销"后悔"的 token。

## 方法详解

### 整体框架

每步两阶段：(1) 自适应加速——用动态阈值 $\tau_t$（已解遮蔽 token 的平均置信度）决定哪些新 token 可被解遮蔽；(2) 回溯重遮蔽——重新评估已解遮蔽 token 在新上下文下的置信度，将置信度下降最大的 $\mu_t$ 个 token 重新遮蔽。

### 关键设计

1. **自适应动态阈值加速**:

    - 功能：根据生成进度自然调整并行度
    - 核心思路：$\tau_t = \frac{1}{|\mathcal{U}_{t-1}|} \sum_{j \in \mathcal{U}_{t-1}} c_j^{\text{unmask}}$，即已解遮蔽 token 的平均解遮蔽时置信度。所有置信度超过 $\tau_t$ 的遮蔽 token 被选入草稿集 $\mathcal{D}_t$
    - 设计动机：$\tau_t$ 自然随进度上升——早期上下文稀疏时均值低只解遮蔽最确定的 token，后期上下文丰富时均值高允许大量并行

2. **回溯增强重遮蔽**:

    - 功能：撤销在新上下文下被证伪的早期决策
    - 核心思路：重遮蔽数量 $\mu_t = \max(1, \lfloor |\mathcal{D}_t| / \mu \rfloor)$ 与当前步的激进程度成正比。对每个已解遮蔽 token 计算置信度下降 $\Delta_j = c_j^{t-1} - c_j^t$，选择下降最大的 $\mu_t$ 个重新遮蔽
    - 设计动机：传统 DLM 采样不可逆——过早锁定的错误 token 会破坏后续所有步骤的上下文。回溯机制使模型能"反悔"，从根本上解决错误传播问题

3. **无训练设计**:

    - 功能：可直接应用于任何 DLM 无需重新训练
    - 核心思路：Saber 仅修改采样过程中的 token 选择和撤销策略，不修改模型权重或架构
    - 设计动机：与改进 DLM 训练的研究正交——Saber 可以叠加在任何 DLM 上

### 损失函数 / 训练策略

免训练方法。在 LLaDA-8B-Instruct 上实验，温度 0，生成长度 256 token。

## 实验关键数据

### 主实验

**代码生成 Pass@1 和推理速度**

| 方法 | HumanEval Pass@1 | MBPP Pass@1 | 平均步数 | 相对加速 |
|------|----------------|------------|---------|---------|
| Confidence (标准) | 43.29 | 42.86 | 256 | 1.0x |
| Fast-dLLM | 38.54 | 38.95 | ~80 | ~3.2x |
| Saber | **45.12** | **44.76** | ~72 | ~3.5x |

### 消融实验

| 配置 | HumanEval Pass@1 | 说明 |
|------|----------------|------|
| Saber (完整) | 45.12 | 完整模型 |
| w/o 回溯 | 42.68 | 去掉回溯，质量下降 |
| w/o 自适应 | 43.89 | 去掉自适应，速度下降 |
| 固定阈值 | 40.12 | 静态阈值最差 |

### 关键发现

- Saber 同时提升质量（+1.9% Pass@1）和速度（251.4% 加速）——打破了 DLM 的速度-质量权衡
- 回溯机制是质量提升的主要来源——允许模型修正早期错误避免了级联失败
- 自适应加速是速度提升的主要来源——后期阶段大量并行解遮蔽
- Saber 在不同 DLM（LLaDA、Dream）上均有效——模型无关性

## 亮点与洞察

- "谨慎→激进"的自适应策略非常直觉且有效——上下文越丰富模型越自信，应该允许更多并行
- 回溯重遮蔽是 DLM 领域的重要创新——打破了"一旦决定不可撤销"的限制
- 两个策略协同作用——自适应加速允许激进并行，回溯机制确保激进不会导致灾难

## 局限与展望

- 回溯增加了每步的计算开销（需要重新评估已解遮蔽 token 的置信度）
- 超参数 $\mu$（回溯比例）需要调优
- 仅在代码生成上验证，自然语言生成的效果未知
- DLM 整体仍落后于 ARM，Saber 只是缩小了差距

## 相关工作与启发

- **vs Fast-dLLM**: 固定阈值加速，Saber 用动态阈值更精确
- **vs ReMDM**: 分阶段重遮蔽，Saber 逐步回溯更细粒度
- **vs ARM Speculative Decoding**: 解决不同问题——ARM 加速单token生成，Saber 优化DLM的并行解遮蔽

## 评分

- 新颖性: ⭐⭐⭐⭐ 自适应+回溯的组合在DLM领域是首创
- 实验充分度: ⭐⭐⭐⭐⭐ 5个代码基准+多DLM+详细消融
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，算法伪代码完整
- 价值: ⭐⭐⭐⭐ 对DLM实用化有显著推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration](specbound_adaptive_bounded_self-speculation_with_layer-wise_confidence_calibrati.md)
- [\[CVPR 2026\] SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](../../CVPR2026/llm_efficiency/sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)
- [\[ACL 2026\] Native Hybrid Attention for Efficient Sequence Modeling](native_hybrid_attention_for_efficient_sequence_modeling.md)
- [\[ACL 2025\] DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](../../ACL2025/llm_efficiency/dive_moe_reconstruction.md)
- [\[ACL 2026\] Lizard: An Efficient Linearization Framework for Large Language Models](lizard_an_efficient_linearization_framework_for_large_language_models.md)

</div>

<!-- RELATED:END -->
