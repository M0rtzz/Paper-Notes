---
title: >-
  [论文解读] Scaling with Collapse: Efficient and Predictable Training of LLM Families
description: >-
  [ICLR 2026][预训练][训练损失曲线崩塞] 证明 LLM 家族的训练损失曲线在优化超参数与数据预算匹配时会“崩塞”到同一条通用曲线上，并利用这一现象实现两个实用应用：(1) 偏离崩塞作为训练病理的早期诊断信号，(2) 崩塞曲线的可预测性实现大规模超参调优的早停。
tags:
  - "ICLR 2026"
  - "预训练"
  - "训练损失曲线崩塞"
  - "超参缩放"
  - "训练诊断"
  - "早停"
  - "Cerebras"
---

# Scaling with Collapse: Efficient and Predictable Training of LLM Families

**会议**: ICLR 2026  
**arXiv**: [2509.25087](https://arxiv.org/abs/2509.25087)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 训练损失曲线崩塞, 超参缩放, 训练诊断, 早停, Cerebras

## 一句话总结
证明 LLM 家族的训练损失曲线在优化超参数与数据预算匹配时会“崩塞”到同一条通用曲线上，并利用这一现象实现两个实用应用：(1) 偏离崩塞作为训练病理的早期诊断信号，(2) 崩塞曲线的可预测性实现大规模超参调优的早停。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：Scaling law 可预测最终损失，μP 可转移学习率，但完整的训练损失曲线（TLC）的可预测性尚未在实际 LLM 规模下验证。

**现有痛点**：

### 核心矛盾

**核心矛盾**：Qiu et al. 发现损失曲线崩塞现象但仅在小规模验证，未测试实际 LLM 训练配方

### 现有痛点

**现有痛点**：前沿规模无法直接实验——需要从小规模推断

### 解决思路

**解决思路**：训练病理（loss spike）的诊断仍依赖人工判断

**核心发现**：损失曲线崩塞的充要条件是优化超参数对给定数据预算是最优的——崩塞是计算最优训练的“特征标记”。

**切入角度**：当所有模型以相同的 tokens-per-parameter (TPP=D/N) 训练且 AdamW 时间尺度 τ 设为最优时，不同大小的模型的 TLC 在简单归一化后落在同一条曲线上。

## 方法详解

### 整体框架
本文的核心观察是：当一个 LLM 家族里不同大小的模型都用相同的 tokens-per-parameter（$\text{TPP}=D/N$）训练、且优化超参数对各自数据预算都设为最优时，它们的完整训练损失曲线（TLC）在简单归一化后会“崩塞”到同一条通用曲线上。围绕这一现象，作者把它从一个视觉规律提炼成可操作的工具：先刻画崩塞成立的充要条件，再用偏离崩塞作为训练病理的在线诊断信号，最后用崩塞曲线的可预测性把昂贵的超参搜索改造成早停流程，并据此训练出 Celerity 模型家族。

### 关键设计

**1. 崩塞条件：把“计算最优训练”变成一个可观测的特征标记。** 已有 scaling law 只能预测最终损失这一个标量，而完整的损失曲线形状是否可跨规模转移此前没有在真实 LLM 规模上验证过。作者发现 TLC 崩塞并非总会发生，而是有严格前提：所有模型必须共享相同的 TPP，且学习率、batch size、权重衰减按 scaling law 联合缩放到最优。其中一个被忽视的关键变量是 AdamW 的 EMA 时间尺度 $\tau = 1/(\eta\lambda)$，它把学习率 $\eta$ 与权重衰减 $\lambda$ 的效果统一了起来——只有当 $\tau$ 对给定 TPP 取最优时曲线才会崩塞，$\tau$ 偏离最优会拉伸或压缩 TLC，不同的 LR 衰减形状同样会破坏崩塞。反过来，崩塞与否就成了“超参是否计算最优”的充要判据：曲线一旦崩塞，就说明这套配方处在效率前沿。

**2. 偏离诊断：用残差把人工盯 loss 曲线换成客观早警。** 传统上判断一次 loss spike 是否需要回退依赖人工经验，往往等异常在原始曲线上肉眼可见时才反应。作者改为先从小模型拟合出通用崩塞曲线，再让大模型在训练中把自己归一化后的 TLC 实时与之相减，监控崩塞残差。由于数值稳定性问题（如 bf16 精度不足导致的梯度累积漂移）会先在残差里显形，这套方法能在原始 TLC 出现明显异常前数百步就检测到偏离——在 Celerity 1.8B 的训练里正是靠它提前定位了一处 bf16 精度问题，修复后曲线重新回到崩塞轨迹上。

**3. 早停超参调优：靠崩塞曲线的可预测性外推最终损失。** 崩塞意味着完整曲线的形状是可参数化、可预测的，因此不必把每组超参都训练到底。作者对一批候选配置只训练前 10–20% 的 token，用崩塞曲线的参数化模型拟合这段部分 TLC 并外推到最终损失，外推误差控制在 1% 以内，从而能提前排除预期表现最差的大多数配置、只对最优的少数完成全量训练，把超参搜索的计算量节省 80% 以上。Celerity 家族正是在 Cerebras CS-3 上利用这些洞察训练出来的，落在同规模模型的效率前沿。

## 实验关键数据

### 主实验

| 现象 | 结果 |
|------|------|
| Llama-2（不同 TPP） | TLC 不崩塞 |
| Celerity（相同 TPP + 最优超参） | **TLC 完美崩塞** |
| 偏离诊断 | 比人工判断更早检测 loss spike |
| 早停超参 | 从 20% TLC 外推最终损失，误差 <1% |

### 关键发现
- **崩塞是计算最优训练的充要条件**——仅当超参按 scaling law 设为最优时才出现
- 偏离诊断可更早发现数值稳定性问题（如 bf16 精度不足）
- 早停节省 80%+ 超参搜索计算

## 消融实验与深入分析

### 崩塞条件验证

| 条件 | 是否崩塞 | 说明 |
|------|---------|------|
| 固定 TPP + 最优 $\tau$ + 固定 LR schedule | ✓ 崩塞 | Celerity 家族 |
| 不同 TPP（如 Llama-2） | ✗ 不崩塞 | 不同 D/N 比导致 TLC 形状不同 |
| 固定 TPP + 非最优 $\tau$ | ✗ 不崩塞 | $\tau$ 偏离最优会拉伸或压缩 TLC |
| 固定 TPP + 不同 LR schedule | ✗ 不崩塞 | LR 衰减形状直接影响 TLC 形状 |

### 偏离诊断的实际案例
- 在 Celerity 1.8B 训练中，缓存中旧的 loss 显示了轻微的上升趋势
- 通过崩塞残差分析（将 TLC 归一化后与通用曲线比较），在原始 TLC 出现明显异常前数百步就检测到了偏离
- 诊断结果：bf16 数值精度问题导致的梯度累积不稳定
- 修复后 TLC 重新回到崩塞曲线上

### 早停超参调优
- 对 20+ 组超参配置仅训练前 20% 的 token
- 用崩塞曲线参数化模型拟合部分 TLC → 外推最终 loss
- 排除预期最终 loss 最差的 80% 配置，仅对 top 20% 完成全量训练
- 节省 80%+ 超参搜索的计算量，外推误差 <1%

### Celerity 在效率前沿的位置

| 模型 | 参数量 | 训练 token | 平均准确率 |
|------|--------|-----------|-----------|
| 典型同规模模型 | 同等 | 同等 | 基线 |
| **Celerity** | 同等 | 同等 | **效率前沿** |

## 亮点与洞察
- **崩塞作为"健康标志"**是一个简单但强大的工程工具——如果 TLC 不崩塞就说明超参或训练配方有问题。这比任何 metric 都更直觉化。
- **崩塞 = 计算最优训练**的充要关系是核心理论贡献——将一个视觉现象连接到了优化理论
- **偏离诊断的实用性**：传统方法需要人工判断 loss spike 是否需要回退，崩塞曲线提供了客观参考
- **早停超参调优**：外推最终 loss 的可靠性使得大规模超参搜索成本大幅降低
- **$\tau$ 的统一作用**：AdamW 的 EMA 时间尺度 $\tau = 1/(\eta\lambda)$ 是一个被忽视但极其重要的超参——它统一了学习率和权重衰减的效果

## 局限与展望
- 崩塞条件要求所有模型 TPP 相同——实际中不同模型可能有不同最优 TPP（如 Chinchilla 的 20 vs 其他估计）
- 仅验证了预训练 loss——下游任务性能的崩塞未探索（loss 崩塞不保证下游 accuracy 也崩塞）
- 早停外推依赖参数化崩塌曲线模型的准确性——对于非常不同的训练配方可能需要重新拟合
- 所有实验在 Cerebras CS-3 上运行——不同硬件（如 GPU）上的崩塞行为可能略有差异（精度、通信模式等）
- 目前仅验证了 μP 参数化下的崩塞——其他参数化方案（如 SP）下是否成立未知

## 相关工作与启发
- **vs Chinchilla (Hoffmann et al.)**：Chinchilla 预测最终损失的缩放律（一个标量）；本文预测完整训练曲线的形状（一条曲线）——是缩放律的"时间序列版"
- **vs Qiu et al. (2025) Supercollapse**：他们在小规模自回归任务上发现崩塞；本文将其推广到实际 LLM 训练，并揭示了崩塞的充要条件（TPP + $\tau$ 最优）
- **vs μP (Yang & Hu)**：μP 使学习率可跨规模转移；本文发现在 μP 下整个 TLC 形状都可跨规模转移——是 μP 的更强推论
- **vs Wang & Aitchison (2024) AdamW EMA**：他们发现 $\tau$ 在图像任务上跨规模稳定；本文发现 $\tau$ 的最优值取决于 TPP，在 LLM 中是 TLC 崩塞的关键控制变量
- **启发**：崩塞理论可以推广到其他序列训练场景——如扩散模型、强化学习的训练曲线是否也存在类似的通用形状

## 评分
- 新颖性: ⭐⭐⭐⭐ 崩塞条件的发现和实用应用有独特洞察力
- 实验充分度: ⭐⭐⭐⭐⭐ 大规模 Cerebras 实验，多模型大小验证，实际训练诊断案例
- 写作质量: ⭐⭐⭐⭐⭐ Figure 1 的三列对比极其直观，行文清晰
- 价值: ⭐⭐⭐⭐⭐ 对大规模 LLM 训练的实际工程指导价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Scaling Inference-Efficient Language Models](../../ICML2025/llm_pretraining/scaling_inference-efficient_language_models.md)
- [\[ICML 2026\] Annotations Mitigate Post-Training Mode Collapse](../../ICML2026/llm_pretraining/annotations_mitigate_post-training_mode_collapse.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ACL 2026\] SAGE: Sign-Adaptive Gradient for Memory-Efficient LLM Optimization](../../ACL2026/llm_pretraining/sage_sign-adaptive_gradient_for_memory-efficient_llm_optimization.md)
- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](../../NeurIPS2025/llm_pretraining/vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)

</div>

<!-- RELATED:END -->
