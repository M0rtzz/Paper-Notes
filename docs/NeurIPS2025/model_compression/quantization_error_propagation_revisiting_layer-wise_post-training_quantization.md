---
title: >-
  [论文解读] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization
description: >-
  [NeurIPS 2025][模型压缩][量化] 识别现有逐层 PTQ 方法忽略量化误差跨层累积和增长的关键瓶颈，提出 QEP 框架通过误差传播和补偿显式纠正累积误差，在极低比特（INT2/INT3）下实现大幅性能提升。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 量化
  - LLM compression
  - low-bit
---

# Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization

**会议**: NeurIPS 2025  
**arXiv**: [2504.09629](https://arxiv.org/abs/2504.09629)  
**代码**: [GitHub](https://github.com/FujitsuResearch/qep)  
**领域**: model_compression  
**关键词**: post-training quantization, LLM compression, quantization error propagation, layer-wise quantization, low-bit

## 一句话总结

识别现有逐层 PTQ 方法忽略量化误差跨层累积和增长的关键瓶颈，提出 QEP 框架通过误差传播和补偿显式纠正累积误差，在极低比特（INT2/INT3）下实现大幅性能提升。

## 研究背景与动机

逐层后训练量化（Layer-wise PTQ）因其简单高效而成为 LLM 压缩的主流方法，如 GPTQ、AWQ、QuIP 等。然而近期该方向的进展趋于饱和。本文回到逐层 PTQ 的核心设计，发现一个根本性问题：现有方法将每层的量化视为独立优化问题（最小化 $\|W_l X_l - \hat{W}_l X_l\|_F^2$），既不考虑上游层传播来的量化误差，也不纠正已累积的误差。实验表明量化误差在层间近似指数级增长，且即使在未量化层中仍持续增长。

## 方法详解

### 整体框架

QEP 是一个通用的、轻量级的框架，可与任何逐层 PTQ 方法无缝结合。核心思想是：将逐层独立优化重新表述为考虑误差传播的联合优化，通过权重校正项补偿前序层的累积量化误差。

### 关键设计

1. **问题重新表述（Problem Reformulation）**: 将原始目标 $\min \|W_l X_l - \hat{W}_l X_l\|_F^2$（共享同一输入 $X_l$）改为 $\min \|W_l X_l - \hat{W}_l \hat{X}_l\|_F^2$，其中 $X_l$ 是全精度输入，$\hat{X}_l$ 是量化输入。这样量化权重不仅要近似全精度权重，还要补偿上游累积的量化误差。关键区别在于：原始目标的平凡最优解是 $\hat{W}_l = W_l$，而新目标的最优解一般 $\hat{W}_l \neq W_l$，显式实现了误差纠正。

2. **权重校正（Weight Correction, Proposition 5.1）**: 将连续松弛后的最优解推导为闭式表达式：$W_l^* = W_l + W_l \delta_l \hat{X}_l^T \hat{H}_l^{-1}$，其中 $\delta_l = X_l - \hat{X}_l$ 是累积量化误差，$\hat{H}_l = \hat{X}_l \hat{X}_l^T$ 是基于量化激活的 Hessian。校正后的权重保持了与原始 PTQ 相同的二次优化结构（式7），因此可直接复用现有的 Hessian 加速方法。

3. **传播强度控制（Controlling Propagation Strength）**: 引入可调参数 $\alpha_l \in [0,1]$：$W_l^*(\alpha_l) = W_l + \alpha_l W_l \delta_l \hat{X}_l^T \hat{H}_l^{-1}$。$\alpha_l=1$ 恢复完全校正，$\alpha_l=0$ 退化为原始方法。Proposition 5.3 证明了 $\alpha_l$ 等价于正则化参数，可有效防止过拟合，特别在 MLP 层（参数量大）中尤为重要。对大型模型（如 Llama-2 70B）的 MLP 层设置 $\alpha_l=0$ 既减少计算开销（约1/3到1/2），又起到隐式正则化作用。

### 损失函数 / 训练策略

QEP 本身不引入新的训练过程。核心优化目标（式7）为：$\min_{\hat{W}_l} \|W_l^* \hat{X}_l - \hat{W}_l \hat{X}_l\|_F^2$，与原始逐层 PTQ 结构相同，只是将 $W_l$ 替换为校正后的 $W_l^*$。Theorem 5.2 理论保证了 QEP 的输出量化误差 $\|f_\theta(X) - f_{\hat{\theta}_{QEP}}(X)\|_F \leq \|f_\theta(X) - f_{\hat{\theta}_{BASE}}(X)\|_F$。

## 实验关键数据

### 主实验

| 模型 | 方法 | INT4 PPL | INT3 PPL | INT2 PPL |
|------|------|----------|----------|----------|
| Llama-2-7B | QuIP | 8.434 | 12.048 | 65.593 |
| Llama-2-7B | QuIP+QEP | 5.753 | 6.154 | 11.972 |
| Llama-2-7B | GPTQ | 6.083 | 10.881 | 13051.5 |
| Llama-2-7B | GPTQ+QEP | 5.933 | 7.898 | 7214.3 |
| Llama-2-7B | AWQ | 5.831 | 15.299 | 199448.8 |
| Llama-2-7B | AWQ+QEP | 5.756 | 11.131 | - |
| Llama-3-8B | QuIP | 6.998 | 8.288 | 70.518 |
| Llama-3-8B | QuIP+QEP | 6.650 | 7.703 | 27.326 |
| Llama-2-7B | FP16基线 | 5.472 | 5.472 | 5.472 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| BASE (无QEP) | 标准PTQ误差 | 误差近似指数级增长 |
| With QEP | 误差显著降低 | 前10个block后误差增长被有效抑制 |
| $\alpha_l=0$ (MLP) | 减少计算+防过拟合 | 对70B模型特别重要 |
| $\alpha_l=1/2$ (默认) | 平衡校正和正则化 | 大部分模型的推荐值 |
| RTN+QEP INT3 | 539.9→17.3 | 在最简单基线上也有巨大提升 |

### 关键发现

- QEP 在所有测试方法（RTN/GPTQ/AWQ/QuIP）和所有比特宽度（INT2/3/4）上均有提升，具有极强通用性
- 低比特场景下提升最为显著：QuIP INT2 从 PPL 65.6 降到 12.0（Llama-2-7B），接近实用水平
- GPTQ INT3 从 10.9 降到 7.9（Llama-2-7B），甚至超过原始 AWQ INT3 的 15.3
- 量化误差传播图（Figure 2）直观展示了 QEP 有效遏制了误差在量化层和未量化层间的累积与增长
- Zero-shot 任务上 QEP 同样带来一致性提升，验证了 PPL 改善可转化为下游任务能力

## 亮点与洞察

- 视角独特：回到逐层 PTQ 最基础的优化目标重新审视，发现前人忽略的误差累积问题
- 理论与实践统一：Proposition 5.1 给出闭式校正解，Theorem 5.2 提供理论保证，且计算开销可控
- $\alpha_l$ 参数设计精妙：既可防过拟合，又能控制计算量，Proposition 5.3 揭示其与正则化的等价关系
- 正交于现有所有 PTQ 改进（非线性量化、旋转矩阵等），可叠加使用

## 局限与展望

- $\alpha_l$ 目前采用简单的固定值策略（1/2 或 MLP 层设0），未来可开发自适应、逐层、数据感知的调节策略
- 在 INT2 场景下，AWQ+QEP 有时未必优于原始 QuIP+QEP，说明不同 PTQ 方法与 QEP 的兼容性有差异
- 校正项计算依赖于校准数据集的质量和大小，小校准集可能导致过拟合
- 未探索与块级（block-wise）PTQ 或 QAT 的结合

## 相关工作与启发

- GPTQ 开创的 Hessian-based 逐层量化是 QEP 的直接改进对象
- AWQ 的 salience-based 权重缩放和 QuIP 的旋转预处理是正交技术，可与 QEP 叠加
- 误差传播的思想在深度学习训练（如 batch normalization 缓解梯度消失）中有类似思路，但首次应用于 PTQ

## 补充讨论

- QEP 的核心洞察可类比为：原始逐层 PTQ 像逐帧编辑视频但忽略帧间连贯性，QEP 则引入帧间误差补偿机制
- 仅需额外计算一个矩阵乘法 $\delta_l \hat{X}_l^T$，即可获得显著的量化精度提升，成本效益比极高
- 在 Llama-3-8B INT4 上，GPTQ 的 PPL 从 147.9 降到 9.5，说明原始 GPTQ 在该模型上存在严重的误差累积问题
- QuIP+QEP 在 INT2 下的 PPL（11.97）甚至优于 GPTQ 的 INT3（10.88），展示了低比特量化的巨大潜力
- Figure 2 的误差传播可视化是论文中最具说服力的证据，直观展示了误差如何在未量化层中持续增长

## 评分

- 新颖性: ⭐⭐⭐⭐ 从最基础的优化目标出发识别问题，解法虽简洁但此前被忽略
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型(7B-70B)、多方法(4种PTQ)、多比特(INT2/3/4)的全面评估
- 写作质量: ⭐⭐⭐⭐ 从问题发现到理论分析到实验验证的逻辑链条完整
- 价值: ⭐⭐⭐⭐⭐ 作为通用插件显著提升所有逐层PTQ方法性能，尤其在极低比特下价值巨大
