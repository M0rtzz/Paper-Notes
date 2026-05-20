---
title: >-
  [论文解读] First-Order Error Matters: Accurate Compensation for Quantized Large Language Models
description: >-
  [AAAI 2026][模型压缩][后训练量化] 本文揭示了LLM后训练量化中逐列补偿过程会导致一阶梯度项不可忽略的问题，提出FOEM方法通过将一阶项纳入误差补偿公式来提升量化精度，在3-bit量化下将Llama3-8B的困惑度降低17.3%，且几乎不增加计算开销。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "后训练量化"
  - "大语言模型"
  - "一阶误差补偿"
  - "GPTQ"
  - "权重量化"
---

# First-Order Error Matters: Accurate Compensation for Quantized Large Language Models

**会议**: AAAI 2026  
**arXiv**: [2507.11017](https://arxiv.org/abs/2507.11017)  
**代码**: [https://github.com/Xingyu-Zheng/FOEM](https://github.com/Xingyu-Zheng/FOEM)  
**领域**: 模型压缩  
**关键词**: 后训练量化, 大语言模型, 一阶误差补偿, GPTQ, 权重量化

## 一句话总结

本文揭示了LLM后训练量化中逐列补偿过程会导致一阶梯度项不可忽略的问题，提出FOEM方法通过将一阶项纳入误差补偿公式来提升量化精度，在3-bit量化下将Llama3-8B的困惑度降低17.3%，且几乎不增加计算开销。

## 研究背景与动机

### 领域现状

大语言模型（LLM）在语言理解、对话系统、代码生成等领域取得了显著成功，但其庞大的参数量带来了巨大的内存和计算负担。量化作为经典的模型压缩技术，通过将高精度浮点参数转换为低比特定点格式来减少内存使用并加速计算。在各种量化方法中，后训练量化（PTQ）因不需要梯度微调而特别适合大模型。

### 现有痛点

GPTQ是LLM权重量化的代表方法，它使用Taylor展开的二阶项来建模量化误差，并通过Hessian矩阵逐列量化权重、补偿后续列的误差。**然而，这些方法都假设全精度模型已经充分优化，因此一阶项可以忽略不计。**

### 核心矛盾

本文发现了一个被忽视的关键问题：在逐列量化-补偿过程中，先前列的量化补偿会导致后续未量化列的潜在权重 $\mathbf{W}$ 与原始全精度权重 $\mathbb{W}$ 之间产生显著偏差。这种累积偏差使得后续列在量化时实际上具有不可忽略的梯度 $g = \frac{\partial E}{\partial w}$，因此忽略一阶项的假设从根本上是有缺陷的。

### 本文切入角度

既然一阶项不可忽略，那么应该显式地将其纳入量化误差补偿。但直接计算梯度需要反向传播，这对大模型来说计算成本过高。FOEM巧妙地利用权重偏差和Hessian的关系来近似梯度，且在代入理论解后Hessian项相互抵消，最终只需简单的权重差计算即可。

## 方法详解

### 整体框架

FOEM沿用GPTQ的逐列量化框架，但在补偿项的计算中引入了一阶梯度修正。整体流程为：计算Hessian矩阵 → Cholesky分解 → 逐列量化 → **带一阶修正的误差补偿** → 更新后续列权重。

### 关键设计

#### 1. 一阶项的理论推导

保留Taylor展开中的一阶项后，量化误差模型变为：

$$\delta E = g \delta w^\top + \frac{1}{2} \delta w \mathbf{H} \delta w^\top$$

通过拉格朗日乘子法求解约束优化问题，得到最优补偿项：

$$\delta w = -\frac{w_q - \hat{w}_q - g H^{-1} e_q^\top}{\mathbf{T}_{qq}} \mathbf{T}_{q,q:} - g \mathbf{H}^{-1}$$

相比GPTQ的解（只有第一项的分子中没有梯度项，且没有第二项），FOEM引入了额外的梯度相关修正。

- **设计动机**：量化补偿过程中累积的权重偏差会产生不可忽略的梯度，忽略它们会导致次优补偿

#### 2. 梯度的高效近似

直接通过反向传播计算梯度不现实。FOEM利用一阶Taylor展开来近似梯度：

$$g(\mathbf{W}) \approx g(\mathbb{W}) + (\mathbf{W} - \mathbb{W})\mathbf{H}$$

由于原始全精度权重已经优化到局部最优，$g(\mathbb{W}) \approx 0$，因此：

$$g(\mathbf{W}) \approx (\mathbf{W} - \mathbb{W})\mathbf{H}$$

引入稳定化因子 $\beta$（默认0.1）：

$$g = \beta (\mathbf{W} - \mathbb{W}) \mathbf{H}$$

- **核心思路**：用权重偏差乘以Hessian来近似梯度，避免反向传播
- **设计动机**：$\beta$ 控制梯度修正的强度，防止过大的近似梯度放大误差

#### 3. Hessian项的消除

将梯度近似代入理论解后，最终补偿公式变为：

$$\delta w = -\frac{(w_q - \hat{w}_q) - \beta(w_q - \mathbb{W} e_q^\top)}{\mathbf{T}_{qq}} \mathbf{T}_{q,q:} - \beta(\mathbf{W} - \mathbb{W})$$

关键发现：**Hessian $\mathbf{H}$ 和逆Hessian $\mathbf{H}^{-1}$ 在代入过程中完全抵消**。这意味着：
- 不需要显式计算或求逆Hessian用于梯度项
- 只需要简单的权重差计算（$\mathbf{W} - \mathbb{W}$）
- 可以复用GPTQ已有的Cholesky分解结果
- GPTQ的懒更新机制等策略完全兼容

### 损失函数 / 训练策略

FOEM是后训练量化方法，不涉及训练。使用128个C4数据集样本（序列长度2048）进行校准。唯一的超参数 $\beta$ 固定为0.1，在所有模型架构和量化配置上表现稳健。

## 实验关键数据

### 主实验

**Llama3-8B 3-bit权重量化（WikiText2困惑度）**：

| 方法 | WikiText2 PPL↓ | C4 PPL↓ | 0-shot Avg↑ | MMLU↑ |
|------|----------------|---------|-------------|-------|
| FP16 | 6.13 | 9.61 | 70.7 | 64.9 |
| RTN | 13.10 | 20.50 | 57.2 | 38.9 |
| GPTQ | 9.86 | 12.94 | 64.4 | 55.4 |
| GPTAQ | 8.92 | 12.82 | 63.3 | 53.8 |
| **FOEM** | **8.32** | **12.37** | **65.5** | **56.1** |

**Llama2-7B 3-bit权重量化**：

| 方法 | WikiText2 PPL↓ | C4 PPL↓ | 0-shot Avg↑ | MMLU↑ |
|------|----------------|---------|-------------|-------|
| FP16 | 5.48 | 6.90 | 66.9 | 45.8 |
| GPTQ | 6.38 | 7.85 | 63.5 | 40.3 |
| GPTAQ | 6.41 | 7.93 | 64.0 | 35.1 |
| **FOEM** | **6.27** | **7.81** | **64.5** | **42.0** |

**W4A4KV4 量化（结合SpinQuant）**：

| 模型 | 方法 | WikiText2↓ | C4↓ | 0-shot Avg↑ |
|------|------|-----------|-----|-------------|
| Llama3-8B | GPTQ | 8.55 | 13.24 | 64.1 |
| Llama3-8B | GPTAQ | 8.50 | 13.13 | 64.1 |
| Llama3-8B | **FOEM** | **8.35** | **12.94** | **64.3** |

### 消融实验

**$\beta$ 敏感性分析（Llama3-8B 3-bit）**：

| $\beta$ | WikiText2↓ | C4↓ | 0-shot Avg↑ | 说明 |
|---------|-----------|-----|-------------|------|
| 0 (GPTQ) | 9.86 | 12.94 | 64.4 | 无一阶修正 |
| 0.1 | **8.32** | **12.37** | 65.5 | 默认设置 |
| 0.2 | 8.87 | 12.90 | 65.7 | 仍有提升 |
| 0.3 | 8.43 | 12.88 | **65.8** | PPL稍升但Avg最优 |
| 0.5 | 8.52 | 12.51 | 64.9 | 性能开始下降 |
| 0.8 | 10.09 | 13.09 | 61.5 | 显著退化 |
| 1.0 | 10.37 | 14.41 | 61.5 | 完全退化 |

**效率分析（Llama3-8B W4A4KV4）**：

| 方法 | 量化时间(s) | WikiText2↓ |
|------|-----------|-----------|
| GPTQ | 825.5 | 8.55 |
| GPTAQ | 1112.2 | 8.50 |
| FOEM | 828.9 | 8.35 |

### 关键发现

1. **一阶项不可忽略**：逐列补偿过程中累积的权重偏差使得后续列存在显著梯度，忽略它们会导致次优的量化结果
2. **$\beta < 0.5$ 时稳定有效**：超过0.5后性能显著退化，符合理论分析——过大的梯度会放大近似误差
3. **计算开销几乎为零**：与GPTQ的量化时间几乎相同（828.9s vs 825.5s），而GPTAQ需要1112.2s
4. **跨架构泛化**：在Transformer（Llama/Qwen/Mistral/Phi）和SSM（Mamba）上均有效
5. **与SpinQuant互补**：在W4A4KV4这种极端量化设置下仍能提供额外增益

## 亮点与洞察

1. **发现被忽视的问题**：所有基于补偿的PTQ方法都假设一阶项可忽略，但补偿过程本身会破坏这个假设——这是一个非常优雅的观察
2. **数学上的巧妙消除**：通过Taylor展开近似梯度后代入拉格朗日解，Hessian项恰好完全抵消，这不是刻意设计而是自然推导的结果
3. **极简改动带来显著提升**：只需在GPTQ的补偿公式中额外减去一个简单的权重差项，就能获得显著的性能提升
4. **理论分析扎实**：从数学推导到实验验证形成完整闭环

## 局限与展望

1. **$\beta$ 固定为0.1**：虽然实验表明不同$\beta$值在<0.5时都有效，但并未探索自适应的$\beta$选择机制
2. **梯度近似的假设**：假设 $g(\mathbb{W}) \approx 0$ 在实际预训练模型中可能并不完全成立，特别是对于未充分训练的模型
3. **二阶以上的高阶项**：本文只修正了一阶项，高阶项的累积效应尚未考虑
4. **极低比特（2-bit）场景未探索**：当量化更激进时，累积误差可能更显著，一阶修正可能不够

## 相关工作与启发

- **GPTQ** → 基于OBS的逐列量化补偿框架，本文的基础
- **GPTAQ** → 使用真实梯度信息的改进版，但计算开销大（+35%时间），本文用近似梯度实现了更好的效果
- **SpinQuant** → 基于旋转的激活量化方法，与FOEM可以无缝结合
- **启发**：对于任何基于补偿的迭代优化方法，都值得重新审视"一阶项可忽略"的假设——补偿本身会改变优化景观

## 评分

- 新颖性: ⭐⭐⭐⭐ （观察精准，数学推导优雅，但改进基于已有框架）
- 实验充分度: ⭐⭐⭐⭐⭐ （覆盖多个模型系列、多种量化配置、效率分析、敏感性分析、跨架构验证）
- 写作质量: ⭐⭐⭐⭐⭐ （数学推导清晰，逻辑链完整）
- 价值: ⭐⭐⭐⭐ （即插即用的量化方法改进，实用价值高）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](../../NeurIPS2025/model_compression/restoring_pruned_large_language_models_via_lost_component_compensation.md)
- [\[AAAI 2026\] Failures to Surface Harmful Contents in Video Large Language Models](failures_to_surface_harmful_contents_in_video_large_language_models.md)
- [\[ICML 2025\] SlimLLM: Accurate Structured Pruning for Large Language Models](../../ICML2025/model_compression/slimllm_accurate_structured_pruning_for_large_language_models.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[AAAI 2026\] QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching](quept_quantized_elastic_precision_transformers_with_one-shot_calibration_for_mul.md)

</div>

<!-- RELATED:END -->
