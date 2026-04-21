---
title: >-
  [论文解读] Scaling Inference-Efficient Language Models
description: >-
  [ICML 2025][LLM效率][Scaling Laws] 本文提出了推理感知的 Scaling Law，通过在 Chinchilla 损失函数中引入模型宽高比（aspect ratio）项来联合优化参数量、训练 token 数和模型形状，训练 63 个模型拟合该定律后指导设计了 Morph-1B 模型，在保持下游任务精度的同时实现 1.8× 推理延迟提升。
tags:
  - ICML 2025
  - LLM效率
  - Scaling Laws
  - 推理效率
  - 模型架构
  - 宽浅模型
  - Morph-1B
---

# Scaling Inference-Efficient Language Models

**会议**: ICML 2025  
**arXiv**: [2501.18107](https://arxiv.org/abs/2501.18107)  
**代码**: https://github.com/Waterpine/open-lm-morph (有)  
**领域**: LLM效率  
**关键词**: Scaling Laws, 推理效率, 模型架构, 宽浅模型, Morph-1B

## 一句话总结

本文提出了推理感知的 Scaling Law，通过在 Chinchilla 损失函数中引入模型宽高比（aspect ratio）项来联合优化参数量、训练 token 数和模型形状，训练 63 个模型拟合该定律后指导设计了 Morph-1B 模型，在保持下游任务精度的同时实现 1.8× 推理延迟提升。

## 研究背景与动机

**领域现状**：Scaling Laws（尤其是 Chinchilla）已成为预测 LLM 性能的核心工具，指导在固定计算预算下平衡模型参数量 $N$ 和训练 Token 数 $D$，得出经典的 $D \approx 20N$ 结论。

**现有痛点**：
   - 现有 Scaling Law 仅关注**训练成本**，完全忽略了**推理成本**，而在实际部署中推理开销远大于训练（模型生命周期内要反复推理）。
   - FLOPs 约束不符合实际：实践中模型大小由部署设备内存决定，训练 Token 数由可用数据量决定（例如 LLaMA-3-8B 用了 15T tokens，远超 Chinchilla 推荐的 160B），两者并不受 FLOPs 预算约束。
   - 虽然 Sardana et al. (2023) 尝试考虑推理 FLOPs，但其方法需要预估模型生命周期内推理的 Token 总量，实际不可行。

**核心矛盾**：相同参数量的模型，推理延迟可以相差高达 **3.5 倍**——例如 MiniCPM-1B 的延迟甚至高于 Qwen2.5-14B。这说明参数量并非影响推理效率的唯一因素，**模型架构（形状）** 才是关键变量，但现有 Scaling Law 完全没有建模这一项。

**本文目标**
   - 子问题 1：如何在 Scaling Law 中引入模型架构变量（宽度 $d_{\text{model}}$ vs 深度 $n_{\text{layers}}$）？
   - 子问题 2：如何将推理延迟预算 $T_C$ 纳入优化约束？
   - 子问题 3：Scaling Law 预测的 loss 和下游任务精度之间存在 gap，如何选出真正最优的模型配置？

**切入角度**：作者通过大量实验观察到推理延迟随层数线性增长（因为层间计算必须串行），但宽度增大的影响远小于深度——同参数量下，**更宽更浅**的模型推理更快。

**核心 idea**：在 Chinchilla 损失函数上乘以一个关于宽高比 $R = d_{\text{model}} / n_{\text{layers}}$ 的修正项 $(1 + \varepsilon R^\gamma)$，从而联合优化 loss 和推理效率。

## 方法详解

### 整体框架

整个方法可分为三个阶段：

- **Phase A（候选生成）**：给定参数预算（如 1B），生成多组不同宽高比的模型配置候选（如 12 层×3072 hidden、16 层×2560 hidden、24 层×2048 hidden 等）。
- **Phase B（预测 + 测量）**：对每个候选，(1) 用推理感知 Scaling Law 预测其 loss；(2) 实际测量其推理延迟。
- **Phase C（排序 + 训练 + 评估）**：根据 loss 预测排名和延迟约束选取 top-k 候选，全量训练后在下游任务上评估，发布最优模型。

核心输入：参数量上限 $N_C$、训练 Token 上限 $D_C$、推理延迟预算 $T_C$。  
核心输出：满足约束的最优模型架构配置。

### 关键设计

1. **推理感知 Scaling Law 公式**

    - 功能：预测给定参数量 $N$、训练 Token 数 $D$ 和宽高比 $R$ 的模型训练 loss。
    - 核心思路：在 Chinchilla 的 $L(N, D) = E + AN^{-\alpha} + BD^{-\beta}$ 基础上，乘以架构修正因子：
    $L(N, D, R) = (E + AN^{-\alpha} + BD^{-\beta}) \cdot (1 + \varepsilon R^\gamma)$
      其中 $R = d_{\text{model}} / n_{\text{layers}}$，$A, B, E, \alpha, \beta, \gamma, \varepsilon$ 为可学习参数。实验中令 $\alpha = \beta = \gamma$ 以简化拟合。
    - 设计动机：实验发现宽高比 $R$ 对 loss 的影响呈现可用 $(1 + \varepsilon R^\gamma)$ 捕捉的规律——过宽（$R$ 过大）的模型 loss 会略微上升，但换取显著的推理加速。这个乘法形式使得架构修正对不同 $(N, D)$ 组合都能均匀地施加影响。

2. **约束重写：从 FLOPs 约束到三重约束**

    - 功能：将 Scaling Law 的优化目标从不实用的 FLOPs 约束改为参数量 + Token 数 + 推理延迟三重约束。
    - 核心思路：原始 Chinchilla 目标是 $\arg\min_{N,D} L(N,D) \text{ s.t. } \text{FLOPs}(N,D) = C$。本文改为：
    $\arg\min_{N,D} L(N,D) \text{ s.t. } N \leq N_C, \; D \leq D_C, \; T_{\text{inf}} \leq T_C$
      这样就把推理延迟预算显式地纳入了优化。推理延迟 $T_{\text{inf}}$ 通过实际在目标硬件上测量获得（几分钟即可）。
    - 设计动机：实际部署中，模型大小受设备内存限制，训练数据量受语料规模限制，两者都是硬性约束而非 trade-off。延迟预算由应用场景决定（如聊天机器人需要低延迟）。

3. **Predict-Rank-Select 模型选择方法**

    - 功能：解决"loss 相近但下游任务表现差异大"的问题，从多个候选中选出真正好的模型。
    - 核心思路：不依赖 Scaling Law 的绝对 loss 值来直接预测下游精度（noise 太大），而是利用其**排序能力**——预测 loss 越低的排名越靠前。选取 top-k 候选全量训练，再用下游评估做最终决定。
    - 设计动机：作者在 PIQA、BoolQ、HellaSwag 上观察到，loss 和 accuracy 的关系呈现不同模式——有些近似线性、有些"阶梯状"——使得精确预测 accuracy 非常困难。但 Scaling Law 的排序准确性远高于绝对预测准确性（Spearman 相关系数达到 1.0）。

### 训练策略

- **训练数据**：使用 DCLM-Baseline 数据集的均匀采样子集，单 epoch 无重复训练。
- **优化器**：AdamW，bfloat16 精度。
- **Scaling Law 拟合**：训练 63 个模型（80M-339M 参数，1.6B-12.8B tokens），用 Levenberg-Marquardt 算法对公式做最小二乘拟合。
- **过训练数据的重要性**：消融发现仅用 Chinchilla-optimal 数据（$D=20N$）拟合效果差，加入过训练数据（$D=160N$）后预测精度大幅提升（MSE 从 0.1165 降至 0.0006）。
- **Morph-1B 最终配置**：$d_{\text{model}}=3072$，$n_{\text{layers}}=12$，$n_{\text{heads}}=16$，中间层大小 8192，在 30B tokens 上训练。

## 实验关键数据

### 主实验：Morph-1B 与同量级开源模型对比

| 模型 | $d_{\text{model}}$ | $n_{\text{layers}}$ | 下游平均精度 | 推理延迟 (s) |
|------|----:|----:|:----:|:----:|
| Open-LM-1B | 2048 | 24 | 0.49 | 3.61 |
| OPT-1.3B | 2048 | 24 | 0.50 | 2.55 |
| Pythia-1.3B | 2048 | 22 | 0.49 | 3.28 |
| NeoX-1.3B | 2048 | 24 | 0.49 | 3.99 |
| OPT-IML-1.3B | 2048 | 24 | 0.54 | 2.54 |
| Morph-1B-v1 | 2048 | 24 | 0.52 | 3.61 |
| Morph-1B-v2 | 2560 | 16 | 0.52 | 2.57 |
| **Morph-1B** | **3072** | **12** | **0.52** | **1.96** |

Morph-1B 以仅 12 层的超浅架构，在保持 0.52 平均精度（与 v1/v2 一致）的同时，推理延迟仅 1.96s，比标准 24 层架构（3.61s）快 **1.8 倍**。OPT-IML-1.3B 略优的精度（0.54）来自于 6 倍的训练数据量（180B vs 30B）和指令微调。

### Scaling Law 拟合精度对比

| 指标 | Chinchilla | 推理感知 Scaling Law |
|------|:----:|:----:|
| MSE | 0.0033 | **0.0006** |
| $R^2$ | 0.9895 | **0.9982** |
| 相对预测误差 | 2.7%–4.1% | **< 1.2%** |
| Spearman (1B预测) | -0.40 | **1.00** |

推理感知 Scaling Law 在所有指标上大幅优于 Chinchilla。尤其关键的是 Spearman 相关系数从 -0.40（完全错误排序）提升到 1.00（完美排序），说明其排名预测能力极强。

### 消融实验

| 消融设置 | MSE | $R^2$ | Spearman | 说明 |
|---------|:---:|:---:|:---:|------|
| Full（含过训练数据） | 0.0006 | 0.9982 | 1.00 | 完整模型 |
| 去除过训练数据 (Chinchilla) | 0.9825 | -2.1259 | — | 完全失效 |
| 去除过训练数据 (推理感知) | 0.1165 | 0.6293 | — | 大幅退化但仍优于 Chinchilla |
| 随机模型形状 (Chinchilla) | 0.0198 | 0.9369 | — | 误差大 |
| 随机模型形状 (推理感知) | 0.0008 | 0.9973 | — | 仍然鲁棒 |

### 关键发现

- **过训练数据至关重要**：仅用 $D=20N$ 的 Chinchilla-optimal 数据拟合 Scaling Law 效果很差（两种方法都显著退化），加入 $D=160N$ 的过训练点后精度恢复。这说明实际部署中的过训练现象必须在 Scaling Law 中被建模。
- **推理感知 Scaling Law 更鲁棒**：即使随机选择模型形状来拟合，推理感知版本仍保持 $R^2 > 0.99$；而 Chinchilla 降到 0.93，预测误差高达 11.8%–13.4%。
- **宽浅模型一致性好**：在 1B/3B/7B 三个量级、不同 batch size、HuggingFace/vLLM 两种框架、A100/A30 两种 GPU 上，更宽更浅的模型都展现出更低延迟和更高吞吐，规律非常一致。
- **仅需 6 个数据点 + 85 GPU 小时**即可拟合出足够准确的推理感知 Scaling Law，成本极低。

## 亮点与洞察

- **乘法修正因子设计巧妙**：在 Chinchilla 公式上乘以 $(1 + \varepsilon R^\gamma)$ 而非加法，使得架构修正与 $(N, D)$ 的基础预测自然耦合，既简洁又有效，拟合仅多 2 个参数。
- **排序优于绝对预测**的洞察非常实用：承认 Scaling Law 的绝对 loss 预测不够精确（无法可靠映射到下游 accuracy），转而利用其排序能力做候选筛选，是一种务实且有效的方法论。
- **将推理延迟"免费"测量并入选择流程**：推理延迟的实测只需几分钟，几乎没有额外开销，但能直接排除延迟不达标的候选，这个设计思路可以迁移到任何需要硬件感知的模型选择场景。
- **宽浅模型的直觉解释**：层间计算必须串行（attention → FFN → 下一层），层数越多串行步骤越多；而单层内的矩阵运算可以并行，所以增加宽度不明显增加延迟。这一简洁的物理直觉贯穿全文。

## 局限与展望

- **规模受限**：受资源限制，最大只训练到 1.5B 参数。对于 7B/13B/70B 等更大规模，宽高比对 loss 的影响模式是否一致尚未验证。
- **架构优化单一**：仅考虑了标准 MHA + FFN 的 Transformer，未纳入 GQA、MQA、MLA 等高效注意力变体。虽然作者声称其框架可推广到这些架构，但未提供实验验证。
- **推理系统依赖**：延迟测量基于 HuggingFace generate（非量化、非优化），在实际部署系统（如 TensorRT-LLM、vLLM 量化模式）下，宽浅架构的优势是否保持需要进一步验证。不过附录中 vLLM 实验表明趋势一致。
- **下游任务有限**：评估仅覆盖了知识型/常识型 benchmark（ARC、PIQA、HellaSwag 等），未涉及生成质量、指令遵循、代码等更实际的评测。
- **未考虑 KV Cache 大小**：宽浅模型虽然延迟低，但更大的 $d_{\text{model}}$ 意味着更大的 KV cache，在长序列和大 batch 场景下可能成为内存瓶颈。

## 相关工作与启发

- **vs Chinchilla (Hoffmann et al., 2022)**：Chinchilla 只优化 $N$ 和 $D$ 的比例，忽略架构和推理成本。本文在其公式上增加架构变量，更贴合实际。Chinchilla 的 Spearman 排序在预测 1B 模型时甚至为负值，说明其对不同架构的预测完全失效。
- **vs Beyond Chinchilla-Optimal (Sardana et al., 2023)**：Sardana 通过预估推理总 FLOPs 来纳入推理成本，但需要预知模型生命周期推理量——这在实际中不可行。本文直接测量延迟，更实用。
- **vs DCLM/Gadre et al. (2024)**：Gadre 将 Scaling Law 扩展到下游任务 error 预测，但本文发现 loss→accuracy 映射在 loss 接近时噪声很大，因此改用排序方法，更鲁棒。
- **vs MiniCPM (Hu et al., 2024)**：MiniCPM 也训练小模型并关注效率，但其 1B 模型由于深窄架构导致延迟甚至高于某些 14B 模型，恰好说明了本文的核心观点。

## 评分

- 新颖性: ⭐⭐⭐⭐ 在 Scaling Law 中引入架构变量的思路并非全新（Kaplan 2020 已讨论），但将其形式化为乘法修正项并结合推理延迟约束是有价值的贡献。
- 实验充分度: ⭐⭐⭐⭐⭐ 训练 63 个模型，覆盖 5 种参数量 × 多种宽高比 × 3 种训练长度，附录含 A30 GPU + vLLM + TTFT 等大量补充实验，非常完整。
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，动机推导自然。公式表达有少量 LaTeX 排版瑕疵但不影响理解。
- 价值: ⭐⭐⭐⭐ 对 LLM 部署中的模型架构选择有直接指导意义，"宽浅模型推理更快"的结论简洁实用，可直接应用于模型设计。

<!-- RELATED:START -->

## 相关论文

- [DISC: Dynamic Decomposition Improves LLM Inference Scaling](../../NeurIPS2025/llm_efficiency/disc_dynamic_decomposition_improves_llm_inference_scaling.md)
- [Star Attention: Efficient LLM Inference over Long Sequences](star_attention_efficient_llm_inference_over_long_sequences.md)
- [Ladder Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference](ladder-residual_parallelism-aware_architecture_for_accelerating_large_model_infe.md)
- [Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](../../ACL2025/llm_efficiency/scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)
- [On the Entropy Calibration of Language Models](../../NeurIPS2025/llm_efficiency/on_the_entropy_calibration_of_language_models.md)

<!-- RELATED:END -->
