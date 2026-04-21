---
title: >-
  [论文解读] Wanda++: Pruning Large Language Models via Regional Gradients
description: >-
  [ACL 2025][模型压缩] 提出 Wanda++——基于 decoder block 级别区域梯度的轻量级 LLM 剪枝框架，通过区域梯度评分（RGS）改进剪枝准则 + 区域优化（RO）最小化稠密/稀疏块输出差异，在 2:4 稀疏下 WikiText 困惑度较 Wanda 最高降低 32%，单 H100 GPU 10 分钟内完成 7B 模型剪枝。
tags:
  - ACL 2025
  - 模型压缩
---

# Wanda++: Pruning Large Language Models via Regional Gradients

**会议**: ACL 2025  
**arXiv**: [2503.04992](https://arxiv.org/abs/2503.04992)  
**代码**: 无  
**领域**: 模型压缩  

## 一句话总结

提出 Wanda++——基于 decoder block 级别区域梯度的轻量级 LLM 剪枝框架，通过区域梯度评分（RGS）改进剪枝准则 + 区域优化（RO）最小化稠密/稀疏块输出差异，在 2:4 稀疏下 WikiText 困惑度较 Wanda 最高降低 32%，单 H100 GPU 10 分钟内完成 7B 模型剪枝。

## 背景与动机

1. **LLM 推理瓶颈**：大模型参数量庞大（如 LLaMA-2 70B 需 140GB 显存），推理延迟高，迫切需要模型压缩。
2. **现有剪枝方法精度损失大**：后训练剪枝方法（SparseGPT、Wanda）虽然高效，但在 2:4 半结构化稀疏下性能退化严重，远不如量化方法（AWQ 近乎无损 4× 压缩）。
3. **梯度信息有价值但获取昂贵**：GBLM、Pruner-Zero 已证明梯度信息能显著改善剪枝效果，但它们依赖**全模型反向传播**，GPU 时间和显存需求不可接受（GBLM 剪 7B 需 5800 秒 vs Wanda 55 秒）。
4. **逐层剪枝的线性假设有缺陷**：Wanda 在每层独立评估权重重要性，忽略层间的累积误差传播效应。
5. **核心问题**：能否在保持轻量级的同时有效利用梯度信息？

## 方法详解

### 整体框架

Wanda++ 对每个 decoder block 逐块执行两阶段处理：**Regional Gradient Score (RGS) 剪枝** + **Regional Optimization (RO) 权重修复**，迭代 K 轮。

### 1. Regional Gradient Score (RGS)

核心思想：用 decoder block 级别的"区域梯度"代替全模型梯度，大幅降低计算成本。

**区域损失定义**（无需标签）：
$$\mathcal{L}_{RGS}^l(\mathbf{X}_n^l) = \|f^l(\mathbf{X}_n^l)\|_2$$

即第 $l$ 个 decoder block 输出的 L2 范数。对此损失做一次反向传播即可获得该 block 内所有权重的梯度。

**RGS 剪枝准则**：
$$S_{ij} = (\alpha \cdot G_{ij} + \|\mathbf{X}_j\|_2) \cdot |W_{ij}|$$

- $G_{ij}$：区域梯度幅值（RMS over N 个样本）
- $\|\mathbf{X}_j\|_2$：Wanda 原始的输入激活范数
- $\alpha = 100$：平衡梯度和激活的缩放因子
- 区域梯度每个 block 只计算一次，与逐层更新的激活范数融合，兼顾效率和准确性

### 2. Regional Optimization (RO)

在每轮 RGS 剪枝后，微调 block 内权重以修复剪枝引入的误差：

$$\mathcal{L}_{ro}^{l,k}(\hat{\mathbf{X}}_m^l) = (f^l(\hat{\mathbf{X}}_m^l) - \hat{f}_k^l(\hat{\mathbf{X}}_m^l))^2$$

- 稠密块输出 vs 剪枝块输出的 MSE 损失
- 从 128 个校准样本中随机选 32 个做 RO
- 使用 RMSprop 优化器，学习率 3e-7
- 每个 block 迭代 K=5 轮（RGS剪枝 → RO优化 → 再剪枝恢复稀疏）

### 3. 算法流程

对 L 个 decoder block 顺序处理：
1. 计算区域梯度 G（一次反向传播）
2. K 轮迭代：RGS 剪枝 → RO 权重更新
3. 最终 RGS 剪枝恢复稀疏约束
4. 更新下一 block 的输入隐状态

## 实验结果

### 表1：WikiText 困惑度（↓越低越好）

| 方法 | LLaMA-1 7B (2:4) | LLaMA-1 13B (2:4) | OpenLLaMA 3B (2:4) | LLaMA-3.1 8B (2:4) |
|:---|:---:|:---:|:---:|:---:|
| Dense 基线 | 5.68 | 5.09 | 7.27 | 6.39 |
| Wanda | 11.53 | 9.58 | 28.04 | 24.83 |
| GBLM | 11.33 | 9.16 | 24.75 | 24.34 |
| SparseGPT | 11.00 | 9.11 | 15.91 | - |
| **Wanda++** | **9.43 (-19%)** | **7.75 (-20%)** | **19.03 (-32%)** | **18.32 (-26%)** |

小模型（3B/7B）提升最显著；2:4 稀疏比非结构化和 4:8 更受益。

### 表2：零样本下游任务精度（LLaMA-1 7B, 2:4 稀疏）

| 方法 | MRPC | HellaSwag | ARC-e | RTE | MMLU |
|:---|:---:|:---:|:---:|:---:|:---:|
| Dense | 69.12 | 56.96 | 75.29 | 66.43 | 35.10 |
| Wanda | 46.81 | 41.66 | 59.34 | 49.82 | 25.85 |
| **Wanda++** | **68.38 (+46%)** | **45.31 (+8%)** | **63.72 (+7%)** | **62.09 (+24%)** | **27.52 (+6%)** |

MRPC 和 RTE 任务几乎恢复到稠密基线水平。

### 剪枝效率（LLaMA-1 7B）

| 方法 | 时间 | 显存 |
|:---|:---:|:---:|
| GBLM | 5801s | 26 GB |
| SparseGPT | 322s | 23 GB |
| Wanda | 55s | 22 GB |
| **Wanda++** | **290s** | **25 GB** |

显存与 Wanda 相当，时间为 GBLM 的 1/20。

## 亮点

- **区域梯度概念巧妙**：通过 block 级别反向传播避免全模型 BP，将梯度计算成本从 O(全模型) 降至 O(单block)，且显存不随模型总层数增长
- **RGS + RO 双阶段互补**：RGS 改善剪枝决策，RO 修复剪枝误差，两者结合效果 > 各自单独
- **与 LoRA 微调正交**：Wanda++ 剪枝后再做 LoRA 微调可进一步提升，两种技术不冲突
- **可扩展至超大模型**：理论分析 530B 模型单 block 优化仅需 ~40GB 显存，单 GPU 可行

## 局限性

- **2:4 稀疏仍有明显精度损失**：即便 Wanda++ 仍然 7B 2:4 困惑度从 5.68 升到 9.43，远未达到量化的"几乎无损"水平
- **对校准数据量敏感**：小校准集（<64 样本）时 RO 效果不稳定
- **RO 迭代增加时间**：290s vs Wanda 的 55s，约 5× 慢（但仍比 GBLM 快 20×）
- **仅评估 LLaMA 系列**：未在 Qwen、Mistral 等其他架构上验证泛化性

## 相关工作对比

| 维度 | Wanda++ | Wanda | GBLM | SparseGPT |
|:---|:---|:---|:---|:---|
| 梯度信息 | 区域梯度（block级） | 无 | 全模型梯度 | 二阶 Hessian 近似 |
| 权重更新 | RO（block级 MSE） | 无 | 无 | 逐列 OBS 更新 |
| 7B 剪枝时间 | ~5 min | ~1 min | ~97 min | ~5 min |
| 显存需求 | 与 Wanda 相当 | 最低 | 全模型加载 | 中等 |
| 2:4 困惑度改善 | 最优（-19 ~ -32%） | 基线 | 微弱 | 中等 |

## 评分

- ⭐⭐⭐⭐ 新颖性：区域梯度替代全模型梯度的思路简单而有效，block 级 MSE 优化降低误差传播
- ⭐⭐⭐⭐ 实用性：10 分钟/7B 单 GPU、与 LoRA 正交、可扩展至超大模型，工程部署友好
- ⭐⭐⭐⭐ 实验充分度：4 个模型系列、3 种稀疏模式、困惑度+下游任务+效率+延迟+消融全覆盖
- ⭐⭐⭐ 写作质量：方法描述清晰但部分实验图表不够直观，零样本结果部分任务反而下降未充分解释

<!-- RELATED:START -->

## 相关论文

- [IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)
- [TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)

<!-- RELATED:END -->
