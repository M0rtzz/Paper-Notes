---
title: >-
  [论文解读] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets
description: >-
  [ACL 2025][LLM/NLP][集成学习] 提出 EnsembleLoRA——一种面向多数据集微调的高效集成方法，利用一阶 Taylor 近似快速估计任务亲和度将数据集分组，为每组训练一个 adapter 后加权组合，在 10 个 SuperGLUE 任务上以仅 9% 额外计算代价将 QLoRA 的平均测试准确率提升 10%。
tags:
  - ACL 2025
  - LLM/NLP
  - 集成学习
  - LoRA
  - 多数据集微调
  - 任务亲和度
  - 梯度近似
  - 参数高效微调
---

# Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets

**会议**: ACL 2025  
**arXiv**: [2505.21930](https://arxiv.org/abs/2505.21930)  
**代码**: [https://github.com/VirtuosoResearch/EnsembleLoRA](https://github.com/VirtuosoResearch/EnsembleLoRA)  
**作者**: Dongyue Li, Ziniu Zhang, Lu Wang, Hongyang R. Zhang  
**机构**: Northeastern University, University of Michigan  
**领域**: NLP / 高效微调  
**关键词**: 集成学习, LoRA, 多数据集微调, 任务亲和度, 梯度近似, 参数高效微调

## 一句话总结

提出 EnsembleLoRA——一种面向多数据集微调的高效集成方法，利用一阶 Taylor 近似快速估计任务亲和度将数据集分组，为每组训练一个 adapter 后加权组合，在 10 个 SuperGLUE 任务上以仅 9% 额外计算代价将 QLoRA 的平均测试准确率提升 10%。

## 研究背景与动机

**领域现状**：参数高效微调（PEFT）方法如 LoRA/QLoRA 在单数据集适配时高效。但实际场景中评估标准常涉及多个数据集/任务的混合，如何高效适配成为问题。

**现有方法的不足**：
   - **单 adapter 训练**所有数据集 → 任务间负迁移。实验表明，在 45 个两两组合中，QLoRA 有 33 个出现性能下降
   - **先预训练再微调（MTL-FT）**→ 计算开销翻倍（预训练 + 任务特定微调）
   - **每任务一个 adapter** → 内存开销与任务数线性增长（10 个任务 = 10 个 adapter）
   - 核心挑战：如何理解数据集间的关系，找到合适的分组策略

**核心动机**：利用 LoRA 微调后权重与基础模型极为接近的性质（< 0.2% 相对距离），通过一阶展开快速估计任务间的亲和度，高效分组后用少量 adapter 实现集成。

## 方法详解

### 整体框架

给定 $n$ 个数据集和一个基础适配方法（如 QLoRA），算法分三步：

1. **任务亲和度分组**：估计 $n \times n$ 任务亲和度矩阵 → 聚类分为 $m$ 组（$m \ll n$）
2. **Adapter 训练**：每组训练一个 adapter → $m$ 个 adapter
3. **梯度提升精化**：对损失最大的组添加额外 adapter → 最终 $M = m + b$ 个 adapter → 加权集成

### 关键设计一：一阶近似估计微调性能

**核心观察**：LoRA/Adapter 等 PEFT 方法微调后权重与基础模型的相对距离极小：

| 方法 | Llama-3-1B | Llama-3-3B | Llama-3-8B |
|------|-----------|-----------|-----------|
| LoRA | 0.16% | 0.14% | 0.12% |
| QLoRA | 0.18% | 0.16% | 0.11% |
| Adapter | 0.09% | 0.05% | 0.08% |
| QAdapter | 0.11% | 0.08% | 0.07% |

因此可以用一阶 Taylor 展开近似模型输出：

$$h_X(s, y) \approx h_{\theta^*}(s, y) + [\nabla_X h_{\theta^*}(s, y)]^{\top}(X - \theta^*)$$

实验表明该近似在 Llama/GPT-J 模型（最大 34B 参数）上误差 < 1%（LoRA/Adapter）或 < 3%（QLoRA/QAdapter）。

### 关键设计二：梯度投影 + 回归估计

1. 在基础模型上对所有训练样本计算一次梯度
2. 使用随机投影将梯度降维至几百维（Johnson-Lindenstrauss 保距性质）
3. 对任意数据集子集 $S$，通过求解 logistic 回归（在 CPU 上几秒完成）估计微调后的 adapter 权重 $\hat{\theta}_S$
4. 用 $\hat{\theta}_S$ 评估各任务的估计性能 $\hat{f}_i(S)$
5. 遍历 $k$ 个随机子集，计算任务亲和度矩阵 $T_{i,j}$

**估计结果**（投影维度 $d$ vs. 估计误差）：

| $d$ | Llama-3-1B | Llama-3-3B | Llama-3-8B | 加速比 |
|-----|-----------|-----------|-----------|--------|
| 200 | 8.2% | 8.1% | 7.0% | $10^5 \times$ |
| 400 | 4.7% | 4.8% | 4.3% | $10^5 \times$ |
| 800 | 4.6% | 4.4% | 4.2% | $10^5 \times$ |

以 $10^5$ 倍的加速代价，估计误差控制在 5% 以内。

### 关键设计三：聚类分组

基于半正定规划松弛的聚类算法最大化组内亲和度密度，并使用 trace 正则化自动确定组数 $m$。

### 关键设计四：梯度提升精化

完成初始分组训练后，对训练损失最大的组添加新 adapter：
- 拟合残差（负梯度 $1 - p_s$）
- 仍使用梯度近似求解线性回归
- 第一步提升即可降低 18% 训练误差，对应 0.4% 测试准确率提升

### 计算/内存开销对比

| 方法 | 运行时间 | 内存 |
|------|---------|------|
| 基础微调 | $T$ | $A$ |
| 先预训练再微调 | $\approx 2T$ | $nA$ |
| **EnsembleLoRA** | $T + G$ | $MA$ |

其中 $T$ 是基础微调时间，$G$ 是一次梯度计算（$G \ll T$），$M$ 是 ensemble 中 adapter 数量。

## 实验关键数据

### 实验设置

- 基础模型：Llama-3.1-8B, CodeLlama-34B-Instruct
- 10 个 SuperGLUE 任务（分 5 类：句子完成、自然语言推理、共指消解、问答、词义歧义）
- 对比基线：Base fine-tuning, MTL-FT (Liu et al., 2019), TAG (Fifty et al., 2021)

### Llama-3-8B + QLoRA 主要结果

| 方法 | 平均准确率 | FLOPs | GPU 内存 |
|------|-----------|-------|---------|
| Full FT | 84.6% | $6.0 \times 10^{19}$ | 73.0 GB |
| QLoRA (单 adapter) | ~70% | 基准 | 基准 |
| MTL-FT | 与 Ours 相当 | 高 45% | 高 45% |
| **EnsembleLoRA** | **QLoRA +10%** | QLoRA +9% | QLoRA +9GB |

### 关键发现

1. **单 adapter 的负迁移严重**：在 QLoRA 的 45 个两两组合中，73% (33/45) 出现负迁移
2. **集成有效但需控制规模**：朴素地训练 10 个 adapter 集成可提升 10.8%，但内存开销增至 4 倍
3. **分组 + 梯度提升 = 高效集成**：EnsembleLoRA 以约 3 个 adapter 实现了相当的效果
4. **可扩展到 500 个数据集**（联邦学习设置），计算减少 90%，内存减少 91%
5. **泛化理论分析**：小 rank 的 LoRA adapter 具有最低泛化误差，集成进一步降低

### CodeLlama-34B 结果

在 34B 参数模型上，EnsembleQLoRA 相比单 QLoRA 提升 3% 准确率，仅增加 8% FLOPs。

## 亮点与洞察

1. **核心洞察精彩**：PEFT 方法微调后权重距基础模型 < 0.2%，使得一阶近似成为可能——这个观察将复杂的多数据集微调问题转化为梯度空间的回归问题
2. **$10^5$ 倍加速估计**非常实用——所有亲和度估计在 CPU 上几秒完成
3. **方法通用性强**：适用于 LoRA, Adapter, QLoRA, QAdapter 四种 PEFT 方法
4. **理论与实践结合**：不仅有泛化误差分析，还用 sharpness 衡量验证了小 rank adapter 的泛化优势

## 局限性

1. 仅评估分类任务（SuperGLUE），生成任务（如摘要、翻译）未验证
2. 一阶近似对 QLoRA 的误差稍大（< 3% vs < 1%），可能在极端量化下失效
3. 聚类算法基于半正定规划，在任务数很多时（如 500）计算量可能较大
4. 未讨论数据集大小严重不均匀时的处理策略
5. 梯度提升步数 $b$ 和组数 $m$ 的选择仍需调优

## 相关工作

- **参数高效微调**：LoRA (Hu et al., 2021), QLoRA (Dettmers et al., 2023), Adapter-tuning (Houlsby et al., 2019)
- **多任务学习**：MTL-FT (Liu et al., 2019), TAG (Fifty et al., 2021)
- **影响函数/数据建模**：Koh & Liang (2017), Ilyas et al. (2022), Park et al. (2023)

## 评分

⭐⭐⭐⭐⭐ (5/5)

这是一篇技术贡献非常扎实的工作。核心观察（PEFT 权重极接近基础模型）简洁而深刻，由此推导出的方法既有理论保证又在实践中高效。实验覆盖面广，从 1B 到 34B 模型、从 10 到 500 个数据集均有验证。在多数据集微调这一实际且重要的场景中给出了实用的解决方案。

<!-- RELATED:START -->

## 相关论文

- [A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)
- [Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](salience_sparse_fine_tuning.md)
- [HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)

<!-- RELATED:END -->
