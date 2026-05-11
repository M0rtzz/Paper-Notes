---
title: >-
  [论文解读] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training
description: >-
  [NeurIPS 2025][data mixture] NVIDIA 提出 CLIMB 框架，通过嵌入聚类 + 迭代自举搜索自动发现最优预训练数据混合比例，在 1B 模型上超过 Llama-3.2-1B 达 2.0%，并发布了 1.2T token 的 ClimbLab 语料库和 400B token 的 ClimbMix 高质量数据集。
tags:
  - NeurIPS 2025
  - data mixture
  - pre-training
  - clustering
  - LLM预训练
  - LLM
---

# Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training

**会议**: NeurIPS 2025  
**arXiv**: [2504.13161](https://arxiv.org/abs/2504.13161)  
**代码**: [数据已公开](https://huggingface.co/nvidia/Nemotron-ClimbMix)  
**领域**: LLM预训练  
**关键词**: data mixture, pre-training, clustering, iterative optimization, LLM

## 一句话总结

NVIDIA 提出 CLIMB 框架，通过嵌入聚类 + 迭代自举搜索自动发现最优预训练数据混合比例，在 1B 模型上超过 Llama-3.2-1B 达 2.0%，并发布了 1.2T token 的 ClimbLab 语料库和 400B token 的 ClimbMix 高质量数据集。

## 研究背景与动机

大规模预训练数据集（如 Common Crawl）缺乏显式的领域标签，手工标注（如 The Pile）又极其耗时。即使有了领域标签，数据混合比例与模型性能之间存在复杂的非线性关系，最优混合仍然是一个开放性难题。现有方法面临的核心挑战包括：

**缺乏领域划分**：Web 规模数据无法直接提供领域信息，传统的困惑度或教育价值过滤不够精准

**混合策略复杂**：优化编程能力不仅需要代码数据，还需要数学、推理、安全等互补知识

**搜索代价高昂**：穷举不同混合比例并训练完整模型在计算上不可行

已有工作如 DoReMi 和 RegMix 尝试了数据混合优化，但前者依赖预定义领域，后者仅执行单次搜索。CLIMB 的核心创新在于**自动发现领域**并**迭代优化混合权重**。

## 方法详解

### 整体框架

CLIMB 框架分为两个阶段：**数据预处理**（嵌入 + 聚类）和**迭代自举搜索**（混合权重优化）。

**阶段一：数据预处理**

1. **文本嵌入**：使用 stella_en_400M_v5 模型将文档 $\hat{D} = \{D_1, D_2, \dots, D_n\}$ 映射到嵌入空间，得到向量集 $E = \{E_1, E_2, \dots, E_n\}$
2. **嵌入聚类**：使用 FAISS 中的 K-means 算法将嵌入聚为 $K_{\text{init}} = 1000$ 个初始簇
3. **簇合并**：先用 fasttext 模型评估数据质量（整体质量、教育价值、信息价值、广告得分），以阈值 3.0 剪枝到 $K_{\text{pruned}} = 240$ 个簇，再根据欧氏距离阈值 1.5 合并为 $K_{\text{enhanced}}$ 个超级簇

**阶段二：迭代自举搜索**

### 关键设计

核心思想是将数据混合构建为**双层优化问题**：

$$\min_{\alpha \in A} \ell_{val}(\alpha, \omega^*(\alpha)) \quad \text{s.t.} \quad \omega^*(\alpha) = \arg\min_{\omega} \ell_{train}(\alpha, \omega), \quad \sum_{i=1}^{k} \alpha_i = 1, \alpha_i \geq 0$$

直接求解需要为每个 $\alpha$ 训练完整模型，计算代价极高。CLIMB 使用**预测器** $f_\theta(\alpha)$ 近似目标函数 $\ell(\alpha, \omega)$，将问题转化为：

$$\min_{\alpha \in A} f(\alpha | S) \quad \text{s.t.} \quad f = \arg\min_{S, f \in \tilde{\mathcal{F}}} \sum_{s \in S} \mathcal{L}(f(s), \ell(s, w^*))$$

不同于 RegMix 的单次搜索，CLIMB 通过**坐标下降法**交替优化采样策略 $S$ 和预测器 $f_\theta$：

- **子例程 1（配置采样）**：在第 $k+1$ 次迭代中，用预测器 $f_k$ 排序所有未测试配置，从 Top-N 中随机采样 $M$ 个新配置，兼顾利用与探索
- **子例程 2（弱预测器拟合）**：用 LightGBM 回归模型在累积样本 $S^{k+1}$ 上训练预测器 $f_{\theta}^{k+1}$

迭代搜索分三轮进行，分别评估 64、32、16 个候选配置（共 112 次搜索），使用 Dirichlet 分布初始化采样。LightGBM 设置了 L1/L2 正则化、最大深度 4、最少 5 个叶子样本、早停 20 轮等防过拟合策略。

### 损失函数 / 训练策略

- **基础模型训练**：先在 10T token（DCLM + TxT360）上进行 Phase-1 预训练，使用 WSD 学习率调度
- **代理模型**：使用 62M 和 350M 模型进行高效搜索，主实验使用 350M 代理
- **目标模型评估**：以 PIQA、ARC_E、HellaSwag 验证集为优化目标，在测试集上评估泛化

## 实验关键数据

### 主实验

| 模型规模 | 方法 | piqa | arc_c | arc_e | hellaswag | winogrande | siqa | 平均 |
|---------|------|------|-------|-------|-----------|------------|------|------|
| 350M | Random | 71.16 | 30.54 | 62.50 | 52.14 | 55.40 | 41.29 | 52.17 |
| 350M | DoReMi | 70.29 | 33.53 | 66.41 | 52.25 | 55.95 | 41.86 | 53.38 |
| 350M | RegMix | 71.92 | 33.42 | 66.12 | 53.69 | 55.27 | 42.23 | 53.78 |
| 350M | **CLIMB** | **72.21** | **34.87** | **67.25** | **55.32** | **56.79** | **42.54** | **54.83** |
| 1B | Random | 74.05 | 37.12 | 70.24 | 62.90 | 60.77 | 42.48 | 57.93 |
| 1B | DoReMi | 74.91 | 40.01 | 72.34 | 63.53 | 61.08 | 43.09 | 59.16 |
| 1B | RegMix | 75.22 | 40.42 | 71.32 | 64.73 | 62.33 | 42.22 | 59.37 |
| 1B | **CLIMB** | **75.78** | **40.98** | **72.97** | **66.01** | **63.32** | **43.37** | **60.41** |

在 400B token 训练下，950M 的 CLIMB 模型平均得分 53.54%，超过 Llama-3.2-1B 的 51.56% 达 2.0%。

### 消融实验

| 消融维度 | 配置 | 平均准确率 |
|---------|------|-----------|
| 搜索计算量 100% | 64:32:16 | 60.41 |
| 搜索计算量 150% | — | 60.72 |
| 搜索计算量 200% | — | 61.12 |
| 计算分配 6:1 | 2 轮 | 60.05 |
| 计算分配 4:2:1 | 3 轮 | **60.41** |
| 计算分配 2:2:1:1 | 4 轮 | 60.14 |
| 代理模型 62M | — | 60.11 |
| 代理模型 132M | — | 60.19 |
| 代理模型 350M | — | **60.41** |
| 初始化：Random | — | 60.21 |
| 初始化：Dirichlet | — | **60.41** |

### 关键发现

1. **迭代搜索显著优于单次搜索**：CLIMB 每次迭代都带来稳定提升，iter1→iter3 在 Social Sciences 上从 40.18% 提至 41.79%
2. **领域特化效果显著**：针对 Social Sciences 优化可比随机采样提升 5%
3. **代理模型规模影响有限**：62M 代理仅比 350M 低 0.3%，说明小代理即可高效搜索
4. **计算分配平衡很关键**：4:2:1 的三轮迭代是最佳深度-广度平衡
5. **从头训练需要更均衡的混合**：与持续训练不同，从头训练需要更多样化的聚类覆盖

## 亮点与洞察

- **端到端自动化**：不依赖人工领域标签，从嵌入聚类到混合搜索全自动
- **弱预测器 + 迭代精化**：借鉴 boosting 思想，用多轮弱预测器逐步聚焦高质量搜索空间
- **开源贡献**：发布 1.2T ClimbLab（20 个语义簇）和 400B ClimbMix 数据集，为数据混合研究提供统一实验平台
- **实用洞察**：分析了最优混合中各簇的权重分布和主题特征，发现通用推理任务主要依赖 C8（数学/逻辑）、C9（科技新闻）、C18（教育）、C19（百科）四个簇

## 局限与展望

1. **搜索空间受限于聚类粒度**：21 个超级簇可能无法充分表达细粒度领域差异
2. **代理模型与目标模型的对齐假设**：虽然实验验证了 350M→1B 的迁移，但更大模型（7B+）是否仍然有效尚未验证
3. **静态混合 vs 动态课程**：CLIMB 找到的是全局固定混合比例，未探索训练过程中动态调整比例的潜力
4. **评估覆盖面**：主要在推理 benchmark 上验证，对代码生成、对话能力等实用场景的影响未深入分析

## 相关工作与启发

- **RegMix**（Liu et al.）：CLIMB 的直接前身，CLIMB 将其单次搜索扩展为迭代框架
- **DoReMi**（Xie et al.）：基于 DRO 的在线混合优化，依赖预定义领域
- **WebOrganizer**（Wettig et al.）：并行工作，用分类器标注 Web 数据，而 CLIMB 用聚类更直接
- 启发：数据质量 > 数据数量的范式在持续强化，自动化的数据 curation pipeline 是 LLM 预训练的重要研究方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 迭代自举 + 嵌入聚类的组合有新意，但每个组件单独看并不全新
- 实验充分度: ⭐⭐⭐⭐⭐ 消融极其详尽（计算量、分配、代理、聚类、初始化），覆盖 350M-1B 多尺度验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，图示直观
- 价值: ⭐⭐⭐⭐⭐ 发布高质量开源数据集 + 实用的自动混合框架，对 LLM 预训练社区价值极大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)
- [\[ICML 2025\] Metadata Conditioning Accelerates Language Model Pre-training](../../ICML2025/llm_pretraining/metadata_conditioning_accelerates_language_model_pre-training.md)
- [\[NeurIPS 2025\] Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)
- [\[ICML 2025\] The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training](../../ICML2025/llm_pretraining/the_sharpness_disparity_principle_in_transformers_for_accelerating_language_mode.md)
- [\[ACL 2025\] Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models](../../ACL2025/llm_pretraining/optimizing_pre-training_data_mixtures_with_mixtures_of_data_expert_models.md)

</div>

<!-- RELATED:END -->
