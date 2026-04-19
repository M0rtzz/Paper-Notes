---
title: >-
  [论文解读] Protriever: End-to-End Differentiable Protein Homology Search for Fitness Prediction
description: >-
  [ICML 2025][医学图像][蛋白质同源搜索] 提出 Protriever，首个端到端可微的蛋白质同源序列检索框架，将检索器与阅读器联合训练，在蛋白质适应性预测任务上达到序列模型 SOTA，同时比传统 MSA 检索快两个数量级。
tags:
  - ICML 2025
  - 医学图像
  - 蛋白质同源搜索
  - 端到端可微检索
  - 蛋白质适应性预测
  - 向量检索
  - 蛋白质语言模型
---

# Protriever: End-to-End Differentiable Protein Homology Search for Fitness Prediction

**会议**: ICML 2025  
**arXiv**: [2506.08954](https://arxiv.org/abs/2506.08954)  
**领域**: 蛋白质建模 / 计算生物学  
**关键词**: 蛋白质同源搜索, 端到端可微检索, 蛋白质适应性预测, 向量检索, 蛋白质语言模型

## 一句话总结

提出 Protriever，首个端到端可微的蛋白质同源序列检索框架，将检索器与阅读器联合训练，在蛋白质适应性预测任务上达到序列模型 SOTA，同时比传统 MSA 检索快两个数量级。

## 研究背景与动机

- **蛋白质同源序列检索**是蛋白质建模（适应性预测、蛋白质设计、结构预测、蛋白质-蛋白质相互作用）的基础步骤
- 传统工作流是**两阶段流水线**：先通过多序列比对（MSA）检索同源序列，再在这些比对上训练模型
- 传统 MSA 方法（JackHMMER、MMseqs2）存在根本性局限：
    - **漏检远亲序列**：低于比对显著性阈值的远亲序列被遗漏，丢失有价值的进化上下文
    - **难以处理复杂 indel**：包含大量插入、缺失或结构重排的序列难以可靠比对
    - **与下游任务脱节**：检索基于固定的序列相似性启发式规则，而非为特定任务优化
    - **计算昂贵**：每个蛋白质家族需要单独构建 MSA 并训练模型，不适合大规模应用
- **蛋白质语言模型（pLM）**如 ESM、Tranception 等提供了无比对替代方案，但单序列模型在突变效应预测中常逊于家族特异性方法
- **混合方法**虽然结合了 pLM 和进化信息，但检索仍是静态的，无法反向传播以优化检索选择
- **核心洞察**：将 NLP 中的 RAG（检索增强生成）范式引入蛋白质建模——让模型自己学习哪些同源序列对下游任务最有用

## 方法详解

### 整体框架

Protriever 由三个核心组件构成（如 Figure 1 所示）：

1. **Retriever（检索器）**：将查询序列编码为向量，通过向量相似度搜索从索引中检索同源序列
2. **Index（索引）**：预计算的蛋白质序列嵌入库，支持快速近似最近邻搜索
3. **Reader（阅读器）**：接收检索到的序列集合，执行下游任务（如自回归解码查询序列）

训练时 Reader 计算每个检索文档的相关性得分 $p_{\text{LM}}(\mathbf{q} \mid \mathbf{d}_k)$，梯度反传给 Retriever 以调整嵌入空间。

### 检索器模块

**初始化**：
- 使用 ESM-2 预训练权重初始化 Transformer encoder（35M 参数）
- 最后一层输出做 average pooling 得到 480 维向量表示
- 用余弦相似度 $s(\mathbf{d}, \mathbf{q})$ 计算序列间相似性

**DPR 预训练**：
- 采用 Dense Passage Retrieval（DPR）范式进一步预训练
- 目标：构建嵌入空间使同源序列相似度高、非同源序列相似度低
- 在 UniRef50 上用 BLAST all-vs-all 搜索构建训练数据
- 对比学习损失：给定查询 $\mathbf{q}_i$，最大化正例相似度，最小化负例相似度

$$\mathcal{L}_{\text{DPR}} = -\log \frac{e^{s(\mathbf{q}_i, \mathbf{d}_i^+)}}{e^{s(\mathbf{q}_i, \mathbf{d}_i^+)} + \sum_j e^{s(\mathbf{q}_i, \mathbf{d}_{i,j}^-)}}$$

**困难负例挖掘**：
- 从索引中检索 top-K 相似但非同源的序列作为困难负例
- 随训练异步更新索引以保持负例质量

### 阅读器模块

- 使用自回归蛋白质语言模型（ProtGPT2, 36M 参数）
- Reader 以拼接的检索序列为条件，自回归重建查询序列
- 对每个检索文档独立计算条件似然，作为相关性得分
- Reader 损失为加权负对数似然

$$\mathcal{L}_{\text{reader}} = -\sum_{t=1}^{T} \log p(\mathbf{q}_t \mid \mathbf{q}_{<t}, \mathcal{D}_{K_f})$$

### 联合训练

- **端到端梯度传播**：Reader 的梯度通过相关性得分反传给 Retriever
- 多种混合损失组合策略：Reader-only loss、DPR + Reader joint loss、加权组合
- **采样策略**：探索不同的检索序列采样方式（top-K、带温度的采样等）以优化效果

### 加速检索

- **倒排文件索引（IVF）**：将向量空间划分为 Voronoi 单元，仅搜索最近的几个分区
- **乘积量化（PQ）**：将 480 维向量压缩为短编码，大幅减少存储和距离计算开销
- FAISS 库实现高效近似最近邻搜索

## 实验关键数据

### 表1：ProteinGym 基准上的适应性预测性能（Spearman rho）

| 方法 | 类别 | 检索方式 | 平均 Spearman rho |
|------|------|----------|----------------|
| EVE | 家族特异性 | MSA (JackHMMER) | 0.456 |
| DeepSequence | 家族特异性 | MSA | 0.440 |
| ESM-1v | 单序列 pLM | 无 | 0.421 |
| ESM-2 (650M) | 单序列 pLM | 无 | 0.434 |
| Tranception (L) | 混合 | MSA（推理时） | 0.462 |
| PoET | 混合 | MSA (JackHMMER) | 0.467 |
| ProtMamba | 混合 | MSA | 0.459 |
| **Protriever** | **端到端检索** | **向量搜索** | **0.471** |

> Protriever 在序列模型中取得最佳，超越所有基于 MSA 检索的方法。

### 表2：检索速度对比

| 检索方法 | 数据库大小 | 每序列检索时间 | 相对速度 |
|----------|-----------|---------------|---------|
| JackHMMER | UniRef90 | ~300s | 1x |
| MMseqs2 (CPU) | UniRef90 | ~30s | 10x |
| MMseqs2-GPU | UniRef90 | ~3s | 100x |
| **Protriever (FAISS)** | **UniRef50** | **~0.3s** | **1000x** |

> 向量检索比 JackHMMER 快约三个数量级，比 MMseqs2 快约两个数量级。

### 表3：消融实验——训练策略与组件影响

| 配置 | Spearman rho | 说明 |
|------|-----------|------|
| Reader only（随机检索） | 0.412 | 无有意义检索 |
| Reader + 冻结 ESM-2 检索器 | 0.438 | 检索器不更新 |
| Reader + DPR 预训练检索器（冻结） | 0.451 | DPR 预训练有效 |
| Reader + DPR 检索器（微调） | 0.463 | 端到端微调进一步提升 |
| **Protriever（完整联合训练）** | **0.471** | 联合训练最优 |

> 消融实验证明：(1) DPR 预训练提供良好检索器初始化；(2) 端到端联合训练相比冻结检索器增益显著（+0.020）。

### 表4：不同检索数据库与推理灵活性

| 推理时数据库 | Spearman rho | 说明 |
|-------------|-----------|------|
| UniRef50 | 0.471 | 训练同库 |
| UniRef90 | 0.468 | 更大库，泛化良好 |
| BFD | 0.465 | 跨库泛化 |
| 只用 top-10 序列 | 0.464 | 少量检索也有效 |
| 用 top-100 序列 | 0.471 | 更多检索序列略优 |

> Protriever 在推理时可灵活切换数据库和检索数量，体现了架构不可知性和任务不可知性。

## 亮点与创新

1. **首个端到端可微的蛋白质同源检索**：打破传统 MSA 到模型的两阶段范式，让检索器和阅读器联合优化，使模型自主学习对下游任务最有价值的同源序列
2. **极速检索**：通过 FAISS 向量索引（IVF + PQ）实现比 JackHMMER 快约1000倍的检索，使大规模蛋白质分析成为可能
3. **架构和任务不可知**：检索器与阅读器可独立替换，推理时可切换不同蛋白质数据库，灵活适配多种下游任务
4. **NLP RAG 到生物的成功迁移**：受 REALM/RAG 启发，将 retriever-reader 可微框架从 NLP 迁移到蛋白质序列建模，并针对蛋白质场景做了关键适配
5. **轻量高效**：检索器仅 35M 参数（ESM-2 小模型），阅读器 36M 参数，总模型规模远小于 ESM-2 650M 等大模型

## 局限性

1. **检索数据库依赖**：检索质量仍受限于索引中的蛋白质序列覆盖范围，对于极端小众蛋白质家族可能效果有限
2. **索引异步更新**：训练过程中索引不是实时更新的（周期性重建），存在检索一致性滞后问题
3. **仅验证适应性预测**：虽声称架构和任务不可知，但目前仅在 fitness prediction 一个下游任务上做了验证
4. **Reader 容量限制**：ProtGPT2（36M）作为 Reader 规模较小，更大的 Reader 可能带来更好的性能但增加开销
5. **近似检索引入噪声**：IVF + PQ 压缩加速检索的同时会引入近似误差，可能影响少数难例

## 相关工作

- **比对方法**：PSSM、HMM -> EVE、DeepSequence -> 依赖 MSA 的家族特异性模型
- **蛋白质语言模型**：UniRep -> ESM、ESM-2、ProGen、Tranception、ProtGPT2 -> 无比对但缺乏家族特异性
- **混合方法**：MSA Transformer、PoET、ProtMamba -> 结合 pLM 与进化信息但检索仍静态
- **检索增强**：NLP 中的 DPR、REALM、RAG -> 蛋白质领域的 AIDO.RAG、RSA -> 但都未实现端到端联合训练
- **MSA 检索工具**：JackHMMER、MMseqs2、BLAST -> 传统序列相似性搜索

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐⭐：首次实现蛋白质同源检索的端到端可微训练，概念突破显著
- **实验充分性** ⭐⭐⭐⭐：ProteinGym 标准基准上全面对比，消融实验详尽
- **实用价值** ⭐⭐⭐⭐⭐：检索速度提升 2~3 数量级同时保持 SOTA 性能，工业应用价值大
- **写作质量** ⭐⭐⭐⭐：逻辑清晰、方法描述详尽
- **局限**: 仅验证了适应性预测一个下游任务；小规模 Reader 可能限制上限

<!-- RELATED:START -->

## 相关论文

- [Revisiting End-to-End Learning with Slide-level Supervision in Computational Pathology](../../NeurIPS2025/medical_imaging/revisiting_end-to-end_learning_with_slide-level_supervision_in_computational_pat.md)
- [UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection](../../NeurIPS2025/medical_imaging/unisite_the_first_cross-structure_dataset_and_learning_framework_for_end-to-end_.md)
- [Steering Generative Models with Experimental Data for Protein Fitness Optimization](../../NeurIPS2025/medical_imaging/steering_generative_models_with_experimental_data_for_protein_fitness_optimizati.md)
- [Steering Protein Language Models](steering_protein_language_models.md)
- [GraphFLA: Augmenting Biological Fitness Prediction Benchmarks with Landscape Features](../../NeurIPS2025/medical_imaging/augmenting_biological_fitness_prediction_benchmarks_with_landscapes_features_fro.md)

<!-- RELATED:END -->
