---
title: >-
  [论文解读] Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI
description: >-
  [ICLR 2026][脑机接口] 提出 FunctionalMap 框架，通过对比学习从颅内局部场电位（LFP）中学习被试无关的功能嵌入作为"功能坐标系"，替代不可靠的 MNI 解剖坐标，结合 Transformer 实现跨被试、跨电极的神经数据聚合和信号重建，在 20 名被试的多脑区 SEEG 数据集上验证有效。
tags:
  - ICLR 2026
  - 脑机接口
  - SEEG
  - 功能嵌入
  - 对比学习
  - Transformer
  - 跨被试建模
  - 神经信号
---

# Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI

**会议**: ICLR 2026  
**arXiv**: [2510.27090](https://arxiv.org/abs/2510.27090)  
**代码**: [GitHub](https://github.com/ICLR-Functional-Embedding/ICLR2026_Functional_Map)  
**领域**: llm_nlp  
**关键词**: 脑机接口, SEEG, 功能嵌入, 对比学习, Transformer, 跨被试建模, 神经信号

## 一句话总结

提出 FunctionalMap 框架，通过对比学习从颅内局部场电位（LFP）中学习被试无关的功能嵌入作为"功能坐标系"，替代不可靠的 MNI 解剖坐标，结合 Transformer 实现跨被试、跨电极的神经数据聚合和信号重建，在 20 名被试的多脑区 SEEG 数据集上验证有效。

## 研究背景与动机

颅内神经记录（如 SEEG/DBS）的跨被试建模面临两大核心困难：

**解剖变异性和不一致的电极覆盖**：电极的数量、位置和覆盖区域因临床需求而异。标准的 MNI 图谱对齐假设空间对应等于功能相似，但**匹配解剖坐标处的记录常捕获不同的功能角色**，极端情况下甚至是完全不同的脑区。

**多区域记录的异质性**：现代 DBS 手术同时从多个基底节和丘脑核团采样（GPi、STN、VO、VA、VIM 等），提供了研究区间通信的独特机会，但其异质性放大了对齐问题。

**核心假设**：神经信号通过其功能特征（而非解剖坐标）可以更可靠地跨被试对齐。

## 方法详解

### 整体框架

FunctionalMap 分两阶段：
1. **功能嵌入学习**：Siamese 编码器 + 对比学习，从 LFP 信号中学习 32 维被试无关的功能身份嵌入
2. **功能 Transformer**：利用功能嵌入作为 token 坐标，对可变数量通道建模区间关系，执行掩蔽区域重建

### 关键设计

**功能嵌入网络**：轻量级 CNN 编码器 $f_\theta: \mathbb{R}^T \to \mathbb{R}^d$（$d=32$），将 10 秒 LFP 片段映射到嵌入空间。

**配对 Siamese 对比（PSC）**：同区域配对拉近，不同区域推开至距离 $m=0.5$ 以上。

**改进有监督对比（MSC）**：多正样本 InfoNCE + 类内方差惩罚 $\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_{\text{var}} \mathcal{L}_{\text{var}}$，在超球面上操作，强调角度分离。MSC 在跨通道泛化上更优。

**功能 Transformer**：
- 任务：遮蔽某脑区所有通道，从其他区域预测
- 1D 卷积 tokenizer 将源通道转为时间 patch 特征，与功能嵌入融合
- 标准 pre-LN encoder-decoder，无被试 ID
- 损失：$\mathcal{L} = \text{MSE}(\hat{\mathbf{Y}}, \mathbf{Y}) + \lambda(1 - \rho(\hat{\mathbf{Y}}, \mathbf{Y}))$

### 损失函数 / 训练策略
- 功能嵌入：对比损失（PSC 或 MSC），10 秒 LFP 片段
- Transformer：MSE + Pearson 相关损失，跨 11 名被试联合训练
- 单一共享模型，无被试特异性微调

## 实验关键数据

### 主实验

| 设置 | 留出时间段准确率 | 留出通道准确率 |
|------|---------------|-------------|
| 单被试 | 75.78% ± 17.90% | 45.79% ± 18.44% |
| **多被试联合** | **80.71% ± 11.41%** | **49.18% ± 12.11%** |

### 消融实验

| 坐标系 | Pearson 相关 r | 显著性 |
|--------|---------------|--------|
| MNI 坐标 | 基线 | — |
| Functional-1 (PSC) | 正趋势 | 不显著 |
| **Functional-2 (MSC)** | **显著优于 MNI** | $p \approx 0.002$ |

### 关键发现
- 功能嵌入成功跨被试聚类脑区，零样本迁移到未见通道
- 功能坐标显著优于解剖坐标（MSC embedding）
- MNI 的失败案例：共享相同 MNI 坐标的通道被功能嵌入正确区分

## 亮点与洞察
- "功能作为坐标系"替代解剖坐标是有力新范式
- 对比学习几何差异有意义：PSC 的紧凑质心 vs MSC 的角度分离
- 掩蔽区域重建不需要行为标签，纯利用区间神经回路信息

## 局限与展望
- 依赖区域标签进行对比训练
- 仅限基底节-丘脑回路，未验证皮层 ECoG
- 仅验证信号重建，未测试行为解码等下游任务

## 相关工作与启发
- **vs MNI坐标方法**: MNI 假设解剖=功能，本文证明功能坐标更可靠
- **vs PopT**: PopT 聚合冻结单通道嵌入，本文学习可迁移的功能坐标

## 评分
- 新颖性: ⭐⭐⭐⭐ 功能坐标系替代解剖坐标是有力的新范式
- 实验充分度: ⭐⭐⭐⭐ 模拟验证 + 真实数据 + 多层消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富
- 价值: ⭐⭐⭐⭐ 对临床神经科学和 BCI 有直接价值

<!-- RELATED:START -->

## 相关论文

- [GRADIEND: Feature Learning within Neural Networks Exemplified through Biases](gradiend_feature_learning_within_neural_networks_exemplified_through_biases.md)
- [Scalable Multi-Task Low-Rank Model Adaptation](scalable_multi-task_low-rank_model_adaptation.md)
- [Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems](stop_wasting_your_tokens_towards_efficient_runtime_multi-agent_systems.md)
- [Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)
- [BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)

<!-- RELATED:END -->
