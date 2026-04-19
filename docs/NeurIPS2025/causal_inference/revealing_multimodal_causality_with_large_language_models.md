---
title: >-
  [论文解读] Revealing Multimodal Causality with Large Language Models
description: >-
  [NeurIPS 2025][Causal Discovery] 提出 MLLM-CD，首个面向多模态非结构化数据的因果发现框架，通过对比因子发现识别跨模态因果变量，结合统计因果结构推断，并利用 MLLM 的世界知识生成多模态反事实样本来迭代消除结构歧义，在合成和真实数据集上均显著优于现有方法。
tags:
  - NeurIPS 2025
  - Causal Discovery
  - 多模态
  - Counterfactual Reasoning
  - Contrastive Factor Discovery
  - Unstructured Data
---

# Revealing Multimodal Causality with Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.17784](https://arxiv.org/abs/2509.17784)  
**代码**: https://github.com/JinLi-i/MLLM-CD  
**领域**: 因果推理  
**关键词**: Causal Discovery, Multimodal LLM, Counterfactual Reasoning, Contrastive Factor Discovery, Unstructured Data  

## 一句话总结

提出 MLLM-CD，首个面向多模态非结构化数据的因果发现框架，通过对比因子发现识别跨模态因果变量，结合统计因果结构推断，并利用 MLLM 的世界知识生成多模态反事实样本来迭代消除结构歧义，在合成和真实数据集上均显著优于现有方法。

## 研究背景与动机

**领域现状**：因果发现（Causal Discovery, CD）旨在从观测数据中推断因果结构图（DAG）。传统 CD 方法（PC、FCI、GES 等）依赖预定义的结构化变量，无法处理非结构化数据。COAT 是目前唯一将 LLM 引入非结构化数据 CD 的工作，但仅限于单模态文本。

**核心痛点**：
   - 现实场景中数据日益多模态化（文本+图像+音频），缺乏能处理图文混合数据的 CD 方法
   - 简单将 COAT 等单模态方法适配到多模态效果很差——只能发现少量因子，且推断的因果边缺乏方向信息
   - 观测数据固有的马尔可夫等价类问题在变量增多时更加严重

**两大挑战**：
   - **CH1（模态交互探索）**：因果变量可能隐含在不同模态的交互中，需要主动探索模态内和模态间交互才能发现（例如"较小的苹果对应较低的评分"揭示了 size 这个变量）
   - **CH2（结构歧义消解）**：多个不同的因果结构可能产生相同的统计依赖关系，纯靠观测数据无法区分

**核心思路**：将 MLLM 作为多模态理解和推理引擎，通过对比样本发现隐含因子，通过反事实推理引入超越观测数据的额外证据来消除歧义。

## 方法详解

### 问题定义

给定多模态数据集 $\mathcal{D} = \{\mathbf{X}_1, \ldots, \mathbf{X}_n\}$，每个样本包含多个模态（文本、图像等），需要完成两个任务：
1. **因子识别**：从非结构化数据中发现潜在因果变量 $\mathbf{V}$ 并标注值，将非结构化数据转为结构化数据 $\mathcal{D}_S$
2. **结构发现**：推断变量间的因果关系，构建因果图 $\mathcal{G} = (\mathbf{V} \cup \{Y\}, \mathbf{E})$

### 整体框架：三模块迭代

MLLM-CD 包含三个核心模块，以迭代方式运行：

**模块一：对比因子发现 (Contrastive Factor Discovery, CFD)**

核心思想是通过展示差异最大的样本对，迫使 MLLM 关注不同维度的变化，从而发现隐含因子。

- **语义表征提取**：用 CLIP 等预训练模型对每个模态提取语义 embedding $\mathbf{e}_{ki} = f_i(\mathbf{x}_{ki})$
- **模态内对比探索**：在每个模态内选择余弦距离最大的 top-K 样本对 $\mathcal{P}_i$，提交给 MLLM 分析"这两个样本在哪些维度不同"，从而发现该模态内的因果变量
- **模态间对比探索**：选择跨模态语义最不对齐的样本对（综合考虑 embedding 距离和目标变量差异），让 MLLM 分析跨模态的不一致性，揭示跨模态关联的因子
- **因子合并与标注**：用 MLLM 对所有候选因子去重合并，得到最终因子集 $\mathbf{V}^{(t)}$；再为每个样本标注各因子的值，形成结构化数据 $\mathcal{D}_S^{(t)}$

设计动机：对比样本对类似"最大化信息增益"——语义差异最大的一对样本能暴露最多的潜在变量维度。

**模块二：因果结构发现 (Causal Structure Discovery)**

- 在结构化数据 $\mathcal{D}_S^{(t)}$ 上运行统计 CD 算法（本文采用 FCI 算法）推断因果图 $\mathcal{G}^{(t)}$
- 选择 FCI 是因为它对潜在混杂变量鲁棒，适合真实场景中可能存在未观测变量的情况
- 可替换为其他 CD 算法（PC、GES、NOTEARS 等）

**模块三：多模态反事实推理 (Multimodal Counterfactual Reasoning, MCR)**

针对因果图中的不确定边，利用 MLLM 的世界知识生成反事实样本作为额外因果证据：

- **反事实生成**：对不确定因子 $V_a$ 做"what if"干预，MLLM 预测干预后其他因子如何变化，并生成对应的多模态反事实样本（文本由 MLLM 直接修改，图像通过描述+图像生成模型 Φ 生成）
- **语义合理性验证**：确保反事实样本与原始样本的平均 embedding 相似度 $\geq \tau_{sem}$，过滤掉语义不合理的生成
- **因果一致性验证**：检查干预变量的因果非后代节点中发生变化的比例 $\leq \tau_{causal}$，确保反事实符合当前因果图结构
- **迭代更新**：通过双重验证的反事实样本加入数据集 $\mathcal{D}^{(t+1)} = \mathcal{D}^{(t)} \cup \mathcal{D}_{CF}^{(t)}$，重新进入下一轮因子发现和结构推断

### 训练策略

- 无需神经网络训练——纯推理框架，基于 MLLM 的 prompt 调用
- 支持 GPT-4o、Gemini 2.0、LLaMA 4 Maverick、Grok-2v 等多种 MLLM
- 迭代 3-5 轮通常收敛

## 实验关键数据

### 数据集

| 数据集 | 类型 | 样本数 | 因子数 | 模态 |
|--------|------|--------|--------|------|
| MAG (Multimodal Apple Gastronome) | 合成 | 200 | 9 (3视觉+5文本+1目标) | 文本+图像 |
| Lung Cancer | 真实 (MedPix) | 60 | 5 (2文本+1视觉+1目标) | 临床文本+CT 图像 |

### 主实验 — MAG 数据集（四种 MLLM 平均值）

| 方法 | 因子 NF↑ | 结构 AF↑ | ESHD↓ |
|------|---------|---------|-------|
| META (零样本) | 0.52 | 0.46 | 21.25 |
| Pairwise | - | 0.40 | 34.08 |
| Triplet | - | 0.39 | 39.58 |
| COAT | 0.53 | 0.29 | 16.42 |
| **MLLM-CD** | **0.89** | **0.57** | **13.42** |

### 主实验 — Lung Cancer 数据集（四种 MLLM 平均值）

| 方法 | 因子 NF↑ | 结构 AF↑ | ESHD↓ |
|------|---------|---------|-------|
| META | 0.45 | 0.59 | 19.25 |
| COAT | 0.52 | 0.32 | 9.42 |
| **MLLM-CD** | **0.90** | **0.68** | **5.33** |

### 消融实验（Gemini 2.0）

| 变体 | MAG NF / AF / ESHD | Lung NF / AF / ESHD |
|------|---------------------|---------------------|
| 无 CFD + 无 MCR | 0.54 / 0.41 / 16.33 | 0.55 / 0.13 / 9.67 |
| 无 CFD | 0.73 / 0.47 / 15.00 | 0.62 / 0.36 / 8.00 |
| 无 MCR | 0.81 / 0.52 / 15.67 | 0.94 / 0.38 / 5.33 |
| **完整 MLLM-CD** | **0.87 / 0.60 / 14.00** | **0.97 / 0.87 / 4.67** |

### CFD 采样策略消融（Gemini 2.0，MAG 因子发现 NF）

| 策略 | NF |
|------|----|
| 随机采样 | 0.73 |
| 简单配对（目标变量差异最大） | 0.75 |
| 仅模态内对比 | 0.82 |
| 仅模态间对比 | 0.63 |
| **模态内+模态间** | **0.87** |

### 关键发现

- MLLM-CD 因子发现 recall 远超 COAT（0.86 vs 0.39），对比探索有效发现了隐含因子
- 反事实推理对小数据集（Lung Cancer）提升尤为显著：AF 从 0.38 升至 0.87
- 模态内和模态间对比互为补充，缺一不可
- 跨四种 MLLM 一致有效，Gemini 2.0 表现最优

## 亮点与洞察

1. **首个多模态非结构化因果发现框架**：填补了重要空白——现有 CD 方法要么处理结构化数据，要么仅限单模态文本，本文首次将因果发现扩展到真正的多模态非结构化场景。
2. **对比因子发现设计精巧**：利用语义空间中距离最远的样本对迫使 MLLM 关注差异维度，本质上是"最大化信息增益"的思想在因果变量发现中的应用。
3. **MLLM 世界知识作为因果证据源**：创新性地将 MLLM 的知识通过反事实生成转化为额外的因果证据，弥补了纯观测数据的不足。这是统计因果推断与大模型知识的巧妙桥接。
4. **反事实的双重验证**（语义合理性 + 因果一致性）有效缓解了 MLLM 幻觉问题，确保引入的反事实证据可靠。
5. **模块化可扩展**：FCI 可替换为其他 CD 算法，MLLM 可更换为任意多模态大模型，框架具有良好的通用性。

## 局限性与改进方向

- **基准数据集规模小**：MAG 200 样本、Lung Cancer 60 样本，大规模场景下的表现有待验证
- **依赖 MLLM 质量**：在专业领域（如罕见病诊断）MLLM 知识可能不足，反事实质量下降
- **因子标注噪声**：MLLM 为每个样本标注因子值可能引入系统性偏差
- **计算成本高**：多轮迭代中大量 MLLM API 调用，成本较高
- **模态范围受限**：受 MLLM 能力限制，难以处理传感器数据、基因组序列等专业模态
- **改进方向**：与 RAG/知识图谱结合增强领域知识；开发不确定性量化机制；探索开源 MLLM 降成本

## 相关工作对比

| 方法 | 处理多模态 | 因子发现 | 结构消歧 | 非结构化数据 |
|------|-----------|---------|---------|------------|
| PC / FCI / GES | ✗ | ✗ | ✗ | ✗ |
| Pairwise / Triplet prompting | ✗ | ✗ | 部分 | ✗ |
| COAT | ✗ (仅文本) | ✓ | ✗ | ✓ |
| CRL (因果表示学习) | 部分 | 隐变量 | ✗ | ✓ |
| **MLLM-CD** | **✓** | **✓ (可解释)** | **✓ (反事实)** | **✓** |

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个多模态非结构化因果发现框架，问题定义和方法设计均有原创性
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据集、4 种 MLLM、完整消融实验、采样策略对比
- 写作质量: ⭐⭐⭐⭐ 框架清晰，模块化设计好，问题动机论证充分
- 实用价值: ⭐⭐⭐⭐⭐ 将因果发现扩展到多模态非结构化数据有重大实际意义

## 一句话总结

<!-- RELATED:START -->

## 相关论文

- [Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [From Black-box to Causal-box: Towards Building More Interpretable Models](from_black-box_to_causal-box_towards_building_more_interpretable_models.md)
- [Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features](causality-induced_positional_encoding_for_transformer-based_representation_learn.md)

<!-- RELATED:END -->
