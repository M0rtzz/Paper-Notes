---
title: >-
  [论文解读] mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules
description: >-
  [ICLR 2026][医学图像][化学语言模型] 提出 mCLM（模块化化学语言模型），通过将分子表示为可合成构建模块的序列，使 LLM 能生成同时满足药理功能和自动化合成可行性的分子，在 430 种 FDA 批准药物上显著改善了药代动力学和毒性性质。
tags:
  - ICLR 2026
  - 医学图像
  - 化学语言模型
  - 分子优化
  - 自动化合成
  - 模块化设计
  - 药物发现
---

# mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules

**会议**: ICLR 2026  
**arXiv**: [2505.12565](https://arxiv.org/abs/2505.12565)  
**代码**: 有（论文中提供）  
**领域**: 医学 / 分子生成  
**关键词**: 化学语言模型, 分子优化, 自动化合成, 模块化设计, 药物发现

## 一句话总结
提出 mCLM（模块化化学语言模型），通过将分子表示为可合成构建模块的序列，使 LLM 能生成同时满足药理功能和自动化合成可行性的分子，在 430 种 FDA 批准药物上显著改善了药代动力学和毒性性质。

## 研究背景与动机

**领域现状**：LLM 已展现出理解化学知识的能力，但在生成功能性小分子方面仍然有限——生成的分子通常不兼容自动化合成方法。

**现有痛点**：现有分子生成方法使用原子级或片段级表示（如 SMILES），生成的分子虽然可能满足药理目标，但几乎无法通过自动化合成流程制造。这导致计算预测到实验验证之间存在巨大鸿沟。

**核心矛盾**：分子的"功能"（药效、毒性等）和"可制造性"（合成路径已知、构建模块可用）是两个独立优化目标，传统方法只关注前者。

**本文目标** 让 LLM 学习一种新的分子语言，使其生成的分子同时具有优化的化学功能和保证的合成可行性。

**切入角度**：将分子表示为来自预定义构建模块库的组合序列，每个模块对应已知的可自动合成的化学片段。

**核心 idea**：用模块化分子语言替代传统 SMILES，让 LLM 在受限的合成空间中搜索功能最优分子。

## 方法详解

### 整体框架
mCLM 将目标分子分解为构建模块序列，每个模块来自与自动化合成兼容的库。输入为初始分子的模块分解，输出为优化后的模块序列。训练使用配对数据（原始→优化）进行 seq2seq 风格的微调。

### 关键设计

1. **模块化分子表示 (Modular Molecular Representation)**:

    - 做什么：将分子从原子级 SMILES 转化为构建模块级序列
    - 核心思路：定义构建模块库 $\mathcal{B} = \{b_1, ..., b_N\}$，每个 $b_i$ 是一个可自动合成的化学片段。分子 $M$ 被分解为 $M = b_{i_1} \oplus b_{i_2} \oplus ... \oplus b_{i_k}$，其中 $\oplus$ 表示化学键连接。使用逆合成分析来确定分解方式
    - 设计动机：模块级表示天然保证合成可行性——只要每个模块可合成、连接规则有效，整个分子就可合成

2. **功能引导的编辑训练**:

    - 做什么：训练模型学习如何编辑分子模块来改善药理性质
    - 核心思路：构建配对训练数据：对 FDA 药物进行模块分解后，用性质预测器评估不同模块替换方案的药理改善。选择保持分子骨架相似但改善目标性质（如 AMES 毒性↓、BBBP 透过率↑、HIA 吸收率↑）的替换作为正例
    - 设计动机：直接从头生成太难，编辑式学习保留原分子的有效结构同时定向优化

3. **超越训练模块库的泛化**:

    - 做什么：使模型能使用训练时未见过的新构建模块
    - 核心思路：在测试时将模块库扩展到分布外 (OOD) 的构建模块集合。模型通过学习模块间的语义关系（化学性质相似性）来泛化
    - 设计动机：固定模块库限制了可搜索的化学空间，泛化到新模块大幅扩展实用性

### 损失函数 / 训练策略
标准 seq2seq 交叉熵损失，基于 LLaMA 架构微调。训练数据通过药理性质预测器筛选。

## 实验关键数据

### 主实验

| 模型 | AMES ↓ | BBBP ↑ | CYP3A4 ↓ | DILI ↓ | HIA ↑ | PGP ↓ | 平均改善 |
|------|--------|--------|----------|--------|-------|-------|---------|
| FDA 药物 (原始) | 47.8 | 61.4 | - | - | - | - | - |
| mCLM | 改善 | 改善 | 改善 | 改善 | 改善 | 改善 | 显著 |
| MoleculeSTM | - | - | - | - | - | - | 远不如 mCLM |

### 消融实验

| 配置 | 平均改善 | 说明 |
|------|---------|------|
| 完整 mCLM | 最优 | 全模块化+功能引导 |
| OOD 模块库 (122药物) | 仍有效 | 泛化到未见模块 |
| SMILES 基线 | 不保证合成 | 功能可能更优但不可制造 |
| 无编辑训练 | 随机替换 | 无定向优化 |

### 关键发现
- mCLM 在 430 种 FDA 药物上在所有 6 个药代动力学/毒性指标上都有改善
- 在 122 种 OOD 药物上使用仅兼容自动合成的模块仍然有效，证明泛化能力
- 大幅超越 MoleculeSTM 等文本分子编辑基线
- 模块化表示使合成路径自动可得，消除了"计算→实验"鸿沟

## 亮点与洞察
- **合成可行性作为硬约束**：将可制造性从"事后检查"变为"架构保证"，这是药物发现管线的关键实用需求
- **模块化语言让化学搜索变为 token 序列优化**：将连续化学空间的搜索转化为离散模块序列的编辑，天然适合 LLM

## 局限与展望
- 模块库的覆盖范围限制了可到达的化学空间
- 性质预测器的准确性直接影响训练数据质量
- 仅优化药代动力学/毒性，未涉及药效（结合亲和力等）
- 合成路径的实际执行成功率未验证

## 相关工作与启发
- **vs MoleculeSTM**: 传统文本-分子方法在 SMILES 空间操作，不保证合成可行性
- **vs RetroGPT**: 逆合成规划关注"如何合成给定分子"，mCLM 关注"在可合成空间中找最优分子"，是互补方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 模块化分子表示+LLM 结合的思路新颖
- 实验充分度: ⭐⭐⭐ 多性质评估但缺少实验验证
- 写作质量: ⭐⭐⭐⭐ 跨学科但讲解清晰
- 价值: ⭐⭐⭐⭐⭐ 对 AI 辅助药物发现有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [Reverse Distillation: Consistently Scaling Protein Language Model Representations](reverse_distillation_consistently_scaling_protein_language_model_representations.md)
- [How to Make the Most of Your Masked Language Model for Protein Engineering](how_to_make_the_most_of_your_masked_language_model_for_protein_engineering.md)
- [Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](../../NeurIPS2025/medical_imaging/mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)
- [AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)

<!-- RELATED:END -->
