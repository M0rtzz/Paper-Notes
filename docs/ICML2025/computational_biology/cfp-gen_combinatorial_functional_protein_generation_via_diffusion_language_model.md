---
title: >-
  [论文解读] CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models
description: >-
  [ICML 2025][计算生物][蛋白质设计] 提出 CFP-Gen——一种大规模扩散语言模型，通过注释引导特征调制（AGFM）和残基级控制编码（RCFE）实现多模态功能约束（功能注释 + 序列基序 + 3D 结构）的组合蛋白质生成，F1 分数比 ESM3 提升 30%。 领域现状：蛋白质语言模型（PLM）在 de nov…
tags:
  - "ICML 2025"
  - "计算生物"
  - "蛋白质设计"
  - "扩散语言模型"
  - "多功能约束"
  - "组合条件生成"
  - "功能注释"
---

# CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models

**会议**: ICML 2025  
**arXiv**: [2505.22869](https://arxiv.org/abs/2505.22869)  
**代码**: 无  
**领域**: 医学图像/蛋白质设计  
**关键词**: 蛋白质设计, 扩散语言模型, 多功能约束, 组合条件生成, 功能注释

## 一句话总结
提出 CFP-Gen——一种大规模扩散语言模型，通过注释引导特征调制（AGFM）和残基级控制编码（RCFE）实现多模态功能约束（功能注释 + 序列基序 + 3D 结构）的组合蛋白质生成，F1 分数比 ESM3 提升 30%。

## 研究背景与动机

**领域现状**：蛋白质语言模型（PLM）在 de novo 蛋白质设计中展现巨大潜力。现有方法大多基于单一条件（仅功能标签、仅结构、或仅序列模式）生成蛋白质。

**现有痛点**：
   - 单条件生成无法同时满足多个功能约束——实际蛋白质工程需要同时满足催化活性（EC number）、结构域（IPR domain）、亚细胞定位（GO term）等多种约束
   - 迭代过滤/多步优化管道在多约束下效率低下且成功率低
   - 当数据有限时这种pipeline更不可行

**核心矛盾**：真实世界的蛋白质工程是多目标优化问题，但现有生成模型只支持单目标。

**本文目标**：在统一模型中同时处理多模态功能约束的蛋白质生成。

**切入角度**：将扩散语言模型与多模态条件编码结合——不同类型的功能约束通过不同的模块注入生成过程。

**核心 idea**：AGFM（注释→特征分布调制）+ RCFE（关键残基→残基级控制）+ 结构编码器（3D 骨架→几何约束），三者组合实现多目标蛋白质设计。

## 方法详解

### 整体框架
CFP-Gen 基于 ESM 架构的扩散语言模型：
1. 输入：加噪的蛋白质序列 + 多种功能条件（GO terms, IPR domains, EC numbers, 序列基序, 3D 结构）
2. 迭代去噪生成目标蛋白质序列
3. 功能条件通过三个专用模块注入每个 ESM block

### 关键设计

1. **注释引导特征调制 (AGFM)**:

    - 功能：将功能注释（GO/IPR/EC）动态调制到加噪序列的特征分布中
    - 核心思路：功能标签编码为 one-hot → 通过可学习仿射变换得到 scale $\gamma$ 和 shift $\beta$ → 对 LayerNorm 后的特征做 $\gamma \cdot x + \beta$（类似 AdaIN/FiLM）
    - 关键优势：多个注释标签可自由组合——训练时联合优化确保功能-序列的严格对齐，推理时支持灵活的注释组合
    - 设计动机：比 classifier-guided 方法更直接——AGFM 嵌入模型内部，不需要额外的分类器

2. **残基控制功能编码 (RCFE)**:

    - 功能：对关键氨基酸残基（序列基序/功能域）提供残基级精确控制
    - 核心思路：用 ESM-like Transformer 编码器处理标记了功能域的部分序列，捕获残基间的表位关系和进化关联，作为条件注入主网络
    - 设计动机：某些残基是功能的关键（如活性位点）——全局的功能注释不足以精确控制这些局部位置

3. **3D 结构条件**:

    - 功能：将参考蛋白质的 3D 骨架原子坐标作为几何约束
    - 核心思路：使用现成的结构编码器（如 ESM-IF）将骨架坐标编码为特征向量→注入生成过程
    - 设计动机：逆折叠（inverse folding）场景——保持结构不变的同时优化序列功能

### 损失函数 / 训练策略
- 离散扩散：在 token 空间添加/去除噪声
- 交叉熵损失：预测去噪后的氨基酸
- 多条件联合训练：随机 mask 部分条件（dropout），增强鲁棒性
- 基于 ESM2-650M 架构扩展

## 实验关键数据

### 主实验
功能蛋白质生成（由领先功能预测器评估）：

| 方法 | F1 (GO-MF) ↑ | F1 (EC) ↑ | 序列新颖性 ↑ |
|------|------------|----------|------------|
| EvoDiff | 0.35 | 0.28 | 高 |
| DPLM | 0.42 | 0.38 | 高 |
| ESM3 | 0.48 | 0.45 | 中 |
| **CFP-Gen** | **0.62** | **0.58** | **高** |

### 逆折叠（Inverse Folding）

| 方法 | AAR (氨基酸恢复率) ↑ | 结构一致性 (TM-Score) |
|------|-------|-----|
| ProteinMPNN | 0.45 | 0.89 |
| DPLM | 0.48 | 0.87 |
| **CFP-Gen** | **0.57** | **0.91** |

### 多功能蛋白质设计

| 约束组合 | 成功率 ↑ | 说明 |
|---------|---------|------|
| EC + GO (双功能酶) | 72.5% | 催化活性+亚细胞定位 |
| EC + IPR (酶+结构域) | 68.3% | 催化活性+特定结构域 |
| EC + 序列基序 | 78.1% | 催化活性+保守位点 |

### 消融实验

| 配置 | F1 (GO-MF) | 说明 |
|------|-----------|------|
| 无 AGFM（无注释调制） | 0.45 | 退化为非条件生成 |
| 无 RCFE（无残基控制） | 0.55 | 缺少局部精确控制 |
| 无结构条件 | 0.58 | 无几何约束 |
| **完整 CFP-Gen** | **0.62** | 三种条件互补 |

### 关键发现
- CFP-Gen 的 F1 比 ESM3 提升 30%——多模态约束的联合优化远优于单约束
- AAR（逆折叠）改进 9%——结构条件有效指导序列设计
- 多功能蛋白质设计成功率 68-78%——首次展示大规模多功能酶设计的可行性
- 条件 dropout 训练使推理时支持任意条件组合——灵活性极高
- 生成的蛋白质序列新颖度高——不是简单地复制训练集中的已知蛋白质

## 亮点与洞察
- **多模态条件的统一处理**是关键贡献——将分散的功能/序列/结构约束整合到单一生成框架
- AGFM 的 FiLM 式调制简洁有效——从图像生成领域借鉴的成熟技术
- RCFE 的残基级控制填补了全局注释和局部精度之间的鸿沟
- 多功能蛋白设计（如同时具有两种催化活性的酶）是蛋白质工程的重要前沿
- 基于 ESM 架构确保了可扩展性和与现有蛋白质 AI 生态的兼容性

## 局限与展望
- 功能验证仅通过计算预测器，实际实验验证（wet lab）缺失
- 生成的蛋白质是否真的可折叠和可表达未验证
- 训练数据偏向已知蛋白质功能，对全新功能的泛化能力未知
- 3D 结构条件依赖外部编码器的质量
- 生成序列的多样性 vs 功能满足之间的权衡未深入探讨

## 相关工作与启发
- **vs ESM3**: 单条件生成的 SOTA，CFP-Gen 通过多条件约束显著提升
- **vs ProteinMPNN**: 专注逆折叠，不支持功能约束；CFP-Gen 在逆折叠上也更优
- **vs RFdiffusion/Chroma**: 结构为中心的设计方法，CFP-Gen 功能为中心更实用于功能导向的蛋白质工程
- **启发**：多模态条件生成的思路可推广到其他分子设计（药物、材料等）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 组合功能蛋白生成是重要的新范式
- 实验充分度: ⭐⭐⭐⭐ 多任务（功能生成/逆折叠/多功能设计），但缺 wet lab
- 写作质量: ⭐⭐⭐⭐ 框架清晰，模块设计合理
- 价值: ⭐⭐⭐⭐⭐ 对蛋白质工程有重大实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Steering Protein Language Models](steering_protein_language_models.md)
- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](../../ACL2025/computational_biology/concept_bottleneck_language_models_for_protein_design.md)
- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](elucidating_the_design_space_of_multimodal_protein_language_models.md)
- [\[ICML 2025\] Training Flexible Models of Genetic Variant Effects from Functional Annotations using Accelerated Linear Algebra](training_flexible_models_of_genetic_variant_effects_from_functional_annotations_.md)
- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](../../NeurIPS2025/computational_biology/fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)

</div>

<!-- RELATED:END -->
