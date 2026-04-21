---
title: >-
  [论文解读] A Variational Approach for Mitigating Entity Bias in Relation Extraction
description: >-
  [ACL 2025][NLP理解][实体偏差] 提出基于变分信息瓶颈（VIB）的实体去偏方法，将实体token映射为高斯分布以选择性压缩实体特定信息、保留上下文语义，在通用/金融/生物医学三个领域的关系抽取数据集上均取得SOTA，特别是在OOD场景下BioRED提升5.3个F1点。
tags:
  - ACL 2025
  - NLP理解
  - 实体偏差
  - 变分信息瓶颈
  - 关系抽取
  - 去偏
  - PLM
---

# A Variational Approach for Mitigating Entity Bias in Relation Extraction

**会议**: ACL 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: NLP理解 / 关系抽取  
**关键词**: 实体偏差, 变分信息瓶颈, 关系抽取, 去偏, PLM

## 一句话总结

提出基于变分信息瓶颈（VIB）的实体去偏方法，将实体token映射为高斯分布以选择性压缩实体特定信息、保留上下文语义，在通用/金融/生物医学三个领域的关系抽取数据集上均取得SOTA，特别是在OOD场景下BioRED提升5.3个F1点。

## 研究背景与动机

**领域现状**：关系抽取（RE）是信息抽取的核心任务，主流方法基于预训练语言模型（PLM）在标注数据上微调。然而，模型容易过度利用实体本身携带的信息（如"Smith"常出现在"就职于"关系中）来捷径式地预测关系，而非真正理解上下文语义。

**现有痛点**：现有去偏方法存在明显的性能-泛化权衡。实体掩码直接用通用标记替换实体，虽减小域内-域外差距但严重损害域内性能（TACRED上降7.5个F1）。实体替换式数据增强无法系统性解决偏差问题。当前SOTA方法SCM通过构造实体邻居的凸包中心替换实体嵌入，但缺乏理论基础和可解释性。

**核心矛盾**：实体信息并非全部有害——实体类型、角色等信息对关系预测有正向贡献，问题在于实体的身份信息（identity）导致虚假关联。如何在保留有用信息的同时压缩有害偏差，是一个信息论层面的权衡问题。

**本文目标** 设计一种有原则的、可解释的实体去偏框架，能精准控制实体信息的保留与压缩程度，同时在域内和域外场景均保持高性能。

**切入角度**：变分信息瓶颈（VIB）天然适合这一需求——它通过最大化输出与输入的互信息同时最小化中间表示与输入的互信息，实现信息压缩。将VIB仅作用于实体token，非实体token保持不变。

**核心 idea**：用VIB将实体token嵌入映射为高斯分布进行随机化压缩，方差自然反映模型对实体vs上下文的依赖程度，兼具去偏效果和可解释性。

## 方法详解

### 整体框架

1. 输入文本经PLM编码器获得初始词嵌入 $X$
2. 使用二元实体掩码 $M$ 识别实体token位置
3. 对实体token应用VIB：通过SLP将实体嵌入映射为高斯分布 $\mathcal{N}(\mu, \sigma)$
4. 使用重参数化技巧采样 $z = \mu + \epsilon \cdot \sigma$
5. 通过混合因子 $\beta$ 将原始嵌入与压缩表示融合
6. 送入PLM编码器获取上下文化表示，拼接主/客体标签表示后分类

### 关键设计

1. **选择性VIB压缩**:

    - 功能：仅对实体token施加信息瓶颈压缩，精准限制实体信息流入
    - 核心思路：使用二元实体掩码 $M$ 识别实体位置，通过单层感知机将实体嵌入映射为高斯分布 $\mathcal{N}(\mu, \sigma)$，使用重参数化技巧采样 $z = \mu + \epsilon \cdot \sigma$。非实体token完全保留原始嵌入
    - 设计动机：全局压缩会损害上下文理解能力，选择性压缩确保只有实体特定的身份信息被噪声化，而上下文语义信号不受影响

2. **嵌入混合与自适应损失**:

    - 功能：通过混合因子控制压缩程度，通过自适应权重平衡分类与正则化损失
    - 核心思路：混合公式 $x' = x \cdot (1-M) + x \cdot M \cdot (1-\beta) + z \cdot M \cdot \beta$，$\beta$ 控制原始嵌入与压缩表示的配比。总损失 $\mathcal{L} = L_{CE} + \alpha \cdot L_{VIB}$，其中 $\alpha$ 自适应设为CE与VIB损失的比值，确保两项贡献动态平衡
    - 设计动机：$\beta$ 提供了域内/域外性能的连续调节杆；自适应 $\alpha$ 避免了手动调参，使训练过程更稳健

3. **方差可解释性机制**:

    - 功能：提供模型对实体依赖程度的定量可解释指示器
    - 核心思路：$\sigma^2$ 的大小直接反映模型判断——低方差表示模型认为该实体本身携带强关系信号（高依赖），高方差表示模型更多利用上下文推理。通过统计不同方差区间的样本分布和对应关系类型，验证可解释性
    - 设计动机：现有去偏方法（如SCM）是黑盒操作，无法告诉用户模型为什么做出特定预测。方差作为天然副产物，零额外成本地提供了可解释性

## 实验关键数据

### 主实验

| 方法 | TACRED ID | TACRED OOD | REFinD ID | REFinD OOD | BioRED ID | BioRED OOD |
|------|-----------|------------|-----------|------------|-----------|------------|
| LUKE-Large | 71.1 | 63.8 | 75.0 | 73.4 | 56.9 | 51.8 |
| + Entity Mask | 63.6 | 61.7 | 71.4 | 71.4 | 53.2 | 40.2 |
| + SCM | 68.6 | 64.8 | 74.5 | 73.8 | 58.3 | 53.4 |
| + **VIB (ours)** | **70.4** | **66.5** | **75.4** | **74.8** | **61.2** | **58.7** |
| RoBERTa-Large | 70.8 | 61.5 | 75.1 | 72.7 | 57.7 | 47.9 |
| + SCM | 70.5 | **67.5** | 74.9 | 73.7 | 57.3 | **52.5** |
| + **VIB (ours)** | **70.7** | 67.2 | **75.4** | **74.4** | **63.0** | 52.5 |

### 消融实验

| 方差区间 | 样本比例 | 主要关系类型 |
|----------|----------|-------------|
| 0.0-0.1 | 4.6% | pers:title (高置信度实体依赖) |
| 0.1-0.2 | 85.8% | 大多数关系的平衡区域 |
| 0.2-0.3 | 9.6% | org:date:formed_on (更多依赖上下文) |
| 0.3-0.4 | 0.1% | 高上下文依赖关系 |

### 关键发现

1. **VIB在LUKE-Large上优势明显**: 在BioRED OOD上比SCM高5.3个F1点，说明VIB更擅长利用实体增强型骨干
2. **域内-域外差距显著缩小**: 实体掩码方法虽缩小差距、但牺牲域内性能；VIB保持域内性能的同时大幅提升OOD
3. **方差分析验证可解释性**: 低方差实体（如人名+职位）说明模型认为该实体本身就携带强关系信号；高方差实体说明模型更靠上下文推理

## 亮点与洞察

- **VIB框架的创新应用**：将信息论中的变分信息瓶颈方法创新性地应用于实体去偏场景，理论基础扎实，提供了一个有原则的去偏框架
- **方差作为免费可解释性工具**：无需额外设计，$\sigma^2$ 天然反映模型对实体vs上下文的依赖程度，可定量分析每个样本的决策依据
- **跨领域广泛验证**：在通用NLP（TACRED）、金融（REFinD）、生物医学（BioRED）三个差异显著的领域均有效，证明方法的通用性

## 局限与展望

- 仅在PLM（RoBERTa/LUKE）上验证，未扩展到LLM和生成式关系抽取框架
- 仅英文实验，未验证多语言场景下实体偏差的模式差异
- VIB增加了额外的推理时间（需要SLP计算均值和方差）
- 可探索将VIB与对比学习结合，进一步增强实体无关的上下文表示学习

## 相关工作与启发

- **vs 实体掩码 (Zhang et al., 2017)**：他们完全移除实体信息，本文选择性压缩保留有用部分。实体掩码在TACRED上域内降7.5个F1，VIB仅降0.7
- **vs SCM (Wang et al., 2023a)**：SCM用邻居实体凸包中心替换原始嵌入，是黑盒操作。VIB提供理论保证和方差可解释性，且在BioRED OOD上高出5.3个F1

## 评分
- 新颖性: ⭐⭐⭐⭐ VIB应用于实体去偏是新角度，理论动机清晰
- 实验充分度: ⭐⭐⭐⭐ 三领域六设置的全面验证加方差可解释性分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，理论与实验结合紧密
- 价值: ⭐⭐⭐ 方法有效但应用场景较窄，缺少在LLM上的验证

<!-- RELATED:START -->

## 相关论文

- [Towards a More Generalized Approach in Open Relation Extraction](generalized_open_relation_extract.md)
- [Generating Diverse Training Samples for Relation Extraction with Large Language Models](generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)
- [Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](analyzing_political_bias_in_llms_via_target-oriented_sentiment_classification.md)
- [Dynamic Order Template Prediction for Generative Aspect-Based Sentiment Analysis](dot_absa_template.md)
- [Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](embqa_embedding_odqa.md)

<!-- RELATED:END -->
