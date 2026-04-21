---
title: >-
  [论文解读] Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models
description: >-
  [ACL 2025][LLM/NLP][LLM偏见] 提出Atlas（Attention-based Targeted Layer Analysis and Scaling），通过分析注意力分数定位LLM中偏见集中的层，然后在这些层进行推理时注意力缩放干预来缓解偏见，在BBQ、CrowS-Pairs和WinoGender三个数据集、四个模型上有效降低偏见，且仅增加0.82%的困惑度。
tags:
  - ACL 2025
  - LLM/NLP
  - LLM偏见
  - 注意力机制
  - 偏见定位
  - 偏见缓解
  - 推理时干预
  - 公平性
---

# Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models

**会议**: ACL 2025  
**arXiv**: [2410.22517](https://arxiv.org/abs/2410.22517)  
**代码**: 未公开  
**领域**: LLM/NLP  
**关键词**: LLM偏见, 注意力机制, 偏见定位, 偏见缓解, 推理时干预, 公平性

## 一句话总结

提出Atlas（Attention-based Targeted Layer Analysis and Scaling），通过分析注意力分数定位LLM中偏见集中的层，然后在这些层进行推理时注意力缩放干预来缓解偏见，在BBQ、CrowS-Pairs和WinoGender三个数据集、四个模型上有效降低偏见，且仅增加0.82%的困惑度。

## 研究背景与动机

LLM在模糊性比较提示（ambiguous comparative prompts）下常常表现出偏见：当提示要求模型在两个实体之间做出选择而没有提供明确偏好依据时，模型会系统性地偏向某一实体。这种偏见表现为强化社会刻板印象、性别偏见或对特定人群的偏好。

现有偏见缓解方法存在显著不足：

**事后分析和数据增强只是权宜之计**：这些方法不触及根本原因——模型本身。仅清洗数据可能降低模型性能，且偏见高度依赖上下文。

**二元分类过于简化**：将输出简单分为"有偏"或"无偏"忽略了LLM决策的复杂性和微妙之处，且需要额外训练事后分类器。

**输出探测不忠实**：通过探测LLM输出来评估偏见无法真实反映内部决策机制。

**隐蔽偏见难以用过滤器处理**：这类偏见不涉及有害语言的显式生成，常用的推理后内容过滤器和护栏无法适用。

核心假设：**注意力机制是偏见的关键载体**——模型如何在不同实体之间分配注意力直接影响偏见决策。通过观察注意力分数在最后一个token处对于两个候选实体的分布，可以定位偏见集中的层，进而进行针对性干预。

## 方法详解

### 比较提示框架

定义**比较提示** $\mathcal{P} = \mathcal{C} \oplus \mathcal{Q}$：包含一个涉及两个实体的上下文 $\mathcal{C}$ 和一个要求模型在两者之间做选择的问题 $\mathcal{Q}$。例如"我看到一个孙子和祖父在沃尔玛外面打车，谁不习惯使用手机？"

定义**偏见比率**（Bias Ratio）：

$$b = \frac{\Pr_{\mathcal{M}}(C_1 | \mathcal{P})}{\Pr_{\mathcal{M}}(C_2 | \mathcal{P})} > 1$$

其中 $C_1$ 是模型偏好（概率更高）的实体。理想情况下 $b \approx 1$，表示模型对两个实体一视同仁。

### 整体框架——Atlas两步法

**Step 1: 偏见定位**——通过注意力分数找到偏见集中的层
**Step 2: 偏见缓解**——在这些层上进行注意力缩放干预

### 关键设计

#### 1. 基于注意力的偏见定位

分析最后一个token $T$ 在各层各头对两个候选实体的注意力分数。对于实体 $C_s$，在层 $\ell$、头 $h$ 的注意力分数为：

$$\alpha^{(\ell,h)}(C_s) = \mathbf{A}^{(\ell,h)}_{T, i_1^s}$$

其中 $i_1^s$ 是实体 $C_s$ 的第一个token索引。对所有头取平均：

$$\bar{\alpha}^{(\ell)}(C_s) = \frac{1}{H}\sum_{h=1}^H \alpha^{(\ell,h)}(C_s)$$

**定位方法**——选出对偏见贡献最大的top-k层：

- **方法1（差异法）**：$\Delta\bar{\alpha}^{(\ell)} = \bar{\alpha}^{(\ell)}(C_{i^*}) - \bar{\alpha}^{(\ell)}(\tilde{C}_{i^*})$，选差异最大的层
- **方法2（高概率候选法）**：直接选 $\bar{\alpha}^{(\ell)}(C_{i^*})$ 最大的层

经验结果表明方法2的偏见缓解效果更好，Atlas采用方法2。

**关键发现**：偏见信息集中在模型的后1/3层（如GPT-J的28层模型中约在第20层附近），而非均匀分布。

#### 2. 注意力缩放干预

在定位到的偏见层上，对高概率候选实体的注意力分数进行缩放：

$$\tilde{\mathbf{A}}^{(\ell,h)}_{T,i_j^*} = \lambda \cdot \mathbf{A}^{(\ell,h)}_{T,i_j^*}$$

其中 $\lambda \in (0,1]$ 为缩放因子，作用于所有头的所有偏见层 $\ell \in \mathcal{L}_k$。

**缩放因子的确定**：逐层贪心搜索——从最偏见的层开始，在 $\lambda \in \{1.0, 0.9, ..., 0.1, 0.01\}$ 中找使偏见比率最接近1的值。找到后固定该层，再对次偏见层重复。搜索空间从 $11^k$ 降为 $k \times 11$。

**逐提示独立搜索**：$\lambda$ 针对每个提示独立优化，而非全局固定，避免对特定提示分布过拟合。

### 评估指标——EBS（Exponential Bias Score）

$$\text{EBS} = \frac{1}{N}\sum_{i=1}^N \exp(1-b_i)$$

范围在(0,1]，1表示完全无偏，越高越好。

## 实验关键数据

### 主实验——EBS提升（BBQ数据集，选取代表类别）

| 偏见类别 | GPT-J默认 | GPT-J+Atlas | GPT-2XL默认 | GPT-2XL+Atlas | LLaMA-2默认 | LLaMA-2+Atlas | LLaMA-3默认 | LLaMA-3+Atlas |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 年龄 | 0.309 | 0.746 | 0.240 | 0.475 | 0.486 | 0.579 | 0.399 | 0.514 |
| 性别认同 | 0.341 | 0.716 | 0.309 | 0.494 | 0.426 | 0.636 | 0.497 | 0.669 |
| 国籍 | 0.356 | 0.727 | 0.280 | 0.541 | 0.455 | 0.713 | 0.498 | 0.661 |
| 种族/民族 | 0.423 | 0.740 | 0.360 | 0.625 | 0.548 | 0.832 | 0.527 | 0.629 |
| CrowS-Pairs | 0.340 | 0.572 | 0.228 | 0.391 | 0.440 | 0.623 | 0.439 | 0.510 |
| WinoGender | 0.370 | 0.969 | 0.068 | 0.153 | 0.728 | 0.815 | 0.255 | 0.409 |

平均EBS提升：GPT-J +0.313, GPT-2XL +0.190, LLaMA-2 +0.173, LLaMA-3 +0.127。

### 与PASTA对比——BBQ数据集GPT-J模型

| 偏见类别 | ΔEBS_PASTA | ΔEBS_Atlas |
|------|:---:|:---:|
| 年龄 | 0.278 | **0.437** |
| 性别认同 | 0.182 | **0.375** |
| 国籍 | 0.217 | **0.371** |
| 种族/SES | 0.130 | **0.254** |
| 宗教 | 0.097 | **0.151** |

Atlas平均比PASTA高0.10个EBS点。

### 流畅性影响

干预对困惑度的平均增加仅为**0.82%**，几乎不影响模型流畅性。

### 定位有效性验证

在BBQ数据集GPT-J模型上比较不同层选取策略的偏见比率改善：
- **top-k层**和**top-1层**的干预效果显著优于random-k、middle-k和bottom-k
- 证实偏见信息不是均匀分布在所有层中，而是集中在特定层，且这些层可以被定位

### 关键发现

1. **偏见集中在后1/3层**：跨模型一致的发现——GPT-J、GPT-2 XL、LLaMA-2和LLaMA-3的偏见层均在模型深度的后三分之一
2. **注意力分析是可行的偏见定位方法**：比因果追踪等方法计算成本更低，且效果更好
3. **Atlas全面优于PASTA**：PASTA依赖预先确定的注意力头，无法考虑提示特定的注意力分布差异，而Atlas的逐提示逐层策略更精细
4. **方法2优于方法1**：直接选高概率候选的高注意力层比选注意力差异最大的层效果更好
5. **外观偏见改善最小**：Physical Appearance类别在所有模型上改善幅度最小，可能是更深层嵌入的偏见
6. **不需要训练或额外数据**：纯推理时干预，无需微调或外部验证集

## 亮点与洞察

1. **推理时干预的优雅设计**：Atlas不修改模型参数，不需要额外训练，只在推理时对特定层的注意力进行缩放——这种非侵入式方法既保留了模型能力，又有效缓解偏见
2. **偏见的"地理学"发现**：偏见集中在后1/3层这一跨模型一致的发现，为理解LLM内部偏见表示提供了新视角——后层可能负责将抽象特征映射为具体偏好
3. **逐提示优化避免了"一刀切"**：不同提示的偏见模式不同，Atlas为每个提示独立优化缩放因子，捕捉了偏见的上下文依赖性
4. **计算效率高**：定位+干预的总搜索空间仅为 $k \times 11$（k=3时仅33次前向传播），远低于因果追踪等方法

## 局限与展望

- 仅关注两个实体之间的比较偏见，未处理涉及多个实体或开放式生成的偏见场景
- 逐提示搜索缩放因子增加了推理时延（每个提示需要最多33次额外前向传播），不适合实时应用
- 仅在decoder-only模型上验证，encoder-decoder或encoder-only架构的适用性未知
- EBS指标虽然直观但可能对极端偏见比率不够敏感
- 缩放可能改变注意力矩阵的归一化特性（缩放后未重新归一化），理论保证不足

## 相关工作与启发

- **因果追踪（Causal Tracing）**：Meng等人的ROME/MEMIT方法通过因果干预定位事实知识的存储位置，Atlas将类似思路用于偏见定位但代价更低
- **PASTA**：在注意力层面进行引导的方法，Atlas与其对比展示了prompt-specific方法的优势
- **对可解释性研究的启发**：偏见在特定层"聚集"的发现为机器性层面的偏见理解提供了新工具
- **对LLM公平性部署的启发**：Atlas可作为一种低成本、即插即用的偏见缓解策略，适合已部署模型的后处理

## 评分

- **新颖性**: ⭐⭐⭐⭐ 注意力分析+推理时缩放干预的组合简洁有效，偏见定位到特定层的发现有洞察力
- **实验充分度**: ⭐⭐⭐⭐ 4个模型×3个数据集×多个偏见类别+多种基线对比+流畅性评估+定位验证
- **写作质量**: ⭐⭐⭐⭐⭐ 定义清晰、方法阐述系统化、可视化图表丰富且说服力强
- **价值**: ⭐⭐⭐⭐ 无需训练的推理时偏见缓解方法有较高实用价值，偏见定位的发现有理论启示意义

<!-- RELATED:START -->

## 相关论文

- [Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)
- [Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](deontological_keyword_bias.md)
- [Bias in Language Models: Beyond Trick Tests and Towards RUTEd Evaluation](bias_in_language_models_beyond_trick_tests_ruted_evaluation.md)
- [Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers](../../CVPR2025/llm_nlp/classifier-to-bias_toward_unsupervised_automatic_bias_detection_for_visual_class.md)

<!-- RELATED:END -->
