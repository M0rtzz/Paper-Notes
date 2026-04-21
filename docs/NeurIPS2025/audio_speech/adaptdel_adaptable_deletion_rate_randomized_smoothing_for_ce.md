---
title: >-
  [论文解读] AdaptDel: Adaptable Deletion Rate Randomized Smoothing for Certified Robustness
description: >-
  [NeurIPS 2025][语音][certified robustness] 提出 AdaptDel 方法，将随机平滑中用于离散序列的固定删除率扩展为根据输入长度等属性自适应调整的可变删除率，在理论上证明了可变率下认证的 soundness，实验在 NLP 序列分类任务上实现认证区域基数最高 30 个数量级的提升。
tags:
  - NeurIPS 2025
  - 语音
  - certified robustness
  - randomized smoothing
  - edit distance
  - adaptable deletion rate
  - sequence classification
---

# AdaptDel: Adaptable Deletion Rate Randomized Smoothing for Certified Robustness

**会议**: NeurIPS 2025  
**arXiv**: [2511.09316](https://arxiv.org/abs/2511.09316)  
**代码**: 无  
**领域**: AI安全 / 认证鲁棒性  
**关键词**: certified robustness, randomized smoothing, edit distance, adaptable deletion rate, sequence classification

## 一句话总结
提出 AdaptDel 方法，将随机平滑中用于离散序列的固定删除率扩展为根据输入长度等属性自适应调整的可变删除率，在理论上证明了可变率下认证的 soundness，实验在 NLP 序列分类任务上实现认证区域基数最高 30 个数量级的提升。

## 研究背景与动机

**领域现状**：认证鲁棒性（certified robustness）旨在为分类器提供可证明的对抗攻击防御保证。随机平滑（randomized smoothing）是最具扩展性的认证方法之一：通过对输入施加随机扰动并对多次预测进行多数投票来平滑分类器，从而获得可证明的鲁棒性半径。对于连续输入（如图像），常用扰动是高斯噪声；对于离散序列（如文本），常用扰动方式是随机删除 token。

**现有痛点**：现有的离散序列随机平滑方法采用固定的删除率——对所有输入无论长短一视同仁。然而自然语言中的文本长度变化很大：短文本（如 5 个词的句子）如果使用高删除率可能丢失几乎所有信息，导致分类失败；长文本（如 50 个词的段落）如果使用低删除率，每次采样保留的 token 过多，鲁棒性探索不足，认证半径很小。固定删除率无法同时兼顾二者。

**核心矛盾**：删除率的选择存在信息保留与鲁棒性之间的 trade-off，而这个 trade-off 的最优平衡点随输入长度等属性的不同而变化。使用单一固定率必然在部分输入上严重次优。本文提出将删除率设计为输入属性的函数，通过数学推导证明可变删除率下随机平滑认证仍然 sound。

## 方法详解

### 整体框架
AdaptDel 扩展了随机平滑的理论框架以支持可变删除率。给定一个基础分类器和输入序列，首先根据输入属性（如长度 n）计算自适应删除率 p(n)，然后以概率 p(n) 独立删除每个 token 生成多个扰动副本，对所有副本的预测取多数投票得到平滑分类器的输出，最后通过扩展的理论框架计算在编辑距离攻击下的认证半径。

### 关键设计

1. **可变删除率理论框架**:

    - 功能：将固定删除率的随机平滑理论推广到输入依赖的删除率，证明认证的数学正确性
    - 核心思路：证明当删除率随输入变化时，平滑分类器在编辑距离扰动下的认证保证仍然成立。关键挑战在于编辑距离攻击（插入/删除/替换）会改变序列长度从而改变删除率，使分析比固定率更复杂。通过分析攻击序列与原始序列之间删除后分布的关系推导可变率认证界
    - 设计动机：固定率是可变率的特例，推广到可变率后可以为每个输入选择最优的信息-鲁棒性平衡点

2. **自适应删除率函数设计**:

    - 功能：根据输入属性动态确定最优删除率
    - 核心思路：将删除率建模为序列长度的函数 p = f(n)。短序列用更低的删除率保留分类信号；长序列用更高的删除率增强鲁棒性（因为信号冗余度更高）。具体函数形式可通过理论分析或实验搜索确定
    - 设计动机：自然语言中句子长度差异很大，固定删除率对变长输入是次优的——这一直觉虽然自然，但之前未被严格形式化

3. **编辑距离认证区域计算**:

    - 功能：在可变删除率设定下计算认证区域，即在保持正确分类的前提下允许的最大编辑距离
    - 核心思路：认证区域的基数取决于删除率和多数投票的置信度。在可变率下需要额外考虑攻击引起的长度变化对删除率的影响，这增加了计算的复杂性但不影响认证的 soundness
    - 设计动机：认证区域基数是衡量认证鲁棒性的核心指标，其数量级提升直接反映方法有效性

### 损失函数 / 训练策略
AdaptDel 本身不涉及新的训练流程，而是一种推理时的认证方法。基础分类器可使用标准方式训练（如在删除增强的数据上训练或微调预训练模型）。AdaptDel 的贡献在于推理时的认证框架而非训练策略。

## 实验关键数据

### 主实验

| 设定 | 指标 | AdaptDel | 固定率 SOTA | 提升幅度 |
|------|------|----------|-----------|---------|
| NLP 序列分类 | 认证区域基数中位数 | 显著更高 | 基线 | 最高 **30 个数量级** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 自适应率 vs 固定率 | 认证区域基数 | 自适应率在变长输入上全面优于固定率 |
| 不同自适应策略 | 认证区域基数 | 基于长度的策略最直接有效 |
| 短文本测试 | 认证改善幅度 | 短序列从低删除率中受益最大 |
| 长文本测试 | 认证改善幅度 | 长序列从高删除率中受益于增大鲁棒性半径 |

### 关键发现
- 认证区域基数提升 30 个数量级是质变而非增量改善，反映了固定率在变长输入上的严重次优性
- 改善主要来自对不同长度输入使用匹配的删除率，避免了短文本过度删除和长文本删除不足
- 理论保证（soundness）确保可变率方法的认证不会比声称的更弱

## 亮点与洞察
- 30 个数量级的提升在认证鲁棒性领域极为罕见，说明固定删除率在变长输入场景下造成了巨大的认证浪费
- 理论贡献扎实：将可变率 soundness 证明与编辑距离攻击分析结合是非平凡的数学工作
- 核心洞察简洁有力：固定删除率对变长输入次优——自然直觉但之前未被形式化解决
- 方法与基础分类器无关（classifier-agnostic），可直接应用于任何现有文本分类模型

## 局限与展望
- 自适应策略的具体函数形式可能需要针对不同任务和数据集调优
- 认证计算开销随输入长度和采样次数增长，大规模部署可能面临效率瓶颈
- 主要在 NLP 序列分类上验证，其他离散序列任务（音频分类、DNA 序列分析等）的适用性待探索
- 论文 33 页（含附录），理论内容丰富但对读者数学背景要求较高

## 相关工作与启发
- **固定率 Randomized Smoothing for Discrete Sequences**：AdaptDel 是其严格推广，可变率在变长输入上大幅优越
- **Gaussian-based Randomized Smoothing**（Cohen et al., 2019）：连续空间经典方法，AdaptDel 处理离散序列
- **IBP / CROWN**：确定性认证方法，通常针对连续空间和神经网络层，不适用于离散编辑距离攻击
- 启发：将超参数从固定值变为输入依赖函数是一种通用的改进策略，可推广到其他需要固定超参数的方法

## 评分
- 新颖性: ⭐⭐⭐⭐ 自适应删除率是自然但之前未被形式化的扩展，理论推导非平凡
- 实验充分度: ⭐⭐⭐ 30个数量级的提升有说服力，但详细实验设置和数据集受限于摘要信息
- 写作质量: ⭐⭐⭐⭐ 33页 camera-ready 版本，理论严谨完整
- 价值: ⭐⭐⭐⭐ 对离散序列认证鲁棒性领域贡献显著，思路可推广

<!-- RELATED:START -->

## 相关论文

- [The Impact of Scaling Training Data on Adversarial Robustness](the_impact_of_scaling_training_data_on_adversarial_robustness.md)
- [AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding](../../AAAI2026/audio_speech/say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)
- [Target Speaker Extraction Through Comparing Noisy Positive and Negative Audio Enrollments](target_speaker_extraction_through_comparing_noisy_positive_and_negative_audio_en.md)
- [VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](vita-15_towards_gpt-4o_level_real-time_vision_and_speech_interaction.md)

<!-- RELATED:END -->
