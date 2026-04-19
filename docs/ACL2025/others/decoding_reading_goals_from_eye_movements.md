---
title: >-
  [论文解读] Decoding Reading Goals from Eye Movements
description: >-
  [ACL 2025] 本文首次提出从眼动轨迹中解码读者阅读目标（信息检索 vs. 普通阅读）的任务，通过 12 种模型的系统比较发现基于 Transformer 的扫视路径+语言建模方案（RoBERTa-Eye-F）最优，可在阅读早期即实现高精度实时预测。
tags:
  - ACL 2025
---

# Decoding Reading Goals from Eye Movements

**会议**: ACL 2025  
**作者**: Omer Shubi, Cfir Avraham Hadar, Yevgeni Berzak
**机构**: Technion - Israel Institute of Technology; MIT
**arXiv**: [2410.20779](https://arxiv.org/abs/2410.20779)  
**领域**: 眼动追踪 / 阅读认知 / 多模态分类

---

## 一句话总结

本文首次提出从眼动轨迹中解码读者阅读目标（信息检索 vs. 普通阅读）的任务，通过 12 种模型的系统比较发现基于 Transformer 的扫视路径+语言建模方案（RoBERTa-Eye-F）最优，可在阅读早期即实现高精度实时预测。

---

## 研究背景与动机

### 问题定义

人们在阅读时可能有不同的目标：普通理解阅读（ordinary reading）或信息检索（information seeking，即带着特定问题阅读）。这两种阅读模式是否会在眼动数据中留下可区分的信号？此前研究已发现两种模式在群体平均水平上存在显著差异（阅读速度、注视模式等），但**单次阅读**中能否自动解码阅读目标，尚无系统性研究。

### 动机

- **教育场景**：实时监测学生阅读参与度，区分"认真阅读"和"查找答案"
- **用户界面**：根据用户阅读目标动态调整内容呈现（如高亮关键信息）
- **辅助技术**：帮助特殊人群（如老年用户）导航复杂网页
- **认知科学**：深入理解不同阅读模式的眼动差异机制

### 与现有工作的差距

已有工作（Hahn & Keller, 2023; Malmaud et al., 2020）仅做描述性统计分析，未尝试自动分类。Hollenstein et al. (2023) 的分类工作局限于单句级别且任务为特殊语言学标注，不具备日常阅读代表性。

---

## 方法详解

### 任务形式化

给定参与者 $S$ 对段落 $P$ 的眼动记录 $E_P^S$，预测阅读目标：

$$h: (E_S^P, P) \rightarrow \{\text{Information Seeking}, \text{Ordinary Reading}\}$$

其中段落文本 $P$ 为可选输入，分类器不接收问题内容或参与者身份信息。

### 模型体系（12 + 1 集成）

**仅眼动模型（4 个）**：
1. **Logistic Regression**：9 维全局眼动特征（平均注视时长、扫视幅度等）
2. **BEyeLSTM-No Text**：基于注视序列的 LSTM，不使用文本
3. **ViT**：将扫视路径渲染为图像，用 Vision Transformer 分类
4. **ConvNext v2**：类似 ViT，使用 ConvNext v2 架构

**眼动+文本模型（8 个）**：
- **RoBERTa-Eye-W**：词级眼动特征与词嵌入在输入层融合
- **RoBERTa-Eye-F**：注视级表示（fixation-level），每个注视点独立编码
- **MAG-Eye**：在 Transformer 中间层注入眼动特征
- **PLM-AS**：按注视顺序重排词嵌入，用 RNN 处理
- **Haller RNN**：注视顺序词嵌入拼接眼动特征，RNN 处理
- **BEyeLSTM**：注视序列+全局特征，LSTM+线性投影
- **Eyettention**：RoBERTa 编码器与 LSTM 注视编码器间做交叉注意力
- **PostFusion-Eye**：RoBERTa 词表示与卷积注视特征通过交叉注意力融合

**Logistic Ensemble**：以 12 个模型的输出概率为特征的逻辑回归集成。

### 数据集

使用 **OneStop Eye Movements** 数据集：
- 360 名母语英语成人参与者，EyeLink 1000 Plus 眼动仪
- 30 篇 Guardian 新闻文章（高级版/简化版），54 段落
- 被试间设计：信息检索 vs. 普通阅读
- 共 19,438 个 trial（平衡：9,718 普通阅读 + 9,720 信息检索）
- 每段落被 120 人阅读（各 60 人两种条件）

### 评估协议

10 折交叉验证，三种泛化评估：
- **New Item**：未见段落，已见参与者
- **New Participant**：未见参与者，已见段落
- **New Item & Participant**：两者均未见（零样本）

---

## 实验结果

### 主实验：模型性能对比

| 模型 | New Item | New Participant | New Item & Part. | All |
|------|----------|-----------------|-------------------|-----|
| Majority/Chance | 50.0 | 50.0 | 50.0 | 50.0 |
| Reading Time | 59.0 | 58.9 | 60.4 | 59.0 |
| Logistic Regression | 62.4 | 60.6 | 60.8 | 61.5 |
| BEyeLSTM-No Text | 71.5 | 61.0 | 61.5 | 65.9 |
| ViT | 70.6 | 64.4 | 64.4 | 67.3 |
| ConvNext v2 | 70.4 | 63.7 | 64.0 | 66.9 |
| RoBERTa-Eye-W | 64.6 | 62.5 | 62.0 | 63.5 |
| PostFusion-Eye | 88.5 | 90.3 | 86.0 | 89.3 |
| **RoBERTa-Eye-F** | **89.9** | **90.9** | **88.2** | **90.3** |
| **Logistic Ensemble** | **91.3** | **91.6** | **88.0** | **91.3** |

**关键发现**：RoBERTa-Eye-F 在所有评估设置下均为最优单模型（90.3%），Logistic Ensemble 进一步提升至 91.3%。性能最优的模型共享三个要素：Transformer 架构 + 注视级编码 + 文本建模。

### 实时预测：早期注视数据即可准确分类

| 注视比例 | 1% | 5% | 10% | 25% | 50% | 100% |
|---------|-----|-----|------|------|------|------|
| 平均时间(秒) | 0.5 | 1.5 | 2.7 | 6.3 | 12.4 | 24.3 |
| 准确率 | 61.0 | 77.6 | 78.9 | 82.3 | 84.9 | 90.3 |

仅用前 5% 注视数据（约 1.5 秒）即可达到 77.6% 准确率，证明任务可在线实时执行。

### 误差分析：混合效应建模

通过混合效应模型分析 RoBERTa-Eye-F 预测正确性与 trial 特征的关系：
- **关键发现**：临界区间（critical span）前后的阅读速度是最重要的预测因子
- 信息检索中，读者在关键区间前后读得越快越容易被正确分类
- 普通阅读中，关键区间内读得越慢越容易被正确分类
- 段落越长，普通阅读越容易分类
- 临界区间越短，信息检索越容易分类（目标更明确）

---

## 亮点与创新

- **首次系统性任务定义**：将阅读目标解码形式化为二分类任务，具有明确的实用价值
- **大规模系统比较**：12 种模型涵盖不同架构、数据表示、模态融合策略，提供了全面的方法论 landscape
- **实时可行性验证**：仅 1.5 秒眼动数据即达 77.6% 准确率，支持在线应用
- **混合效应误差分析**：创新的模型性能分析方法，控制多因素后揭示任务难度的可解释轴

---

## 局限性

1. **文本范围有限**：仅涉及 3-10 行段落级新闻文本，未涵盖更短（单句）或更长文本及其他体裁
2. **语言与人群单一**：仅英语母语成人读者，未涵盖二语学习者、不同年龄群体或其他语言
3. **二分类局限**：仅区分两种阅读模式，未尝试解码具体的信息检索问题
4. **泛化至新参与者的性能仍有提升空间**：弱模型在 New Participant 设置下表现较弱
5. **被试间设计**：每位参与者仅参与一种阅读条件，无法完全排除个体效应

---

## 相关工作

- **目标导向阅读**：Just et al. (1982)、Kaakinen & Hyönä (2010) 等研究了略读、速读、校对等任务的眼动差异
- **信息检索阅读分析**：Hahn & Keller (2023)、Malmaud et al. (2020)、Shubi & Berzak (2023) 分析了信息检索与普通阅读的眼动差异，发现关键区间前后差异显著
- **阅读任务分类**：Hollenstein et al. (2023) 在 ZuCo 语料库单句级别尝试分类，但任务为特殊语言学标注
- **眼动预测模型**：RoBERTa-Eye (Shubi et al., 2024)、BEyeLSTM (Reich et al., 2022)、Eyettention (Deng et al., 2023) 等通用眼动预测模型；本文首次将其统一用于阅读目标解码
- **认知状态预测**：眼动用于阅读理解预测 (Reich et al., 2022b; Shubi et al., 2024) 及文档类型/可读性判断

---

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 综合评分 | ⭐⭐⭐⭐ |

> 新任务定义清晰，实验极为全面（12 模型 × 3 泛化设置 × 10 折），混合效应误差分析方法有创新性。数据集虽来自已有资源但利用充分。主要不足在于二分类粒度较粗，且仅限英语新闻体裁。

<!-- RELATED:START -->

## 相关论文

- [SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL](share_text_to_sql_correction.md)
- [Attention Entropy is a Key Factor for Parallel Context Encoding](attention_entropy_parallel_encoding.md)
- [Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)
- [Consultant Decoding: Yet Another Synergistic Mechanism](consultant_decoding_yet_another_synergistic_mechanism.md)

<!-- RELATED:END -->
