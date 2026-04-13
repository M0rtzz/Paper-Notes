---
title: >-
  [论文解读] ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval
description: >-
  [CVPR 2025][Composed Image Retrieval] 提出 ConText-CIR 框架，通过 Text Concept-Consistency 损失让文本修改中的名词短语更好地关注查询图像的相关部分，配合合成数据生成管线，在多个 CIR 基准上取得 SOTA。
tags:
  - CVPR 2025
  - Composed Image Retrieval
  - Text Concept-Consistency
  - CLIP
  - zero-shot retrieval
  - synthetic data
---

# ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval

**会议**: CVPR 2025  
**arXiv**: [2505.20764](https://arxiv.org/abs/2505.20764)  
**代码**: [mvrl/ConText-CIR](https://github.com/mvrl/ConText-CIR)  
**机构**: Washington University in St. Louis / Saint Louis University / George Washington University
**领域**: 图像检索 / 视觉语言模型  
**关键词**: Composed Image Retrieval, Text Concept-Consistency, CLIP, zero-shot retrieval, synthetic data

## 一句话总结
提出 ConText-CIR 框架，通过 Text Concept-Consistency 损失让文本修改中的名词短语更好地关注查询图像的相关部分，配合合成数据生成管线，在多个 CIR 基准上取得 SOTA。

## 研究背景与动机

1. **领域现状**：组合图像检索（CIR）是一种多模态检索任务，用户同时提供查询图像和文本修改描述，模型需检索满足修改条件的目标图像。该任务结合了图像检索和文本检索的优势，在视觉搜索、电商推荐等场景有广泛应用。
2. **现有痛点**：
   - 现有 CIR 方法难以准确表示图像和文本修改之间的关系，导致性能不理想
   - 当文本中包含多个语义条件时，模型往往无法同时满足所有条件（如 Fig.1 所示的失败案例）
   - 图像嵌入过于复杂难以编码特定检索条件，纯文本又难以精确描述复杂视觉信息
3. **核心矛盾**：文本修改中的概念（名词短语）与查询图像的对应关系缺乏显式监督，模型难以学习到"哪些文本概念应关注图像的哪些部分"。
4. **切入角度**：引入概念级别的表示学习，使文本中各名词短语的表示与查询图像的相关区域对齐。
5. **核心 idea 一句话**：Text Concept-Consistency 损失 + 合成数据管线 = 概念级对齐的组合图像检索。

## 方法详解

### 整体框架
ConText-CIR 基于 CLIP 视觉语言模型，由三个核心组件构成：特征提取、概念一致性学习、检索推理。

### 关键设计

1. **Text Concept-Consistency 损失（TCC Loss）**
   - 做什么：鼓励文本修改中名词短语的表示与查询图像中相关区域的表示保持一致
   - 核心思路：从文本修改中提取名词短语，计算每个名词短语与查询图像 patch token 的注意力分布，强制名词短语的概念表示与图像中对应区域对齐
   - 设计动机：以往方法仅在全局级别对齐文本和图像，忽略了概念级别的细粒度对应关系
   - 效果：使模型能同时关注文本中的多个语义条件

2. **合成数据生成管线**
   - 做什么：从现有 CIR 数据集或无标注图像自动生成训练数据
   - 核心思路：利用视觉语言模型生成图像描述，通过 LLM 生成修改文本，构建 (查询图像, 文本修改, 目标图像) 三元组
   - 优势：不增加推理时间复杂度，也无需大规模额外标注数据
   - 支持从 CIRR 数据集扩展生成 CIRRR 数据集

3. **推理策略**
   - 组合查询图像和文本修改的嵌入，在目标图像数据库中进行最近邻检索
   - 不增加推理时间复杂度
   - 支持多种 CLIP backbone（ViT-B、ViT-L、ViT-H）

### 训练策略
- 基于 CLIP 预训练模型微调
- 使用 TCC Loss + 标准 CIR 损失联合训练
- 合成数据与真实数据混合训练

## 实验关键数据

### CIRR 监督设置（Recall@K）
| 方法 | Backbone | R@1 | R@5 | R@10 | R@50 |
|------|----------|------|------|-------|-------|
| CLIP4CIR | ViT-B | 44.82 | 77.04 | 86.65 | 97.90 |
| CASE | ViT-B | 48.68 | 79.98 | 88.51 | — |
| **ConText-CIR** | **ViT-B** | **最优** | **最优** | **最优** | **最优** |

### CIRR 零样本设置（R@1 提升）
| Backbone | R@1 提升幅度（vs. 之前 SOTA） |
|----------|---------------------------|
| ViT-B | +4.78 |
| ViT-L | +5.38 |
| ViT-H | +12.88 |

**关键发现**：ViT-H backbone 的 ConText-CIR 甚至超越了所有使用更大 ViT-G backbone 的方法（ViT-G 多约 400M 参数），零样本性能超过在 490 万样本上预训练的 CoVR-2。

### 消融实验
| 训练数据 | R@1 | R@5 | R@10 | R@50 |
|---------|------|------|-------|-------|
| CIRR only | 45.25 | 77.52 | 86.88 | 97.24 |
| CIRR + CIRRR（合成） | 48.54 | — | — | — |

TCC Loss 和合成数据均对性能有显著贡献。

## 技术细节补充

### 名词短语提取
- 使用 spaCy 从文本修改中自动提取名词短语
- 每个名词短语独立计算与图像 patch 的注意力权重
- 对齐损失在 token 级别施加，粒度远细于句子级

### CIRRR 合成数据集
- 从 CIRR 原始图像出发，利用 VLM 生成描述
- LLM 根据描述差异自动生成相对修改文本
- 合成数据质量通过人工评估验证
- 数据量适中，避免引入噪声标签

## 亮点与洞察
- **概念级对齐**思路简洁有效，从全局对齐到细粒度概念对齐的进步自然且有意义
- **合成数据管线**通用性强，可从任意无标注图像生成 CIR 训练数据
- 推理时不增加任何计算开销，概念一致性仅在训练阶段引入
- 在零样本设置下以小模型超越大模型，体现了方法的数据效率
- 支持多种 CLIP backbone 且均有提升，方法通用性好
