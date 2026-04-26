---
title: >-
  [论文解读] A General Knowledge Injection Framework for ICD Coding
description: >-
  [ACL 2025][ICD编码] > 本文提出 GKI-ICD，一个通用的知识注入框架，通过指南合成和多任务学习机制，无需额外网络模块即可同时整合 ICD Description、Synonym 和 Hierarchy 三种知识，在 MIMIC-III 基准上取得 SOTA 性能。
tags:
  - ACL 2025
  - ICD编码
  - 知识注入
  - 多任务学习
  - 指南合成
  - 医疗文本分类
---

# A General Knowledge Injection Framework for ICD Coding

**会议**: ACL 2025  
**arXiv**: [2505.18708](https://arxiv.org/abs/2505.18708)  
**代码**: [GitHub](https://github.com/xuzhang0112/GKI-ICD)  
**领域**: 知识编辑 (Knowledge Editing)  
**关键词**: ICD编码, 知识注入, 多任务学习, 指南合成, 医疗文本分类  

## 一句话总结

> 本文提出 GKI-ICD，一个通用的知识注入框架，通过指南合成和多任务学习机制，无需额外网络模块即可同时整合 ICD Description、Synonym 和 Hierarchy 三种知识，在 MIMIC-III 基准上取得 SOTA 性能。

## 研究背景与动机

### 现有痛点

**现有痛点**：研究问题：** ICD 编码任务需要将大量医疗代码分配给临床文本，面临代码长尾分布和缺乏代码级证据标注两大挑战。

**现有方法的不足：**

### 领域现状

**领域现状**：单一知识类型：** 现有方法通常只关注一种知识（描述/同义词/层级关系），设计专用模块

### 核心矛盾

**核心矛盾**：模块不兼容：** 为同义词设计的多同义词注意力机制与为层级关系设计的图神经网络难以整合到统一模型

### 解决思路

**解决思路**：扩展性差：** 专用模块的复杂性使其难以扩展到更先进的模型

**核心动机：** 设计一个通用框架，不依赖专用网络模块，统一注入三种互补的 ICD 代码知识。

## 方法详解

### 整体框架

GKI-ICD 包含两个核心组件：
1. **指南合成 (Guideline Synthesis):** 利用代码知识合成训练指南，替代专用网络模块
2. **多任务学习 (Multi-task Learning):** 模型同时从原始样本和合成指南学习，并通过语义一致性约束对齐

### 关键设计

**1. 指南合成：** 给定一份医疗文档及其标注的 ICD 代码集合，执行以下步骤：
- **描述解析：** 提取每个阳性代码的官方 ICD-9 描述，移除 "NOS" 等非标准术语
- **同义词替换：** 从 UMLS 知识库提取每个代码的同义词，随机替换描述以增强多样性
- **层级检索：** 添加代码所属组的层级描述（如 038.9→030-041→001-139）
- **打乱拼接：** 随机打乱代码顺序后拼接为长文本序列作为合成指南 $\hat{x}$

**2. 多任务学习机制：**
- **原始文本预测：** 标准 ICD 编码损失 $L_{raw} = L_{BCE}(f(x), y)$
- **指南预测：** 要求模型也能从合成指南正确预测代码 $L_{guide} = L_{BCE}(f(\hat{x}), y)$
- **语义一致性约束：** 对齐从原始文本和指南提取的代码特定表示 $L_{sim} = 1 - cosine(E, \hat{E})$

### 损失函数

$$L = L_{raw}(x, y) + L_{guide}(\hat{x}, y) + \lambda L_{sim}(E, \hat{E})$$

其中 $\lambda$ 控制语义一致性权重，考虑理论知识与临床表达之间的差距。

## 实验

### 主实验结果

| 模型 | MIMIC-III-Full MacroAUC | MicroAUC | MacroF1 | MicroF1 | P@8 |
|------|---|---|---|---|---|
| CAML | 0.895 | 0.986 | 0.088 | 0.539 | 0.709 |
| PLM-CA | 0.916 | 0.989 | 0.103 | 0.599 | 0.772 |
| MSMN | 0.950 | 0.992 | 0.103 | 0.584 | 0.752 |
| CoRelation | 0.952 | 0.992 | 0.102 | 0.591 | 0.762 |
| **GKI-ICD** | **0.962** | **0.993** | **0.123** | **0.612** | **0.777** |

GKI-ICD 在 MIMIC-III-Full 所有指标上取得 SOTA，MacroAUC 较基座 PLM-CA 提升 4.6%，MacroF1 提升 19.4%。

### 消融实验

| 知识类型组合 | 效果 |
|------|------|
| 无知识 (baseline PLM-CA) | MacroAUC 0.916, MacroF1 0.103 |
| + Description | 提升 |
| + Description + Synonym | 进一步提升 |
| + Description + Synonym + Hierarchy | 最优 (MacroAUC 0.962, MacroF1 0.123) |

三种知识类型逐步贡献提升，验证了多知识整合的必要性和互补性。

### 关键发现

1. **通用框架有效：** 无需专用模块即可注入多种知识，且效果超越使用专用模块的方法
2. **知识互补性强：** 三种知识类型（描述、同义词、层级）逐步贡献增量提升
3. **推理无额外开销：** 知识仅在训练阶段通过合成指南注入，推理时不使用指南，无计算开销增加
4. **超越额外标注方法：** 即便与使用 DRG/CPT 等额外人工标注的方法相比，GKI-ICD 也具有竞争力
5. **长尾改善显著：** MacroF1 大幅提升说明对低频代码的处理能力显著改善

## 亮点与洞察

- 首次在不引入额外网络模块的情况下统一整合三种 ICD 代码知识
- 指南合成方法巧妙地将离散的知识融入连续的文本序列，利用语言模型的语义理解能力
- 训练时注入知识但推理时无额外开销，是非常实用的设计理念
- 在最大公开临床数据集 MIMIC-III 上取得全面 SOTA

## 局限与展望

- 仅在 ICD-9 编码体系上验证，未测试 ICD-10 等更新版本
- 指南合成使用 ground truth 标签，无法直接推广到半监督场景
- 同义词替换的随机性可能引入训练噪声
- 仅使用 RoBERTa-PM 作为编码器，未验证在其他预训练模型上的效果

## 相关工作

- **ICD 编码网络：** CAML (Mullenbach et al., 2018) 首创标签注意力机制；PLM-ICD/PLM-CA (Edin et al., 2022/2024) 引入预训练语言模型
- **知识注入-描述：** ISD (Zhou et al., 2021) 用自蒸馏结合代码描述；KEPTLongformer (Yang et al., 2022) 将描述作为提示
- **知识注入-同义词：** MSMN (Yuan et al., 2022) 用多同义词注意力学习多样化代码表示
- **知识注入-层级：** MSATT-KG (Xie et al., 2019) 用图卷积捕捉代码间层级关系

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 综合评价 | 8.0/10 |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning](memorizing_is_not_enough_deep_knowledge_injection_through_reasoning.md)
- [\[ACL 2025\] Structure-aware Domain Knowledge Injection for Large Language Models](structure-aware_domain_knowledge_injection_for_large_language_models.md)
- [\[ACL 2025\] ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)
- [\[NeurIPS 2025\] KScope: A Framework for Characterizing the Knowledge Status of Language Models](../../NeurIPS2025/knowledge_editing/kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)
- [\[AAAI 2026\] Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing](../../AAAI2026/knowledge_editing/hybrid-dmkg_a_hybrid_reasoning_framework_over_dynamic_multimodal_knowledge_graph.md)

<!-- RELATED:END -->
