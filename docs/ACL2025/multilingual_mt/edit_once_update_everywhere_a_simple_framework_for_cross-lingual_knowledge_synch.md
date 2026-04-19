---
title: >-
  [论文解读] Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs
description: >-
  [ACL 2025][知识编辑] 提出 X-KDE 框架，通过跨语言编辑指令微调（XE-IT）+ 目标语言偏好优化（TL-PO）实现"编辑一种语言、所有语言同步更新"的跨语言知识民主化，在 Bi-ZsRE 和 MzsRE 基准上平均提升 +8.19%，跨语言场景下显著超越所有现有方法。
tags:
  - ACL 2025
  - 知识编辑
  - 跨语言
  - multilingual synchronization
  - 偏好优化
  - 指令微调
---

# Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs

**会议**: ACL 2025  
**arXiv**: [2502.14645](https://arxiv.org/abs/2502.14645)  
**代码**: https://github.com/YukinoshitaKaren/X_KDE  
**领域**: LLM/NLP - 知识编辑  
**关键词**: 知识编辑, 跨语言, multilingual synchronization, 偏好优化, 指令微调

## 一句话总结
提出 X-KDE 框架，通过跨语言编辑指令微调（XE-IT）+ 目标语言偏好优化（TL-PO）实现"编辑一种语言、所有语言同步更新"的跨语言知识民主化，在 Bi-ZsRE 和 MzsRE 基准上平均提升 +8.19%，跨语言场景下显著超越所有现有方法。

## 研究背景与动机

**领域现状**：知识编辑允许在不完全重训练的情况下高效更新 LLM 中的知识。主流方法包括参数修改型（ROME、MEMIT 修改 FFN 参数）、微调型（FT-L、FT-M 微调特定层）和上下文型（IKE 利用 in-context learning、LTE 利用检索增强）。

**现有痛点**：(1) 现有方法主要聚焦于单语言编辑，跨语言知识同步被忽视——在英语中编辑了一条知识，模型在中文/法语/泰语等语言中仍然保留旧知识；(2) 参数修改方法（ROME/MEMIT）在跨语言场景下性能急剧下降；(3) 即使是最好的基线 LTE，在目标语言上的可靠性和可移植性仍有很大提升空间。

**核心矛盾**：LLM 的多语言知识表示并非完全对齐——同一事实在不同语言中的参数编码路径不同，导致单语编辑无法自然传播到其他语言。

**本文目标** 实现真正的跨语言知识同步——在一种（主导）语言中编辑知识后，确保更新自动传播到所有其他语言。

**切入角度**：通过指令微调教会模型"在看到编辑指令时应如何更新知识"的行为模式，再通过偏好优化强化跨语言一致性。

**核心 idea**：XE-IT 让模型学会知识编辑的行为模式，TL-PO 让编辑效果跨语言传播，两阶段联合实现知识的跨语言民主化。

## 方法详解

### 整体框架
两阶段训练流程：(1) 第一阶段 XE-IT（Cross-lingual Edition Instruction Tuning）：在精心构建的平行数据集上微调模型，使其学会根据指令修改目标知识同时保留无关知识；(2) 第二阶段 TL-PO（Target-language Preference Optimization）：使用 DPO 等偏好优化技术确保编辑效果在目标语言中的一致性。

### 关键设计

1. **跨语言编辑指令微调（XE-IT）**:

    - 功能：在构建的平行数据集上微调模型，训练数据包含编辑指令和对应的知识更新示范
    - 核心思路：将知识编辑任务格式化为指令跟随任务——给模型展示"旧事实→新事实"的编辑指令，同时包含同一事实在多种语言中的更新示范。训练目标是让模型学会：(a) 在看到编辑指令时修改指定知识，(b) 保持未涉及知识不变（locality），(c) 将编辑效果推广到相关问题（portability）
    - 设计动机：比传统知识编辑方法（直接修改参数）更灵活——不需要定位和修改特定参数，而是让模型自主学习知识更新行为

2. **目标语言偏好优化（TL-PO）**:

    - 功能：使用 DPO（Direct Preference Optimization）等技术，以目标语言中的正确更新为正样本、旧知识为负样本进行偏好学习
    - 核心思路：XE-IT 可能在源语言上效果好但在目标语言上不够一致。TL-PO 通过偏好优化进一步强化模型在目标语言中选择新知识而非旧知识的倾向
    - 设计动机：即使模型学会了编辑行为模式，跨语言传播仍需额外的对齐信号——偏好优化提供了这种信号

3. **高质量跨语言数据集**:

    - 功能：构建专门设计的平行数据集，包含跨语言知识编辑的训练样本
    - 核心思路：每条数据包含编辑前后的知识、多语言版本的问答对、以及 locality/portability 测试样本
    - 设计动机：现有知识编辑数据集多为单语或简单双语，不足以训练真正的跨语言同步能力

## 实验关键数据

### 主实验（Bi-ZsRE，Llama2-Chat-7B，英语编辑→跨语言测试）
| 方法 | 英语 Reliability | 英语 Portability | 中文 Reliability | 中文 Portability | 总平均 |
|------|-----------------|-----------------|-----------------|-----------------|--------|
| FT-L | 53.51 | 53.31 | 51.81 | 55.14 | 61.90 |
| ROME | 96.09 | 58.87 | 49.94 | 51.81 | 73.43 |
| MEMIT | 95.21 | 57.77 | 52.05 | 52.19 | 74.46 |
| IKE | 99.59 | 71.27 | 67.83 | 58.97 | 73.33 |
| LTE | 99.91 | 77.40 | 76.86 | 67.49 | 84.28 |
| **X-KDE** | **99.93** | **76.41** | **94.81** | **77.43** | **91.04** |

### 消融实验（MzsRE，12语言平均，英语编辑）
| 方法 | 英语 Reliability | 跨语言平均 Reliability | 跨语言平均 Generality |
|------|-----------------|----------------------|---------------------|
| FT-M | 99.96 | 63.65 | 62.04 |
| IKE | 99.65 | 76.78 | 76.65 |
| LTE | 100.00 | 79.38 | 79.16 |
| **X-KDE** | **99.93** | **90.24** | **90.14** |

### 关键发现
- X-KDE 在跨语言场景下取得压倒性优势——中文 Reliability 从 LTE 的 76.86% 提升到 94.81%（+17.95%）
- 在 MzsRE 的 12 语言场景中，跨语言平均 Reliability 从 79.38% 提升到 90.24%（+10.86%）
- 参数修改方法（ROME/MEMIT）在跨语言场景下性能严重退化——中文 Reliability 分别只有 49.94% 和 52.05%
- X-KDE 在保持高编辑可靠性的同时，Locality 指标也保持在 90%+ ，说明不影响无关知识
- 批量编辑和序列编辑场景中 X-KDE 也保持一致的优势

## 亮点与洞察
- "Edit Once, Update Everywhere"的愿景非常实用——多语言 LLM 中知识更新只需一种语言一次编辑
- 两阶段设计巧妙——XE-IT 解决"怎么编辑"，TL-PO 解决"跨语言一致性"，相互补充
- 在 12 种语言上的一致提升证明方法的语言无关性，包括泰语等与英语差异极大的语言

## 局限与展望
- 主要在 7B 模型上验证，更大模型上的效果未知
- 偏好优化阶段需要额外的训练数据和计算资源
- 未讨论知识冲突的情况——如果不同语言中同一事实本身就不一致该如何处理

## 相关工作与启发
- **vs ROME/MEMIT**：参数修改方法在跨语言场景下严重失效，因为不同语言的知识编码在不同参数位置
- **vs LTE (Jiang et al., 2024)**：LTE 通过 SFT+检索增强实现知识编辑，但缺乏跨语言一致性保证。X-KDE 在 LTE 基础上增加偏好优化来弥补这一不足
- **vs IKE (Zheng et al., 2023)**：上下文学习方法灵活但 Locality 较差（56.95% vs X-KDE 的 90.15%）

## 评分
- 新颖性: ⭐⭐⭐⭐ 跨语言知识同步的问题定义有价值，两阶段方案设计清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 两个基准数据集、12种语言、批量/序列编辑、多种基线对比
- 写作质量: ⭐⭐⭐⭐ 方法描述清楚，实验表格丰富
- 价值: ⭐⭐⭐⭐ 对多语言 LLM 的知识维护有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs](cross-lingual_auto_evaluation_for_assessing_multilingual_llms.md)
- [Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)
- [Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation](cross-lingual_transfer_of_debiasing_and_detoxification_in_multilingual_llms_an_e.md)

<!-- RELATED:END -->
