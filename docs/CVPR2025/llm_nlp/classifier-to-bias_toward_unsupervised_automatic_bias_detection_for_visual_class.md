---
title: >-
  [论文解读] Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers
description: >-
  [CVPR 2025][LLM/NLP][偏差发现] 本文提出Classifier-to-Bias（C2B），首个无需任何标注数据的视觉分类器偏差自动发现框架，仅依靠分类任务的文本描述，利用LLM生成偏差候选并通过检索增强的生成-验证流程来识别预训练模型中的系统性偏差。
tags:
  - CVPR 2025
  - LLM/NLP
  - 偏差发现
  - 无监督
  - LLM
  - 视觉分类器
  - 公平性
---

# Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 待确认  
**领域**: 公平性 / 偏差检测  
**关键词**: 偏差发现, 无监督, LLM, 视觉分类器, 公平性

## 一句话总结

本文提出Classifier-to-Bias（C2B），首个无需任何标注数据的视觉分类器偏差自动发现框架，仅依靠分类任务的文本描述，利用LLM生成偏差候选并通过检索增强的生成-验证流程来识别预训练模型中的系统性偏差。

## 研究背景与动机

**领域现状**：深度学习模型在部署时可能携带训练数据中的偏差（如性别、种族偏差），这些偏差会导致不公平的预测结果。现有偏差检测方法通常需要包含任务标签的标注数据集来验证偏差是否存在。

**现有痛点**：(1) 现有方法依赖标注数据——用户下载预训练模型时往往没有相应的标注数据集，也缺乏资源去标注。(2) 偏差发现范围受限——已有方法通常预设特定类型的偏差（如人脸属性），无法发现训练数据中未预见的偏差模式。(3) 非专业用户难以使用——现有工具对ML专业知识要求高。

**核心矛盾**：全面的偏差检测需要覆盖各种可能的偏差维度，但标注所有潜在偏差属性的成本极高且不可穷举；而不使用标注数据则缺乏验证偏差的ground truth。

**本文目标** 如何在完全无标注数据的条件下，自动发现视觉分类器中存在的系统性偏差？

**切入角度**：利用LLM的世界知识生成关于可能偏差的假设（如"分类猫狗时是否对深色背景有偏好"），然后通过检索增强的方法从无标签图像中收集证据来验证这些假设。

**核心 idea**：用LLM生成偏差假设候选，用CLIP/检索模型从无标签数据中验证假设，实现零标注偏差检测。

## 方法详解

### 整体框架

C2B包含三个主要阶段：(1) 偏差假设生成——给定分类任务的文本描述，利用LLM生成涵盖多个维度（场景、外观、共现物体等）的偏差假设及对应caption。(2) 图像检索与伪标注——利用视觉-语言模型（如CLIP）从无标签图像集合中检索与各偏差假设匹配的图像子集，构建伪标注验证集。(3) 偏差评估——将检索到的图像输入目标分类器，统计不同偏差条件下的预测分布差异，量化各假设对应偏差的严重程度。

### 关键设计

1. **LLM驱动的偏差假设生成**：
    - 功能：自动生成覆盖多维度偏差的候选集
    - 核心思路：通过精心设计的prompt，引导LLM（如GPT-4）根据分类任务描述生成可能的偏差来源。例如对"猫 vs 狗"分类器，LLM可能生成"室内/室外场景偏差"、"毛色偏差"等假设。每个假设附带一对contrastive caption用于后续图像检索
    - 设计动机：LLM包含丰富的常识知识，能够联想到人类可能忽略的偏差维度

2. **检索增强的偏差验证**：
    - 功能：在无标注情况下为每个偏差假设收集验证证据
    - 核心思路：利用CLIP等视觉-语言模型计算图像与偏差caption之间的相似度，将无标签图像按偏差属性分组。然后观察目标分类器在不同组上的预测差异——如果分类器在"深色背景"组和"浅色背景"组上表现差异显著，则说明存在背景颜色偏差
    - 设计动机：通过视觉-语言对齐模型的零样本能力实现属性标注的替代

3. **偏差量化与排序**：
    - 功能：将检测到的偏差按严重程度排序
    - 核心思路：对每个偏差假设计算统计指标（如不同偏差条件间的预测概率差异、KL散度等），大的差异意味着分类器对该属性存在强偏好。通过假设检验排除随机波动，最终输出按严重程度排序的偏差列表
    - 设计动机：实际场景中用户需要知道哪些偏差最需要优先修复

## 实验关键数据

### 关键发现

- C2B能在不使用任何标注数据的情况下发现与使用标注数据方法相当的偏差
- 在多个基准分类器（如CelebA、Waterbirds）上成功检测到已知偏差（性别-发色关联、背景-鸟类关联）
- LLM生成的偏差假设覆盖率高，能发现人工预设之外的新偏差
- 该方法对不同CLIP模型和LLM后端均表现稳健

## 亮点与洞察

- **真正的零标注**：突破了偏差检测领域对标注数据的依赖，使任何下载预训练模型的用户都能检查偏差
- **LLM作为偏差先验**：巧妙利用LLM的世界知识替代人工枚举偏差类型
- **实用性强**：整个流程自动化程度高，非ML专家也可使用

## 局限与展望

- 偏差假设的质量受限于LLM的知识覆盖范围，可能遗漏特定领域的偏差
- 检索质量依赖CLIP等模型的对齐精度，细粒度属性可能匹配不准确
- 目前主要验证在图像分类任务上，向检测、分割等任务的扩展待验证
- 无标签图像集的分布可能本身就有偏差，影响验证结果

<!-- RELATED:START -->

## 相关论文

- [Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events](../../ACL2025/llm_nlp/synergizing_unsupervised_episode_detection_with_llms_for_large-scale_news_events.md)
- [Attention Speaks Volumes: Localizing and Mitigating Bias in Language Models](../../ACL2025/llm_nlp/attention_speaks_volumes_localizing_and_mitigating_bias_in_language_models.md)
- [Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)
- [Evaluating Implicit Bias in Large Language Models by Attacking from a Psychometric Perspective](../../ACL2025/llm_nlp/evaluating_implicit_bias_in_large_language_models_by_attacking_from_a_psychometr.md)
- [Test-Time Visual In-Context Tuning](test-time_visual_in-context_tuning.md)

<!-- RELATED:END -->
