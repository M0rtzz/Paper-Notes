---
title: >-
  [论文解读] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry
description: >-
  [ACL 2026][古典诗词检测] 本文构建了首个面向LLM生成古典中文诗词的检测基准ChangAn（含30,664首诗），系统评估了12种AI检测方法在不同文本粒度和生成策略下的表现，揭示了当前中文文本检测器在古典诗词领域的严重局限性。
tags:
  - ACL 2026
  - 古典诗词检测
  - AI生成文本
  - 中文NLP
  - 基准测试
  - 文学创作
---

# Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry

**会议**: ACL 2026  
**arXiv**: [2604.10101](https://arxiv.org/abs/2604.10101)  
**代码**: [GitHub](https://github.com/VelikayaScarlet/ChangAn)  
**领域**: AIGC检测  
**关键词**: 古典诗词检测, AI生成文本, 中文NLP, 基准测试, 文学创作

## 一句话总结

本文构建了首个面向LLM生成古典中文诗词的检测基准ChangAn（含30,664首诗），系统评估了12种AI检测方法在不同文本粒度和生成策略下的表现，揭示了当前中文文本检测器在古典诗词领域的严重局限性。

## 研究背景与动机

**领域现状**：大语言模型在文本生成领域的能力已扩展到文学创作，包括古典中文诗词。AI生成的诗歌作品在各类诗刊和文学平台上引发了关于创作真实性和伦理的广泛争议，多家诗歌期刊已公开谴责未标注的AI投稿行为。

**现有痛点**：现有的AI生成文本检测研究主要聚焦于现代语言和当代文体，尚未涉及古典中文诗词这一特殊领域。通用检测方法在该领域面临三重挑战：(1) 古典诗词具有严格的格律规律（平仄、韵律、对仗），检测模型难以判断这些规律是人类遵循诗歌传统还是AI模仿学习模式；(2) 古典诗词共享广泛的意象系统，导致人类和AI在词汇分布上存在大量重叠；(3) 古典诗词存在灵活的词性活用和倒装句法，偏离现代汉语语法，进一步增加了检测难度。

**核心矛盾**：古典诗词的高度形式化约束使得AI生成与人类创作在表层特征上高度相似，传统基于统计特征或语言模型困惑度的检测方法难以有效区分。

**本文目标**：构建首个专门针对LLM生成古典中文诗词的检测基准，系统评估现有检测方法的能力边界，为该领域的检测研究提供数据和实验基础。

**切入角度**：从文本粒度（单首vs多首）和生成策略（直接生成vs推敲优化）两个维度系统评估检测方法，同时探索LLM作为生成者和零样本检测者的双重角色。

**核心 idea**：古典诗词的严格形式约束有效掩盖了AI生成的统计痕迹，使得现有检测器几乎失效，需要专门的检测基准和方法。

## 方法详解

### 整体框架

构建ChangAn数据集，包含10,276首现代人类创作的古典诗词（282位作者）和20,388首由四个主流LLM（DeepSeek-V3.2、GPT-4.1、Kimi-K2、Doubao Seed-1.6）生成的诗词。在此基础上，从基于判断的检测器（LLM直接判断）和基于概率的检测器（统计方法、监督方法）两类共12种方法进行系统评估，考虑单首/多首检测和直接生成/推敲优化两个维度。

### 关键设计

1. **数据集构建策略**:

    - 功能：提供高质量、无数据泄漏的检测基准
    - 核心思路：人类诗词部分选择现代诗人的创作（而非古代诗人），从小红书、百度贴吧等平台和专业文学出版物收集。这样做是因为古代诗词已被纳入LLM预训练语料，使用古代作品会导致模型通过"记忆"判断而非真正的检测能力。LLM生成部分设计了两种提示策略：直接生成和推敲优化（模拟人类"推敲"过程）
    - 设计动机：使用现代诗人作品有效缓解数据泄漏偏差，确保检测任务聚焦于基于风格和结构特征的真正区分能力

2. **多粒度评估设计**:

    - 功能：评估检测方法在不同数据规模下的性能边界
    - 核心思路：设计单首检测（SPD）和多首检测（MPD，6首和12首）三个粒度。单首检测反映社交媒体等场景，多首检测模拟诗集出版场景。选择6首作为基本单位有文化依据——中国古典诗歌中常见六首组诗（如《三吏三别》《塞下曲六首》等）
    - 设计动机：古典诗词文本极短（每首仅几十字），特征极其有限，通过多首聚合评估是否能改善检测效果

3. **LLM自检测能力探索**:

    - 功能：检验LLM能否识别自己生成的诗词
    - 核心思路：让每个LLM同时充当生成者和检测者，通过交叉检测矩阵分析自我识别能力。实验发现大多数模型在古典诗词领域不具备自检测优势——例如Doubao Seed-1.6对自己生成的诗词召回率仅16.09%
    - 设计动机：验证LLM是否存在可利用的"身份信号"，结果表明古典诗词的形式约束有效掩盖了这种信号

### 损失函数 / 训练策略

本文主要是评估型研究。对于监督检测器RoBERTa，使用Chinese RoBERTa模型微调3个epoch，学习率1e-4，batch size 16，数据集按8:1:1划分训练/验证/测试集。所有实验在2块A100 GPU上运行3次取平均值。

## 实验关键数据

### 主实验

| 检测方法 | 类型 | AUROC | Macro-F1 | 说明 |
|---------|------|-------|----------|------|
| RoBERTa（微调） | 监督 | 95.03 | 86.18 | 最佳检测器 |
| Log-Rank | 概率统计 | 85.86 | 75.56 | 最佳无监督方法 |
| Log-Likelihood | 概率统计 | 83.94 | 73.77 | 表现稳定 |
| Fast-DetectGPT | 概率统计 | 49.67 | 49.35 | 完全失效，等于随机猜测 |
| DeepSeek-V3.2 | 判断型 | - | 39.42 | 最佳LLM判断器 |
| GPT-4.1 | 判断型 | - | 23.31 | 最差LLM判断器 |

### 消融实验

| 配置 | AUROC | Macro-F1 | 说明 |
|------|-------|----------|------|
| Kimi-K2生成 | 74.13 | 67.48 | 最难检测的模型 |
| Doubao Seed-1.6生成 | 76.34 | 68.89 | 中等检测难度 |
| DeepSeek-V3.2生成 | 77.17 | 69.59 | 较易检测 |
| GPT-4.1生成 | 76.34 | 69.41 | 较易检测 |

### 关键发现

- Fast-DetectGPT完全失效（AUROC 49.67%），其依赖的负曲率假设在古典诗词的高约束语言结构下不成立
- 中国开发的LLM（DeepSeek、Kimi、Doubao）作为判断型检测器普遍优于GPT-4.1，可能源于训练语料中古典文学的覆盖差异
- LLM在古典诗词领域几乎不具备自我识别能力，古典诗词的严格形式约束有效掩盖了生成痕迹
- 推敲优化策略总体上增加了检测难度（平均AI召回率下降2.94%），但影响因检测器而异
- RoBERTa微调后在所有生成源上均表现最优（AUROC 93.61-97.95%），证明监督学习可以捕捉到共享的机器特征

## 亮点与洞察

- 首个面向古典中文诗词AI检测的基准数据集，填补了重要的研究空白
- 深入分析了古典诗词的语言特征（格律、意象、句法）为何使通用检测方法失效
- 发现不同LLM生成的古典诗词留下的统计痕迹密度相似，暗示古典形式约束对不同模型施加了类似的生成约束
- 推敲优化策略的设计巧妙模拟了人类创作过程中的"推敲"环节

## 局限与展望

- 数据集仅覆盖古典中文诗词的词、绝句、律诗三种体裁，未涉及楚辞、赋等更多形式
- 检测器的评估主要基于现有公开方法，未提出针对古典诗词的专门检测方法
- 人类诗词来源于网络平台，质量参差不齐可能影响检测基线
- 未深入分析具体哪些诗歌风格或主题最易/最难被检测

## 相关工作与启发

- **vs 英文诗歌检测（Chen et al.）**: 英文诗歌形式约束较弱，检测相对容易；古典中文诗词的强形式约束创造了独特挑战
- **vs 现代中文诗歌检测（Wang et al.）**: 现代诗歌形式自由，内在语言特征差异更明显；古典诗词的检测难度更高
- **vs 俳句检测（Hitsuwari et al.）**: 两者都涉及短文本和高约束形式，但古典中文诗词的意象系统更复杂

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个古典中文诗词AI检测基准，选题独特且有现实意义
- 实验充分度: ⭐⭐⭐⭐ 12种检测方法、4种LLM、多粒度多策略的全面评估
- 写作质量: ⭐⭐⭐⭐ 论文结构完整，对古典诗词的语言学分析深入
- 价值: ⭐⭐⭐⭐ 为AI文学检测这一新兴方向提供了重要基础

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code](../../NeurIPS2025/aigc_detection/classical_planning_with_llm-generated_heuristics_challenging_the_state_of_the_ar.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)
- [\[ACL 2025\] Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection](../../ACL2025/aigc_detection/who_writes_what_ai_detection.md)
- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[ACL 2026\] BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories](biasedtales-ml_a_multilingual_dataset_for_analyzing_narrative_attribute_distribu.md)

<!-- RELATED:END -->
