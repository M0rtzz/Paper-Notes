---
title: >-
  [论文解读] D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Models
description: >-
  [ACL 2025 (Findings)][图像生成][distractor generation] 提出 D-GEN——首个开源干扰项生成模型（LLaMA微调，8B/70B），自动将开放式评测题转为多选题格式，配套排名对齐+熵分析两种评估方法验证干扰项质量，在 MMLU 上 Spearman's ρ=0.99 保持模型排名一致性。
tags:
  - "ACL 2025 (Findings)"
  - "图像生成"
  - "distractor generation"
  - "multiple-choice evaluation"
  - "LLM benchmark"
  - "ranking alignment"
  - "entropy analysis"
---

# D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Models

**会议**: ACL 2025 (Findings)  
**arXiv**: [2504.13439](https://arxiv.org/abs/2504.13439)  
**代码**: [GitHub](https://github.com/emorynlp/D-Gen)  
**领域**: 图像生成  
**关键词**: distractor generation, multiple-choice evaluation, LLM benchmark, ranking alignment, entropy analysis

## 一句话总结
提出 D-GEN——首个开源干扰项生成模型（LLaMA微调，8B/70B），自动将开放式评测题转为多选题格式，配套排名对齐+熵分析两种评估方法验证干扰项质量，在 MMLU 上 Spearman's ρ=0.99 保持模型排名一致性。

## 研究背景与动机

**领域现状**：LLM 评估中 MC 格式（MMLU/HellaSwag）比开放生成更可靠——直接从 logits 提取预测。但构建 MC 数据集需要精心设计的干扰项。

**现有痛点**：(a) 人工标注耗时昂贵；(b) 检索式方法易引入偏差；(c) BLEU/ROUGE 不衡量干扰项的"迷惑性"。

**核心矛盾**：开放式题目多但缺乏 MC 版本——需要自动转换并保证评测效力。

**本文目标**：构建高质量干扰项生成器 + 设计衡量干扰项质量的自动评估方法。

**切入角度**：好的干扰项应让模型排名与人工干扰项一致（排名对齐），且让模型置信度分布相似（熵分析）。

**核心 idea**：用 MMLU 辅助集微调 LLaMA + 自动过滤低质量项 + 排名对齐&熵双重验证。

## 方法详解

### 整体框架
输入：问题+正确答案。输出：3个高质量干扰项。

### 关键设计

1. **D-GEN 训练**

    - 功能：微调LLaMA 8B/70B生成干扰项
    - 核心思路：用MMLU auxiliary set训练，输入"Question+Answer"，输出干扰项列表
    - 设计动机：微调比prompt更稳定

2. **自动过滤与再生成**

    - 过滤条件：与正确答案重复/干扰项间重复/格式错误
    - 对不合格项重新生成直到满足条件

3. **排名对齐 (Ranking Alignment)**

    - 功能：验证D-GEN干扰项是否保持LLM排名
    - 核心思路：分别用原始和D-GEN干扰项评测20+模型→Spearman/Kendall相关
    - 设计动机：保持区分力=好的干扰项

4. **熵分析 (Entropy Analysis)**

    - 功能：比较模型在两组干扰项上的置信度分布
    - 核心思路：$H = -\sum p_i \log p_i$，原始 vs D-GEN的熵分布应相似
    - 设计动机：衡量"迷惑程度"是否一致

## 实验关键数据

### 排名对齐 (MMLU, 20+ models)

| 指标 | D-GEN 8B | D-GEN 70B |
|------|---------|----------|
| Spearman's ρ | 0.98 | **0.99** |
| Kendall's τ | 0.92 | **0.94** |

### 熵分布匹配 (KL散度 vs 原始)

| 配置 | KL散度 |
|------|--------|
| 随机干扰项 | 0.45 |
| GPT-4生成 | 0.12 |
| **D-GEN 70B** | **0.08** |

### 人类评估 (1-5分)

| 维度 | D-GEN 70B |
|------|----------|
| 流利度 | 4.7 |
| 连贯性 | 4.5 |
| 迷惑性 | 4.2 |
| 错误性 | 4.8 |

### 关键发现
- **排名几乎完美保持**：ρ=0.99意味着D-GEN干扰项不改变LLM排名
- **熵分布高度匹配**：模型置信度模式与人工干扰项几乎相同
- **跨域泛化**：在math/commonsense/structured data等非MMLU任务上也有效
- **70B > 8B**：更大模型生成更高质量干扰项
- **自动过滤关键**：过滤5-10%低质量项

## 亮点与洞察
- **"MC化"标准管道**：使任何开放式benchmark自动转为MC格式——消除答案格式解析痛点
- **排名对齐的功能性评估**：不判断单个干扰项好坏，而看整体评测效力是否保持
- **熵分析超越文本相似度**：看"模型被迷惑程度"而非"文本像不像"——更贴近评测目的
- **可用于检测数据污染**：模型在D-GEN版本上表现远低于原始版本=记忆了原始干扰项

## 局限与展望
- **训练数据来自MMLU**：覆盖领域受限
- **仅限英语**：多语言未探索
- **干扰项数量固定为3**
- **依赖logit访问**：不适用于纯API模型

## 相关工作与启发
- **vs GPT-4直接生成**：D-GEN微调模型在熵匹配上更好(KL=0.08 vs 0.12)
- **vs Raina et al. (2023)**：他们评估单个干扰项影响，D-GEN评估整体熵分布
- **启发**：D-GEN+排名对齐可检测benchmark数据污染

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个开源干扰项生成器+排名对齐/熵分析
- 实验充分度: ⭐⭐⭐⭐⭐ 20+模型排名+熵分析+跨域+人类评估
- 写作质量: ⭐⭐⭐⭐ 方法清晰直观
- 价值: ⭐⭐⭐⭐⭐ LLM评测标准化MC转换管道，实用性高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Conf-Gen: Conformal Uncertainty Quantification for Generative Models](../../ICML2026/image_generation/conf-gen_conformal_uncertainty_quantification_for_generative_models.md)
- [\[ICCV 2025\] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](../../ICCV2025/image_generation/adiee_automatic_dataset_creation_and_scorer_for_instruction-guided_image_editing.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[CVPR 2025\] ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting](../../CVPR2025/image_generation/chatgen_automatic_text-to-image_generation_from_freestyle_chatting.md)
- [\[ICCV 2025\] CAP: Evaluation of Persuasive and Creative Image Generation](../../ICCV2025/image_generation/cap_evaluation_of_persuasive_and_creative_image_generation.md)

</div>

<!-- RELATED:END -->
