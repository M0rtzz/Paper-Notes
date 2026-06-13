---
title: >-
  [论文解读] CogSteer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models
description: >-
  [ACL 2025][LLM 其他][层选择干预] 利用认知科学中的眼动数据分析 LLM 各层行为，发现中间层与人类注视相关性最高且最适合语义干预，提出 CogSteer 框架——仅微调最优单层（约 3% 参数）即可达到或超过全层微调的效果，在 GLUE/毒性控制任务上有效。 领域现状：参数高效微调（PEFT）如 Adapt…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "层选择干预"
  - "眼动追踪"
  - "参数高效微调"
  - "语义引导"
  - "去毒化"
---

# CogSteer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models

**会议**: ACL 2025  
**arXiv**: [2410.17714](https://arxiv.org/abs/2410.17714)  
**代码**: [https://github.com/Ethanscuter/CogSteer](https://github.com/Ethanscuter/CogSteer)  
**领域**: LLM NLP  
**关键词**: 层选择干预, 眼动追踪, 参数高效微调, 语义引导, 去毒化

## 一句话总结
利用认知科学中的眼动数据分析 LLM 各层行为，发现中间层与人类注视相关性最高且最适合语义干预，提出 CogSteer 框架——仅微调最优单层（约 3% 参数）即可达到或超过全层微调的效果，在 GLUE/毒性控制任务上有效。

## 研究背景与动机

**领域现状**：参数高效微调（PEFT）如 Adapter 和 LoRA 是 LLM 适配下游任务的主流方式，默认做法是对**所有层**或**最后一层**插入可训练模块。

**现有痛点**：全层微调参数冗余，部分层对特定任务贡献甚微甚至产生负面影响；仅微调最后一层又不是最优选择，因为最后一层主要负责预测而非语义整合。缺乏一个有理论支撑的层选择策略。

**核心矛盾**：LLM 不同层承担不同功能（句法处理、语义整合、推理预测），但现有 PEFT 方法对层的选择是盲目的——要么全选要么只选最后层。

**本文目标** (a) 用可解释的方式揭示 LLM 各层的功能分工；(b) 基于此找到最适合语义干预的单层，实现更高效的微调和推理时引导。

**切入角度**：认知科学中的眼动追踪（eye-tracking）研究显示，人类阅读时的注视时间、回视等指标反映了不同层次的语言处理过程——从句法到语义到推理。作者将这些人类认知指标与 LLM 各层隐藏状态做相关性分析。

**核心 idea**：用眼动数据发现中间层是语义处理的核心层，只在该层做 PEFT 干预，用 1/N 参数达到全层效果。

## 方法详解

### 整体框架
CogSteer 分三步：(1) 认知启发的可解释性分析——计算 LLM 各层隐藏状态与人类眼动指标的 Pearson 相关性，发现中间层相关性最高；(2) 启发式最优层选择——在中间 bucket（第 N/3 到 2N/3 层）中，用验证集搜索最佳干预层 $M'$；(3) 选择性层干预——仅在最优层插入 Adapter/LoRA 进行微调，或在推理时做隐式层对比干预实现去毒化。

### 关键设计

1. **眼动-隐藏状态相关性分析**：

    - 功能：用 5 种眼动指标（SFD 首次注视时长、FFD 第一遍注视、GD 凝视时长、TRT 总阅读时长、GPT 回视路径时间）衡量人类对每个词的认知加工深度，与 LLM 各层 FFN 隐藏状态做 Pearson 相关
    - 核心思路：对每层 $l$，将所有词的隐藏状态 $\mathbf{h}_{l,i}$ 通过 PCA 降维为标量，再与对应的眼动指标 $e_i^{(k)}$ 计算 $\rho_{l,k}$
    - 设计动机：眼动指标是人类可解释的认知加工度量，比 probing/circuit 方法更通用且可扩展

2. **三阶段层行为发现**：

    - 将 N 层等分为三个 bucket：premature（初级）、middle（中间）、mature（成熟）
    - 发现 1：所有模型（GPT-2、Llama2-7B）的中间层与眼动相关性最高，说明中间层负责深层语义整合
    - 发现 2：任务阅读（关系检测）比自然阅读在中间层和成熟层有更高相关性，说明这些层参与推理；Llama2 因为 RLHF 训练在中间层也展现推理能力

3. **启发式最优层选择**：

    - 功能：从中间 bucket 的候选层中，选出在验证集上表现最好的层 $M'$
    - 公式：$M' = \arg\max_{l \in J} Score(D; P(\cdot | x_t, l))$，其中 $J$ 是中间 bucket 的层集合
    - 设计动机：基于认知分析的层功能先验，将搜索空间从 N 层缩小到约 N/3 层，大幅降低搜索成本

4. **隐式层对比干预（推理时去毒化）**：

    - 功能：在推理时不引入额外参数，通过对比毒性模型和原始模型在最优层的 value 向量来引导生成方向
    - 核心思路：计算语义引导方向 $\Delta v^M = v_c^M - v_o^M$，然后更新原始模型的 value 向量 $v'^M = v_o^M - \lambda_{norm}^\alpha \cdot \Delta v^M$，最后归一化保持向量范数
    - 与 Contrastive Decoding 的区别：不在输出层做对比，而是在中间层的 attention value 向量上做隐式干预，更加精细

### 训练策略
- GLUE 任务：在最优层插入 Adapter（GPT-2 用 vanilla Adapter，Llama2 用 LLaMa-Adapter），仅训练该层的 Adapter 参数
- 毒性控制：先在 Jigsaw 毒性数据集上微调得到毒性对比模型，推理时用层对比干预做去毒化

## 实验关键数据

### 主实验（GLUE Benchmark）

| 模型 | 干预方式 | MNLI-M | RTE | SST-2 | Avg. (Test) | 参数量 |
|---|---|---|---|---|---|---|
| GPT-2 Large | 全层微调 | 82.6 | 62.6 | 93.5 | 77.1 | 14.8M (100%) |
| GPT-2 Large | 单层 L19 | 79.3 | 64.6 | 92.4 | 75.8 | 0.4M (2.7%) |
| Llama2-7B | 全层微调 | 89.5 | 58.2 | 93.5 | 78.7 | 1.3M (100%) |
| Llama2-7B | 单层 L14 | 82.9 | 74.7 | 95.2 | **80.5** | 0.04M (3.1%) |
| Mistral-7B | 全层微调 | 89.7 | 52.3 | 97.3 | 77.3 | 134.5M (100%) |
| Mistral-7B | 单层 L12 | 87.1 | 81.0 | 95.9 | **83.2** | 4.2M (3.1%) |

- Llama2 单层干预比全层高 **+1.8** 平均分，Mistral 高 **+5.9**，仅用 3.1% 参数

### 消融实验（毒性控制）

| 配置 | GPT-2 毒化↑ | GPT-2 去毒↓ | Llama2 毒化↑ | Llama2 去毒↓ |
|---|---|---|---|---|
| 全层微调 | 0.86 | 0.60 | 0.86 | 0.62 |
| 最优单层 | **0.87** (+1.2%) | **0.59** | **0.87** (+1.2%) | **0.59** |
| 最后一层 | 0.83 | 0.63 | 0.71 | 0.73 |

### 关键发现
- **最优干预层一致落在中间 bucket**：GPT-2 的 L19（共 36 层），Llama2 的 L14（共 32 层），Mistral 的 L12（共 32 层），与认知分析预测完全吻合
- **效率极高**：平均仅需全层微调一半的训练时间和 3% 的参数
- **去毒化显著**：Llama2 单层去毒比最后一层干预高 +24%，说明中间层语义引导效果远超输出层
- **RTE 任务上提升惊人**：Mistral 单层 L12 比全层高 +28.7%，说明全层微调在某些推理任务上反而有害

## 亮点与洞察
- **认知科学与 AI 的精妙桥接**：用眼动追踪数据来理解 LLM 层行为是全新的视角。眼动指标是经过几十年认知科学验证的人类加工度量，用它来指导模型干预既有理论基础又有实用价值
- **"少即是多"的直觉验证**：直觉上，微调所有层不如只微调最关键的那一层——这篇论文用认知证据和大量实验证实了这一点。可直接迁移到任何 PEFT 场景
- **隐式层对比干预的巧妙之处**：不在输出层做对比（DExperts 等方法），而是在中间层的 value 向量上做对比+归一化，干预更精细且不影响生成流畅性

## 局限与展望
- **仅分析了 FFN 隐藏状态**：未探索 attention block 与眼动的关系，可能遗漏重要信息
- **三等分 bucket 划分过于简化**：不同模型的层功能边界可能不均匀，自适应层聚类可能更优
- **眼动数据的语言局限**：使用的眼动数据集主要是英文，跨语言泛化性未验证
- **实验基于中等规模模型**：GPT-2、Llama2-7B、Mistral-7B，更大规模模型（70B+）的层行为可能不同
- 改进方向：将眼动分析扩展到 attention 模块；探索动态多层干预；结合不同认知任务的眼动数据做更细粒度的层功能分析

## 相关工作与启发
- **vs LoRA/Adapter 全层微调**：传统 PEFT 不选层，CogSteer 用认知分析选层后效果更好且参数更少
- **vs Contrastive Decoding (Li et al. 2023)**：对比解码在输出层做 logit 对比，CogSteer 在中间层 value 向量做对比，干预位置更合理
- **vs DExperts (Liu et al. 2021)**：DExperts 需要额外的 expert/anti-expert 模型，CogSteer 仅用单个对比模型在单层做干预
- 这篇论文的认知分析方法可以迁移到 VLM 的层选择、MoE 的 expert 路由等场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 眼动+LLM 层分析的跨学科视角很新颖，但选择性层微调的 idea 本身不算突破
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个模型 × 多个 benchmark（GLUE+毒性控制）+ 多种 PEFT 方法，非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，认知分析部分的可视化直观
- 价值: ⭐⭐⭐⭐ 实用价值高，任何使用 PEFT 的人都可以用中间层选择策略来节省资源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Multi-Attribute Steering of Language Models via Targeted Intervention](multi_attribute_steering.md)
- [\[ACL 2025\] SConU: Selective Conformal Uncertainty in Large Language Models](sconu_selective_conformal_uncertainty_in_large_language_models.md)
- [\[ACL 2025\] Steering off Course: Reliability Challenges in Steering Language Models](steering_off_course_reliability_challenges_in_steering_language_models.md)
- [\[ACL 2025\] CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models](cognibench_cognitive_faithfulness.md)
- [\[ACL 2025\] From Data to Knowledge: Evaluating How Efficiently Language Models Learn Facts](from_data_to_knowledge_evaluating_how_efficiently_language_models_learn_facts.md)

</div>

<!-- RELATED:END -->
