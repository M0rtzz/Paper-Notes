---
title: >-
  [论文解读] From Teacher to Student: Tracking Memorization Through Model Distillation
description: >-
  [ACL 2025][视频理解][知识蒸馏] 系统研究了知识蒸馏（KD）对大语言模型记忆化行为的影响，发现蒸馏不仅能压缩模型，还能显著降低对训练数据的逐字记忆风险——其中反向 KL 蒸馏（RKLD/MiniLLM）将记忆化比例从 SFT 的 65.4% 降至最低 6.0%。 领域现状：大语言模型已被证明会记忆并可能泄露训练数…
tags:
  - "ACL 2025"
  - "视频理解"
  - "知识蒸馏"
  - "记忆化"
  - "隐私保护"
  - "模型压缩"
  - "GPT-2"
---

# From Teacher to Student: Tracking Memorization Through Model Distillation

**会议**: ACL 2025  
**arXiv**: [2506.16170](https://arxiv.org/abs/2506.16170)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 知识蒸馏, 记忆化, 隐私保护, 模型压缩, GPT-2

## 一句话总结
系统研究了知识蒸馏（KD）对大语言模型记忆化行为的影响，发现蒸馏不仅能压缩模型，还能显著降低对训练数据的逐字记忆风险——其中反向 KL 蒸馏（RKLD/MiniLLM）将记忆化比例从 SFT 的 65.4% 降至最低 6.0%。

## 研究背景与动机

**领域现状**：大语言模型已被证明会记忆并可能泄露训练数据中的敏感信息。此前研究主要关注预训练模型的记忆化问题（Carlini et al., 2021; 2023），而对微调阶段和知识蒸馏过程中的记忆化行为研究甚少。

**现有痛点**：(1) 微调通常使用专业化的敏感数据（如医疗记录、专有数据），数据集更小更集中，记忆化风险更高且更危险；(2) 直接微调（SFT）会导致模型高度记忆训练样本；(3) 知识蒸馏作为主流的模型压缩技术，其对记忆化的影响几乎未被探索。

**核心矛盾**：需要在保持任务性能的同时降低模型对训练数据的记忆化程度，尤其在涉及隐私敏感数据的场景中。

**本文目标**：系统研究不同知识蒸馏方法如何影响从微调教师模型到小学生模型传递过程中的记忆化行为。

**切入角度**：将蒸馏视为一种隐式的隐私保护技术，通过比较 SFT、Word-Level KD、Sequence-Level KD 和 RKLD 四种方法的记忆化行为来验证这一假设。

**核心 idea**：蒸馏过程中的"知识软化"天然地过滤了逐字记忆——学生模型学习的是教师的输出分布（软标签），而非直接记忆训练数据的硬标签。

## 方法详解

### 记忆化定义
基于 Carlini et al. (2023) 的框架，适配为指令跟随场景：给定指令-上下文-回复三元组 (p, c, s)，若模型在给定 p 和 c 后通过贪心解码生成的 s' 与 s 完全匹配（逐字复现），则视为记忆化。记忆化比例 = 被记忆样本数 / 总样本数。

### 四种蒸馏方法对比

1. **SFT（监督微调，基线）**：
    - 学生模型直接在 ground-truth 回复上训练，使用标准 next-token loss
    - 无教师指导，直接学习硬标签，记忆化风险最高

2. **Word-Level KD（词级蒸馏）**：
    - 学生模仿教师在每个位置上的 token 级概率分布（软标签）
    - 损失 = KL(教师分布, 学生分布) + NLL 混合
    - 软分布包含了不确定性和替代可能性，比硬标签提供更丰富的监督信号

3. **Seq-KD（序列级蒸馏）**：
    - 用教师模型 beam search 生成的完整序列替代原始 ground-truth 作为训练目标
    - 学生学习的是教师的输出序列，而非原始训练数据
    - 间接地切断了学生与原始数据的直接联系

4. **RKLD（反向 KL 蒸馏/MiniLLM）**：
    - 最小化学生到教师的反向 KL 散度（而非传统的正向 KL）
    - 反向 KL 会惩罚模型在教师分布概率低的地方给出高概率——即惩罚"过度自信"
    - 附加预训练语言模型损失以保持通用能力

### 评估体系
- **记忆化比例**：从训练集随机抽 3000 样本，token 窗口 k=50，计算逐字复现比例
- **ROUGE 分数**：分别在训练集和测试集上计算 ROUGE-1/2/L，训练集高 + 记忆化高意味着逐字复制，测试集反映泛化能力

## 实验关键数据

### 实验设置
- 教师模型：GPT-2 1.5B，在 DollyEval 数据集 10,000 样本上微调
- 学生模型：GPT-2 760M / 340M / 120M
- 测试集：500 样本；记忆化评估：3,000 样本

### 表1：不同蒸馏方法的记忆化比例

| 模型参数量 | SFT | Word-Level KD | Seq-KD | RKLD |
|:---:|:---:|:---:|:---:|:---:|
| 1.5B (Teacher) | 0.654 | — | — | — |
| 760M | 0.523 | 0.472 | 0.315 | 0.090 |
| 340M | 0.433 | 0.140 | 0.134 | 0.075 |
| 120M | 0.330 | 0.100 | 0.129 | 0.060 |

**关键发现**：RKLD 在所有规模上记忆化最低（6.0%–9.0%），相比同规模 SFT 降低 5–7 倍。

### 表2：ROUGE 分数对比（训练集 vs 测试集）

| 模型参数量 | 方法 | R-1 Train | R-1 Test | R-L Train | R-L Test |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1.5B | SFT | 0.88 | 0.33 | 0.78 | 0.27 |
| 760M | SFT | 0.78 | 0.31 | 0.76 | 0.25 |
| 760M | RKLD | 0.45 | 0.36 | 0.40 | 0.30 |
| 340M | SFT | 0.72 | 0.30 | 0.76 | 0.25 |
| 340M | RKLD | 0.57 | 0.34 | 0.53 | 0.28 |
| 120M | SFT | 0.67 | 0.25 | 0.66 | 0.24 |
| 120M | RKLD | 0.46 | 0.30 | 0.42 | 0.21 |

**关键发现**：SFT 的训练集 ROUGE 远高于测试集（过拟合/记忆化），而 RKLD 在训练集和测试集上更均衡，且测试集 ROUGE 表现更优。

## 亮点与创新

1. **首次系统研究蒸馏与记忆化的关系**：揭示了蒸馏的"隐性隐私保护"效应——不是专门设计的隐私机制，但天然降低记忆化。
2. **RKLD 的双重优势**：反向 KL 蒸馏不仅压缩模型，还将记忆化降至最低（120M 模型仅 6%），同时测试集 ROUGE 甚至优于 SFT。
3. **实用洞察**：为在隐私敏感场景（医疗、法律）部署 LLM 提供了一条低成本路径——用蒸馏替代直接微调即可大幅降低数据泄露风险。
4. **清晰的方法论**：基于 Carlini 等人的记忆化量化框架，结合 ROUGE 双视角（训练/测试）分析，评估体系可复用性强。

## 局限性

1. **数据集单一**：仅在 DollyEval 一个数据集上实验，结论的普适性有待在更多领域和数据类型上验证。
2. **模型架构局限**：仅使用 GPT-2 系列，未验证 LLaMA、Mistral 等现代架构是否有相同规律。
3. **记忆化窗口固定**：采用固定 50-token 窗口评估记忆化，不同窗口长度可能影响结论。
4. **隐式记忆化未考虑**：仅关注逐字复现，未分析模型是否以改写形式"隐式"记忆了训练数据。
5. **缺少正式隐私指标**：蒸馏降低记忆化不等于提供可证明的隐私保证（如差分隐私）。

## 相关工作

- **记忆化量化**：Carlini et al. (2021, 2023) 提出从 LLM 提取训练数据的攻击方法，并建立逐字记忆化的量化框架。
- **微调隐私风险**：Yang et al. (2024) 研究领域专用 LLM 的记忆化和隐私风险，确认微调模型更易记忆敏感内容。
- **知识蒸馏**：Hinton et al. (2015) 奠基；Kim & Rush (2016) 提出词级和序列级蒸馏；Gu et al. (2024) 提出 MiniLLM（反向 KL 蒸馏）。
- **隐私泄露检测**：Lukas et al. (2023)、Kim et al. (2023) 分析 LLM 中个人可识别信息的泄露。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|:---|:---:|:---|
| 新颖性 | 3 | 蒸馏和记忆化分别是成熟方向，组合研究有一定新意但不算突破 |
| 技术深度 | 2 | 无新方法提出，主要是现有方法的实验对比 |
| 实验充分性 | 2 | 仅一个数据集 + GPT-2 系列，缺乏多架构和多数据集验证 |
| 写作质量 | 3 | 结构清晰，但内容较薄 |
| 实用价值 | 3 | 为隐私敏感部署提供实用指导，但缺乏大规模验证 |
| **总评** | **2.5** | 有意义的实证研究但实验规模偏小，发现符合直觉但缺乏深入分析 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples](improving_dialogue_state_tracking_through_combinatorial_search_for_in-context_ex.md)
- [\[CVPR 2025\] ViTED: Video Temporal Evidence Distillation](../../CVPR2025/video_understanding/vited_video_temporal_evidence_distillation.md)
- [\[ECCV 2024\] Boosting 3D Single Object Tracking with 2D Matching Distillation and 3D Pre-training](../../ECCV2024/video_understanding/boosting_3d_single_object_tracking_with_2d_matching_distillation_and_3d_pre-trai.md)
- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[CVPR 2025\] Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding](../../CVPR2025/video_understanding/beyond_single-sample_reliable_multi-sample_distillation_for_video_understanding.md)

</div>

<!-- RELATED:END -->
