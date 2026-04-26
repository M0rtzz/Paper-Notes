---
title: >-
  [论文解读] Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations
description: >-
  [ACL 2026][目标检测][幻觉检测] 本文发现 LLM 内部编码真实性信号存在两条不同的信息通路：Question-Anchored（依赖问题到回答的信息流）和 Answer-Anchored（从生成答案本身提取自包含证据），两者与知识边界紧密关联，并据此提出 Mixture-of-Probes 和 Pathway Reweighting 两种通路感知的幻觉检测方法，AUC 提升达 10%。
tags:
  - ACL 2026
  - 目标检测
  - 幻觉检测
  - 真实性编码
  - 注意力机制
  - 信息通路
  - 知识边界
---

# Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations

**会议**: ACL 2026  
**arXiv**: [2601.07422](https://arxiv.org/abs/2601.07422)  
**代码**: https://github.com/RowanWenLuo/llm-truthfulness-pathways  
**领域**: 幻觉检测/可解释性  
**关键词**: 幻觉检测, 真实性编码, 注意力机制, 信息通路, 知识边界

## 一句话总结

本文发现 LLM 内部编码真实性信号存在两条不同的信息通路：Question-Anchored（依赖问题到回答的信息流）和 Answer-Anchored（从生成答案本身提取自包含证据），两者与知识边界紧密关联，并据此提出 Mixture-of-Probes 和 Pathway Reweighting 两种通路感知的幻觉检测方法，AUC 提升达 10%。

## 研究背景与动机

**领域现状**：LLM 常产生幻觉——看似合理但事实错误的输出。先前工作已证明 LLM 内部表示编码了丰富的真实性信号，可通过线性探测器检测幻觉。但这些信号的来源和工作机制仍不清楚。

**现有痛点**：现有内部探测方法将所有样本视为同质的，使用单一探测器检测所有幻觉。但不同样本的真实性信号可能通过不同机制产生，用统一方法处理会导致次优性能。

**核心矛盾**：显著性分析显示，问题到回答的信息流的重要性呈双峰分布——一部分样本高度依赖问题信息，另一部分几乎不依赖。这暗示存在两种本质不同的真实性编码机制。

**本文目标**：(1) 验证并解耦两条真实性通路；(2) 揭示它们的涌现特性；(3) 利用通路区分提升幻觉检测性能。

**切入角度**：通过注意力敲除（attention knockout）和 token 移植（token patching）两种因果干预实验来解耦和验证两条通路。

**核心 idea**：真实性信号通过两条独立通路产生——Q-Anchored 依赖问题到回答的信息流（适用于模型知识范围内的事实），A-Anchored 从生成文本本身提取自包含证据（适用于知识边界外的长尾事实）。

## 方法详解

### 整体框架

分三阶段：(1) 显著性分析发现双峰分布，提出两条通路假设；(2) 通过注意力敲除和 token 移植验证假设；(3) 探索通路的特性（知识边界关联、自感知能力），并设计通路感知的幻觉检测方法。实验覆盖 12 个模型（base/instruct/reasoning）和 4 个 QA 数据集。

### 关键设计

1. **注意力敲除解耦实验**:

    - 功能：验证两条通路的存在和独立性
    - 核心思路：对于在第 k 层训练的探测器，将 1 到 k 层中从精确问题 token 到后续位置的注意力权重设为 0，阻断问题到回答的信息流。根据探测器预测是否翻转将样本分为 Q-Anchored（翻转）和 A-Anchored（不翻转）。跨所有模型和数据集，两组行为呈明显分叉——一组概率大幅变化，另一组几乎不变。
    - 设计动机：如果真实性信号是同质的，阻断问题信息流应该均匀影响所有样本。双峰行为直接证明了两种不同机制的存在。

2. **通路与知识边界的关联**:

    - 功能：揭示两条通路的认知意义
    - 核心思路：使用三个指标（回答准确率、I-don't-know 率、实体流行度）衡量知识边界。Q-Anchored 样本准确率显著更高、涉及更流行的实体（知识范围内）；A-Anchored 样本准确率低、涉及长尾实体（知识范围外）。这表明：当模型拥有相关知识时主要通过问答信息流编码真实性；当知识不足时，转而从生成文本的内在模式中提取线索。
    - 设计动机：理解通路的认知基础有助于设计更有针对性的检测策略。

3. **通路感知幻觉检测（MoP + PR）**:

    - 功能：利用通路区分提升检测性能
    - 核心思路：(a) Mixture-of-Probes（MoP）：训练多个专家探测器，每个专注于特定的真实性编码机制，利用模型的通路自感知能力（>87% 分类准确率）自动路由到合适的专家；(b) Pathway Reweighting（PR）：根据当前样本属于哪条通路，选择性地增强通路相关的内部信号强度，放大最有信息量的激活维度。两种方法在多个数据集和模型上一致优于单探测器基线。
    - 设计动机：既然两条通路有本质不同的信号来源，用通路特化的检测器比通用检测器更有效。

### 损失函数 / 训练策略

探测器使用二元交叉熵损失训练线性分类器。通路分类器同样在原始内部表示上训练线性探测器，验证模型的自感知能力。

## 实验关键数据

### 主实验

| 方法 | PopQA AUC | TriviaQA AUC | HotpotQA AUC | NQ AUC |
|--------|------|------|----------|------|
| 标准 Probing | 基线 | 基线 | 基线 | 基线 |
| MoP (本文) | +5-10% | +3-8% | +2-5% | +3-7% |
| PR (本文) | 类似提升 | 类似提升 | 类似提升 | 类似提升 |

### 消融实验

| 分析 | 结果 | 说明 |
|------|---------|------|
| 通路自感知准确率 | 75-93% | 模型能从原始表示区分两条通路 |
| Q-Anchored 准确率 | 显著高于 A-Anchored | 知识范围内事实用 Q-Anchored |
| 实体流行度 | Q-Anchored >> A-Anchored | Q-Anchored 涉及高频实体 |
| 随机 token 敲除 | 无显著影响 | 确认效果特异于精确问题 token |

### 关键发现

- **两条通路跨模型跨数据集稳健存在**：从 1B 到 70B，从 base 到 instruct 到 reasoning 模型，双峰模式在所有 12 个模型和 4 个数据集上一致出现。
- **知识边界决定通路选择**：模型"知道答案"时用 Q-Anchored（通过问题理解来判断真实性），"不知道答案"时用 A-Anchored（通过答案本身的统计模式判断）。
- **模型具有通路自感知能力**：内部表示中包含足以区分两条通路的信息，分类准确率 75-93%，这是 MoP 方法的基础。
- **A-Anchored 的"自包含"特性**：移除问题后仅用答案做前向传播，A-Anchored 样本的预测几乎不变，而 Q-Anchored 样本大幅变化。

## 亮点与洞察

- **机制性理解的深度**：不仅证明了两条通路的存在，还揭示了它们与知识边界的关联，提供了认知层面的解释。
- **通路分离的实际应用**：从发现到应用的路径清晰——MoP 和 PR 直接利用机制洞察提升检测性能，不是单纯的分析论文。
- **实验规模**：12 个模型（含最新的 Qwen3）、4 个数据集的全面验证，可信度高。

## 局限与展望

- 目前聚焦于事实性 QA 场景，对开放式生成、多轮对话等场景的通路模式未知。
- 通路自感知准确率并非 100%，错误路由会影响 MoP 性能。
- 未探讨如何通过训练干预来增强特定通路的可靠性。
- 精确 token 的定义依赖语义框架理论，自动化提取可能有噪声。

## 相关工作与启发

- **vs Burns et al. (2023)**: CCS 发现了 LLM 中的线性真实性方向，但未区分信号来源。本文揭示了信号的双通路结构。
- **vs Orgad et al. (2025)**: 他们证明在精确答案 token 上探测效果最好，本文进一步解释了为什么——Q-Anchored 通路的信号集中在精确 token 的信息流中。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次揭示 LLM 真实性编码的双通路结构，发现深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 12个模型4个数据集，因果干预验证严谨
- 写作质量: ⭐⭐⭐⭐⭐ 从假设到验证到应用的叙事逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 对幻觉检测的机制理解和实用改进都有重要贡献

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization](gigacheck_detecting_llm-generated_content_via_object-centric_span_localization.md)
- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)
- [\[CVPR 2026\] Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](../../CVPR2026/object_detection/beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations.md)
- [\[CVPR 2025\] ESCAPE: Equivariant Shape Completion via Anchor Point Encoding](../../CVPR2025/object_detection/escape_equivariant_shape_completion_via_anchor_point_encoding.md)
- [\[ECCV 2024\] Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](../../ECCV2024/object_detection/adaptive_bounding_box_uncertainties_via_twostep_conformal_pr.md)

<!-- RELATED:END -->
