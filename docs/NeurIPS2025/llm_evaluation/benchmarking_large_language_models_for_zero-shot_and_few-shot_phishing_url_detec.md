---
title: >-
  [论文解读] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection
description: >-
  [NeurIPS 2025][LLM评测] 在统一的零样本和少样本 prompt 框架下系统评估 GPT-4o、Claude-3.7 和 Grok-3-Beta 三个商用 LLM 在钓鱼 URL 检测任务上的表现，发现少样本 prompt 可显著提升所有模型性能，Grok-3-Beta 在平衡数据集上取得最佳 F1（0.9399），但不同模型在精度-召回率权衡上呈现差异化行为模式。
tags:
  - NeurIPS 2025
  - LLM评测
  - LLM
  - 零样本学习
  - 少样本学习
  - 提示学习
  - 网络安全benchmark
---

# Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection

**会议**: NeurIPS 2025  
**arXiv**: [2602.02641](https://arxiv.org/abs/2602.02641)  
**代码**: 有  
**领域**: 网络安全 / LLM评估  
**关键词**: 钓鱼URL检测, LLM, 零样本学习, 少样本学习, prompt工程, 网络安全benchmark

## 一句话总结

在统一的零样本和少样本 prompt 框架下系统评估 GPT-4o、Claude-3.7 和 Grok-3-Beta 三个商用 LLM 在钓鱼 URL 检测任务上的表现，发现少样本 prompt 可显著提升所有模型性能，Grok-3-Beta 在平衡数据集上取得最佳 F1（0.9399），但不同模型在精度-召回率权衡上呈现差异化行为模式。

## 研究背景与动机

**威胁升级**：自 2022 年以来钓鱼攻击数量激增超过 4000%，生成式 AI 被网络犯罪者广泛利用制造上下文感知的钓鱼网站和 URL，传统检测手段捉襟见肘。

**传统方法局限**：黑名单和基于手工特征的 ML 分类器（决策树、SVM、集成模型）对新型钓鱼 URL 泛化能力差，严重依赖已知模式和标注数据。

**深度学习进展不足**：CNN/RNN 等深度模型虽能自动学习 URL 序列特征，但仍需大量有监督训练，缺乏零样本推理能力。

**LLM 潜力未被系统评估**：BERT 微调（DomURLs_BERT）和 LoRA 适配（PhishURLDetect）虽展示了语言模型在安全领域的可行性，但缺乏对商用 LLM 在 prompt-only 推理下的系统对比。

**已有评估的缺陷**：先前工作（Nasution et al.）仅评估了 21 个开源 LLM 且限于英文 prompt，未涵盖商用模型；评估条件不一致导致结论可比性差。

**标注数据滞后**：钓鱼策略演变速度远快于标注数据的产出速度，零样本/少样本学习成为应对快速演变威胁的务实选择。

## 方法详解

### 评估框架设计

本文构建了统一的 prompt-based 分类框架，核心包含三个组件：

1. **任务指令 $\mathcal{I}$**：设定角色为网络安全专家，要求仅输出 0（钓鱼）或 1（合法）的二值判断。
2. **查询 $\mathcal{Q}(u)$**：将目标 URL 嵌入标准化查询模板。
3. **示例集 $\mathcal{E}$**（仅少样本）：提供若干带标签的 URL 示例，少样本 prompt 格式为 $\mathcal{P}_{\text{FS}}(u) = \mathcal{I} \oplus \mathcal{E} \oplus \mathcal{Q}(u)$。

### 模型与数据配置

- **评估模型**：GPT-4o（OpenAI）、Claude-3.7-sonnet-20250219（Anthropic）、Grok-3-Beta（xAI），均通过官方 API 调用，temperature=0，max tokens=10。
- **平衡数据集**：从 PhiUSIIL 数据集随机采样 10,000 条 URL（钓鱼/合法各 5,000），零样本全量评估，少样本抽取 6 个示例（3+3）后在剩余 9,994 条上评估。
- **不平衡数据集**：构建 1,000 条测试集，钓鱼比例分别为 1% 和 10%，使用双随机种子（S123/S456）验证稳定性。
- **少样本变体**：在 10% 不平衡条件下测试 $|\mathcal{E}| \in \{1, 3, 9\}$ 的效果。

### 评估指标

采用六项指标全面评价：Accuracy、Macro-Precision、Macro-Recall、Macro-F1、AUROC、AUPRC，所有指标按类别宏平均计算。

## 实验关键数据

### 表1：平衡数据集（10,000 URL）性能对比

| 模型 | 设置 | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|------|------|----------|-----------|--------|------|-------|-------|
| GPT-4o | 零样本 | 0.8752 | 0.8421 | 0.9232 | 0.8808 | 0.8752 | 0.9018 |
| GPT-4o | 少样本 | 0.9050 | 0.8880 | 0.9270 | 0.9071 | 0.9050 | 0.9258 |
| Claude-3.7 | 零样本 | 0.8759 | 0.8778 | 0.8734 | 0.8756 | 0.8759 | 0.9072 |
| Claude-3.7 | 少样本 | 0.9250 | 0.9027 | **0.9526** | 0.9270 | 0.9250 | 0.9395 |
| Grok-3-Beta | 零样本 | 0.8914 | 0.8361 | 0.9735 | 0.8996 | 0.8914 | 0.9114 |
| Grok-3-Beta | 少样本 | **0.9405** | **0.9492** | 0.9307 | **0.9399** | **0.9405** | **0.9573** |

### 表2：不平衡数据集少样本（10% 钓鱼，S123）F1 表现

| 模型 | 零样本 | $\mathcal{E}$=1 | $\mathcal{E}$=3 | $\mathcal{E}$=9 |
|------|--------|-----------------|-----------------|-----------------|
| GPT-4o | 0.785 | 0.709 | 0.801 | 0.861 |
| Claude-3.7 | 0.761 | 0.857 | 0.842 | 0.876 |
| Grok-3-Beta | **0.854** | **0.906** | 0.821 | 0.831 |

### 关键发现

- **少样本一致提升**：6 个示例将所有模型的 F1 提升 3-6 个百分点（平衡集），AUROC 和 AUPRC 同步改善。
- **精度-召回权衡差异**：Grok-3-Beta 从零样本到少样本 Recall 下降（0.9735→0.9307）但 Precision 大幅上升（0.8361→0.9492），表明少样本使模型决策阈值更严格。
- **不平衡场景行为分化**：Grok-3-Beta 仅需 1 个示例即达 F1=0.906 但更多示例反而退化；GPT-4o 则随示例数单调提升；Claude-3.7 呈现先降后升的非单调模式。
- **跨种子稳定性**：双随机种子下 F1 差异均小于 0.05，验证了评估结论的鲁棒性。

## 亮点与洞察

- **统一评估框架**：首次在标准化条件下对比三大商用 LLM 的钓鱼检测能力，确保了模型间的公平可比性。
- **细粒度行为分析**：揭示了不同 LLM 在少样本学习中截然不同的行为模式（单调/非单调/过拟合），这对实际部署中的模型选择具有直接指导意义。
- **低标注成本方案**：仅需 1-6 个标注示例即可获得显著性能提升，展示了 LLM 在标注数据匮乏的安全场景中的实用价值。
- **可复现性**：开源代码和标准化数据集使得结果可复现。

## 局限与展望

- **模型覆盖有限**：仅评估三个商用模型，未包含开源 LLM（如 Llama-3、Mistral），无法评估开源替代方案的性价比。
- **数据集单一**：仅使用 PhiUSIIL 数据集，未在多个钓鱼数据集上验证泛化性，不同数据源的 URL 分布可能影响结论。
- **缺乏对抗分析**：未测试 LLM 对抗性钓鱼 URL（如 Unicode 混淆、同形异义字攻击）的鲁棒性。
- **时效性问题**：钓鱼 URL 快速进化，当前 benchmark 的有效期有限，需要持续更新机制。
- **成本分析缺失**：未对比不同模型的 API 调用成本和延迟，在实际部署中这些是关键因素。

## 与相关工作的对比

| 维度 | 本文 | Nasution et al. (2025) | PhishURLDetect (Ali & Subba, 2025) |
|------|------|------------------------|-------------------------------------|
| 模型类型 | 商用 LLM (GPT-4o/Claude/Grok) | 21 个开源 LLM | 单个微调 LLM (LoRA) |
| 评估范式 | Prompt-only 推理 | Prompt 工程 | 参数高效微调 |
| 数据条件 | 平衡+不平衡+多种子 | 仅英文 prompt | 单一数据集 |
| 少样本分析 | 系统对比 1/3/6/9-shot | 未涉及 | 不适用（微调范式） |
| 行为分析深度 | 精度-召回权衡+非单调模式 | 基础指标对比 | 有限分析 |

| 维度 | 本文 | DomURLs_BERT (Mahdaouy et al., 2024) |
|------|------|--------------------------------------|
| 训练需求 | 无需训练 | 需要微调 |
| 最佳 F1 | 0.9399 (Grok few-shot) | 较高（全监督） |
| 隐私/部署 | API 调用、数据不离开本地 | 需本地训练数据 |

## 评分

- 新颖性: ⭐⭐⭐ 系统评估框架有价值，但方法层面无创新，核心贡献在于实验设计和行为分析
- 实验充分度: ⭐⭐⭐⭐ 平衡/不平衡、零样本/少样本、多种子验证，指标体系完整
- 写作质量: ⭐⭐⭐⭐ 实验设计描述清晰，prompt 构造透明，结果呈现规范
- 价值: ⭐⭐⭐ 对安全领域 LLM 应用有参考价值，行为差异分析有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Few-Shot Pattern Detection via Template Matching and Regression](../../ICCV2025/llm_evaluation/few-shot_pattern_detection_via_template_matching_and_regression.md)
- [\[ICML 2025\] Feedforward Few-shot Species Range Estimation](../../ICML2025/llm_evaluation/feedforward_few-shot_species_range_estimation.md)
- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](../../ICCV2025/llm_evaluation/a_conditional_probability_framework_for_compositional_zerosh.md)
- [\[ACL 2025\] Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance](../../ACL2025/llm_evaluation/language_complexity_measurement_as_a_noisy_zero-shot_proxy_for_evaluating_llm_pe.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](unlocking_transfer_learning_for_open-world_few-shot_recognition.md)

</div>

<!-- RELATED:END -->
