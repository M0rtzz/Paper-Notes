---
title: >-
  [论文解读] WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging
description: >-
  [ACL 2025][LLM评测][多选题基准] 提出 WiCkeD 方法，通过随机将多选题的一个选项替换为"以上都不对"来增加现有基准难度，使18个 LLM 的平均性能下降12.1个百分点，且链式思维推理也无法弥补这一下降。 多选题（MCQ）基准是评估 LLM 的主流方式，但面临一系列挑战： 基准饱和：MMLU、Arc-c…
tags:
  - "ACL 2025"
  - "LLM评测"
  - "多选题基准"
  - "None of the above"
  - "基准饱和"
  - "推理评估"
  - "Wild-Card Distractor"
---

# WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging

**会议**: ACL 2025  
**arXiv**: [2502.18316](https://arxiv.org/abs/2502.18316)  
**代码**: [github.com/ahmedselhady/wicked-benchmarks](https://github.com/ahmedselhady/wicked-benchmarks)  
**领域**: LLM评测  
**关键词**: 多选题基准, None of the above, 基准饱和, 推理评估, Wild-Card Distractor

## 一句话总结

提出 WiCkeD 方法，通过随机将多选题的一个选项替换为"以上都不对"来增加现有基准难度，使18个 LLM 的平均性能下降12.1个百分点，且链式思维推理也无法弥补这一下降。

## 研究背景与动机

多选题（MCQ）基准是评估 LLM 的主流方式，但面临一系列挑战：

**基准饱和**：MMLU、Arc-challenge 等主流基准在 LLM 时代迅速被"做满"，难以区分模型能力差异

**构建新基准成本高**：创建高质量基准需要大量人工标注和验证

**现有增强方法有限**：MMLU-Pro 通过增加干扰项数量来提高难度，但创建合理干扰项本身就困难，且常需人工验证

已有研究发现 MCQ 评估存在固有缺陷：
- 某些 LLM 仅看选项（不看问题）就能正确作答
- LLM 对特定答案键（A/B/C/D）存在先验偏好
- LLM 无法有效识别自己缺乏知识的情况

教育测评领域长期使用"以上都不对"（None of the Above, NOTA）选项来增加测试难度、鼓励深度思考。这启发了 WiCkeD 方法。

## 方法详解

### 整体框架

WiCkeD（Wild-Card Distractor）是一种自动化方法，可将任意 MCQ 基准转化为更具挑战性的版本：
- 保持问题不变
- 随机替换一个选项为 "None of the above"
- 不增加选项数量（替换而非追加）

核心假设：检测正确答案不在选项中，比从现有选项中选择正确答案更难。

### 关键设计

**WiCkeD 算法**：
给定包含 M 个样本、每个样本 N 个选项（1个正确 + N-1个干扰项）的基准：
1. 均匀采样一个选项进行替换
2. 追加 "None of the above" 选项
3. 若正确选项被替换，新的正确答案为 "None of the above"
4. 若干扰项被替换，原正确答案保持不变

**SBA/SCA 分类处理**：
并非所有替换都能产生一致的 WiCkeD 样本。当存在"最佳答案"（Single Best Answer, SBA）而非"唯一正确答案"（Single Correct Answer, SCA）时：
- 移除最佳答案后，次佳答案变为正确，但算法会错误标注 NOTA 为正确
- 解决方案：训练 BERT 分类器检测 SBA 样本（召回率 98.9%），SBA 样本原封不动复制到 WiCkeD 版本

**分类器训练**：
- 从4个基准（MMLU、MMLU-Pro、TruthfulQA、CommonsenseQA）采样4000个样本
- GPT-4o-mini 自动标注 SBA/SCA
- 75%用于训练 BERT 分类器，25%用于评估（含人工标注）
- SBA 比例：MMLU/MMLU-Redux/MMLU-Pro 约20%，其他基准 <5%

**随机性处理**：
- 由于选项替换是随机的，为每个基准生成5个 WiCkeD 变体
- 报告均值和标准差

### 损失函数 / 训练策略

WiCkeD 本身不涉及模型训练。SBA/SCA 检测的 BERT 分类器使用标准的二分类交叉熵损失。

## 实验关键数据

### 主实验

**6个基准上18个模型的 WiCkeD 性能下降**：

| 模型 | 原始平均 | WiCkeD 平均 | 下降 Δ |
|------|---------|-----------|--------|
| Qwen-2.5-72B | 84.6 | 72.6 | -12.0 |
| Qwen-2.5-72B-IT | 82.6 | 69.3 | -13.3 |
| Llama-3.1-70B-IT | 77.1 | 64.5 | -12.6 |
| Qwen-2.5-7B | 74.7 | 54.9 | **-19.7** |
| DS-R1-Qwen7B | 60.8 | 53.4 | **-7.3** |
| 总平均 | 70.78 | 58.52 | -12.2 |

关键发现：
- 所有模型均出现显著性能下降（7.2-19.7个百分点）
- **Qwen-2.5-7B 下降最大（19.7%）**，而其 DeepSeek-R1 蒸馏版仅下降 7.3%
- WiCkeD **重排了模型排名**：Qwen-2.5-7B 原本接近 Llama-3.1-70B，但在 WiCkeD 上落后13%

**WiCkeD 揭示了模型间隐藏的能力差异**：
- Gemma-2-9B-IT 和 Gemma-2-27B-IT 在 WiCkeD 上分别落后 Llama-3.1-70B 9.5% 和 5.3%
- 指令微调没有一致优势，效果因模型家族而异

### 消融实验

**链式思维（CoT）实验**（MMLU、MMLU-Pro、MMLU-Redux）：

| 模型 | 直接 WiCkeD Δ | CoT WiCkeD Δ |
|------|-------------|-------------|
| DS-R1-Llama 8B | -4.1 | -2.0 |
| DS-R1-Qwen 7B | -4.3 | -2.5 |
| Llama-3.1-8B | -3.2 | -5.8 |
| Qwen-2.5-7B | -6.9 | -14.6 |
| Gemma-2-9B | -12.2 | -8.9 |
| 平均 | -5.62 | -5.24 |

关键观察：
- CoT 无法消除 WiCkeD 带来的性能下降——平均仍有约5%的下降
- **推理增强模型（DeepSeek-R1 蒸馏版）受影响最小**（~2%），验证了 WiCkeD 确实测试推理能力
- 指令微调 + CoT 时，Qwen-2.5-7B-IT 和 14B-IT 下降不到2%
- 基座模型在 CoT 下下降反而可能更大（如 Llama-3.1-8B 从-3.2到-5.8）

### 关键发现

1. WiCkeD 是一种零成本的基准增强方法——完全自动化，无需人工标注
2. 性能下降范围大（7-20%），有效应对基准饱和问题
3. WiCkeD 暴露了原始基准掩盖的模型差异——特别是推理能力的差异
4. 推理能力更强的模型（如 DeepSeek-R1）受 WiCkeD 影响最小
5. "检测正确答案缺失"确实是一种独立的、有价值的评估维度

## 亮点与洞察

1. **极简而深刻**：仅一个简单操作（替换为 NOTA）就暴露了 LLM 评估中的深层问题
2. **教育测评的启发**：NOTA 在教育领域是成熟的评估技术，成功迁移到 LLM 评估场景
3. **SBA 检测机制**：细致处理了"多个正确答案"的边界情况，保证了 WiCkeD 变体的内在一致性
4. **对推理能力的精准测试**：WiCkeD 下降幅度与模型推理能力表现出清晰的负相关
5. **可组合性**：可叠加到任何 MCQ 基准上，与其他增强方法（如增加干扰项）正交

## 局限与展望

1. "None of the above" 的位置总是最后一个选项，可能引入位置偏差
2. SBA 检测依赖 GPT-4o-mini 标注，存在约1.1%的噪声
3. 仅实验了6个基准和18个模型，可扩展到更多场景
4. 未探索"以上都对"（All of the above）等其他 wild-card 选项
5. 教育测评研究显示 NOTA 可能降低考生信心——类似效应是否存在于 LLM 有待研究
6. 未分析模型在哪些类型的问题上对 WiCkeD 更敏感

## 相关工作与启发

- **MMLU-Pro (Wang et al., 2024)**：通过增加干扰项数量提高难度，但需人工创建合理干扰项
- **MMLU-Redux (Gema et al., 2024)**：修正 MMLU 中的错误问题并重新标注
- **Balepur et al. (2024)**：发现 LLM 仅看选项就能答题——WiCkeD 一定程度上抵消了这种"捷径"
- **DiBattista & Fortuna (2014)**：教育测评中 NOTA 的研究——WiCkeD 将这一理念引入 AI 评估
- 启发：评估方法的创新有时比构建全新基准更高效、更有洞察力

## 评分

- **创新性**：⭐⭐⭐⭐ — 将教育测评技术简洁迁移到 LLM 评估，idea 小而精妙
- **实用性**：⭐⭐⭐⭐⭐ — 零成本自动化，可立即应用于任何 MCQ 基准
- **实验充分性**：⭐⭐⭐⭐ — 18模型×6基准+CoT分析，5次随机种子
- **写作质量**：⭐⭐⭐⭐ — 动机清晰，SBA处理细致，实验展示得当

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging](financereasoning_benchmarking_financial_numerical_reasoning_more.md)
- [\[ACL 2025\] Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice QA](right_answer_wrong_score_uncovering_the_inconsistencies_of_llm_evaluation_in_mul.md)
- [\[ACL 2026\] BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks](../../ACL2026/llm_evaluation/benchmarker_an_education-inspired_toolkit_for_highlighting_flaws_in_multiple-cho.md)
- [\[ACL 2025\] CoPrUS: Consistency Preserving Utterance Synthesis Towards More Realistic Benchmark](coprus_consistency_preserving_utterance_synthesis_towards_more_realistic_benchma.md)
- [\[ACL 2025\] CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming](culturalbench_a_robust_diverse_and_challenging_cultural_benchmark_by_human-ai_cu.md)

</div>

<!-- RELATED:END -->
