---
title: >-
  [论文解读] Building Reliable Long-Form Generation via Hallucination Rejection Sampling
description: >-
  [ICML 2026][LLM评测][幻觉缓解] 提出 SHARS 框架，在推理时逐句检测并拒绝幻觉内容、仅保留经验证的事实段落继续生成，配合改进的语义熵检测器 HalluSE，在 FactScore 上将事实精度提升约 20–26%，同时保持甚至增加生成中的事实信息量。
tags:
  - "ICML 2026"
  - "LLM评测"
  - "幻觉缓解"
  - "推理时计算"
  - "语义熵"
  - "拒绝采样"
  - "长文本生成"
---

# Building Reliable Long-Form Generation via Hallucination Rejection Sampling

**会议**: ICML 2026  
**arXiv**: [2606.03628](https://arxiv.org/abs/2606.03628)  
**代码**: https://github.com/TreeLLi/hallucination-rejection-sampling  
**领域**: LLM评测  
**关键词**: 幻觉缓解, 推理时计算, 语义熵, 拒绝采样, 长文本生成  

## 一句话总结

提出 SHARS 框架，在推理时逐句检测并拒绝幻觉内容、仅保留经验证的事实段落继续生成，配合改进的语义熵检测器 HalluSE，在 FactScore 上将事实精度提升约 20–26%，同时保持甚至增加生成中的事实信息量。

## 研究背景与动机

**领域现状**：大语言模型在开放式长文本生成中表现出色，但幻觉问题严重影响可靠性。现有缓解方法主要分为训练时方法（如 DPO 偏好优化、FactAlign 句级奖励）和推理时方法（如 DoLa 层间对比解码、RAG 检索增强）。

**现有痛点**：长文本生成中存在 **幻觉雪球效应**（hallucination snowballing）——早期生成的错误会传播并放大后续输出中的错误。现有推理时方法要么需要外部知识库（RAG），要么仅在 token 级别干预（DoLa），无法有效阻断错误的逐句累积。

**核心矛盾**：开放式问题通常有无限多的有效回答信息，但模型实际只使用其中有限子集。如果能过滤掉幻觉内容并引导模型探索剩余信息空间中的真实内容，就能打破错误链式传播。

**本文目标**：设计一个通用的推理时框架，能够 (1) 逐段检测并拒绝幻觉内容，(2) 仅在已验证的事实基础上继续生成，(3) 不依赖外部知识库即可工作。

**切入角度**：作者观察到推理时计算扩展（inference-time compute scaling）范式在事实性方面尚未被充分探索，且用户在高风险场景中愿意用更多推理时间换取更可靠的输出。

**核心 idea**：用分段拒绝采样（segment-wise rejection sampling）逐句过滤幻觉，仅在已验证的事实上继续生成，从源头阻断幻觉雪球效应。

## 方法详解

### 整体框架

SHARS 的 pipeline 为：给定用户查询 $q$，模型逐句生成文本 → 每生成一句调用幻觉检测器 HalluSE 判断该句事实性 → 根据检测结果决定保留/重写/丢弃 → 在已验证文本基础上继续生成下一句。终止条件为：生成 EOS、达到最大 token 预算、或连续 $N$ 次采样全为幻觉。

### 关键设计

1. **分段拒绝采样（Segment-wise Rejection Sampling）**:

    - 功能：逐句检测并过滤幻觉内容，仅保留事实段落用于后续生成
    - 核心思路：与传统 best-of-N 整体拒绝不同，SHARS 对每一句动态执行拒绝采样。对于当前句，检测器将其分解为事实集合并判定每个事实的可信度。若句中全为幻觉则丢弃；若混合了事实和幻觉则用 LLM 重写，仅保留经验证的事实声明（采用"给正面示例让模型重组"而非"告诉模型删除负面内容"，因实验发现前者在中小模型上表现更好）；若全为事实则直接保留。采样新句时使用 **Following 策略**——暂时保留已识别的幻觉句作为上下文让模型继续生成，利用模型自身的内容规划能力避免重复生成同类知识，同时幻觉句不参与语义熵计算以防止污染检测
    - 设计动机：逐句干预能在错误产生的最早阶段阻断幻觉雪球效应，比整体拒绝更高效且更细粒度

2. **HalluSE 幻觉检测器**:

    - 功能：基于语义熵的长文本幻觉检测方法，改进了朴素语义熵的三个关键缺陷
    - 核心思路：流程为 (1) 将生成文本分解为 (实体, 事实声明) 对，而非仅分解为事实声明——解决了朴素方法中探测实体歧义的问题；(2) 为每个事实生成 $Q$ 个探测问题，通过改进的提示策略确保预期答案无歧义；(3) 对每个问题采样 $A$ 个答案，并显式指示 LLM 提供所有有效答案——解决了多有效答案导致的语义熵虚高问题；(4) 计算各问题的语义熵并取平均，若超过阈值 $\theta$ 则判定为幻觉。语义熵 $H_s = -\sum_i p(C_i) \log p(C_i)$，其中 $C_i$ 为语义聚类，$p(C_i) = \sum_{y \in C_i} p(y)$
    - 设计动机：朴素语义熵方法存在实体探测歧义和多有效答案误报两大问题，导致在长文本场景下检测精度不足

3. **动态弃权机制（Dynamic Abstention）**:

    - 功能：当模型对查询缺乏可靠知识时自动拒绝回答，而非强行编造
    - 核心思路：当连续 $N$ 次采样新句子均被判定为全幻觉时，终止生成。这种弃权可以发生在生成开始时（模型对整个问题缺乏知识），也可以发生在生成中途（模型已输出其确信的部分后停止）。通过调节检测阈值 $\theta$ 可以平滑控制响应率与事实精度之间的 trade-off
    - 设计动机：与其让模型生成不可靠的内容再让用户审核，不如让模型主动识别自身知识边界并适时停止

## 实验关键数据

### 主实验

在 FactScore 基准上的无长度约束评测（Qwen3-32B）：

| 方法 | 响应率(%) | 不支持事实数 | 支持事实数 | 事实精度(%) |
|------|-----------|-------------|-----------|------------|
| Greedy | 99.5 | 8.8 | 9.7 | 52.4 |
| DoLa | 95.6 | 9.3 | 8.2 | 53.1 |
| ChatProtect | 98.9 | 8.1 | 6.8 | 54.4 |
| Self-Endorse | 91.8 | 4.9 | 8.4 | 63.2 |
| **SHARS-Info** | **92.9** | **4.2** | **11.7** | **73.5** |
| **SHARS-Prec** | **82.4** | **3.1** | **11.1** | **78.4** |

FactualBio 幻觉检测评测（Qwen3-32B，Major+Minor）：

| 方法 | AUROC | AURAC | Acc@0.8 | Acc@0.9 |
|------|-------|-------|---------|---------|
| Self-Check | 57.6 | 69.3 | 73.5 | 73.5 |
| P(True) | 69.8 | 73.3 | 70.0 | 70.0 |
| Naive SE | 66.2 | 73.1 | 70.5 | 70.5 |
| **HalluSE** | **72.9** | **77.3** | **75.4** | **72.8** |

### 消融实验

| 采样策略 | 重写 | 响应率(%) | 事实精度(%) | 相对耗时 |
|----------|------|-----------|------------|---------|
| Following | 是 | 91.8 | 69.4 | 1.00× |
| Temperature | 是 | 95.6 | 64.8 | 1.01× |
| Following | 否 | 54.4 | 73.5 | 1.60× |
| Temperature | 否 | 40.1 | 76.2 | 1.55× |

### 关键发现

- **SHARS 框架与检测器均有贡献**：即使将语义熵替换为朴素 token 级熵（Ours-NE），事实精度仍达 70.1%，超过最强 baseline Self-Endorse（63.2%），说明框架本身的分段拒绝策略是有效的
- **与训练时方法互补**：在 FactAlign 基础上叠加 SHARS，事实精度从 53.1% 提升至 80.6%（无长度约束），说明推理时和训练时方法可以协同
- **小模型同样有效**：Qwen3-4B 上获得 +16–24% 精度提升，表明方法不依赖强指令跟随能力
- **重写对响应率至关重要**：关闭重写后响应率从 91.8% 骤降至 54.4%，因为混合型句子被整体丢弃导致大量弃权；Following 策略比 Temperature 策略在支持事实数和精度上均更优

## 亮点与洞察

- **推理时事实性扩展的新范式**：首次系统展示了推理时计算扩展在开放式生成事实性上的 scaling 特性——在合理范围内增加推理计算量可以持续提高事实精度，且效率远优于 Self-Endorse 等方法（相同精度下计算量低 2–3 倍）
- **正面示例重写优于负面删除**：发现让 LLM 根据经验证的事实列表重组句子，比给出原句加标注让其删除幻觉部分效果更好，这一发现在中小模型上尤为显著，对其他 LLM 后处理任务也有借鉴意义
- **知识边界的自发感知**：弃权机制不需要额外的校准，模型在反复采样失败后自然停止，本质上实现了对自身参数化知识边界的感知

## 局限与展望

- 方法不引入外部知识，若模型对某主题完全无知，拒绝采样无法产生新的正确信息，仅能选择弃权
- 推理时计算开销仍较高（约 10–50× Greedy），虽然优于同精度的 baseline，但在延迟敏感场景下仍有挑战
- 当前仅用于英文事实性评测，跨语言和非事实性幻觉（如逻辑不一致）场景尚未验证
- 可改进方向：(1) 与 RAG 结合弥补模型知识盲区；(2) 蒸馏检测器为轻量探针降低开销；(3) 探索更高效的批量句子级检测方法

## 相关工作与启发

- **Semantic Entropy** (Farquhar et al., 2024)：HalluSE 的基础方法，本文针对长文本场景做了实体分解、提示改进和多有效答案三方面改进
- **FactAlign** (Huang & Chen, 2024)：训练时句级事实奖励方法，与 SHARS 正交互补
- **DoLa** (Chuang et al., 2024)：层间对比解码，token 级干预粒度较细但无法阻断句级错误传播
- **Self-Endorse** (Wang et al., 2024)：自一致性验证方法，精度较高但计算开销更大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](../../ACL2025/llm_evaluation/atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ICML 2026\] Automatic Layer Selection for Hallucination Detection](automatic_layer_selection_for_hallucination_detection.md)
- [\[ICML 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets (CAIA)](when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[ICLR 2026\] How Reliable is Language Model Micro-Benchmarking?](../../ICLR2026/llm_evaluation/how_reliable_is_language_model_micro-benchmarking.md)
- [\[ACL 2025\] Pap2Pat: Benchmarking Outline-Guided Long-Text Patent Generation with Patent-Paper Pairs](../../ACL2025/llm_evaluation/pap2pat_benchmarking_outline-guided_long-text_patent_generation_with_patent-pape.md)

</div>

<!-- RELATED:END -->
