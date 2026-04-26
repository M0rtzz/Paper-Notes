---
title: >-
  [论文解读] Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation
description: >-
  [ACL 2025][幻觉缓解] 提出 Monitoring Decoding (MD) 框架，在生成过程中动态监控部分响应的事实性，通过监控函数识别易产生幻觉的 token 并利用树搜索策略选择性地修正这些关键 token，从而在保持效率的同时显著提升事实准确性。
tags:
  - ACL 2025
  - 幻觉缓解
  - 解码策略
  - 监控函数
  - 树搜索
  - 部分响应评估
---

# Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation

**会议**: ACL 2025  
**arXiv**: [2503.03106](https://arxiv.org/abs/2503.03106)  
**代码**: 无  
**领域**: NLP / LLM 幻觉缓解  
**关键词**: 幻觉缓解, 解码策略, 监控函数, 树搜索, 部分响应评估

## 一句话总结

提出 Monitoring Decoding (MD) 框架，在生成过程中动态监控部分响应的事实性，通过监控函数识别易产生幻觉的 token 并利用树搜索策略选择性地修正这些关键 token，从而在保持效率的同时显著提升事实准确性。

## 研究背景与动机

大语言模型在问答、摘要和推理等任务中表现出色，但仍然容易产生幻觉——生成看似合理但事实上不正确的内容。现有的幻觉缓解方法面临以下问题：

**全长采样效率低**：Best-of-N (BoN) 策略和自一致性方法需要生成多个完整响应，引入显著的延迟开销。

**过度自信问题**：模型对幻觉 token 可能表现出极高的置信度，导致多次采样仍然生成相同的错误输出。自一致性高并不等于事实正确。

**关键发现**：作者观察到，通常只有少数关键 token 导致幻觉，只需替换这些 token（如将 "24" 替换为 "It"）就可以将错误响应转变为正确响应。这意味着不需要重新采样整个响应。

核心问题：**是否有必要重新采样多个高度相似的全长响应来提高事实性？** 答案是否定的，针对性的 token 级干预即可。

## 方法详解

### 整体框架

MD 框架包含两个核心组件：
- **过程内检测机制 (In-process Detection)**：在生成过程中持续监控每 m 个新生成 token 的事实性
- **树搜索修正机制 (Tree-based Revision)**：对被标记的可疑 token 进行树搜索式重采样和修剪

流程：输入 prompt → 模型每次生成 m 个 token → 监控函数评估 → 若通过则保留继续生成 → 若检测到幻觉风险则触发树搜索修正 → 选择最佳路径继续。

### 关键设计

1. **监控函数 (Monitor Function)**：

    - 核心思路：利用一个更大的参考模型 $f^*$ 来评估目标模型 $f_\theta$ 生成 token 的可信度
    - 计算方式：加权比率 $r_\beta = \sum_{s=1}^{m} w_s^t \cdot \frac{p^*(y_s^t | \mathbf{y}^{<t}, y_{<s}^t)}{p_\theta(y_s^t | \mathbf{y}^{<t}, y_{<s}^t)}$
    - 设计动机：幻觉但过度自信的 token 在目标模型中概率高但在参考模型中概率低，导致该比率较低
    - 权重设计：$w_s^t = 1/|(\mathbf{y}^{<t}, y_{<s}^t)|$，越早的 token 权重越大，因为它们为后续生成奠定基础

2. **带拒绝的生成 (Generation with Rejection)**：

    - 接受概率：$p(\text{accept } \mathbf{y}^t) = \min\{1, r_\beta(\mathbf{y}^t)\}$
    - 自适应阈值：$\gamma^t = \gamma_0 \sum_{s=1}^{m} w_s^t$，其中 $\gamma_0 \in [0,1]$
    - 若接受概率超过阈值则保留，否则触发修正

3. **树搜索修正 (Tree-based Revision)**：

    - 功能：对被拒绝的 m 个 token 逐个进行树搜索式重新生成
    - 核心思路：每步采样 Top-N 个候选 token，用监控函数剪枝保留 Top-K 路径，逐层扩展直到 m 步
    - 设计动机：平衡探索空间和计算效率——不像全长采样那样冗余，也不像贪心解码那样单一

### 损失函数 / 训练策略

MD 是一个 **无需训练** 的推理时框架，不涉及额外训练：
- 目标模型直接使用（如 Llama-2-7B-chat）
- 参考模型选择同架构的更大模型（如 Llama-2-70B-Chat）
- 超参数：采样数 N=2，搜索深度 K=3

## 实验关键数据

### 主实验（表格）

| 模型 | 方法 | TruthfulQA (T×I%) | TriviaQA (EM) | NQ-Open (EM) | GSM8K (Acc) |
|------|------|-------------------|---------------|--------------|-------------|
| Llama-2 | Greedy | 37.9 | 64.8 | 36.6 | 24.2 |
| Llama-2 | USC | 39.4 | 66.8 | 38.6 | 23.4 |
| Llama-2 | **MD** | **44.1 (+6.2)** | **72.1 (+7.6)** | **40.5 (+3.7)** | **27.5 (+3.3)** |
| Llama-3 | Greedy | 42.4 | 72.4 | 39.6 | 81.4 |
| Llama-3 | **MD** | **46.1 (+3.7)** | **80.8 (+8.4)** | **47.4 (+6.8)** | **85.2 (+3.8)** |
| Gemma-2 | Greedy | 43.6 | 54.0 | 23.0 | 60.9 |
| Gemma-2 | **MD** | **50.2 (+6.6)** | **64.6 (+10.6)** | **31.0 (+8.0)** | **79.9 (+19.0)** |

### 效率对比（表格）

| 方法 | 延迟 (ms/token) | 吞吐量 (token/s) |
|------|-----------------|------------------|
| Greedy | 19.94 (×1.00) | 50.68 (×1.00) |
| USC | 245.76 (×12.32) | 4.06 (×0.08) |
| FSC | 316.72 (×15.88) | 3.15 (×0.06) |
| **MD** | **113.78 (×5.70)** | **18.99 (×0.37)** |

### 消融实验（表格）

| 采样数 N | TriviaQA EM |
|----------|-------------|
| 1 (=Greedy) | 64.8 |
| 2 | ~70 |
| 4 | ~71 |
| 6+ | 趋于稳定（~72） |

- 阈值 $\gamma_0$ 从 0 到正值都能稳定提升，方法对该参数鲁棒

### 关键发现

1. MD 在 Gemma-2 上提升最为显著——TriviaQA 提升 10.6%，GSM8K 提升 19.0%，说明对较小模型效果更好
2. 基线方法效果不稳定：DoLa 在推理任务上甚至降低性能（GSM8K -7.7%），ID 在 Llama-2 上 GSM8K -13.3%
3. MD 延迟仅为 USC 的约一半，吞吐量是 USC 的 4.7 倍，效率优势明显
4. 案例研究表明 MD 能精准定位关键幻觉 token，仅修改少量 token 即可纠正整体响应

## 亮点与洞察

1. **粒度洞察**：不是所有 token 都需要修正——大部分"容易" token 跨采样一致，只有少量"困难" token 导致幻觉。这一发现将幻觉缓解从响应级细化到 token 级
2. **过度自信的本质**：模型对幻觉 token 的过度自信使得自一致性策略失效。MD 通过引入外部参考模型绕过了这一问题
3. **效率与效果的平衡**：选择性 token 重采样 + 树搜索剪枝实现了比全长采样更好的效果和更低的开销

## 局限与展望

1. **依赖参考模型**：需要同架构的更大参考模型（如 70B），在实际部署中增加资源需求
2. **知识覆盖**：若训练数据中不存在的事实信息，监控函数也无法检测。可通过引入外部知识库缓解
3. **固定窗口 m**：每次监控 m 个 token，m 的选择可能影响性能，论文未充分探讨最优 m 的设置
4. **可扩展性**：树搜索的 N 和 K 参数如何随任务复杂度调整尚不清楚

## 相关工作与启发

- 与 DoLa（层间对比解码）互补：DoLa 利用模型内部层间信息，MD 利用外部参考模型信息
- SelfCheckGPT 的后验检测思想与 MD 的过程内检测形成对比，启发性地说明了 "在哪个阶段干预" 的重要性
- 与推测解码 (Speculative Decoding) 共享 "小模型生成、大模型验证" 的范式，但动机完全不同

## 评分

- **新颖性**: ⭐⭐⭐⭐ — token 级监控 + 树搜索修正的组合是自然但有效的创新，从 "全长采样" 到 "选择性 token 修正" 的范式转变有意义
- **实验充分度**: ⭐⭐⭐⭐ — 3 个模型、4 个数据集、效率分析、消融实验、案例研究，较为全面
- **写作质量**: ⭐⭐⭐⭐ — 动机阐述清晰（尤其是 Figure 1 的对比），方法描述层次分明
- **价值**: ⭐⭐⭐⭐ — 作为推理时幻觉缓解方案实用性强，但需要额外大模型是部署障碍

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](odysseus_dynamic_focus_decoding.md)
- [\[ACL 2025\] How Does Response Length Affect Long-Form Factuality](how_does_response_length_affect_long-form_factuality.md)
- [\[ICCV 2025\] ChartCap: Mitigating Hallucination of Dense Chart Captioning](../../ICCV2025/llm_safety/chartcap_mitigating_hallucination_of_dense_chart_captioning.md)
- [\[ACL 2025\] Beyond Facts: Evaluating Intent Hallucination in Large Language Models](intent_hallucination_eval.md)
- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)

<!-- RELATED:END -->
