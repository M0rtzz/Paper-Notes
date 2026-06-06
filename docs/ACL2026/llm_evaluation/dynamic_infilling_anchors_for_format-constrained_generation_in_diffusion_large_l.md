---
title: >-
  [论文解读] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models
description: >-
  [ACL 2026][LLM评测][扩散模型] DIA 是一种无需训练的扩散大语言模型格式约束生成方法，通过先预测结束锚点位置再在锚点间迭代填充，显著提升 reasoning template 和 JSON 输出的格式正确率，并缓解固定锚点导致的截断或冗余。
tags:
  - "ACL 2026"
  - "LLM评测"
  - "扩散模型"
  - "格式约束"
  - "动态锚点"
  - "结构化生成"
  - "JSON 生成"
---

# Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models

**会议**: ACL 2026  
**arXiv**: [2606.04535](https://arxiv.org/abs/2606.04535)  
**代码**: https://github.com/Westlake-AGI-Lab/DIA  
**领域**: 扩散语言模型 / 格式约束生成  
**关键词**: diffusion LLM、格式约束、动态锚点、结构化生成、JSON 生成

## 一句话总结
DIA 是一种无需训练的扩散大语言模型格式约束生成方法，通过先预测结束锚点位置再在锚点间迭代填充，显著提升 reasoning template 和 JSON 输出的格式正确率，并缓解固定锚点导致的截断或冗余。

## 研究背景与动机
**领域现状**：diffusion large language models 与自回归 LLM 不同，使用双向注意力和并行去噪生成，天然可以在初始全 mask 序列中预填某些固定 token。因此，它们看起来很适合做结构化输出，例如 `<think>...</think><answer>...</answer>` 或 parseable JSON。

**现有痛点**：直接把 begin/end anchors 放在固定位置虽然能约束格式，但会把生成空间切成固定长度。如果 reasoning span 太短，模型会提前截断；如果 span 太长，模型会重复或生成冗余内容。prompt 约束、后处理和 constrained decoding 也各有问题：prompt 不稳定，后处理可能破坏语义，严格解码又影响效率和灵活性。

**核心矛盾**：格式约束需要结构边界稳定，但高质量生成需要长度可变。固定模板把这两件事绑定在一起，导致结构正确和语义质量难以兼得。

**本文目标**：作者希望利用 dLLM 对全局 mask 序列和结束位置的感知能力，在不微调模型的情况下动态估计 anchor 位置，让模型先规划每个结构段需要多长，再生成段内内容。

**切入角度**：论文观察到 dLLM 可以通过一两步预测估计 eos 或结束位置。因此，结束锚点不必预先写死，可以通过 single-step prediction 和 confidence threshold 动态寻找。

**核心 idea**：把格式约束生成拆成“长度调整”和“锚点内填充”两个阶段：先反复扩展 mask block 直到模型高置信预测出 end anchor，再固定 anchor 并在边界内完成扩散式内容生成。

## 方法详解
DIA 面向已经训练好的 diffusion LLM，不引入新参数。它把输出序列划分为若干 block，每个 block 对应一个结构段，例如 reasoning 段和 answer 段。每段先插入 begin anchor，然后通过一次预测判断 end anchor 可能出现在哪里；如果当前位置空间不足，就继续扩展 block，直到找到高置信 end anchor 或达到最大长度。

### 整体框架
给定用户 query $Q$ 和全 mask 目标序列 $X_L$，DIA 先把 $X_L$ 分成若干 blocks $\mathcal{C}=\{C_1,\dots,C_{|\mathcal{B}|}\}$，并在每个 block 开头放入 begin anchor。Stage 1 对每个 block 执行 length adjustment：调用 dLLM 一步预测，搜索 end anchor 或 partial anchor，如果置信度超过阈值 $c$ 就截断 block 并补全 end anchor，否则按步长 $\Delta$ 扩展。Stage 2 在确定边界后执行 iterative denoising with infilling，在固定 anchor 之间生成具体 reasoning 或 answer 内容。

### 关键设计
1. **动态结束锚点预测**:

    - 功能：为每个结构段自动估计合适长度，而不是手工固定 span。
    - 核心思路：在 block 前放入 begin anchor 后进行 single-step prediction。如果模型在 block 内预测出 end anchor 或 partial end anchor，且置信度超过阈值，就认为当前长度足够；如果没有，则继续添加 $\Delta$ 个 mask tokens。
    - 设计动机：dLLM 在预训练中学到了回答终止位置的先验。利用这个先验可以在真正生成内容前规划边界，避免固定 anchor 太早或太晚。

2. **左侧优先和最大长度约束**:

    - 功能：避免重复 end anchors 和无限扩展。
    - 核心思路：当多个位置都满足置信度阈值时，DIA 选择最靠左的位置作为 end anchor，截断其后的冗余 mask；若一直找不到有效 anchor，则在最大 block length $M$ 处停止扩展。
    - 设计动机：结构化生成最怕模型在同一段中反复开启/关闭标签。左侧优先提供保守边界，最大长度则防止复杂样本带来不可控延迟。

3. **顺序式锚点内迭代填充**:

    - 功能：在已确定结构边界内生成语义连贯内容。
    - 核心思路：对于 think-answer 任务，DIA 先确定并生成 thinking block，再利用已经生成的 reasoning 内容决定 answering block 的长度和内容。生成过程中 anchors 保持固定，只有中间 mask tokens 被迭代更新。
    - 设计动机：answer 的长度和内容依赖 reasoning。如果两个 block 完全独立规划，最终答案可能与推理脱节；顺序处理能把前一段信息传递给后一段。

### 损失函数 / 训练策略
DIA 是 training-free 方法，没有新增训练损失。实验使用 Dream-7B-Base-v0 和 Dream-7B-Instruct-v0，在官方代码基础上修改实现。GSM8K 使用 1,319 个样本，MATH 使用 5,000 个样本；GSM8K max new tokens 为 256，MATH 为 512；confidence threshold 分别为 0.065 和 0.05，expand size $\Delta=4$，max block length $M=512$，diffusion steps 为 512，batch size 分别为 1 和 3。实验环境为 PyTorch 2.5.1、Python 3.10、NVIDIA vGPU 32G/48G。

## 实验关键数据

### 主实验
| 数据集 | 指标 | DIA | 之前方法 | 提升 / 说明 |
|--------|------|------|----------|------|
| GSM8K 0-shot | Format Score | 72.63 | Infilling: 58.83，Base/Instruct: 0.00 | 格式正确率显著提升 |
| GSM8K 0-shot | Accuracy | 46.78 | Infilling: 14.86，Instruct: 15.01，Base: 68.99 | 相比固定 infilling 大幅提升，但低于无格式 Base |
| MATH-500 0-shot | Format Score | 76.82 | Infilling: 29.10，Base/Instruct: 0.00 | 复杂数学场景格式收益最大 |
| MATH-500 0-shot | Accuracy | 20.08 | Infilling: 21.52，Base: 25.14，Instruct: 25.28 | 保持可比但不是最高准确率 |
| WikiBio JSON | Valid JSON / Hallucination | 79.84 / 0.15 | Instruct raw: 52.80 / 4.81，Infilling: 0.01 / 0.00 | 原始匹配和正则提取下结果一致 |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| DIA w/o Stage 1, GSM8K | Acc. 10.31，Format 0.00，Latency 14.99 | 去掉 confidence prediction 后格式几乎崩溃 |
| DIA Full, GSM8K | Acc. 47.54，Format 59.67，Latency 25.86 | 附录超参设置下完整方法明显更好 |
| DIA w/o Stage 1, MATH | Acc. 6.73，Format 0.84，Latency 15.33 | 复杂任务更依赖长度规划 |
| DIA Full, MATH | Acc. 20.20，Format 75.62，Latency 29.37 | 完整两阶段方法保留结构边界 |
| GSM8K latency | DIA 26.52 vs Base 10.72 | 动态规划带来额外延迟 |
| MATH latency | DIA 30.62 vs Base 31.71 | 复杂任务上前置长度规划反而减少冗余计算 |

### 关键发现
- 固定 infilling 能部分保留 anchors，但不能保证整体格式正确，也会压低答案质量。GSM8K 上固定 infilling accuracy 只有 14.86，而 DIA 达到 46.78。
- DIA 在 JSON 生成上效果最稳定。WikiBio 中 raw matching 和 regular expression 都得到 79.84% valid JSON，且 hallucination score 只有 0.15%。
- anchor retention 分析显示 DIA 在 GSM8K 和 MATH 上几乎能稳定保留 `<think>`、`</think>`、`<answer>`、`</answer>` 四类 anchors；Base 和 Instruct 模型的结束锚点保留率会严重下降。
- 超参分析表明 $\Delta$ 是格式严格性和推理深度之间的旋钮。较小 $\Delta$ 可能过早截断，格式率高但 accuracy 低；较大 $\Delta$ 给 reasoning 更多空间，但可能降低 format score 或增加 latency。

## 亮点与洞察
- DIA 把格式控制从“事后修 JSON / 事后修标签”提前到“生成前规划边界”，这很符合 diffusion LLM 的双向和并行生成特性。
- 方法 training-free，适合快速部署在已有 dLLM 上；这比为每种输出 schema 微调模型更轻量。
- 论文很清楚地展示了固定 anchor 的本质问题：不是没有结构 token，而是 token 位置刚性导致内容空间不匹配。
- 对结构化 reasoning 的启发是，模型不只需要知道输出什么格式，还需要知道每个格式段应该有多少生成预算。

## 局限与展望
- DIA 仍依赖人工指定 anchors 及其语义角色。对于结构边界本身会动态变化的开放域对话、多轮工具调用或创意写作，手工 anchor 可能不够灵活。
- 长度调整会引入额外推理开销。GSM8K 上 DIA latency 为 26.52，明显高于 Base 10.72，不适合所有实时场景。
- Accuracy 与 format 存在 trade-off。DIA 大幅提升格式正确率，但在 MATH 上 accuracy 低于无格式 Base/Instruct 和固定 infilling。
- 当前评测集中在 reasoning template 和 WikiBio JSON。未来需要验证更复杂 schema、代码、证明、多模态结构化输出和嵌套工具调用。

## 相关工作与启发
- **vs prompt-based constraints**: 仅靠提示词要求输出 JSON 或标签，在长推理中容易丢边界；DIA 直接在初始 mask 序列中控制 anchors，更贴近解码过程。
- **vs post-processing / repair**: 后处理可以修格式，但可能改变语义或丢失 reasoning。DIA 在生成前规划结构，减少后处理依赖。
- **vs constrained decoding**: grammar 或 FSM decoding 约束强但不灵活，DIA 通过动态长度分配保留一定生成自由度。
- **对 diffusion LLM 的启发**: dLLM 的优势不只是并行加速，还包括可以先规划全局结构再填充内容，这可能是区别于 AR LLM 的重要应用方向。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 动态锚点位置预测很贴合 dLLM 机制，training-free 设计也实用。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖 GSM8K、MATH、WikiBio、stage ablation、超参和 latency，但任务类型仍偏有限。
- 写作质量: ⭐⭐⭐⭐☆ 动机清晰、算法直观，个别主表和附录表的数值设置不同，需要读者区分实验条件。
- 价值: ⭐⭐⭐⭐☆ 对结构化输出、格式约束 reasoning 和 dLLM 解码设计有较强启发，尤其适合需要 parseable output 的应用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)

</div>

<!-- RELATED:END -->
