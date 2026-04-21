---
title: >-
  [论文解读] The First Impression Problem: Internal Bias Triggers Overthinking in Reasoning Models
description: >-
  [ICLR 2026 (Poster)][LLM推理][过度思考] 推理模型在看到问题的瞬间就形成了对答案的"第一印象"（内部偏差），当这个直觉猜测与后续系统推理产生冲突时，模型会反复自我质疑、重新检查，导致推理长度膨胀 21%–43%，而现有所有缓解方法均无法从根本上消除这一效应。
tags:
  - ICLR 2026 (Poster)
  - LLM推理
  - 过度思考
  - 内部偏差
  - 推理模型
  - 因果干预
  - 注意力机制
---

# The First Impression Problem: Internal Bias Triggers Overthinking in Reasoning Models

**会议**: ICLR 2026 (Poster)  
**arXiv**: [2505.16448](https://arxiv.org/abs/2505.16448)  
**作者**: Renfei Dang, Zhening Li, Shujian Huang, Jiajun Chen（南京大学）  
**代码**: 无  
**领域**: LLM 推理 (LLM Reasoning)  
**关键词**: 过度思考, 内部偏差, 推理模型, 因果干预, 注意力机制

## 一句话总结

推理模型在看到问题的瞬间就形成了对答案的"第一印象"（内部偏差），当这个直觉猜测与后续系统推理产生冲突时，模型会反复自我质疑、重新检查，导致推理长度膨胀 21%–43%，而现有所有缓解方法均无法从根本上消除这一效应。

## 研究背景与动机

以 DeepSeek-R1 和 OpenAI o1 为代表的推理模型通过内部 Chain-of-Thought（CoT）进行自我反思和纠错，在复杂推理任务上取得了突破性表现。但它们有一个突出的效率瓶颈——**过度思考（overthinking）**：生成大量冗余的推理步骤（如反复"Wait..."、"Let me re-check..."），既浪费计算资源又不能提升准确率，有时甚至降低最终答案质量。

已有研究主要从**外部行为**层面描述 overthinking（比如推理 token 数与问题难度不匹配），但没有人解释**为什么模型会陷入这种无意义的反思循环**。本文借鉴认知心理学中的**锚定效应（anchoring effect）**，提出了一个全新的因果解释——**"第一印象问题"（The First Impression Problem）**：

- 模型在收到输入问题后、`<think>` 推理正式开始**之前**，其隐状态中已经编码了一个关于答案的初步猜测（称为 **internal bias**）
- 这种"第一印象"不一定被显式输出，但会持久存在于模型的中间层表示中
- 当这个初始猜测恰好与后续推理得出的结论**冲突**时，模型无法"放手"，反复回看原题、反复自我修正，产生大量冗余反思
- 极端情况下，模型甚至会陷入**鹦鹉循环（parroting）**——不断重复相同推理步骤却永远无法收敛到最终答案

## 方法详解

### 整体框架

本文的研究框架采用递进式分析结构：**量化偏差 → 验证关联 → 因果干预 → 机制解释 → 缓解尝试**，每一步都不可或缺。

### 关键设计

1. **内部偏差的量化方法**

    作者通过 **Direct Answer** 方法提取模型的内部偏差：给模型加上 "Answer without thinking more:" 等强制直答提示，在其 `<think>` 推理块启动之前截获输出，得到直觉答案 $a_{bias}$。为消除采样噪声，对每个问题采样 64 次，构建内部偏差分布 $\tilde{a}_{bias}$。然后定义**偏差偏离度（Deviation Degree）** $D_{bias}$ 来度量内部偏差与最终推理答案 $a_{final}$ 之间的差距：

    - 数值型任务：$D_{bias} = \text{mean}(|a_{bias} - a_{final}|)$（MAE）
    - 选择题任务：$D_{bias}$ = 不一致率（初始猜测与最终答案不同的比例）

2. **关联性验证**

    在 DeepSeek-R1-671B、QwQ-32B 等多个推理模型上，横跨 AIME、KnowLogic 等多种推理 benchmark 进行统计分析。核心发现：随着 $D_{bias}$ 增大，推理长度显著增加，**相对长度增幅 $R_\Delta$ 在 21.0%–43.1% 之间**。更关键的是，通过测量模型**首次得出完整答案的位置（First Answer Position）**发现，无论偏差高低，模型到达第一个逻辑结论的时间点基本相同——多出来的 token 几乎**全部**是结论之后的冗余反思。

3. **反事实因果干预**

    为从相关性上升到因果性，设计了两个互补的反事实实验：

    - **干预一：移除输入问题（Question Removal）**。在模型生成第一个完整答案后，立即从上下文中删除原始输入问题，迫使模型只基于自己已生成的推理链来决定是否继续反思。定义冗余缩减比 $r = (L_{ori} - L_{rem}) / (L_{ori} - P_{first})$。结果：在 AIME 2024 上冗余 token **减少了 53.5%**，同时准确率基本不变甚至略有提升
    - **干预二：偏差注入（Bias Injection）**。通过 LoRA 微调，分别将错误偏差注入到模型原本轻松解决的简单题（Low2Wrong），以及将正确偏差注入到原本困难题（High2Correct）。结果：注入错误偏差让简单题开始产生大量冗余反思；注入正确偏差让困难题的 overthinking 显著减少。这从**正反两个方向**确认了因果关系

4. **注意力机制分析**

    通过可解释性实验揭示内部偏差的具体作用机制：

    - **注意力动力学**：在模型触发反思（即将生成 "Wait..." 或 "Let me check..."）的瞬间，对原始输入问题的注意力权重飙升至正常推理时的 **4 倍以上**。这说明模型通过反复"回看"原题来重新激活初始偏差
    - **Logit Lens 探测**：对于模型最终正确推理出 $a_{final} = 3$ 但内部偏差为 $a_{bias} = 5$ 的样本，在中间到后半段层中，偏差答案 "5" 的 internal decoding 概率**始终高于**正确答案 "3"——即使模型在推理链中已经显式写出了 "3"。这揭示了一种持久的**认知失调状态**：直觉猜测与系统推理在模型内部并行存在，互相拉扯

## 实验关键数据

### 偏差-overthinking 关联实验

在多模型、多任务上验证 $D_{bias}$ 与推理长度的关系：

| 模型 | 任务 | 低偏差组推理长度 | 高偏差组推理长度 | 相对增幅 $R_\Delta$ |
|------|------|:---------:|:---------:|:---------:|
| DeepSeek-R1-671B | AIME | 基准 | 显著增长 | 21.0%–43.1% |
| QwQ-32B | AIME | 基准 | 显著增长 | 范围内 |
| DeepSeek-R1 蒸馏版 | KnowLogic | 基准 | 显著增长 | 范围内 |
| 多模型一致 | 多任务 | 基准 | 显著增长 | **均 >20%** |

关键观察：高偏差组与低偏差组到达**第一个完整答案**的位置基本相同，多出来的 token 全部集中在首答之后的冗余反思区间。

### 反事实干预实验结果

| 干预方法 | 具体操作 | 冗余推理变化 | 准确率影响 |
|---------|---------|:---------:|:---------:|
| Question Removal | 首答后删除原题 | **减少 53.5%**（AIME 2024） | 保持/略升 |
| Low2Wrong 注入 | LoRA 注入错误偏差到简单题 | **显著增加** | 下降 |
| High2Correct 注入 | LoRA 注入正确偏差到困难题 | **显著减少** | 提升 |

### 现有缓解方法评估

| 方法 | 类型 | 能否降低推理长度 | 能否消除偏差影响 ($R_\Delta$) | 准确率影响 |
|------|------|:---------:|:---------:|:---------:|
| FCS (SFT+DPO) | 训练时 | 能降低平均长度 | 否，$R_\Delta$ 未下降甚至恶化 | 复杂任务可能下降 |
| SEAL | 推理时 | 能缩短 | 否，偏差影响持续 | 基本保持 |
| PROBE | 推理时 | 能缩短 | 否，偏差影响持续 | 基本保持 |
| 注意力早退出（本文提出） | 推理时 | 能缩短 | **部分有效**：$R_\Delta$ 从 31.5% → 9.4% | 精度损失微小 |

核心结论：FCS / SEAL / PROBE 等方法本质上只是"截断"推理链，并没有解决内部偏差与推理冲突的根对立问题。唯一有初步效果的是本文提出的**注意力早退出机制**——监控模型对原始问题的归一化注意力，当超过阈值（意味着模型正在重新激活偏差）时直接终止推理。

## 亮点与洞察

1. **从认知心理学到 LLM 机制的完整映射**：将"锚定效应"和 Kahneman 的 System 1/System 2 理论映射到推理模型的行为模式上，发现模型确实存在类似人类的快速直觉（System 1）与慢速推理（System 2）的冲突。这不仅是一个诊断工具，更开辟了用认知科学理论理解 LLM 行为的新路径。

2. **因果推断设计的严谨性**：不满足于简单的相关性分析，通过两种互补的反事实干预（移除偏差源 + 注入偏差）从正反两个方向确立因果关系。特别是 LoRA 偏差注入实验的设计非常精巧：Low2Wrong 和 High2Correct 分别控制了问题难度和偏差方向两个混杂变量。

3. **"负面结果"的重大价值**：系统性地证明 FCS、SEAL、PROBE 等现有方法都无法消除内部偏差，这一发现本身具有重要指导意义——告诉社区表面的推理截断治标不治本，需要从模型架构或训练范式层面寻找根本解决方案。

4. **Logit Lens 揭示的"认知失调"**：在模型已经显式推理出正确答案的情况下，错误的初始猜测在中间层仍保持更高的 decoding 概率，这是一个极其有趣的发现，揭示了推理模型内部"直觉"与"逻辑"并行竞争的动力学。

## 局限与展望

1. **缺乏根本性解决方案**：虽然注意力早退出机制有初步效果（$R_\Delta$ 从 31.5% 降至 9.4%），但这仍然是一种推理时的外部干预，而非让模型本身学会忽略错误直觉。未来可探索在 RL 训练阶段加入"偏差解耦"目标。

2. **模型覆盖有限**：主要在 DeepSeek-R1 和 Qwen 系列的开源推理模型上实验，对 o1、Claude 等闭源商业模型的适用性未知。不同训练范式（纯 RL vs. SFT+RL）下偏差形成机制可能不同。

3. **任务范围聚焦有确定答案的推理**：实验集中在数学（AIME）、逻辑（KnowLogic）等有明确正确答案的任务上。对于开放式生成、创意写作等任务中的 overthinking 尚未探索，偏差的定义和量化在这些场景下也需要重新设计。

4. **偏差提取方法的完备性**："Answer without thinking more" 的直答提示可能无法完整捕获模型隐状态中所有形式的偏差信息。更refined 的探测方法（如训练 probing classifier 直接读取隐状态）或许能获得更精确的偏差估计。

5. **未来方向**：设计显式的 System 1 / System 2 推理架构，让快速直觉通路与慢速推理通路在模型内部解耦；或研究注意力正则化技术减少推理过程中对原始输入的过度关注。

## 相关工作与启发

- **Overthinking 分析**：Chen et al. (2024) "Do not think that much for 2+3=?" 首次系统描述了 o1 类模型的 overthinking 现象，本文在此基础上深入到内部机制层面
- **推理不忠实性**：Arcuschin et al. (2025) 发现 CoT 推理在自然场景下并非总是忠实的，最终答案可能由隐含偏差塑造而非推理链决定，与本文发现相互印证
- **Circuit Tracing**：Anthropic 的研究表明语言模型可能存在独立的"快速估算"神经通路，为本文"内部偏差"的存在提供了机制层面的佐证
- **推理效率优化**：SEAL（隐状态引导）和 PROBE（置信度探测早停）等方法被本文证明只能治标不治本
- 启发方向：**认知科学驱动的 LLM 行为分析**是一条值得深耕的研究路线，System 1/System 2 的区分可能启发全新的推理模型架构设计

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （"第一印象问题"是一个极具原创性的发现，认知心理学与 LLM 机制分析的桥接非常新颖）
- 实验充分度: ⭐⭐⭐⭐⭐ （关联分析 + 双向因果干预 + 注意力/Logit Lens 可解释性 + 缓解方法评估 + 新方案探索，实验链条完整且严谨）
- 写作质量: ⭐⭐⭐⭐ （概念阐述清晰，"第一印象"类比直觉友好，递进式实验设计的叙事逻辑流畅）
- 价值: ⭐⭐⭐⭐⭐ （对推理模型 overthinking 的根因理解具有里程碑意义，对未来模型设计和训练范式有直接指导价值）

<!-- RELATED:START -->

## 相关论文

- [Adversarial Manipulation of Reasoning Models using Internal Representations](../../ICML2025/llm_reasoning/adversarial_manipulation_of_reasoning_models_using_internal_representations.md)
- [The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning](../../NeurIPS2025/llm_reasoning/the_virtues_of_brevity_avoid_overthinking_in_parallel_test-time_reasoning.md)
- [BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models](../../AAAI2026/llm_reasoning/badthink_triggered_overthinking_attacks_on_chain-of-thought_reasoning_in_large_l.md)
- [The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](../../NeurIPS2025/llm_reasoning/the_illusion_of_thinking_understanding_the_strengths_and_limitations_of_reasonin.md)
- [On The Fragility of Benchmark Contamination Detection in Reasoning Models](on_the_fragility_of_benchmark_contamination_detection_in_reasoning_models.md)

<!-- RELATED:END -->
