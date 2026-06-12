---
title: >-
  [论文解读] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models
description: >-
  [ACL 2026][幻觉检测][扩散语言模型] 首次系统性地对比扩散大语言模型（dLLM）与自回归（AR）对应模型的幻觉模式，揭示当前 dLLM 幻觉倾向更高，并识别出三种扩散特有的失败模式：过早终止、不完全去噪和上下文入侵。
tags:
  - "ACL 2026"
  - "幻觉检测"
  - "扩散语言模型"
  - "幻觉"
  - "非自回归生成"
  - "失败模式"
  - "推理时计算"
---

# Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models

**会议**: ACL 2026  
**arXiv**: [2604.10556](https://arxiv.org/abs/2604.10556)  
**代码**: [github.com/ZeroLoss-Lab/Lost-in-Diffusion](https://github.com/ZeroLoss-Lab/Lost-in-Diffusion)  
**领域**: 幻觉检测  
**关键词**: 扩散语言模型, 幻觉, 非自回归生成, 失败模式, 推理时计算

## 一句话总结

首次系统性地对比扩散大语言模型（dLLM）与自回归（AR）对应模型的幻觉模式，揭示当前 dLLM 幻觉倾向更高，并识别出三种扩散特有的失败模式：过早终止、不完全去噪和上下文入侵。

## 研究背景与动机

**领域现状**：扩散大语言模型（dLLM）作为非自回归生成范式正迅速崛起，LLaDA、Dream、SDAR 等开源模型在通用 benchmark 上已达到与 AR-LLM 可比的性能水平。理论上，dLLM 的全局规划和双向可见性可以缓解 AR 模型中的"滚雪球效应"和"逆向诅咒"。

**现有痛点**：(1) dLLM 的**可信度**（特别是幻觉问题）几乎未被探索，现有研究主要聚焦于架构优化和推理加速；(2) 扩散过程的随机性可能加剧解码随机性，而随机性本身就是幻觉的已知根因；(3) 缺乏控制变量的公平对比框架。

**核心矛盾**：dLLM 的全局上下文规划理论上应减少幻觉（可回溯修正），但扩散固有噪声可能加剧幻觉——孰对孰错缺乏实证。

**本文目标**：通过严格控制的对比实验回答核心问题：扩散机制究竟是缓解还是加剧了幻觉？

**切入角度**：设计两组精心对照的配对比较——(I) 架构对齐（LLaDA-8B vs LLaMA-3-8B）和 (II) 参数对齐（Dream-7B vs Qwen2.5-7B，Dream 直接从 Qwen 权重初始化），最大化隔离生成机制的影响。

**核心 idea**：dLLM 虽缩小了通用任务的性能差距，但其独特的幻觉机制对模型可靠性构成关键挑战，需要动态序列编辑能力来实现非自回归生成的全部潜力。

## 方法详解

### 整体框架

构建配对比较框架，通过两组对照实验隔离生成范式的影响。使用 HalluLens 基准评估外在幻觉（Extrinsic Hallucination），涵盖精确知识召回、长文本事实一致性和知识边界检测三个维度。

### 关键设计

1. **配对比较框架**：Group I 为架构对齐——LLaDA-8B vs LLaMA-3-8B，两者共享相似架构和参数规模且通用性能可比；Group II 为参数对齐——Dream-7B vs Qwen2.5-7B，Dream 直接从 Qwen 权重初始化，任何幻觉差异可主要归因于扩散生成过程。优先使用预训练（非指令微调）检查点以隔离后训练噪声。设计动机是最大化消除训练数据和模型容量的混淆因素。

2. **标准扩散推理设置**：采用规范扩散设置（canonical diffusion），去噪步数 $T$ 设为等于序列长度 $L$（$T=L$），最大化模型的迭代精化能力。temperature 设为 0 确保可复现性。LLaDA 使用高置信度解码，Dream 使用最小熵解码。设计动机是充分表征 dLLM 的原生生成行为，不使用半自回归或块级加速方法。

3. **推理时计算动态分析**：在 LongWiki 任务上评估不同去噪步数 $T \in \{128, 256, 512, 1024\}$ 的影响，揭示两种 dLLM 的截然不同行为。LLaDA 因准自回归解码（线性噪声调度器 + 高置信度解码强制近似从左到右生成）导致**早期饱和**；Dream 因最小熵解码实现真正的非顺序精化，展现**正向扩展**特性。设计动机是验证 dLLM 理论上的"以计算换质量"能力是否在实践中成立。

### 损失函数 / 训练策略

本文是分析性工作，不涉及模型训练。评估使用 HalluLens 的自动 LLM 评估器，并通过分层子集上的人工标注验证评估器可靠性。排除了内在幻觉任务（如摘要），因为这些任务严重依赖指令遵循能力，会引入混淆因素。

## 实验关键数据

### 主实验

| 模型 | PreciseWikiQA HR ↓ | PreciseWikiQA CR ↑ | LongWiki F1@32 ↑ | NonExistRefusal FA ↓ |
|------|-------------------|-------------------|------------------|---------------------|
| LLaMA-3-8B (AR) | 85.94 | 10.30 | 0.306 | 73.35 |
| LLaDA-8B (dLLM) | **95.13** | **3.92** | 0.272 | **87.10** |
| Qwen2.5-7B (AR) | 89.06 | 9.06 | **0.387** | 94.05 |
| Dream-7B (dLLM) | 92.54 | 6.04 | 0.340 | **98.50** |

### 扩散特有失败模式频率（人工标注 200 例）

| 模型 | 过早终止 (PT) | 不完全去噪 (ID) | 上下文入侵 (CI) |
|------|-------------|----------------|----------------|
| LLaDA-8B | 18.0% | **60.0%** | 38.0% |
| Dream-7B | 13.0% | 44.0% | **58.0%** |

### 关键发现

- **dLLM 在所有三项任务上一致劣于 AR 对应模型**：在精确知识召回中，LLaDA-8B 的正确率仅 3.92%（vs LLaMA-3-8B 10.30%）；在非存在实体拒绝中，Dream-7B 的误接受率高达 98.50%
- **推理时计算的分化动态**：LLaDA 的 F1@32 在所有步数上停滞在 ~0.27（早期饱和），而 Dream 从 128 步到 1024 步单调递增，展现正向扩展
- 早期饱和归因于 LLaDA 的准自回归生成顺序——虽理论上有双向可见性，但实际被迫近似从左到右生成
- **三种扩散特有失败模式**的发现极具启发性：
    - **过早终止**：独立去噪的片段无法语法对齐，模型被迫插入 EOS 或断裂分隔符
    - **不完全去噪**：面对罕见实体时后部序列锚定在无意义 token 上，双向注意力试图合理化连接导致整体崩溃
    - **上下文入侵**：偶尔去噪出高频 token（数字、代码关键词），双向注意力被迫构建通向该伪锚点的逻辑路径，劫持原始查询

## 亮点与洞察

- **首次系统性量化 dLLM 的幻觉问题**，填补了一个重要的研究空白——dLLM 社区此前几乎只关注性能追赶而忽视可靠性
- 配对比较框架设计严谨，特别是 Dream-Qwen 配对（共享初始化权重）提供了近乎理想的控制变量
- **三种失败模式的分类法**为理解 dLLM 的生成行为提供了有价值的词汇和框架
- 推理时计算的分化动态揭示了解码策略（而非架构本身）对 dLLM 行为的关键影响

## 局限与展望

- 无法完全隔离生成范式的影响——即使最轻量的扩散适配也需要权重更新，可能引入微妙的混淆因素
- 使用规范扩散设置（$T=L$），实际部署中的加速方法可能改变幻觉特征
- 排除了指令微调模型和内在幻觉任务，评估覆盖范围受限
- 仅评估了 7-8B 规模模型，更大规模 dLLM 的幻觉行为可能不同
- 未来方向：动态序列编辑能力（插入、删除、re-masking）是 dLLM 实现其全部潜力的关键

## 相关工作与启发

- **滚雪球效应 & 逆向诅咒**：AR-LLM 的已知幻觉机制在 dLLM 中并未被完全消除，而是转化为新形式——早期去噪错误被"固化"
- **Re-masking 方法**（LLaDA2.1 等）正在尝试解决 token 级错误，与本文发现的失败模式直接相关
- **HalluLens 基准**：为 dLLM 幻觉评估提供了标准化工具，但需要扩展以覆盖更多任务类型
- 启示：dLLM 要实现可靠生成，核心挑战不是去噪精度而是**全局一致性维护**——当独立去噪的片段冲突时，缺乏编辑机制是致命弱点

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次系统性对比 dLLM 幻觉，三种失败模式的发现极具原创性和启发性
- **实验充分度**: ⭐⭐⭐⭐ 配对比较框架设计严谨，含人工标注验证，但模型和任务覆盖范围可更广
- **写作质量**: ⭐⭐⭐⭐⭐ 问题定义精确，分析深入，失败模式描述清晰生动，Limitation 部分诚恳且有建设性
- **价值**: ⭐⭐⭐⭐⭐ 对 dLLM 社区敲响可靠性警钟，三种失败模式为后续改进指明方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](../../CVPR2026/hallucination/fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[ACL 2026\] Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](mechanisms_of_prompt-induced_hallucination_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
