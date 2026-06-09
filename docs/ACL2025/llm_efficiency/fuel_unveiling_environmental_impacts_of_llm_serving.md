---
title: >-
  [论文解读] FUEL: Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View
description: >-
  [ACL 2025][LLM效率][碳排放] 提出 FUEL 框架，首次引入生命周期评估中的"功能单元"（Functional Unit）概念作为标准化比较基准，在统一的质量、性能和工作负载约束下评估不同 LLM 服务配置的碳排放，通过模型大小、量化策略和硬件选择三个案例研究揭示了多个反直觉的绿色 AI 洞察…
tags:
  - "ACL 2025"
  - "LLM效率"
  - "碳排放"
  - "LLM服务"
  - "功能单元"
  - "量化"
  - "硬件选择"
---

# FUEL: Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View

**会议**: ACL 2025  
**arXiv**: [2502.11256](https://arxiv.org/abs/2502.11256)  
**代码**: [https://github.com/jojacola/FUEL](https://github.com/jojacola/FUEL)  
**领域**: LLM效率 / 绿色AI  
**关键词**: 碳排放, LLM服务, 功能单元, 量化, 硬件选择

## 一句话总结

提出 FUEL 框架，首次引入生命周期评估中的"功能单元"（Functional Unit）概念作为标准化比较基准，在统一的质量、性能和工作负载约束下评估不同 LLM 服务配置的碳排放，通过模型大小、量化策略和硬件选择三个案例研究揭示了多个反直觉的绿色 AI 洞察。

## 研究背景与动机

**领域现状**：LLM 服务带来显著的环境影响——ChatGPT 处理单个 prompt 产生超过 4 克 CO2eq，是普通搜索查询的 20 倍以上。已有研究（如 LLMCarbon、LLMCO2、GreenLLM）通过建模和 profiling 分析 LLM 的碳排放。

**现有痛点**：现有研究有两个关键局限：（1）聚焦于单个 LLM 而非跨模型比较；（2）缺乏标准化的碳排放比较基准——不同模型在不同质量、延迟和吞吐量条件下的碳排放无法公平对比。仅看每 token 的碳排放忽略了输出质量差异。

**核心矛盾**：一个小模型可能每 token 碳排放低，但如果需要更多 token 才能达到相同质量，总碳排放反而更高。缺乏统一的比较基准导致"哪个配置更绿"这个问题无法严谨回答。

**本文目标**：建立一个考虑质量和性能约束的标准化评估框架，在公平条件下比较不同 LLM 服务配置的碳排放。

**切入角度**：借鉴环境可持续性领域的生命周期评估（LCA）方法论，引入"功能单元"概念——不是比较"每个 token 的碳排放"，而是比较"每个满足约束条件的 token 的碳排放"。

**核心 idea**：将 LLM 服务中的功能单元定义为满足特定工作负载强度、性能约束（TTFT ≤ 1s，TPOT ≤ 200ms）和质量约束（Qscore 达标）的 token，在此基础上计算碳排放强度 CFU。

## 方法详解

### 整体框架

FUEL 框架分四步：（1）输入定义——模型、比较配置（大小/量化/硬件）和服务约束；（2）定义功能单元——基于 QPS、TTFT/TPOT 性能约束和 Qscore 质量约束；（3）Profiling——运行 LLM 收集性能和能耗数据；（4）碳建模——计算每功能单元碳排放 CFU，包含运营碳和体现碳。

### 关键设计

1. **功能单元定义（Functional Unit）**:

    - 功能：建立跨模型碳排放比较的标准化基准
    - 核心思路：一个功能单元代表一个在生成过程中满足服务约束的 token。$N_f = \sum_{i=1}^N \mathbb{I}(Q_i \geq \alpha) \cdot \mathbb{I}(TTFT_i \leq \beta) \cdot \mathbb{I}(TPOT_i \leq \gamma)$，CFU = 总碳排放 / $N_f$。质量用 Skywork reward model 的 Qscore 评估，性能约束为 TTFT ≤ 1s 和 TPOT ≤ 200ms（对齐人类阅读速度）
    - 设计动机：直接比较每 token 碳排放会误导——小模型单 token 便宜但质量差时 token 浪费更多。功能单元将质量和性能纳入考量，使比较更公平

2. **碳排放建模（Operational + Embodied）**:

    - 功能：全面计算 LLM 服务的碳足迹
    - 核心思路：总碳 = 运营碳 + 体现碳。运营碳 $C_{op} = E_{op} \times CI$（能耗 × 碳强度），使用 pynvml/psutil 每 200ms 采样功率。体现碳 $C_{em} = (t / LT) \times C_{em,total}$（运行时间占硬件寿命比例 × 硬件全生命周期碳排放），用 ACT 工具计算
    - 设计动机：仅看运营碳会忽略新硬件制造带来的巨大碳足迹——H100 的体现碳约 29.92 kgCO2eq，而 L40 为 26.6 kgCO2eq

3. **质量评估方法（Reward Model as Quality Evaluator）**:

    - 功能：量化评估 LLM 输出质量作为功能单元的质量约束
    - 核心思路：采用开源的 Skywork reward model（在 RewardBench 上排名靠前），计算每个响应的 Qscore 作为质量度量。高 Qscore 表示输出质量更高、更符合人类偏好
    - 设计动机：传统质量评估依赖特定数据集或参考答案，而 reward model 能跨任务一致地评估质量，且与人类偏好对齐良好

### 损失函数 / 训练策略

非训练型工作。实验使用 vLLM 服务平台，碳强度取 518 gCO2eq/kWh（服务器所在地区 12 个月平均值），温度设为 0。

## 实验关键数据

### 主实验：模型大小案例

| 配置 | 低质量(-5) QPS=1 | 高质量(15) QPS=1 |
|------|-------------------|-------------------|
| Qwen 7B | 最低碳 | 最高碳（1.8× 32B） |
| Qwen 14B | 中等 | 中等 |
| Qwen 32B | 最高碳 | **最低碳（节省40%+）** |

### 量化案例

| 方法 | 描述 | 是否总是更绿？ |
|------|------|----------------|
| AWQ (权重量化) | TPOT 在低 QPS 有加速，TTFT 始终更慢 | **否**——大模型和高 QPS 下反而增加碳排放 |
| W8A8 (激活量化) | TPOT 和 TTFT 均持续加速 | **是**——在所有场景下均持续降低碳排放 |

### 硬件案例

| 硬件 | L40 (2022) | H100 (2023) |
|------|------------|-------------|
| GPU TDP | 300W | 700W |
| 体现碳(GPU) | 26.6 kgCO2eq | 29.92 kgCO2eq |
| 低 QPS 碳效率 | **更优** | 更差 |
| 高 QPS 碳效率 | 更差 | **更优** |

### 关键发现

- **小模型不总是更绿**：当输出质量要求高时，大模型反而碳效率更高，因为小模型满足质量约束的 token 比例更低
- **量化的碳效果取决于类型**：权重量化（AWQ）在某些场景下反而增加碳排放（因为推理时需要反量化回 16-bit），而激活量化（W8A8）始终有效
- **新硬件不总是更绿**：H100 性能强但功耗高、体现碳大，在低 QPS 下老硬件 L40 碳效率更优
- **不存在"万能最绿"配置**：最优选择取决于 QPS、质量要求和模型大小的组合

## 亮点与洞察

- **功能单元的引入是核心创新**：将环境科学中的 LCA 方法论迁移到 AI 领域，解决了"如何公平比较不同配置碳排放"的根本问题。这个框架可以推广到所有 AI 服务的环境影响评估
- **反直觉发现极具实践价值**："小模型更绿"、"新硬件更绿"、"量化更绿"这些直觉都被证明是有条件的，为实际部署决策提供了量化依据
- **体现碳的重要性被揭示**：Intel Xeon 8480+ 的体现碳（42.81 kgCO2eq）是 AMD EPYC 7443（9.98 kgCO2eq）的 4 倍以上，这种差异在仅看运营碳时完全被忽略

## 局限与展望

- 仅测试了 Qwen2.5 和 Llama2 两个模型家族，未覆盖多模态/代码模型
- 实验限于单 GPU，未探索多 GPU 分布式环境的通信开销和碳排放
- 质量评估依赖 reward model，更先进的质量评估方法可能改变结论
- 未来可探索碳感知的自适应服务策略——根据实时碳强度和负载动态选择模型配置

## 相关工作与启发

- **vs LLMCarbon/LLMCO2**: 这些框架提供端到端碳建模但缺乏标准化比较基准，FUEL 通过功能单元解决了跨模型可比性问题
- **vs GreenLLM/Sprout**: 这些工作基于 profiling 优化碳排放但未考虑质量约束，可能导致"减碳但降质"的次优决策

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 功能单元概念的引入是该领域的范式转变
- 实验充分度: ⭐⭐⭐⭐ 三个案例研究覆盖全面，但模型和硬件种类有限
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，每个案例以三个问题驱动，逻辑性强
- 价值: ⭐⭐⭐⭐⭐ 对绿色 AI 部署有直接的实践指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)
- [\[AAAI 2026\] Scaling and Transferability of Annealing Strategies in Large Language Model Training](../../AAAI2026/llm_efficiency/scaling_and_transferability_of_annealing_strategies_in_large_language_model_trai.md)
- [\[ACL 2025\] A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding in Large Language Models](a_drop-in_solution_for_on-the-fly_adaptation_of_speculative_decoding_in_large_la.md)
- [\[ACL 2025\] Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)
- [\[NeurIPS 2025\] Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training](../../NeurIPS2025/llm_efficiency/critical_batch_size_revisited_a_simple_empirical_approach_to_large-batch_languag.md)

</div>

<!-- RELATED:END -->
