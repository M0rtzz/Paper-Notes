---
title: >-
  [论文解读] River-LLM: Large Language Model Seamless Exit Based on KV Share
description: >-
  [ACL 2026][早退机制] 本文提出 River-LLM，一个无需训练的框架，通过构建轻量级 KV 共享退出通道（Exit River）解决了 decoder-only 架构中 Early Exit 的 KV Cache 缺失问题，利用状态转换相似度引导退出决策，实现 1.71×-2.16× 的实际推理加速且保持近无损生成质量。
tags:
  - ACL 2026
  - 早退机制
  - KV缓存
  - 动态推理
  - 模型加速
  - 量化
---

# River-LLM: Large Language Model Seamless Exit Based on KV Share

**会议**: ACL 2026  
**arXiv**: [2604.18396](https://arxiv.org/abs/2604.18396)  
**代码**: 无  
**领域**: LLM效率  
**关键词**: 早退机制, KV缓存, 动态推理, 模型加速, 量化

## 一句话总结
本文提出 River-LLM，一个无需训练的框架，通过构建轻量级 KV 共享退出通道（Exit River）解决了 decoder-only 架构中 Early Exit 的 KV Cache 缺失问题，利用状态转换相似度引导退出决策，实现 1.71×-2.16× 的实际推理加速且保持近无损生成质量。

## 研究背景与动机

**领域现状**：Early Exit 是 LLM 动态推理加速的主流方向，通过根据输入复杂度动态跳过冗余层来减少计算。已有方法如 SkipDecode（单调递减退出）、EE-LLM（批量重计算）、CALM（状态传播）、D-LLM（KV掩码）等从不同角度尝试解决这一问题。

**现有痛点**：在 decoder-only 架构中，Early Exit 的效率受到 **KV Cache 缺失问题**的严重瓶颈。当一个 token 提前退出时，跳过的层无法为后续 token 提供必要的历史 KV 状态。作者的实证分析表明：虽然理论上超过 50% 的 token 可以在早期层退出，但实际 wall-clock 加速微乎其微。

**核心矛盾**：现有四种 KV 恢复策略都存在根本性缺陷：批量重计算引入显著延迟开销；单调递减退出严重限制退出灵活性；状态传播牺牲精度换取速度；KV 掩码导致严重精度损失。没有方法能同时满足"逐 token 自由退出"和"KV 完整性"。

**本文目标**：设计一种"无缝退出"(Seamless Exit) 机制，使单个 token 可以在任意层独立退出（粒度自由），同时跳过层的 KV 缓存作为退出路径执行的副产品自动填充（内在 KV 完整性），无需后退出恢复或重计算。

**切入角度**：受 KV 缓存冗余性研究启发，作者发现可以通过量化后的轻量级退出层复制骨干 decoder 的 KV 生成，以极低开销"代替"跳过的层完成 KV 填充。退出层产出的 KV 与骨干层的余弦相似度保持在 0.97 以上。

**核心 idea**：构建一条与骨干 decoder 一一映射的轻量级"退出河流"（KV-Shared Exit River），使用 4-bit 量化权重加速 token 通过退出通道的速度（2.4× 吞吐提升），同时自然生成与骨干兼容的 KV 缓存。

## 方法详解

### 整体框架
River-LLM 的推理分为两个阶段：Prefill 阶段使用序列级退出（所有 token 统一深度退出以保持并行注意力效率）；Generation 阶段切换为 token 级退出（每个 token 在最优深度终止）。当某 token 触发退出条件后，剩余计算被卸载到量化加速的退出层序列，最终到达原始 LM Head 生成 logits。退出层同时产出完整的 KV 缓存，消除后续 token 的 KV 缺失问题。

### 关键设计

1. **KV 共享退出层 (KV-Shared Exit Layer)**:

    - 功能：作为骨干 decoder 的轻量级替代品，在加速 token 通过的同时生成兼容的 KV 缓存
    - 核心思路：退出层继承骨干层的架构和参数，然后对 Attention 和 FFN 块施加 4-bit 权重量化 (W4A16)，同时保持 KV Cache 为 FP16 格式以保留表示密度。每个退出层与对应的骨干 decoder 共享相同的 KV Cache 寻址方案。通过量化和部分图编译优化的推理内核，退出层实现 2.4× 的吞吐提升，且生成的 KV 与骨干原生 KV 的余弦相似度保持在 0.97 以上
    - 设计动机：核心洞察是 KV 缓存不需要完全精确——4-bit 量化引入的误差在可接受范围内，但计算节省巨大。整个权重迁移过程通常在一分钟内完成，无需任何训练

2. **基于状态转换相似度的退出决策**:

    - 功能：预测累积量化误差，引导精确的退出时机
    - 核心思路：利用 decoder 块输入输出之间的余弦相似度（状态转换相似度）作为退出指标。退出决策定义为 $\mathcal{D}^{(l)} = \mathbb{I}(\min_{b \in \mathcal{B}} s_{t,b}^{(l)} > \tau)$，其中 $s_{t,b}^{(l)} = \frac{\mathbf{h}_{t,b}^{(l-1)\top} \mathbf{h}_{t,b}^{(l)}}{\|\mathbf{h}_{t,b}^{(l-1)}\| \|\mathbf{h}_{t,b}^{(l)}\|}$。作者发现早期层的状态转换相似度与最终层的骨干-退出值向量相似度存在中等正相关 ($r=0.5536$)，因此可以用前者预测后者
    - 设计动机：状态转换相似度大致呈单调递增趋势，符合 Early Exit 的客观规律（退出层之后的大多数层也满足退出条件）。退出判定的计算复杂度仅为 $\mathcal{O}(d)$，约 100 微秒，仅占总推理时间的 0.0688%

3. **骨干卸载 (Backbone Offloading)**:

    - 功能：进一步减少 GPU 显存占用
    - 核心思路：由于绝大多数 token 在早期阶段终止骨干遍历，框架可自动将后续稀疏激活的骨干层从主显存中驱逐。模型在接近全量化基线的内存占用下运行，同时退出河流常驻显存提供连续的语义补全
    - 设计动机：River-LLM 相比全模型量化的优势在于选择性计算保真度——"困难"或高熵 token 以全精度通过骨干，"简单" token 卸载到退出河流

### 损失函数 / 训练策略
River-LLM 是完全无需训练的框架。退出层权重直接从骨干层复制并施加 PTQ 量化，无需任何微调。通过调节阈值 $\tau$ 实现精度-速度的灵活权衡。

## 实验关键数据

### 主实验
在 GSM8K、MATH、HumanEval 上的实际 wall-clock 加速对比。

| 模型 | 任务 | Backbone Acc | Full Quant. Acc | River-LLM Acc | River-LLM 加速 |
|------|------|------|------|------|------|
| Llama3.2 1B | GSM8K | 33.2 | 25.1 | 29.3 | 2.16× |
| Llama3.2 1B | MATH | 17.8 | 12.2 | 14.6 | 1.88× |
| Llama3.1 8B | GSM8K | 78.2 | 69.8 | 74.4 | 1.78× |
| Llama3.1 8B | HumanEval | 57.3 | 50.2 | 55.5 | 1.77× |
| Ministral3 8B | MATH | 48.1 | 46.0 | 46.6 | 1.85× |

### 消融实验

| KV 策略 | 实际延迟 | 精度保持 | 说明 |
|---------|---------|---------|------|
| KV Mask | 最高骨干延迟 | 差 | 需执行更深层来补偿精度损失 |
| KV Recompute | 高计算开销 | 好 | 长序列生成中开销累积 |
| State Propagation | 中等 | 中等 | 精度-速度折中 |
| Mono-Decreasing | 中等 | 好 | 限制退出灵活性 |
| KV Share (Ours) | 最低 | 好 | 无需恢复操作 |

### 关键发现
- River-LLM 平均只执行 3-4 个骨干层即可达到与全模型接近的精度，在 Llama3.1 8B 上大部分任务在中位层之前终止
- 在 HumanEval 上 River-LLM 甚至超过全模型基线（57.3 vs 55.5），可能是通过跳过冗余深层减少了累积噪声或"过度思考"
- 相比全量化基线，River-LLM 吞吐略低约 10%，但精度保持远优于全量化
- 退出决策逻辑仅约 100 微秒，占总推理时间 0.0688%，开销可忽略
- GPU 内存消耗显著低于骨干模型和现有 Early Exit 基线，接近全量化模型

## 亮点与洞察
- **"无缝退出"的概念定义**很有价值：粒度自由 + 内在 KV 完整性，清晰地将 River-LLM 与所有先前方法区分开来。这个定义本身就是对 Early Exit 研究的贡献
- **量化退出层作为 KV 代理**的想法非常巧妙：不追求精确恢复 KV，而是用 4-bit 量化层"近似"生成，0.97+ 的余弦相似度足以维持自回归生成质量。这利用了 KV 缓存的内在冗余性
- 完全无需训练是一大实用优势，权重迁移在一分钟内完成，可以即插即用于任何 decoder-only 模型
- 量化后端可替换（HQQ→AWQ 后精度进一步提升），框架具有良好的可扩展性

## 局限与展望
- 当前评估仅覆盖最大 8B 参数模型，24B 和 70B 模型上的行为未验证
- 对 prefill 主导的任务（如 MMLU）加速不明显，因为 prefill 阶段使用序列级退出
- 退出阈值 $\tau$ 需要手动选择，不同模型和任务的最优值可能不同
- 累积量化误差在非常早的退出点仍然存在（虽然可控），对极长序列生成的影响未充分研究

## 相关工作与启发
- **vs LayerSkip/SpecEE**: 这些方法将 Early Exit 与投机解码结合，但受限于序列级退出或短 draft 序列，River-LLM 实现了真正的 token 级自由退出
- **vs CALM**: CALM 使用状态传播填充 KV，这是一种精度-速度折中；River-LLM 通过量化退出层生成高保真 KV，消除了这种折中
- **vs 全模型量化**: 全量化对所有 token 施加均匀精度损失，River-LLM 选择性地让"难" token 走全精度骨干、"易" token 走量化退出河流，实现了更优的帕累托前沿

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ KV 共享退出河流的概念新颖且优雅，清晰解决了 Early Exit 的核心瓶颈
- 实验充分度: ⭐⭐⭐⭐ 四个模型、多个基准、与全量化和现有策略的对比充分，但缺少 >8B 模型验证
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，图表信息量大，但部分内容有重复

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] KV Cache Transform Coding for Compact Storage in LLM Inference](../../ICLR2026/code_intelligence/kv_cache_transform_coding_for_compact_storage_in_llm_inference.md)
- [\[ACL 2026\] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)
- [\[ACL 2025\] CoCo-Bench: A Comprehensive Code Benchmark for Multi-task Large Language Model Evaluation](../../ACL2025/code_intelligence/coco-bench_a_comprehensive_code_benchmark_for_multi-task_large_language_model_ev.md)
- [\[ACL 2026\] KoCo-Bench: Can Large Language Models Leverage Domain Knowledge in Software Development?](koco-bench_can_large_language_models_leverage_domain_knowledge_in_software_devel.md)
- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)

<!-- RELATED:END -->
