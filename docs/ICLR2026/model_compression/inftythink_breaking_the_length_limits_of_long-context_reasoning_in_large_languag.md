---
title: >-
  [论文解读] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models
description: >-
  [ICLR 2026][模型压缩][长上下文推理] 提出 InftyThink，一种将整体式长推理转化为迭代式短推理+中间摘要的新范式，在不修改模型架构的前提下实现理论上无界的推理深度、显著降低计算成本，Qwen2.5-Math-7B 在 AIME24 上提升11%。
tags:
  - ICLR 2026
  - 模型压缩
  - 长上下文推理
  - 迭代推理
  - 摘要压缩
  - 计算效率
  - 推理范式
---

# InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2503.06692](https://arxiv.org/abs/2503.06692)  
**代码**: [Project Page](https://zju-real.github.io/InftyThink)  
**领域**: 模型压缩  
**关键词**: 长上下文推理, 迭代推理, 摘要压缩, 计算效率, 推理范式

## 一句话总结
提出 InftyThink，一种将整体式长推理转化为迭代式短推理+中间摘要的新范式，在不修改模型架构的前提下实现理论上无界的推理深度、显著降低计算成本，Qwen2.5-Math-7B 在 AIME24 上提升11%。

## 研究背景与动机
以DeepSeek-R1、o1为代表的推理模型通过长链思维实现了卓越性能，但长上下文推理面临三个根本问题：

**二次方计算扩展**：Decoder-based LLM的计算复杂度随序列长度呈二次增长，推理阶段资源消耗巨大

**上下文长度天花板**：推理过程受max_length约束，经常被截断而无法得出结论

**超出训练窗口后性能退化**：大多数模型预训练窗口仅4k-8k tokens，推理超过此范围时性能明显下降

现有解决方案（如CoT-Valve压缩推理链、TokenSkip删除冗余token、LightThinker用特殊token动态压缩）仍在"单次连续推理"范式内优化，未触及根本的计算扩展问题。

核心idea：借鉴人类认知——复杂问题分解为可管理的部分并总结中间进展。将整体推理分为多个边界长度的段落，每段后生成摘要，下一段基于摘要继续推理，形成"锯齿形"内存模式。

## 方法详解

### 整体框架
InftyThink 将推理分为多轮迭代：第1轮生成推理段 $RP_1$ + 摘要 $S_1$；后续轮以前一轮摘要作为历史上下文，生成新推理段 $RP_i$ + 新摘要 $S_i$；最后一轮生成推理段 $RP_n$ + 最终结论 $C$。

### 关键设计

1. **迭代推理范式 (Iterative Reasoning with Summarization)**:

    - 功能：将单次整体推理替换为多轮有界推理
    - 核心思路：
        - 首轮：`<|U|>Q<|A|><think>RP₁</think><summary>S₁</summary>`
        - 后续轮：`<|U|>Q<|A|><history>Sᵢ₋₁</history><think>RPᵢ</think><summary>Sᵢ</summary>`
        - 终轮：`<|U|>Q<|A|><history>Sₙ₋₁</history><think>RPₙ</think>C`
    - 设计动机：每轮保持有界上下文长度（锯齿形内存模式），理论上支持无限推理深度。简单问题可在第一轮直接得出结论，自然退化为传统范式

2. **数据重构流水线 (Data Reconstruction)**:

    - 功能：将现有长推理数据集转化为InftyThink格式
    - 核心思路：三步流水线——
        - Step I *推理分割*：基于超参数 $\eta$（最大段长度），在语义边界（句子/段落）处切分
        - Step II *摘要生成*：用Meta-Llama-3.3-70B-Instruct为每段生成摘要，摘要考虑所有先前段的上下文以保持推理连续性
        - Step III *训练实例构建*：组装为多个训练实例，首段实例包含 $(Q, RP_1, S_1)$，中间段 $(Q, S_{i-1}, RP_i, S_i)$，末段 $(Q, S_{n-1}, RP_n, C)$
    - 设计动机：从OpenR1-Math（220K样本）重构为333K InftyThink格式样本（$\eta$=4k），利用已有高质量推理数据避免从零生成

3. **推理时的执行机制**:

    - 功能：推理时模型迭代生成推理段和摘要，直到产生结论
    - 核心思路：每轮输出被解析，摘要成为下一轮上下文。设置 max_iters=10 防止无限循环，实验表明训练良好的模型自然会在合理迭代次数内收敛
    - 设计动机：无需架构修改，任何decoder-only模型均可使用

### 训练策略
使用instruction fine-tuning在OpenR1-Math-Inf（InftyThink格式）上训练多种基础模型。$\eta$ = 4k, max_iters = 10。

## 实验关键数据

### 主实验（base models, pass@16, temperature=0.7）
| 模型 | 格式 | MATH500 ACC | AIME24 ACC | GPQA ACC | Avg ACC |
|------|------|------------|-----------|---------|---------|
| Qwen2.5-Math-1.5B | Vanilla | 75.24 | 16.04 | 26.48 | 59.54 |
| Qwen2.5-Math-1.5B | InftyThink | **79.57** | **26.04** | **35.89** | **65.48** |
| Qwen2.5-Math-7B | Vanilla | 89.51 | 32.92 | 43.94 | 74.78 |
| Qwen2.5-Math-7B | InftyThink | **91.29** | **43.96** | **52.97** | **78.92** |
| Llama-3.1-8B | Vanilla | 82.10 | 20.83 | 41.35 | 68.49 |
| Llama-3.1-8B | InftyThink | **82.28** | **34.17** | **47.51** | **70.84** |

### 延迟对比（推理耗时）
| 模型 | MATH500延迟 Vanilla→InftyThink | AIME24延迟 |
|------|------|------|
| Qwen2.5-Math-7B | 1.26s→0.76s | 4.15s→4.66s |
| Qwen2.5-14B | 1.49s→1.43s | 11.30s→7.11s |

### 关键发现
- Qwen2.5-Math-7B 在 AIME24 上提升11%（32.92→43.96），GPQA上提升9%（43.94→52.97）
- 小模型（1.5B）获益更大：AIME24提升10%，GPQA提升9.4%
- MATH500延迟从1.26s降至0.76s，计算效率显著提升（曲线下面积更小）
- 模型规模越大（14B/32B），InftyThink的accuracy提升趋于平缓但延迟收益仍然显著
- 摘要生成模型的规模对最终性能影响不大（70B vs 更小模型差异有限）

## 亮点与洞察
- "锯齿形内存模式"概念直观且强大——周期性压缩使计算复杂度可控
- 无需架构修改、无需专门训练基础设施，仅需数据重构和SFT即可获得显著提升
- 挑战了"推理深度与计算效率必须权衡"的假设——两者可同时改善

## 局限与展望
- 摘要质量如何影响推理正确性缺乏系统分析——信息丢失可能在长推理链中累积
- $\eta$（段长度）固定为4K，动态调整可能更优（简单段无需4K，困难段可能不够）
- 依赖SFT训练，若结合RL（如GRPO）可能释放更大潜力
- 多轮摘要的可靠性在数值推理vs语言推理中可能表现不同

## 相关工作与启发
- **vs CoT-Valve**: CoT-Valve需预设压缩比，InftyThink自适应判断何时结束
- **vs LightThinker**: LightThinker压缩为隐式表示，InftyThink保持文本可解释性
- **vs TokenSkip**: TokenSkip删token会损失推理性能，InftyThink通过摘要保留关键信息

## 评分
- 新颖性: ⭐⭐⭐⭐ 迭代推理范式简单但有效，概念清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 5种基础模型、多benchmark、延迟分析、消融丰富
- 写作质量: ⭐⭐⭐⭐⭐ 图示优秀，锯齿形对比图直观易懂
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高，可直接用于现有模型
