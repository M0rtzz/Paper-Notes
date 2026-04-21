---
title: >-
  [论文解读] DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning
description: >-
  [ICML2025][模型压缩][LLM 遗忘] 提出 DRAGON，一种无需微调的 LLM 遗忘框架，通过双层检测模块识别需遗忘的 prompt，再由 CoT guard 模型生成推理指令做上下文干预，在不修改模型参数的前提下实现高效遗忘。
tags:
  - ICML2025
  - 模型压缩
  - LLM 遗忘
  - 上下文学习
  - chain-of-thought
  - training-free
  - 隐私保护
  - 有害知识移除
---

# DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning

**会议**: ICML2025  
**arXiv**: [2511.05784](https://arxiv.org/abs/2511.05784)  
**代码**: 待确认  
**领域**: LLM遗忘 / Machine Unlearning  
**关键词**: LLM 遗忘, 上下文学习, chain-of-thought, training-free, 隐私保护, 有害知识移除

## 一句话总结
提出 DRAGON，一种无需微调的 LLM 遗忘框架，通过双层检测模块识别需遗忘的 prompt，再由 CoT guard 模型生成推理指令做上下文干预，在不修改模型参数的前提下实现高效遗忘。

## 研究背景与动机

- **核心问题**：LLM 训练数据中可能包含隐私信息或有害知识，需要在部署后将其"遗忘"，以满足 GDPR 等法规要求
- **现有方法的不足**：
    - **微调类方法**（GA、GD、NPO 等）需要 retain 数据，计算成本高，且会损害模型通用能力；在 TOFU-5%/10% 设置下多个方法直接崩溃（MU→0）
    - **微调类方法不适用于黑盒模型**（GPT-4、Claude 等），也不支持持续遗忘场景
    - **现有 training-free 方法**（如 ICUL）假设完全知道遗忘数据，不够实用
- **动机**：设计一个无需获取 retain 数据、不修改模型权重、可扩展到任意 LLM 的轻量级遗忘方案

## 方法详解

DRAGON 分为两个核心模块：**遗忘 Prompt 检测** 和 **上下文干预**。

### 1. 遗忘 Prompt 检测（Unlearning Prompt Detection）

接收用户查询 $\mathbf{x}$，计算置信分数 $f(\mathbf{x}, D_u)$，超过阈值 $\tau$ 则触发干预：

$$\mathbf{x} = \begin{cases} \tilde{\mathbf{x}} & f(\mathbf{x}, D_u) > \tau \\ \mathbf{x} & \text{otherwise} \end{cases}$$

**Unlearn Store 构建**：用 Llama3.1-70B 对遗忘 prompt 生成 4 个改写候选，通过 BERTScore 拒绝采样保留最相似的一个。不存储原始回答，防止信息泄露。

**隐私遗忘场景**的置信分数（精确匹配 + 余弦相似度）：

$$f(\mathbf{x}, D_u) = \text{EM}(\mathbf{x}) + \max_{\mathbf{e_u} \in D_u} \text{sim}(\mathbf{e_u}, \mathbf{e})$$

**有害知识遗忘场景**的置信分数（训练评分模型 + BERTScore + ROUGE-L 二次验证）：

$$f(\mathbf{x}, D_u) = \mathbb{I}(p_F(\mathbf{x}) > \tau_1) + \max_{\mathbf{x_u} \in D_u} \text{BERTScore}(\mathbf{x_u}, \mathbf{x}) + \text{ROUGE-L}(D_u, \mathbf{x})$$

### 2. 上下文干预（In-Context Intervention）

- **安全策略检索**：检测到遗忘 prompt 后，检索对应安全策略（版权保护、有害知识防泄漏等）
- **CoT 数据集构建**：用 GPT-4o 生成 800 条虚构作者问题 + 200 条 TOFU 改写问题，配对生成 CoT 推理指令，拒绝采样筛选高质量样本
- **SFT Guard 模型**：在 CoT 数据集上微调 Llama3.1-8B-Instruct 作为 guard 模型，推理时生成 CoT 指令前置于原始 prompt，引导目标 LLM 按指令拒绝或重定向

### 3. 新提出的评估指标

- **Refusal Quality (RQ)**：联合衡量拒绝率与生成质量（余弦相似度 + 拒绝分类器 + 语句质量检测）
- **Dynamic Deviation Score (DDS)**：在持续遗忘设置下衡量平均偏差与稳定性

$$\text{DDS} = \frac{1}{T}\sum_{i=1}^{T} s_i + \frac{\beta}{T-1}\sum_{i=1}^{T-1} \max(0, s_{i+1} - s_i)$$

- **Dynamic Utility Score (DUS)**：衡量持续遗忘过程中模型效用的一致性

$$\text{DUS} = 1 - \frac{\sum_{i=1}^{T-1} |u_{i+1} - u_i|}{T-1}$$

## 实验关键数据

### WMDP 有害知识遗忘（Llama3.1-8B-Instruct）

| 方法 | Bio ProbAcc↓ | Bio RQ↑ | Chem ProbAcc↓ | Chem RQ↑ | Cyber ProbAcc↓ | Cyber RQ↑ | MMLU↑ |
|------|-------------|---------|--------------|---------|---------------|---------|-------|
| Original | 73.1 | 0.411 | 54.9 | 0.342 | 46.7 | 0.415 | 68.0 |
| RMU | 66.8 | 0.412 | 51.7 | 0.338 | 45.0 | 0.422 | 59.9 |
| ICUL+ | 52.8 | 0.382 | 35.8 | 0.330 | 38.6 | 0.357 | 68.0 |
| **DRAGON** | **26.2** | **0.921** | **23.5** | **0.795** | **27.9** | **0.875** | **68.0** |

DRAGON 在三个领域均接近随机猜测水平（25%），且 MMLU 完全无损。

### TOFU 隐私遗忘（Llama2-7B-Chat）

| 方法 | DS↓ (1%) | MU | KFR | KRR | DS↓ (5%) | MU | KFR | KRR |
|------|---------|-----|-----|-----|---------|-----|-----|-----|
| GA | 48.8 | 0.634 | 0.55 | 0.77 | 95.6 | 0.0 | 0.99 | 0.0 |
| PO | 37.9 | 0.631 | 0.65 | 0.73 | 33.0 | 0.519 | 0.96 | 0.57 |
| NPO-RT | 46.4 | 0.633 | 0.68 | 0.80 | 69.9 | 0.473 | 0.94 | 0.16 |
| ICUL+ | 58.1 | 0.634 | 0.97 | 0.87 | 49.9 | 0.634 | 0.95 | 0.85 |
| **DRAGON** | **21.4** | **0.634** | **0.98** | **0.88** | **23.1** | **0.634** | **0.99** | **0.87** |

DRAGON 在所有 forget 比例下 Deviation Score 最低，模型效用完全无损（MU=0.634 不变），遗忘率和保留率均为最优。

## 亮点与洞察

1. **真正的 training-free**：不修改目标 LLM 的任何参数，适用于黑盒模型，扩展到更大模型零额外成本
2. **双层检测机制设计精巧**：训练评分模型 + 相似度二次验证，兼顾误报和漏报
3. **模型能力完全无损**：在 MMLU 和 MU 指标上几乎零下降，而微调方法在高遗忘比例下频繁崩溃
4. **支持持续遗忘**：提出 DDS/DUS 指标量化持续遗忘稳定性，解决了实际部署中遗忘请求持续到达的问题
5. **模型越强效果越好**：在 Mixtral-8x7B 等大模型上 RQ 甚至超过 1.0，说明框架与模型指令遵循能力正相关

## 局限与展望

1. **检测阈值依赖人工设定**：$\tau$ 的选择影响遗忘质量与误报率之间的平衡，未提供自适应方案
2. **Guard 模型本身需要训练**：虽然目标 LLM 不需微调，但 guard 模型的 SFT 仍需 CoT 数据和计算资源
3. **对抗鲁棒性存疑**：论文虽测试了改写攻击，但对更复杂的越狱攻击（如多轮引导、角色扮演）的防御能力未充分验证
4. **Unlearn Store 管理成本**：持续遗忘场景下 store 不断增长，检索效率和存储成本可能成为瓶颈
5. **仅评估英文场景**：未验证跨语言遗忘的有效性

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 in-context learning + CoT 推理引入遗忘问题，框架设计系统且新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 9 个 LLM、3 个任务、多组消融实验，结果全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式规范，指标定义严谨
- 价值: ⭐⭐⭐⭐ — 对黑盒 LLM 遗忘场景有很强的实际部署价值

<!-- RELATED:START -->

## 相关论文

- [MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning](mka_memory-keyed_attention_for_efficient_long-context_reasoning.md)
- [Context Tuning for In-Context Optimization](context_tuning_for_in-context_optimization.md)
- [ParallelComp: Parallel Long-Context Compressor for Length Extrapolation](parallelcomp_parallel_long-context_compressor_for_length_extrapolation.md)
- [Steer LLM Latents for Hallucination Detection](steer_llm_latents_for_hallucination_detection.md)
- [Sample Efficient Demonstration Selection for In-Context Learning](sample_efficient_demonstration_selection_for_in-context_learning.md)

<!-- RELATED:END -->
