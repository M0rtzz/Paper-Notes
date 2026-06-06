---
title: >-
  [论文解读] Hogwild! Inference: Parallel LLM Generation via Concurrent Attention
description: >-
  [NeurIPS 2025][LLM Agent][并行推理] 提出 Hogwild! Inference——一种无需预定义协作框架的并行 LLM 推理协议，多个 LLM 实例通过共享的并发 KV 缓存实时同步，利用 RoPE 位置编码避免重计算，在数学推理和编程任务上以更少的串行步骤达到更高精度。
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "并行推理"
  - "共享KV缓存"
  - "协作推理"
  - "RoPE位置编码"
  - "LLM加速"
---

# Hogwild! Inference: Parallel LLM Generation via Concurrent Attention

**会议**: NeurIPS 2025  
**arXiv**: [2504.06261](https://arxiv.org/abs/2504.06261)  
**代码**: [https://github.com/eqimp/hogwild_llm](https://github.com/eqimp/hogwild_llm)  
**领域**: LLM Agent  
**关键词**: 并行推理, 共享KV缓存, 协作推理, RoPE位置编码, LLM加速

## 一句话总结
提出 Hogwild! Inference——一种无需预定义协作框架的并行 LLM 推理协议，多个 LLM 实例通过共享的并发 KV 缓存实时同步，利用 RoPE 位置编码避免重计算，在数学推理和编程任务上以更少的串行步骤达到更高精度。

## 研究背景与动机
现代 LLM 在推理、长文本生成、工具使用等复杂任务中需要大量推理时计算（逐 token 生成）。人类解决复杂问题时常用的策略是**协作**：拆分子任务、并行探索不同策略、实时交流调整。

现有并行推理方法各有局限：

**Self-Consistency**（投票聚合）：独立生成多个解再投票，当某个线程较慢时效率低下

**Skeleton-of-Thought**（子任务并行）：先规划再并行执行子任务，但要求问题可立即拆分，且初始计划错误无法修正

**PASTA 等异步子任务**：需要专门微调，单个子任务过长时陷入等待

核心洞察：**没有单一协作策略适用于所有任务**。与其预定义协作框架，不如让 LLM 自己决定如何协作。这受人类协作启发——动态重新规划、中途放弃不佳方案、讨论策略调整。

## 方法详解

### 整体框架
Hogwild! Inference（灵感来自 Hogwild! SGD 的异步更新思想）：
- 多个 LLM 实例（"workers"）使用**相同权重**并行生成
- 共享一个**并发更新的 KV 缓存**
- 每个 worker **即时看到**其他 worker 的生成内容
- 通过 prompt 引导 worker 自主决定协作方式
- **无需额外微调**，现有推理模型开箱即用

### 关键设计

#### 1. 共享 KV 缓存与并发注意力（Section 3.1）

以 2 个 worker（Alice 和 Bob）为例：
- 缓存分为多个 block：公共 prompt block + 每个 worker 的生成 block
- Alice 看到：公共 prompt → Bob 的 token → Alice 自己的 token
- Bob 看到：公共 prompt → Alice 的 token → Bob 自己的 token

挑战：同一 KV 对在不同 worker 视角中出现在不同位置，且随生成推进相对位置会变化。

**利用 RoPE 避免重计算**：大多数现代 LLM 使用旋转位置编码（RoPE），其中 key 和 query 被旋转到与绝对位置成比例的角度。Hogwild! Inference 不重新编码 token，而是**旋转 query**：
$$\rho(q, i_q)[\rho(A, i_k^A) \oplus \rho(B, i_k^B) \oplus \rho(C, i_k^C)] = \rho(q, i_q-i_k^A)A \oplus \rho(q, i_q-i_k^B)B \oplus \rho(q, i_q-i_k^C)C$$

只需旋转**当前step的 query（单个 token）**，而非包含数千 token 的 cache block。

#### 2. 缓存结构设计（Section 3.2）

**Combined Layout（主方法）**——结合了聊天室式步骤共享和实时 token 可见性：

- **Common Cache**：存储系统 prompt、任务描述、以及所有 worker 过去的推理步骤历史
- **Other Workers**：其他 worker 当前未完成步骤的 KV（按 worker 顺序排列）
- **Current Worker**：当前 worker 自己的未完成步骤

每个 block 以 `\n\n` + worker ID header 开始。当 worker 完成一个推理步骤时，其 KV 被旋转移入共享历史。

这种设计解决了"距离问题"：简单拼接会导致 worker 间最新输出相距甚远，降低注意力效果。

#### 3. Zero-Shot 协作提示（Section 3.3）

两部分提示策略：
1. **System Prompt**：描述共享缓存的"规则"，建议 workers 协作
2. **s1-like 干预提示**：每生成 1024 个 token，向随机 worker 插入"Wait, am I doing redundant work? (yes/no):"

后者至关重要——推理模型可能过于"专注"于当前生成而忽略其他实例的进展。直接询问时，它们通常能发现冗余并调整策略。

### 损失函数 / 训练策略
- **无需训练/微调**——完全是零样本推理协议
- 使用现有推理模型开箱即用（QwQ-32B, DeepSeek-R1, Qwen3, Phi-4 Reasoning Plus）
- 实现基于 Flash-Decoding，自定义 GPU kernel 处理多 cache block 的注意力计算
- 每个 cache block 中 KV 存储在位置 0,1,...,len(block)，通过 query 旋转处理实际位置

## 实验关键数据

### 主实验

**LIMO 数据集（817道数学推理题）**——QwQ-32B：

Hogwild! Inference（2 workers）在相同串行步骤预算下始终优于所有基线：
- 在中低 token 预算下优势明显（更快收敛到正确答案）
- 随 worker 数量增加（2→3→4），准确率进一步提升
- Self-Consistency 有改善但不如 Hogwild!
- Skeleton-of-Thought 对无明显子任务的问题效果有限

**跨模型泛化**：
- QwQ-32B: 显著改善 ✓
- Phi-4-Reasoning-Plus (14B): 改善 ✓
- Qwen3-8B: 改善 ✓
- Qwen3-4B: 部分改善
- Qwen3-1.7B: 较小模型难以适应共享缓存任务

**GSM8k×5 合成基准**（5道独立题打包）：
- Hogwild! 和 SoT 都能有效加速此类可拆分任务
- 验证了 KV 缓存操作不破坏模型推理能力

### 消融实验

**OlympiadBench 数学+物理**：
- 数学部分：QwQ-32B, Qwen3-14B, Qwen3-8B 均有改善
- 物理部分：QwQ-32B 和 Qwen3-8B 改善，但 Qwen3-14B 在 ~4096 token 后过度思考

**LiveCodeBench 代码生成**（279题）：
- QwQ-32B: 改善明显
- Phi-4-Reasoning-Plus: 改善
- Qwen3-8B: 改善

**AIME'25**（大模型）：
- Qwen3-235B-A22B: 改善
- DeepSeek-R1: 改善

**缓存布局消融**：
- Combined（token-wise 同步 + 步骤历史）> Interleaved（仅步骤级同步）> Contiguous（仅 token 级同步）

### 关键发现

**协作能力量化**（GPT-4o 评分，1-6 分）：

| 设置 | QwQ-32B | Phi-4-R+ | Qwen3-8B |
|------|---------|----------|----------|
| 无同步（独立生成） | ~1.2 | ~1.1 | ~1.1 |
| 步骤级同步 | ~2.5 | ~2.2 | ~2.0 |
| **Token 级同步（完整 Hogwild!）** | **~3.5** | **~3.0** | **~2.5** |

Token 级即时同步显著优于仅步骤级同步。

**推理性能**（QwQ-32B-AWQ, L40S GPU）：

| Workers | 1024 tokens | 8192 tokens | 16384 tokens |
|---------|-------------|-------------|--------------|
| 1 (基线) | 20.1 tok/s | 19.3 tok/s | 18.3 tok/s |
| 2 workers | 36.3 tok/s | 36.1 tok/s | 34.3 tok/s |
| 4 workers | 68.9 tok/s | 66.3 tok/s | 60.3 tok/s |

2 workers 接近 **2x 加速**，4 workers **3.2-3.6x 加速**，overhead 很小。

## 亮点与洞察
1. **范式创新**：不预定义协作框架，让 LLM 自己决定如何协作——这是对并行推理策略的根本性反思
2. **零样本协作能力**：现有推理模型（QwQ, DeepSeek-R1）无需微调就能"推理以协调"——制定和遵循计划、纠正错误、利用彼此的关键发现
3. **工程实现优雅**：利用 RoPE 的旋转特性，仅旋转当前 query 而非所有 cache block，避免了 O(n³) 的重计算开销
4. **即时可见性的力量**：token 级同步（worker 生成过程中即可被其他 worker 看到）远优于步骤级同步
5. **s1-like 干预提示**简单却有效——推理模型容易"过度专注"，定期提醒"你在做冗余工作吗？"能触发策略调整

## 局限与展望
1. **小模型鲁棒性差**：Qwen3-1.7B 难以适应共享缓存设置，暗示存在模型规模下限
2. **长上下文鲁棒性降低**：随上下文增长，协作效果可能减弱
3. **自动评估依赖 GPT-4o**：协作质量评分依赖私有模型，可能影响可复现性
4. **未探索微调**：通过 RL 或专门数据微调可能进一步提升协作能力
5. **缓存增长管理**：长时间推理时共享历史缓存可能过大，需要选择性遗忘机制
6. **人机交互潜力**：KV 缓存重排可用于人类异步给予反馈，但未深入探索

## 相关工作与启发
- **Hogwild! SGD**：名称灵感来源，异步更新的思想从训练迁移到推理
- **Self-Consistency / Skeleton-of-Thought**：直接对比的并行推理方法
- **Paged Attention（vLLM）**：类似的 KV 缓存分段管理，但 Hogwild! 实现跨 worker 注意力
- **s1（muennighoff2025s1）**：提供了"budget thinking"的灵感
- 启发：(1) LLM 的涌现协作能力值得深入研究；(2) 推理时计算的并行化是 LLM 加速的重要方向；(3) 共享记忆模型可扩展到代码协作、工具使用等场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (并行推理的新范式，共享 KV 缓存 + 零样本协作极具创意)
- 实验充分度: ⭐⭐⭐⭐⭐ (多模型×多基准×消融×协作评分×性能基准，极其全面)
- 写作质量: ⭐⭐⭐⭐ (主体清晰，但技术细节较密集)
- 价值: ⭐⭐⭐⭐⭐ (开辟了推理时并行化的新方向，实用性强，开源代码)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution](../../AAAI2026/llm_agent/cook_and_clean_together_teaching_embodied_agents_for_paralle.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](../../ACL2025/llm_agent/metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2025\] GeAR: Graph-enhanced Agent for Retrieval-augmented Generation](../../ACL2025/llm_agent/gear_graph-enhanced_agent_for_retrieval-augmented_generation.md)
- [\[ICML 2026\] Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](../../ICML2026/llm_agent/towards_feedback-to-plan_decisions_for_self-evolving_llm_agents_in_cuda_kernel_g.md)
- [\[ACL 2026\] Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models](../../ACL2026/llm_agent/dynamic_generation_of_multi-llm_agents_communication_topologies_with_graph_diffu.md)

</div>

<!-- RELATED:END -->
