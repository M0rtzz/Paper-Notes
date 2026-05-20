---
title: >-
  [论文解读] FocusLLM: Precise Understanding of Long Context by Dynamic Condensing
description: >-
  [ACL 2025][LLM效率][长上下文理解] 提出FocusLLM框架，通过将长文本分块并为每块注入动态提示（dynamic prompt），用可训练的候选token浓缩各块的关键信息，再通过并行解码机制聚合到本地上下文中生成下一个token…
tags:
  - "ACL 2025"
  - "LLM效率"
  - "长上下文理解"
  - "上下文压缩"
  - "并行解码"
  - "动态浓缩"
  - "无信息丢失"
---

# FocusLLM: Precise Understanding of Long Context by Dynamic Condensing

**会议**: ACL 2025  
**arXiv**: [2408.11745](https://arxiv.org/abs/2408.11745)  
**代码**: [https://github.com/leezythu/focusllm](https://github.com/leezythu/focusllm)  
**领域**: LLM效率  
**关键词**: 长上下文理解, 上下文压缩, 并行解码, 动态浓缩, 无信息丢失

## 一句话总结
提出FocusLLM框架，通过将长文本分块并为每块注入动态提示（dynamic prompt），用可训练的候选token浓缩各块的关键信息，再通过并行解码机制聚合到本地上下文中生成下一个token，仅用8K训练长度和0.5B训练预算即可扩展LLaMA-2到400K上下文，在LongBench和∞-Bench上超越所有基线。

## 研究背景与动机
- **领域现状**：LLM长上下文处理是核心挑战。现有方法分三类：(1) 位置编码修改（PI、NTK、YaRN）有外推限制；(2) 继续训练长序列（LongChat、LongAlpaca）计算代价高；(3) 上下文压缩（StreamingLLM、Activation Beacon）通过丢弃/压缩token减少开销
- **现有痛点**：
    - 位置编码方法在极长序列上PPL爆炸
    - 压缩方法存在**信息丢失**问题：在Passkey Retrieval中，压缩方法随上下文增长准确率骤降
    - 根源在于token重要性在解码过程中**动态变化**——当前步不重要的token可能在未来步变得关键
    - 现有压缩方法"一次性"决定保留/丢弃，无法适应这种动态性
- **核心矛盾**：低计算开销 vs. 无信息丢失 vs. 训练高效
- **切入角度**：不丢弃任何token，而是通过动态提示让模型在每个解码步从各块中"提取"当前最需要的信息
- **核心idea**：每步解码时，将本地上下文片段（动态提示）拼接到每个块后方，通过新增的可训练参数生成候选token，融合所有块的候选token后再生成最终token

## 方法详解

### 整体框架
给定长文本 $\{x_1,...,x_S\}$：
1. **分块**：将前m个token分为k个chunk $C_1,...,C_k$，剩余token作为local context
2. **动态浓缩**：每个 $C_i$ 后拼接dynamic prompt（local context片段），经修改后的decoder生成候选token
3. **并行解码**：k个chunk并行处理，k个候选token的KV拼接到local context中
4. **冻结decoder生成**：用原始（冻结）decoder在拼接后的表示上生成下一个token

### 关键设计
1. **Dynamic Prompt Injection（动态提示注入）**:

    - 在每个chunk后拼接local context的末512个token（推理时）
    - 动机：让模型根据**当前解码上下文**决定从chunk中提取什么信息
    - 每步生成新token后，该token追加到dynamic prompt末尾（可丢弃最旧的token保持固定长度）
    - 关键：dynamic prompt随解码过程**动态演化**，因此每步提取的信息不同——避免了信息丢失

2. **Candidate Token Generation（候选token生成）**:

    - 候选token = 每个chunk中最后一个local token $x_S$ 对应的**可训练hidden states**
    - 引入新的线性投影参数 $\{W_Q^c, W_K^c, W_V^c, W_O^c\}_l$（每层）
    - 候选token的query用新参数生成，但attend to的key/value包含原始token + 候选token自身
    - 原始模型参数完全冻结，仅训练新增参数
    - 候选token本质上是对chunk内容的**条件压缩表示**——条件来自dynamic prompt

3. **Parallel Decoding（并行解码）**:

    - k个chunk的候选token生成过程**相互独立**，可并行前向传播
    - 所有候选token的K/V逐层拼接到local context的KV缓存中
    - 最终用冻结的原始decoder在拼接后的表示上生成next token
    - 计算复杂度从 $O(L^2)$ 降至 $O((L/n)^2)$

4. **双损失联合训练**:

    - **Continuation Loss**：local context是memory tokens的自然续写，训练模型生成后续token
    - **Reconstruction Loss**：随机选L个连续memory tokens作为local context，训练模型重建这些token
    - 两者缺一不可：仅Continuation导致Passkey Retrieval准确率从99%降到1.69%；仅Reconstruction导致生成能力退化

### 损失函数 / 训练策略
- 自回归损失：仅在local context的token上计算loss
- 训练数据：RedPajama，序列长度3K-8K
- 训练预算：仅0.5B token（vs LongLlama的7B token）
- 可训练参数：约1/3（仅新增的attention projection参数）
- 基座模型：LLaMA-2-7B (chat/base)

## 实验关键数据

### 主实验

| 数据集 | 指标 | FocusLLM | Activation Beacon | InfLLM | StreamingLLM |
|--------|------|------|----------|------|------|
| ∞-Bench平均 | Acc | **44.03** | - | 43.05 | 15.64 |
| ∞-Bench Retrieve.Number | Acc | **83.56** | - | 81.69 | 4.41 |
| ∞-Bench Retrieve.PassKey | Acc | **95.76** | - | 99.15 | 4.92 |
| ∞-Bench Retrieve.KV | Acc | **12.40** | - | 0.60 | 0.00 |
| LongBench平均(Vicuna) | Score | **36.17** | - | 33.24 | 31.92 |
| LongBench HotpotQA | F1 | **40.65** | - | 22.53 | 22.17 |
| LongBench平均(LLaMA) | Score | **39.01** | 38.54 | - | - |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅Continuation Loss | PassKey=1.69% | 丧失信息恢复能力，Passkey任务几乎完全失败 |
| 仅Reconstruction Loss | PassKey=91.19% | 新token生成能力退化，NarrativeQA等任务下降 |
| Local context 1K vs 2K | 略有下降 | 候选token无法完全替代local context中的信息 |
| Chunk size (256-2048) | PPL稳定 | 不同块大小对PPL影响不大，说明可用更大块配合更长上下文模型 |

### 关键发现
- Passkey Retrieval至400K准确率保持~99%，优于所有基线（包括YaRN-128K的92.71%）
- StreamingLLM和Activation Beacon在∞-Bench的检索类任务上接近0分——信息丢失致命
- FocusLLM在语言建模上PPL不随长度增长显著恶化（100K时PG19:10.59 vs Activation Beacon:8.68）
- 可视化分析：Passkey任务中模型只对包含答案的chunk的候选token给高注意力；QA任务中多个候选token被关注，说明模型学会了聚合多chunk信息
- 内存和速度：相比Standard方法内存增长更平缓，比CEPE和LongLlama效率更高

## 亮点与洞察
- "动态浓缩"的核心思想与注意力机制的本质一致：让模型根据当前需求自主选择信息
- 训练效率极高：仅在8K长度上训练就能泛化到400K，这得益于chunk处理的天然外推性
- 与RefreshKV的对比：RefreshKV在解码时刷新小KV缓存；FocusLLM则在每步动态提取信息——思路相似但实现路径不同
- 双损失设计揭示了长文本理解的两个必要能力：续写能力 + 信息恢复能力

## 局限与展望
- 硬件限制仅测试到400K，实际上限未知
- 训练数据规模较小（0.5B token），增大数据量应能进一步提升
- local context大小固定，未探索自适应调整
- 每步解码需要k次前向传播（虽可并行），增加了计算开销
- **可探索方向**：(1) 在更长上下文模型（如LLaMA-2-32K）上使用更大chunk size；(2) 与KV量化结合进一步降低开销；(3) 探索候选token数量的自适应选择（某些chunk可能不需要候选token）

## 相关工作与启发
- Activation Beacon通过beacon token压缩上下文，但是一次性压缩导致信息丢失
- CEPE用小encoder逐块编码长文本再通过cross-attention融入decoder，但memory不额外推
- AutoCompressor递归压缩上下文，但在长序列上PPL爆炸
- InfLLM将处理过的上下文存入memory unit并用注意力分数检索，思路相近但FocusLLM更端到端

## 评分
- 新颖性: ⭐⭐⭐⭐ 动态提示+候选token+并行解码的组合设计新颖，但chunking+compression的大框架不新
- 实验充分度: ⭐⭐⭐⭐⭐ 语言建模、LongBench、∞-Bench、Passkey Retrieval全覆盖，消融分析充分，可视化直观
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，方法描述系统，但公式较多可能增加阅读负担
- 价值: ⭐⭐⭐⭐⭐ 高训练效率+强性能+可扩展到400K，实用价值非常高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](longbench_v2_towards_deeper_understanding_and_reasoning_on_realistic_long-contex.md)
- [\[ACL 2025\] Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)
- [\[ACL 2025\] Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention](efficient_many-shot_in-context_learning_with_dynamic_block-sparse_attention.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[ACL 2025\] Literary Evidence Retrieval via Long-Context Language Models](literary_evidence_retrieval_via_long-context_language_models.md)

</div>

<!-- RELATED:END -->
