---
title: >-
  [论文解读] Context Tuning for In-Context Optimization
description: >-
  [ICML 2025][模型压缩][上下文学习] 提出 Context Tuning，用少样本示例初始化可训练的 prompt/KV 前缀，通过梯度优化上下文表示（而非模型参数）来增强 LLM 的 few-shot 适应能力，CT-KV 变体在线性时间复杂度下达到与 TTT 竞争的精度。
tags:
  - ICML 2025
  - 模型压缩
  - 上下文学习
  - 提示学习
  - Prefix Tuning
  - KV Cache 优化
  - 参数高效适应
---

# Context Tuning for In-Context Optimization

**会议**: ICML 2025  
**arXiv**: [2507.04221](https://arxiv.org/abs/2507.04221)  
**代码**: [https://agenticlearning.ai/context-tuning](https://agenticlearning.ai/context-tuning)  
**领域**: 模型压缩  
**关键词**: 上下文学习, prompt tuning, Prefix Tuning, KV Cache 优化, 参数高效适应

## 一句话总结

提出 Context Tuning，用少样本示例初始化可训练的 prompt/KV 前缀，通过梯度优化上下文表示（而非模型参数）来增强 LLM 的 few-shot 适应能力，CT-KV 变体在线性时间复杂度下达到与 TTT 竞争的精度。

## 研究背景与动机

大语言模型（LLM）通过 In-Context Learning（ICL）可以零参数更新地适应新任务，但 ICL 仅依赖前向传播解读示例，面对复杂推理或领域偏移时表现不佳。现有改进路线有两条：

**Prompt/Prefix Tuning**: 在输入前附加可训练向量，用梯度下降优化。但这些向量通常用随机 token 初始化，完全没有利用 few-shot 示例中的任务相关信息。

**Test-Time Training (TTT)**: 在推理时用 LoRA 微调模型权重，效果好但计算代价是示例数的**二次方**（需要对示例做排列组合）。

核心洞察：ICL 擅长从上下文中提取任务信息，prompt-based 方法擅长梯度优化——能否把二者桥接起来？即**用 few-shot 示例初始化可训练上下文，再用梯度优化这个上下文**？

## 方法详解

### 整体框架：In-Context Optimization (ICO)

作者提出 ICO 统一框架，将 few-shot 学习形式化为：

$$\min \sum_{i=1}^{k} -\log p_\phi(y_i \mid [\theta_{\text{context}}^{(i)}; x_i])$$

其中 $\theta_{\text{context}}^{(i)}$ 是从示例集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^k$ 导出的上下文表示。ICO 的关键在于**上下文表示来自 few-shot 示例**并通过梯度优化，这是与传统 Prompt Tuning（随机初始化）和 ICL（无梯度优化）的本质区别。

在 ICO 框架下，TTT 可以被视为一个实例：它将上下文设为 $\theta_{\text{context}}^{(i)} = \mathcal{C}^{-i}$（去掉第 $i$ 个示例的随机排列拼接），然后通过 LoRA 更新模型参数 $\phi$。

### 关键设计

Context Tuning 冻结模型参数 $\phi$，仅优化上下文表示 $\theta_{\text{context}}$，提出两个变体：

#### CT-Prompt

将所有示例对拼接后送入模型，取其 prompt embedding 作为初始化：

$$\theta_{\text{context}} = P_{\text{CT}} \leftarrow \text{Embed}(\mathcal{C})$$

然后像标准 Prompt Tuning 一样优化这个 soft prompt，但初始化包含了任务相关的语义信息。**缺点**：每步优化需要重新计算所有层的 KV，时间复杂度为 $O(k^2)$（与 TTT 相同）。

#### CT-KV（核心贡献）

将示例对送入模型，直接提取每一层的 KV cache 作为可训练前缀：

$$\theta_{\text{context}} = \Theta_{\text{CT}} = \{K_j, V_j\}_{j=1}^L$$

优化 KV 前缀而非 prompt embedding。由于 KV 前缀直接在每层注入，**无需重新计算层间传播**，时间复杂度降为 $O(k)$（线性），同时因为每层都有独立的条件化信号，效果优于仅在输入层操作的 CT-Prompt。

| 特性 | CT-Prompt | CT-KV | TTT |
|------|-----------|-------|-----|
| 优化对象 | 输入层 soft prompt | 每层 KV 前缀 | 模型权重 (LoRA) |
| 初始化来源 | 示例 embedding | 示例 KV cache | 预训练权重 |
| 时间复杂度 | $O(k^2)$ | $O(k)$ | $O(k^2)$ |
| 模型参数冻结 | ✅ | ✅ | ❌ |
| 每层条件化 | ❌ | ✅ | ✅ |

#### Leave-One-Out Masking

直接优化包含所有示例的上下文时，模型可以"偷看"——当要预测第 $i$ 个示例的 $y_i$ 时，$\theta_{\text{context}}$ 中已经编码了 $(x_i, y_i)$ 的信息，模型只需从上下文中检索答案即可。

解决方案：构造 $\theta_{\text{context}}^{(i)}$，在优化第 $i$ 个示例时，将对应的 KV token（CT-KV）或 prompt token（CT-Prompt）从注意力视野中 **mask 掉**。这迫使模型从其他 $k-1$ 个示例中学习任务模式，而非简单记忆。

与 TTT 的 leave-one-out 策略区别：TTT 是在上下文中**物理删除**一个示例再更新权重，而 Context Tuning 是在已导出的连续向量上做**注意力 mask**，不改变模型参数。

#### Token Dropout

由于 Context Tuning 引入的可训练 token 数通常远多于传统 Prompt Tuning（与示例对的 token 数成正比），容易过拟合。引入 Token Dropout：在优化时以固定概率随机丢弃 $\theta_{\text{context}}^{(i)}$ 中的 token，鼓励学到的上下文表示具有鲁棒性和冗余性。

### 损失函数 / 训练策略

最终优化目标（以 CT-KV 为例）：

$$\Theta_{\text{CT}}^* = \arg\min_{\Theta_{\text{CT}}} \sum_{i=1}^{k} -\log p_\phi(y_i \mid [\text{TokenDrop}(\Theta_{\text{CT}}^{-i}); x_i])$$

推理时使用全部优化后的前缀：$\hat{y}_q = \arg\max_y p_\phi(y \mid [\Theta_{\text{CT}}^*; x_q])$。

**TTT + CT-KV 组合**：由于 TTT 优化模型权重、CT-KV 优化上下文表示，二者互补。先用 TTT 更新模型权重 $\phi^*$，再用 CT-KV 在 $\phi^*$ 上优化上下文，可进一步提升性能。

## 实验关键数据

### 主实验

评估覆盖 4 个基准、模型规模从 1B 到 8B、示例数 $k$ 从 2 到 16：

| 方法 | NLP-LR (Acc%) | NLP-LR (T/s) | MMLU (Acc%) | MMLU (T/s) | BBH (Acc%) | BBH (T/s) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-Shot | 34.9 | 0 | 35.8 | 0 | 40.9 | 0 |
| ICL | 35.6 | 0 | 41.2 | 0 | 50.4 | 0 |
| Prompt Tuning (m=32) | 41.4 | 147 | 39.2 | 15 | 50.8 | 7 |
| Prefix Tuning (m=32) | 42.0 | 123 | 39.9 | 5 | 52.7 | 7 |
| LoRA | 42.8 | 156 | 40.1 | 16 | 51.7 | 9 |
| DoRA | 42.9 | 161 | 40.3 | 16 | 52.6 | 9 |
| TTT | 44.1 | 342 | 43.6 | 30 | 57.8 | 14 |
| **CT-Prompt** | 43.2 | 228 | 43.6 | 33 | 56.3 | 14 |
| **CT-KV** | **44.2** | **145** | **43.7** | **9** | **57.9** | **7** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| CT-KV (完整) | 最优 | Leave-One-Out + Token Dropout 均启用 |
| 去掉 Leave-One-Out | 显著下降 | 模型直接从上下文中检索答案，无法泛化 |
| 去掉 Token Dropout | 轻微下降 | 过拟合风险增加，尤其示例数少时 |
| ARC 上关闭 Leave-One-Out | 反而提升 | ARC 示例极少（<4），mask 反而丢失关键信息 |
| CT-KV 随机初始化 | 大幅下降 | 验证了 ICL 初始化的必要性 |

### 关键发现

1. **CT-KV 在精度和效率上实现最优权衡**：在 NLP-LR 上用 145s 达到 44.2%（TTT 用 342s 达到 44.1%），训练时间减少 57%、精度持平。在 BBH 上用 7s 达到 57.9%（TTT 用 14s 达到 57.8%），速度快一倍。
2. **ICL 初始化 > 随机初始化**：相同优化过程，用示例 KV cache 初始化比随机初始化高出 2-5 个百分点，证明 ICL 提取的信息是有用的梯度优化起点。
3. **TTT + CT-KV 互补**：组合后性能超过任一单独方法，在 ARC 上 TTT 23.8% → TTT+CT-KV 进一步提升，说明优化权重和优化上下文是正交的改进方向。
4. **线性 vs 二次复杂度差异显著**：随着示例数 $k$ 增大，CT-KV 的效率优势更加明显；CT-Prompt 和 TTT 的计算代价会迅速增长。

## 亮点与洞察

- **概念极简但有效**：核心想法就是"用示例初始化再优化"，没有引入任何新架构，纯属对已有技术的巧妙组合。
- **KV cache 是比 prompt embedding 更好的优化对象**：每层独立条件化 + 避免层间重计算，同时带来精度和效率双重优势。
- **ICO 框架的统一视角**：将 ICL、Prompt Tuning、TTT 统一到一个优化目标下，清晰地揭示了它们的联系与区别，具有理论启发性。
- **Leave-One-Out Masking 的自适应性**：在示例充足时有效防止信息泄露；在示例极少时（如 ARC）反而应关闭，体现了对任务特性的细粒度理解。

## 局限与展望

1. **仅在分类/选择题型上验证**：生成式任务（摘要、翻译、开放式问答）的效果未知。
2. **依赖 few-shot 示例质量**：KV cache 初始化本质上放大了 ICL 对示例选择的敏感性。
3. **单任务适应**：每个任务独立优化上下文，无法跨任务共享或迁移。
4. **与更大模型的 scaling**：实验最大用到 Llama3-8B，在 70B+ 模型上的行为有待验证。
5. **Token Dropout 概率需要调优**：不同任务/模型可能需要不同的 dropout 率，增加了超参搜索成本。

## 相关工作与启发

- **Prompt Tuning / Prefix Tuning**: Context Tuning 的直接改进对象，核心差异在于初始化策略。
- **TTT (Akyürek et al., 2024)**: 互补方法；TTT 优化权重，CT 优化上下文，组合效果最佳。
- **ICL 理论 (Dai et al., 2023)**: ICL 可被解释为隐式梯度下降，Context Tuning 则将其变为显式优化。
- 启发：在其他模态（视觉 Transformer、多模态模型）中，用任务示例初始化 KV 前缀进行 test-time 适应可能同样有效。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 想法简单但 ICO 统一框架有理论贡献，CT-KV 的线性复杂度分析有价值
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4 个基准、4 种模型规模、大量消融、5 seed 统计
- **写作质量**: ⭐⭐⭐⭐⭐ — 数学形式化清晰，framework presentation 优秀
- **价值**: ⭐⭐⭐⭐ — 即插即用、训练高效、可与 TTT 组合，实用性强

<!-- RELATED:START -->

## 相关论文

- [Core Context Aware Transformers for Long Context Language Modeling](core_context_aware_transformers_for_long_context_language_modeling.md)
- [Sample Efficient Demonstration Selection for In-Context Learning](sample_efficient_demonstration_selection_for_in-context_learning.md)
- [SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](../../ACL2025/model_compression/see_strategic_exploration_exploitation_prompt_optimization.md)
- [ParallelComp: Parallel Long-Context Compressor for Length Extrapolation](parallelcomp_parallel_long-context_compressor_for_length_extrapolation.md)
- [DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)

<!-- RELATED:END -->
