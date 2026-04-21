---
title: >-
  [论文解读] Core Context Aware Transformers for Long Context Language Modeling
description: >-
  [ICML 2025][模型压缩][长上下文建模] 提出 Core Context Aware (CCA) Attention，通过全局感知池化将输入 token 动态压缩为少量核心 token，结合局部保持模块捕获邻近细粒度信息，实现即插即用地替换标准自注意力，在 128K 上下文下获得 7.9× 加速和 46% 显存节省，同时保持建模性能。
tags:
  - ICML 2025
  - 模型压缩
  - 长上下文建模
  - 高效注意力
  - KV缓存压缩
  - 核心上下文
  - 线性复杂度
---

# Core Context Aware Transformers for Long Context Language Modeling

**会议**: ICML 2025  
**arXiv**: [2412.12465](https://arxiv.org/abs/2412.12465)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 长上下文建模, 高效注意力, KV缓存压缩, 核心上下文, 线性复杂度

## 一句话总结

提出 Core Context Aware (CCA) Attention，通过全局感知池化将输入 token 动态压缩为少量核心 token，结合局部保持模块捕获邻近细粒度信息，实现即插即用地替换标准自注意力，在 128K 上下文下获得 7.9× 加速和 46% 显存节省，同时保持建模性能。

## 研究背景与动机

当 Transformer 的上下文长度 $L$ 扩展到极大值（如 128K）时，自注意力机制面临两个核心问题：

**冗余信息激增**：并非所有上下文对目标 token 都同等重要。注意力得分在大多数层和头中呈现高度稀疏分布，大量计算被浪费在冗余上下文上。

**计算与存储开销**：标准自注意力的计算复杂度为 $O(L^2)$，KV 缓存存储为 $O(L)$，在超长上下文下成为严重瓶颈。

现有方法的不足：

- **StreamingLLM / LM-Infinite**：仅保留首尾若干 token 的注意力，完全忽略中间 token 间的信息交换，在需要全文理解的 QA 任务中性能严重下降。
- **MInference**：采用离线确定的固定稀疏模式（A-shape、vertical-slash、block-sparse），无法动态适应不同输入。
- **LongLoRA**：分组注意力+移位策略，但组间通信仍然受限。
- **线性注意力（RetNet 等）**：需要从头训练，无法复用预训练 LLM 的知识。

作者观察到，注意力得分中存在明显的"核心上下文"与"冗余上下文"之分（如 Figure 1 所示），核心上下文集中在少量高注意力 token 上。这一观察启发了动态识别并聚焦核心上下文的方法设计思路。

## 方法详解

### 整体框架

CCA-Attention 由两个互补模块组成，通过可微融合策略结合：

1. **全局感知池化模块 (Globality-aware Pooling Module)**：将输入 token 序列分组，在每组内动态压缩为一个核心 token，用核心 token 代替原始 token 进行注意力计算，捕获长程全局依赖。
2. **局部保持模块 (Locality-preserving Module)**：保留查询 token 邻近的 $s$ 个 token 进行细粒度注意力，补充全局模块丢失的局部信息。
3. **可微融合 (Differentiable Fusion)**：将两个模块的 Key-Value 拼接后统一做 softmax 注意力，保证每个 token 都能访问所有前序 token 的信息。

CCA-Attention 是即插即用的：输入输出维度与标准自注意力完全一致，可直接替换预训练 LLM 的注意力层，仅需少量微调。

### 关键设计

#### 1. 全局感知池化：动态核心 token 生成

给定输入序列 $\mathbf{X} = [\mathbf{x}_1; \mathbf{x}_2; \dots; \mathbf{x}_L]$，将其分为 $m = \lfloor L/g \rfloor$ 个组，每组包含 $g$ 个 token。

对第 $i$ 组，利用组内最后一个 token $\mathbf{x}_{ig}$ 的 query 来评估组内每个 token 的重要性，通过加权池化生成核心 token $\mathbf{c}_i$：

$$\mathbf{c}_i = \text{softmax}\left(\frac{\mathbf{Q}_{ig} \mathbf{K}_{\mathcal{I}_i}^{\prime\top}}{\sqrt{d}}\right) \mathbf{X}_{\mathcal{I}_i}^{G}$$

其中 $\mathbf{Q}_{ig} = \mathbf{x}_{ig} \mathbf{W}^Q$，$\mathbf{K}_{\mathcal{I}_i}^{\prime} = \mathbf{X}_{\mathcal{I}_i}^{G} \mathbf{W}^K$。

**设计动机**：注意力图可视化表明，重要 token 从后续 token 那里一致地获得高注意力得分，因此用组内最后一个 token 作为"评估者"是自然且有效的选择。

核心 token 序列 $\mathbf{C} = [\mathbf{c}_1; \dots; \mathbf{c}_m]$ 将原始 $L \times d$ 的表示压缩为 $m \times d$，显著降低后续注意力的计算量和 KV 缓存大小。

用核心 token 计算全局 Key 和 Value：

$$\mathbf{K}^G = \mathbf{C} \mathbf{W}^K, \quad \mathbf{V}^G = \mathbf{C} \mathbf{W}^V$$

对查询 $\mathbf{Q}_i$，全局注意力仅使用索引 $j = \max(0, \lfloor(i-s)/g\rfloor)$ 之前的核心 token，排除距离较近的 token（这部分由局部模块负责）。

#### 2. 局部保持模块：细粒度邻近注意力

每个查询 $\mathbf{Q}_i$ 对其前面至少 $s$ 个 token 进行完整注意力。局部窗口大小自适应调整为 $s + ((i-s) \bmod g)$，确保所有 token 都参与注意力计算不遗漏。

$$\mathbf{K}_{\mathcal{U}_i}^L = [\mathbf{K}_k^L; \cdots; \mathbf{K}_i^L], \quad k = \max(1, i - s - ((i-s) \bmod g))$$

**重要设计**：局部模块与全局模块共享投影参数 $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$，不引入额外参数。

#### 3. 可微融合策略

将全局和局部两个模块的 K/V 拼接，通过单个 softmax 统一计算注意力输出：

$$\mathbf{Att}_i = \text{softmax}\left(\frac{\mathbf{Q}_i [\widetilde{\mathbf{K}}_{\mathcal{T}_i}^G; \widetilde{\mathbf{K}}_{\mathcal{U}_i}^L]^\top}{\sqrt{d}}\right) [\widetilde{\mathbf{V}}_{\mathcal{T}_i}^G; \widetilde{\mathbf{V}}_{\mathcal{U}_i}^L]$$

这种拼接式融合（而非加权求和）保证了全局与局部信息在统一的 softmax 归一化下竞争，实现端到端可微的自适应分配。论文证明（Proposition 1），CCA-Attention 等价于一种特殊的全注意力变体，每个 token 都能访问所有前序 token 的信息，保证信息完整性。

#### 4. 推理阶段的灵活性

- 推理时可动态调整 $g$ 和 $s$，生成不同效率-精度权衡的模型变体，适应不同流量场景。
- 使用 Triton kernel 实现，加速训练和推理的并行计算。
- KV 缓存仅存储核心 token 的 $\mathbf{K}^G, \mathbf{V}^G$ 和局部窗口的 $\mathbf{K}^L, \mathbf{V}^L$，存储复杂度从 $O(L)$ 降为 $O(L/g + s)$。

### 损失函数 / 训练策略

CCA-Attention 支持三种训练策略：

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| **从头训练** | 在大规模语料上全量训练 CCA-Attention | 追求最佳性能，资源充足 |
| **全量微调** | 基于预训练 LLM 参数，微调所有参数 | 性能与效率的平衡 |
| **部分微调** | 仅微调 $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ | 资源受限，快速适配 |

核心优势：与线性注意力（需从头训练）不同，CCA-Attention 可直接利用预训练 LLM 的知识，仅需少量微调即可部署。

## 实验关键数据

### 主实验

在 LongBench-E 基准上的 LLaMA2-7B-32K 结果（32K 上下文）：

| 方法 | 平均分 | 首 token 延迟(s) | 显存(GB) | 加速比 | 显存节省 |
|------|--------|------------------|----------|--------|----------|
| Vanilla Self-Attention | 22.11 | 9.15 | 35.58 | 1.0× | — |
| StreamingLLM | 14.95 | 5.75 | 22.94 | 1.6× | 35%↓ |
| LM-Infinite | 18.76 | 4.72 | 26.35 | 1.9× | 26%↓ |
| MInference | 21.14 | 4.20 | 33.52 | 2.2× | 6%↓ |
| **CCA-LLM (Ours)** | **21.86** | **2.59** | **19.12** | **3.5×** | **46%↓** |

在 LLaMA2-7B-80K 上的 64K 上下文结果：

| 方法 | 平均分 | 首 token 延迟(s) | 显存(GB) | 加速比 | 显存节省 |
|------|--------|------------------|----------|--------|----------|
| Vanilla Self-Attention | 22.42 | 32.43 | 60.03 | 1.0× | — |
| StreamingLLM | 14.94 | 9.04 | 37.45 | 3.6× | 37%↓ |
| LM-Infinite | 21.20 | 8.27 | 41.54 | 3.9× | 31%↓ |
| MInference | 22.08 | 8.14 | 54.09 | 4.0× | 10%↓ |
| **CCA-LLM (Ours)** | **22.24** | **6.42** | **33.86** | **5.7×** | **44%↓** |

在最新模型上的验证（LLaMA3.1-8B-128K 和 Qwen2.5-7B-128K，32K 上下文）：

| 方法 | 基座模型 | 平均分 | 加速比 | 显存节省 |
|------|----------|--------|--------|----------|
| Vanilla | LLaMA3.1-8B | 37.93 | 1.0× | — |
| MInference | LLaMA3.1-8B | 37.74 | 1.9× | 11%↓ |
| **CCA-LLM** | **LLaMA3.1-8B** | **37.81** | **3.1×** | **49%↓** |
| Vanilla | Qwen2.5-7B | 38.38 | 1.0× | — |
| MInference | Qwen2.5-7B | 36.72 | 2.2× | 8%↓ |
| **CCA-LLM** | **Qwen2.5-7B** | **38.08** | **3.9×** | **45%↓** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 仅全局模块 | 性能下降明显 | 缺少局部细粒度信息，短程依赖建模不足 |
| 仅局部模块 | 长程任务退化 | 无法捕获远距离语义关联 |
| 全局+局部拼接融合 | 最佳性能 | 统一 softmax 下的竞争式融合优于加权求和 |
| $g$=8, $s$=512 | 精度高、速度中等 | 组大小较小保留更多信息 |
| $g$=64, $s$=512 | 速度最快、精度微降 | 压缩比更大，适合极端效率场景 |
| 部分微调 vs 全量微调 | 精度差距小 | 部分微调仅调 QKV 权重即可接近全量微调效果 |

### 关键发现

1. **效率-性能权衡优于所有基线**：CCA-LLM 在保持与全注意力接近的平均分的同时，实现了最大的加速和显存节省。MInference 虽然保持了精度，但显存节省仅 6-11%，远不及 CCA 的 44-49%。
2. **StreamingLLM / LM-Infinite 精度损失严重**：它们通过丢弃中间 token 获得效率，但平均分从 22 降至 14-18，在多文档 QA 和摘要任务上几乎不可用。
3. **跨模型泛化性强**：在 LLaMA2、LLaMA3.1、Qwen2.5 三个不同架构的模型上均展现一致优势。
4. **128K 上下文可达 7.9× 加速**：随着上下文长度增加，CCA-Attention 的效率优势更为显著。

## 亮点与洞察

1. **核心 token 的动态生成**：不同于静态稀疏模式（如 BigBird 的固定 strided attention），CCA 用组内加权池化动态生成核心 token，使压缩后的表示能自适应地保留最重要的信息。这是本文最核心的创新。
2. **即插即用设计**：CCA-Attention 与标准 self-attention 保持相同的输入输出接口和参数维度，可直接替换预训练 LLM 中的注意力层，仅需部分微调 QKV 权重。这大幅降低了部署门槛。
3. **全局-局部互补的思路**：粗粒度全局信息 + 细粒度局部信息的双模块设计，通过拼接+统一 softmax 的融合方式，优雅地保证了信息完整性（每个 token 都能间接访问所有前序 token）。
4. **推理时可动态调参**：$g$ 和 $s$ 的灵活调整允许同一模型在不同场景下切换效率模式，这在实际部署中非常实用。

## 局限与展望

1. **核心 token 生成的信息损失不可避免**：组内加权池化虽然动态，但仍是有损压缩；对于需要精确引用原文细节（如精确数字、代码片段）的任务，可能丢失关键信息。
2. **组大小 $g$ 的选择缺乏理论指导**：$g$ 的最优值在不同任务和模型上可能不同，当前主要依赖经验调参。
3. **Triton kernel 的硬件依赖**：自定义的 Triton 实现限制了在不同 GPU 架构和推理框架上的通用性。
4. **与 GQA/MQA 的兼容性**：论文未明确讨论 CCA-Attention 在 grouped-query attention 或 multi-query attention 架构下的效果。
5. **可探索方向**：将核心 token 思路推广到多模态场景（视觉 token 压缩）或结合 KV 缓存驱逐策略（如 H2O）进一步压缩。

## 相关工作与启发

- **高效注意力方法线**：Longformer → BigBird → LongLoRA → MInference → CCA，趋势从静态稀疏模式走向动态自适应。
- **KV 缓存压缩**：与 H2O、SnapKV 等 KV 缓存驱逐方法互补；CCA 从注意力机制层面压缩，而它们从缓存管理层面压缩。
- **上下文压缩**：与 gist token（Mu et al.）和 AutoCompressor（Chevalier et al.）有思路上的相似性，但 CCA 的核心 token 不需要额外的辅助网络或特殊 token。
- **启发**：核心 token 的思路可推广到多模态场景（视觉 token 压缩），以及 MoE 架构中的 expert 选择问题。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 全局池化+局部保持的双模块思路有一定新意，但属于稀疏注意力的自然延伸
- 实验充分度: ⭐⭐⭐⭐ — 跨多个模型和基准验证，消融完整，但缺少更多 128K+场景的详细分析
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰，公式推导严谨，图示直观
- 价值: ⭐⭐⭐⭐ — 即插即用+大幅效率提升的组合有很强的实用价值，但需要更多实际部署验证

<!-- RELATED:START -->

## 相关论文

- [LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models](lacache_ladder-shaped_kv_caching_for_efficient_long-context_modeling_of_large_la.md)
- [ParallelComp: Parallel Long-Context Compressor for Length Extrapolation](parallelcomp_parallel_long-context_compressor_for_length_extrapolation.md)
- [RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [Context Tuning for In-Context Optimization](context_tuning_for_in-context_optimization.md)
- [MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning](mka_memory-keyed_attention_for_efficient_long-context_reasoning.md)

<!-- RELATED:END -->
