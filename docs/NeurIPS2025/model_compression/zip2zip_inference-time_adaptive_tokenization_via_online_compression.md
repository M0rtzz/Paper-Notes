---
title: >-
  [论文解读] zip2zip: Inference-Time Adaptive Tokenization via Online Compression
description: >-
  [NeurIPS 2025][模型压缩][自适应分词] 提出 zip2zip，将经典 LZW 在线无损压缩算法深度集成到 LLM 的推理流程中，通过在解码过程中持续将高频共现 Token 合并为可复用的"hypertoken"来动态扩展词表，配合动态嵌入层和压缩空间语言建模训练，仅需 10 GPU-hours 的 LoRA 微调即可使现有 LLM 获得推理时自适应分词能力，实现输入输出序列长度缩减 15-40%、端到端解码延迟降低最高 40%，且下游任务性能几乎无损。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 自适应分词
  - LZW压缩
  - 超级Token
  - 推理加速
  - 词表扩展
  - 动态嵌入
---

# zip2zip: Inference-Time Adaptive Tokenization via Online Compression

**会议**: NeurIPS 2025  
**arXiv**: [2506.01084](https://arxiv.org/abs/2506.01084)  
**代码**: https://github.com/epfl-dlab/zip2zip  
**领域**: 模型压缩  
**关键词**: 自适应分词, LZW压缩, 超级Token, 推理加速, 词表扩展, 动态嵌入

## 一句话总结

提出 zip2zip，将经典 LZW 在线无损压缩算法深度集成到 LLM 的推理流程中，通过在解码过程中持续将高频共现 Token 合并为可复用的"hypertoken"来动态扩展词表，配合动态嵌入层和压缩空间语言建模训练，仅需 10 GPU-hours 的 LoRA 微调即可使现有 LLM 获得推理时自适应分词能力，实现输入输出序列长度缩减 15-40%、端到端解码延迟降低最高 40%，且下游任务性能几乎无损。

## 研究背景与动机

大型语言模型（LLM）的分词器（tokenizer）是影响模型性能与推理成本的关键组件。当前主流 LLM 几乎都使用基于 BPE（Byte Pair Encoding）的静态分词器，其词表在大规模通用 Web 语料上进行一次性优化后即固定不变。这种"一刀切"的静态词表在处理通用文本时表现尚可，但面对特定领域或特定语言的输入时，分词效率会显著下降。例如在生物医学文本、编程代码、数学公式等领域，静态分词器常常将专业术语拆分为过多的碎片化 Token，导致同一段文本的 Token 序列长度比通用文本高出 2-3 倍，直接推高了推理的计算成本、延迟和用户费用。

更紧凑的分词带来三大核心优势：（1）更大的有效上下文窗口——同样的上下文长度限制下可以容纳更多内容；（2）更低的计算和经济成本——注意力机制的复杂度与序列长度直接相关；（3）更短的响应时间——自回归解码的延迟与输出 Token 数量成正比。

针对静态分词效率不足的问题，已有工作探索了两条路线但都存在局限：第一条是**领域适配分词器**，即为每个目标领域或语言训练专用的扩展词表。这种方法虽然有效，但需要为每个领域单独维护分词器，不具备可扩展性。第二条是**扩大通用词表**——商业 LLM 的词表规模已从 32K 扩展到 128K（Llama-3）甚至 200K（Phi-4），但研究表明简单扩大词表的收益是递减的，过大的词表甚至可能损害模型性能。

这些局限指向一个核心需求：一种**推理时自适应**的分词机制，能够根据当前输入文本的特征动态调整词表，无需重新训练模型或维护多套分词器。但实现这一目标面临架构层面的挑战——Transformer 模型的嵌入层和语言建模头都是与固定词表大小绑定的静态矩阵，如何在推理时动态扩展词表并让模型正确处理新 Token，是本文要解决的核心问题。本文受 ZIP 压缩工具所用的 LZW 算法启发，提出了 zip2zip 框架，实现了真正的推理时自适应分词。

## 方法详解

### 整体框架

zip2zip 的名称致敬 seq2seq，其中两个"zip"分别代表输入端压缩（第一个 zip）和输出端压缩（第二个 zip），即同时减少输入和输出的 Token 数量。整体框架包含三大核心组件，协同工作实现推理时的动态分词。

系统工作流程如下：输入文本首先通过标准 BPE 分词器转化为基础 Token 序列（base tokens），然后经过在线 LZW 压缩模块将高频共现的 Token 合并为 hypertoken，形成更紧凑的表示。由于 hypertoken 不在原始词表中，其嵌入向量由 hyper-encoder 在推理时动态计算。嵌入后的 base token 和 hypertoken 混合序列送入标准 Transformer 层处理，得到上下文化的隐藏状态。在输出层，hypertoken 的 unembedding 向量被动态追加到原始语言建模头的矩阵中，使模型能够在 base vocabulary 和 hyper vocabulary 的并集上进行联合 softmax 采样。采样结果可能是 base token 也可能是 hypertoken，生成结束后通过 LZW 解码将所有 hypertoken 还原为 base token 序列。

关键的设计约束是**词表一致性更新**：嵌入层和 unembedding 层的 hypertoken 集合必须保持同步，否则会出现两类错误——要么生成了无法解码的 hypertoken，要么试图解码一个不存在的 hypertoken。此外系统采用**hyper-embedding 缓存**机制：由于 hypertoken 嵌入是上下文无关的（只取决于其组成的基础 Token），可以像 KV-cache 一样跨解码步骤缓存复用，每步只需计算新创建的 hypertoken 嵌入，保持了常数级的每步计算开销。

### 关键设计

**LZW 分词器：在线增量词表构建。** zip2zip 采用 Lempel-Ziv-Welch（LZW）压缩算法作为动态词表扩展的核心引擎。LZW 是一种基于字典的无损压缩方法，通过增量构建码本（codebook）将重复出现的序列模式编码为更短的符号。在 zip2zip 中，码本用 base vocabulary 初始化，然后在推理过程中持续扫描 Token 流，将高频共现的 Token 对合并为新的 hypertoken 并加入码本。最大合并长度 $M$ 限制了单个 hypertoken 可以包含的最大基础 Token 数量（默认 $M=3$），用于控制动态词表的增长速度。

LZW 算法相比其他压缩方法（如 BPE 本身）具有三个关键优势使其特别适合 zip2zip：（1）**在线性**——在步骤 $t$ 创建的 hypertoken 可以在步骤 $t+1$ 立即被复用，而 BPE 需要访问完整序列、只能离线运行；（2）**自包含性**——压缩后的 Token 序列可以独立解码还原原始输入，无需额外传输码本，保持了与现有 LLM 库和接口的兼容性；（3）**无歧义性**——当 base token 和 hypertoken 都可用时，LZW 算法提供了一致、确定性的选择规则。此外 hypertoken 还支持递归合并——已有的 hypertoken 可以与其他 Token 进一步合并形成嵌套的超级 Token，实现多层级压缩。

**动态嵌入层（Hyper-Encoder）：从基础嵌入到超级嵌入的映射。** hypertoken 不在原始模型的嵌入矩阵中，因此需要一种方法在运行时计算其嵌入向量。zip2zip 引入了 hyper-encoder $f_\phi: \mathcal{V}^M \to \mathbb{R}^d$，这是一个两层 Transformer encoder，接受 hypertoken 所包含的 $M$ 个基础 Token 的嵌入向量作为输入，输出对应的 hypertoken 嵌入 $h = f_\phi(y_{1:M}) \in \mathbb{R}^d$。对于包含少于 $M$ 个基础 Token 的 hypertoken，输入序列会被 padding 到长度 $M$。从数学角度看，hyper-encoder 本质上是将 $M \times d$ 维空间（拼接的基础 Token 嵌入）映射到 $d$ 维空间的非线性降维器。对于输出端的 unembedding 层，如果模型采用 tied embedding（嵌入矩阵与 unembedding 矩阵共享），则可以直接复用同一个 hyper-encoder 的输出；否则需要训练一个独立的 hyper-encoder 来生成 unembedding 向量。

### 损失函数与训练策略

zip2zip 的训练采用联合优化目标，包含两部分损失：

**主损失：压缩序列上的语言建模损失。** zip2zip 在 LZW 压缩后的 Token 序列上进行训练，而非在原始基础 Token 序列上训练。给定语料分布 $\mathcal{D}$ 和在线压缩算法 $\mathcal{C}$（即 LZW），训练目标为最小化压缩序列上的因果语言建模损失：

$$\min_{\theta, \phi} \mathbb{E}_{y \sim \mathcal{D}} \left[ -\log \pi_{\theta, \phi}(\mathcal{C}(y)) \right]$$

其中 $\theta$ 是基础模型参数，$\phi$ 是 hyper-encoder 参数。训练数据的预处理只需一次：先用标准分词器对语料分词，再应用 LZW 压缩，压缩在文档级别独立进行以避免跨文档学习无关模式。训练过程仍然完全可并行——利用标准因果掩码机制，模型可以在每个位置并行预测下一个 Token（无论是 base token 还是 hypertoken）。为了避免推理时需要按序更新码本，训练时预计算完整输入序列的固定码本，确保效率和与标准训练流水线的兼容性。

**辅助损失：自编码重建损失。** 为了确保 hypertoken 嵌入保留其组成基础 Token 的语义信息，zip2zip 引入了一个自编码性质的辅助损失。具体而言，引入一个解码器 $f_\psi: \mathbb{R}^d \to \mathcal{V}^M$，训练它从 hypertoken 嵌入中重建原始基础 Token。完整的联合优化目标为：

$$\min_{\theta, \phi, \psi} \mathbb{E}_{y \sim \mathcal{D}} \left[ -\log \pi_{\theta, \phi}(\mathcal{C}(y)) \right] + \lambda \mathbb{E}_{y_{1:M}} \left[ \Delta(y_{1:M}, f_\psi(f_\phi(y_{1:M}))) \right]$$

其中 $\Delta$ 是重建损失函数（交叉熵损失），$\lambda = 0.1$ 控制两部分损失的权衡。重建损失可以理解为一种自编码约束——hypertoken 作为压缩的潜在表示，重建目标鼓励它保持语义内容完整，使压缩过程尽可能无损。实验中观察到重建损失在训练过程中收敛到接近零，表明模型几乎可以完美恢复 hypertoken 对应的原始基础 Token 序列，验证了习得的压缩具有高度的信息保持性。

**参数高效训练策略。** 为了降低训练成本，zip2zip 采用 LoRA 对基础模型进行参数高效微调，同时训练新引入的 hyper-embedding 和 hyper-unembedding 模块。整个训练过程对 Phi-3-4B 仅需约 10 H100-GPU 小时、对 Phi-3-14B 需约 40 H100-GPU 小时，使用 1 亿个训练 Token 即可完成。这意味着任何一个拥有少量 GPU 资源的研究实验室都可以将现有 LLM 升级为具备自适应分词能力的 zip2zip 模型。

**熵不变性理论保证。** 论文证明了一个重要的理论结果——对于任意无损映射 $g$，在压缩空间中总存在一个对应的传输模型分布，其交叉熵与原始空间中完全相同。形式化地，若 $g$ 是双射的无损压缩，则 $H(P_Z) = H(P_X)$，$H(P_Z, p_\gamma) = H(P_X, p_\theta)$。这意味着无损压缩不会从根本上改变模型的可达性能——最优模型在压缩表示下可以达到与原始空间相同的似然度。当然实际训练中由于优化的挑战（梯度下降可能收敛更慢或陷入次优区域），训练出的模型可能无法完美逼近理论最优的传输模型。

## 实验关键数据

### 主实验：Token 效率提升

论文在 5 个代表性领域（代码、数学、聊天、多语言、Web）上评估了 4 种主流 LLM 分词器加上 zip2zip 后的 Token 效率（bytes per token，越高越好）：

| 分词器 | 领域 | 基础效率 | +zip2zip | 提升幅度 |
|--------|------|---------|---------|---------|
| Llama-3-128K | 代码 | 4.1 | 6.3 | +54% |
| Llama-3-128K | 数学 | 2.7 | 4.0 | +48% |
| Llama-3-128K | 聊天 | 5.1 | 6.4 | +25% |
| Llama-3-128K | 多语言 | 3.8 | 4.7 | +24% |
| Llama-3-128K | Web | 4.6 | 5.4 | +17% |
| Qwen-2-150K | 代码 | 4.0 | 6.2 | +55% |
| Qwen-2-150K | 数学 | 2.3 | 3.7 | +61% |
| Phi-4-200K | 代码 | 4.1 | 6.3 | +54% |
| Phi-4-200K | 数学 | 2.7 | 4.1 | +52% |
| Gemma-3-256K | 代码 | 3.3 | 5.6 | +70% |
| Gemma-3-256K | 数学 | 2.3 | 3.7 | +61% |

zip2zip 在所有分词器和所有领域上都显著提升了 Token 效率。结构化领域（代码和数学）的增益最为突出，Gemma-3 在代码领域甚至达到了 70% 的提升。值得注意的是，词表更大的分词器并不总是有更好的 Token 效率——Gemma-3（256K 词表）的基础代码效率 3.3 反而低于 Llama-3（128K 词表）的 4.1，进一步证明了简单扩大词表并不能有效解决领域适配问题。

### 消融实验：下游任务性能与困惑度对比

| 模型 | 方法 | ARC-c | ARC-e | HellaSwag | OBQA | PIQA | WG | GSM8K |
|------|------|-------|-------|-----------|------|------|-----|-------|
| Phi-3.5-4B | 基线 | 0.60 | 0.83 | 0.66 | 0.46 | 0.79 | 0.75 | 0.82 |
| Phi-3.5-4B | 持续预训练 | 0.60 | 0.82 | 0.63 | 0.47 | 0.82 | 0.75 | 0.40 |
| Phi-3.5-4B | zip2zip | 0.57 | 0.83 | 0.61 | 0.46 | 0.82 | 0.75 | 0.15 |
| Phi-3-14B | 基线 | 0.62 | 0.80 | 0.70 | 0.51 | 0.83 | 0.76 | 0.84 |
| Phi-3-14B | 持续预训练 | 0.62 | 0.88 | 0.66 | 0.52 | 0.87 | 0.80 | 0.52 |
| Phi-3-14B | zip2zip | 0.62 | 0.86 | 0.68 | 0.51 | 0.85 | 0.79 | 0.25 |

Byte-level 困惑度（越低越好）：

| 模型 | 方法 | Wikitext | The Pile | mC4 | dC4 |
|------|------|---------|---------|-----|-----|
| Phi-3.5-4B | 基线 | 1.58 | 1.79 | 1.88 | 1.74 |
| Phi-3.5-4B | zip2zip | 1.69 | 1.95 | 2.00 | 1.82 |
| Phi-3-14B | 基线 | 1.43 | 1.72 | 1.82 | 1.67 |
| Phi-3-14B | zip2zip | 1.56 | 1.90 | 1.96 | 1.75 |

### 推理效率（H100 GPU 上 Phi-3.5-4B）

| 设置 | 方法 | Prefill (tok/s) | Decode (tok/s) | Prefill 提升 | Decode 提升 |
|------|------|----------------|----------------|-------------|-------------|
| 256+256 | 基线 | 700.9 | 56.2 | - | - |
| 256+256 | zip2zip | 936.6 | 61.4 | +33.6% | +9.3% |
| 512+256 | 基线 | 1347.2 | 54.4 | - | - |
| 512+256 | zip2zip | 2722.1 | 79.8 | +102.6% | +46.6% |
| 1024+256 | 基线 | 2689.4 | 52.8 | - | - |
| 1024+256 | zip2zip | 4326.7 | 61.5 | +60.9% | +16.6% |
| 2048+256 | 基线 | 4993.2 | 53.1 | - | - |
| 2048+256 | zip2zip | 9258.1 | 61.9 | +85.4% | +16.5% |

### 关键发现

1. **Token 效率提升在结构化文本中最为显著**——代码和数学领域的 bytes/token 提升达 48-70%，因为这些领域包含大量重复的固定模式（如函数名、关键字、公式符号）。聊天和 Web 文本增益较低（15-28%），因为内容更加多样化、重复模式较少。

2. **下游任务性能总体保持稳健**——在大多数 NLP 基准（ARC、PIQA、WinoGrande 等）上性能损失极小，通常在 1-3 个百分点以内。但 GSM8K（数学计算任务）出现了显著退化（Phi-3.5-4B 从 0.82 降至 0.15），这可能是因为数值计算本身就依赖 Token 级的精确操作，自适应分词可能改变了数字的 Token 分割方式从而干扰了计算能力。

3. **困惑度有温和上升但幅度可控**——byte-level perplexity 增加约 5-10%，这与理论预测一致：虽然熵不变定理保证了理论最优性能不变，但实际优化未能完美逼近传输模型。

4. **推理吞吐量提升显著，尤其在数据中心级 GPU 上**——在 H100 上 prefill 吞吐量提升最高达 102.6%（512+256 设置），decode 吞吐量最高提升 46.6%。在消费级硬件（Apple M1）上改善更温和但仍然正向。

5. **Hypertoken 具有语义意义**——观察到的 hypertoken 大多对应领域特定的语义单元，如代码中的"torch"、"Attention"、"MultiHead"，生物医学中的"mRNA"、"cellular"，法语中的"Eiffel"、"Gustave"等，表明 LZW 的合并策略在实践中捕获了有意义的语言模式。

6. **仅约 25% 的 hypertoken 在生成过程中被复用**，说明码本管理策略（如剪枝无用条目、预填充）有较大优化空间。

## 亮点与洞察

- **LZW 的在线性质是关键创新点**：与 BPE 等需要全局统计的离线方法不同，LZW 允许在步骤 $t$ 创建的 hypertoken 在步骤 $t+1$ 立即可用，这与自回归生成过程天然匹配。这种"即创即用"的特性使得动态词表扩展可以无缝嵌入到标准 LLM 解码循环中，无需修改推理架构。

- **熵不变定理为实用性提供了理论背书**：证明无损压缩不会从根本上损害模型性能，为整个方法论提供了坚实的理论基础——性能损失来源于优化不充分而非表示能力不足，意味着随着训练预算增加性能有望进一步恢复。

- **极低的训练成本是突出的实用优势**：10 GPU-hours + 100M tokens 的微调成本甚至低于许多领域适配微调的开销，使得 zip2zip 成为一种真正"即插即用"的推理加速技术。配合开源的 Rust 高性能 LZW 分词器和兼容 HuggingFace Transformers 的 drop-in 模型架构，方法具有极高的工程可复现性和可部署性。

- **输入和输出双端压缩的设计理念**：名称中的双"zip"体现了核心洞察——与只能压缩输入的 prompt compression 方法不同，zip2zip 同时减少输出 Token 数量，而在低批次负载下输出 Token 通常才是推理延迟的主要瓶颈，因此减少输出 Token 的收益远大于只压缩输入。

- **hyper-embedding 缓存与 KV-cache 类比精妙**：hypertoken 嵌入的上下文无关性使其可以像 KV-cache 一样增量更新，每步仅计算一个新 Token 嵌入，保持了 $O(1)$ 的每步额外开销，避免了随词表动态增长带来的计算爆炸。

## 局限性

- **最大合并长度 $M=3$ 限制了压缩率上限**：当前设计中单个 hypertoken 最多包含 3 个基础 Token，这限制了极端情况下的压缩潜力。虽然 hypertoken 可以递归合并，但实际中递归合并的深度有限，更大的 $M$ 可能带来更高压缩率但也需要更强的 hyper-encoder。

- **数学计算类任务性能严重退化**：GSM8K 上从 0.82 降至 0.15 的崩溃式下降表明，自适应分词可能根本性地干扰了模型的数值推理能力。数字 Token 的合并可能改变了模型习得的数字运算模式，这对依赖精确数值计算的应用场景构成了重大限制。

- **byte-level 困惑度的温和上升反映了优化差距**：虽然理论证明无损压缩不应损害性能，但实际训练未能完全恢复理论最优模型，表明在压缩表示空间中的优化景观可能更加复杂，需要更长的训练或更精细的优化策略。

- **码本效率低**：仅约 25% 的创建的 hypertoken 被实际复用，大量码本条目是浪费的。当前缺乏码本剪枝或选择性保留策略，限制了内存效率。

- **对非结构化文本增益有限**：在文本多样性高、重复模式少的对话和 Web 领域，Token 效率提升仅为 15-20%，显著低于代码和数学领域的 48-70%，说明 LZW 的字典匹配机制在缺乏重复模式时效用受限。

- **模型适用性有限**：当前仅在 Phi-3 系列（4B 和 14B）上验证，尚未在 Llama、Qwen 等更多模型架构和更大规模模型上进行验证。

## 相关工作

**领域自适应分词器。** Zhao et al. (2024)、Kim et al. (2024)、Liu et al. (2023, 2024a) 等工作通过为 Llama 分词器添加中文、韩文或特定领域（心理健康、法律）的新 Token 来适配，但这些方法产生的仍然是固定词表，不会在推理时自适应调整。

**输入压缩方法。** Gist tokens（Mu et al., 2023）、LLMLingua（Jiang et al., 2023）、In-context Autoencoders（Ge et al., 2024）等 prompt compression 方法通过有损压缩减少上下文长度，可以实现更高压缩率，但只能压缩输入 Token 而无法压缩输出 Token。Lester et al. (2024) 提出在算术编码压缩的文本上训练 LLM。

**动态嵌入 Transformer。** 多项工作探索了减少 Transformer 计算成本的动态嵌入方案：Hourglass（Nawrot et al., 2022）采用沙漏状的 Token 合并-展开架构，MegaByte（Yu et al., 2023）用多尺度 Transformer 处理字节序列，BLT（Pagnoni et al., 2025）基于字节级别的动态分块。与 zip2zip 概念上最接近的是 Dynamic Vocab（Liu et al., 2024b），它也在生成过程中动态扩展词表，但在词表构建算法和训练流程上与 zip2zip 有显著差异。ZeTT（Minixhofer et al., 2024）和 Dynamic Tokenization（Feher et al., 2025）也探索了动态分词的方向，但采用了不同的架构和分块策略。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将经典 LZW 压缩算法与 LLM 推理深度集成，首次实现了真正在线的推理时自适应分词，概念上非常优雅
- 实验充分度: ⭐⭐⭐⭐ 覆盖了多模型多领域多维度的评估（Token 效率、困惑度、下游任务、推理吞吐），但 GSM8K 的严重退化需要更深入分析
- 写作质量: ⭐⭐⭐⭐⭐ 概念阐述清晰，图示直观易懂，理论和实验结合紧密，LZW 工作流程的可视化尤其出色
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高——10 GPU-hours 即可升级现有模型，开源完整工具链，即插即用的推理加速方案

<!-- RELATED:START -->

## 相关论文

- [Inference-Time Hyper-Scaling with KV Cache Compression](inference-time_hyper-scaling_with_kv_cache_compression.md)
- [When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization](when_worse_is_better_navigating_the_compression-generation_tradeoff_in_visual_to.md)
- [A Partition Cover Approach for Tokenization](a_partition_cover_approach_to_tokenization.md)
- [CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)
- [Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making](online_mixture_of_experts_no-regret_learning_for_optimal_collective_decision-mak.md)

<!-- RELATED:END -->
