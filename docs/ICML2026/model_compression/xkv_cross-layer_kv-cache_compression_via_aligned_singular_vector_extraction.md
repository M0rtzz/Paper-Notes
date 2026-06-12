---
title: >-
  [论文解读] xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction
description: >-
  [ICML 2026][模型压缩][KV-cache 压缩] xKV 发现 LLM 不同层的 KV-cache 虽然逐 token 余弦相似度不高，但主奇异向量高度对齐，因此用跨层共享低秩基同时压缩多层 KV-cache，并结合选择性重构在长上下文推理中取得最高 8 倍压缩和 4.23 倍端到端吞吐提升。
tags:
  - "ICML 2026"
  - "模型压缩"
  - "KV-cache 压缩"
  - "长上下文推理"
  - "跨层低秩分解"
  - "选择性重构"
  - "CKA"
---

# xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction

**会议**: ICML 2026  
**arXiv**: [2503.18893](https://arxiv.org/abs/2503.18893)  
**代码**: https://github.com/abdelfattah-lab/xKV  
**领域**: 模型压缩 / LLM 推理加速  
**关键词**: KV-cache 压缩, 长上下文推理, 跨层低秩分解, 选择性重构, CKA  

## 一句话总结
xKV 发现 LLM 不同层的 KV-cache 虽然逐 token 余弦相似度不高，但主奇异向量高度对齐，因此用跨层共享低秩基同时压缩多层 KV-cache，并结合选择性重构在长上下文推理中取得最高 8 倍压缩和 4.23 倍端到端吞吐提升。

## 研究背景与动机
**领域现状**：长上下文 LLM 的主要瓶颈已经从参数存储转向推理时不断膨胀的 KV-cache。现有压缩路线大致包括低比特量化、token eviction、单层低秩分解、动态 token 选择以及 CPU offloading；这些方法可以降低显存占用，但大多把每一层的 cache 当成彼此独立的对象处理。

**现有痛点**：单层方法只能利用层内冗余，压缩率进一步提高时容易丢失长程检索、变量追踪和多轮对话中需要保留的细粒度信息。跨层共享 KV 的方法也存在两类问题：CLA、YOCO 这类架构改造需要重新预训练，MiniCache 这类后处理方法又依赖相邻层 token 表示的余弦相似度，而论文实测发现这种逐 token 相似度并不稳定，导致即使 1.3 倍压缩也可能严重掉点。

**核心矛盾**：KV-cache 的冗余不是简单的“同一个 token 在相邻层很像”，而更像是“不同层的 token 空间由相近的主方向张成”。如果只看 token 级余弦相似度，就会低估跨层共享的空间；如果要显式利用这个共享空间，又必须避免为每一步解码重构完整长序列带来的计算开销。

**本文目标**：作者希望在不改模型结构、不微调模型的条件下压缩已有长上下文 LLM 的 KV-cache，同时维持 RULER、LongBench 和多轮 NIAH 等任务的准确率，并把显存节省真正转化为吞吐提升，而不是被重构计算或 PCIe 传输抵消。

**切入角度**：论文用 CKA 检查不同层 KV-cache 的整体几何结构，而不是比较每个 token 的向量方向。结果显示，Llama、Qwen 乃至混合注意力架构上，不同层 KV-cache 的主奇异向量有明显对齐现象；把多个相邻层水平拼接后，保留 95% 谱能量所需的相对秩还会随窗口变大而下降。

**核心 idea**：用跨层 SVD 提取一组共享 token basis，让同一层组内的多层 KV-cache 共享主空间，再用每层自己的小重构矩阵恢复需要参与注意力的 token。

## 方法详解

### 整体框架
xKV 面向 decoder-only LLM 的长上下文推理。输入是一段长 prompt 在预填充阶段产生的各层 key/value cache；输出不是新的模型权重，而是一个压缩后的 cache 表示。它把模型层按连续窗口分组，对每个窗口里的多层 cache 做联合低秩分解，存储共享 basis 和层特定重构矩阵。解码时，模型可以选择密集重构整层 cache，也可以只重构当前 query 最可能关注的少量 token，从而在显存和计算之间取得更好的平衡。

更具体地说，设某个层的 key 或 value cache 为 $X_\ell \in \mathbb{R}^{L \times d}$，其中 $L$ 是上下文长度，$d$ 是 KV hidden size。xKV 把窗口 $\mathcal{W}_k$ 内的 $W$ 层 cache 横向拼接为 $X_k^{cat}=[X_{kW},\ldots,X_{kW+W-1}]$，然后做低秩近似 $X_k^{cat}\approx A_k[B_{kW},\ldots,B_{kW+W-1}]$。其中 $A_k\in\mathbb{R}^{L\times r}$ 是这个层组共享的 token basis，$B_\ell\in\mathbb{R}^{r\times d}$ 是每层自己的重构矩阵。整条流水线可概括为：预填充阶段靠跨层对齐先分组、再做跨层低秩分解得到共享基；解码阶段在密集重构与选择性重构之间二选一。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["长 prompt 预填充<br/>产出各层 K/V cache"] --> B["跨层对齐分析（CKA）<br/>主奇异向量跨层对齐 → 相邻 W 层分组（默认 W=4）"]
    B --> C["跨层低秩分解（CLF）<br/>窗口内多层横向拼接 → 跨层 SVD<br/>得共享基 A_k 与每层重构矩阵 B_ℓ"]
    C --> D{解码时重构}
    D -->|密集重构：全 L token<br/>O(L) 计算瓶颈| E["X̂_ℓ = A_k · B_ℓ"]
    D -->|选择性重构（SR）| F["近似注意力选 query 相关 token 集 S<br/>仅重构 S 对应行（|S| ≪ L）"]
    E --> G["参与注意力 → 长上下文推理加速"]
    F --> G
```

### 关键设计

**1. 用 CKA 发现跨层共享的主奇异向量：把“跨层相似”从 token 级改成子空间级**

xKV 的全部前提是先回答一个问题——不同层的 KV-cache 到底像不像。MiniCache 这类方法直接比相邻层逐 token 的余弦相似度，得出“挺像、可以合并”的结论，但实测在 Llama 上 1.3 倍压缩就掉到 45 分。xKV 改用 Centered Kernel Alignment（CKA）：它比较的不是单个 token 向量，而是两层 token 嵌入的中心化 Gram 矩阵，刻画整体几何结构是否一致。结果发现，逐 token 余弦相似度确实不高，但许多层对的 CKA 很高，而高 CKA 在数学上等价于两层主左奇异向量高度对齐（附录给了证明）。换句话说，真正可压缩的冗余不在“同一个 token 在相邻层很像”，而在“不同层的 token 空间由相近的主方向张成”。只看余弦相似度会严重低估跨层压缩空间，这既是 MiniCache 脆弱的根因，也直接决定了 xKV 选择“共享主方向”而非“合并相似 token”——这正是框架图里第一步“分组依据”的来源。

**2. 跨层低秩分解（Cross-Layer Factorization，CLF）：让一组层共享一套 token basis**

既然主奇异向量跨层对齐，xKV 就把相邻 $W$ 层（默认 $W=4$）编为一个窗口，在 prefill 阶段对窗口内多层 cache 做联合低秩分解。做法是把各层 cache 横向拼接成 $X_k^{cat}$，再做跨层 SVD，得到一个层组共享的 token basis $A_k\in\mathbb{R}^{L\times r}$ 和每层各自的小重构矩阵 $B_\ell\in\mathbb{R}^{r\times d}$。这样原本 $W$ 层要存 $O(WLd)$ 的 cache，被压成 $O(Lr+Wrd)$ 的“一份共享基 + $W$ 份层特定系数”；key 和 value 走同一流程，默认 key/value rank 按 1:1.5 配置。对比单层 SVD——它给每层各存一份高度相似的主方向，压缩率一上去就快速掉点（Llama 上 8.4 倍只有 45.71）——跨层分解把这些重复 basis 合并成一份，同等压缩率下保留的有效信息显著更多。这也解释了窗口从 1 扩到 4 时密集 xKV 平均分从 45.71 跳到 88.50：收益主要来自“跨层共享”本身，而不是“低秩”。

**3. 选择性重构（Selective Reconstruction，SR）：把重构成本从全序列降到 query 相关的少数 token**

压缩省了显存，但解码时若每步都把整层 $L$ 个 token 的 cache 重构出来（$\hat{X}_\ell=A_k B_\ell$），重构 FLOPs 会随 $L$ 线性增长、落到关键路径上，反而拖垮速度——论文实测密集重构在 122k 上下文把 kernel 速度压到约 0.4 倍。Selective Reconstruction 借助 LLM 注意力天然稀疏的特性：每个解码步、每个 head、每层先用近似注意力（landmark-guided chunk selector）选出当前 query 最可能关注的 token 集 $\mathcal{S}_{t,\ell,g}$，只重构这些行 $\hat{X}_{\ell,g}[\mathcal{S}_{t,\ell,g},:]=A_k[\mathcal{S}_{t,\ell,g},:]B_{\ell,g}$。由于 $|\mathcal{S}|$ 固定且远小于 $L$，重构开销不再随上下文长度增长。ShadowKV 最早提出选择性重构，但它底层是单层 SVD、保真不足，压缩 value 后掉点明显，只能把 value offload 到 CPU、走 PCIe 传输；xKV-SR 因为跨层分解保真更高，可以把 key/value 都留在 GPU HBM，避开传输瓶颈，最终在 122k 上下文拿到 4.23 倍端到端吞吐提升。

### 损失函数 / 训练策略
xKV 不引入训练损失，也不需要微调。它是 post-training、plug-and-play 的推理时压缩方法：prefill 后对真实 prompt 产生的 cache 做在线低秩分解，解码时根据 query 选择性重构。作者还实现了自定义 randomized SVD kernel，用 16-bit GEMM 和 shifted Cholesky QR 降低在线分解开销；在 Llama-3.1-8B 的 64k 到 256k 上下文里，窗口大小为 4 时 SVD 时间约为 prefill 的 5.72% 到 1.24%。

## 实验关键数据

### 主实验
RULER 64K 是主实验核心。下表保留平均分和压缩率，能直接看出 xKV 与单层、token eviction、量化和跨层合并 baseline 的差异。

| 模型 | 方法 | 类型 | KV 压缩率 | RULER Avg. | 说明 |
|------|------|------|-----------|------------|------|
| Llama-3.1-8B-Instruct | Full Attention | 无压缩 | 1.00x | 91.89 | 原始 KV-cache |
| Llama-3.1-8B-Instruct | KIVI-2 | 层内量化 | 7.10x | 86.87 | 2-bit KV 量化 |
| Llama-3.1-8B-Instruct | SnapKV | 层内 token eviction | 8.00x | 89.68 | 首轮注意力驱动裁剪 |
| Llama-3.1-8B-Instruct | Single SVD | 层内低秩 | 8.40x | 45.71 | 每层独立 SVD，掉点严重 |
| Llama-3.1-8B-Instruct | MiniCache | 跨层 token 合并 | 1.30x | 45.04 | 余弦相似假设不稳 |
| Llama-3.1-8B-Instruct | xKV | 跨层低秩 | 8.03x | 88.50 | 接近强 token eviction baseline |
| Qwen2.5-14B-Instruct-1M | Full Attention | 无压缩 | 1.00x | 93.36 | 原始 KV-cache |
| Qwen2.5-14B-Instruct-1M | SnapKV | 层内 token eviction | 6.00x | 91.66 | Qwen 的 KV heads 更少，压缩更难 |
| Qwen2.5-14B-Instruct-1M | Single SVD | 层内低秩 | 6.35x | 71.79 | 单层低秩仍损失明显 |
| Qwen2.5-14B-Instruct-1M | MiniCache | 跨层 token 合并 | 1.30x | 13.78 | 在该模型上几乎不可用 |
| Qwen2.5-14B-Instruct-1M | xKV-4 | 跨层低秩 | 6.21x | 90.19 | 与 full attention 差距约 3.17 分 |

### 消融实验
选择性重构和窗口大小是最关键的分析。第一组结果说明 xKV-SR 在保留准确率的同时把压缩 cache 放回 GPU；第二组说明“跨层”本身确实贡献很大。

| 配置 | 压缩/显存效果 | RULER Avg. | 说明 |
|------|---------------|------------|------|
| Full Attention | 1.00x | 91.89 | Llama-3.1-8B-Instruct 原始 baseline |
| Quest | 1.00x KV，约 8.00x GPU memory reduction | 84.87 | 只做动态 token loading，不真正压缩 cache |
| ShadowKV | 1.64x KV，约 9.08x GPU memory reduction | 87.17 | 单层 SVD + value offloading |
| xK-SR | 1.63x KV，约 8.90x GPU memory reduction | 89.70 | 用跨层 key 压缩替换 ShadowKV 的单层 key 压缩 |
| ShadowKV‡ | 5.51x KV | 70.94 | 进一步压缩 value 后准确率大幅下降 |
| xKV-SR | 5.35x KV | 89.69 | key/value 都跨层压缩并保持在 GPU 上 |

| 窗口大小 | xKV Avg. | xK-SR Avg. | xKV-SR Avg. | 结论 |
|----------|----------|-------------|--------------|------|
| 1 | 45.71 | 87.17 | 72.27 | 退化成单层压缩时保真不足 |
| 2 | 75.15 | 88.43 | 86.06 | 跨两层已明显提升 |
| 4 | 88.50 | 89.70 | 89.69 | 主实验默认，精度和开销较平衡 |
| 8 | 88.91 | 89.74 | 89.72 | 继续扩大窗口收益接近饱和 |

### 关键发现
- xKV 的核心优势不是单纯“低秩”，而是把低秩从每层独立改成跨层共享；窗口大小从 1 到 4 时，密集 xKV 平均分从 45.71 提升到 88.50，直接证明跨层 basis 才是主要收益来源。
- MiniCache 在 Llama 上 1.3 倍压缩只有 45.04，在 Qwen 上只有 13.78，说明 token 级余弦相似度不足以支撑跨层 cache 合并；这也反向支持作者用 CKA 而不是 cosine 做动机分析。
- xKV-SR 的系统价值很清楚：它以 5.35 倍 KV 压缩维持 89.69 平均分，并在 122k 上下文取得 4.23 倍端到端吞吐提升；如果只做密集重构，显存瓶颈会缓解，但重构 FLOPs 会压低速度。
- 多轮 NIAH 中 SnapKV/PyramidKV 会在后续轮次明显掉点，因为它们按首轮 query 的注意力裁剪 token；xKV 保留压缩后的全局信息，所以跨轮更稳定。

## 亮点与洞察
- 论文最好的洞察是把“跨层相似”从 token 表示相似改成子空间相似。这个视角解释了为什么已有跨层合并方法看起来合理却效果脆弱，也让 SVD 这种老工具在 KV-cache 场景里有了新用法。
- xKV 把算法压缩和系统吞吐放在同一个闭环里讨论。很多 cache 压缩方法只报显存或准确率，但这里明确指出密集重构会变成计算瓶颈，必须靠选择性重构和 GPU 常驻才能兑现速度收益。
- 该方法是训练无关的，所以对已有长上下文模型很友好。它不要求重新预训练 CLA/YOCO 式模型，也不假设用户能拿到大规模训练资源。
- 窗口大小消融很有解释力。窗口从 1 到 4 的跃升说明跨层冗余真实存在，而 8 的收益饱和提醒实际部署不必盲目扩大层组，避免 prefill 缓冲和 SVD 开销增加。

## 局限与展望
- 论文主要研究 long-prefill 场景，即初始长上下文被压缩，后续生成 token 不压缩。对于长篇生成或 test-time scaling，生成过程中不断增长的新 KV-cache 仍可能成为瓶颈。
- rank、窗口大小和 key/value 压缩比例基本是固定策略。附录显示不同任务对 key/value 压缩的容忍度不同，未来可以做任务感知或上下文感知的动态 rank 分配。
- 方法依赖 prefill 后的在线 SVD，虽然长上下文中相对开销较小，但在短上下文、高 QPS 或小 GPU 场景中，分解开销和实现复杂度仍需要额外评估。
- 选择性重构需要先选 token，实际效果会受近似注意力质量影响。若任务需要非常分散的证据或 query 模式频繁变化，固定 token budget 可能不够稳。

## 相关工作与启发
- **vs KIVI / KV 量化**: 量化降低每个元素的 bit 数，xKV 降低需要存储的子空间维度；两者正交，论文附录还显示 xKV 可以叠加 4-bit/3-bit 量化进一步压缩。
- **vs SnapKV / PyramidKV**: token eviction 直接丢弃部分历史 token，适合注意力集中且单轮查询稳定的场景；xKV 保留全局 cache 的低秩近似，因此在多轮检索和需要后续重新关注旧信息时更稳。
- **vs MiniCache**: MiniCache 用相邻层 token 余弦相似度做合并，xKV 用 CKA 发现共享主奇异向量。前者压缩假设过强，后者对层间表示旋转和 token 级差异更鲁棒。
- **vs ShadowKV**: ShadowKV 的选择性重构思想很有价值，但底层是单层 SVD，压缩 value 后掉点明显；xKV 替换成跨层 factorization 后，可以把 key/value 都压缩并尽量留在 GPU。
- **启发**: 很多推理加速问题可能不该只比较激活的逐元素或逐 token 相似度，而应检查表征子空间、Gram 结构和谱能量分布；这对 MoE activation cache、视觉 token cache 和扩散模型 feature cache 都有迁移价值。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 用 CKA 引出跨层共享奇异向量很有辨识度，SVD 本身传统但组合方式新。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 RULER、LongBench、多轮 NIAH、窗口大小、key/value 压缩、量化叠加和吞吐测量。
- 写作质量: ⭐⭐⭐⭐☆ 动机链条清楚，系统实验扎实；少量公式和表格排版在文本缓存里略显拥挤。
- 价值: ⭐⭐⭐⭐⭐ 对长上下文 LLM 部署很实用，尤其适合已有模型的无训练压缩和 GPU batch 扩展。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] The Pitfalls of KV Cache Compression](../../ACL2026/model_compression/the_pitfalls_of_kv_cache_compression.md)
- [\[ICML 2026\] Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in.md)
- [\[ICML 2026\] FlattenGPT: Depth Compression for Transformer with Layer Flattening](flattengpt_depth_compression_for_transformer_with_layer_flattening.md)
- [\[AAAI 2026\] KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](../../AAAI2026/model_compression/kvmix_gradient-based_layer_importance-aware_mixed-precision_.md)
- [\[ICML 2026\] EpiCache: Episodic KV Cache Management for Long-Term Conversation on Resource-Constrained Environments](epicache_episodic_kv_cache_management_for_long-term_conversation_on_resource-con.md)

</div>

<!-- RELATED:END -->
