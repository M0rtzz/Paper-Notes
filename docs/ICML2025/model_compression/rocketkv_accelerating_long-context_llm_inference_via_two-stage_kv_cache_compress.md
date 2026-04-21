---
title: >-
  [论文解读] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression
description: >-
  [ICML 2025][模型压缩][KV缓存压缩] 提出 RocketKV，一种无需训练的两阶段 KV 缓存压缩方法：第一阶段用 SnapKV 做粗粒度永久驱逐，第二阶段用混合稀疏注意力（HSA）做细粒度动态 top-k 选择，在 Mistral-7B 等模型上实现高达 400× 压缩比、3.7× 端到端加速和 32.6% 峰值内存节省，精度损失可忽略。
tags:
  - ICML 2025
  - 模型压缩
  - KV缓存压缩
  - 长上下文推理
  - 稀疏注意力
  - 推理加速
  - SnapKV
---

# RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression

**会议**: ICML 2025  
**arXiv**: [2502.14051](https://arxiv.org/abs/2502.14051)  
**代码**: [https://github.com/NVlabs/RocketKV](https://github.com/NVlabs/RocketKV)  
**领域**: 模型压缩  
**关键词**: KV缓存压缩, 长上下文推理, 稀疏注意力, 推理加速, SnapKV

## 一句话总结

提出 RocketKV，一种无需训练的两阶段 KV 缓存压缩方法：第一阶段用 SnapKV 做粗粒度永久驱逐，第二阶段用混合稀疏注意力（HSA）做细粒度动态 top-k 选择，在 Mistral-7B 等模型上实现高达 400× 压缩比、3.7× 端到端加速和 32.6% 峰值内存节省，精度损失可忽略。

## 研究背景与动机

### 1. KV 缓存的瓶颈

Transformer 推理中，KV 缓存存储历史 key-value 对以避免重复计算，但大小与序列长度和批大小成正比。例如 Llama3.1-70B 在 batch=32、context=32K 时需约 320GB 的 KV 缓存，远超当前硬件容量。

### 2. 现有方案的两难

现有 KV 缓存压缩方案分两类，各有致命缺陷：
- **永久驱逐**（H2O、SnapKV）：保留重要 token，节省内存和带宽，但被丢弃的 token 可能后续需要
- **动态选择**（Quest、SparQ）：保留所有 token 但每步动态选 top-k，避免信息丢失但节省内存有限且需额外索引

### 3. 关键观察与核心 Idea

作者分析 Mistral-7B 在 qasper 基准上的注意力模式发现：虽然序列长度可达 25000，但 oracle top-k（k=256）跨所有解码步选择的唯一 token 索引仅约 1200 个。这意味着：
1. 永久驱逐在 token 预算 ~1200 时应能接近 oracle 精度
2. 在驱逐后的更小集合上做动态选择会大幅降低 top-k 预测难度

**核心 Idea**：先粗粒度永久驱逐（较大预算）移除低重要性 token，再在残余集合上做细粒度动态选择——融合两类方案的优势。

## 方法详解

### 整体框架

RocketKV 采用两阶段级联架构：
- 输入：完整 KV 缓存（长序列）
- 第一阶段：SnapKV 粗粒度永久驱逐 → 保留重要 token 子集
- 第二阶段：HSA 细粒度动态 top-k 选择 → 极低带宽的稀疏注意力
- 输出：在极高压缩比下保持近似无损精度的注意力输出

框架高度灵活，各阶段可插拔不同方法（第一阶段可用 Ada-KV，第二阶段可用 SparQ/Loki）。

### 关键设计

#### 1. 第一阶段：SnapKV 粗粒度驱逐

- **功能**：利用输入上下文末尾的观察窗口计算聚合注意力分数，永久移除低分 token
- **GQA 适配**：原始 SnapKV 逐注意力头选择 token，造成同一 GQA 组内冗余存储。改进为按注意力组选择，共享 token 集合
- **池化核调整**：原 SnapKV 用小核（7）保留信息完整性，但本文第一阶段仅做粗粒度驱逐，实验发现最优核大小为 63，显著简化计算

#### 2. 第二阶段：混合稀疏注意力（HSA）

- **功能**：在驱逐后的残余 token 上进行二维约化（序列维 + 头维）的 top-k 估计
- **核心算法（三步）**：
    - **Step 1**：将 key 张量沿序列维分组为连续页面，存储每页逐元素最大值 $K_{\max}$ 和最小值 $K_{\min}$
    - **Step 2**：对查询 $q$ 找头维的 $k_1$ 个最大绝对值位置，根据 $q$ 对应位置的符号从 $K_{\max}$ 或 $K_{\min}$ 取值，计算 $\max(q \times K_{\max}, q \times K_{\min})$ 近似每页最高注意力分数，选 $k_2$ 个最高分页面
    - **Step 3**：从 $k_2$ 个选中页面的原始 KV 对上执行稀疏注意力
- **设计动机**：一维约化（Quest 沿序列维、SparQ 沿头维）压缩比有限，超过阈值精度急剧下降。二维约化同时利用两个维度的稀疏性，更准确地近似 top-k token
- **GQA 兼容**：所有选择在注意力组级别进行，通过对组维求和保证组内所有头做同一选择

#### 3. 自适应压缩分解

- **功能**：给定目标压缩比 $c = S/t$，智能分配两阶段的 token 预算
- **核心思路**：将 token 预算定义为包含 top-k 估计（Step 2）和稀疏注意力（Step 3）的总内存流量，均匀分配两步。这比仅计算稀疏注意力的旧定义更准确反映实际开销
- **GQA 适配**：token 预算定义在整个注意力组而非单个头上

### RocketKV-MT：多轮对话变体

- **问题**：永久驱逐在多轮对话中的致命问题 —— 早期轮驱逐的 token 可能在后续轮变得关键
- **方案**：第一阶段不永久丢弃，保留全部 KV token 但仍只动态选择第一阶段过滤出的子集做解码，跨轮保留完整 KV 历史
- **效果**：每轮解码速度与 RocketKV 相同，但无内存存储节省；精度接近 oracle top-k

## 实验关键数据

### 主实验结果

| 模型 | 方法 | 压缩比 | 端到端加速 | 峰值内存节省 | 精度影响 |
|------|------|--------|-----------|-------------|---------|
| Mistral-7B | Full KV | 1× | 1× | 0% | 基准 |
| Mistral-7B | SnapKV 仅驱逐 | ~16× | 适中 | 有 | 明显下降 |
| Mistral-7B | Quest 仅动态 | ~32× | 适中 | 0% | 明显下降 |
| Mistral-7B | **RocketKV** | **400×** | **3.7×** | **32.6%** | **可忽略** |
| 多模型 | RocketKV-MT | 高 | 近似 RocketKV | 无 | 接近 oracle top-k |

论文在 LongBench 多个子任务（qasper、HotpotQA 等）上的 Mistral-7B 实验表明，当 token 预算低于 1024 时其他方法精度剧降，而 oracle top-k 即使 k=256 仍能保持。RocketKV 通过两阶段成功逼近 oracle 水平。

### 消融分析

| 配置 | 表现趋势 | 说明 |
|------|---------|------|
| 仅第一阶段（SnapKV） | 中等精度，中等压缩 | 无法在极低预算下保持质量 |
| 仅第二阶段（HSA） | 高精度但压缩有限 | 一维约化的天花板 |
| 两阶段联合（RocketKV） | 精度最高，压缩比最大 | 两阶段互补的验证 |
| HSA 替换为 Quest/SparQ | 精度下降 | 二维约化优于一维 |
| SnapKV 核大小 7 vs 63 | 核 63 更优 | 粗粒度场景下大核更合适 |

注：缓存在方法章节末尾截断，完整的数值表格未能从缓存获取。上述实验趋势来自论文摘要和方法动机中提供的对比分析。

### 关键发现

- 两阶段联合方案在任意给定的总压缩比下都优于单阶段方案，因为永久驱逐缩小了搜索空间，使动态选择更准确
- HSA 的二维约化相比一维方法（Quest/SparQ）在相同压缩比下精度显著提升
- 多轮场景中 RocketKV-MT 完全规避永久驱逐的精度退化，与 oracle top-k 精度接近

## 亮点与洞察

- **观察驱动的设计**：对注意力模式的 CDF 分析（unique indices 远少于 sequence length）直接催生了两阶段方案，"让数据说话"的方法论值得学习
- **训练无关、即插即用**：不需要任何额外训练或 draft model，部署门槛极低
- **GQA 原生支持**：在注意力组级别操作的设计自然适配 GQA/MQA，无需特殊改造
- **自适应分解**：将 top-k 估计的流量也计入 token 预算，使压缩比定义更诚实、性能估计更准确
- **模块化框架**：各阶段可插拔不同方法，便于社区后续改进

## 局限与展望

- 缓存在 Section 3.6 之后截断，完整的实验部分（不同上下文长度 / 模型规模的 benchmark、运行时间成本、内存占用详细分析）未能获取
- RocketKV-MT 为保精度保留全部 KV 历史，在内存严格受限场景价值有限——是否可以做"部分保留"的中间方案？
- 极低 token 预算（k < 64）下精度仍有可见下降，对需要全局均匀注意的特殊推理任务效果待验证
- SnapKV 第一阶段依赖末尾观察窗口，对"信息均匀分布在全文"的场景可能驱逐过多重要 token

## 相关工作与启发

- **vs SnapKV**：本文第一阶段的基础，改进了 GQA 支持和池化参数
- **vs Quest / SparQ**：一维稀疏选择方法，本文 HSA 整合其思路并推广到二维
- **vs H2O**：经典的保留 Heavy Hitter + 最近 token 策略，但在低预算精度有限
- **vs Ada-KV**：自适应预算分配方案，可作为第一阶段替代
- **vs MInference**：专注 prefill 阶段加速，与 RocketKV 的 decode 阶段加速互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 两阶段组合 + 二维约化 HSA 设计新颖，但各组件有先驱工作
- 实验充分度: ⭐⭐⭐ 缓存中看到的摘要数字很强，但完整表格未获取
- 写作质量: ⭐⭐⭐⭐⭐ 观察清晰、动机逻辑链完整、方法表述系统
- 价值: ⭐⭐⭐⭐⭐ 训练无关即插即用，NVIDIA 开源，工程价值极高

<!-- RELATED:START -->

## 相关论文

- [ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](../../NeurIPS2025/model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [Inference-Time Hyper-Scaling with KV Cache Compression](../../NeurIPS2025/model_compression/inference-time_hyper-scaling_with_kv_cache_compression.md)
- [Core Context Aware Transformers for Long Context Language Modeling](core_context_aware_transformers_for_long_context_language_modeling.md)
- [KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](../../NeurIPS2025/model_compression/kvzip_query-agnostic_kv_cache_compression_with_context_reconstruction.md)
- [KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](../../NeurIPS2025/model_compression/keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)

<!-- RELATED:END -->
