---
title: >-
  [论文解读] MoH: Multi-Head Attention as Mixture-of-Head Attention
description: >-
  [ICML 2025][LLM效率][Mixture-of-Experts] 本文将多头注意力（MHA）重新表述为求和形式，借鉴 MoE 思想提出 Mixture-of-Head Attention（MoH），通过路由器为每个 token 动态选择最相关的注意力头子集，仅激活 50%~90% 的头即可匹配甚至超越标准 MHA 性能，并证明预训练模型（如 LLaMA3-8B）可通过 continue-tuning 转换为 MoH 模型。
tags:
  - ICML 2025
  - LLM效率
  - Mixture-of-Experts
  - 注意力头路由
  - 稀疏激活
  - 多头注意力
  - 推理加速
---

# MoH: Multi-Head Attention as Mixture-of-Head Attention

**会议**: ICML 2025  
**arXiv**: [2410.11842](https://arxiv.org/abs/2410.11842)  
**代码**: https://github.com/SkyworkAI/MoH (有)  
**领域**: LLM效率 / 注意力机制  
**关键词**: Mixture-of-Experts, 注意力头路由, 稀疏激活, 多头注意力, 推理加速

## 一句话总结

本文将多头注意力（MHA）重新表述为求和形式，借鉴 MoE 思想提出 Mixture-of-Head Attention（MoH），通过路由器为每个 token 动态选择最相关的注意力头子集，仅激活 50%~90% 的头即可匹配甚至超越标准 MHA 性能，并证明预训练模型（如 LLaMA3-8B）可通过 continue-tuning 转换为 MoH 模型。

## 研究背景与动机

**领域现状**：多头注意力（MHA）是 Transformer 的核心组件，广泛用于 NLP 和 CV。标准做法是所有注意力头并行计算后拼接/求和输出，每个 token 都经过全部头的处理。

**现有痛点**：大量研究表明多头注意力中存在显著的冗余。Voita et al. 证明许多注意力头可以被剪枝而不影响精度；Michel et al. 发现即使大幅剪枝也不会显著降低性能。这意味着标准 MHA 在推理时做了大量不必要的计算。

**核心矛盾**：MHA 对所有 token 一视同仁地激活全部注意力头，但不同 token 实际上只需要部分头的处理。这种"一刀切"设计既浪费计算资源，也限制了头的专业化程度——因为所有头在相同数据上训练，容易学到相似特征。

**本文目标** (a) 如何让每个 token 动态选择需要的注意力头？(b) 如何在减少激活头数的同时保持甚至提升性能？(c) 如何将已有预训练模型高效转换为稀疏头模型？

**切入角度**：作者注意到 MHA 可以从标准的拼接形式改写为**求和形式**，即输出等于各头输出的简单相加。既然是相加，就可以像 MoE 一样为各项加权、稀疏激活，自然引出"头即专家"的类比。

**核心 idea**：把注意力头视为 MoE 中的专家，用 router 为每个 token 选择 Top-K 头，并通过 shared head + 两阶段路由增强路由稳定性。

## 方法详解

### 整体框架

MoH 在标准 Transformer 架构中替换 MHA 层。输入仍是 token 序列 $\mathbf{X} \in \mathbb{R}^{T \times d_{in}}$，输出维度不变。每个 MoH 层包含：(1) $h$ 个注意力头（和标准 MHA 相同），(2) 一个路由器（router），(3) shared heads 子集。对每个 token，router 在非 shared 头中选择 Top-K 个激活，shared 头始终激活。最终输出是所有激活头输出的**加权求和**（而非标准 MHA 的等权求和），权重由 router 给出。

### 关键设计

1. **求和形式的 MHA 重写**

    - 功能：将标准 MHA 从拼接形式转为等价的求和形式
    - 核心思路：将输出投影矩阵 $\mathbf{W}_O \in \mathbb{R}^{d_v \times d_{out}}$ 按行分解为 $h$ 个子矩阵 $\mathbf{W}_O^i$，则 $\text{MultiHead}(\mathbf{X}, \mathbf{X}') = \sum_{i=1}^{h} \mathbf{H}^i \mathbf{W}_O^i$
    - 设计动机：求和形式揭示了各头的独立性——既然输出是各头贡献的简单加法，就可以自然地对部分项置零（稀疏化）或重新加权，为引入 MoE 路由机制提供了理论基础

2. **Heads as Experts（头即专家）**

    - 功能：将 $h$ 个注意力头视为 MoE 中的 $h$ 个专家，用 router 动态激活 Top-K 个
    - 核心思路：MoH 输出为 $\text{MoH}(\mathbf{X}, \mathbf{X}') = \sum_{i=1}^{h} g_i \mathbf{H}^i \mathbf{W}_O^i$，其中 $g_i$ 是路由得分，非激活头的 $g_i = 0$。与标准 MoE 不同的是，MoH 不增加头的数量，总参数量与 MHA 基本相同
    - 设计动机：不同于 MoE 追求参数扩展，MoH 的核心目标是减少冗余头的激活以提升推理效率。加权求和取代等权求和增加了灵活性，释放了额外的性能潜力

3. **Shared Heads（共享头）**

    - 功能：指定 $h_s$ 个头为共享头，对所有 token 始终激活
    - 核心思路：共享头捕获跨上下文的通用知识（如语法规则），其路由得分由独立的投影矩阵 $\mathbf{W}_s \in \mathbb{R}^{h_s \times d_{in}}$ 通过 Softmax 计算。实验表明共享头比例在 13.9%~74.0% 范围内模型性能稳定
    - 设计动机：灵感来自 DeepSeekMoE。共享头集中处理共性知识，使得路由头可以更专注于领域/任务特定信息，减少路由头之间的冗余。共享头也可视为 Soft MoE 的一种形式，作者建议共享头比例 > 40%

4. **Two-Stage Routing（两阶段路由）**

    - 功能：分两级计算路由得分——先在各类型内部 Softmax 归一化，再用可学习系数 $\alpha_1, \alpha_2$ 平衡 shared 与 routed 头的贡献
    - 核心思路：路由得分定义为分段函数。对 shared 头 $i$：$g_i = \alpha_1 \cdot \text{Softmax}(\mathbf{W}_s \mathbf{x}_t)_i$；对激活的 routed 头：$g_i = \alpha_2 \cdot \text{Softmax}(\mathbf{W}_r \mathbf{x}_t)_{i-h_s}$。平衡系数 $[\alpha_1, \alpha_2] = \text{Softmax}(\mathbf{W}_h \mathbf{x}_t)$，其中 $\mathbf{W}_h \in \mathbb{R}^{2 \times d_{in}}$ 可学习
    - 设计动机：单级路由无法动态调节 shared 头和 routed 头的整体重要性。两阶段设计让模型根据输入自适应分配两类头的贡献权重

5. **Continue-Tuning 策略（预训练模型转换）**

    - 功能：将已有预训练 MHA 模型（如 LLaMA3-8B）转换为 MoH 模型
    - 核心思路：解决三个挑战——(a) 共享头选择：直接取每层前 16 个头作为 shared heads；(b) 无参数路由器初始化：用每个头的 query 的 $\ell_2$ 范数作为路由得分，无需随机初始化；(c) 路由得分量化：用 straight-through estimator 处理量化 $g_i^q = \mathbb{1}(\text{Token } \mathbf{x} \text{ selects Head } i)$，避免加权和对输出分布的剧烈改变
    - 设计动机：从头训练成本巨大。通过巧妙的初始化和两阶段训练（先 300B tokens 适配数据分布，再 100B tokens 转化为 MoH），仅用约 3% 的原始预训练数据量即可完成转换

### 损失函数 / 训练策略

- **Load Balance Loss**：防止路由坍缩（大部分 token 路由到少数头），公式为 $\mathcal{L}_b = \sum_{i=h_s+1}^{h} P_i f_i$，其中 $P_i$ 是头 $i$ 的平均路由概率，$f_i$ 是头 $i$ 被选中的 token 比例
- **总训练目标**：$\mathcal{L} = \mathcal{L}_{task} + \beta \mathcal{L}_b$，其中 $\beta = 0.01$（所有任务统一）
- **非均匀头激活预算**：浅层激活更少的头，深层激活更多的头（借鉴 TransNeXt 的设计）
- **Continue-Tuning 两阶段**：第一阶段用 300B tokens 适配数据分布，第二阶段用 100B tokens 训练 MoH 路由

## 实验关键数据

### 主实验 1：ViT 图像分类（ImageNet-1K）

| 方法 | 参数量(M) | 激活头比例 | Top-1 Acc(%) |
|------|-----------|-----------|--------------|
| TransNeXt-S | 50 | 100% | 84.7 |
| **MoH-ViT-S** | **50** | **80%** | **84.7** |
| MoH-ViT-S | 50 | 75% | 84.6 |
| TransNeXt-B | 90 | 100% | 84.8 |
| **MoH-ViT-B** | **90** | **75%** | **84.9** |
| MoH-ViT-B | 90 | 50% | 84.7 |

### 主实验 2：Continue-Tuning LLaMA3-8B（14 个 Benchmark 平均）

| 方法 | 激活头比例 | MMLU | CEVAL | CMMLU | GSM8K | TruthfulQA | HellaSwag | ARC-C | 14项平均 |
|------|-----------|------|-------|-------|-------|------------|-----------|-------|---------|
| LLaMA3-8B | 100% | 65.2 | 52.3 | 50.7 | 49.5 | 35.4 | 81.9 | 59.0 | 61.6 |
| **MoH-LLaMA3-8B** | **75%** | **65.8** | **61.5** | **64.4** | **56.9** | **44.0** | 80.1 | **60.1** | **64.0** |

MoH-LLaMA3-8B 仅用 75% 的注意力头，在 14 个 benchmark 上平均提升了 **2.4%**，尤其在中文任务（CEVAL +9.2, CMMLU +13.7）和数学推理（GSM8K +7.4）上提升显著。

### 消融实验

| Shared Heads | Two-Stage Routing | ViT Acc(%) | DiT FID↓ |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | 75.6 | 71.97 |
| ✓ | ✗ | 78.3 | 69.54 |
| **✓** | **✓** | **78.6** | **69.42** |

### 关键发现

- **Shared Heads 贡献最大**：引入 shared heads 将 ViT 精度从 75.6% 提升至 78.3%（+2.7%），是性能提升的关键因素
- **小模型中更少的头可能更好**：MoH-LLM-S 激活 50% 的头反而优于 75%（45.4% vs 44.6%），作者认为在小模型+小数据场景下，更少的头起到了正则化效果
- **图像生成任务对头的需求更大**：DiT 模型在 75% 激活率下性能下降，因为像素级密集预测需要更多头捕获细粒度关系
- **推理加速实际可测量**：序列长度 512 时，MoH 50% 激活头推理时间 0.863ms vs MHA 1.376ms，加速约 37%
- **头负载可视化**显示不同类别/任务有不同的头分配模式，证明 MoH 实现了头的专业化

## 亮点与洞察

- **求和形式的洞察极为关键**：将 MHA 改写为求和形式看似简单，却为引入 MoE 路由机制打开了大门。这种对已有公式的重新解读是值得学习的研究思路——很多创新不需要发明新东西，只需换个角度看老问题
- **Continue-tuning 策略设计精巧**：无参数路由器（用 query 的 L2 范数）+ 量化路由得分 + straight-through estimator 的组合，巧妙解决了向预训练模型注入新模块的冷启动问题。这套策略可以迁移到任何需要在预训练模型中引入稀疏路由的场景
- **跨模态一致验证增强可信度**：在 ViT、DiT、LLM 三种截然不同的框架上验证同一方法，且都显示正向收益，说明注意力头冗余是一个普遍现象，MoH 的解决方案具有通用性
- **Shared heads 可视为 Soft MoE**：这个视角将 shared heads 与 Puigcerver et al. 的 Soft MoE 联系起来，为理解 shared heads 的作用提供了理论支撑

## 局限与展望

- **激活率仍较高**：当前最低也需 50% 的头才能保持性能，未来需探索更激进的稀疏化（<50%）
- **头大小固定**：所有注意力头有相同的 hidden size，未探索异构头（不同大小的头用于不同功能），这是一个自然的扩展方向
- **Continue-tuning 数据量仍不小**：需要 400B tokens（300B 适配 + 100B 转换），对算力要求仍然较高
- **仅验证了 decoder-only LLM**：未测试 encoder-only（BERT类）或 encoder-decoder（T5类）模型
- **多模态场景未探索**：视觉和文本 token 在注意力中有不同模式，MoH 的路由机制如何处理多模态输入是开放问题
- **推理加速依赖稀疏矩阵乘法**：实际部署需要算子层面的支持（稀疏 QKV），当前硬件对稀疏运算的优化仍不充分

## 相关工作与启发

- **vs MoA (Zhang et al., 2022)**：MoA 也将注意力头与 MoE 结合，但目标是扩展参数量（类似标准 MoE），且需共享 K/V 因此必须从头训练。MoH 不增加参数、支持 continue-tuning，适用性更广
- **vs SwitchHead (Csordás et al., 2024)**：同样采用 MoE 风格激活注意力头，但 MoH 额外引入了 shared heads 和两阶段路由，提供了更稳定的训练和更好的性能
- **vs DuoAttention (Xiao et al., 2024)**：DuoAttention 区分 retrieval heads 和 streaming heads 来优化 KV cache，关注长上下文推理。MoH 则在更基础的层面优化头的激活，两者思路可结合
- **vs 头剪枝方法 (Voita et al., 2019; Michel et al., 2019)**：传统头剪枝是静态的（对所有输入剪掉相同的头），MoH 是动态的（不同 token 激活不同头），更灵活但引入了路由开销

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 MHA 重写为求和形式后引入 MoE 路由是自然但有效的创新，shared heads + 两阶段路由 + continue-tuning 方案有较多工程贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 跨 ViT/DiT/LLM 三大框架验证，包含从头训练和 continue-tuning，消融完整，可视化分析深入
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，从求和形式的推导到 MoH 的提出一气呵成，公式排版规范
- 价值: ⭐⭐⭐⭐ MoH 作为 MHA 的即插即用替代方案具有广泛的应用前景，continue-tuning 方案进一步降低了采用门槛

<!-- RELATED:START -->

## 相关论文

- [Mixture of Lookup Experts](mixture_of_lookup_experts.md)
- [From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers](../../NeurIPS2025/llm_efficiency/from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in.md)
- [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](../../ACL2025/llm_efficiency/native_sparse_attention.md)
- [Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](efficient_length-generalizable_attention_via_causal_retrieval_for_long-context_l.md)
- [LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](../../ICLR2026/llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid_speculative_deco.md)

<!-- RELATED:END -->
