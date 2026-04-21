---
title: >-
  [论文解读] Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search
description: >-
  [NeurIPS 2025][模型压缩][注意力机制] NVIDIA 提出 PostNAS 流水线——从预训练全注意力模型出发，冻结 MLP 权重，通过四步搜索（全注意力层放置→线性注意力块选择→新注意力块 JetBlock 设计→硬件感知超参搜索）得到混合架构 Jet-Nemotron，2B 模型在 MMLU-Pro 上超越 Qwen3-1.7B 同时生成吞吐提升 47×。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 注意力机制
  - 神经架构搜索
  - efficient LLM
  - KV cache
---

# Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search

**会议**: NeurIPS 2025  
**arXiv**: [2508.15884](https://arxiv.org/abs/2508.15884)  
**代码**: https://github.com/NVlabs/Jet-Nemotron  
**领域**: model_compression  
**关键词**: hybrid attention, linear attention, neural architecture search, efficient LLM, KV cache

## 一句话总结
NVIDIA 提出 PostNAS 流水线——从预训练全注意力模型出发，冻结 MLP 权重，通过四步搜索（全注意力层放置→线性注意力块选择→新注意力块 JetBlock 设计→硬件感知超参搜索）得到混合架构 Jet-Nemotron，2B 模型在 MMLU-Pro 上超越 Qwen3-1.7B 同时生成吞吐提升 47×。

## 研究背景与动机

**领域现状**：LLM 推理效率是部署瓶颈。全注意力 $O(n^2)$ 复杂度在长上下文生成时 KV cache 膨胀严重。已有大量线性注意力 ($O(n)$) 设计（Mamba2、RWKV7、GLA 等），以及混合架构（保留少量全注意力层 + 线性注意力层）。

**现有痛点**：现有高效模型从头预训练成本高且精度明显落后全注意力 SOTA——尤其在 MMLU-Pro、数学推理和检索任务上差距很大。从头训练架构设计风险高、周期长。

**核心矛盾**：架构探索需要预训练验证，但预训练成本极高，导致学术界和中小团队难以参与 LLM 架构创新。

**本文切入角度**：不从头训练，而是从现有全注意力模型出发，冻结 MLP（已学到知识的主体）只探索注意力块设计。大幅降低成本的同时保持对 SOTA 的竞争力。

**核心 idea**：Post Neural Architecture Search (PostNAS)——一个从粗到细的四步架构搜索流水线，从预训练全注意力模型继承 MLP 知识，系统搜索最优注意力块配置。

## 方法详解

### 整体框架
PostNAS 从预训练全注意力模型（如 Qwen2.5-1.5B）出发，冻结所有 MLP 权重，通过四个步骤逐步优化注意力块设计：(1) 学习全注意力层最优放置位置→(2) 选择最优线性注意力块→(3) 设计新的 JetBlock→(4) 硬件感知超参搜索。最终产出 Jet-Nemotron 系列模型。

### 关键设计

1. **全注意力层放置学习**：

    - 功能：确定哪些层保留全注意力，哪些替换为线性注意力
    - 核心思路：构建 Once-for-All 超级网络——每层同时有全注意力路径和线性注意力路径，训练时随机采样子网络，用特征蒸馏损失训练。训练完成后用 beam search 在给定约束下（如只保留 2 层全注意力）搜索最优放置
    - 设计动机：不同任务对全注意力的需求不同（检索任务需第 15/20 层全注意力，MMLU 需要滑动窗口注意力），学习式放置显著优于均匀间隔放置（MMLU 提升多个点）

2. **线性注意力块选择**：

    - 功能：在 6 个 SOTA 线性注意力块中选最优
    - 核心思路：在 PostNAS 框架下对比 RetNet、Mamba2、GLA、Deltanet、Gated DeltaNet 的精度和效率。无需小模型代理实验，直接在目标规模验证
    - 结论：Gated DeltaNet 综合最优——得益于数据依赖门控 + Delta 规则（增量更新隐状态以节省有限状态内存）

3. **JetBlock 设计（新线性注意力块）**：

    - 功能：在 Gated DeltaNet 基础上引入动态卷积增强表达力
    - 核心思路：(a) 引入核生成器（Kernel Generator）——输入经线性降维（ratio=8）→ SiLU 激活 → 线性层输出动态卷积核权重；(b) 动态卷积仅作用于 Value tokens（对 Q/K 无效）；(c) 去除 Q/K 上的静态卷积（引入动态卷积后冗余）
    - 设计动机：先前线性注意力的卷积核是静态的，无法根据输入自适应调整特征提取模式；动态卷积能根据上下文动态调整

4. **硬件感知超参搜索**：

    - 功能：优化 key/value 维度和注意力头数
    - 核心发现：**KV cache 大小比参数量更影响生成吞吐**（因解码阶段通常是内存带宽瓶颈）
    - 核心思路：固定 KV cache 大小（匹配原始设计），在 key 维度、value 维度和头数上做网格搜索。最终设计使用更少 key 维度、更多 value 维度和更多头数，参数量更大但 KV cache 大小不变→吞吐一致但精度更高
    - 最优配置：$d_K=96, d_V=256, n_{\text{head}}=12$（vs 原始 $d_K=192, d_V=192, n_{\text{head}}=8$）

### 训练策略
- 第一阶段：冻结 MLP，用蒸馏损失训练注意力块，50B tokens（Nemotron-CC + Redstone-QA）
- 第二阶段：全模型训练，加入数学和代码数据，350B tokens

## 实验关键数据

### 主实验 —— MMLU(-Pro) 和 BBH

| 模型 | 类型 | 参数(B) | Cache(MB) | 吞吐(tok/s) | MMLU | MMLU-Pro | BBH |
|------|------|---------|-----------|-------------|------|----------|-----|
| Qwen2.5-1.5B | $O(n^2)$ | 1.5 | 1,792 | 241 | 59.5 | 28.9 | 44.1 |
| Qwen3-1.7B-Base | $O(n^2)$ | 1.7 | 7,168 | 61 | 60.3 | 37.8 | 54.2 |
| Llama3.2-3B | $O(n^2)$ | 3.0 | 7,168 | 60 | 54.9 | 25.0 | 47.1 |
| Mamba2-2.7B | $O(n)$ | 2.7 | 80 | 2,507 | 25.1 | 8.6 | 25.7 |
| RWKV7-1.5B | $O(n)$ | 1.5 | 24 | 3,050 | 41.0 | 13.4 | 15.9 |
| **Jet-Nemotron-2B** | **Hybrid** | **2.0** | **154** | **2,885** | **60.8** | **39.0** | **58.3** |
| Jet-Nemotron-4B | Hybrid | 4.0 | 258 | 1,271 | 65.2 | 44.2 | 65.0 |

### PostNAS 各步骤消融

| 步骤 | MMLU 提升 | 数学提升 | 检索提升 | 常识提升 |
|------|-----------|----------|----------|----------|
| 全注意力放置 | +5.3 | — | +7.8 | — |
| 线性注意力块选择 | — | +6.3 | — | +0.6 |
| JetBlock 动态卷积 | +0.7 | +0.5 | +0.6 | -0.2 |
| 硬件感知搜索 | +1.8 | +2.1 | +0.5 | +1.0 |
| **总提升** | **+5.3** | **+8.4** | **+7.8** | **+3.2** |

### 数学任务

| 模型 | 吞吐 | Avg | GSM8K | MATH | MathQA |
|------|------|-----|-------|------|--------|
| Qwen3-1.7B-Base | 61 | 42.3 | 62.8 | 16.7 | 46.0 |
| Jet-Nemotron-2B | 2,885 | **49.6** | **76.2** | **23.3** | **53.8** |

### 关键发现
- Jet-Nemotron-2B 甚至超越参数量更大的 MoE 模型（DeepSeek-V3-Small 2.2B/15B 在 MMLU 上仅 53.3 vs 60.8）
- 在 256K 上下文长度下，Jet-Nemotron-2B 相比 Qwen3 实现 6.14× prefilling 加速和 53.6× 解码加速
- KV cache 大小是生成吞吐的关键因素（而非参数量）——Jet-Nemotron-2B 仅 154MB cache vs Qwen3 的 7168MB
- MMLU 类多选任务主要依赖 softmax 的模式匹配能力，滑动窗口注意力即可保持精度

## 亮点与洞察
- **PostNAS 范式创新**：不从头训练而是从现有模型出发做架构搜索，大幅降低探索成本和风险。如果新设计在这个框架下都不行，基本也不值得从头训练
- **JetBlock 的动态卷积**：用可学习的核生成器替代固定卷积核，开销极小（降维比 1/8）但精度提升在检索和数学上可观
- **硬件感知设计**："KV cache 大小比参数量更影响吞吐"这一发现非常实用，意味着可以用更多参数换更高精度而不牺牲吞吐
- **全注意力放置的 task-specific 重要性**：不同任务需要不同位置的全注意力层，统一放置是次优的

## 局限与展望
- PostNAS 的起点模型决定了天花板——当 Qwen2.5-1.5B 基座不够强时后续搜索也受限
- 仅探索了注意力块设计，MLP 冻结策略限制了对整体架构的优化
- 训练需要 350B tokens 的第二阶段全模型训练，成本仍然不低
- JetBlock 的动态卷积核尺寸、降维比等超参未做系统搜索
- 长上下文真实应用（如 RAG、长文档 QA）的端到端评测不够

## 相关工作与启发
- **vs Mamba2/RWKV7**：纯线性注意力模型吞吐高但精度差（MMLU 差 20-35 点），Jet-Nemotron 混合设计两全其美
- **vs Qwen3/Gemma3**：全注意力 SOTA 精度高但吞吐低 40-50×，Jet-Nemotron 在精度更优的前提下大幅提升吞吐
- **vs Hymba/Zamba2**：先前混合模型精度仍显著低于全注意力 SOTA，PostNAS 首次让混合模型追平甚至超越
- **vs 从头训练 NAS**：传统 NAS 需要搜索+完整预训练，PostNAS 仅需搜索+少量重训练

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ PostNAS 范式新颖，JetBlock 动态卷积有创意，硬件感知搜索有独到见解
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 MMLU/数学/常识/检索/代码/长上下文六大类，对比全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，但某些细节需查附录
- 价值: ⭐⭐⭐⭐⭐ 实用性极强，为高效 LLM 架构设计提供了可复现的完整流水线

<!-- RELATED:START -->

## 相关论文

- [Learning to Better Search with Language Models via Guided Reinforced Self-Training](learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)
- [Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [NADER: Neural Architecture Design via Multi-Agent Collaboration](../../CVPR2025/model_compression/nader_neural_architecture_design_via_multi-agent_collaboration.md)
- [PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery](../../ICLR2026/model_compression/paser_post-training_data_selection_for_efficient_pruned_large_language_model_rec.md)
- [Vision-centric Token Compression in Large Language Model](vision-centric_token_compression_in_large_language_model.md)

<!-- RELATED:END -->
