---
title: >-
  [论文解读] TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering
description: >-
  [ACL 2025][模型压缩] 提出 TaDA——无需训练的 KV cache 压缩方法，通过对 K/V 激活做 head 维度均值中心化后量化偏差（而非原始激活），自动消除离群值问题，配合逐层自适应量化精度搜索，将 KV cache 压缩至原始 16 位的 27% 同时保持接近基线的精度。
tags:
  - ACL 2025
  - 模型压缩
---

# TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering

**会议**: ACL 2025  
**arXiv**: [2506.04642](https://arxiv.org/abs/2506.04642)  
**代码**: 无  
**领域**: LLM效率  

## 一句话总结

提出 TaDA——无需训练的 KV cache 压缩方法，通过对 K/V 激活做 head 维度均值中心化后量化偏差（而非原始激活），自动消除离群值问题，配合逐层自适应量化精度搜索，将 KV cache 压缩至原始 16 位的 27% 同时保持接近基线的精度。

## 背景与动机

1. **KV cache 是长上下文推理的主要瓶颈**：随序列长度线性增长，推理模型（DeepSeek-R1 等）的长思维链进一步加剧问题。
2. **现有量化方法需要单独处理离群值**：KIVI、GEAR、KVQuant 等方法在低精度量化时必须额外管理稀疏的离群元素（单独矩阵或低秩近似），增加了实现复杂度和存储开销。
3. **逐层量化精度未被充分探索**：现有方法大多对所有层使用统一量化精度，忽视了不同层对量化误差的敏感度差异。
4. **核心洞察**：对 KV 激活做 head 维度均值中心化后，偏差值分布更均匀、无极端离群值，可以直接低精度量化而无需额外处理。

## 方法详解

### 整体框架

TaDA 包含三个核心组件：**均值中心化**、**偏差量化**、**自适应精度搜索**。

### 1. 均值中心化

对 K/V 激活沿 head 维度计算均值：
$$K_m = \frac{1}{H}\sum_{i=1}^{H} K^i, \quad V_m = \frac{1}{H}\sum_{i=1}^{H} V^i$$

均值 $K_m, V_m$ 以全精度（16 位）存储，仅需 $\frac{1}{H}$ 的存储开销（如 32 头模型仅需 3.1%）。

### 2. 偏差计算与量化

计算每个头相对于均值的偏差并量化到低精度：
$$D_K^i = K_m - K^i, \quad D_V^i = V_m - V^i$$

推理时从均值和量化偏差重建：$\hat{K}^i = K_m - \text{quantize}(D_K^i)$

**存储分析**（以 LLaMA2-7B 32 头、4 位量化为例）：
- 均值：$\frac{1}{32} \approx 3.1\%$
- 量化偏差：$\frac{4}{16} = 25\%$  
- 缩放因子：$\frac{2}{128} \approx 1.6\%$
- **总计 ≈ 29%**

### 3. 辅助技术

**残差 token**：保留最近 R 个 token 的 KV 不压缩（全精度），超过阈值后批量压缩，减轻低精度对最近 token 的影响。

**逐层量化精度搜索**：用训练集小样本（1000 条 hotpotqa）通过随机搜索为每层找最优量化精度（{2, 4, 8}-bit），不同层可使用不同精度，进一步压缩 KV cache。

### 4. 高效实现

开发 3 个 Triton kernel 优化在线计算：
- 融合 RoPE 与 K 压缩
- 融合投影计算与 V 压缩  
- 适配 flash-decoding 的 TaDA 自注意力 kernel

## 实验结果

### 表1：LongBench 评估（平均分，8 个任务）

| 模型 | 方法 | KV cache 占比 | 平均分 |
|:---|:---|:---:|:---:|
| LLaMA2-7B | BF16 基线 | 100% | 46.03 |
| LLaMA2-7B | KIVI-2bit | 25% | 43.09 |
| LLaMA2-7B | GEAR | 31% | 45.38 |
| LLaMA2-7B | **TaDA** | **27%** | **45.87** |
| LLaMA3-8B-it | BF16 基线 | 100% | 49.51 |
| LLaMA3-8B-it | Quanto-4bit | 37% | 49.61 |
| LLaMA3-8B-it | **TaDA** | **35%** | **49.43** |
| Mistral-7B-it | BF16 基线 | 100% | 49.27 |
| Mistral-7B-it | **TaDA** | **35%** | **49.07** |

TaDA 以更少的 KV cache（27-35%）达到与 BF16 基线几乎相同的精度。

### 表2：GSM8k Chain-of-Thought 评估

| 模型 | 方法 | KV cache | GSM8k |
|:---|:---|:---:|:---:|
| LLaMA2-7B | BF16 | 100% | 21.30 |
| LLaMA2-7B | KIVI-2bit | 25% | 18.31 |
| LLaMA2-7B | **TaDA** | **27%** | **21.26** |
| LLaMA3-8B-it | BF16 | 100% | 67.62 |
| LLaMA3-8B-it | GEAR | 31% | 54.76 |
| LLaMA3-8B-it | **TaDA** | **35%** | **66.73** |

CoT 场景下 TaDA 几乎无损（LLaMA2 仅下降 0.04），远优于 GEAR（-12.86）。

## 亮点

- **均值中心化消除离群值**：简单优雅的核心思想——量化偏差而非原始激活，自动避免离群值问题，无需单独矩阵存储
- **无需训练**：纯推理时技术，不需要微调模型，即插即用
- **逐层自适应精度**：通过搜索为不同层分配不同量化位宽，更精细地平衡精度和压缩率
- **Triton kernel 融合实现**：三个自定义 kernel 隐藏在线压缩/解压延迟，实际部署友好

## 局限性

- **MHA vs GQA 效果差异**：在 MHA 模型（LLaMA2）上优势明显（head 多→均值化效果好），GQA 模型（LLaMA3、Mistral）上优势缩小
- **仅评估 7-8B 模型**：未在更大模型（70B+）上验证，压缩效果和精度保持可能不同
- **搜索成本**：需要用训练数据做逐层精度搜索，虽然只需一次但增加了部署流程
- **残差 token 引入额外复杂度**：极低精度下需要全精度缓冲区管理最近 token

## 相关工作对比

| 维度 | TaDA | KIVI | GEAR | KVQuant |
|:---|:---|:---|:---|:---|
| 量化方式 | 均值中心化+偏差量化 | per-channel K + per-token V | 均匀量化+低秩误差修复 | Sub-4bit 非均匀量化 |
| 离群值处理 | 自动消除（无需额外处理） | 需要管理 | 低秩+稀疏矩阵 | 旋转+非均匀码本 |
| 逐层自适应精度 | ✓（搜索） | ✗ | ✗ | ✗ |
| 需要训练 | ✗ | ✗ | ✗ | ✗ |
| KV cache 占比 | 27-35% | 25-37% | 31% | ~25% |

## 评分

- ⭐⭐⭐⭐ 新颖性：均值中心化消除离群值的思路简单但有效，逐层自适应精度搜索是新增量
- ⭐⭐⭐⭐ 实用性：无需训练、Triton kernel 已实现、AMD GPU 验证，部署路径清晰
- ⭐⭐⭐ 实验充分度：4 个模型、长上下文+CoT 两类场景，但缺少大模型和推理延迟/吞吐的端到端评估
- ⭐⭐⭐ 写作质量：方法清晰但 GQA 场景效果接近 Quanto 的分析不够深入

<!-- RELATED:START -->

## 相关论文

- [IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)
- [RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)

<!-- RELATED:END -->
