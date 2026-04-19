---
title: >-
  [论文解读] BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference
description: >-
  [ICML 2025][模型压缩][混合格式量化] 提出 BlockDialect——对权重和激活进行块级细粒度混合格式量化，为每个 block 从 FP4 变体（方言）格式书中选择最优数值格式，在 LLaMA3-8B 上比 MXFP4 准确率提升 10.78%，仅比全精度低 5.45%。
tags:
  - ICML 2025
  - 模型压缩
  - 混合格式量化
  - FP4变体
  - 细粒度量化
  - 能效推理
  - 激活量化
---

# BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference

**会议**: ICML 2025  
**arXiv**: [2501.01144](https://arxiv.org/abs/2501.01144)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 混合格式量化, FP4变体, 细粒度量化, 能效推理, 激活量化

## 一句话总结
提出 BlockDialect——对权重和激活进行块级细粒度混合格式量化，为每个 block 从 FP4 变体（方言）格式书中选择最优数值格式，在 LLaMA3-8B 上比 MXFP4 准确率提升 10.78%，仅比全精度低 5.45%。

## 研究背景与动机

**领域现状**：4-bit 量化是 LLM 部署的关键技术。硬件支持的细粒度块量化（如 Microscaling/MX 格式）正在成为主流。

**现有痛点**：(a) 现有方法主要关注"如何缩放"（scale），忽略了"如何表示"（represent）; (b) 固定的数值格式（如 FP4 E2M1）无法适应不同数据块的分布差异; (c) 激活量化仍然困难——动态范围大、通道异常值问题。

**核心矛盾**：既然每个 block 有独立的 scale factor，为什么不也给它独立的数值格式？

**本文目标**：设计块级混合数值格式量化。

**切入角度**：设计 FP4 变体（方言）来适应不同 block 的数据分布，用两阶段方法实现在线最优格式选择。

**核心 idea**：格式书（formatbook）包含多个 FP4 变体，每个 block 的最优格式由分布决定，且所有变体可表示为缩放整数，保证硬件能效。

## 方法详解

### 整体框架
1. 通过 block 级数据分布分析，设计 DialectFP4 格式书
2. 两阶段在线格式选择：粗分类→精选格式
3. 用逻辑操作实现量化，保持 FP4 MAC 单元的面积和能效

### 关键设计

1. **DialectFP4 格式书**:

    - 功能：设计一组 FP4 变体，覆盖不同 block 数据分布
    - 核心思路：分析发现 FP4 E2M1 的可表示值 {0, 0.5, 1, 1.5, 2, 3, 4, 6} 基本匹配矩阵级分布，但不同 block 的最大值分布差异很大→设计变体调整大值区域的表示密度
    - 设计动机：关键洞察是所有变体的值都选为 0.5 的倍数（缩放整数），确保可用低精度整数 MAC 计算

2. **两阶段在线格式选择**:

    - 功能：为每个激活 block 实时选择最优格式
    - 核心思路：第一阶段根据 block 最大值粗分类；第二阶段在候选格式中精选（基于简单统计）
    - 设计动机：避免 MSE 暴力搜索的计算开销，零样本性能接近 MSE 方法

3. **全路径量化**:

    - 功能：不仅量化线性层的权重-激活乘法，还量化注意力中的激活-激活乘法
    - 核心思路：将 KV cache 和中间激活都用 DialectFP4 量化
    - 设计动机：真正的端到端低精度，最大化能效收益

### 损失函数 / 训练策略
- 纯 PTQ，无需微调
- 格式选择在推理时在线执行

## 实验关键数据

### 主实验
LLaMA3-8B 零样本准确率：

| 方法 | 位宽 | 准确率 | vs FP16 |
|------|------|--------|---------|
| FP16 (全精度) | 16-bit | 69.23% | 基线 |
| MXFP4 | 4-bit | 53.00% | -16.23% |
| **BlockDialect** | **4-bit** | **63.78%** | **-5.45%** |

### 消融实验
| 配置 | 准确率 | 说明 |
|------|--------|------|
| 单一 FP4 E2M1 | 53.00% | 固定格式 |
| 混合 2 种格式 | 58.2% | 有限覆盖 |
| 完整 DialectFP4 | **63.78%** | 充分覆盖 |
| 仅线性层量化 | 67.47% | -1.76%，已很接近 FP16 |

### 关键发现
- 混合格式带来 10.78% 的准确率提升（53→63.78%）
- 全路径量化（含注意力）也保持可接受精度（-5.45%）
- 格式书中的变体数量 4-6 个就足够

## 亮点与洞察
- **"为什么不给每个 block 一个格式"**——这个问题非常自然但之前没人系统解决
- 缩放整数约束确保硬件兼容性，是实用化的关键
- 两阶段在线选择使方法可部署，不依赖离线校准

## 局限与展望
- 格式元数据（每 block 2-3 bits）增加存储开销
- 自定义 MAC 单元尚未有公开硬件实现
- 格式书设计依赖经验分析，自动化搜索可能更优

## 相关工作与启发
- **vs any4**: any4 用 LUT 做任意非均匀量化，BlockDialect 用格式书做结构化非均匀量化
- **vs MXFP4**: 标准细粒度格式，BlockDialect 在此基础上加混合格式
- **vs SmoothQuant**: 处理"如何缩放"，BlockDialect 处理"如何表示"

## 评分
- 新颖性: ⭐⭐⭐⭐ 混合数值格式量化视角新颖
- 实验充分度: ⭐⭐⭐⭐ 多模型、全路径量化、硬件分析
- 写作质量: ⭐⭐⭐⭐ 分析透彻，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 对下一代量化硬件设计有指导意义

<!-- RELATED:START -->

## 相关论文

- [MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](../../ACL2025/model_compression/moqae_mixed_precision_kv_cache.md)
- [Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation](sketch_to_adapt_fine-tunable_sketches_for_efficient_llm_adaptation.md)
- [OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference](orthorank_token_selection_via_sink_token_orthogonality_for_efficient_llm_inferen.md)
- [TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree](treelora_efficient_continual_learning_via_layer-wise_loras_guided_by_a_hierarchi.md)
- [ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](../../ICLR2026/model_compression/paroquant_pairwise_rotation_quantization_for_efficient_reasoning_llm_inference.md)

<!-- RELATED:END -->
