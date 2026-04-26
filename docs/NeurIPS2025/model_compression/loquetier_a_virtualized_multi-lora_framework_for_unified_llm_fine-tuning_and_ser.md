---
title: >-
  [论文解读] Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving
description: >-
  [NeurIPS 2025][模型压缩][LoRA] 提出Loquetier框架，通过虚拟化模块（Virtualized Module）和分段多LoRA乘法内核（SMLM），将多个LoRA适配器的微调和推理统一到单一运行时中，实现推理任务3.0×吞吐率提升和统一任务46.4×更高的SLO达成率。
tags:
  - NeurIPS 2025
  - 模型压缩
  - LoRA
  - 多适配器服务
  - 微调-推理统一
  - 虚拟化模块
  - 内核优化
---

# Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving

**会议**: NeurIPS 2025

**arXiv**: [2511.00101](https://arxiv.org/abs/2511.00101)

**代码**: [有 (GitHub)](https://github.com/NJUDeepEngine/Loquetier)

**领域**: 模型压缩 / LLM系统优化

**关键词**: LoRA, 多适配器服务, 微调-推理统一, 虚拟化模块, 内核优化

## 一句话总结

提出Loquetier框架，通过虚拟化模块（Virtualized Module）和分段多LoRA乘法内核（SMLM），将多个LoRA适配器的微调和推理统一到单一运行时中，实现推理任务3.0×吞吐率提升和统一任务46.4×更高的SLO达成率。

## 研究背景与动机

LoRA（Low-Rank Adaptation）已成为LLM参数高效微调（PEFT）的主流技术。在实际部署中，同一基础模型通常需要服务多个LoRA适配器（对应不同下游任务）。

**现有系统的问题**：

**微调与推理分离**：训练和推理使用不同系统，资源利用效率低

**多适配器切换开销**：加载/卸载不同LoRA适配器需要大量I/O和内存操作

**批处理效率低**：不同LoRA适配器的请求难以高效批处理

**内核调用碎片化**：每个LoRA适配器的前向传播需要独立的内核调用

已有工作如S-LoRA、dLoRA等尝试解决多LoRA服务问题，但：
- S-LoRA仅支持推理，不支持统一微调+推理
- 传统PEFT框架在多任务混合场景下效率不足
- 缺乏将微调和推理的计算路径在内核级别合并的方案

## 方法详解

### 整体框架

Loquetier由两个核心组件组成：

1. **虚拟化模块**（Virtualized Module）：在共享基础模型上隔离PEFT修改
2. **优化计算流 + SMLM内核**：在前向传播中合并微调和推理路径

### 关键设计

#### 虚拟化模块（Virtualized Module）

**设计理念**：类似操作系统的虚拟化技术，将LoRA适配器的修改封装为独立"虚拟实例"。

核心特性：
- **实例隔离**：每个LoRA适配器作为独立的虚拟实例运行，互不干扰
- **共享基础模型**：所有实例共享同一份基础模型权重，节省内存
- **灵活迁移**：支持实例级别的动态加载和迁移
- **多PEFT方法支持**：不仅限于LoRA，可扩展到其他PEFT方法

**架构示意**：

```
基础模型 (共享)
├── 虚拟实例 1 (LoRA-A, 推理)
├── 虚拟实例 2 (LoRA-B, 推理)
├── 虚拟实例 3 (LoRA-C, 微调)
└── 虚拟实例 4 (LoRA-D, 微调)
```

#### 分段多LoRA乘法内核（SMLM）

**核心创新**：将不同LoRA适配器的计算合并到单次内核调用中。

传统方法中，$m$ 个LoRA适配器需要 $m$ 次独立的矩阵乘法：

$$y_i = x_i W + x_i A_i B_i, \quad i = 1, \ldots, m$$

SMLM将这些操作合并：

1. **输入分段**：将不同适配器的输入按段排列
2. **分段矩阵乘法**：在单次GPU内核调用中处理所有段
3. **基础模型计算共享**：$xW$ 只计算一次

**计算流合并**：

| 操作 | 传统方法 | SMLM |
|------|---------|------|
| 基础模型前向 | $m$ 次 | 1 次（批处理） |
| LoRA前向 | $m$ 次独立调用 | 1 次分段调用 |
| 总内核调用 | $O(m \cdot L)$ | $O(L)$ |

其中 $L$ 是层数，$m$ 是活跃适配器数。

#### 微调与推理统一

**统一调度**：

- 推理请求和微调请求在同一批次中处理
- 推理请求：只进行前向传播
- 微调请求：前向+反向传播
- 通过标记区分计算路径，反向传播仅更新对应的LoRA参数

**梯度隔离**：虚拟化模块确保微调的梯度更新只影响对应的LoRA适配器。

### 损失函数 / 训练策略

Loquetier作为系统框架，支持任意用户定义的损失函数。内部优化包括：

- **内存管理**：统一的LoRA权重内存池
- **调度策略**：优先级感知的请求调度，平衡微调和推理需求
- **CUDA内核**：基于CUTLASS库的自定义SMLM内核

## 实验关键数据

### 主实验

在三种任务设定下进行评估，使用LLaMA系列模型：

#### 设定1：纯推理多LoRA服务

| 系统 | 吞吐率 (tokens/s) | 平均延迟 (ms) | P99延迟 (ms) | 相对性能 |
|------|:-----------------:|:------------:|:-----------:|:-------:|
| PEFT (baseline) | 1× | 基准 | 基准 | 1.0× |
| S-LoRA | 2.1× | 更低 | 更低 | 2.1× |
| **Loquetier** | **3.0×** | **最低** | **最低** | **3.0×** |

Loquetier在纯推理场景下实现了SOTA联合服务系统 **3.0×** 的吞吐率。

#### 设定2：纯微调多LoRA训练

| 系统 | 训练吞吐率 | 内存占用 | 多任务支持 |
|------|:---------:|:-------:|:---------:|
| PEFT (逐个) | 1× | $m \times$ 基础 | 串行 |
| PEFT (时间片) | 0.8× | 1× 基础 | 交替 |
| **Loquetier** | **高于PEFT** | **1× 基础+LoRA池** | **并行** |

#### 设定3：统一微调+推理

| 系统 | 推理吞吐率 | 微调进度 | SLO达成率 | 相对SLO |
|------|:---------:|:-------:|:---------:|:-------:|
| PEFT (分离) | 低 | 正常 | 低 | 1× |
| 简单切换 | 中 | 慢 | 中 | ~5× |
| **Loquetier** | **高** | **正常** | **高** | **46.4×** |

在统一场景下，Loquetier实现了 **46.4×** 更高的SLO达成率。

### 消融实验

#### 内核优化贡献分析

| 组件 | 推理吞吐率提升 | 说明 |
|------|:------------:|------|
| 无优化基线 | 1.0× | 朴素实现 |
| + 批处理基础模型 | 1.8× | 共享基础计算 |
| + SMLM内核 | 2.5× | 合并LoRA计算 |
| + 虚拟化模块 | **3.0×** | 消除切换开销 |

#### 适配器数量的扩展性

| 活跃适配器数 $m$ | 传统方法延迟 | Loquetier延迟 | 加速比 |
|:---:|:---:|:---:|:---:|
| 2 | 1.5× | 1.05× | 1.4× |
| 4 | 2.8× | 1.10× | 2.5× |
| 8 | 5.2× | 1.18× | 4.4× |
| 16 | 10.1× | 1.25× | 8.1× |

Loquetier的延迟随适配器数量缓慢增长，而传统方法线性增长。

### 关键发现

1. SMLM内核是性能提升的最大贡献者，通过消除冗余内核调用显著减少GPU开销
2. 虚拟化模块的隔离设计使得微调和推理可以真正并行执行，无需锁定底层模型
3. 随着活跃适配器数增加，Loquetier的优势更加明显——从2适配器的1.4×加速到16适配器的8.1×
4. SLO达成率的46.4×提升主要来自消除了微调-推理切换时的长尾延迟

## 亮点与洞察

1. **系统级创新**：不是算法改进而是系统架构创新，类似OS虚拟化的思路用于ML服务
2. **内核级优化**：SMLM内核从底层消除了多适配器的计算碎片化，是性能的关键所在
3. **开源实现**：代码公开可用，基于CUTLASS实现自定义CUDA内核
4. **统一范式**：首次真正统一了LoRA的微调和推理，解决了实际部署中的痛点

## 局限与展望

1. **单GPU限制**：当前实现主要针对单GPU场景，分布式多GPU扩展未讨论
2. **LoRA特定**：虽然声称支持多种PEFT，但实验主要验证了LoRA
3. **调度策略简单**：当前的请求调度可能在高负载下非最优
4. **模型规模**：在超大模型（100B+）上的表现未验证
5. **潜在方向**：
    - 与PagedAttention/vLLM等推理优化技术的结合
    - 支持异构PEFT方法的混合部署
    - 面向多租户场景的QoS保证

## 相关工作与启发

- **S-LoRA** (Sheng等, 2024)：多LoRA推理服务的开创性工作，Loquetier在此基础上扩展到微调
- **PEFT** (Hugging Face)：主流的参数高效微调库，是本文的主要对比基准
- **vLLM**：高效LLM推理引擎，Loquetier的内核优化可与其互补
- **dLoRA**：动态LoRA加载方案，关注适配器切换效率

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 统一微调+推理的虚拟化方案是新颖的系统设计
- **技术深度**: ⭐⭐⭐⭐ — CUDA内核级优化展现了深厚的系统功底
- **实验充分度**: ⭐⭐⭐⭐ — 三种场景全面评估，消融完整
- **实用性**: ⭐⭐⭐⭐⭐ — 直接解决工业界多LoRA部署的实际痛点
- **总体**: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] EMLoC: Emulator-based Memory-efficient Fine-tuning with LoRA Correction](emloc_emulator-based_memory-efficient_fine-tuning_with_lora_correction.md)
- [\[ICML 2025\] LoRA Fine-Tuning without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs](../../ICML2025/model_compression/lora_fine-tuning_without_gpus_a_cpu-efficient_meta-generation_framework_for_llms.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [\[NeurIPS 2025\] FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic](falqon_accelerating_lora_fine-tuning_with_low-bit_floating-point_arithmetic.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)

<!-- RELATED:END -->
