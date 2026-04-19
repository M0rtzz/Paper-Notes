---
title: >-
  [论文解读] Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models
description: >-
  [模型压缩][量化] OSP（Outlier-Safe Pre-Training）框架通过三项创新——Muon 优化器（消除特权基方向）、Single-Scale RMSNorm（防止通道放大）和可学习嵌入投影层（重分布嵌入层激活），在预训练阶段主动防止异常值形成，训练的 1.4B 模型在 1T tokens 上实现近零超额峰度（0.04 vs 标准模型的 1818.56），在激进4-bit量化下平均分 35.7（Adam 为 26.5），仅 2% 训练开销。
tags:
  - 模型压缩
  - 量化
  - 异常值消除
  - 预训练
  - Muon优化器
  - 低比特推理
---

# Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models

| 会议 | 领域 | arXiv | 代码 |
|------|------|-------|------|
| ACL2025 | Model Compression / Quantization | [2506.19697](https://arxiv.org/abs/2506.19697) | [GitHub](https://github.com/dmis-lab/Outlier-Safe-Pre-Training) |

**关键词**: 量化, 异常值消除, 预训练, Muon优化器, 低比特推理

## 一句话总结

OSP（Outlier-Safe Pre-Training）框架通过三项创新——Muon 优化器（消除特权基方向）、Single-Scale RMSNorm（防止通道放大）和可学习嵌入投影层（重分布嵌入层激活），在预训练阶段主动防止异常值形成，训练的 1.4B 模型在 1T tokens 上实现近零超额峰度（0.04 vs 标准模型的 1818.56），在激进4-bit量化下平均分 35.7（Adam 为 26.5），仅 2% 训练开销。

## 研究背景与动机

### 量化面临的根本障碍：激活异常值

LLM 在标准训练过程中不可避免地产生激活异常值（activation outliers），这些异常值会灾难性地膨胀量化缩放因子 $s$，导致严重的舍入误差和信息损失。对于 $n$-bit 量化：

$$\hat{x}_\text{int} = \text{clip}\left(\left\lfloor \frac{x}{s} \right\rceil + z, 0, 2^n - 1\right)$$

异常值使 $s$ 过大，正常值的量化精度被严重损害。

### 现有方法的局限

- **PTQ（后训练量化）**：如 GPTQ、QuaRot、SpinQuant 等，是"补救"而非"预防"，默认接受异常值为 LLM 的固有属性
- **QAT（量化感知训练）**：在训练中模拟量化误差，但不消除异常值本身，且显著拖慢训练
- **已有研究规模小**：大多限于 < 1B 参数或 < 100B tokens，未考虑生产级可扩展性

### 异常值的三种成因假说

**归一化层的通道缩放因子**（Kovaleva et al., 2021; Wei et al., 2022）

**注意力汇聚现象（Attention Sinks）**（Bondarenko et al., 2023）

**对角优化器（Adam/AdaFactor）的自适应梯度缩放**（He et al., 2024; Guo et al., 2024）

## 方法详解

### 整体框架

OSP 的三个组件协同工作，目标是在保持训练效率和架构兼容性的同时，从源头消除异常值。

### 组件一：Muon 优化器

**原理**：Adam 等对角优化器维护逐参数梯度方差统计，进行逐元素标准化——这引入了"特权基"（privileged basis），某些通道不成比例地累积幅值。

Muon 使用 Newton-Schulz 算法近似正交化梯度矩阵：

$$G = U\Sigma V^T \mapsto UV^T$$

关键优势：
- 仅维护梯度动量，不进行逐元素缩放
- 通过全秩线性变换更新参数，消除特权基
- 达到 Adam **97.9%** 的训练吞吐量，内存减少 33%

| 优化器 | 相对吞吐量 | 内存 | 编译时间 |
|--------|-----------|------|---------|
| Adam | 100% | $O(36LD^2)$ | 2m30s |
| Muon | 97.9% | $O(24LD^2)$ | 3m48s |
| Shampoo | 75.5% | $O(\frac{338}{3}LD^2)$ | 24m24s |

这是首次在 10 亿参数、万亿 token 规模上验证 Muon 的可扩展性。

### 组件二：Single-Scale RMSNorm（SSNorm）

**问题**：RMSNorm 的通道维度可学习缩放参数 $\gamma_i$ 构成显式的基对齐，某些通道可能被放大。

Simple RMSNorm（去掉所有可学习参数，除以 $\sqrt{d}$）能防止异常值，但带来两个问题：
- 初期训练：激活幅值被严重抑制，收敛缓慢
- 后期训练：固定缩放导致不稳定

**SSNorm 解决方案**：保留一个全局标量缩放参数 $\gamma \in \mathbb{R}$：

$$\text{SSNorm}(x) = \gamma \cdot \frac{x}{\|x\|_2}$$

所有维度共享同一个 $\gamma$，既保持动态缩放能力，又从根本上防止特权坐标的出现。

### 组件三：解耦嵌入优化 + 可学习嵌入投影（EmbProj）

**问题**：词嵌入矩阵维度极高，对 Muon 造成计算瓶颈（额外 6% 吞吐量下降）。且 Jordan et al. (2024) 表明解耦嵌入用 Adam 训练收敛更好。但这会重新引入异常值。

**解决方案**：
- 嵌入层仍用 Adam 训练（保持效率和收敛性）
- 在嵌入层之后、反嵌入层之前添加全秩投影矩阵
- 投影矩阵重分布异常值到不同维度，防止其集中和传播
- 训练后投影矩阵可吸收到相邻嵌入中，保持推理时计算不变

### 异常值量化指标

使用超额峰度（Excess Kurtosis）量化异常值程度：

$$\text{Kurt}[X] - 3 = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^4\right] - 3$$

## 实验

### 消融实验（100B tokens）

| 优化器 | SSNorm | EmbProj | Ex. Kurt | 4-4-4 Avg |
|--------|--------|---------|----------|-----------|
| Adam | ✗ | ✗ | 1818.56 | 26.8 |
| Muon | ✗ | ✗ | 1575.12 | 29.0 |
| Muon | ✓ | ✗ | 66.69 | 36.4 |
| Muon | ✗ | ✓ | 703.23 | 30.4 |
| **Muon (OSP)** | **✓** | **✓** | **0.04** | **38.9** |

关键发现：
- 三个组件必须**同时启用**才能将峰度维持在近零水平
- 任何单独组件都不足以完全消除异常值——一旦在网络某处形成，就会传播到整个架构

### 规模化到 1T tokens

在 12 个同规模开源 LLM 上对比 4-bit 量化性能：

| 模型 | Params | Tokens | 4-bit Avg |
|------|--------|--------|-----------|
| Pythia | 1.4B | 0.3T | 26.5 |
| TinyLlama | 1.1B | 2T | 26.4 |
| OLMo | 1.2B | 3T | 27.6 |
| Qwen 2 | 1.5B | 7T | 29.3 |
| LLaMA 3.2 | 1.2B | – | 28.1 |
| Adam (from scratch) | 1.4B | 1T | 26.5 |
| **Muon (OSP)** | **1.4B** | **1T** | **35.7** |

多数模型在 4-bit 量化下 ARC、CSQA 降到近随机水平（~25%），OSP 显著保持性能。

### 与 PTQ 方法的互补性

| 量化方法 | Adam PPL | OSP PPL |
|----------|----------|---------|
| RTN | 14475.51 | 45.92 |
| + GPTQ | 3723.46 | 14.29 |
| + QuaRot | 16.62 | **14.38** |
| + SpinQuant | 14.94 | **13.66** |

OSP 与所有 PTQ 方法互补，组合后进一步提升——因为 OSP 消除了异常值这一根本障碍，为 PTQ 校准提供了更好的起点。

### 注意力汇聚分析

**重要发现**：在消除异常值后，**注意力汇聚（attention sinks）仍然存在**，但机制不同：
- Adam 模型：通过将注意力 logit 驱向负无穷来实现"no-op"操作，产生巨大激活
- OSP 模型：通过集中正注意力在特定 token 上实现相同功能，无需极端激活

这挑战了"注意力汇聚导致异常值"的假设——注意力汇聚本身不是异常值的原因，而是有异常值倾向的模型采用了极端 logit 策略。

## 亮点与洞察

1. **根本性转变**："异常值不是 LLM 的固有属性，而是训练策略的后果"——这一结论具有范式转变意义
2. **实用性极强**：仅 2% 训练开销，保持标准 Transformer 架构，完全兼容现有推理管线
3. **首个生产级无异常值 LLM**：1.4B 参数 + 1T tokens，超越了之前小规模实验
4. **与 PTQ 正交**：不是替代 PTQ，而是提供更好的量化基础
5. **注意力汇聚的新理解**：证明注意力汇聚和异常值是可分离的现象

## 局限性

1. **仅验证 1.4B 规模**：未扩展到 3B 和 7B，这些是移动端部署的常见目标
2. **缺乏与其他二阶方法的广泛比较**：TPU 编译时间过长限制了消融实验
3. **SSNorm 的通用性**：单标量缩放是否在所有架构和规模上都有效尚待验证
4. **嵌入层仍用 Adam**：这是一个折中，可能在超大词汇表场景下引入问题

## 相关工作

- **量化方法**：RTN、GPTQ（Hessian 优化权重舍入）、SmoothQuant/AWQ（激活统计）、QuaRot/SpinQuant（旋转矩阵重分布）
- **异常值成因**：Elhage et al. (2023) 的特权基理论、Bondarenko et al. (2023) 的注意力汇聚假说
- **优化器**：K-FAC（Fisher 信息近似 Hessian）、Shampoo/SOAP（张量维度解耦预条件器）、Muon（Newton-Schulz 正交化，本文首次验证 trillion-scale）
- **归一化层**：Simple RMSNorm（Qin et al., 2023），He et al. (2024) 的无归一化方案

## 评分

⭐⭐⭐⭐⭐（5/5）

这是一项极具影响力的工作：从根本上证明 LLM 的激活异常值是可预防的而非不可避免的，并给出了实用的生产级解决方案。方法设计精巧（三个组件各解决一个异常值来源），实验扎实（从 100B 到 1T tokens 的全面验证），分析深入（注意力汇聚与异常值的关系重新审视）。对 LLM 高效部署具有重大意义。

<!-- RELATED:START -->

## 相关论文

- [EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)
- [PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](ptq161_low_bit_quantization.md)
- [Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [Accurate KV Cache Quantization with Outlier Tokens Tracing](accurate_kv_cache_quantization_with_outlier_tokens_tracing.md)
- [UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)

<!-- RELATED:END -->
