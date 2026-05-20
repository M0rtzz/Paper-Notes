---
title: >-
  [论文解读] Mamba-3: Improved Sequence Modeling using State Space Principles
description: >-
  [ICLR 2026][视频理解][状态空间模型] 从SSM视角提出三项核心改进：指数-梯形离散化、复值状态空间、多输入多输出(MIMO)公式化，在不增加解码延迟的前提下显著提升模型质量和状态追踪能力，推进性能-效率Pareto前沿。
tags:
  - "ICLR 2026"
  - "视频理解"
  - "状态空间模型"
  - "Mamba"
  - "序列建模"
  - "推理效率"
  - "MIMO"
---

# Mamba-3: Improved Sequence Modeling using State Space Principles

**会议**: ICLR 2026  
**arXiv**: [2603.15569](https://arxiv.org/abs/2603.15569)  
**代码**: [有](https://github.com/state-spaces/mamba)  
**领域**: 视频理解  
**关键词**: 状态空间模型, Mamba, 序列建模, 推理效率, MIMO

## 一句话总结

从SSM视角提出三项核心改进：指数-梯形离散化、复值状态空间、多输入多输出(MIMO)公式化，在不增加解码延迟的前提下显著提升模型质量和状态追踪能力，推进性能-效率Pareto前沿。

## 研究背景与动机

推理时计算(test-time compute)已成为LLM性能的关键驱动力，思维链推理和迭代细化等技术使推理效率成为模型设计的核心关注点。Transformer虽然是当前行业标准，但受制于：
- **二次计算复杂度**：自注意力机制
- **线性内存需求**：KV cache随序列长度线性增长

亚二次模型（SSM、线性注意力）虽提供常数内存和线性计算，但仍有三大不足：

**表达力受限**：Mamba-2为提升训练速度牺牲了部分表达力，较Mamba-1表现下降

**缺乏状态追踪能力**：无法解决简单的奇偶校验(parity)等任务

**硬件效率低**：解码阶段算术密度(arithmetic intensity)仅约2.5 ops/byte，大量硬件闲置

## 方法详解

### 整体框架

Mamba-3在Mamba-2基础上引入三项由SSM视角驱动的核心改进，外加若干架构优化。整体架构沿用Llama风格，交替排列Mamba-3块和SwiGLU MLP块，采用pre-norm。

### 关键设计

#### 1. 指数-梯形离散化 (Exponential-Trapezoidal Discretization)

**背景**：将连续时间SSM转化为离散递推。Mamba-1/2使用的离散化缺乏理论证明。

**本文贡献**：
- 形式化了Mamba-1/2的启发式离散化为"指数-Euler"方法（一阶近似，误差 $O(\Delta_t^2)$）
- 提出"指数-梯形"方法（二阶近似，误差 $O(\Delta_t^3)$）

指数-梯形递推：

$$\mathbf{h}_t = e^{\Delta_t A_t}\mathbf{h}_{t-1} + (1-\lambda_t)\Delta_t e^{\Delta_t A_t}\mathbf{B}_{t-1}x_{t-1} + \lambda_t\Delta_t\mathbf{B}_t x_t$$

其中 $\lambda_t \in [0,1]$ 是数据依赖的标量。当 $\lambda_t=1$ 时退化为Mamba-2的Euler方法，$\lambda_t=\frac{1}{2}$ 时为经典梯形法。

**等价卷积视角**：该递推等价于对状态输入 $\mathbf{B}_t x_t$ 施加一个宽度为2的数据依赖卷积，然后进入线性递推。这与Mamba中外部施加的标准短卷积本质不同——它是递推核心内部的卷积。

**并行形式**：通过SSD框架，新递推对应的结构化掩码 $\mathbf{L}$ 是1-半可分矩阵与2-带矩阵的乘积（特殊的2-半可分矩阵），支持高效的矩阵乘法并行计算。

#### 2. 复值状态空间模型 (Complex-Valued SSM)

**动机**：实值SSM（如Mamba-2）的转移矩阵特征值受限于实数，无法表示"旋转"动力学，例如奇偶校验可用旋转矩阵 $\mathbf{R}(\pi x_t)$ 表达。

**方法**：将SSM的底层参数扩展为复数值。关键等价性：

复值SSM离散化后等价于一个实值SSM加上数据依赖的旋转嵌入(RoPE)：

$$\mathbf{h}_t = e^{\Delta_t A_t}\mathbf{R}_t\mathbf{h}_{t-1} + \Delta_t\mathbf{B}_t x_t$$

其中 $\mathbf{R}_t$ 是块对角旋转矩阵，旋转角度由数据投影产生。

**RoPE trick**：通过将旋转矩阵累积应用于B/C投影（对应注意力中的Q/K），可高效实现复值SS，计算开销极小。这建立了复值SSM与数据依赖RoPE之间的理论联系。

#### 3. 多输入多输出 (MIMO) SSM

**动机分析**：SSM解码的算术密度极低。标准SISO的算术密度约2.5 ops/byte，而H100的matmul峰值约295 ops/byte，说明SSM解码严重内存受限。

**SISO→MIMO转换**：
- 将 $\mathbf{B}_t \in \mathbb{R}^N \to \mathbb{R}^{N \times R}$
- 将 $\mathbf{x}_t \in \mathbb{R}^P \to \mathbb{R}^{P \times R}$
- 外积 $\mathbf{B}_t\mathbf{x}_t^\top$ 变为矩阵乘法（利用tensor core）

效果：FLOPs增加 $R$ 倍但墙钟延迟几乎不变（因计算与内存IO重叠），算术密度从 $\Theta(1)$ 提升至 $\Theta(R)$。

**训练**：MIMO可分解为 $R^2$ 个SISO的并行调用。通过调整chunk size为 $C_{\text{MIMO}} = \frac{1}{R}C_{\text{SISO}}$，总FLOPs增长仅为 $R$ 倍（而非 $R^2$）。

**参数匹配**：MIMO增加的参数通过减小MLP宽度补偿（1.5B模型中MLP仅缩减6.6%）。

#### 4. 架构优化

- **BC归一化**：在B/C投影后添加RMSNorm（类似Transformer的QKNorm），可移除post-gate RMSNorm
- **B/C偏置**：添加可学习头特定偏置，提供数据无关成分（类卷积行为）
- **移除短卷积**：指数-梯形离散化 + B/C偏置的组合使已有的短卷积可以完全移除

### 损失函数 / 训练策略

- 标准语言建模训练：100B FineWeb-Edu tokens，Llama-3.1 tokenizer，2K上下文
- 所有规模使用相同训练协议，便于公平比较
- MIMO rank $R=4$，通过减小MLP宽度保持参数量匹配

## 实验关键数据

### 主实验

1.5B参数模型在100B FineWeb-Edu tokens上训练，8个下游任务平均准确率：

| 模型 | FW-Edu ppl↓ | 下游Avg Acc↑ |
|------|:-:|:-:|
| Transformer-1.5B | 10.51 | 55.4 |
| GDN-1.5B | 10.45 | 55.8 |
| Mamba-2-1.5B | 10.47 | 55.7 |
| **Mamba-3-SISO-1.5B** | **10.35** | **56.4** |
| **Mamba-3-MIMO-1.5B** | **10.24** | **57.6** |

Mamba-3 SISO比次佳模型GDN提升 +0.6点；MIMO进一步提升 +1.2点，共 +1.8点。MIMO的PPL改善0.11。

| 模型 | 180M | 440M | 880M | 1.5B |
|------|:-:|:-:|:-:|:-:|
| Mamba-2 | 42.9 | 49.6 | 53.4 | 55.7 |
| Mamba-3 SISO | 43.4 | 49.8 | 54.4 | 56.4 |
| Mamba-3 MIMO | 43.5 | 51.0 | 55.3 | 57.6 |

所有模型规模上Mamba-3均优于基线。

### 消融实验

**组件消融**（440M规模）：

| 变体 | PPL↓ |
|------|:-:|
| Mamba-3 - bias - trap | 16.68 |
| Mamba-3 - bias | 16.49 |
| Mamba-3 | 15.72 |
| Mamba-3 + conv | 15.85 |

B/C偏置和梯形离散化协同作用，使短卷积变为可选（加卷积反而PPL更高）。

**状态追踪任务**：

| 模型 | Parity↑ | 算术(无括号)↑ | 算术(有括号)↑ |
|------|:-:|:-:|:-:|
| Mamba-2 | 0.90 | 47.81 | 0.88 |
| Mamba-3 (w/o RoPE) | 2.27 | 1.49 | 0.72 |
| Mamba-3 (w/ Std. RoPE) | 1.56 | 20.70 | 2.62 |
| **Mamba-3** | **100.00** | **98.51** | **87.75** |
| GDN [-1,1] | 100.00 | 99.25 | 93.50 |

数据依赖RoPE是状态追踪的关键：Mamba-2完全失败（接近随机），Mamba-3近乎完美解决Parity。

**状态大小实验**：Mamba-3 MIMO在 $d_{\text{state}}=64$ 时达到Mamba-2在 $d_{\text{state}}=128$ 时的PPL，效果为用一半延迟达到相同质量。

### 关键发现

- 指数-梯形离散化的 $\lambda_t$ 不强制约束为 $\frac{1}{2}$ 时实证效果更好
- BF16精度下Mamba-3 SISO核实际比Mamba-2和GDN更快（0.156ms vs 0.203ms vs 0.257ms）
- MIMO $R=4$ 仅增加约2×训练耗时，但解码延迟几乎不变
- 混合模型（Mamba-3 + NoPE注意力，5:1比例）在检索任务上显著优于纯线性模型

## 亮点与洞察

1. **SSM视角的统一性**：三项改进都自然源于SSM连续时间视角，但从线性注意力或测试时回归视角难以得出
2. **理论贡献实质性**：首次严格证明Mamba-1/2的离散化为"指数-Euler"，并给出更优的梯形推广
3. **复值SSM → RoPE**：建立了复值SSM与数据依赖RoPE的等价关系，统一了两个独立发展方向
4. **MIMO的实用价值**：增加计算但不增加延迟，完美利用了解码阶段的硬件闲置

## 局限与展望

- 纯线性模型在半结构化/非结构化数据提取(SWDE, FDA)上仍明显弱于Transformer
- 混合模型中norm类型和位置的选择仍不明确，存在竞争性权衡
- 仅在≤1.5B规模和100B tokens上验证，大规模结果待确认
- MIMO的最优rank $R$ 选择尚需理论指导
- 长上下文外推能力需要额外的RMSNorm层（增加复杂度）

## 相关工作与启发

- 与Gated DeltaNet(GDN)竞争性能但方法论完全不同（SSM离散化 vs. delta规则）
- 复值SSM的RoPE等价性可能启发Transformer中位置编码的新设计
- MIMO的算术密度优化思路可泛化到其他内存受限计算场景
- 已被NVIDIA Nemotron、阿里Qwen3等大规模混合模型采用，验证了工业可行性

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 三项改进均有理论新意，复值SSM-RoPE等价性尤为漂亮
- 技术深度：⭐⭐⭐⭐⭐ — 从连续时间ODE到离散递推再到高效核实现，全链条覆盖
- 实验充分度：⭐⭐⭐⭐⭐ — 4个规模+合成任务+检索任务+核性能benchmark
- 实用价值：⭐⭐⭐⭐⭐ — 已开源并被工业界采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](../../NeurIPS2025/video_understanding/pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[AAAI 2026\] MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](../../AAAI2026/video_understanding/state-space_hierarchical_compression_with_gated_attention_an.md)
- [\[ICLR 2026\] The Expressive Limits of Diagonal SSMs for State-Tracking](the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)
- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](../../ECCV2024/video_understanding/videomamba_state_space_model_for_efficient_video_understanding.md)

</div>

<!-- RELATED:END -->
