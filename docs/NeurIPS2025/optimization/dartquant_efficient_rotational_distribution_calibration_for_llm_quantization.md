---
title: >-
  [论文解读] DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization
description: >-
  [NeurIPS 2025][优化/理论][LLM量化] DartQuant 提出基于分布校准的旋转矩阵优化方法，通过 Whip 损失将激活值分布推向均匀分布以减少量化误差，并用 QR-Orth 替代昂贵的流形优化器，在 70B 模型上实现 47× 加速和 10× 内存节省…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "LLM量化"
  - "旋转矩阵"
  - "分布校准"
  - "正交优化"
  - "后训练量化"
---

# DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization

**会议**: NeurIPS 2025  
**arXiv**: [2511.04063](https://arxiv.org/abs/2511.04063)  
**代码**: [https://github.com/CAS-CLab/DartQuant](https://github.com/CAS-CLab/DartQuant)  
**领域**: 优化  
**关键词**: LLM量化, 旋转矩阵, 分布校准, 正交优化, 后训练量化

## 一句话总结
DartQuant 提出基于分布校准的旋转矩阵优化方法，通过 Whip 损失将激活值分布推向均匀分布以减少量化误差，并用 QR-Orth 替代昂贵的流形优化器，在 70B 模型上实现 47× 加速和 10× 内存节省，首次在单张 3090 GPU 上完成大模型旋转校准。

## 研究背景与动机

**领域现状**：LLM 量化（PTQ）是部署大模型的关键技术，激活值中的极端异常值是量化精度下降的主因。旋转矩阵变换可有效平滑异常值，且因可逆性和范数保持特性可无损嵌入模型架构。

**现有痛点**：SpinQuant、OSTQuant 将旋转矩阵视为网络参数做端到端微调，需要 Cayley SGD 等专用优化器。70B 模型需数百 GiB 显存和数十 GPU 小时。

**核心矛盾**：端到端微调面临三重困难——(a) 计算资源消耗与 PTQ 快速部署目标矛盾；(b) 小样本微调容易过拟合校准集分布；(c) Cayley/Riemannian SGD 计算开销约为标准优化器的 2×。

**本文目标** 如何在极低资源下快速优化旋转矩阵，同时避免过拟合。

**切入角度**：从分布变换角度重新定义旋转优化——不微调任务损失，而是直接约束旋转后激活值分布趋向均匀分布。

**核心 idea**：用 Whip 损失驱动激活值分布从 Laplace 转向均匀分布来减少异常值，用 QR 分解替代流形优化保证正交性。

## 方法详解

### 整体框架
DartQuant 流程：(1) 少量校准数据前向传播得到各层激活值；(2) Token 采样；(3) 用 QR-Orth 优化器最小化 Whip 损失，间接优化旋转矩阵 $R$；(4) $R$ 融合到模型权重中，零额外推理开销。整个过程独立于任务损失，每层独立校准，无需端到端反向传播。

### 关键设计

1. **旋转分布校准（Rotational Distribution Calibration）**:

    - 功能：通过约束旋转后激活值的分布形状来优化旋转矩阵
    - 核心思路：原目标为 $\min_R \sum_{i=1}^{c_{in}} \mathbb{I}(|(Rx)_i| > \tau)$，指示函数不可微。方差受范数不变性约束难优化，峰度收敛慢，需要新的替代目标
    - 设计动机：摆脱端到端微调对任务损失的依赖，从分布层面直接优化

2. **Whip 损失函数**:

    - 功能：将激活值分布从 Laplace 推向均匀分布
    - 核心思路：激活值近似 $\text{Laplace}(0, b)$，其 CDF 变换 $U_X(x) = \tau[\exp(x/b)-1]$（$x \leq 0$）扩展零附近、压缩远端。受此启发：$\text{Whip} = \sum_{i=1}^{c_{in}} \exp(-|x_i|)$
    - 设计动机：Whip 在零附近梯度最大，将小值激活推离零点。范数不变约束下小值增大则异常值自然压缩，产生"聚合效应"，分布趋向均匀

3. **QR-Orth 优化方案**:

    - 功能：用 QR 分解替代 Cayley/Riemannian SGD 实现正交约束
    - 核心思路：对任意矩阵 $Z$ 做 QR 分解得正交矩阵 $R$，优化潜在参数 $Z$（标准梯度下降），每步重新分解得 $R$。复杂度 $\frac{4}{3}n^3$ 远低于 Cayley 的 $6n^3$
    - 设计动机：避免流形投影运算，可用任意标准优化器（SGD/Adam），实测 1.4× 加速

### 训练策略
- 校准数据：128 条 WikiText2 样本，token 长度 2048
- 优化器：SGD + QR-Orth，权重重建搭配 GPTQ
- 旋转矩阵 $R_1, R_2$ 由 QR-Orth 优化，$R_3, R_4$ 使用随机 Hadamard

## 实验关键数据

### 主实验

| 模型 | 量化配置 | 方法 | PPL ↓ | 0-shot Acc ↑ |
|------|---------|------|-------|-------------|
| Llama-2 7B | W4A4KV16 | QuaRot | 20.63 | 57.90 |
| Llama-2 7B | W4A4KV16 | SpinQuant | 19.90 | 57.85 |
| Llama-2 7B | W4A4KV16 | OSTQuant | 19.24 | 57.94 |
| Llama-2 7B | W4A4KV16 | **DartQuant** | **18.53** | **58.05** |
| Llama-3 70B | W4A4KV16 | SpinQuant | 9.61 | 66.06 |
| Llama-3 70B | W4A4KV16 | OSTQuant | 7.67 | 67.94 |
| Llama-3 70B | W4A4KV16 | **DartQuant** | 7.99 | **69.39** |

### 优化成本对比

| 指标 | 方法 | 7B | 13B | 70B |
|------|------|-----|------|------|
| 时间(GPU hr) | SpinQuant | 0.30 | 0.70 | 42.90 |
| 时间(GPU hr) | OSTQuant | 0.30 | 0.80 | 44.00 |
| 时间(GPU hr) | **DartQuant** | **0.14** | **0.23** | **0.91** |
| 显存(GiB) | SpinQuant | 19.98 | 33.73 | 238.89 |
| 显存(GiB) | OSTQuant | 42.25 | 239.16 | 583.86 |
| 显存(GiB) | **DartQuant** | **17.41** | **21.40** | **23.47** |

70B 模型：仅需 0.91 GPU hr + 23.47 GiB，首次在单张 3090 上完成（约 3 小时）。

### 消融实验

| 优化目标 | 效果 | 说明 |
|---------|------|------|
| 量化损失 | 几乎无变化 | 梯度信号弱 |
| 方差 | 几乎无变化 | 范数不变性约束 |
| 峰度 | 略有改善 | 收敛慢 |
| **Whip** | **显著改善** | 分布趋近均匀，快速收敛 |

| 优化器方案 | 100步耗时 | 加速比 |
|-----------|----------|--------|
| Cayley SGD | 8.2h | 1.0× |
| QR-Orth SGD | 5.7h | 1.44× |
| QR-Orth SGD (含收敛加速) | — | **41×** |

### 关键发现
- Whip 是唯一能显著改变激活分布的优化目标，其他目标几乎无效
- DartQuant 对校准数据集不敏感（WikiText2/PTB/C4 结果一致），端到端微调方法波动显著
- QR-Orth 每步更快（1.4×）且收敛更快（6 步 ≈ Cayley 100 步），综合加速 41×

## 亮点与洞察
- **分布校准替代端到端微调**：从"优化任务损失"转向"优化激活分布形状"，大幅降低成本且避免过拟合，可推广到其他正交约束场景
- **Whip 损失的理论基础**：利用 Laplace → 均匀分布的 CDF 变换形式，巧妙将分布变换转化为简单指数损失
- **QR-Orth 的通用性**：任何需要正交矩阵优化的场景都可用 QR 分解间接优化，彻底绕过流形优化复杂性

## 局限与展望
- 仅验证 Llama 系列和少量 MoE 模型，对 Qwen、Gemma 等架构泛化性未验证
- Whip 损失假设激活值近似 Laplace 分布，分布偏移大的模型效果可能下降
- 在线旋转 $R_3, R_4$ 仍用随机 Hadamard，可探索分布校准优化
- 未讨论 2-bit 等更极端配置的表现

## 相关工作与启发
- **vs QuaRot**: QuaRot 用随机 Hadamard 旋转，DartQuant 通过分布校准学习最优旋转矩阵
- **vs SpinQuant**: SpinQuant 端到端微调旋转矩阵，资源消耗大且过拟合风险高；DartQuant 逐层校准，成本降低 47×
- **vs OSTQuant**: OSTQuant 同时优化旋转和缩放但依赖端到端训练；DartQuant 独立于任务损失

## 评分
- 新颖性: ⭐⭐⭐⭐ 分布校准视角是旋转量化新思路，但旋转矩阵量化领域已有不少工作
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型多配置全面对比，消融详尽，效率对比有说服力
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰图表丰富，动机推导自然
- 价值: ⭐⭐⭐⭐⭐ 70B 模型单卡 3090 可校准，极大降低量化部署门槛，工程价值突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] OrthoGrad Improves Neural Calibration](orthograd_improves_neural_calibration.md)
- [\[CVPR 2025\] Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](../../CVPR2025/optimization/automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)
- [\[NeurIPS 2025\] Efficient Adaptive Experimentation with Noncompliance](efficient_adaptive_experimentation_with_noncompliance.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](efficient_adaptive_federated_optimization.md)
- [\[ICML 2026\] Memory-Efficient LLM Pretraining via Minimalist Optimizer Design](../../ICML2026/optimization/memory-efficient_llm_pretraining_via_minimalist_optimizer_design.md)

</div>

<!-- RELATED:END -->
