---
title: >-
  [论文解读] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization
description: >-
  [ICLR 2026][模型压缩][零阶优化] 提出QZO方法，通过对量化缩放因子（而非离散权重）做零阶扰动来估计梯度，配合方向导数裁剪稳定训练，实现4-bit/2-bit LLM的极致内存高效微调，总内存降低18倍以上。
tags:
  - ICLR 2026
  - 模型压缩
  - 零阶优化
  - 量化模型微调
  - 内存高效训练
  - 量化缩放因子
  - 梯度方差
---

# Fine-tuning Quantized Neural Networks with Zeroth-order Optimization

**会议**: ICLR 2026  
**arXiv**: [2505.13430](https://arxiv.org/abs/2505.13430)  
**代码**: [GitHub](https://github.com/maifoundations/QZO)  
**领域**: 模型压缩/高效微调  
**关键词**: 零阶优化, 量化模型微调, 内存高效训练, 量化缩放因子, 梯度方差

## 一句话总结
提出QZO方法，通过对量化缩放因子（而非离散权重）做零阶扰动来估计梯度，配合方向导数裁剪稳定训练，实现4-bit/2-bit LLM的极致内存高效微调，总内存降低18倍以上。

## 研究背景与动机

**领域现状**：LLM微调需要存储权重、梯度、优化器状态和激活值，典型7B模型需56GB。现有方法分别压缩不同组件：LoRA减参数、GaLore压缩优化器状态、MeZO用零阶优化消除梯度存储。

**现有痛点**：这些方法只解决了部分内存问题。权重本身仍占大量内存（7B模型bfloat16需14GB），即使用MeZO消除梯度也仍需14GB存权重。最直接的解决方案是量化权重（如int4只需3.5GB），但量化后权重是离散的，无法直接做零阶扰动。

**核心矛盾**：零阶优化需要在连续空间扰动权重→量化权重是离散的；估计的梯度是连续的→无法更新离散权重（需要反量化-再量化循环）。

**本文目标**：如何在量化模型上应用零阶优化，同时最大化内存压缩（权重+梯度+优化器状态全压缩）？

**切入角度**：观察到量化的本质是 $w = \Delta \cdot \bar{w}$，其中 $\Delta$ 是连续的缩放因子，$\bar{w}$ 是离散整数。可以扰动连续的 $\Delta$ 而保持 $\bar{w}$ 不变。

**核心 idea**：扰动连续的量化缩放因子做零阶梯度估计，用方向导数裁剪控制梯度方差。

## 方法详解

### 整体框架
QZO = Q-SPSA(量化零阶梯度估计) + DDC(方向导数裁剪)。保持量化整数权重 $\bar{\theta}$ 不变，只更新连续的缩放因子 $\Delta$。两次前向传播估计梯度，无需反向传播，无需梯度存储，无需优化器状态。

### 关键设计

1. **Q-SPSA (量化同步扰动随机近似)**:

    - 功能：将SPSA扩展到量化模型，扰动连续的缩放因子而非离散权重
    - 核心思路：$\hat{\nabla}_{\Delta}\mathcal{L} = \frac{\mathcal{L}((\Delta+\epsilon z)\odot\bar{\theta}) - \mathcal{L}((\Delta-\epsilon z)\odot\bar{\theta})}{2\epsilon}z$，其中 $z \sim \mathcal{N}(0, I_d)$
    - 设计动机：$\Delta$ 是连续的，可自然扰动和更新。反量化 $w = \Delta \cdot \bar{w}$ 与正常前向传播一致，无需修改推理代码。适用于scalar-based(GPTQ)和codebook-based(AQLM)两类量化方法。

2. **DDC (方向导数裁剪, Directional Derivative Clipping)**:

    - 功能：裁剪零阶梯度估计中的方向导数标量 $d$，稳定训练
    - 核心思路：$d' = \text{clip}(d, -C, C)$，梯度估计变为 $\hat{\nabla} = d' \cdot z$
    - 设计动机：零阶梯度估计有高方差问题（MeZO也存在）。Theorem 1证明裁剪后仍是无偏估计，且 $\text{Var}[\hat{\nabla}'] \leq \text{Var}[\hat{\nabla}]$（因为 $d'^2 \leq d^2$）。

3. **内存种子技巧**:

    - 功能：用随机种子重现扰动向量 $z$，避免存储
    - 核心思路：与MeZO相同，用种子编号代替 $z$ 的存储
    - 设计动机：$z$ 与模型同维度，存储它会抵消节省

### 损失函数 / 训练策略
- ZO-SGD更新：$\Delta_{t+1} = \max(\Delta_t - \eta \cdot d' \cdot z, 0)$（确保缩放因子非负）
- 学习率 $10^{-7}$，扰动尺度 $\epsilon = 10^{-3}$，裁剪阈值 $C = 100$
- 可选：Q-SPSA更新 $\Delta$ + SPSA联合更新未量化部分

## 实验关键数据

### 主实验
4-bit GPTQ量化模型（SST-2/RTE/CB/BoolQ/SQuAD）：

| 方法 | 精度 | 内存 | SST-2 | RTE | SQuAD |
|------|------|------|-------|-----|-------|
| Zero-Shot | 16bit | 14GB | 基线 | 基线 | 基线 |
| Fine-tuning+AdamW | 16bit | 56GB | 上界 | 上界 | 上界 |
| MeZO | 16bit | 14GB | 好 | 好 | 好 |
| QZO (4bit) | 4bit | **<3GB** | 接近MeZO | 接近MeZO | 接近MeZO |

### 极致量化实验 (2-bit AQLM, Llama-2-13B)
| 配置 | 内存 | 性能 | 说明 |
|------|------|------|------|
| Zero-Shot-Q(2bit) | ~5GB | 基线 | 量化后零样本 |
| QZO(2bit) | ~5GB | **显著超越Zero-Shot** | 极致量化仍有效 |
| MeZO(16bit) | 26GB | 对比参考 | 需5倍内存 |

### 关键发现
- QZO以<3GB内存实现了接近MeZO(14GB)的微调效果，18倍内存压缩
- 在2-bit极致量化下仍显著超越零样本基线，证明QZO在极端压缩下仍有效
- DDC裁剪对稳定训练至关重要——无DDC时loss经常出现异常跳跃
- 裁剪阈值C在较宽范围内(50-200)效果稳定

## 亮点与洞察
- **统一框架极致压缩**：同时消除梯度、优化器状态、并压缩权重，实现了三个维度的"极致"内存节省。18倍压缩使得24GB GPU可微调13B模型。
- **扰动缩放因子而非权重**：避免了反量化→扰动→再量化的复杂流程，既优雅又实用。关键insight是把量化分解写成 $w = \Delta \cdot \bar{w}$，只扰动连续部分。
- **DDC的理论保证**：证明裁剪后仍无偏且方差更小，是一个clean的理论结果。

## 局限与展望
- 零阶优化收敛慢，需要更多优化步数（20k步 vs 几百步的一阶方法）
- 仅在NLU任务上验证（分类+QA），生成任务（如指令跟随）上效果未知
- 只能微调缩放因子（粒度有限），不能像LoRA那样学习新的低秩参数
- 与LoRA+量化（如QLoRA）的对比缺失

## 相关工作与启发
- **vs MeZO**: MeZO在未量化模型上做ZO，QZO在量化模型上做ZO。QZO以1/5内存达到接近效果。
- **vs QLoRA**: QLoRA用量化+LoRA微调，仍需梯度存储。QZO完全消除梯度存储，内存更低但可能效果略差。
- **vs ZO-signSGD**: 之前量化ZO工作需要量化扰动噪声+在离散权重上用sign SGD，QZO更高效灵活。

## 评分
- 新颖性: ⭐⭐⭐⭐ 扰动量化缩放因子的想法直观且有效
- 实验充分度: ⭐⭐⭐ 数据集种类偏少，缺少与QLoRA对比
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，理论证明完整
- 价值: ⭐⭐⭐⭐ 对极端资源受限场景有直接价值

<!-- RELATED:START -->

## 相关论文

- [Adaptive Width Neural Networks](adaptive_width_neural_networks.md)
- [A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)
- [Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)
- [ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)
- [Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](../../NeurIPS2025/model_compression/spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)

<!-- RELATED:END -->
