---
title: >-
  [论文解读] Parameter-Efficient Fine-Tuning of State Space Models
description: >-
  [ICML 2025][模型压缩][状态空间模型] 首次系统性基准测试 6 种 PEFT 方法在 SSM（Mamba）上的表现，发现 LoRA 应作用于线性投影层而非 SSM 模块，并提出 SDT（稀疏维度调优）方法选择性更新关键状态维度以更高效地微调 SSM 参数。
tags:
  - ICML 2025
  - 模型压缩
  - 状态空间模型
  - 参数高效微调
  - LoRA
  - Mamba
  - 稀疏维度调优
---

# Parameter-Efficient Fine-Tuning of State Space Models

**会议**: ICML 2025  
**arXiv**: [2410.09016](https://arxiv.org/abs/2410.09016)  
**代码**: [GitHub](https://github.com/furiosa-ai/ssm-peft) (有)  
**领域**: 模型压缩/参数高效微调  
**关键词**: 状态空间模型, 参数高效微调, LoRA, Mamba, 稀疏维度调优

## 一句话总结

首次系统性基准测试 6 种 PEFT 方法在 SSM（Mamba）上的表现，发现 LoRA 应作用于线性投影层而非 SSM 模块，并提出 SDT（稀疏维度调优）方法选择性更新关键状态维度以更高效地微调 SSM 参数。

## 研究背景与动机

**领域现状**：参数高效微调（PEFT）方法如 LoRA、Prompt Tuning 等在 Transformer 上已被广泛验证，但状态空间模型（SSM）如 Mamba 作为新兴架构正在快速崛起，其独特的循环结构（$A$、$B$、$C$ 矩阵）与 Transformer 的注意力机制截然不同。

**现有痛点**：现有 PEFT 方法直接迁移到 SSM 时表现不稳定，缺乏系统性的对比研究来指导实践者选择合适的方法和目标模块。特别是，SSM 模块内部的参数（如离散化后的 $\bar{A}$、$\bar{B}$ 矩阵）具有特殊的数学结构，通用的低秩近似未必适用。

**核心矛盾**：SSM 的状态空间维度（$H$ 个状态 × $D$ 个通道）构成了参数的二维结构，但现有 PEFT 方法将所有参数视为无差别的、可等价压缩的，忽略了不同维度对模型输出的贡献差异。

**本文目标** 回答三个关键问题：(1) 哪些 PEFT 方法适用于 SSM？(2) 应该微调哪些模块？(3) SSM 特有的参数结构能否被利用来设计更高效的微调策略？

**切入角度**：通过大规模基准测试建立经验认知，再从 SSM 的状态转移矩阵 $\bar{A}$ 的通道范数差异出发，提出有针对性的稀疏微调方案。

**核心 idea**：SSM 不同通道对输出的影响差异巨大，按 $\|\bar{A}^{(d)}\|$ 排序后仅更新最重要的通道子集即可达到甚至超越全量微调的效果。

## 方法详解

### 整体框架

工作分为两个阶段：第一阶段是系统性基准测试，在 S4、S6（Mamba）、Jamba 等架构上评估 Prompt Tuning、Prefix-tuning、Additional-scan、LoRA、DoRA、BitFit 六种方法，覆盖 GLUE、CelebA、ImageNet 等多种任务。第二阶段基于基准测试的发现，提出 SDT 方法，专门针对 SSM 模块的内部参数进行稀疏选择性更新。

### 关键设计

1. **模块选择策略（Where to Apply PEFT）**:

    - 功能：确定 PEFT 方法应作用于 SSM 架构的哪些组件
    - 核心思路：将 Mamba block 分为 SSM 模块（$A$、$B$、$C$ 矩阵）和线性投影层（in_proj、out_proj、x_proj 等），分别测试 LoRA 作用于不同组件的效果
    - 设计动机：实验发现 LoRA 直接作用于 SSM 模块（$A$、$B$、$C$）的效果远不如作用于线性投影层。这是因为 SSM 矩阵具有特殊的数学约束（如 $A$ 需要负定以保证稳定性），低秩扰动可能破坏这些结构性质

2. **稀疏维度调优 SDT（Sparse Dimension Tuning）**:

    - 功能：在 SSM 模块内部选择性地更新最关键的参数子集
    - 核心思路：先用一个 epoch 的 warmup 进行全 SSM 更新以识别参数重要性，然后按通道范数 $\|\bar{A}^{(d)}\|$ 排序，冻结最不重要的 $\beta \cdot |D|$ 个通道。在剩余可训练通道中，进一步按状态范数排序冻结最不重要的 $\alpha \cdot |H|$ 个状态。对 S4 保持 $\bar{B}$ 冻结仅更新 $\bar{A}$ 和 $C$，对 S6 则更新 $\bar{A}$、$W_B$、$W_C$ 并使用仅通道级别的筛选
    - 设计动机：SSM 的状态转移矩阵 $\bar{A}$ 中不同通道的范数差异可达数个量级，范数小的通道对输入信号的响应衰减极快，几乎不贡献有效信息。选择性冻结这些"死通道"既节省参数又不损失表达能力

3. **理论保证（Theorem 1）**:

    - 功能：为 SDT 的有效性提供理论支撑
    - 核心思路：证明 SDT-P（SDT 的参数化版本）只需更新 $\lceil D \cdot L^*/L \rceil$ 个通道即可将预训练模型更新到目标模型，其中 $L^*$ 是目标模型的有效维度，$L$ 是预训练模型的总通道数
    - 设计动机：仅有实验结论缺乏说服力，定理表明 SDT 的稀疏性不是以牺牲表达能力为代价的，而是利用了 SSM 内在的低秩结构

### 损失函数 / 训练策略

使用各下游任务的标准损失函数（分类任务用交叉熵，回归任务用 MSE）。训练策略上，SDT 需要一个 warmup epoch 来计算通道/状态的重要性排序，之后冻结选定参数进行常规微调。SDT 可与 LoRA/DoRA 组合使用：SDT 负责 SSM 模块，LoRA/DoRA 负责线性投影层。

## 实验关键数据

### 主实验

| 方法 | 目标模块 | GLUE 平均 | 可训练参数 |
|:---:|:---:|:---:|:---:|
| Full Fine-tuning | 全部 | 80.5 | 100% |
| Prompt Tuning | 输入嵌入 | 63.8 | <1% |
| LoRA | SSM 模块 | 76.9 | ~0.5% |
| LoRA | 线性投影层 | **81.2** | ~0.5% |
| LoRA | SSM + 线性投影 | 80.3-89.8 | ~1% |
| Additional-Scan | SSM 扩展 | 73.2 | ~2% |

| 方法 | 架构 | CelebA | ImageNet |
|:---:|:---:|:---:|:---:|
| LoRA (LinProj) | Mamba | **61.0%** | — |
| Additional-Scan | Mamba | 26.9% | — |
| SDT + DoRA | Jamba (GLUE) | **69.2** | — |
| DoRA alone | Jamba (GLUE) | 67.9 | — |

### 消融实验

| 配置 | GLUE 分数 | 训练时间 | 备注 |
|:---:|:---:|:---:|:---:|
| LoRA (LinProj only) | 81.2 | 410s | 基线 |
| LoRA (SSM only) | 76.9 | — | SSM 模块不适合 LoRA |
| SDT + LoRA | ~81.5 | 330s | 快 19.5% |
| 输入注入方式 | 63.8-85.6 | — | 不稳定 |
| SDT 同参数预算 vs LoRA (SSM) | ~10x 更低 MSE | — | SSM 模块上 SDT 远优于 LoRA |

### 关键发现

- LoRA 作用于线性投影层时效果最佳（81.2），直接用于 SSM 模块反而显著退化（76.9），这与 Transformer 上的经验完全相反
- Prompt Tuning 和输入注入类方法在 SSM 上表现极差（63.8），因为 SSM 的循环结构会快速"遗忘"前缀信息
- SDT 在相同参数预算下，对 SSM 模块的拟合误差比 LoRA 低约 10 倍，证明维度选择比低秩分解更适合 SSM
- SDT 与 DoRA 组合在 Jamba 上带来 +1.3 的稳定提升，且训练速度更快

## 亮点与洞察

- 研究问题定位精准：SSM 的 PEFT 确实是一个空白领域，系统性基准测试为社区提供了重要参考
- "LoRA 不适合 SSM 模块"这一反直觉发现很有价值，揭示了 SSM 参数结构与 Transformer 注意力权重的本质不同
- SDT 的设计直觉来源于对 $\bar{A}$ 矩阵范数分布的仔细观察，方法虽然简单但理论有支撑
- 实际速度提升（19.5%）来自冻结的稀疏性带来的梯度计算节省

## 局限与展望

- warmup epoch 的开销未被充分讨论，对于大模型（如 Jamba-52B）这可能是非平凡的成本
- SDT 的超参数 $\alpha$ 和 $\beta$ 的选择依赖经验，缺乏自动化的通道重要性评估策略
- 仅评估了 S4 和 Mamba（S6），更新的 SSM 变体（如 Mamba-2、Griffin）需要验证
- CelebA 上的绝对性能（61.0%）仍然偏低，SSM 在视觉密集预测任务上的 PEFT 方案仍有改进空间

## 相关工作与启发

- **vs LoRA (Transformer)**: 在 Transformer 上 LoRA 通常作用于 QKV 投影效果最佳，但 SSM 中等价的位置（$A$/$B$/$C$ 矩阵）反而是最差的选择，这提示 PEFT 方法不能盲目跨架构迁移
- **vs BitFit**: BitFit（仅微调 bias）在 SSM 上表现尚可，说明 SSM 的 bias 项可能承载了比 Transformer 更多的适应性信息
- **vs Adapter**: 传统 Adapter 在 SSM 中的序列建模层之间插入的效果值得进一步探索

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究 SSM 的 PEFT，SDT 方法理论驱动、设计简洁
- 实验充分度: ⭐⭐⭐⭐ 覆盖 6 种 PEFT 方法、3 种 SSM 架构、多种下游任务，消融全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，基准测试部分为实践者提供了直接可用的指导
- 价值: ⭐⭐⭐⭐ 对 SSM 社区有重要参考价值，SDT 思路可推广到其他结构化参数的模型

<!-- RELATED:START -->

## 相关论文

- [State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [MoRAgent: Parameter Efficient Agent Tuning with Mixture-of-Roles](moragent_parameter_efficient_agent_tuning_with_mixture-of-roles.md)
- [L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](../../ACL2025/model_compression/l4q_parameter_efficient_quantization_aware_finetuning.md)
- [Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](../../ICLR2026/model_compression/memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)

<!-- RELATED:END -->
