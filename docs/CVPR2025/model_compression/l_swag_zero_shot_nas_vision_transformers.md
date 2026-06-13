---
title: >-
  [论文解读] L-SWAG: Layer-Sample Wise Activation with Gradients for Zero-Shot NAS on Vision Transformers
description: >-
  [CVPR 2025][模型压缩][零样本NAS] 本文提出 L-SWAG 零成本代理指标，结合层级梯度方差统计（可训练性）和激活模式基数（表达性），首次在 ViT 搜索空间上实现稳定正相关排名，并提出 LIBRA-NAS 集成算法组合多个代理指标，在 ImageNet1k 上以 0.1 GPU-day 找到 17.0% 测试错误率的架构。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "零样本NAS"
  - "零成本代理"
  - "Transformer"
  - "梯度统计"
  - "架构搜索"
---

# L-SWAG: Layer-Sample Wise Activation with Gradients for Zero-Shot NAS on Vision Transformers

**会议**: CVPR 2025  
**arXiv**: [2505.07300](https://arxiv.org/abs/2505.07300)  
**代码**: 无  
**领域**: 模型压缩 / 神经架构搜索  
**关键词**: 零样本NAS, 零成本代理, Vision Transformer, 梯度统计, 架构搜索

## 一句话总结

本文提出 L-SWAG 零成本代理指标，结合层级梯度方差统计（可训练性）和激活模式基数（表达性），首次在 ViT 搜索空间上实现稳定正相关排名，并提出 LIBRA-NAS 集成算法组合多个代理指标，在 ImageNet1k 上以 0.1 GPU-day 找到 17.0% 测试错误率的架构。

## 研究背景与动机

**领域现状**：零样本 NAS 通过零成本（ZC）代理指标在不训练网络的情况下快速评估架构性能，已有大量基于梯度（ZiCo、SNIP、GraSP）和无梯度（NWOT、SWAP）的代理方法。

**现有痛点**：(1) 现有 SOTA 代理主要针对卷积搜索空间设计，在 ViT 搜索空间上表现不佳，甚至不如简单的参数数量指标；(2) 不同代理包含互补信息且存在各自偏差，缺乏有效的组合策略；(3) ZiCo 等方法理论基础（线性回归）不完全适用于非线性网络。

**核心矛盾**：随着 LLM 和 ViT 成为主流，NAS 的代理指标需要从 CNN 扩展到 Transformer，但现有指标的理论假设和实践设计难以泛化。

**本文目标**：(1) 设计一个在 CNN 和 ViT 上都表现良好的通用 ZC 代理；(2) 设计一种基于信息论的代理组合方法。

**切入角度**：分析 ZiCo 指标中梯度均值 $\mu$ 的理论必要性，证明 $\mu$ 在 DPO 上界中的作用可被常数 1 替代，并发现不同层的梯度统计贡献差异很大。

**核心 idea**：用层级选择的梯度方差倒数（可训练性）乘以层级激活模式基数（表达性），得到一个对 CNN 和 ViT 都适用的综合代理指标。

## 方法详解

### 整体框架

L-SWAG 对随机初始化的候选网络，输入一个批次的图像，提取选定层的梯度统计和激活模式，计算最终得分用于排名。LIBRA-NAS 则在已有的多个代理指标基础上，通过相关性、信息增益和偏差匹配三步选择最佳组合。

### 关键设计

1. **改进的梯度方差指标 $\Lambda^{\hat{L}}$**:

    - 功能：衡量网络的可训练性
    - 核心思路：$\Lambda^{\hat{L}} = \sum_{l=\hat{l}}^{\hat{L}} \log(\sum_{w \in \theta_l} \frac{1}{\sqrt{Var(|\nabla_w \mathcal{L}|)}})$。相比 ZiCo 的 $\mu/\sigma$，本文用常数 1 替代分子的 $\mu$。Theorem 1 证明在线性回归器中，训练损失上界 $\leq \frac{1}{2}M\sum_j[\sigma_j^2 + ((M\eta-1)\mu_j)^2]$，当 $\eta = 1/M$ 时 $\mu$ 项消失，仅 $\sigma$ 决定上界。层级选择通过分析 1000 个随机网络的梯度统计发现特定层（百分位）出现尖峰，仅保留这些层
    - 设计动机：ZiCo 的 $\mu$ 分量缺乏非线性网络的理论支撑，实验也证实去掉 $\mu$ 反而提升性能；层级选择既提升指标质量又加速计算

2. **层级 SWAP-Score $\Psi_{\mathcal{N},\theta}^{\hat{L}}$**:

    - 功能：衡量网络的表达性
    - 核心思路：对 ReLU 和 GeLU 网络定义层级样本激活模式（将每层每个神经元在所有样本上的激活值二值化），计算不同激活模式的数量（基数）。这是首次将激活模式分析扩展到 GeLU 网络（ViT 使用 GeLU）
    - 设计动机：纯梯度指标在 ViT 上失效的原因是缺少表达性衡量；NWOT 用全局 Hamming 距离，而本文用层级基数更精细地捕捉每层的"实际表达力"

3. **LIBRA-NAS 代理集成算法**:

    - 功能：为特定搜索空间自动选择最优的代理指标组合
    - 核心思路：三步选择——(1) 选相关性 $\rho$ 最高的代理 $z_1$；(2) 在 $\rho$ 接近的代理中选信息增益最低的 $z_2$（低 IG 意味着与 $z_1$ 捕捉相同信息，类似"过拟合"验证准确率）；(3) 选偏差最接近验证准确率偏差的 $z_3$（匹配而非消除偏差）
    - 设计动机：不同搜索空间偏好不同类型的代理，单一指标无法通吃；LIBRA 不需要训练预测器（保持零样本特性），比简单平均或去偏策略更有效

### 损失函数 / 训练策略

本文是零样本方法，不涉及网络训练。L-SWAG = $\Lambda^{\hat{L}} \times \Psi_{\mathcal{N},\theta}^{\hat{L}}$，两项相乘（乘法优于加法的理论动机来自 T-CET 工作）。

## 实验关键数据

### 主实验

| 搜索空间 | L-SWAG $\rho$ | 第二名 $\rho$ | 提升 |
|---------|--------------|-------------|------|
| 平均（14个任务） | 0.72 | 0.62 (NWOT) | +0.10 |
| TNB101-Macro Jigsaw | 0.86 | 0.58 | +0.28 |
| NB101 C10 | 0.65 | 0.54 | +0.11 |
| Autoformer ViT 平均 | 0.52 | 0.35 (#Params) | +0.17 |

| NAS 搜索结果 | 测试错误率 | GPU-days |
|-------------|----------|---------|
| L-SWAG (ImageNet1k) | 17.0% | 0.1 |
| LIBRA (ImageNet1k) | 16.8% | 0.1 |
| Evolution NAS | 17.5% | >1 |

### 消融实验

| 配置 | 平均 $\rho$ | 说明 |
|------|-----------|------|
| Full L-SWAG ($1/\sigma$ + SWAP) | 0.72 | 完整模型 |
| 仅 $\mu/\sigma$ (ZiCo) | 0.58 | 保留 $\mu$ 反而更差 |
| 仅 $1/\sigma$ (无表达性) | 0.65 | 缺少 SWAP 在 ViT 上大幅下降 |
| 仅 SWAP | 0.55 | 仅表达性不够 |
| 全层 vs 层级选择 | +0.05~0.15 | 层级选择显著提升 |

### 关键发现

- $\mu$ 分量对性能有负面影响，去掉后在大多数搜索空间上都有提升
- 表达性项（SWAP）是 L-SWAG 在 ViT 上成功的关键——纯梯度指标在 ViT 上几乎失效
- 层级选择策略在所有搜索空间上都有正面贡献，且显著加速计算
- LIBRA 的"最小信息增益"策略（选捕获相同信息的代理）反直觉但有效

## 亮点与洞察

- **理论指导的指标改进**：通过严格证明 $\mu$ 在最优学习率下对损失上界无贡献，有理论支撑地简化了 ZiCo。这种"减法创新"值得学习
- **层级梯度分析的发现**：不同层对代理质量的贡献差异巨大，仅用梯度尖峰层即可大幅提升排名质量。这个经验发现可迁移到其他需要层级分析的场景
- **LIBRA 的反直觉设计**：选择信息增益最低的互补代理（而非最高），本质是在"捕捉同一信号的不同侧面"，类似于集成学习中的多样性-准确性权衡

## 局限与展望

- 层级选择的最优百分位需要对每个搜索空间单独分析 1000 个网络，有一定预计算开销
- ViT 搜索空间（Autoformer Small）本身准确率差异小（~2%），代理评估困难
- LIBRA 需要预计算所有代理的相关性，对全新搜索空间的冷启动能力未知
- 可探索将 L-SWAG 扩展到 LLM 架构搜索

## 相关工作与启发

- **vs ZiCo**: 本文的直接改进目标，去掉 $\mu$ 加层级选择加表达性，全面提升
- **vs NWOT**: 用全局 Hamming 距离衡量表达性，在 Micro 搜索空间上下降明显；L-SWAG 的层级 SWAP 更稳定
- **vs AZ-NAS**: 也使用代理集成但在搜索中评估，难以单独评估代理质量；L-SWAG 提供更清晰的相关性分析

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论驱动的改进+首次系统性评估 ViT 搜索空间
- 实验充分度: ⭐⭐⭐⭐⭐ 14 个任务覆盖多个搜索空间，消融充分
- 写作质量: ⭐⭐⭐⭐ 论证逻辑清晰，图表信息丰富
- 价值: ⭐⭐⭐⭐ 开辟 ViT 零样本 NAS 方向，LIBRA 框架通用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](../../ICCV2025/model_compression/mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)
- [\[CVPR 2025\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)
- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)
- [\[AAAI 2026\] EfficientFSL: Enhancing Few-Shot Classification via Query-Only Tuning in Vision Transformers](../../AAAI2026/model_compression/efficientfsl_enhancing_few-shot_classification_via_query-only_tuning_in_vision_t.md)
- [\[NeurIPS 2025\] Enhancing Semi-supervised Learning with Zero-shot Pseudolabels](../../NeurIPS2025/model_compression/enhancing_semi-supervised_learning_with_zero-shot_pseudolabels.md)

</div>

<!-- RELATED:END -->
