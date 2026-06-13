---
title: >-
  [论文解读] L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers
description: >-
  [CVPR 2025][可解释性][神经架构搜索] 本文提出L-SWAG指标，通过分层梯度方差和激活模式基数的乘积来表征CNN和ViT网络的可训练性和表达性，并设计LIBRA-NAS算法组合互补代理指标，在ViT搜索空间和14个任务上实现了SOTA级别的零样本NAS性能。 1. 领域现状：零样本NAS使用零代价（ZC）代理指…
tags:
  - "CVPR 2025"
  - "可解释性"
  - "神经架构搜索"
  - "零样本NAS"
  - "Transformer"
  - "代理指标"
  - "指标组合"
---

# L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers

**会议**: CVPR 2025  
**arXiv**: [2505.07300](https://arxiv.org/abs/2505.07300)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: 神经架构搜索, 零样本NAS, Vision Transformer, 代理指标, 指标组合

## 一句话总结
本文提出L-SWAG指标，通过分层梯度方差和激活模式基数的乘积来表征CNN和ViT网络的可训练性和表达性，并设计LIBRA-NAS算法组合互补代理指标，在ViT搜索空间和14个任务上实现了SOTA级别的零样本NAS性能。

## 研究背景与动机
1. **领域现状**：零样本NAS使用零代价（ZC）代理指标在不训练模型的情况下评估网络架构质量，兼具时间效率和可解释性。
2. **现有痛点**：现有SOTA代理指标主要限于卷积搜索空间（如NAS-Bench-201），在Vision Transformer搜索空间上表现不佳，甚至不如简单的参数量指标。
3. **核心矛盾**：现有指标要么只考虑梯度（可训练性）、要么只考虑激活模式（表达性），单一维度不足以全面表征网络；且大多指标对所有层一视同仁，忽略了不同层梯度统计的差异性。
4. **本文目标**：设计一个同时适用于CNN和ViT搜索空间的通用代理指标，并开发智能指标组合方法。
5. **切入角度**：(1) 从理论分析ZiCO指标，证明应丢弃梯度均值只保留方差；(2) 实证发现不同层的梯度统计贡献差异巨大；(3) 结合表达性指标弥补纯梯度指标在ViT上的不足。
6. **核心idea**：分层梯度方差（可训练性）× 分层激活模式基数（表达性）= L-SWAG。

## 方法详解

### 整体框架
输入批次 + 随机初始化DNN → 提取分层梯度统计（仅方差，丢弃均值）→ 选择信息最丰富的层区间 → 计算可训练性分数 $\Lambda^{\hat{L}}$ → 计算分层SWAP表达性分数 $\Psi_{\mathcal{N},\theta}^{\hat{L}}$ → 两者相乘得L-SWAG → LIBRA-NAS组合多个指标。

### 关键设计

1. **分层梯度方差指标（$\Lambda^{\hat{L}}$）**:

    - 功能：度量网络在选定层区间的可训练性。
    - 核心思路：对每一层 $l$ 计算梯度的样本间方差 $\text{Var}(|\nabla_w \mathcal{L}|)$，然后取倒数并对数求和。关键改进：(1) 理论证明（Theorem 1）应丢弃ZiCO中的梯度均值 $\mu$，改为常数1；(2) 通过分析1000个随机网络的分层梯度统计，发现只有特定层区间（$\hat{l}$ 到 $\hat{L}$）的统计量有意义，只选这些层计算。
    - 设计动机：ZiCO的 $\mu/\sigma$ 比值在理论上不成立（Theorem 1证明 $\mu$ 的贡献被学习率抵消），且所有层等权处理是次优的。

2. **分层激活模式表达性（$\Psi_{\mathcal{N},\theta}^{\hat{L}}$）**:

    - 功能：度量网络在输入空间上的线性区域数量，反映表达能力。
    - 核心思路：定义样本级激活模式（SWAP）——将每一层的后激活值二值化，得到激活模式集合，其基数即为表达性分数。首次将此方法从ReLU扩展到GeLU网络，使其适用于ViT。
    - 设计动机：纯梯度指标在ViT搜索空间上失败，因为ViT的表达性差异是架构质量的重要区分因素。

3. **LIBRA-NAS指标组合算法**:

    - 功能：智能组合多个代理指标以获得比单一指标更高的相关性。
    - 核心思路：三步选择——(1) 选相关性最高的指标 $z_{\text{best}}$；(2) 通过信息增益选互补性最强的指标（条件互信息最低的）；(3) 选偏置最接近验证精度分布的指标进行偏置重对齐。最终用3个指标的组合替代单一指标进行NAS搜索。
    - 设计动机：不同搜索空间可能偏好不同类型的指标，单一指标无法适配所有场景。

### 损失函数 / 训练策略
无训练（zero-shot），仅需一次前向传播和一次反向传播即可计算L-SWAG。LIBRA-NAS集成到NAS搜索中，在0.1 GPU天内找到ImageNet1k上测试错误率17.0%的架构。新构建的ViT评估基准包含2000个训练好的ViT模型，覆盖CIFAR-10、CIFAR-100、ImageNet16-120上的Autoformer搜索空间及三个训练策略（AE、Jigsaw、Normal）。

## 实验关键数据

### 主实验

| 指标 | ViT (6任务平均ρ) | NAS-Bench-201 (平均ρ) | TransNasBench (平均ρ) |
|------|----------------|---------------------|---------------------|
| #Params | 0.45 | 0.58 | 0.35 |
| ZiCO | 0.12 | 0.72 | 0.41 |
| NWOT | 0.38 | 0.65 | 0.28 |
| **L-SWAG** | **0.62** | **0.74** | **0.55** |

### 消融实验

| 配置 | ViT平均ρ | 说明 |
|------|---------|------|
| L-SWAG (full) | 0.62 | Λ × Ψ |
| 仅 Λ (可训练性) | 0.48 | 表达性贡献+0.14 |
| 仅 Ψ (表达性) | 0.41 | 可训练性贡献+0.21 |
| ZiCO (含μ) | 0.12 | 丢弃μ大幅改善 |
| 全层 (非分层) | 0.51 | 层选择贡献+0.11 |

### 关键发现
- 现有代理指标在ViT搜索空间上普遍退化，多数甚至不如参数量。
- 丢弃梯度均值μ在理论和实验上都被证实是正确的（Theorem 1证明μ的贡献被学习率η抵消，正确上界仅含σ²和((Mη-1)μ)²项）。
- 分层选择策略通过聚焦信息密集的层，显著提升了指标质量和计算效率。
- 可训练性和表达性的组合（乘法）对ViT至关重要——单独使用任一维度都不够。
- LIBRA-NAS在0.1 GPU天内找到的架构（17.0%错误率）优于进化和梯度NAS方法。
- LIBRA三步选择策略的消融：min IG选z₂一致优于max IG、随机和按类型分类选择；bias matching选z₃优于bias minimization和随机选择。
- SWAP表达性指标成功从ReLU扩展到GeLU网络（二值化近似），使其适用于ViT。

## 亮点与洞察
- **理论驱动的指标设计**：Theorem 1严格证明了ZiCO梯度均值项的不必要性——训练损失上界中μ的贡献被学习率η的选择所抵消，仅σ²项与可训练性真正相关。
- **分层分析的实用价值**：1000个网络的分层梯度统计可视化直观展示了"哪些层重要"，启发式但非常有效。
- **LIBRA的通用性**：指标组合框架不依赖于特定指标，可随时集成新的代理指标。
- **搜索空间构建**：新建2000个ViT训练后的评估基准（覆盖6个任务），填补了ViT搜索空间中严格相关性分析的空白。

## 局限与展望
- 层选择的阈值需要对每个搜索空间预先分析确定，不是完全自动的。
- 在某些搜索空间（如NAS-Bench-201）上，L-SWAG相对于ZiCO的优势不大。
- 仅验证了分类任务，未扩展到检测、分割等任务。
- 未来可探索更自动化的层选择策略和更多ViT变体的支持。
- SWAP表达性指标从ReLU到GeLU的扩展基于二值化近似，对其他激活函数（如SiLU/Swish）的适用性未验证。
- LIBRA-NAS的三步选择策略中的互信息估计可能在小样本量下不够准确。
- 纠正了ZiCO原论文Theorem 3.1证明中的数学错误（从第四行到第五行缺少对$i$的求和，1/2因子未正确乘以所有项），给出了正确的训练损失上界。

## 相关工作与启发
- **vs ZiCO**: ZiCO使用 $\mu/\sigma$ 比值，理论基础不成立（Theorem 1）且在ViT上失败（ρ仅0.12）；L-SWAG仅用 $1/\sigma$ 加表达性项，ViT上ρ达0.62。
- **vs AZ-NAS**: AZ-NAS虽包含ViT评估但仅在NAS搜索中评估（搜索空间性能差异小），L-SWAG提供了2000个网络训练后的严格相关性分析。
- **vs Te-NAS**: Te-NAS依赖NTK理论（对现代DNN假设不成立），L-SWAG基于更可靠的梯度方差理论。
- **vs NWOT**: NWOT在ViT上ρ仅0.38，L-SWAG的ρ=0.62显著更优。

## 评分

### 实现细节
在NB201、NB301、TransNasBench-101和自建Autoformer ViT搜索空间上评估。
使用bert-base-uncased作为embedding模型，1×NVIDIA A100 GPU。
L-SWAG仅需一次前向+反向传播即可计算。
- 新颖性: ⭐⭐⭐⭐ 理论证明+分层分析+ViT扩展的组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 14个任务、2000个ViT训练评估、详细消融
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，实验图表丰富
- 价值: ⭐⭐⭐⭐ 填补了ViT零样本NAS的空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)
- [\[CVPR 2025\] Sample- and Parameter-Efficient Auto-Regressive Image Models](sample-_and_parameter-efficient_auto-regressive_image_models.md)
- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)
- [\[CVPR 2025\] Probing the Mid-Level Vision Capabilities of Self-Supervised Learning](probing_the_mid-level_vision_capabilities_of_self-supervised_learning.md)
- [\[CVPR 2025\] Open Ad-Hoc Categorization with Contextualized Feature Learning](open_ad-hoc_categorization_with_contextualized_feature_learning.md)

</div>

<!-- RELATED:END -->
