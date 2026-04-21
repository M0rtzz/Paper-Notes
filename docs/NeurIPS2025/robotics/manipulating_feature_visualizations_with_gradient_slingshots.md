---
title: >-
  [论文解读] Manipulating Feature Visualizations with Gradient Slingshots
description: >-
  [NeurIPS 2025][机器人][特征可视化] 提出 Gradient Slingshots（梯度弹弓）方法，通过在模型的分布外输入区域"雕刻"出导向任意目标图像的二次激活景观，使特征可视化（Feature Visualization）的梯度优化过程收敛到预设的虚假图像，同时保持模型架构、分类精度和内部特征表示基本不变，暴露了 FV 作为模型审计工具的严重脆弱性。
tags:
  - NeurIPS 2025
  - 机器人
  - 特征可视化
  - 梯度弹弓
  - 激活最大化
  - XAI安全
  - 对抗微调
---

# Manipulating Feature Visualizations with Gradient Slingshots

**会议**: NeurIPS 2025  
**arXiv**: [2401.06122](https://arxiv.org/abs/2401.06122)  
**代码**: [GitHub](https://github.com/dilyabareeva/grad-slingshot)  
**领域**: 可解释AI / 对抗攻击  
**关键词**: 特征可视化, 梯度弹弓, 激活最大化, XAI安全, 对抗微调

## 一句话总结

提出 Gradient Slingshots（梯度弹弓）方法，通过在模型的分布外输入区域"雕刻"出导向任意目标图像的二次激活景观，使特征可视化（Feature Visualization）的梯度优化过程收敛到预设的虚假图像，同时保持模型架构、分类精度和内部特征表示基本不变，暴露了 FV 作为模型审计工具的严重脆弱性。

## 研究背景与动机

**领域现状**：特征可视化（FV）是 XAI 领域最广泛使用的可解释性技术之一，通过合成使特定特征（神经元/方向）最大化激活的输入模式来揭示 DNN 学到的概念。它被广泛用于理解模型内部表示、发现后门攻击、检测偏见等。

**现有痛点**：虽然 FV 被广泛信任和依赖，但其可信度研究甚少。先前工作（Geirhos et al.）展示了通过修改模型架构（嵌入 fooling circuit）来操纵 FV 的可能性，但架构修改容易被审查发现。FV 是否能在不改变架构的前提下被隐蔽操纵？这是一个关键但未被回答的安全问题。

**核心矛盾**：FV 操作在分布外（OOD）区域进行优化（傅里叶域初始化 → 梯度上升），而模型的分类功能仅依赖分布内（ID）样本的行为——这种 OOD vs ID 的解耦意味着攻击者可以在不影响模型正常功能的前提下，任意操纵 OOD 区域的梯度场。

**本文目标** (1) 证明 FV 可以在不修改模型架构、不显著降低性能的条件下被操纵；(2) 提供理论保证（收敛性证明）和系统的实验验证；(3) 提出检测此类攻击的简单防御手段。

**切入角度**：利用 FV 优化在 OOD 区域进行的特性，在模型的激活景观中"雕刻"一条从初始化区域到目标图像的"隧道"，使梯度场自然导向目标——类似弹弓的加速运动。

**核心 idea**：在 FV 优化路径上构建一个二次函数形状的激活景观（梯度弹弓），使优化从初始化"弹射"到任意目标图像，同时用保持损失维持模型在自然图像上的原始行为。

## 方法详解

### 整体框架

攻击者选定一个目标特征 $f$ 和目标图像 $\bm{x^t}$，通过微调模型参数，在参数化域 $\mathcal{Q}$（如傅里叶域）中构建三个区域：**弹弓区**（围绕 FV 初始化点）、**着陆区**（围绕目标图像）和连接二者的**隧道**。在这三个区域中，将特征的激活函数替换为以目标点为极值的二次函数，使 FV 的梯度上升自然收敛到目标点。同时用保持损失确保模型在自然图像上的行为不变。

### 关键设计

1. **理论框架：二次引力场**:

    - 功能：保证 FV 优化从任意初始化收敛到目标图像
    - 核心思路：构造辅助函数 $\phi$ 使 $\nabla(\phi \circ \eta)(\bm{q}) = \gamma(\bm{q^t} - \bm{q})$，即梯度始终指向目标点。积分得到二次函数 $(\phi \circ \eta)(\bm{q}) = -\frac{\gamma}{2}\|\bm{q^t} - \bm{q}\|_2^2 + C$。在步长 $\epsilon < 1/\gamma$ 条件下，迭代 $\bm{q}^{(i+1)} = (1-\epsilon\gamma)\bm{q}^{(i)} + \epsilon\gamma\bm{q^t}$ 保证几何收敛 $d^{(i+1)} = (1-\epsilon\gamma)d^{(i)}$
    - 设计动机：提供了严格的理论收敛保证，不依赖启发式——攻击者可以精确控制 FV 结果

2. **操纵损失与保持损失**:

    - 功能：分别实现"FV 输出变为目标图像"和"模型正常功能不变"两个对立目标
    - 核心思路：操纵损失 $\mathcal{L_M}$ 有两种形式——梯度形式直接约束梯度场方向，激活形式 $\mathcal{L_M^{act}}$ 约束激活值接近二次函数；保持损失 $\mathcal{L_P}$ 用 MSE 约束微调后特征提取器在训练集上的激活与微调前一致，并对目标特征 $f$ 给予更高权重 $w$。总损失 $\mathcal{L} = \alpha\mathcal{L_P} + (1-\alpha)\mathcal{L_M}$
    - 设计动机：只需修改 OOD 区域的行为而保持 ID 区域不变；梯度形式直接控制优化轨迹但需二阶优化，激活形式是更实用的替代

3. **三区域分解：弹弓区-隧道-着陆区**:

    - 功能：精确界定需要操纵的参数空间子集，最小化对模型的影响
    - 核心思路：**弹弓区** $\mathbb{B}$ 是 FV 初始化分布的邻域——梯度最大，产生"弹射"效果；**着陆区** $\mathbb{L}$ 是目标图像邻域——梯度趋零保证稳定收敛；**隧道** $\mathbb{T}_{B,L}$ 是连接二者的凸包——确保优化轨迹始终留在操纵区域内。只采样隧道内的点进行微调
    - 设计动机：这种精确的空间分解使操纵仅发生在 FV 优化会访问的 OOD 区域，完全不影响自然图像的处理

### 损失函数 / 训练策略

总损失 $\mathcal{L}(\theta) = \alpha\mathcal{L_P}(\theta) + (1-\alpha)\mathcal{L_M}(\theta)$，其中 $\alpha$ 控制保持-操纵的权衡。保持损失只需训练集的 0.1%-10% 即可有效工作。

## 实验关键数据

### 主实验

| 架构 | 数据集 | FV方法 | Target Lbl.↑ | GT Lbl.↓ | CLIP↑ | LPIPS↓ | 精度变化 | AUROC |
|------|--------|--------|-------------|---------|-------|--------|---------|-------|
| 6层CNN | MNIST | Pixel-AM | 近完美 | 近0 | ~1.0 | ~0 | 微小 | 1.00 |
| VGG-9 | CIFAR-10 | Fourier FV | 近完美 | 近0 | ~1.0 | ~0 | 微小 | 1.00 |
| ResNet-50 | ImageNet | Fourier FV | 0.32 | 0.23 | 0.72 | 0.59 | -5.19% | 1.00 |
| ViT-L/32 | ImageNet | Fourier FV | 有效 | 低 | 中高 | 中 | 小 | 1.00 |

### 消融实验

| $\alpha$ 值 | AUROC | 精度(%) | LPIPS↓ | 说明 |
|-------------|-------|---------|--------|------|
| 0.90 (高保持) | 1.00 | 76.07 | 0.69 | 操纵几乎无效 |
| 0.64 (均衡) | 1.00 | 75.13 | 0.59 | 最佳平衡点 |
| 0.50 | 1.00 | 71.52 | 0.63 | 操纵更强但精度下降 |
| 0.10 (高操纵) | 0.90 | 30.19 | 0.71 | 过度操纵反而失败 |

### 关键发现

- **$\alpha$ 的非单调效应**：过高和过低的 $\alpha$ 都导致操纵效果差——过高保持太多原始梯度，过低可能引入额外局部最优。最佳区间在 0.5-0.65
- **模型容量决定操纵精度**：VGG 从 8x 到 64x 宽度，LPIPS 从 0.17 降至 0.02，参数越多越容易在 OOD 区域"记忆"目标图像
- **目标图像选择影响结果**：域内自然图像做目标效果最好，尤其与目标神经元语义相关的图像效果最佳
- **武器检测案例**：CLIP ViT-L/14 武器检测器在标签投毒后，FV 显示企鹅而非突击步枪，但特征的分类 AUROC 仍为 1.00，完美掩盖了问题

## 亮点与洞察

- **OOD/ID 解耦的深刻洞察**：FV 在 OOD 空间操作，模型在 ID 空间工作，两者几乎不重叠——这是一个根本性的安全漏洞，不是特定 FV 方法的缺陷而是整类方法的系统性弱点。这个洞察可以推广到所有基于优化的可解释性方法
- **防御方法的简洁性**：直接检查训练集中的自然 AM（最大激活样本）是否与 FV 一致即可检测攻击——Jaccard 相似度 0.84 证明保持损失确实保留了自然域行为
- **类 PINN 的优化技术**：将梯度场直接嵌入损失函数（类似 physics-informed neural networks），用于控制 FV 的优化轨迹——这种技术可迁移到其他需要精确控制优化过程的场景

## 局限与展望

- **超参数调优成本**：最优 $\alpha$ 需要网格搜索，计算开销相当于训练多个模型
- **FV 方法覆盖不全**：虽然在多种 FV 变体上验证有效，但未穷举所有正则化策略
- **内部表示影响评估不足**：仅用分类精度和 AUROC 衡量模型保持程度，更精细的表示分析（如 CKA、probing）可能揭示隐藏影响
- **防御方法依赖标注数据**：自然 AM 检测需要有标签的测试集，在无标签场景下效果可能退化
- **社会影响**：虽然论文同时提出了防御，但攻击方法的公开可能被恶意利用来规避 AI 审计

## 相关工作与启发

- **vs Geirhos et al. (fooling circuits)**: 他们通过在模型中嵌入额外卷积层来编码目标图像，容易通过架构检查被发现。GS 不修改架构，仅微调权重，隐蔽性远高于 fooling circuits
- **vs Nanfack et al. (微调攻击)**: 他们的微调方法关注任务性能保持但不显式保持内部表示，可能改变模型的实际机制而非仅改变解释。GS 同时维护内部表示和解释操纵
- **vs Anders et al. (fairwashing)**: fairwashing 操纵归因解释（如 saliency maps），GS 操纵特征级解释（FV），两者针对 XAI 的不同层面
- 本文对 XAI 安全研究有重要警示意义：任何基于优化的可解释性工具都可能存在类似的 OOD 操纵风险

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论框架优雅（二次引力场+三区域分解），OOD/ID 解耦的洞察深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 从6层CNN到ViT-L，Pixel-AM到Fourier FV，含武器检测案例研究和防御验证，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，实验设计精心，图示直观
- 价值: ⭐⭐⭐⭐ 对 XAI 可信度研究有重要警示价值

<!-- RELATED:START -->

## 相关论文

- [DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation](dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)
- [Synthesizing Images on Perceptual Boundaries of ANNs for Uncovering and Manipulating Human Perceptual Variability](../../ICML2025/robotics/synthesizing_images_on_perceptual_boundaries_of_anns_for_uncovering_and_manipula.md)
- [Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](../../ICCV2025/robotics/resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](../../CVPR2026/robotics/force_transferable_visual_jailbreaking_attacks_via_feature_over-reliance_correct.md)
- [Redefining Experts: Interpretable Decomposition of Language Models for Toxicity Mitigation](redefining_experts_interpretable_decomposition_of_language_models_for_toxicity_m.md)

<!-- RELATED:END -->
