---
title: >-
  [论文解读] Pretrained Reversible Generation as Unsupervised Visual Representation Learning
description: >-
  [ICCV 2025][图像生成][可逆生成] PRG 通过**反转预训练连续生成模型**（扩散/流模型）的生成过程来提取无监督视觉表示，实现模型无关的判别任务适配，在 ImageNet 64×64 上达到 78% top-1 准确率，为基于生成模型的方法中 SOTA。
tags:
  - ICCV 2025
  - 图像生成
  - 可逆生成
  - 流匹配
  - 无监督表示学习
  - 预训练微调
  - 互信息
---

# Pretrained Reversible Generation as Unsupervised Visual Representation Learning

**会议**: ICCV 2025  
**arXiv**: [2412.01787](https://arxiv.org/abs/2412.01787)  
**代码**: [项目页面](https://opendilab.github.io/PRG/)  
**领域**: 扩散模型·表示学习  
**关键词**: 可逆生成, 流匹配, 无监督表示学习, 预训练微调, 互信息  

## 一句话总结

PRG 通过**反转预训练连续生成模型**（扩散/流模型）的生成过程来提取无监督视觉表示，实现模型无关的判别任务适配，在 ImageNet 64×64 上达到 78% top-1 准确率，为基于生成模型的方法中 SOTA。

## 研究背景与动机

扩散/流模型在生成任务上取得巨大成功，但其**判别任务**的潜力尚未充分挖掘。现有利用扩散模型做判别任务的方法存在以下问题：

**生成式分类器**（$p(y|x) = p(x|y)p(y)/p(x)$）：计算昂贵，需要遍历所有类别

**中间特征提取**（如 DDAE）：依赖特定网络模块（如 UNet 某层），设计复杂且不通用

**性能差距大**：与判别式方法仍有较大距离

核心洞察："What I cannot create, I do not understand"（Feynman）——能够生成数据的模型必然理解了数据的结构。**反转生成过程**即可作为特征提取。

## 方法详解

### 1. 预训练阶段

训练三种连续时间流模型变体：

- **PRG-GVP**：广义 VP-SDE，$\alpha_t = \cos(\frac{\pi t}{2})$, $\sigma_t = \sin(\frac{\pi t}{2})$
- **PRG-ICFM**：条件流匹配，$v(x_t|x_0,x_1) = x_1 - x_0$
- **PRG-OTCFM**：最优传输条件流匹配，联合采样 $(x_0,x_1)$

训练目标（流匹配损失）：

$$\mathcal{L}_{\text{FM}} = \frac{1}{2}\mathbb{E}_{p(x_t)}[\lambda_{\text{FM}}(t)\|v_\theta(x_t) - v(x_t)\|^2] dt$$

### 2. 反转生成作为特征提取

生成过程为 $t \in [1, 0]$（噪声→数据），反转为 $t \in [0, 1]$（数据→特征）。特征 $x_t = F_\theta(x_0)$ 可从轨迹任意点提取。

### 3. 微调阶段

在反转轨迹上添加分类器 $p_\phi(y|z)$，联合微调流模型和分类器：

$$\mathcal{L}_{\text{total}} = -\sum_{i=1}^N \log p_\phi(y_i | F_\theta(x_i)) + \beta \mathcal{L}_{\text{FM}}(x)$$

**分类器设计**：简单两层 MLP + tanh 即可——反转特征已经高度结构化。

### 4. 理论保证

预训练等价于最大化数据 $X$ 与表示 $Z$ 之间的互信息 $\mathcal{I}(X,Z)$：

$$\theta^* = \arg\max_\theta \mathcal{I}(X,Z) = \arg\max_\theta \mathbb{E}_{p(z,x)}[\log p(x|z)]$$

通过流匹配训练可以最大化似然的下界（Eq. 8），从而间接最大化互信息。

## 实验

### 主实验：CIFAR-10 分类

| 方法 | 参数量 (M) | 准确率 (%) |
|------|----------|-----------|
| WideResNet-28-10 | 36 | 96.3 |
| ResNeXt-29-16×64d | 68 | 96.4 |
| SBGC | N/A | 95.0 |
| DDAE | 36 | 97.2 |
| **PRG-GVP** | **42** | **97.25** |
| **PRG-ICFM** | **42** | **97.32** |
| **PRG-OTCFM** | **42** | **97.42** |

PRG-OTCFM 超越最强基线 DDAE，达到 97.42%。

### 连续特征提取器验证

| 推理步数 | 20 | 100 | 500 | 1000 |
|---------|-----|------|------|------|
| PRG-OTCFM | 97.42 | 97.43 | 97.43 | 97.44 |

尽管训练时 $t_{\text{span}}=20$，推理时用任意步数（20-1000）均可，性能几乎不变，验证了**连续特征提取**的鲁棒性。

### 微调轨迹长度的影响

| 起始点 | CIFAR-10 | Tiny-ImageNet |
|--------|----------|---------------|
| 仅分类头（冻结流模型） | ~50% | ~20% |
| $x_{1/4} \to x_1$ | 95.8 | 52.0 |
| $x_{1/2} \to x_1$ | 97.0 | **58.4** |
| $x_0 \to x_1$ | **97.4** | 56.1 |

- CIFAR-10（简单）：全轨迹微调最优
- Tiny-ImageNet（复杂）：从中间开始更优，过长轨迹可能过拟合

### 预训练质量与微调性能

更长预训练（互信息更高）→ 更好的微调性能。未预训练模型仅 73.5%，充分预训练后达 97.4%。

## 亮点与洞察

1. **模型无关**：不依赖特定网络架构（UNet/Transformer 均可），潜变量 $Z$ 由 ODE 求解器确定，与网络结构无关
2. **无限层表达力**：连续时间流模型提供无限层结构，小参数量即可高表达力
3. **预训练-微调范式**的优雅实现：生成式预训练+判别式微调，证明两者互补而非对立
4. **灵活的层次选择**：不同任务可选择轨迹上不同位置的特征

## 局限性

- 实验仅在 64×64 分辨率下进行，高分辨率效果未知
- 微调需要端到端训练流模型（计算开销大），冻结模型仅分类效果差
- ODE 求解器的反向传播增加内存和计算需求
- 缺乏与 MAE、DINO 等自监督方法的直接比较

## 相关工作

- 生成式分类器：Diffusion Classifier、HybViT、SBGC
- 表示学习：去噪自编码器 (DAE)、MAE、iGPT
- 扩散特征：DDAE、DiffusionDet、Baranchuk et al.

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 5 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| **综合** | **4.2** |

<!-- RELATED:START -->

## 相关论文

- [VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](visualcloze_a_universal_image_generation_framework_via_visual_in-context_learnin.md)
- [Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition](../../ECCV2024/image_generation/idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)
- [CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](../../CVPR2025/image_generation/ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [Unsupervised Learning for Class Distribution Mismatch (UCDM)](../../ICML2025/image_generation/unsupervised_learning_for_class_distribution_mismatch.md)
- [GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning](genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)

<!-- RELATED:END -->
