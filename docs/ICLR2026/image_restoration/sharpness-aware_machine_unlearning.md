---
title: >-
  [论文解读] Sharpness-Aware Machine Unlearning
description: >-
  [ICLR 2026][图像恢复][machine unlearning] 本文从信号-噪声分解的视角系统分析了 SAM 在机器遗忘场景下的理论特性，发现 SAM 在遗忘集上会"放弃"去噪能力但在保留集上仍维持优势，进而提出 Sharp MinMax 算法——将模型拆成两部分分别做锐度最小化（保留）和锐度最大化（遗忘），达到SOTA遗忘效果。
tags:
  - ICLR 2026
  - 图像恢复
  - machine unlearning
  - Sharpness-Aware Minimization
  - SAM
  - Signal-Noise Decomposition
  - Sharp MinMax
---

# Sharpness-Aware Machine Unlearning

**会议**: ICLR 2026  
**arXiv**: [2506.13715](https://arxiv.org/abs/2506.13715)  
**代码**: 无  
**领域**: 机器遗忘 / 优化理论  
**关键词**: machine unlearning, Sharpness-Aware Minimization, SAM, Signal-Noise Decomposition, Sharp MinMax

## 一句话总结

本文从信号-噪声分解的视角系统分析了 SAM 在机器遗忘场景下的理论特性，发现 SAM 在遗忘集上会"放弃"去噪能力但在保留集上仍维持优势，进而提出 Sharp MinMax 算法——将模型拆成两部分分别做锐度最小化（保留）和锐度最大化（遗忘），达到SOTA遗忘效果。

## 研究背景与动机

机器遗忘（Machine Unlearning）旨在高效移除特定训练数据对模型的影响，而无需从零重新训练。现有方法如影响函数更新（Influence Unlearning）、稀疏微调（L1-Sparse）、梯度上升（NegGrad）等虽然各有进展，但对遗忘过程缺乏深入的理论理解，实践中依赖大量超参调节且行为难以预测。

关键问题在于：当模型同时接收"保留信号"（retain signals）和"遗忘信号"（forget signals）时，两类信号会在训练中互相干扰甚至抵消。特别地，**如何在保留数据的准确率和遗忘数据的彻底性之间取得平衡**，一直缺乏可靠的理论指导。

Sharpness-Aware Minimization（SAM）已被证明能寻找更平坦的损失最小值，有效抑制噪声记忆化，提升泛化性能。一个自然的假设是：擅长抑制记忆化的优化器是否也更擅长"遗忘"？本文对此进行了深入的理论和实验研究。

## 方法详解

### 整体框架

本文基于两层 CNN 的信号-噪声分解框架，将模型权重更新分解为信号学习系数 $\kappa$ 和噪声学习系数 $\zeta$，系统分析 SGD 与 SAM 在 NegGrad 遗忘方案下的行为差异。

具体地，输入图像的每个 patch 要么包含类别信号 $y_i \varphi$，要么是噪声向量 $\xi_i$。模型权重可以分解为信号分量和噪声分量，训练目标是增加 $\kappa$（信号学习）同时控制 $\zeta$（噪声记忆化）。

### 关键设计

1. **SAM 的去噪失效定理（Lemma 3.1）**: 在 NegGrad 遗忘方案下，SAM 的扰动项 $\hat{\epsilon}$ 会使保留集 $\mathcal{R}$ 的噪声神经元去激活（保持去噪特性），但遗忘集 $\mathcal{F}$ 的噪声神经元仍然保持激活。这意味着 SAM 在遗忘集上"退化"为 SGD 的行为——对遗忘集过拟合的程度与 SGD 相当。核心原因是 NegGrad 对遗忘集做梯度上升，逆转了 SAM 扰动的作用方向。

2. **差异化测试误差界（Theorem 3.2 & 3.3）**: SGD 在 NegGrad 下，当信号强度 $\|\varphi\|_2 \geq C_1 d^{1/4} n^{-1/4} P \sigma_p$ 时可达良性过拟合（benign overfitting），否则有害过拟合导致测试误差 $\geq 0.1$。SAM 的优势在于：即使在更弱的信号条件 $\|\varphi\|_2 \geq \Omega(1)$ 下，只要保留集信号充足，SAM 仍能保证低测试误差。这得益于 SAM 在保留集上的去噪特性未被破坏。

3. **信号余量与 $\alpha$ 阈值（Lemma 3.4）**: SAM 在保留集上的信号学习速率是 SGD 的 $\Theta(\|\varphi\|_2^2)$ 倍，因此 SAM 允许更小的保留权重 $\alpha$（即更强调遗忘）。在良性过拟合区间，SGD 与 SAM 所需的 $\alpha$ 差距为 $O(\sqrt{d/n})$ 量级。这一发现非常实际：**$\alpha$ 的选择不仅取决于保留集与遗忘集的大小比，还取决于信号强度和数据维度**。

4. **Sharp MinMax 算法**: 受上述理论启发——SAM 在保留集上有优势但在遗忘集上退化——作者提出将模型拆分为两部分：

    - **保留模型**：用 SAM（锐度最小化）训练，利用其去噪特性保持泛化
    - **遗忘模型**：用锐度最大化（Sharpness Maximization）训练，故意让模型在遗忘集上过拟合到更尖锐的损失景观，从而更彻底地"记住然后遗忘"

   模型拆分基于梯度幅值排名：计算参数对遗忘集梯度的幅值，高幅值参数分配给遗忘模型。

### 损失函数 / 训练策略

- **NegGrad 双目标损失**: $\mathcal{L}_{\text{NegGrad}} = \alpha \cdot \mathcal{L}(\mathcal{R}) - (1-\alpha) \cdot \mathcal{L}(\mathcal{F})$，保留集梯度下降 + 遗忘集梯度上升
- **Sharp MinMax**: 保留部分用 SAM 的 $\min_W \mathcal{L} + [\max_{\hat{\epsilon}} \mathcal{L}(W+\hat{\epsilon}) - \mathcal{L}(W)]$，遗忘部分用锐度最大化 $\min_W \mathcal{L} - [\max_{\hat{\epsilon}} \mathcal{L}(W+\hat{\epsilon}) - \mathcal{L}(W)]$
- **遗忘难度量化**: 使用 Feldman & Zhang (2020) 的记忆化得分 $\text{mem}(\mathcal{A}, \mathcal{S}, i)$ 衡量每个样本的遗忘难度，高记忆化样本更难遗忘

## 实验关键数据

### 实验设置
- 数据集：CIFAR-100、ImageNet-1K（主实验）、CIFAR-10、Tiny-ImageNet（补充实验）
- 模型：ResNet-50
- 遗忘集大小 $|\mathcal{F}| \approx 5\%|\mathcal{S}|$，按记忆化得分分为 $\mathcal{F}_{\text{high}}, \mathcal{F}_{\text{mid}}, \mathcal{F}_{\text{low}}$
- 评价指标：ToW（Tug-of-War），综合衡量保留准确率、遗忘准确率、测试准确率
- 对比方法：NegGrad, RL, SalUn, L1-Sparse, SCRUB

### 主实验
| 方法 | ImageNet AVG ToW | CIFAR-100 AVG ToW | 说明 |
|------|-----------------|-------------------|------|
| NegGrad+SGD | 83.83 | 81.80 | Baseline |
| NegGrad+SAM 0.1 | 83.68 | 72.78 | SAM 作为 $\mathcal{U}$ |
| NegGrad+ASAM 1.0 | 84.84 | 83.11 | 最佳 NegGrad 变体 |
| Sharp MinMax+ASAM 1.0 | ~87.90 | ~87.90 | **新 SOTA** |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 减小 $\alpha$ | SAM 抗衰退能力更强 | SGD 最先崩溃，ASAM 1.0 最鲁棒 |
| MIA 正确率 | SAM 一致降低 | 遗忘集更难被成员推断攻击识别 |
| 特征纠缠度 $E_{Wp}$ | SAM < SGD | SAM 遗忘后保留/遗忘特征分离更好 |

### 关键发现

1. **SAM 一致提升所有遗忘方法**: 无论作为预训练算法还是遗忘算法，SAM 都能提升 ToW 指标
2. **过拟合可以有益于遗忘**: 在隐私/版权等严格的样本级遗忘场景中，故意让模型对遗忘集过拟合反而更有效——这挑战了"过拟合总是有害"的常规认知
3. **SGD 有时遗忘准确率更低**: 尽管 SAM 的 ToW 更高，SGD 有时能达到更低的遗忘集准确率，印证了 SGD 在遗忘集上更深度过拟合的理论预测
4. **损失景观可视化**: SAM 预训练模型更平坦，但有趣的是 SGD 在遗忘后变得更平坦，可能存在隐式正则化效应

## 亮点与洞察

- **理论贡献突出**: 首次在严格的信号-噪声框架下分析 SAM 在机器遗忘中的行为，证明了 SAM 去噪特性的"选择性失效"——这是一个反直觉但重要的发现
- **连接优化与遗忘**: 将锐度感知优化与机器遗忘深度结合，提供了 $\alpha$ 选择的理论指导，不再依赖纯启发式调参
- **Sharp MinMax 设计精巧**: 利用"过拟合=好遗忘"这一洞察，将模型拆分为互补的两部分，既保持泛化又增强遗忘
- **Wasserstein 纠缠度指标**: 提出基于最优传输的特征纠缠度度量 $E_{Wp}$，比方差纠缠度 $E_{\text{Var}}$ 更能区分不规则形状的特征分布

## 局限与展望

1. **弱信号区间理论缺失**: 当保留信号强度为 $O(1)$ 时，SAM 的行为未被完全刻画，可能存在有害过拟合
2. **$\alpha$ 与模型拆分比例的交互**: 两个超参数（保留权重 $\alpha$ 和遗忘模型占比）之间的交互关系未被理论分析
3. **两层CNN假设**: 理论分析基于两层CNN，向深层网络的推广需要额外工作
4. **遗忘后SGD的"正则化"效应**: 论文观察到SGD遗忘后损失景观变平坦但未给出解释
5. **计算开销**: SAM 本身需要两次前向/反向传播，Sharp MinMax 还增加了模型拆分的成本

## 相关工作与启发

- **与 SalUn 的关系**: SalUn 也做参数选择性遗忘，但用随机标签翻转；Sharp MinMax 用锐度最大化替代，更有理论基础
- **与 SCRUB 的关系**: SCRUB 基于知识蒸馏+NegGrad，SAM 可以直接作为插件提升其性能
- **对隐私遗忘的启发**: "过拟合有益遗忘"的发现对设计满足差分隐私约束的遗忘算法有重要指导意义
- **对 LLM 遗忘的启发**: 论文的框架可能延伸到大模型遗忘（如知识编辑、概念擦除），尤其是 SAM 在 LLM 微调中已有广泛应用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)
- [Classic Video Denoising in a Machine Learning World: Robust, Fast, and Controllable](../../CVPR2025/image_restoration/classic_video_denoising_in_a_machine_learning_world_robust_fast_and_controllable.md)
- [MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring](../../CVPR2026/image_restoration/motionaware_animatable_gaussian_avatars_deblurring.md)
- [Outlier-Aware Post-Training Quantization for Image Super-Resolution](../../ICCV2025/image_restoration/outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal](../../ECCV2024/image_restoration/oapt_offset-aware_partition_transformer_for_double_jpeg_artifacts_removal.md)

<!-- RELATED:END -->
