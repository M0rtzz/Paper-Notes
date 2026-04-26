---
title: >-
  [论文解读] Sign-In to the Lottery: Reparameterized Sparse Training from Scratch
description: >-
  [NeurIPS 2025][稀疏训练] 本文发现稀疏网络从头训练(PaI)性能差的根本原因是无法像dense-to-sparse方法那样学习正确的参数符号，为此提出Sign-In重参数化方法（θ=m⊙w），通过引入内部自由度来促进符号翻转，理论证明其能解决一种互补于过参数化的符号翻转情况，实验中显著提升了稀疏从头训练的性能。
tags:
  - NeurIPS 2025
  - 稀疏训练
  - 彩票假说
  - 符号翻转
  - 重参数化
  - 初始化剪枝
---

# Sign-In to the Lottery: Reparameterized Sparse Training from Scratch

**会议**: NeurIPS 2025  
**arXiv**: [2504.12801](https://arxiv.org/abs/2504.12801)  
**代码**: 无  
**领域**: 模型压缩 / 稀疏训练  
**关键词**: 稀疏训练, 彩票假说, 符号翻转, 重参数化, 初始化剪枝

## 一句话总结
本文发现稀疏网络从头训练(PaI)性能差的根本原因是无法像dense-to-sparse方法那样学习正确的参数符号，为此提出Sign-In重参数化方法（θ=m⊙w），通过引入内部自由度来促进符号翻转，理论证明其能解决一种互补于过参数化的符号翻转情况，实验中显著提升了稀疏从头训练的性能。

## 研究背景与动机
随着神经网络规模扩展到数十亿参数，高效训练流水线至关重要。网络稀疏化是实现高效性的关键手段。当前最好的稀疏化方法（如AC/DC、STR、CAP）依赖"先密后稀"的训练范式：先训练完整密集网络，再逐步剪枝。这要求在密集训练阶段承受完整的计算和内存开销。

彩票假说(LTH)提出：随机初始化中存在可以直接从头训练到与密集网络相当性能的稀疏子网络（即"彩票"）。但实际中，剪枝初始化(PaI)方法（如SNIP、GraSP、Synflow）与dense-to-sparse方法之间始终存在显著性能差距。

本文从参数符号(sign)的角度揭示了这个差距的根本原因：dense-to-sparse方法在早期训练阶段就完成了参数符号对齐，而PaI方法无法稳定地学习正确的符号。符号比幅度(magnitude)包含更多关键信息——用学到的符号+随机幅度初始化可以恢复baseline性能，但用学到的幅度+随机符号则不能。

## 方法详解

### 整体框架
Sign-In将每个权重参数θ重参数化为两个参数的逐元素乘积：θ = m ⊙ w。这引入了一个内部自由度β（由初始化决定），满足m² - w² = β·1。这个重参数化在原始参数空间中诱导了黎曼梯度流：dθ_t = -√(θ_t² + β) ⊙ ∇L(θ) dt。关键是√(θ_t² + β)使得即使θ接近零，梯度仍有非零的缩放因子，从而可能穿过零点实现符号翻转。

### 关键设计

1. **符号对齐的重要性**：通过广泛实验验证了三个关键发现：(a) 在密集训练的前10个epoch，大量符号就已经翻转并稳定下来；(b) 用学到的符号+学到的mask+随机幅度初始化稀疏网络，可以恢复baseline性能；(c) 用学到的幅度+随机符号则基本等价于随机初始化。这说明符号是参数-mask耦合中的关键信息。

2. **Sign-In重参数化**：每个参数θ替换为m⊙w的乘积。这个重参数化在黎曼梯度流中引入了√(θ² + β)的预条件器。当β>0时，即使θ=0处梯度也不为零（传统参数化中θ²=0会阻止符号翻转）。但训练过程中随机噪声和weight decay会使β收缩到0，因此需要周期性重置。

3. **动态缩放重置（Scaling Reset）**：每p个epoch重置内部缩放β=1（不改变实际权重θ=m⊙w的值，只重新分配m和w的数值）。通过解析解实现重置，使权重有更多机会对齐正确的符号。这是Sign-In的关键组成部分——消融实验显示去掉重置后性能提升大幅减少。

### 理论分析
在单神经元两层网络的简化设置中：
- **标准梯度流**（Theorem 5.1）：只有a>0且w>0时才能恢复真实值，其他三种符号组合都失败
- **Sign-In梯度流**（Theorem 5.2）：额外地在a<0, w>0时也能恢复真实值（互补于过参数化能解决的情况）
- **不可能性定理**（Theorem 5.4）：不存在任何连续重参数化能在w<0时恢复真实值——这从根本上说明仅靠重参数化无法完全替代密集训练

## 实验关键数据

### 主实验：Sign-In改善稀疏从头训练

| 数据集+模型 | 稀疏度 | Random Mask | Random + Sign-In | 提升 |
|------------|--------|------------|-----------------|------|
| CIFAR10 + ResNet20 | 80% | 88.25 | **89.37** | +1.12 |
| CIFAR10 + ResNet20 | 90% | 86.25 | **87.83** | +1.58 |
| CIFAR10 + ResNet20 | 95% | 83.56 | **84.74** | +1.18 |
| CIFAR100 + ResNet18 | 80% | 73.95 | **75.32** | +1.37 |
| CIFAR100 + ResNet18 | 90% | 72.96 | **73.94** | +0.98 |
| CIFAR100 + ResNet18 | 95% | 71.36 | **72.51** | +1.15 |
| ImageNet + ResNet50 | 80% | 73.87 | **74.12** | +0.25 |
| ImageNet + ResNet50 | 90% | 71.56 | **72.19** | +0.63 |
| ImageNet + ResNet50 | 95% | 68.72 | **69.38** | +0.66 |

### 符号对齐验证：ResNet50 on ImageNet, 90%稀疏度

| 初始化方式 | AC/DC | STR | RiGL |
|-----------|-------|-----|------|
| Baseline (dense-to-sparse) | 74.68 | 73.65 | 73.75 |
| Epoch 10 sign + learned mask | 73.97 | 71.7 | 73.32 |
| Epoch 30 sign + learned mask | **74.89** | **73.91** | 73.7 |
| Final sign + learned mask | 74.88 | 73.77 | **73.74** |
| Final mag + learned mask | 70.94 | 68.35 | 72.40 |
| Random init + learned mask | 70.6 | 68.38 | 71.89 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Sign-In有/无缩放重置 | 准确率 | 没有重置时提升不明显，重置是关键 |
| Sign-In + AC/DC | 准确率 | 进一步提升dense-to-sparse方法 |
| Sign-In + RiGL | 准确率 | 提升动态稀疏训练方法 |
| DeiT Small (ViT) | 准确率 | Sign-In在Vision Transformer上也有>1%提升 |
| β=1 统一缩放 vs 分层缩放 | 成功率 | 理论需β₁>β₂，但实践中统一β=1即可 |

### 关键发现
- 符号比幅度重要得多：用学到的符号+随机幅度可恢复baseline，反之不行
- 符号在训练早期（~10-30个epoch的warmup阶段）就已稳定
- Sign-In在训练全程促进更多符号翻转，找到更平坦的最小值（Hessian最大特征值更小）
- Sign-In在靠近输出层的位置促进更多符号翻转，符合理论预测
- 不可能性定理揭示了PaI与dense-to-sparse差距的根本性：没有任何重参数化能完全替代密集训练

## 亮点与洞察
- **问题诊断精准**：将PaI性能差归因于"符号对齐失败"，并通过大量实验提供了convincing的证据
- **理论与实践结合出色**：从单神经元理论分析出发，每个定理都有清晰的实践含义，且不可能性定理诚实地指出了方法的本质局限
- **正交性**：Sign-In与dense-to-sparse方法互补——它促进的符号翻转与密集训练促进的不同，两者组合可以进一步提升
- **计算开销低**：虽然参数量翻倍，但在ResNet50 ImageNet训练中仅增加约5%的训练时间，且推理时合并回单参数表示

## 局限与展望
- 尽管有进步，Sign-In仍无法缩小PaI与dense-to-sparse之间的全部差距
- 不可能性定理表明，纯粹基于重参数化的方法存在根本限制
- 实验主要在视觉任务上验证，缺乏NLP任务（如语言模型训练）的验证
- 理论分析局限于单神经元设置，多层深度网络的符号翻转动力学仍待研究
- 缩放重置的频率p是一个额外超参数，虽然实验中β=1和固定p表现稳定，但可能对不同任务需要调整

## 相关工作与启发
- 与Gadhikar & Burkholz (2024)的Learning Rate Rewinding分析密切相关，后者首先在单神经元上分析了密集训练促进符号翻转的机制
- 连续稀疏化方法（STR、spred、PILoT）使用相同的m⊙w重参数化来诱导稀疏性，而Sign-In反其道而行之用它来促进符号翻转
- 对高效训练的实践启发：如果未来能找到更好的"符号初始化"方法，可能真正实现稀疏网络从头训练达到密集网络水平

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Sparse Training from Random Initialization: Aligning Lottery Ticket Masks using Weight Symmetry](../../ICML2025/others/sparse_training_from_random_initialization_aligning_lottery_ticket_masks_using_w.md)
- [\[CVPR 2025\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2025/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[NeurIPS 2025\] Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)
- [\[NeurIPS 2025\] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?](are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)
- [\[ACL 2025\] Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](../../ACL2025/others/instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)

<!-- RELATED:END -->
