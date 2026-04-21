---
title: >-
  [论文解读] SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning
description: >-
  [ECCV 2024][自监督学习][单应性估计] 提出 SCPNet，通过模内自监督学习（intra-modal self-supervised learning）、相关性网络和一致性特征图投影三个关键组件的协同，首次在卫星-地图等大模态差距数据集上实现了有效的无监督跨模态单应性估计，MACE 比监督方法 MHN 低 14%。
tags:
  - ECCV 2024
  - 自监督学习
  - 单应性估计
  - 无监督学习
  - 跨模态
  - 多光谱图像
---

# SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning

**会议**: ECCV 2024  
**arXiv**: [2407.08148](https://arxiv.org/abs/2407.08148)  
**代码**: https://github.com/RM-Zhang/SCPNet  
**领域**: 自监督学习  
**关键词**: 单应性估计, 无监督学习, 跨模态, 自监督学习, 多光谱图像

## 一句话总结
提出 SCPNet，通过模内自监督学习（intra-modal self-supervised learning）、相关性网络和一致性特征图投影三个关键组件的协同，首次在卫星-地图等大模态差距数据集上实现了有效的无监督跨模态单应性估计，MACE 比监督方法 MHN 低 14%。

## 研究背景与动机

**领域现状**：单应性估计（Homography Estimation）旨在计算图像之间的全局透视变换，是图像配准、融合、导航等任务的基础。监督方法（DHN、MHN、IHN、RHWF）在大偏移量下表现优异，但获取跨模态图像的真值标注非常困难。无监督方法（UDHN、CA-UDHN、biHomE）通过像素或特征级别的强度一致性损失训练，无需真值标注。

**现有痛点**：现有无监督方法主要依赖"跨模态强度学习"——将源图像按预测单应性 warp 后与目标图像比较。但当模态差距大（如卫星图 vs 地图）且偏移量大（±32 像素/128×128 图像）时，跨模态的强度相似性高度非凸，优化极易不收敛。在 GoogleMap 数据集上，现有无监督方法全部失败或性能极差。

**核心矛盾**：无监督学习的核心矛盾是"缺乏直接监督信号"——跨模态图像外观差异巨大，像素/特征级相似度度量不可靠，而真值单应性又无法获取。

**本文目标** 如何在大模态差距和大偏移量下实现有效的无监督跨模态单应性估计？

**切入角度**：作者提出了一个关键洞察——虽然跨模态真值单应性难以获取，但模内（intra-modal）的真值单应性可以通过模拟变形轻松生成。通过在两个模态内部分别进行自监督学习，共享权重的网络可以将单应性估计知识从模内泛化到跨模态。先导实验验证了这一发现：仅用模内自监督训练的网络在跨模态测试上的表现反而优于直接跨模态强度学习。

**核心 idea**：通过"模内自监督学习"为跨模态单应性估计提供可靠的间接监督，结合相关性网络和一致性特征图投影构建强大的无监督学习框架。

## 方法详解

### 整体框架
SCPNet 的训练框架包含三个分支：两个模内自监督学习分支（分别对模态 A 和模态 B 进行模拟单应性变换后做有监督回归）和一个跨模态学习分支（通过一致性特征图比较实现无监督约束）。三个分支共享可学习模块的权重，同时训练。推理时仅使用跨模态预测分支。

### 关键设计

1. **模内自监督学习 (Intra-modal Self-supervised Learning)**:

    - 功能：为跨模态单应性估计提供间接但可靠的监督信号
    - 核心思路：对模态 A 的图像 $\mathbf{I}_A$ 施加随机模拟单应性变形得到 $\mathbf{I}_{A'}$，构建 $(\mathbf{I}_A, \mathbf{I}_{A'}, \mathbf{H}_{GT,A})$ 三元组；模态 B 同理。用共享权重的网络分别预测两个模态内部的单应性，以 L1 损失监督：$\mathcal{L}_S = \|\mathbf{O} - \mathbf{O}_{GT}\|_1$，其中 O 是四角点偏移量参数化。关键发现：即使不使用任何跨模态损失，仅靠模内自监督训练的网络在跨模态测试上也能取得可比性能（MACE=13.06 vs 跨模态训练不收敛）
    - 设计动机：灵感来自多任务学习——模内自监督任务与跨模态估计高度相关，共享表示中隐式学到了跨模态的结构对应关系。这相当于用"免费的"模拟数据提供高质量监督，绕过了跨模态强度损失不可靠的核心难题

2. **相关性网络 (Correlation-based Homography Estimation Network)**:

    - 功能：通过显式计算特征相关性来约束网络学习更清晰、更可迁移的知识
    - 核心思路：用共享权重的特征提取器分别产出两个模态的特征图 $\mathbf{F}_A$ 和 $\mathbf{F}_B$，在局部窗口内计算内积相关性 $\mathbf{C}(\mathbf{x}, \mathbf{r}) = \text{ReLU}(\mathbf{F}_A(\mathbf{x})^T \mathbf{F}_B(\mathbf{x}+\mathbf{r}))$，$\|\mathbf{r}\|_\infty \leq R$。相关性图输入单应性估计器预测偏移量。相比直接拼接图像对输入网络的传统做法，相关性将网络分解为"特征提取 → 相似度编码 → 单应性解码"三个明确环节
    - 设计动机：相关性将特征相似度显式编码，使得模内和跨模态的单应性解码知识统一化——无论是哪种模态，相关性图的模式是一致的，这极大促进了模内知识向跨模态的迁移

3. **一致性特征图投影 (Consistent Feature Map Projection)**:

    - 功能：将跨模态图像从强度变化空间投影到强度不变的潜在空间，使跨模态强度监督变得可行
    - 核心思路：用卷积块（3×3 conv + 残差块 + 1×1 conv）将输入图像投影为单通道的一致性特征图。跨模态损失在投影后的特征图上计算：$\mathcal{L}_C = \|\mathbf{P}_A - \mathbf{P}_{B,W}\|_1 / \|\mathbf{P}_A - \mathbf{P}_B\|_1$，分子最小化 warp 后的相似度，分母最大化未 warp 的差异以防止退化解
    - 设计动机：直接在原始像素或简单特征空间比较跨模态图像会因模态差异而失败。一致性投影学习了一个两个模态共享的结构表示，使得强度监督重新可用。同时分母项的"反退化"设计很巧妙——防止网络将所有图像投影为常数

### 损失函数 / 训练策略
总损失为三个分支的加权和：$\arg\min_{\xi,\zeta} \mathcal{L}_C(\delta_\zeta(\mathbf{I}_A), \mathcal{W}(\delta_\zeta(\mathbf{I}_B), \psi_\xi(\cdot))) + \lambda \mathcal{L}_S(\psi_\xi(\delta_\zeta(\mathbf{I}_A), \delta_\zeta(\mathbf{I}_{A'})), \mathbf{H}_{GT,A}) + \lambda \mathcal{L}_S(\psi_\xi(\delta_\zeta(\mathbf{I}_B), \delta_\zeta(\mathbf{I}_{B'})), \mathbf{H}_{GT,B})$，其中 $\lambda=0.1$。使用 AdamW 优化器，学习率 $4\times10^{-4}$，batch size 8，训练 120K 迭代。

## 实验关键数据

### 主实验

**跨模态数据集 (GoogleMap, [-32,+32] offset)**:

| 方法 | 类型 | Easy↓ | Moderate↓ | Hard↓ | Mean MACE↓ |
|------|------|-------|-----------|-------|------------|
| SIFT | 手工 | 19.17 | 23.87 | 29.04 | 24.53 |
| UDHN | 无监督 | 18.63 | 21.55 | 26.89 | 22.84 |
| CA-UDHN | 无监督 | NC | NC | NC | NC |
| biHomE | 无监督 | NC | NC | NC | NC |
| DHN | 监督 | - | - | - | ~6.9 |
| MHN | 监督 | - | - | - | ~5.06 |
| **SCPNet** | **无监督** | - | - | - | **4.35** |

**跨光谱数据集 (Harvard)**:

| 方法 | 类型 | Mean MACE↓ |
|------|------|------------|
| MHN | 监督 | ~10.2 |
| **SCPNet** | **无监督** | 比 MHN 低 25.2% |

### 消融实验

| Setting | Self | Correlation | Projection | Cross | MACE↓ |
|---------|------|-------------|------------|-------|-------|
| 1 | ✗ | ✗ | ✗ | ✓ | NC |
| 2 | ✗ | ✓ | ✗ | ✓ | NC |
| 3 | ✗ | ✗ | ✓ | ✓ | 24.64 |
| 5 | ✓ | ✗ | ✗ | ✗ | 13.06 |
| 6 | ✓ | ✓ | ✗ | ✗ | 9.68 |
| 8 | ✓ | ✓ | ✓ | ✗ | 7.70 |
| **9** | **✓** | **✓** | **✓** | **✓** | **4.35** |

### 关键发现
- 模内自监督学习是整个框架的基石：没有它（Setting 1-4），训练要么不收敛要么精度很差；仅用它（Setting 5）就能达到 13.06 的 MACE
- 三个组件逐步叠加带来持续改善：13.06 → 9.68（+相关性）→ 7.70（+投影）→ 4.35（+跨模态损失）
- 相关性网络促进了一致性特征图的生成质量——对比实验表明，使用相关性时投影的特征图结构更清晰
- 计算开销：模内自监督训练将训练时间从 3.18h 增加到 6.98h，显存从 4.6GB 增加到 9.1GB，但推理时无额外开销

## 亮点与洞察
- **模内自监督学习的核心洞察极为精彩**：将"免费"的模内模拟监督转化为跨模态能力，本质上是利用了共享权重网络在多任务学习中的正迁移效应。这个思路可以推广到任何跨域/跨模态的无监督学习场景——只要域间任务共享某些底层知识，就可以用模内自监督作为跳板
- **损失函数的分母设计**：$\mathcal{L}_C$ 的分母项 $\|\mathbf{P}_A - \mathbf{P}_B\|_1$ 防止退化解很巧妙——既保证了投影的有意义性，又避免了额外的正则化项
- 消融实验的设计非常系统，完整展示了每个组件的独立和协同贡献，堪称范例

## 局限与展望
- 作者指出可以引入多尺度、迭代优化和 Transformer 架构替代 CNN，进一步提升精度
- 在 Flash/no-flash 数据集上不如监督方法 LocalTrans、IHN、RHWF，说明在模态差异较小时无监督方法的上限仍有差距
- 训练时间翻倍（3.18h → 6.98h），可探索更高效的自监督训练策略（如渐进式加入模内分支）
- 仅验证了 128×128 分辨率下 ±32 的偏移范围，更大分辨率和变形范围的表现有待验证

## 相关工作与启发
- **vs UDHN/CA-UDHN**: 这些方法仅用跨模态强度损失，在大模态差距下完全失效（NC）。SCPNet 的模内自监督学习从根本上解决了监督信号缺失的问题
- **vs biHomE**: 改用感知损失也无法在 GoogleMap 上收敛。说明问题的根源不在于损失函数的设计，而在于跨模态直接比较本身的不可靠性
- **vs MHN（监督）**: SCPNet 在 GoogleMap 上比 MHN 低 14% MACE，是一个无监督方法超越监督方法的罕见案例，凸显了模内自监督学习的潜力

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 模内自监督学习的idea极具创意，先导实验的设计和洞察发现过程令人信服
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个数据集、14 种对比方法、系统化的 9 组消融实验
- 写作质量: ⭐⭐⭐⭐ 从先导实验引出方法设计的叙事结构很好，但公式和符号较多
- 价值: ⭐⭐⭐⭐ 无监督超越监督的案例有标杆意义，思路可迁移到其他跨模态任务

<!-- RELATED:START -->

## 相关论文

- [TRIDENT: Tri-Modal Molecular Representation Learning with Taxonomic Annotations and Structural Relationships](../../NeurIPS2025/self_supervised/trident_tri-modal_molecular_representation_learning_with_taxonomic_annotations_a.md)
- [ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [Self-supervised Video Copy Localization with Regional Token Representation](self-supervised_video_copy_localization_with_regional_token_representation.md)
- [Rethinking Unsupervised Outlier Detection via Multiple Thresholding](rethinking_unsupervised_outlier_detection_via_multiple_thresholding.md)
- [TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation](../../CVPR2026/self_supervised/teflow_enabling_multi-frame_supervision_for_self-supervised_feed-forward_scene_f.md)

<!-- RELATED:END -->
