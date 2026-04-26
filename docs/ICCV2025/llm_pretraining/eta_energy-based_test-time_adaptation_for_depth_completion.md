---
title: >-
  [论文解读] ETA: Energy-based Test-time Adaptation for Depth Completion
description: >-
  [ICCV 2025][测试时适配] 提出ETA方法，利用能量模型量化深度预测属于源域分布的可能性，并在测试时通过最小化目标域预测的能量值来引导预训练深度补全模型适配到新环境，在室外和室内场景平均比先前SOTA分别提升6.94%和10.23%。
tags:
  - ICCV 2025
  - 测试时适配
  - 能量模型
  - 深度补全
  - 域自适应
  - 对抗扰动
---

# ETA: Energy-based Test-time Adaptation for Depth Completion

**会议**: ICCV 2025  
**arXiv**: [2508.05989](https://arxiv.org/abs/2508.05989)  
**代码**: https://fuzzythecat.github.io/eta  
**领域**: LLM / NLP (分类为depth completion，实际为3D视觉/深度补全)  
**关键词**: 测试时适配, 能量模型, 深度补全, 域自适应, 对抗扰动

## 一句话总结

提出ETA方法，利用能量模型量化深度预测属于源域分布的可能性，并在测试时通过最小化目标域预测的能量值来引导预训练深度补全模型适配到新环境，在室外和室内场景平均比先前SOTA分别提升6.94%和10.23%。

## 研究背景与动机

深度补全是从RGB图像和稀疏点云估计稠密深度图的多模态3D重建任务，广泛应用于自动驾驶、物理AI代理和扩展现实。然而，在源域训练好的模型部署到新环境时，由于光照、遮挡等因素变化导致的协变量偏移，性能会显著下降。

现有域适应方法的局限性：（1）传统域适应需要源域和目标域的标注数据；（2）无源域适应虽不需要源数据但需多次遍历目标数据集；（3）这些方法都不适合实时空间应用的在线场景。测试时适配（TTA）允许模型在无标注的流式数据上逐批更新参数，是更实际的选择。

核心挑战在于：如何在不了解目标域分布的情况下量化预测偏差？本文的关键洞察是——稀疏点云/深度估计的协变量偏移远小于RGB图像，因此可以在深度预测空间上定义能量函数，以稀疏深度为条件评估预测的合理性。

## 方法详解

### 整体框架

ETA包含两个阶段：（1）准备阶段：在源数据上训练能量模型，利用对抗扰动生成分布外样本作为训练信号；（2）测试时适配阶段：冻结能量模型和深度补全模型主体参数，仅更新插入到RGB编码器中的适配层参数，通过最小化能量值将预测对齐到源域分布。

### 关键设计

1. **基于区域的深度能量模型**: 定义能量函数 $E_\phi: (\hat{d}; z) \rightarrow e$，将稠密深度预测 $\hat{d}$ 和稀疏深度 $z$ 作为输入，输出低分辨率能量图 $e: \Omega \rightarrow [0, 1]$。低能量表示预测属于源域分布（较小误差），高能量表示分布外预测（较大误差）。相比全图单一能量值，区域级能量图能定位错误区域实现靶向更新。能量目标值通过Gibbs分布映射：
    $y = 1 - \exp(-\Delta / \tau)$
   其中 $\Delta$ 是每个patch的MSE误差，$\tau$ 为温度参数。

2. **对抗扰动生成分布外样本**: 训练能量模型的关键困难是缺乏分布外数据。本文巧妙地利用FGSM对抗扰动作为探索数据空间的机制，而非对目标分布做假设：
    $\tilde{I_s} = I_s + \epsilon_I \cdot \text{sign}(\nabla_{I_s} \mathcal{L}_{\text{sup}}(\hat{d_s}, d_s))$
    $\tilde{z_s} = z_s + \epsilon_z \cdot \text{sign}(\nabla_{z_s} \mathcal{L}_{\text{sup}}(\hat{d_s}, d_s))$
   对抗扰动使输入偏向数据空间的盲区高密度区域，导致模型产生分布外错误。这种方法覆盖范围广于单一"目标域"，允许同一能量模型在不同测试数据集上复用。

3. **能量引导的测试时适配**: 在预训练的深度补全模型图像编码器中插入轻量级适配模块 $m_\psi$，测试时仅更新 $\psi$ 和BatchNorm统计量。适配损失整合三个组件：

    - 能量最小化损失：$\ell_{\text{energy}} = -\frac{1}{|\Omega_p|}\sum_{x} \log(1 - E_\phi(\hat{d}_t, z_t)(x))$
    - 稀疏深度一致性：$\ell_{\text{sparse}} = \frac{1}{|\Omega_z|}\sum_{x \in \Omega_z}|\hat{d}_t(x) - z_t(x)|$
    - 局部平滑性：边缘感知的深度梯度L1惩罚，用图像梯度加权保留物体边界

### 损失函数 / 训练策略

能量模型训练使用交叉熵损失：$\mathcal{L}_{\text{energy}} = -\frac{1}{|\Omega_p|}\sum_x y(x)\log(y(x)/e(x))$。测试时适配损失为 $\mathcal{L}_{\text{adapt}} = w_e \ell_{\text{energy}} + w_z \ell_{\text{sparse}} + w_s \ell_{\text{smooth}}$。注意：为每个深度补全模型单独训练对应的能量模型，因为能量分布依赖于深度补全模型的参数。

## 实验关键数据

### 主实验

| 深度补全模型 | 适配方法 | VKITTI-FOG MAE | nuScenes MAE | NYUv2 MAE | ScanNet MAE |
|------------|---------|---------------|-------------|----------|-----------|
| NLSPN | Pre-trained | 1.309 | 2.656 | 0.388 | 0.233 |
| NLSPN | ProxyTTA | 0.686 | 2.589 | 0.124 | 0.074 |
| NLSPN | **ETA** | **0.545** | **2.359** | **0.105** | **0.067** |
| BP-Net | Pre-trained | 0.893 | 2.787 | 0.234 | 0.123 |
| BP-Net | ProxyTTA | 0.571 | 2.373 | 0.174 | 0.102 |
| BP-Net | **ETA** | **0.544** | **2.281** | **0.161** | **0.093** |
| CostDCNet | Pre-trained | 1.042 | 3.064 | 0.189 | 0.144 |
| CostDCNet | ProxyTTA | 0.512 | 2.062 | 0.095 | 0.068 |
| CostDCNet | **ETA** | **0.508** | **2.048** | **0.089** | **0.059** |

### 消融实验

| 能量更新策略 | MAE | RMSE | 说明 |
|-----------|------|------|------|
| 全局(图像级) | 1.406 | 4.226 | 单一能量值无法定位错误区域 |
| **局部(区域级)** | **0.703** | **2.996** | 细粒度能量图引导靶向更新 |
| 基线(无能量) | 2.842 | 6.557 | - |

| 跨域适配场景 | 方法 | NYUv2 MAE | SceneNet MAE | ScanNet MAE |
|-----------|------|----------|-------------|-----------|
| KITTI→室内 | Pre-trained | 1.987 | 1.432 | 2.657 |
| KITTI→室内 | ProxyTTA | 1.380 | 0.401 | 0.311 |
| KITTI→室内 | **ETA** | **1.322** | **0.340** | **0.272** |

### 关键发现

- ETA在室外和室内场景上分别比前SOTA ProxyTTA平均提升5.36%和10.13%的MAE
- 相比直接将分类TTA方法TEA搬运到深度估计，ETA提升了24.9%的MAE，证明了面向回归任务定制设计的必要性
- 初始误差越大的模型从ETA中获益越多（如MSG-CHN在VKITTI-FOG上），能量模型能有效"拉回"严重偏离的预测
- 即使面临极端的室外→室内跨域场景，ETA仍能一致保持最佳性能

## 亮点与洞察

- 用对抗扰动代替对目标域的假设来生成分布外数据，是一个非常聪明的设计——它使得单一能量模型可跨多种目标域复用
- 区域级能量图不仅提升了适配效果，还提供了一定程度的可解释性（高能量区域即为模型不确信的区域）
- 稀疏深度在域偏移下的稳定性是本文的关键假设和洞察，将能量条件定义在稀疏深度上而非RGB特征上

## 局限与展望

- 需要为每个深度补全模型单独训练能量模型，部署成本与模型数量线性增长
- 对抗扰动的强度参数 $\epsilon_I, \epsilon_z$ 需要调优，不同场景可能需要不同设置
- 能量模型的patch大小影响定位精度和计算开销之间的trade-off
- 在LiDAR点云极其稀疏（如手机SfM）的场景下，稀疏深度条件化可能不够可靠

## 相关工作与启发

- 能量模型在生成模型和密度估计领域有成熟应用，本文将其巧妙嫁接到TTA这一新应用场景
- 对抗扰动作为数据增强/探索工具的思路可推广到其他缺乏OOD样本的任务
- 区域级能量引导的思想可能适用于其他密集预测任务（如语义分割、光流）的域适应

## 评分

- 新颖性: ⭐⭐⭐⭐ 能量引导TTA + 对抗扰动探索数据空间的组合创新
- 实验充分度: ⭐⭐⭐⭐⭐ 4个深度补全模型×6个数据集×5个基线方法，覆盖室内/室外/跨域
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，消融完善，但整体较长
- 价值: ⭐⭐⭐⭐ 为深度补全领域的域适应提供了实用且有效的TTA方案

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Implicit Bias and Loss of Plasticity in Matrix Completion: Depth Promotes Low-Rank](../../ICLR2026/llm_pretraining/implicit_bias_and_loss_of_plasticity_in_matrix_completion_depth_promotes_low-ran.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](../../NeurIPS2025/llm_pretraining/the_curse_of_depth_in_large_language_models.md)
- [\[ICML 2025\] Language Model Developers Should Report Train-Test Overlap](../../ICML2025/llm_pretraining/language_model_developers_should_report_train-test_overlap.md)
- [\[ACL 2025\] TokAlign: Efficient Vocabulary Adaptation via Token Alignment](../../ACL2025/llm_pretraining/tokalign_vocab_adaptation.md)
- [\[AAAI 2026\] Learning Time in Static Classifiers](../../AAAI2026/llm_pretraining/learning_time_in_static_classifiers.md)

<!-- RELATED:END -->
