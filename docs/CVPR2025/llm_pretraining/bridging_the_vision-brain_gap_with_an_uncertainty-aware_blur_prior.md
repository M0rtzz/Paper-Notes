---
title: >-
  [论文解读] Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior
description: >-
  [CVPR 2025][脑信号解码] 提出不确定性感知模糊先验(UBP)方法，通过估计脑信号与视觉刺激间的不确定性动态模糊图像高频细节，在零样本脑-图像检索任务上大幅超越SOTA。
tags:
  - CVPR 2025
  - 脑信号解码
  - EEG
  - 不确定性量化
  - 对比学习
  - 模糊先验
---

# Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior

**会议**: CVPR 2025  
**arXiv**: [2503.04207](https://arxiv.org/abs/2503.04207)  
**代码**: https://github.com/ (待公开)  
**领域**: 多模态VLM / 脑科学  
**关键词**: 脑信号解码, EEG, 不确定性感知, 模糊先验, 视觉-脑域差距

## 一句话总结

首次提出"系统差距"（System GAP）和"随机差距"（Random GAP）的概念来描述脑信号与视觉刺激之间的信息不匹配，通过不确定性感知的模糊先验（UBP）动态调整图像模糊程度来缓解训练中的过拟合，在 200-way 零样本脑-图像检索任务上实现 50.9% top-1 准确率，超越前 SOTA 13.7 个百分点。

## 研究背景与动机

**领域现状**：视觉神经解码通过 EEG/fMRI 等脑信号来检索或重建原始视觉刺激。主流方法（如 BraVL、NICE）通过对比学习将脑信号编码器的输出与 CLIP 图像特征对齐。

**现有痛点**：现有方法直接对齐脑信号和原始图像特征，忽略了二者之间存在的信息差距。人脑处理视觉信息时会丢失高频细节（System GAP），而注意力波动、认知联想、信号采集噪声等因素进一步引入随机变异（Random GAP）。

**核心矛盾**：在配对数据稀缺的情况下，模型被迫学习弥合这些差距，导致过拟合训练集并在新数据上泛化失败。

**本文目标**：引入先验知识缓解两种差距的影响，改善脑-视觉对比学习的对齐质量。

**切入角度**：人眼视觉系统本身就是一个低通滤波器——中央凹区域分辨率高、周边分辨率低，因此脑信号中不包含图像的全部高频信息。模糊处理可以让图像更接近脑信号的信息水平。

**核心 idea**：对训练图像施加高斯模糊来模拟信息丢失（缓解 System GAP），并通过估计每对样本的不确定性动态调整模糊强度（缓解 Random GAP）。

## 方法详解

### 整体框架

UBP 在标准的视觉-脑对比学习框架上增加两个组件：(1) 模糊先验——对图像施加中央凹式高斯模糊后再提取特征；(2) 不确定性量化——根据配对样本相似度动态调整模糊半径 $r$。

### 关键设计

1. **中央凹模糊先验（Blur Prior）**:

    - 功能：模拟人眼视觉系统的信息丢失，移除图像高频细节
    - 核心思路：先对图像施加均匀高斯模糊 $x_\text{blur}$，然后与原图按距离衰减权重混合：$\tilde{x}_v = \alpha \cdot x + (1-\alpha) \cdot x_\text{blur}$，其中 $\alpha(i,j) = \exp(-\lambda \cdot d(i,j)/L)$，$d(i,j)$ 是像素到中央凹（图像中心）的距离。这样中心保持清晰、周边逐渐模糊，模拟人眼的分辨率分布。模糊程度由高斯核半径 $r$ 控制。
    - 设计动机：直接对齐高频细节丰富的原图和缺失这些细节的脑信号会导致模型被迫学习不可能的映射，模糊先验降低了对齐的难度。

2. **不确定性感知的动态模糊**:

    - 功能：根据每对配对样本的不确定性自适应调整模糊强度
    - 核心思路：计算同批次配对样本的相似度矩阵 $M = h_b \cdot h_v^\top \cdot \text{softplus}(\tau)$，取对角线得到 $N$ 个配对的相似度 $S$。$S$ 近似服从正态分布 $\mathcal{N}(\hat\mu, \hat\sigma^2)$，落在置信区间外的样本为高不确定性样本。相似度过低的样本（Random GAP 大）增大模糊半径 $r_0 + c$，过高的样本减小 $r_0 - c$，正常范围内保持 $r_0$。
    - 设计动机：Random GAP 因素（注意力偏移、认知联想、信号噪声）导致脑信号质量参差不齐，需要对"匹配不好"的样本施加更强的模糊来降低对齐目标的难度。

### 损失函数 / 训练策略

使用对称交叉熵损失（SCE）进行对比学习，冻结 CLIP 视觉编码器 $f_V$，训练脑信号编码器 $f_B$。模糊半径在训练过程中通过不确定性估计动态更新。

## 实验关键数据

### 主实验

| 方法 | Top-1 Avg | Top-5 Avg |
|------|-----------|-----------|
| BraVL | 5.8 | 17.5 |
| NICE | 17.2 | 44.4 |
| ATM-S | 37.2 | 69.9 |
| **UBP (ours)** | **50.9** | **79.7** |

*THINGS-EEG 200-way 零样本检索，UBP 超越前 SOTA 13.7/9.8 个百分点*

### 消融实验

| 配置 | Top-1 | Top-5 |
|------|-------|-------|
| Baseline (无模糊) | 37.2 | 69.9 |
| +Blur Prior (固定 $r$) | 46.3 | 76.1 |
| +Uncertainty-aware | **50.9** | **79.7** |

### 关键发现
- 单独添加模糊先验即带来 9.1% top-1 提升，证实了 System GAP 的存在与影响
- 不确定性感知的动态调整再额外贡献 4.6%，验证了 Random GAP 的重要性
- 不同被试间性能差异与 EEG 信号变异性负相关（变异性大→性能低），支持了 Random GAP 理论
- 方法简单且通用，可以即插即用地应用于任何脑-视觉对比学习框架

## 亮点与洞察
- **System GAP 和 Random GAP 的概念化**：首次系统分析脑信号与视觉刺激不匹配的来源，为该领域提供了理论框架
- **中央凹模糊的生物学合理性**：模拟人眼视觉系统的分辨率下降，是少见的将感知神经科学知识融入深度学习训练的案例
- **方法极其简单**：核心就是加模糊+动态调整，几乎零额外计算开销，但效果巨大，体现了"正确思路>复杂方法"

## 局限与展望
- 仅在 EEG 信号上验证，fMRI 等高空间分辨率模态的适用性待测试
- 模糊先验假设脑信号主要丢失高频信息，但实际的信息丢失模式可能更复杂
- 不确定性量化基于简单的正态分布假设和三段式调整，更精细的建模可能更有效

## 相关工作与启发
- **vs ATM-S**: ATM-S 通过更好的脑信号编码器提升性能，UBP 从对齐目标（图像侧）出发优化，两者互补
- **vs NICE**: NICE 直接对比学习，UBP 引入先验减少对齐目标中的噪声信息
- 思路可迁移到其他噪声配对数据的对比学习场景（如弱标注、远监督）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ System/Random GAP 的概念和模糊先验的解决方案都值得称道
- 实验充分度: ⭐⭐⭐⭐ 10 个被试、消融充分、分析深入
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、理论与实验结合紧密
- 价值: ⭐⭐⭐⭐ 大幅推进了脑信号解码的性能
---
title: >-
  [论文解读] Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior
description: >-
  [CVPR 2025][LLM/NLP][视觉-脑域差距] 提出不确定性感知的模糊先验，为从脑信号（fMRI）重建视觉刺激提供物理合理的图像退化模型，缓解大脑编码过程中高频信息丢失对重建质量的影响。
tags:
  - CVPR 2025
  - LLM/NLP
  - 视觉-脑域差距
  - 不确定性感知
  - 模糊先验
  - fMRI解码
  - 图像重建
---

# Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: https://github.com/HaitaoWuTJU/Uncertainty-aware-Blur-Prior  
**领域**: 脑信号解码 / 视觉重建  
**关键词**: 视觉-脑域差距, 不确定性感知, 模糊先验, fMRI解码, 图像重建

## 一句话总结
提出不确定性感知的模糊先验，为从脑信号（fMRI）重建视觉刺激提供物理合理的图像退化模型，缓解大脑编码过程中高频信息丢失对重建质量的影响。

## 研究背景与动机
**领域现状**：从 fMRI 脑信号重建受试者所看到的图像是神经科学和 CV 的交叉热点。近期方法利用扩散模型从 fMRI 信号条件生成图像，取得了视觉上的进步。

**现有痛点**：人脑在编码视觉信息时天然存在信息损失 — 注意力资源有限、视觉记忆容量有限 — 导致高频细节被"模糊化"。现有方法忽略了这种物理退化过程。

**核心矛盾**：脑信号本身是对原始视觉刺激的有损编码，直接从有损信号重建会产生与真实图像不匹配的伪细节。

**本文目标** 如何在重建过程中显式建模大脑编码的信息损失，使重建结果在忠实于脑信号的同时避免产生错误的高频细节。

**切入角度**：将大脑的信息损失建模为一种"模糊退化"先验，并引入不确定性估计来自适应地控制重建的细节程度。

**核心 idea**：给扩散模型引入不确定性感知的模糊先验，在脑信号不确定的区域抑制细节生成，在确定的区域保留细节。

## 方法详解

### 整体框架
在标准的 fMRI-to-Image 扩散生成框架上，增加一个不确定性估计模块。该模块预测脑信号对每个空间区域的编码置信度，生成空间变化的模糊核（blur prior），条件化扩散模型的生成过程。

### 关键设计
1. **空间不确定性估计**：从 fMRI 特征中预测每个空间位置的编码不确定性，不确定区域施加更强的模糊约束。
2. **自适应模糊先验**：将不确定性映射为空间变化的模糊核，作为扩散模型的条件信号引导生成。
3. **多尺度脑信号融合**：在扩散模型的不同去噪阶段注入不同粒度的脑信号特征。

## 实验关键数据

### 关键发现
- 引入模糊先验后，重建图像在低阶感知指标（SSIM、PSNR）上显著提升
- 高频伪细节减少，人类评估中重建图像的忠实度评分提高
- 不确定性图与实际的信息损失区域高度吻合

## 亮点与洞察
- 从认知科学角度出发为技术方法提供物理合理的先验设计
- 不确定性感知的生成控制思路可推广到其他退化条件下的图像生成

## 局限与展望
- fMRI 信号的时空分辨率本身有限，不确定性估计的精度受约束
- 模糊先验假设大脑的信息损失主要表现为高频模糊，但实际退化模式可能更复杂
- 在不同受试者之间的泛化性有待验证

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution](../../ICCV2025/llm_pretraining/image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution.md)
- [\[CVPR 2025\] SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)
- [\[CVPR 2025\] 3D Prior is All You Need: Cross-Task Few-shot 2D Gaze Estimation](3d_prior_is_all_you_need_cross-task_few-shot_2d_gaze_estimation.md)
- [\[ICML 2025\] Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks](../../ICML2025/llm_pretraining/bayesian_neural_scaling_law_extrapolation_with_prior-data_fitted_networks.md)
- [\[NeurIPS 2025\] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?](../../NeurIPS2025/llm_pretraining/does_object_binding_naturally_emerge_in_large_pretrained_vision_transformers.md)

<!-- RELATED:END -->
