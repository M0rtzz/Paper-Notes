---
title: >-
  [论文解读] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference
description: >-
  [CVPR 2026][医学图像][去噪] 提出 RelativeFlow，基于 flow matching 的框架，通过将绝对噪声到干净映射分解为相对更噪到噪声映射，结合一致传输约束和基于模拟的速度场，从异质噪声参考中学习统一的去噪流，突破参考偏差限制。
tags:
  - "CVPR 2026"
  - "医学图像"
  - "去噪"
  - "flow matching"
  - "noisy reference"
---

# RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference

**会议**: CVPR 2026  
**arXiv**: [2604.15459](https://arxiv.org/abs/2604.15459)  
**代码**: [github.com/Deliver0/RelativeFlow](https://github.com/Deliver0/RelativeFlow)  
**领域**: 医学影像  
**关键词**: medical image denoising, flow matching, noisy reference, CT denoising, MR denoising

## 一句话总结

提出 RelativeFlow，基于 flow matching 的框架，通过将绝对噪声到干净映射分解为相对更噪到噪声映射，结合一致传输约束和基于模拟的速度场，从异质噪声参考中学习统一的去噪流，突破参考偏差限制。

## 研究背景与动机

医学图像去噪 (MID) 缺乏绝对干净的图像用于监督——只有来自不同采集协议和扫描仪配置的相对高质量参考，其质量在类别间异质变化。现有三种范式的局限：(1) SimSDL 将噪声参考当干净目标，导致次优收敛或参考偏差学习；(2) SSL 的独立噪声假设在医学成像中难以满足；(3) SimSGL 同样天真地将噪声参考当生成目标。核心问题：如何从异质质量的噪声参考中学习统一的高质量去噪映射。

## 方法详解

### 整体框架

医学图像去噪没有绝对干净的金标准，只有质量参差的"噪声参考"。RelativeFlow 的破题点是：与其硬把噪声参考当干净目标去学一个绝对的"噪声→干净"映射，不如把它拆成一串"更噪→噪声"的相对去噪步。对每个噪声参考 $x_t$，用模态特定的退化算子生成更噪样本 $x_{t-\Delta t}$，构建局部相对去噪步；再用两个组件分别保证这些相对流能一致地拼接（CoT 位移映射）、且能被神经网络学到（基于模拟的速度场），配合渐进式步长课程逐步铺满质量谱，最终组合成覆盖完整质量谱的统一去噪流。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["噪声参考 x_t<br/>（异质质量，无干净金标准）"] --> B
    B["一致传输 CoT 位移映射<br/>指数时间线性插值，相对流可传递组合为绝对流"] --> SVF
    subgraph SVF["基于模拟的速度场（SVF）"]
        direction TB
        C["模态退化算子 D_Δt<br/>CT 泊松-高斯 / MR Rician → 更噪样本 x_t−Δt"] --> D["速度目标<br/>u = (x_t − D_Δt(x_t)) / (e^Δt − 1)"]
    end
    SVF --> E["网络 N_θ 预测速度，L2 逼近 u"]
    F["渐进式步长课程<br/>逐轮扩大 [Δt_min, Δt_max]"] -.调度 Δt.-> C
    E --> G["采样：多步欧拉积分串接相对去噪步<br/>→ 统一高质量去噪输出"]
```

### 关键设计

**1. 一致传输（CoT）位移映射：让任意两段相对流都能严丝合缝拼成绝对流**

相对去噪要能用，前提是"从质量 $t_i$ 到 $t_j$"的各段流必须可传递、可组合，否则拼起来会错位。CoT 在指数时间空间用线性插值定义概率路径 $p_t = \lambda p_{t_i} + (1-\lambda)p_{t_j}$，权重 $\lambda = \frac{e^{-t}-e^{-t_j}}{e^{-t_i}-e^{-t_j}}$，由此保证嵌套插值的传递性——任意质量区间 $t_i \to t_j$ 的相对流都是绝对流 $\psi_{0 \to +\infty}$ 的一个分量，且连续的相对流可逐步组合为绝对流。论文从数学上证明了这两条性质（分量性与组合性），它们正是"从异质噪声参考学统一去噪"能成立的理论保证。

**2. 基于模拟的速度场（SVF）：用模态退化模型造出可监督的速度目标**

flow matching 需要一个速度场训练目标，但这里没有干净图像可算真值。SVF 转而利用模态特定的退化算子 $D_{\Delta t}$（CT 用泊松-高斯噪声、MR 用 Rician 噪声）从噪声参考模拟更噪样本，构造速度目标 $u = \frac{x_t - D_{\Delta t}(x_t)}{e^{\Delta t} - 1}$，训练时让网络预测去逼近它（L2 距离）。把领域知识编码进退化算子，使速度场目标在没有金标准的情况下依然可计算、且贴合各模态真实噪声分布。

**3. 渐进式步长课程：从小退化到大退化逐步覆盖完整质量谱**

如果一上来就让模型学大跨度退化，目标过难、收敛不稳。课程策略在训练中逐步扩大退化步长范围 $[\Delta t_{min}, \Delta t_{max}]$（每轮按系数 $\alpha$ 乘除调整），先学小退化的速度场再扩到大退化，逐步铺满整个质量谱；采样时用多步欧拉积分把这些相对步串成完整去噪轨迹。

### 损失函数 / 训练策略

训练损失为预测速度与模拟目标速度的 L2 距离：

$$\mathcal{L}_{RF} = \mathbb{E}_{x_t, \Delta t}\left[\left\|\mathcal{N}_\theta(D_{\Delta t}(x_t), \Delta t) - \frac{x_t - D_{\Delta t}(x_t)}{e^{\Delta t} - 1}\right\|_2^2\right]$$

配合上述渐进式步长课程学习策略。

## 实验关键数据

### 主实验

在 CT 和 MR 去噪任务上与 10 种方法（NID + MID）对比：

| 任务 | 指标 | 最优先前方法 | RelativeFlow |
|------|------|------------|-------------|
| CT 去噪 | PSNR/SSIM | 次优 | **最优** |
| MR 去噪 | PSNR/SSIM | 次优 | **最优** |

在 PSNR、SSIM、RMSE、LPIPS 四个指标上全面最优。

### 消融实验

- CoT 一致性约束是突破参考偏差的核心
- SVF 的模态特定退化建模优于通用噪声模型
- 渐进式步长课程优于固定步长训练

### 关键发现

- RelativeFlow 将不同质量水平的图像统一去噪到一致高质量
- CoT 的数学性质（分量性和组合性）为统一去噪提供了理论保证
- 模态特定退化算子使框架可扩展到不同医学成像模态

## 亮点与洞察

- 首次将噪声参考问题形式化并通过 flow matching 系统解决
- CoT 的数学分析（分量性+组合性证明）理论深度出色
- 跨 CT 和 MR 两种模态的验证展示了框架的通用性

## 局限与展望

- 退化算子的设计需要领域知识（泊松-高斯 vs Rician）
- 采样时的多步积分比判别式方法慢
- 理论上的 $t \to +\infty$ 极限在实践中的收敛质量需验证

## 相关工作与启发

- 相对流组合的思路对其他缺乏干净标签的学习场景有启发
- CoT 的数学框架可推广到其他渐进式质量提升任务
- 模态特定退化建模与通用 flow matching 的结合范式值得推广

## 评分

8/10 — 理论深度和实际效果兼备，是医学图像去噪领域的重要方法论贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graphbased_learning_of_inter_and_intraview_de.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)

</div>

<!-- RELATED:END -->
