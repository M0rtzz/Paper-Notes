---
title: >-
  [论文解读] UOTIP：无须配对的反演问题的非平衡最优传输映射
description: >-
  [ICML 2026][图像恢复][非平衡最优传输] 提出 UOTIP 方法——通过非平衡最优传输（UOT）框架将无须配对的图像反演问题表述为从有噪声测量分布到干净信号分布的映射学习，通过引入似然成本函数和二次项成本获得鲁棒性和理论保证。
tags:
  - "ICML 2026"
  - "图像恢复"
  - "非平衡最优传输"
  - "无须配对学习"
  - "反演问题"
  - "图像去模糊"
  - "超分辨率"
---

# UOTIP：无须配对的反演问题的非平衡最优传输映射

**会议**: ICML 2026  
**arXiv**: [2605.21094](https://arxiv.org/abs/2605.21094)  
**代码**: 待确认  
**领域**: 图像恢复 / 最优传输  
**关键词**: 非平衡最优传输, 无须配对学习, 反演问题, 图像去模糊, 超分辨率

## 一句话总结
提出 UOTIP 方法——通过非平衡最优传输（UOT）框架将无须配对的图像反演问题表述为从有噪声测量分布到干净信号分布的映射学习，通过引入似然成本函数和二次项成本获得鲁棒性和理论保证。

## 研究背景与动机

**领域现状**：图像反演问题（去模糊、超分辨率、HDR 重建等）通常采用贝叶斯框架，通过学习先验分布和似然函数求解。近年采用生成模型（GAN、VAE、扩散模型、最优传输）学习数据先验。

**现有痛点**：标准最优传输（OT）方法在无须配对设置下存在三个主要问题——（1）要求严格边际分布匹配，对数据不平衡敏感；（2）多级噪声场景下无法自适应处理；（3）线性反演问题中 ill-posed 特性可能导致 OT 映射不存在或非唯一。

**核心矛盾**：图像反演需同时满足两目标——保证重建结果符合干净信号分布（先验保真度）与最大化对特定测量的似然（数据保真度）。标准 OT 框架虽能保证前者但需精心设计成本函数才能融合后者；Monge 问题的非凸性与 ill-posedness 威胁理论存在性。

**本文目标**：设计无须配对的反演问题求解器，兼具理论保证与实践鲁棒性。

**切入角度**：非平衡最优传输（UOT）通过放松边际约束能自然处理分布不匹配、类别不平衡等实际问题。关键洞察：将反演问题成本函数设计为似然成本 + 二次项成本，其中似然项编码 MAP 估计的数据保真性，二次项满足扭转条件以保证映射存在唯一性。

**核心 idea**：用 UOT 将反演问题表述为学习分布间映射，通过混合成本函数实现全局无须配对的 MAP 估计。

## 方法详解

### 整体框架

UOTIP 要解的是无须配对的图像反演：只有一堆有噪声测量 $\mathbf{y} = \mathcal{A}(\mathbf{x}) + \mathbf{n}$ 和一堆与之不对应的干净信号，目标是恢复 $\mathbf{x}$。它把这件事重写成学一个映射 $T: \mathcal{Y} \to \mathcal{X}$，让传过去的测量分布对齐干净分布（$T_\# \mu \approx \nu$），并用非平衡最优传输（UOT）放松这个对齐约束。关键是成本函数的设计——似然项负责"贴合这次测量"、二次项负责"理论上映射存在唯一"，二者叠在一起后再用半对偶对抗优化求解。

### 关键设计

**1. 似然成本函数：把 MAP 估计塞进 OT 成本**

标准 OT 的二次成本只看像素距离，对反演问题的物理结构（退化算子 $\mathcal{A}$、噪声模型）一无所知。UOTIP 的切入点是借 MAP 估计 $\mathbf{x}_{MAP}(\mathbf{y}_0) = \arg\min_{\mathbf{x}}[-\log p(\mathbf{y}_0|\mathbf{x})-\log p(\mathbf{x})]$ 的结构：在高斯噪声假设下负对数似然 $-\log p(\mathbf{y}|\mathbf{x})$ 正好是 $\|\mathcal{A}(\mathbf{x})-\mathbf{y}\|_2^2$，于是直接拿它当 OT 成本 $c_l(\mathbf{y},\mathbf{x})=\|\mathcal{A}(\mathbf{x})-\mathbf{y}\|_2^2$。这样一来，最小化 OT 传输成本就等价于最小化负对数后验 $-\log p(\mathbf{x}|\mathbf{y})$，而 OT 的边际约束 $T_\#\mu=\nu$ 又自动把干净信号先验 $p(\mathbf{x})$ 灌了进来——数据项由似然成本管，先验项由分布对齐管，整个 MAP 估计被严格地搬到了 OT 框架里。

**2. 二次成本 + 非平衡松弛：补存在性、扛不平衡**

光有似然成本会出问题：线性反演本身 ill-posed，$\mathcal{A}$ 不可逆时合成成本可能不满足扭转条件（twist condition），最优传输映射既不保证存在也不保证唯一。UOTIP 补一个二次项 $c_q(\mathbf{y},\mathbf{x})=\|\mathbf{y}-\mathbf{x}\|_2^2$，合成出最终成本 $c(\mathbf{y},\mathbf{x}) = \tau(c_l + c_q)$，论文 Prop. 3.1 证明这个二次项足以让合成成本满足扭转条件，从而恢复映射的存在唯一性。另一半好处来自 UOT 把硬边际换成 f-散度软约束 $D_{\Psi_i}(\pi_0\|\mu) + D_{\Psi_i}(\pi_1\|\nu)$：边际不必严格匹配，模型可以通过缩放因子 $\Psi^*_i'$ 对源样本自动重加权，于是多级噪声、类别不平衡、分布异质这些标准 OT 会卡死的场景都能软着陆。

**3. 半对偶 UOT 求解器：用对抗迭代逼近最优映射**

直接解 Monge 形式是非凸的，难优化。UOTIP 把 UOT 的 Kantorovich 对偶展成半对偶形式，引入 c-变换 $v^c(\mathbf{y})=\inf_{\mathbf{x}}[c(\mathbf{y},\mathbf{x})-v(\mathbf{x})]$ 做凸化，然后用两个网络参数化：势函数 $v_\phi$ 当判别侧，映射 $T_\theta$ 当生成侧，且让 $T_\theta(\mathbf{y}) \in \arg\inf_{\mathbf{x}}[c(\mathbf{y},\mathbf{x})-v_\phi(\mathbf{x})]$ 显式满足 c-变换的最优性条件。两者对抗迭代，最终 $T_\theta$ 收敛到最优 UOT 映射。

## 实验关键数据

### 主实验

| 任务 | 方法 | FFHQ PSNR | FFHQ SSIM | FFHQ FID | AFHQ PSNR | AFHQ SSIM | AFHQ FID |
|------|------|-----------|-----------|----------|-----------|-----------|----------|
| Gaussian 去模糊 | NOT | 20.11 | 0.6035 | 52.901 | 19.99 | 0.5472 | 58.927 |
| | OTUR | 23.82 | 0.7106 | 24.337 | 23.91 | 0.6777 | 30.773 |
| | RCOT | 22.07 | 0.5492 | 123.692 | 22.34 | 0.5365 | 132.465 |
| | **UOTIP** | **24.06** | **0.7139** | **21.210** | **24.22** | **0.6804** | **12.566** |
| 4× 超分辨率 | NOT | 20.13 | 0.6257 | 50.066 | 20.14 | 0.5833 | 44.252 |
| | OTUR | 24.09 | 0.7243 | 22.751 | 24.71 | 0.7079 | 19.575 |
| | RCOT | 24.05 | 0.6820 | 118.776 | 25.04 | 0.7137 | 69.072 |
| | **UOTIP** | **24.35** | **0.7371** | **19.475** | **24.97** | **0.7142** | **15.939** |

### 消融与鲁棒性分析

| 配置 | Gaussian 去模糊 | HDR 重建 | 非线性去模糊 | 说明 |
|------|----------------|--------|-------------|------|
| 完整模型 (UOT+$c_l$+$c_q$) | 24.06 | 26.02 | 28.52 | 基准 |
| w/o 二次成本 | 22.18 | 24.31 | 26.74 | 去掉 $c_q$ 后 ill-posed 恶化 |
| w/o 似然成本 | 23.41 | 25.18 | 27.65 | 去掉 $c_l$ 后数据保真下降 |
| 标准 OT（边际硬约束） | 23.12 | 25.04 | 27.31 | UOT 松弛优于严格约束 |

### 关键发现
- 多级噪声处理——3 个不同噪声级训练单一模型，UOTIP 保持稳定性能（PSNR 波动 <0.5dB）。
- 类别不平衡——不同 class 分布比例下 UOT 的自适应边际匹配使其优于 OT。
- 超分辨率的实践成功——虽超分破坏扭转条件理论保证但网络归纳偏置使 UOTIP 仍达 SOTA。
- 纹理保留——定性对比显示 UOTIP 相比 OTUR（过度平滑）和 RCOT（伪影）更好保留细节。

## 亮点与洞察
- **巧妙的 MAP-OT 桥接**：似然成本函数天然对应高斯噪声的负对数似然，通过 OT 约束隐含引入数据先验——首次将 MAP 估计严格映射到 OT 框架。
- **理论保证的实用策略**：Proposition 3.1 证明二次成本 $\lambda c_q$ 确保扭转条件，使 ill-posed 问题也有唯一映射。
- **UOT 的显著优势**：通过允许边际软匹配自然处理三类实际困难——多级噪声、数据不平衡、分布异质性。

## 局限与展望
- 超分辨率任务中修改的二次成本不满足理论扭转条件，存在性/唯一性无保证。
- 成本强度参数 $\tau$ 需手工调整。
- 与 GAN 相比计算成本更高（需对势函数优化）。
- 改进：扩展扭转条件对非 L-Lipschitz 算子的适用性；设计自适应 $\tau$；结合扩散模型作为先验补充。

## 相关工作与启发
- **vs NOT**：都用神经 OT 框架，但 NOT 用标准二次成本缺乏反演问题特异性；UOTIP 引入似然成本更精准编码数据项。
- **vs OTUR**：GAN 方法纹理细节易过度平滑或扭曲；UOTIP 通过 OT 的全局最优性与 UOT 的分布松弛更稳健。
- **vs 标准贝叶斯反演**：传统方法依赖手工先验；UOTIP 通过数据驱动学习分布先验。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  首次系统将 UOT 与反演问题结合，引入似然成本与扭转条件理论保证。
- 实验充分度: ⭐⭐⭐⭐  覆盖 4 种反演任务与多数据集 + 消融与鲁棒性分析。
- 写作质量: ⭐⭐⭐⭐⭐  论文脉络清晰从 MAP 原理出发循序引入。
- 价值: ⭐⭐⭐⭐⭐  统一反演问题与最优传输两个领域，理论与应用兼具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)
- [\[ICML 2026\] Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging](phy-cosf_physics-guided_continuous_spectral_fields_reconstruction_and_super-reso.md)
- [\[ICML 2026\] Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)
- [\[ICML 2026\] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)
- [\[ICML 2026\] One-shot Conditional Sampling: MMD meets Nearest Neighbors](one-shot_conditional_sampling_mmd_meets_nearest_neighbors.md)

</div>

<!-- RELATED:END -->
