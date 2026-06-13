---
title: >-
  [论文解读] Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing
description: >-
  [CVPR 2025][图像生成][逆问题求解] 本文提出解耦退火后验采样（DAPS），通过在扩散采样过程中解耦相邻步骤的样本依赖关系，允许大幅度的非局部跳跃来修正早期采样错误，在非线性逆问题（如相位恢复）上大幅超越现有方法。 领域现状：基于扩散模型的贝叶斯逆问题求解已成为主流方法。现有方法（如 DPS、DDRM、DDNM、…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "逆问题求解"
  - "后验采样"
  - "噪声退火"
  - "扩散模型"
  - "图像恢复"
---

# Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing

**会议**: CVPR 2025  
**arXiv**: [2407.01521](https://arxiv.org/abs/2407.01521)  
**代码**: [https://github.com/zhangbingliang2019/DAPS](https://github.com/zhangbingliang2019/DAPS)  
**领域**: 扩散模型  
**关键词**: 逆问题求解, 后验采样, 噪声退火, 扩散模型, 图像恢复

## 一句话总结

本文提出解耦退火后验采样（DAPS），通过在扩散采样过程中解耦相邻步骤的样本依赖关系，允许大幅度的非局部跳跃来修正早期采样错误，在非线性逆问题（如相位恢复）上大幅超越现有方法。

## 研究背景与动机

**领域现状**：基于扩散模型的贝叶斯逆问题求解已成为主流方法。现有方法（如 DPS、DDRM、DDNM、DiffPIR）在扩散采样的每一步中修改反向 SDE，将测量约束融入去噪过程，从而从后验分布 $p(\mathbf{x}_0|\mathbf{y})$ 中采样。

**现有痛点**：现有方法在每个去噪步骤中从 $p(\mathbf{x}_t|\mathbf{x}_{t+\Delta t}, \mathbf{y})$ 采样，由于反向 SDE 的步长 $\Delta t$ 很小，$\mathbf{x}_t$ 和 $\mathbf{x}_{t+\Delta t}$ 被强制接近。这意味着每步只能做局部修正——可以修正前一步的小错误，但无法修正需要大幅修改的全局错误。

**核心矛盾**：采样步骤间的强耦合限制了解空间的探索范围。在非线性逆问题（如相位恢复）中，测量函数高度非线性，早期步骤容易陷入错误模式，后续步骤无法跳出，最终收敛到与测量一致但概率很低的错误解。

**本文目标** 如何在保证采样质量的同时，允许连续采样步骤之间有足够大的变化来修正全局错误？

**切入角度**：作者意识到关键在于不需要求解连续的反向 SDE，而是可以直接从时间边缘分布 $p(\mathbf{x}_t|\mathbf{y})$ 中采样。通过引入中间变量 $\mathbf{x}_0$ 将 $\mathbf{x}_t$ 和 $\mathbf{x}_{t+\Delta t}$ 解耦——给定 $\mathbf{x}_0$ 后两者条件独立。

**核心 idea**：用解耦的噪声退火过程替代反向 SDE 求解，通过"逆扩散→Langevin 采样→前向加噪"的三步循环递归采样时间边缘分布。

## 方法详解

### 整体框架

DAPS 不求解反向 SDE，而是递归地从时间边缘分布 $p(\mathbf{x}_t|\mathbf{y})$ 采样，噪声水平逐步退火到零。给定噪声调度和时间离散化 $\{t_i\}$，从 $\mathbf{x}_T \sim \mathcal{N}(0, \sigma_T^2 I)$ 开始，迭代采样 $\mathbf{x}_{t_i}$：（1）从 $\mathbf{x}_{t_{i+1}}$ 通过 ODE 求解得到 $\hat{\mathbf{x}}_0$；（2）用 Langevin 动力学从 $p(\mathbf{x}_0|\mathbf{x}_t, \mathbf{y})$ 采样得到 $\mathbf{x}_{0|y}$；（3）对 $\mathbf{x}_{0|y}$ 加噪到 $\sigma_{t_i}$ 得到 $\mathbf{x}_{t_i}$。最终 $\sigma_t \to 0$ 得到后验样本。

### 关键设计

1. **解耦噪声退火过程**:

    - 功能：允许连续采样步骤间的大幅变化，打破传统方法的局部约束
    - 核心思路：核心命题证明，若 $\mathbf{x}_{t_1}$ 服从时间边缘分布 $p(\mathbf{x}_{t_1}|\mathbf{y})$，那么 $\mathbf{x}_{t_2} \sim \mathbb{E}_{\mathbf{x}_0 \sim p(\mathbf{x}_0|\mathbf{x}_{t_1}, \mathbf{y})}[\mathcal{N}(\mathbf{x}_0, \sigma_{t_2}^2 I)]$ 也服从 $p(\mathbf{x}_{t_2}|\mathbf{y})$。这使得从任意噪声水平跳到另一个噪声水平成为可能，$\mathbf{x}_t$ 和 $\mathbf{x}_{t+\Delta t}$ 通过 $\mathbf{x}_0$ 解耦，条件独立。
    - 设计动机：传统方法的耦合采样（$\mathbf{x}_t$ 紧跟 $\mathbf{x}_{t+\Delta t}$）在 2D 合成实验中明确展示了收敛到错误解的问题，而 DAPS 允许轨迹上的点有显著变化，能逃离错误模式。

2. **Langevin 动力学采样 $p(\mathbf{x}_0|\mathbf{x}_t, \mathbf{y})$**:

    - 功能：在给定当前噪声样本和测量的条件下采样干净信号
    - 核心思路：利用贝叶斯规则将 $p(\mathbf{x}_0|\mathbf{x}_t, \mathbf{y}) \propto p(\mathbf{x}_0|\mathbf{x}_t) p(\mathbf{y}|\mathbf{x}_0)$。其中 $p(\mathbf{x}_0|\mathbf{x}_t)$ 用高斯近似 $\mathcal{N}(\hat{\mathbf{x}}_0(\mathbf{x}_t), r_t^2 I)$，$\hat{\mathbf{x}}_0$ 通过概率流 ODE 求解估计。Langevin 更新包含两项梯度：先验梯度（拉向 $\hat{\mathbf{x}}_0$）和似然梯度（拉向满足测量 $\mathbf{y}$ 的方向），加上随机噪声项。
    - 设计动机：高斯近似虽不如扩散得分估计精确，但实验表明二者效果相当且高斯近似计算效率高得多。Langevin 的计算开销主要来自评估测量函数 $\mathcal{A}$，对于图像恢复任务远快于评估扩散模型。

3. **LatentDAPS 潜空间扩展**:

    - 功能：将 DAPS 扩展到预训练的潜空间扩散模型（LDM），支持高分辨率图像恢复
    - 核心思路：根据潜空间扩散的概率图模型进行分解，Langevin 更新在潜空间进行：先验梯度拉向 $\hat{\mathbf{z}}_0$，似然梯度通过解码器 $\mathcal{D}$ 将测量约束传回潜空间，即 $\nabla_{\mathbf{z}_0} \log p(\mathbf{y}|\mathcal{D}(\mathbf{z}_0))$。
    - 设计动机：LDM 可利用大规模预训练模型（如 Stable Diffusion），扩展到高分辨率任务。LatentDAPS 是 DAPS 的自然延伸，比现有潜空间方法（PSLD、ReSample）性能更好。

### 损失函数 / 训练策略

DAPS 是一个推理时方法，不需要额外训练，直接利用预训练的扩散模型。DAPS-1K（线性任务）使用 44 步 ODE + 250 步退火，DAPS-4K（非线性任务）使用 10 步 ODE + 400 步退火。Langevin 步数：DAPS 每次 100 步，LatentDAPS 每次 50 步。不同任务单独调节学习率。

## 实验关键数据

### 主实验

| 任务 | 数据集 | 指标 | DAPS | 之前SOTA | 提升 |
|------|--------|------|------|----------|------|
| 相位恢复 | FFHQ 256 | PSNR↑ | 30.72 | 28.74 (DPS) | +1.98 |
| HDR 恢复 | FFHQ 256 | PSNR↑ | 27.12 | 22.73 (DPS) | +4.39 |
| 超分辨率 4× | FFHQ 256 | PSNR↑ | 29.07 | 28.66 (DCDP) | +0.41 |
| 高斯去模糊 | FFHQ 256 | PSNR↑ | 29.19 | 28.63 (DCDP) | +0.56 |
| 随机 inpainting | FFHQ 256 | PSNR↑ | 31.12 | 30.69 (DCDP) | +0.43 |
| CS-MRI | - | PSNR↑ | 31.49 | 28.79 (之前SOTA) | +2.70 |

DAPS 在几乎所有线性和非线性任务上均取得最优结果，尤其在非线性任务上优势巨大。

### 消融实验

| 配置 | PSNR (相位恢复) | 说明 |
|------|----------------|------|
| DAPS (高斯近似) | 30.72 | 默认配置 |
| DAPS (扩散得分估计) | ~30.7 | 效果相当但计算量大得多 |
| DPS baseline | 28.74 | 耦合采样限制修正能力 |
| DAPS-1K (NFE=1000) | 适用线性任务 | 效率与质量平衡 |
| DAPS-4K (NFE=4000) | 适用非线性任务 | 更充分的退火 |

### 关键发现

- 非线性逆问题是 DAPS 的最大优势场景，相位恢复和 HDR 上分别领先 DPS 1.98 和 4.39 dB
- 高斯近似与扩散得分估计效果相当，但效率高得多，验证了近似策略的实用性
- DAPS 在约 100 次神经网络评估下就能取得好结果，效率-质量平衡优于竞争方法
- 2D 合成实验直观展示了 DAPS 的轨迹变化远大于 DPS，能更好地逼近真实后验

## 亮点与洞察

- **解耦思想非常优雅**：通过引入中间变量 $\mathbf{x}_0$ 将采样步骤解耦，数学上干净（一个命题就说清楚了），实际效果也好。这种"打断依赖链+退火"的思路可以推广到其他序列采样问题
- **与优化方法的联系**：DAPS 在特定参数设置下退化为 ReSample 等优化方法的 MAP 估计，但 DAPS 本质是后验采样而非点估计，理论上更完备
- **MCMC 引入的开销很小**：Langevin 步骤的计算主要是评估测量函数而非扩散模型，对于图像恢复任务几乎零额外开销

## 局限与展望

- 对于每个任务需要手动调节 Langevin 学习率和先验/似然的方差参数 $r_t$ 和 $\beta_y$
- 非线性任务需要 4000 次 NFE，虽然比直觉中少但仍有优化空间
- 高斯近似在噪声水平很高时可能不够精确，但实验中没有成为瓶颈
- 作者在离散扩散模型上做了初步探索（分类数据），可进一步推广到其他数据模态

## 相关工作与启发

- **vs DPS**: DPS 在反向 SDE 上加似然梯度约束，每步局部修正；DAPS 完全解耦，允许全局跳跃。DPS 在线性任务上尚可但非线性任务崩溃
- **vs ReSample/DiffPIR**: 这些方法也交替去噪-优化-重采样，但本质是 MAP 估计；DAPS 是后验采样，当 $\beta_y \to 0$ 时退化为它们
- **vs DDRM/DDNM**: 这些方法通过 SVD 在频域操作，仅限线性逆问题，无法处理非线性测量

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 解耦噪声退火的思路简洁深刻，理论有支撑，是逆问题求解范式的突破
- 实验充分度: ⭐⭐⭐⭐⭐ 8种逆问题任务、像素+潜空间、两个数据集、与众多 baseline 对比
- 写作质量: ⭐⭐⭐⭐⭐ 论文写作清晰，2D 合成实验的可视化特别直观，理论推导完整
- 价值: ⭐⭐⭐⭐⭐ 对扩散逆问题领域有重要推进，尤其在非线性逆问题上开辟了新可能

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](../../NeurIPS2025/image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[ICML 2025\] Integrating Intermediate Layer Optimization and Projected Gradient Descent for Solving Inverse Problems with Diffusion Models](../../ICML2025/image_generation/integrating_intermediate_layer_optimization_and_projected_gradient_descent_for_s.md)
- [\[AAAI 2026\] Constrained Particle Seeking: Solving Diffusion Inverse Problems with Just Forward Passes](../../AAAI2026/image_generation/constrained_particle_seeking_solving_diffusion_inverse_problems_with_just_forwar.md)
- [\[CVPR 2025\] Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification](divide_and_conquer_heterogeneous_noise_integration_for_diffusion-based_adversari.md)

</div>

<!-- RELATED:END -->
