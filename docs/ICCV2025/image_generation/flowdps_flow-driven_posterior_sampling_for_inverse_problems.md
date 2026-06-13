---
title: >-
  [论文解读] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems
description: >-
  [ICCV 2025][图像生成][Flow Matching] FlowDPS 通过推导 Flow 模型的 Tweedie 公式将 Flow ODE 分解为干净图像估计和噪声估计两个分量，在干净图像分量中注入似然梯度、在噪声分量中引入随机噪声，实现了基于 Flow 模型的后验采样逆问题求解，在 SD3.0 上的四种线性逆问题中超越所有已有方法。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Flow Matching"
  - "后验采样"
  - "逆问题"
  - "Tweedie公式"
  - "扩散模型"
---

# FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems

**会议**: ICCV 2025  
**arXiv**: [2503.08136](https://arxiv.org/abs/2503.08136)  
**代码**: [GitHub](https://github.com/FlowDPS-Inverse/FlowDPS)  
**领域**: 扩散模型/逆问题求解  
**关键词**: Flow Matching, 后验采样, 逆问题, Tweedie公式, Stable Diffusion 3.0

## 一句话总结
FlowDPS 通过推导 Flow 模型的 Tweedie 公式将 Flow ODE 分解为干净图像估计和噪声估计两个分量，在干净图像分量中注入似然梯度、在噪声分量中引入随机噪声，实现了基于 Flow 模型的后验采样逆问题求解，在 SD3.0 上的四种线性逆问题中超越所有已有方法。

## 研究背景与动机

逆问题（super-resolution、去模糊等）的目标是从退化观测 $\mathbf{y} = \mathbf{A}\mathbf{x}_0 + \mathbf{n}$ 中恢复原始信号。该问题本质上是病态的，需要先验知识约束解空间。

**扩散模型方案的成功**：DPS、DDRM 等方法利用扩散模型学到的图像先验，通过引导采样轨迹实现后验采样。在 LDM 框架下也衍生出 PSLD、ReSample、DAPS 等方法。

**Flow 模型的兴起与挑战**：Flow Matching 已成为生成建模的主流范式（如 SD3.0 基于 Flow），但在逆问题求解方面缺乏严格的理论分析和有效方法。现有尝试如 FlowChef 和 PnP-Flow 都是启发式方法，缺乏与后验采样的理论联系。

核心矛盾在于：Flow 模型的 ODE 结构不同于扩散模型的 SDE 结构，现有扩散逆问题方法不能直接迁移到 Flow 框架中。

本文的核心 idea：**推导 Flow 版的 Tweedie 公式，将 Flow ODE 分解为去噪和加噪两个分量，从而让后验采样在 Flow 框架中自然实现**。

## 方法详解

### 整体框架

FlowDPS 在 Flow 模型的反向采样过程中：
1. 用 Tweedie 公式从速度场估计干净图像 $\hat{\mathbf{x}}_{0|t}$ 和噪声 $\hat{\mathbf{x}}_{1|t}$
2. 在干净图像估计上加入数据一致性梯度 → $\tilde{\mathbf{x}}_{0|t}$
3. 在噪声估计上混入随机噪声 → $\tilde{\mathbf{x}}_{1|t}$
4. 两个分量按系数组合得到下一步采样点

### 关键设计

1. **Flow 版 Tweedie 公式**:

    - 功能：从训练好的速度场 $v_t(\mathbf{x}_t)$ 推导干净图像和噪声的条件期望估计
    - 核心思路：对仿射条件流 $\mathbf{x}_t = a_t \mathbf{x}_0 + b_t \mathbf{x}_1$，速度场的边际形式为：
    $v_t(\mathbf{x}) = \dot{a}_t \mathbb{E}[\mathbf{x}_0|\mathbf{x}_t] + \dot{b}_t \mathbb{E}[\mathbf{x}_1|\mathbf{x}_t]$
      由此得到 Tweedie 公式：
    $\hat{\mathbf{x}}_{0|t} = \left[a_t - \dot{a}_t \frac{b_t}{\dot{b}_t}\right]^{-1}\left(\mathbf{x}_t - \frac{b_t}{\dot{b}_t} v_t(\mathbf{x}_t)\right)$
    - 设计动机：这个分解揭示了 Flow ODE 的内在结构与扩散模型的几何相似性，为后验采样奠定理论基础

2. **后验速度场推导**:

    - 功能：将观测约束融入 Flow ODE 的采样过程
    - 核心思路：利用 Bayes 规则，后验速度场为：
    $v_t(\mathbf{x}_t|\mathbf{y}) = v_t(\mathbf{x}_t) - \zeta_t \nabla_{\mathbf{x}_t} \log p_t(\mathbf{y}|\mathbf{x}_t)$
      结合流形投影假设简化 Jacobian 计算，最终采样公式为：
    $\mathbf{x}_{t+dt} = C_1(t)\tilde{\mathbf{x}}_{0|t} + C_2(t)\tilde{\mathbf{x}}_{1|t}$
      其中 $\tilde{\mathbf{x}}_{0|t} = \hat{\mathbf{x}}_{0|t} - \beta_t \nabla_{\hat{\mathbf{x}}_{0|t}} \log p(\mathbf{y}|\hat{\mathbf{x}}_{0|t})$ 实现数据一致性
    - 设计动机：将似然梯度准确地注入到干净图像分量（而非整体 $\mathbf{x}_t$），使得梯度引导更加精准且具有自适应步长 $\beta_t$

3. **随机噪声注入与 Latent FlowDPS**:

    - 功能：在噪声分量中混合确定性估计和随机噪声，并扩展到隐空间 Flow 模型
    - 核心思路：
    $\tilde{\mathbf{x}}_{1|t} = \sqrt{1-\eta_t}\hat{\mathbf{x}}_{1|t} + \sqrt{\eta_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$
      隐空间中用多步共轭梯度优化 $\hat{\mathbf{z}}_{0|t}(\mathbf{y})$ 并通过插值保持轨迹稳定性
    - 设计动机：随机噪声类似于 DDIM→DDPM 的推广，增强多样性和鲁棒性；隐空间操作实现高分辨率推理

### 损失函数 / 训练策略

FlowDPS 是**零样本方法**，不需要额外训练。直接利用预训练的 Flow 模型（SD3.0）作为先验。关键超参数：
- 随机噪声比例 $\eta_t$ 控制确定性/随机性平衡
- 步长 $\beta_t = \frac{\zeta_t}{a_t}\frac{dt}{C_1(t)}$ 自适应调整（前期大、后期趋零）
- 插值参数 $\gamma$ 控制数据一致性更新的强度

## 实验关键数据

### 主实验

| 数据集/任务 | 指标 | FlowDPS | FlowChef | ReSample | 提升 |
|--------|------|------|----------|------|------|
| AFHQ 768² SR×12 Avg | FID↓ | **16.85** | 21.14 | 41.17 | -4.29 vs FlowChef |
| AFHQ 768² SR×12 Avg | LPIPS↓ | **0.198** | 0.249 | 0.300 | -0.051 |
| AFHQ 768² SR×12 Bic | FID↓ | **15.71** | 21.31 | 39.94 | -5.60 |
| AFHQ 768² Deblur Gauss | FID↓ | **23.00** | 36.46 | 44.22 | -13.46 |
| FFHQ 768² SR×12 Avg | FID↓ | **33.78** | 41.50 | 102.7 | -7.72 |
| FFHQ 768² SR×12 Bic | FID↓ | **33.75** | 39.75 | 102.4 | -5.99 |
| FFHQ 768² Deblur Motion | FID↓ | **38.14** | 104.7 | 95.16 | -66.56 |

### 消融实验

| 配置 | PSNR | FID | 说明 |
|------|---------|------|------|
| FlowDPS (ODE, 无随机噪声) | 基准 | 基准 | 纯确定性后验采样 |
| FlowDPS (SDE, 有随机噪声) | 更高 | **更低** | 随机噪声改善 FID |
| FlowDPS w/ 多步 CG | **最高** | **最低** | 多步梯度提升数据一致性 |
| FlowDPS w/ 插值 | 最优平衡 | 最优 | 轨迹插值防止偏离 |
| FlowChef (直接引导 $\mathbf{x}_t$) | 较低 | 较高 | 缺乏分解和自适应步长 |

### 关键发现

- FlowDPS 在所有四种线性逆问题（SR-Avgpool、SR-Bicubic、高斯去模糊、运动去模糊）上均大幅领先
- 与 FlowChef 相比，分解式梯度注入（仅修改 $\hat{\mathbf{x}}_{0|t}$）比直接修改 $\mathbf{x}_t$ 效果好得多
- $\beta_t$ 的自适应行为：前期强调数据一致性，后期回归生成先验，这对高质量重建至关重要
- 从 PSLD（FID > 140）到 FlowDPS（FID < 40）的巨大提升说明 Flow 框架比传统 LDM 逆问题方法更有效

## 亮点与洞察

1. **理论贡献扎实**：Flow 版 Tweedie 公式是完整的理论推导，揭示了 Flow ODE 与扩散模型在几何结构上的深层联系
2. **零样本即插即用**：无需训练，直接利用 SD3.0，具有极高的实用价值
3. **分解思想的优雅性**：将后验采样分解为"干净估计 + 数据一致性"和"噪声估计 + 随机性注入"，简洁且有效
4. **自适应步长 $\beta_t$** 自然地从理论推导中浮现，避免了手工调参

## 局限与展望

- 目前仅验证了线性逆问题（$\mathbf{y} = \mathbf{A}\mathbf{x}_0 + \mathbf{n}$），非线性逆问题的扩展有待探索
- 隐空间操作需要通过 decoder 计算梯度，增加了显存开销
- 超参数 $\gamma$（插值系数）和 CG 步数仍需根据任务调整
- 多步 CG 优化增加了推理时间

## 相关工作与启发

- 本文建立了 Flow 模型与扩散模型在逆问题求解上的理论桥梁，为未来统一框架提供了可能
- Tweedie 公式的 Flow 版本可以推广到其他 Flow-based 条件生成任务（如图像编辑、修复等）
- 分解后验采样的思路可以借鉴到视频、3D 等更广泛的逆问题场景中

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Flow 版 Tweedie 公式和 ODE 分解是该领域的重要理论贡献
- 实验充分度: ⭐⭐⭐⭐ 跨三个数据集四个任务的全面对比，但缺少非线性逆问题实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，从背景到方法到实验逻辑清晰，图解直观
- 价值: ⭐⭐⭐⭐⭐ 为 Flow 模型的逆问题求解提供了原则性框架，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICCV 2025\] LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling](ld-rps_zero-shot_unified_image_restoration_via_latent_diffusion_recurrent_poster.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](../../NeurIPS2025/image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)
- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](../../ICML2026/image_generation/saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[ICML 2026\] Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems](../../ICML2026/image_generation/zeroth-order_non-log-concave_sampling_with_variance_reduction_and_applications_t.md)

</div>

<!-- RELATED:END -->
