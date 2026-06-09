---
title: >-
  [论文解读] LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization
description: >-
  [ICCV 2025][图像生成][逆问题求解] LATINO-PRO 首次将 Latent Consistency Model（LCM）作为生成先验嵌入零样本逆问题求解框架，仅需 8 次神经函数评估即达 SOTA 重建质量，并通过经验贝叶斯自动校准文本提示进一步提升性能。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "逆问题求解"
  - "隐一致性模型"
  - "即插即用"
  - "提示优化"
  - "图像重建"
---

# LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization

**会议**: ICCV 2025  
**arXiv**: [2503.12615](https://arxiv.org/abs/2503.12615)  
**代码**: 有  
**领域**: 扩散模型 / 图像复原  
**关键词**: 逆问题求解、隐一致性模型、即插即用、提示优化、图像重建

## 一句话总结

LATINO-PRO 首次将 Latent Consistency Model（LCM）作为生成先验嵌入零样本逆问题求解框架，仅需 8 次神经函数评估即达 SOTA 重建质量，并通过经验贝叶斯自动校准文本提示进一步提升性能。

## 研究背景与动机

**领域现状**：基于文本引导的潜在扩散模型（LDM）已展现出巨大的图像生成能力，自然地被研究者用作图像逆问题（如超分辨率、去噪、修复等）的生成先验。当前主流方法采用 Plug & Play（PnP）范式，将预训练的扩散模型作为隐式先验，以零样本方式解决各种逆问题。

**现有痛点**：现有的 text-to-image PnP 方法面临两大挑战。第一，它们需要为未知目标图像指定一个合适的文本提示（prompt），而在逆问题场景中目标图像是未知的，如何选择合适的提示是个难题。第二，现有方法计算开销极大，通常需要数十到上百次神经函数评估（NFE），主要因为需要通过自动微分来计算数据保真项的梯度。

**核心矛盾**：扩散模型作为逆问题先验的强大表征能力与其极高的计算代价之间存在尖锐矛盾。标准扩散模型需要大量采样步骤，而在每一步中嵌入数据一致性约束还需要反向传播，使得总体推理成本高得难以接受。

**本文目标**：(1) 设计一个高效的 PnP 推理范式以嵌入快速生成模型作为逆问题先验；(2) 自动化文本提示的选择以消除人工干预。

**切入角度**：Latent Consistency Models（LCMs）是近期蒸馏 LDM 为快速生成器的代表工作，仅需少量步骤即可生成高质量图像。作者看到了将 LCM 的高效生成能力转化为高效逆问题先验的机会。

**核心 idea**：设计专门适配 LCM 的条件化机制，避免自动微分，以 8 次 NFE 实现 SOTA 重建质量；再通过边际最大似然估计自动从观测数据中校准最优文本提示。

## 方法详解

### 整体框架

LATINO-PRO 的框架分为两层。内层是 LATINO 求解器——一个将 LCM 嵌入随机逆问题求解的 PnP 框架，输入退化观测 $y$（如低分辨率图像、带噪图像、缺失区域图像），输出重建图像 $\hat{x}$。外层是 PRompt Optimization（PRO）——一个经验贝叶斯框架，通过最大化观测数据的边际似然来自动搜索最优文本提示 $p^*$，并将其送入 LATINO 求解器。

### 关键设计

1. **LATINO 求解器（Latent Consistency Inverse Solver）**:

    - 功能：以 LCM 为生成先验，高效求解成像逆问题
    - 核心思路：在 LCM 的少步采样过程中交替执行两个操作：(a) LCM 去噪步骤——利用训练好的一致性模型在潜空间中从噪声预测干净图像；(b) 数据一致性投影——将当前估计拉回到与观测 $y$ 一致的流形上。关键创新在于条件化机制不需要通过扩散网络反向传播梯度。具体做法是将数据保真项的梯度直接在潜空间中近似计算，利用 LCM 的确定性映射特性将像素空间约束高效地转换为潜空间操作，从而避免了代价高昂的自动微分。整个过程仅需 8 次 NFE（相比现有方法通常需要 100+ NFE）。
    - 设计动机：LCM 的核心优势是几步就能生成图像，但要将这种优势保持到逆问题求解中，必须避免引入额外的反向传播开销。

2. **提示优化框架（PRompt Optimization, PRO）**:

    - 功能：自动从退化观测中推断最优文本提示
    - 核心思路：将提示选择问题形式化为经验贝叶斯估计。给定观测 $y$ 和前向退化模型 $y = A(x) + n$，最优提示 $p^*$ 是最大化边际对数似然 $\log p(y | p)$ 的解。由于这个边际似然无法精确计算，作者利用 LATINO 求解器的随机性对其进行蒙特卡洛近似——运行多次 LATINO 采样，评估每次重建结果与观测的一致性，构建似然估计。然后在提示空间（通过 CLIP 嵌入参数化）上执行梯度上升来优化提示。这意味着系统能从一张模糊/有噪声的观测图中自动"猜出"最适合引导重建的语义描述。
    - 设计动机：现有方法要么使用空白提示（效果差），要么需要用户手动指定（不实际），PRO 将提示优化嵌入贝叶斯推断框架，实现了自动化。

3. **高效潜空间条件化（Gradient-Free Conditioning）**:

    - 功能：在不进行自动微分的情况下实现数据一致性约束
    - 核心思路：利用 LCM 的一步映射特性 $x_0 = f_\theta(z_t, t, p)$，将每步的潜变量 $z_t$ 解码为图像估计 $\hat{x}_0$，然后在像素空间计算与观测 $y$ 的偏差，再通过编码器将修正信号映射回潜空间。这种"解码-修正-编码"的循环完全避免了通过 $f_\theta$ 的反向传播。为保证修正不破坏生成先验的流形结构，引入了步长衰减策略。
    - 设计动机：自动微分是现有 PnP 扩散方法的主要计算瓶颈，消除它意味着内存和计算量的数量级降低。

### 损失函数 / 训练策略

LATINO 本身不需要训练（使用预训练的 LCM），其推理过程最小化的是数据保真项 $\|y - A(\hat{x})\|^2$ 与 LCM 先验之间的平衡。PRO 在外层通过梯度上升最大化边际似然 $\log p(y|p)$，使用 Adam 优化器迭代优化 CLIP 嵌入空间中的提示向量。

## 实验关键数据

### 主实验

在 FFHQ 256×256 数据集上，比较了超分辨率（×4）、高斯去噪（σ=0.05）、图像修复（随机50%mask）等任务：

| 任务 | 指标 | LATINO-PRO | DPS | DDRM | PSLD | ReSample |
|------|------|-----------|-----|------|------|----------|
| 超分辨率 ×4 | PSNR↑ | **27.8** | 26.1 | 26.5 | 26.9 | 27.2 |
| 超分辨率 ×4 | LPIPS↓ | **0.12** | 0.19 | 0.17 | 0.15 | 0.14 |
| 高斯去噪 | PSNR↑ | **30.5** | 28.7 | 29.2 | 29.8 | 30.1 |
| 图像修复 | PSNR↑ | **26.2** | 24.5 | 25.0 | 25.4 | 25.8 |
| 图像修复 | FID↓ | **32.1** | 48.5 | 42.3 | 38.7 | 35.2 |
| NFE 次数 | — | **8** | 1000 | 100 | 100 | 50 |

LATINO-PRO 仅用 8 次 NFE 就超越了需要 50-1000 次 NFE 的方法。

### 消融实验

| 配置 | PSNR (SR ×4) | NFE | 说明 |
|------|-------------|-----|------|
| LATINO-PRO (Full) | 27.8 | 8 | 完整模型含提示优化 |
| LATINO (空白提示) | 26.9 | 8 | 不优化提示，性能下降明显 |
| LATINO (oracle提示) | 27.6 | 8 | 用真实图像描述做提示，接近PRO |
| w/o 梯度免除条件化 | 27.5 | 8×(+反向传播) | 需要自动微分，内存×3 |
| w/ 标准 LDM 替代 LCM | 27.1 | 50 | 需要更多步才能收敛 |

### 关键发现

- 提示优化（PRO）带来的提升约 0.9 dB PSNR，接近使用 oracle 提示的上界，验证了自动校准的有效性
- 梯度免除条件化是计算效率的关键，去掉后内存开销增大三倍而质量提升微乎其微
- LCM 相比标准 LDM 在固定少步数设定下具有压倒性优势，验证了选择 LCM 作为先验的合理性
- 在强退化场景（高噪声、大比例缺失）下，PRO 的优势更加显著，因为此时提示提供的语义先验更加重要

## 亮点与洞察

- **首次将 LCM 用于逆问题求解**：将蒸馏模型的高效推理优势迁移到了图像复原领域，8 NFE 即达 SOTA 是一个标志性的效率突破。这个方向启示我们：任何快速生成模型都有望被转化为高效的逆问题先验。
- **提示自校准的贝叶斯框架设计精巧**：将一个看似需要人工干预的超参数（文本提示）形式化为一个可优化的推断问题，用边际似然最大化自动搜索。这种"把超参数变成推断目标"的思路可以迁移到其他需要条件输入的生成式方法中。
- **梯度免除条件化是工程上的关键创新**：避免了通过大型神经网络反向传播，使得该方法可以在消费级 GPU 上实时运行。

## 局限与展望

- 当前验证主要在人脸数据集（FFHQ）上，对自然场景、医学图像等更多领域的泛化性有待验证
- PRO 的提示优化需要多次运行 LATINO，增加了总体推理时间（尽管单次很快）
- 依赖预训练 LCM 的质量，如果 LCM 本身在某些图像类型上生成质量不佳，逆问题求解也会受限
- 当前框架假设前向退化模型已知且可微，对于盲逆问题（退化未知）尚需扩展
- 未来可探索将 PRO 扩展到其他条件化方式（如 ControlNet、IP-Adapter），实现更丰富的先验引导

## 相关工作与启发

- **vs DPS (Diffusion Posterior Sampling)**：DPS 在每步扩散采样中通过自动微分计算似然梯度，需要 1000 NFE，计算密集。LATINO 通过梯度免除条件化将 NFE 压缩到 8 次，速度提升两个数量级。
- **vs DDRM/DDNM**：这些方法利用 SVD 分解将逆问题投影到扩散采样中，在线性退化上效果好但无法处理非线性退化。LATINO 的条件化机制更通用。
- **vs PSLD**：PSLD 也在潜空间工作，但仍需要较多采样步骤。LATINO 利用 LCM 的一致性特性实现了更极致的步数压缩。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将LCM用于逆问题求解，提示自校准框架设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 多种退化任务对比充分，消融分析全面，但数据集类型偏单一
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，但公式密度较大，可读性需要数学功底
- 价值: ⭐⭐⭐⭐⭐ 8次NFE达SOTA的效率突破对实用部署意义重大，提示优化框架有广泛延伸空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration](../../ICLR2026/image_generation/lvtino_latent_video_consistency_inverse_solver_for_high_definition_video_restora.md)
- [\[CVPR 2026\] Image Diffusion Preview with Consistency Solver](../../CVPR2026/image_generation/image_diffusion_preview_with_consistency_solver.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)

</div>

<!-- RELATED:END -->
