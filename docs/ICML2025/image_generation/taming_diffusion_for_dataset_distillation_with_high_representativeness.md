---
title: >-
  [论文解读] Taming Diffusion for Dataset Distillation with High Representativeness (D³HR)
description: >-
  [ICML2025][图像生成][dataset distillation] 提出 D³HR 框架，通过 DDIM 反演将 VAE 潜在空间的复杂混合高斯分布映射到高正态性的噪声空间，再结合组采样策略生成高代表性的蒸馏数据集，在 CIFAR、Tiny-ImageNet、ImageNet-1K 上全面超越现有 SOTA。
tags:
  - "ICML2025"
  - "图像生成"
  - "dataset distillation"
  - "扩散模型"
  - "DDIM inversion"
  - "distribution matching"
  - "group sampling"
---

# Taming Diffusion for Dataset Distillation with High Representativeness (D³HR)

**会议**: ICML2025  
**arXiv**: [2505.18399](https://arxiv.org/abs/2505.18399)  
**代码**: [lin-zhao-resoLve/D3HR](https://github.com/lin-zhao-resoLve/D3HR)  
**领域**: 图像生成  
**关键词**: dataset distillation, diffusion models, DDIM inversion, distribution matching, group sampling

## 一句话总结

提出 D³HR 框架，通过 DDIM 反演将 VAE 潜在空间的复杂混合高斯分布映射到高正态性的噪声空间，再结合组采样策略生成高代表性的蒸馏数据集，在 CIFAR、Tiny-ImageNet、ImageNet-1K 上全面超越现有 SOTA。

## 研究背景与动机

数据集蒸馏旨在生成一个小型数据集替代原始大数据集进行训练，在高压缩率下比数据剪枝更有优势。近期基于扩散模型的方法（D4M、Minimax）因强大的生成能力成为主流，它们在 VAE 潜在空间中提取代表性潜变量，消除了对特定教师模型架构的依赖。

然而，现有扩散方法在保证蒸馏数据集的代表性方面存在三个核心问题：

**分布匹配不准确**：VAE 潜在空间的分布是多组分高斯混合分布（Lemma 3.1：每个图像对应一个独立高斯组分 $q(z_i|x_i) \sim \mathcal{N}(\mu_i, \sigma_i^2)$），正态性低，难以用简单分布精确描述。D4M 用 K-means 假设球形聚类，Minimax 用余弦相似度忽略概率密度差异，均无法准确匹配

**随机噪声导致的分布偏移**：现有方法通过 DDPM 前向过程添加随机噪声或从随机噪声出发生成蒸馏图像，噪声的随机性会破坏 VAE 空间中的结构和代表性信息，导致去噪后的潜变量发生分布偏移

**独立采样**：蒸馏数据逐个匹配原始分布的局部，缺乏对整体分布的约束，有限的 $n$ 个样本可能无法完整表征目标分布

## 方法详解

### 整体框架

D³HR 包含三个阶段：域映射（Domain Mapping）→ 分布匹配（Distribution Matching）→ 组采样（Group Sampling）。先将图像经 VAE 编码为潜变量，再通过 DDIM 反演映射到高正态性噪声空间，用高斯分布匹配该空间的分布，最后通过组采样策略选出最具代表性的子集，经 DDIM 采样和 VAE 解码生成蒸馏图像。使用预训练 DiT 作为扩散模型骨干。

### 关键设计1：DDIM 反演域映射

核心思想是将难以拟合的 VAE 潜在空间 $\mathcal{Z}_{0,\mathcal{C}}$ 映射到正态性更高的噪声空间 $\mathcal{Z}_{T,\mathcal{C}}$。对每个类别 $\mathcal{C}$ 中的潜变量 $z_0$，执行 DDIM 反演：

$$z_{t+1} = \sqrt{\frac{\alpha_{t+1}}{\alpha_t}} z_t + \sqrt{\alpha_{t+1}} \left(\sqrt{\frac{1}{\alpha_{t+1}} - 1} - \sqrt{\frac{1}{\alpha_t} - 1}\right) \varepsilon_\theta(z_t, t, \mathcal{C})$$

相比 DDPM 前向过程直接加随机噪声，DDIM 反演具有两个关键优势：
- **信息保持**：映射是确定性双射，$\mathcal{Z}_{T,\mathcal{C}}$ 与 $\mathcal{Z}_{0,\mathcal{C}}$ 一一对应，避免关键特征丢失
- **结构一致性**：映射后的潜变量保留原始空间的结构信息，保证分布对齐

实验中使用 31 步 DDIM 反演。作者证明（Lemma 4.1）当步数 $T$ 足够大时 $\mathcal{Z}_{T,\mathcal{C}}$ 可近似为高斯分布。消融实验表明 DDIM 反演比 DDPM 前向过程带来 **11.5%** 的准确率提升。

### 关键设计2：高斯分布匹配

在噪声空间中，由于各维度独立且满足高正态性，可用高斯分布精确描述。计算 $\mathcal{Z}_{T,\mathcal{C}}$ 的均值 $\mu_{T,\mathcal{C}}$ 和方差 $\sigma^2_{T,\mathcal{C}}$，构建 $\hat{\mathcal{Z}}_{T,\mathcal{C}} \sim \mathcal{N}(\mu_{T,\mathcal{C}}, \sigma^2_{T,\mathcal{C}})$，概率密度：

$$f(\hat{\mathbf{z}}_{T,\mathcal{C}}) = \prod_{i=1}^{d} \frac{1}{\sqrt{2\pi (\sigma^i_{T,\mathcal{C}})^2}} \exp\left(-\frac{(\hat{z}^i_{T,\mathcal{C}} - \mu^i_{T,\mathcal{C}})^2}{2(\sigma^i_{T,\mathcal{C}})^2}\right)$$

### 关键设计3：组采样策略

用 Ziggurat 算法从 $\hat{\mathcal{Z}}_{T,\mathcal{C}}$ 中采样 $n$ 个潜变量组成子集，但单次随机采样的 $n$ 个样本整体分布可能偏离目标。为此重复采样 $m$ 次得到 $m$ 个候选子集，用统计评估指标选择最优：

$$\mathcal{L}_{T,\mathcal{C}} = \lambda_\mu \cdot \mathcal{L}_{\mu} + \lambda_\sigma \cdot \mathcal{L}_{\sigma} + \lambda_{\gamma_1} \cdot \mathcal{L}_{\gamma_1}$$

三个分量分别衡量子集与目标分布在均值、标准差、偏度上的差异（偏度 $\gamma_1 = 0$，因为高斯分布完全对称）。最终选择 $j = \arg\min_{1 \leq k \leq m} \mathcal{L}^k_{T,\mathcal{C}}$。

该过程可在 GPU 上并行执行，ImageNet-1K（IPC=10, $m=10^6$）仅需每类 **2.6 秒**（单卡 RTX A6000）。超参数 $\lambda_\mu = 1, \lambda_\sigma = 1, \lambda_{\gamma_1} = 0.5$。

## 实验关键数据

### 表1：ImageNet-1K 与更多 SOTA 方法对比（Table A7）

| 架构 | IPC | SRe2L | DWA | TEDDY | D³HR (Ours) |
|---|---|---|---|---|---|
| ResNet-18 | 10 | 31.4±0.5 | 32.7±0.2 | 34.1±0.1 | **44.3±0.3** |
| ResNet-18 | 50 | 51.8±0.4 | 52.5±0.1 | 52.5±0.1 | **59.4±0.1** |
| ResNet-18 | 100 | 55.7±0.4 | 56.2±0.2 | 56.5±0.1 | **62.5±0.0** |
| ResNet-101 | 10 | 38.2±0.4 | 40.0±0.1 | 40.3±0.1 | **52.1±0.4** |
| ResNet-101 | 50 | 61.0±0.4 | — | — | **66.1±0.1** |
| ResNet-101 | 100 | 63.7±0.2 | — | — | **68.1±0.0** |

D³HR 在 ImageNet-1K IPC=10 上较 SRe2L 提升 **12.9%**，较 TEDDY 提升 **10.2%**。

### 表2：CIFAR 数据集对比（Table A6，1000-epoch 验证）

| 数据集 | 架构 | IPC | SRe2L | RDED | D³HR (Ours) |
|---|---|---|---|---|---|
| CIFAR-10 | ResNet-18 | 10 | 53.5±0.6 | 69.8±0.4 | **69.8±0.5** |
| CIFAR-10 | ResNet-18 | 50 | 59.2±0.4 | 75.8±0.6 | **85.2±0.4** |
| CIFAR-10 | ConvNetW128 | 10 | 46.5±0.7 | — | **55.2±0.5** |
| CIFAR-10 | ConvNetW128 | 50 | 54.3±0.3 | — | **66.8±0.4** |

CIFAR-10 IPC=50 上 D³HR 较最优基线 RDED 提升 **12.5%**（ResNet-18）和 **17.4%**（ResNet-101）。

### 消融实验（ImageNet-1K, ResNet-18, IPC=10, Table 3）

| 配置 | 说明 |
|---|---|
| Base-DDPM | DDPM 前向 + 随机采样，基线最低 |
| Base-RS | DDIM 反演 + 随机采样，较 Base-DDPM 提升 **11.5%** |
| + $\mathcal{L}_\mu$ 约束 | 引入均值约束后进一步提升 |
| + $\mathcal{L}_\mu + \mathcal{L}_\sigma$ | 加入标准差约束继续提升 |
| D³HR (全部) | 三项指标（均值+标准差+偏度）组合达到最优 |

## 关键发现

- **反演步数的权衡**：步数 $t=20$ 时分布仍为混合高斯，单高斯匹配不准；$t=40$ 时正态性好但结构信息损失大、重建质量下降。最优步数 **$t=31$**
- **跨架构泛化**：D³HR 一次生成即可适用于 ResNet-18/101、MobileNet-V2、VGG-11、EfficientNet-B0、ShuffleNet-V2、DeiT-Tiny 等多种架构，且在 VGG-11 上也有效（SRe2L 因缺 BN 层而失效）
- **鲁棒性**：不同随机种子运行 10 次，D³HR 较 D4M 平均高出 **~27.5%**，且方差更小
- **存储效率**：只需存储统计参数（$\mu, \sigma$）+ DiT 预训练权重（约 0.016 GB），即可生成任意 IPC 的蒸馏数据集，远小于直接存储蒸馏图像
- **多软标签增益**：使用 5 个教师模型的软标签（与 D3S 对比），D³HR 在 ImageNet-1K 上持续优于 D3S

## 亮点与洞察

- **VAE 空间本质的深刻分析**：理论证明 VAE 潜在空间是多组分高斯混合分布（每个样本一个组分），并通过 t-SNE 可视化直观展示正态性差异，为域映射提供坚实理论动机
- **DDIM 反演的巧妙利用**：将 DDIM 反演从图像编辑领域引入数据集蒸馏，利用其确定性双射性质同时解决信息保持和结构一致性问题，是"变换到简单空间再操作"思路的优雅实现
- **组采样的实用性**：用统计量约束子集整体分布而非逐个匹配，以极低计算成本（GPU 并行采样，每类 <3 秒）显著提升代表性和稳定性
- **完全不依赖教师模型进行蒸馏**：与 SRe2L/DWA/RDED 依赖特定教师模型不同，D³HR 仅依赖预训练扩散模型，真正实现一次蒸馏、多架构通用

## 局限性

- **依赖预训练扩散模型**：使用 DiT 作为骨干，在非 ImageNet 数据集上需额外微调 400 epochs，引入训练成本；扩散模型本身对某些类别的建模质量会限制蒸馏效果
- **高斯近似的局限**：Lemma 4.1 要求反演步数足够大才能保证近似准确性，但步数与重建质量存在权衡，最优点需实验搜索
- **软标签依赖**：虽然蒸馏过程不依赖教师模型，但验证阶段仍使用教师模型的软标签作为训练监督，未完全脱离教师
- **方差调整无效**：实验表明增大或减小采样方差（±50%）均导致性能下降，分布形状的灵活性受限于单高斯假设
- **部分结果缺失**：IPC=100 时某些基线因参数设置问题无结果，对比不完整

## 相关工作与启发

- **数据集蒸馏演进**：从双层优化（梯度/分布/轨迹匹配）到高效方法（SRe2L 利用 BN 统计量、RDED 拼接真实 patch），再到基于扩散的方法（D4M 聚类 VAE latent、Minimax 优化余弦相似度），D³HR 通过域映射+组采样进一步推进
- **扩散模型的新角色**：不仅是生成工具，其正向/反向过程的数学性质（确定性映射、分布变换）可创造性地用于数据压缩和分布建模
- **启发**："将复杂分布映射到简单空间再操作"的思路在域适应、迁移学习、少样本数据增强等需要分布匹配的场景中均有应用潜力

## 评分

- 新颖性: ⭐⭐⭐⭐ — DDIM 反演用于域映射的思路新颖，组采样策略设计实用巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ — 4 数据集、6+ 架构、多基线对比、详尽消融/鲁棒性/存储分析和可视化
- 写作质量: ⭐⭐⭐⭐ — 三个问题的动机论证清晰，理论证明与实验呼应良好
- 价值: ⭐⭐⭐⭐ — 为大规模数据集蒸馏提供高效实用方案，跨架构泛化和存储效率优势突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CaO2: Rectifying Inconsistencies in Diffusion-Based Dataset Distillation](../../ICCV2025/image_generation/cao2_rectifying_inconsistencies_in_diffusion-based_dataset_distillation.md)
- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](../../CVPR2026/image_generation/learnability-guided_diffusion_for_dataset_distillation.md)
- [\[ECCV 2024\] A High-Quality Robust Diffusion Framework for Corrupted Dataset](../../ECCV2024/image_generation/a_highquality_robust_diffusion_framework_for_corrupted_datas.md)
- [\[ICCV 2025\] DiffSim: Taming Diffusion Models for Evaluating Visual Similarity](../../ICCV2025/image_generation/diffsim_taming_diffusion_models_for_evaluating_visual_similarity.md)
- [\[ICML 2025\] Taming Rectified Flow for Inversion and Editing](taming_rectified_flow_for_inversion_and_editing.md)

</div>

<!-- RELATED:END -->
