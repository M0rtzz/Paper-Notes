---
title: >-
  [论文解读] Purrception: Variational Flow Matching for Vector-Quantized Image Generation
description: >-
  [ICLR 2026][图像生成][变分流匹配] 提出 Purrception，一种将变分流匹配（Variational Flow Matching）适配到向量量化（VQ）隐空间的图像生成方法，通过在连续嵌入空间中计算速度场的同时学习编码本索引上的分类后验分布，桥接了连续传输动力学和离散监督…
tags:
  - "ICLR 2026"
  - "图像生成"
  - "变分流匹配"
  - "向量量化"
  - "离散扩散"
  - "分类后验"
---

# Purrception: Variational Flow Matching for Vector-Quantized Image Generation

**会议**: ICLR 2026  
**arXiv**: [2510.01478](https://arxiv.org/abs/2510.01478)  
**代码**: 无  
**领域**: 图像生成 (Image Generation)  
**关键词**: 变分流匹配, 向量量化, 离散扩散, 分类后验, 图像生成

## 一句话总结

提出 Purrception，一种将变分流匹配（Variational Flow Matching）适配到向量量化（VQ）隐空间的图像生成方法，通过在连续嵌入空间中计算速度场的同时学习编码本索引上的分类后验分布，桥接了连续传输动力学和离散监督，在 ImageNet-1k 256×256 上实现了更快的训练收敛和与 SOTA 可比的 FID 分数。

## 研究背景与动机

图像生成领域的核心范式正在经历深刻变革。在隐空间（latent space）中进行生成已成为主流方法，而**如何在隐空间中建模生成过程**是一个核心设计选择。目前存在两大技术路径，各有优劣：

### 连续方法（Continuous Methods）

以 Flow Matching 和扩散模型为代表，在连续空间中定义从噪声到数据的传输路径。

- **优势**：具有几何感知能力（geometric awareness），传输路径在连续空间中有良好的数学性质，梯度估计平滑
- **劣势**：无法提供对离散编码本索引的显式监督信号；当底层隐空间是离散的（如 VQ-VAE 的编码本）时存在天然不匹配

### 离散方法（Discrete Methods）

以 Discrete Flow Matching 和掩码语言模型为代表，直接在离散 token 空间中建模。

- **优势**：提供在编码本索引上的显式分类监督，与 VQ 隐空间天然匹配
- **劣势**：缺乏连续空间中的几何结构信息，训练可能不够高效

本文的核心动机是：**能否结合两种方法的优势？**具体来说，能否在保持连续传输动力学的同时，提供离散编码本索引上的显式分类监督？

变分流匹配（Variational Flow Matching, VFM）提供了一个天然的框架来实现这种桥接——它在连续流匹配的基础上引入了变分推断，允许定义关于离散变量的后验分布。Purrception 正是将 VFM 适配到 VQ 图像生成中的首次尝试。

## 方法详解

### 整体框架

Purrception 的生成流程建立在**向量量化自编码器（VQ-VAE/VQ-GAN）**的隐空间之上。与现有方法不同，Purrception 同时在两个层面上操作：

- **连续层面**：在编码本嵌入的连续向量空间中定义和学习速度场（velocity field），实现从噪声分布到数据分布的传输
- **离散层面**：在编码本索引的分类空间中学习后验分布（categorical posterior），提供离散监督信号

这两个层面通过变分流匹配框架**统一建模**，共享底层的参数和优化过程。

### 关键设计

**1. 分类后验：用概率分布代替确定性码字查找。**

VQ 隐空间的麻烦在于，编码本里常有多个语义相近的码字，传统的"取距离最近码字"（argmin）是一次硬决策，会把这种相近性压成噪声。Purrception 不再让模型在每个空间位置选定一个索引，而是给定噪声→数据传输路径上的某个中间状态后，参数化一个编码本索引上的分类分布，预测该位置"最可能落在哪些码字"。这样模型可以表达"这里可能是编码 42 也可能是编码 87"的犹豫，天然带来不确定性量化；同时这种软化处理让训练信号更平滑，是后面收敛加速的来源之一。

**2. 连续速度场：在嵌入空间而非索引空间上学传输。**

光有离散监督还不够——纯离散方法丢掉了连续空间的几何结构。Purrception 把 Flow Matching 的数学框架搬到编码本嵌入的连续向量空间里，学习一个描述"从噪声分布流向数据分布"方向的速度场。之所以在嵌入空间而不是索引空间上操作，是因为离散 token 空间里梯度估计很困难，而连续速度场保留了 Flow Matching 平滑传输路径、训练稳定的几何优势。这条连续信号和上面的离散分类监督互补，构成模型的双重学习信号。

**3. 温度控制：一个参数直接调多样性-质量权衡。**

既然每个位置已经是一个分类后验，那么调节它的温度参数 $T$ 就能直接控制生成行为：低温（$T \to 0$）时后验趋向确定性选择，生成更确定但多样性下降；高温（$T > 1$）时后验更均匀，探索更多码字组合，多样性升高但质量可能下降；适中温度则在两者间平衡。这是概率模型自带的好处——相比 classifier-free guidance 的 scale 参数，温度的物理意义更直观（直接控制后验的"尖锐程度"），而且不需要额外的后处理步骤。

**4. VFM 适配：让变分流匹配吃上编码本的离散结构。**

原始的变分流匹配（VFM）定义在一般的连续/离散混合空间上，要落到 VQ 图像生成需要三处适配。其一是利用编码本的有限离散结构来高效参数化分类后验，而不是在无界空间上硬算；其二是利用嵌入向量之间的距离关系（如欧氏距离）来定义连续传输路径的目标，让几何结构真正进入监督；其三是把分类后验和速度场的参数放进同一个模型联合学习，共享底层的视觉特征表示。正是这三点把前面的离散后验和连续速度场缝合进一个统一的变分推断框架，而不是简单拼两个损失。

### 损失函数 / 训练策略

训练目标包含两个互补的组成部分：

1. **连续流匹配损失**：标准的速度场回归损失，推动模型学习正确的传输方向
$$\mathcal{L}_{FM} = \mathbb{E}\left[\|v_\theta(x_t, t) - u(x_t | x_1)\|^2\right]$$

2. **分类后验损失**：交叉熵损失，推动模型在每个空间位置上正确预测编码本索引的分布
$$\mathcal{L}_{cat} = -\mathbb{E}\left[\sum_k q(k|x_t) \log p_\theta(k|x_t, t)\right]$$

两种损失在变分流匹配框架下自然统一，联合优化。

## 实验关键数据

### 主实验

在 ImageNet-1k 256×256 无条件/类条件图像生成上的评估：

| 方法 | FID↓ | 训练收敛速度 | 类型 |
|------|------|------------|------|
| Continuous Flow Matching | 基线 | 慢 | 纯连续 |
| Discrete Flow Matching | 基线 | 慢 | 纯离散 |
| **Purrception** | **可比SOTA** | **更快** | 连续+离散桥接 |
| 其他SOTA模型 | 参考值 | - | 各类方法 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅连续流匹配 | FID较差，收敛慢 | 缺少离散监督 |
| 仅离散监督 | FID较差 | 缺少连续几何信息 |
| Purrception（完整） | 最优 FID + 最快收敛 | 双重信号互补 |
| 温度=0.5 | 质量高，多样性低 | 低温确定性选择 |
| 温度=1.0 | 质量-多样性平衡 | 标准设定 |
| 温度=1.5 | 多样性高，质量略降 | 高温软化后验 |
| 无分类后验 | 收敛变慢 | 验证离散监督的加速作用 |

### 关键发现

1. **训练收敛加速**：Purrception 比纯连续流匹配和纯离散流匹配基线都更快收敛。分类后验提供的离散监督信号为模型提供了更"尖锐"的学习目标，加速了参数更新的方向性。

2. **质量可比 SOTA**：在 FID 分数上与当前最先进方法可比，证明桥接连续和离散方法不会牺牲生成质量。

3. **温度可控性**：通过单一温度参数即可平滑控制生成的多样性-质量权衡，提供了直观的生成调节手段。

4. **不确定性量化**：分类后验天然提供了对每个空间位置可能编码的不确定性估计，这在其他方法中通常不可用。

## 亮点与洞察

1. **优雅的理论桥接**：Purrception 通过变分流匹配框架在理论上优雅地桥接了连续和离散两种生成范式。不是简单地拼接两种损失，而是在统一的变分推断框架下自然融合。

2. **实用的速度提升**：训练收敛加速是非常实际的贡献——在大规模 ImageNet 训练中，训练效率的提升直接转化为 GPU 小时和成本的节省。

3. **概率化的码字选择**：将确定性的码字查找（argmin 距离）替换为概率化的分类后验，不仅提升了训练效率，还引入了不确定性量化能力。这种"软量化"思想在 VQ 领域具有推广价值。

4. **温度控制的可解释性**：与 classifier-free guidance 的 scale 参数相比，温度参数的物理意义更加直观——直接控制后验分布的"尖锐程度"，可解释性更强。

5. **命名的巧思**："Purrception"（Purr = 猫叫声 + Perception）是一个有趣的命名，暗示了模型对编码本的"感知"（perception）过程是柔和的（purr），而非硬决策。

## 局限与展望

1. **仅在 ImageNet 256×256 上验证**：当前实验规模相对有限。在更高分辨率（如 512×512 或 1024×1024）、更大规模数据集上的表现尚未验证。

2. **与最新 SOTA 的差距**：虽然 FID 与 SOTA "可比"，但可能仍有差距。需要更详细的定量对比才能准确定位。

3. **编码本大小的敏感性**：分类后验的计算复杂度与编码本大小线性相关。对于非常大的编码本（如 16384 个码字），可能面临效率挑战。

4. **文本条件生成**：当前主要在类条件生成上验证，文本条件图像生成（text-to-image）的效果尚不清楚。

5. **与现代编码器架构的兼容性**：如何与最新的 VQ 编码器（如改进的 VQGAN、FSQ 等）以及 continuous latent VAE（如 SD 的 KL-VAE）结合，值得探索。

6. **可扩展至视频生成**：将 Purrception 的框架扩展到视频 VQ 隐空间中，可能在时序一致性上获得额外优势。

## 相关工作与启发

- **Flow Matching**：Lipman et al. 的 Flow Matching 框架，定义了连续传输路径的理论基础
- **Variational Flow Matching**：在 Flow Matching 中引入变分推断的框架，Purrception 的理论基石
- **Discrete Flow Matching / Discrete Diffusion**：在离散 token 空间中进行生成建模
- **VQ-VAE / VQ-GAN**：向量量化隐空间的构建方法，提供了 Purrception 操作的底层空间
- **Masked Image Modeling (MIM)**：如 MaskGIT，在 VQ token 上使用掩码预测
- **Autoregressive VQ 生成**：如 VQVAE + Transformer，按序列生成 VQ token
- 启发方向：**变分推断作为统一框架**允许在单一模型中同时处理连续和离散结构，这一范式可能在其他涉及混合连续-离散空间的生成任务中也有价值（如分子生成、程序合成等）

## 评分

- 新颖性: ⭐⭐⭐⭐ （将变分流匹配适配到 VQ 空间是自然但非平凡的贡献，桥接思路清晰）
- 实验充分度: ⭐⭐⭐ （仅在 ImageNet-1k 256×256 上验证，规模有限，需要更多基准对比）
- 写作质量: ⭐⭐⭐⭐ （理论推导清晰，动机阐述到位）
- 价值: ⭐⭐⭐⭐ （训练效率提升有实际意义，为 VQ 生成提供了新范式，但影响范围取决于 VQ 方法的后续发展）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](latent_diffusion_model_without_variational_autoencoder.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](../../CVPR2026/image_generation/freqflow_frequency_aware_flow_matching.md)
- [\[ICLR 2026\] FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching](flowcast_advancing_precipitation_nowcasting_with_conditional_flow_matching.md)
- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)

</div>

<!-- RELATED:END -->
