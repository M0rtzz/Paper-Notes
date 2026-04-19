---
title: >-
  [论文解读] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion
description: >-
  [CVPR 2026][图像生成][图像生成] 提出 PixelRush，一种无需训练的高分辨率图像生成框架，通过部分反演（partial inversion）+ 少步扩散模型 + 高斯滤波拼接 + 噪声注入四大组件，将 4K 图像生成速度从数分钟压缩到约 20 秒（10×–35× 加速），同时在 FID/IS 指标上超越现有 SOTA。
tags:
  - CVPR 2026
  - 图像生成
  - training-free
  - 扩散模型
  - patch-based inference
---

# PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion

**会议**: CVPR 2026  
**arXiv**: [2602.12769](https://arxiv.org/abs/2602.12769)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: high-resolution image generation, training-free, diffusion models, few-step diffusion, patch-based inference

## 一句话总结

提出 PixelRush，一种无需训练的高分辨率图像生成框架，通过部分反演（partial inversion）+ 少步扩散模型 + 高斯滤波拼接 + 噪声注入四大组件，将 4K 图像生成速度从数分钟压缩到约 20 秒（10×–35× 加速），同时在 FID/IS 指标上超越现有 SOTA。

## 研究背景与动机

预训练扩散模型（如 SDXL）在生成高质量图像方面表现出色，但受限于固定的训练分辨率（SDXL 为 1024×1024）。直接在超出训练分辨率的尺度上推理会导致严重的结构伪影和质量退化。微调到目标分辨率的方案面临三大障碍：高分辨率数据稀缺、训练计算代价巨大、以及模型被锁定在特定分辨率上。

现有的无训练高分辨率生成方法分为两类：

**直接推理方法**（如 ScaleCrafter、FreeScale）：在完整高分辨率 latent 上操作，通过修改卷积膨胀率或频域干预来缓解物体重复问题。但频域操作会引入不自然纹理，且内存占用随 latent 尺寸增长，通常限制在 8K 以下。

**Patch-based 方法**（如 DemoFusion、MultiDiffusion）：将高分辨率 latent 切分为与模型原生分辨率匹配的重叠 patch 分别处理，突破了内存瓶颈。但与直接推理方法一样，它们都依赖完整的多步反向扩散（如 50 步），导致生成一张 4K 图像需要数分钟、8K 则超过一小时。

已有加速尝试效果有限：CutDiffusion 通过减少 patch 数量获得微小加速但牺牲质量；LSNR 需要额外训练插件模块，且仅能将步数从 50 降到 30。**核心矛盾在于：现有方法与快速少步采样不兼容**，这是实用化的最大障碍。

PixelRush 的核心洞察是：既然反向扩散过程是按频率层次重建的（先低频全局结构，后高频细节），而粗糙上采样图像已经包含了低频信息，那么**将 latent 扰动到完全高斯噪声再全程去噪是冗余的**。只需从一个浅层中间噪声水平出发，专注于高频细节的合成即可。

## 方法详解

### 整体框架

PixelRush 采用经典的两阶段管线：基础生成 + 级联上采样。

**基础生成阶段**：给定文本 prompt 和目标分辨率，先用多步扩散模型（如 SDXL）在原生分辨率生成基础图像。

**级联上采样阶段**：每次将分辨率翻倍（面积 4×），采用像素空间上采样 → VAE 编码获得粗糙 latent → **精炼阶段**（Refinement Stage）合成高频细节 → VAE 解码输出。多次级联可达 4K、8K 等目标分辨率。

精炼阶段是 PixelRush 的核心创新所在，由四个关键组件构成：

### 关键设计一：部分反演（Partial Inversion）

**核心问题**：现有方法将粗糙 latent 扰动到完全高斯噪声 $\mathbf{z}_T$（$t=T$），然后执行完整的 50 步反向扩散。但反向扩散是分频率层次重建的——早期步骤主要恢复低频全局结构，后期步骤才合成高频细节。对于已具备全局结构的粗糙 latent，早期去噪步骤是冗余的。

**解决方案**：仅将粗糙 latent 通过 DDIM inversion 映射到一个浅层中间噪声水平 $\mathbf{z}_K$（$K \ll T$），而非完全高斯噪声。例如，在 $t=259$（而非 999）处截断，可节省约 75% 的计算量。实验证实（Table 3），将 50 步替换为 15 步部分反演，推理时间从 67 秒降至 18 秒（3.7× 加速），且 FID 从 54.70 改善到 52.90。

### 关键设计二：少步模型加速

部分反演形成的短截反向轨迹天然适配少步扩散模型（如 SDXL-Turbo），因为少步模型能在每步中产生大幅更新，在极短轨迹内合成所需高频细节。

具体做法：前向扰动和反向精炼都使用**单步**完成。选择少步模型对应的中间时间步 $K$（如 SDXL-Turbo 的 4 步时间表中的 $K=249$），执行一步 DDIM inversion + 一步反向去噪。采用确定性 DDIM inversion 而非随机 $q$-sampling，以保留基础图像的结构信息。

这一设计实现了约 10×–35× 的加速，但也带来了两个新问题：patch 边界的棋盘格伪影和过度平滑。

### 关键设计三：高斯滤波拼接（Gaussian Filter Blending）

**问题根源**：传统 patch 拼接（如 MultiDiffusion 的平均混合）在多步去噪中效果尚可，但在少步/单步模式下失效。因为少步反向过程在每个 patch 内部产生剧烈、尖锐的更新，简单平均仅模糊了差异却无法调和，导致明显接缝。

**解决方案**：受图像羽化（image feathering）启发，将硬二值重叠 mask 与高斯模糊核卷积，生成平滑连续的权重 mask。拼接时，靠近某 patch 中心的像素更多地采用该 patch 的值，实现平滑渐变过渡。即使在单步设定下也能完全消除边界伪影。

### 关键设计四：噪声注入（Noise Injection）

**问题根源**：少步模型的去噪步长大，可能无法充分恢复高频细节，产生过度平滑的输出。

**解决方案**：在反向去噪步骤中，将模型预测的噪声 $\epsilon_\gamma(\mathbf{x},t)$ 与随机噪声 $\epsilon_{\text{rand}}$ 进行球面插值（slerp），系数 $\lambda=0.95$：

$$\epsilon'_\gamma(\mathbf{x},t) = \text{slerp}(\epsilon_\gamma(\mathbf{x},t),\, \epsilon_{\text{rand}},\, \lambda)$$

注入随机性有助于展平数据分布，促进高频成分的合成。选用 slerp 而非 lerp 是因为操作在 latent 空间中进行。

**注意**：此技术专门针对少步 patch 管线的过平滑问题。在多步管线中使用会因误差累积导致质量退化。

### 损失函数/训练策略

PixelRush 是**完全无训练**的方法，不涉及任何损失函数或训练过程。所有组件都是对推理过程的干预。唯一需要设定的超参数包括：部分反演时间步 $K$（推荐 249）、噪声注入插值系数 $\lambda$（固定 0.95）、以及 patch 的重叠比例。

## 实验关键数据

### 主实验：与 SOTA 方法的定量对比

| 方法 | 2K FID↓ | 2K IS↑ | 2K 时间(s) | 4K FID↓ | 4K IS↑ | 4K 时间(s) |
|------|---------|--------|-----------|---------|--------|-----------|
| SDXL-DI | 73.34 | 10.93 | 28 | 153.53 | 7.32 | 247 |
| FouriScale | 72.65 | 12.31 | 87 | 98.97 | 8.54 | 680 |
| DemoFusion | 68.46 | 13.15 | 75 | 74.75 | 12.57 | 507 |
| FreeScale | 52.87 | 13.56 | 53 | 58.28 | 13.35 | 323 |
| **PixelRush** | **50.13** | **14.32** | **4** | **54.67** | **13.75** | **20** |

PixelRush 在所有指标上全面超越 SOTA，2K 生成仅需 4 秒（比 FreeScale 快 13×），4K 仅需 20 秒（比 FreeScale 快 16×，比 FouriScale 快 34×）。

### 消融实验：各组件贡献

| 配置 | 去噪步数 | FID↓ | IS↑ | 时间(s) |
|------|---------|------|-----|---------|
| Baseline (50-step DDIM) | 50 | 54.70 | 13.92 | 67 |
| + Partial Inversion | 15 | 52.90 | 13.89 | 18 |
| + Few-step Model | 1 | 57.23 | 13.65 | 4 |
| + Gaussian Blend | 1 | 56.16 | 13.77 | 4 |
| + Noise Injection (PixelRush) | 1 | **50.13** | **14.32** | **4** |

- 部分反演：3.7× 加速且质量提升（FID 54.70→52.90）
- 引入少步模型：进一步加速到 4 秒，但 FID 退化到 57.23
- 高斯拼接：修复棋盘格伪影，FID 降至 56.16
- 噪声注入：修复过平滑，FID 大幅降至 50.13，全面超越 baseline

### 关键发现

1. **部分反演时间步的选择至关重要**：$K=249$（最浅层）效果最优，FID 50.13；随着 $K$ 增大（499→749→999），性能持续退化（FID 66.24→72.34→79.45），因为多步 DDIM inversion 与少步模型存在不兼容性。
2. **模型选择的鲁棒性**：使用不同 base/refinement 模型组合（SDXL+SDXL-Turbo、SDXL+SD-Turbo、SANA+SDXL-Turbo、SDXL+Pixart-δ），性能一致稳定，证明方法的通用性。
3. **定性分析**：SDXL-DI 产生严重物体重复和不自然纹理；DemoFusion 出现结构伪影（如龙头重复）；FouriScale/FreeScale 引入网格/噪声纹理；PixelRush 保持尖锐自然的细节且结构完整。

## 亮点与洞察

1. **核心洞察价值极高**：扩散模型的频域层次重建特性意味着精炼任务不需要完整反向过程——这个观察虽简单但被之前所有方法忽略，由此带来了数量级的加速。
2. **组件设计的针对性**：每个组件精准解决一个具体问题（部分反演→冗余计算、少步模型→进一步加速、高斯拼接→边界伪影、噪声注入→过平滑），层层递进、逻辑清晰。
3. **打破速度-质量权衡**：PixelRush 在大幅加速的同时还提升了生成质量（FID 从 52.87 降到 50.13），颠覆了"加速必然牺牲质量"的传统认知。
4. **实用性突破**：首次实现单 A100 GPU 上 100 秒内生成 8K 图像，使高分辨率生成从离线任务变为实际可用的实时工具。

## 局限与展望

1. **依赖蒸馏模型**：加速能力依赖于少步蒸馏模型（如 SDXL-Turbo）的可用性，对于没有蒸馏版本的新模型架构（如 DiT-based 模型），适配性有待验证。
2. **固定超参数**：噪声注入系数 $\lambda=0.95$ 对所有实验固定，不同内容/风格可能需要自适应调整。
3. **评估指标有限**：仅使用 FID 和 IS 评估，缺少对文本对齐度（如 CLIP score）和用户偏好的评估。
4. **结构一致性**：patch-based 方法本质上对全局一致性有挑战，极端情况下（如全景图中的几何连续性）可能仍存在问题。
5. **可扩展到视频**：当前仅针对图像生成，将类似策略扩展到高分辨率视频生成是有价值的方向。

## 相关工作与启发

- **MultiDiffusion / DemoFusion**：确立了 patch-based 高分辨率生成范式，但受限于多步去噪的计算开销。PixelRush 保留了 patch-based 框架但通过部分反演+少步模型实现了质的加速。
- **FreeScale / FouriScale**：通过频域干预解决物体重复，但引入纹理伪影。PixelRush 用 DDIM-inverted latent 作为结构先验，自然规避了物体重复问题。
- **SDXL-Turbo / 一致性蒸馏**：少步扩散模型的发展为 PixelRush 提供了基础设施支撑，证明蒸馏模型不仅能加速常规生成，也能被巧妙地整合到高分辨率精炼管线中。
- **HiWave**：提出使用 DDIM-inverted noise 替代随机高斯噪声来保留结构信息，这一思路被 PixelRush 继承并推进到少步极端。

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-----------|------|
| 创新性 | 8 | 部分反演+少步模型的结合是首创，洞察简洁而深刻 |
| 技术含量 | 7 | 各组件设计目标明确、分析扎实，但单个技术复杂度不高 |
| 实验充分性 | 8 | 主实验+多维消融+模型鲁棒性分析+定性对比，覆盖全面 |
| 实用价值 | 9 | 10×–35× 加速 + 质量提升，直接解决实际部署瓶颈 |
| 写作质量 | 8 | 动机清晰、逻辑递进、图表丰富 |
| **总分** | **8.0** | 实用性极强的工作，以简洁优雅的方式解决高分辨率生成的速度瓶颈 |

<!-- RELATED:START -->

## 相关论文

- [Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)
- [Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)

<!-- RELATED:END -->
