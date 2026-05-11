---
title: >-
  [论文解读] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation
description: >-
  [ICCV 2025][视频生成] 本文提出 FVGen，一个将多步视频扩散模型（VDM）蒸馏为仅需 4 步采样的快速学生模型的框架，通过 GAN 目标的学生初始化和软化反向 KL 散度优化，实现了保持甚至超越教师模型视觉质量的同时减少 90% 以上的采样时间。
tags:
  - ICCV 2025
  - 视频生成
  - 视频扩散蒸馏
  - 对抗训练
  - 软化反向KL散度
  - 少步采样
---

# FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation

**会议**: ICCV 2025  
**arXiv**: [2508.06392](https://arxiv.org/abs/2508.06392)  
**代码**: [https://wbteng9526.github.io/fvgen/](https://wbteng9526.github.io/fvgen/)  
**领域**: 视频生成  
**关键词**: 新视角合成, 视频扩散蒸馏, 对抗训练, 软化反向KL散度, 少步采样

## 一句话总结

本文提出 FVGen，一个将多步视频扩散模型（VDM）蒸馏为仅需 4 步采样的快速学生模型的框架，通过 GAN 目标的学生初始化和软化反向 KL 散度优化，实现了保持甚至超越教师模型视觉质量的同时减少 90% 以上的采样时间。

## 研究背景与动机

**领域现状**：基于 NeRF 和 3D Gaussian Splatting 的新视角合成在稠密视角输入下已能生成高质量 3D 场景，但在稀疏视角（2-5 张输入图像）下常出现伪影。近期，视频扩散模型（VDM）被用于稀疏视角合成——通过生成输入视角之间的连续视角序列来填补未观察区域，如 ViewCrafter 利用 DUSt3R 构建初始点云来引导 VDM 生成新视角。

**现有痛点**：VDM 的核心问题是**采样速度极慢**——ViewCrafter 生成 16 帧需要约 13.2 秒（50 步去噪），这使得 VDM 方法无法用于实时 3D 重建、动态场景重建或需要大量新视角的大规模场景。扩散模型蒸馏方法（如 DMD、DMD2）在图像领域效果不错，但直接迁移到多视角视频生成时面临严重的**训练不稳定和模式坍塌**问题，这可能因为多视角视频数据集规模远小于通用视频数据集。

**核心矛盾**：VDM 在稀疏视角合成中质量优秀但速度极慢；蒸馏加速是自然选择，但现有蒸馏方法在小规模多视角数据上不稳定。DMD 损失的本质是反向 KL 散度最小化，而反向 KL 的模式寻找特性（mode-seeking）容易导致学生模型只捕获教师分布的部分模式。

**本文目标**：设计一个稳定高效的视频扩散蒸馏框架，将 50 步的 ViewCrafter 蒸馏为 4 步的学生模型，同时保持视觉质量不下降。

**切入角度**：作者发现两个关键点：(1) 用 GAN 目标初始化学生模型比传统的 ODE solver + 回归损失更有效且更快；(2) 将反向 KL 散度"软化"可以保持模式寻找特性的同时避免模式坍塌。

**核心 idea**：先用 GAN 对抗训练（教师作判别器）初始化学生模型，再用软化反向 KL 散度优化分布匹配，两阶段训练实现稳定高效的 4 步视频生成。

## 方法详解

### 整体框架

FVGen 的训练分为两个阶段。输入是稀疏视角图像和目标相机轨迹，输出是 4 步生成的新视角视频序列。第一阶段（GAN 初始化）：学生模型作为生成器，教师模型作为判别器，通过对抗训练获得有效的权重初始化。第二阶段（分布匹配蒸馏）：使用软化反向 KL 散度优化学生与教师的分布差异，同时动态更新伪分数函数。教师、学生和伪分数函数三个网络均基于 ViewCrafter 的稀疏模型初始化。

### 关键设计

1. **GAN 目标的学生初始化**:

    - 功能：为学生模型提供有效的权重初始化，使其在 DMD 训练开始时就能生成接近真实数据的样本
    - 核心思路：将学生模型 $G_\theta$ 视为生成器，教师模型作为判别器 $D$。学生从噪声 $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ 出发，用少步去噪生成视频，注入噪声后送入教师模型提取中间层特征，经 3D 卷积分类器 $f$ 判断真伪。损失函数为标准 GAN 对抗损失：$\mathcal{L}_D = \mathbb{E}[\log f(D(F(\mathbf{x}, t)))] - \mathbb{E}[\log f(D(F(G_\theta(\mathbf{z}), t)))]$。关键设计：(a) 使用**真实样本**而非教师生成的样本作为正样本，避免学生被教师质量上限限制；(b) 判别器使用教师 UNet 的中间层特征而非额外网络，节省计算同时利用教师的语义表示；(c) 分类器 $f$ 采用 3D 卷积以适应视频数据的时空结构
    - 设计动机：之前的蒸馏方法（如 CausVid）用 ODE solver 生成噪声-样本对再训练回归损失来初始化，这非常耗时且学生质量被教师上限限制。GAN 初始化更快速且直接面向真实数据分布

2. **软化反向 KL 散度（Soften Reverse KL-Divergence）**:

    - 功能：替代标准反向 KL 散度进行分布匹配蒸馏，解决训练不稳定和模式坍塌问题
    - 核心思路：标准 DMD 损失最小化 $D_{\text{KL}}(p_{\text{fake}} \| p_{\text{real}})$（反向 KL），这是模式寻找的——学生可以通过完全忽略教师分布中不太显著的模式来最小化损失。软化版本改为最小化 $D_{\text{KL}}(\frac{1}{2}p_{\text{real}} + \frac{1}{2}p_{\text{fake}} \| p_{\text{real}})$，即比较"真假分布的均匀混合"与"真实分布"。这保持了反向 KL 的模式寻找特性，但因为混合中包含真实分布的一半，学生不能完全忽略任何真实模式。梯度公式为 $\nabla \mathcal{L} = -\mathbb{E}_t[\frac{1}{r(\mathbf{x},t)}(s_{\text{real}} - s_{\text{fake}}) \frac{dG_\theta(\mathbf{z})}{d\theta}]$，其中密度比 $r = p_{\text{real}} / p_{\text{fake}}$ 直接由 GAN 阶段训练好的判别器估计
    - 设计动机：在多视角视频这种小数据场景中，反向 KL 的模式坍塌尤为严重——因为数据分布的模式本身就不够强，学生模型很容易集中到少数模式上。软化 KL 的惩罚机制有效缓解了这一问题

3. **两阶段解耦训练策略**:

    - 功能：保证训练稳定性和密度比估计准确性
    - 核心思路：不同于 DMD2 将 GAN 训练和 DMD 训练端到端联合优化，FVGen 严格将 GAN 初始化（4000 iterations）和 DMD 蒸馏（5000 iterations）分为两个阶段。GAN 阶段只训练学生和判别器分类头，DMD 阶段继续训练学生和伪分数函数。这种解耦确保了两个目标不互相干扰：GAN 阶段获得准确的密度比估计和良好的初始化，DMD 阶段利用已稳定的密度比进行精细蒸馏
    - 设计动机：DMD2 的端到端训练在多视角视频数据上展现出显著的训练方差和不稳定性，作者实验发现解耦训练是稳定性的关键

### 损失函数 / 训练策略

GAN 阶段：标准对抗损失 + 双时间尺度更新规则。DMD 阶段：软化反向 KL 散度梯度 + 伪分数函数的扩散损失更新。全流程在 8× NVIDIA H100 上训练约 1 天，batch size 4，分辨率 512×320。训练数据使用 DL3DV-10K 构建的 20,000 个多视角视频-点云对。

## 实验关键数据

### 主实验

| 数据集 | 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | 时间(s) ↓ |
|--------|------|--------|--------|---------|-------|----------|
| MipNeRF360 | DepthSplat | 11.23 | 0.213 | 0.715 | 32.45 | 4.2 |
| MipNeRF360 | MVSplat360 | 12.28 | 0.285 | 0.682 | 25.69 | 87.2 |
| MipNeRF360 | ViewCrafter | 16.35 | 0.346 | 0.433 | 16.28 | 66.3 |
| MipNeRF360 | **FVGen** | **16.28** | **0.352** | **0.429** | 17.44 | **5.1** |
| TNT | DepthSplat | 12.43 | 0.263 | 0.677 | 35.88 | 4.3 |
| TNT | MVSplat360 | 14.18 | 0.301 | 0.532 | 25.23 | 87.3 |
| TNT | ViewCrafter | 18.69 | 0.402 | 0.208 | 23.94 | 65.9 |
| TNT | **FVGen** | **18.72** | **0.411** | **0.210** | 23.64 | **5.0** |

### 消融实验

| GAN Init | DMD | Soften KL | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|----------|-----|-----------|--------|--------|---------|-------|
| ✗ | ✓ | ✓ | 8.62 | 0.154 | 0.880 | 40.17 |
| ✓ | ✗ | ✗ | 16.23 | 0.369 | 0.375 | 21.48 |
| ✓ | ✓ | ✗ | 16.85 | 0.385 | 0.337 | 21.05 |
| ✓ | ✓ | ✓ | **17.50** | **0.382** | **0.320** | **20.54** |

### 关键发现

- **FVGen 在质量上与教师 ViewCrafter 相当甚至更优**，但速度提升 **13x**（66s → 5s）。在 MipNeRF360 上 SSIM 和 LPIPS 均超过 ViewCrafter
- **GAN 初始化是不可或缺的**——去掉后 PSNR 从 17.50 暴跌到 8.62，说明 DMD 无法从头训练学生模型
- **分布匹配蒸馏（DMD）在初始化基础上依然重要**——仅用 GAN（不用 DMD）的 PSNR 为 16.23，加 DMD 后提升到 16.85，说明 DMD 进一步精细化了分布对齐
- **软化 KL 的贡献稳定但显著**——PSNR 从 16.85 提升到 17.50，FID 从 21.05 降到 20.54，证实了软化缓解模式坍塌的有效性
- 与其他蒸馏方法的对比：DMD2 在 ViewCrafter 上训练不稳定，最终 PSNR 仅 9.29-10.27；CausVid 更稳定但仍有模式坍塌，PSNR 为 15.77-17.33。FVGen 全面超越

## 亮点与洞察

- **"教师当判别器"的设计一箭双雕**——既省掉了额外判别器网络的开销，又巧妙地利用了教师模型中间特征的语义表示能力。这个 trick 可迁移到任何扩散蒸馏场景
- **GAN 初始化为 DMD 提供的不只是好权重，还有准确的密度比估计**——这是论文中一个被低估但极重要的洞察。密度比 $r(x,t) \approx f(D(x,t))/(1-f(D(x,t)))$ 直接来自判别器，使得软化 KL 的计算变得免费
- **软化反向 KL 与标准反向 KL 的差异**在小数据场景中被放大——这暗示在任何数据有限的蒸馏场景（如领域特定的扩散模型）中，软化 KL 都可能优于标准 KL
- **90% 的速度提升几乎无质量损失**，这对实际应用的制作管线意味着巨大的效率收益

## 局限与展望

- **继承了 ViewCrafter 的固有局限**——在极端稀疏输入（如单图）下结构完整性和一致性仍会退化，FVGen 无法超越教师的能力上限
- **仅训练了 16 帧的短视频生成**——受计算资源限制，无法覆盖非常大的场景。扩展到更长序列可能需要分段生成并解决帧间一致性
- **三个视频扩散模型**（教师、学生、伪分数函数）的显存需求较高，限制了更大分辨率或更长视频的训练
- 未评估**下游 3D 重建的最终质量**——虽然新视角质量相当，但蒸馏引入的微小分布差异在 3DGS 重建后是否会被放大有待验证
- 未来可以探索：(1) 将蒸馏扩展到更强的 VDM 教师（如 CogVideoX）；(2) 与 3DGS 端到端联合优化；(3) 自适应步数蒸馏——简单场景用 2 步，复杂场景用 4 步

## 相关工作与启发

- **vs ViewCrafter**: 教师模型，50 步 DDIM 采样。FVGen 将其蒸馏为 4 步，速度提升 13x，质量持平。ViewCrafter 使用 DUSt3R 点云引导生成
- **vs DMD2**: DMD2 同时优化 GAN、学生和伪分数函数，但端到端训练在多视角数据上不稳定。FVGen 的关键改进是解耦训练和 3D 判别器
- **vs CausVid**: CausVid 用 ODE solver 生成对进行初始化，再用标准 DMD 蒸馏。FVGen 的 GAN 初始化更高效，软化 KL 比标准 KL 更稳定
- **vs MVSplat360**: MVSplat360 用 VDM 精修 3DGS 渲染，但仅支持低分辨率。FVGen 直接加速 VDM 采样，更通用
- 本文证明了视频扩散蒸馏在 3D 视觉任务中的可行性，为未来实时 VDM 驱动的 3D 重建奠定了基础

## 评分

- 新颖性: ⭐⭐⭐⭐ GAN 初始化+软化 KL 的组合在视频蒸馏中是首次提出，但各组件借鉴了图像蒸馏的思路
- 实验充分度: ⭐⭐⭐⭐⭐ 两个标准数据集、多种基线对比、与蒸馏方法的专门比较、完整消融，实验非常充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学推导完整，图表信息密度高
- 价值: ⭐⭐⭐⭐⭐ 90%速度提升无质量损失对实际应用价值巨大，且框架可推广到其他 VDM 任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_i.md)
- [\[ECCV 2024\] SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](../../ECCV2024/video_generation/sv3d_novel_multi-view_synthesis_and_3d_generation_from_a_single_image_using_late.md)
- [\[CVPR 2025\] StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models](../../CVPR2025/video_generation/streetcrafter_street_view_synthesis_with_controllable_video_diffusion_models.md)
- [\[ICCV 2025\] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](dollar_fewstep_video_generation_via_distillation_and_latent.md)
- [\[ICCV 2025\] Causal-Entity Reflected Egocentric Traffic Accident Video Synthesis](causal-entity_reflected_egocentric_traffic_accident_video_synthesis.md)

</div>

<!-- RELATED:END -->
