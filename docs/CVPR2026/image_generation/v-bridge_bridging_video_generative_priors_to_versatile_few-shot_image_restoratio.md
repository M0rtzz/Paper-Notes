---
title: >-
  [论文解读] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration
description: >-
  [CVPR 2026][图像生成][视频生成先验] 将图像修复重新定义为渐进式视频生成过程，利用预训练视频模型（Wan2.2-TI2V-5B）的丰富视觉先验，仅用 1,000 个多任务训练样本（不到现有方法的 2%）就实现了多种退化类型的全能修复，超越了在百万级数据上训练的专用架构。 领域现状：图像修复（去噪、去模糊、去雾等…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "视频生成先验"
  - "图像修复"
  - "Few-shot学习"
  - "渐进式修复"
  - "漂移校正"
---

# V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration

**会议**: CVPR 2026  
**arXiv**: [2603.13089](https://arxiv.org/abs/2603.13089)  
**代码**: 有（开源项目）  
**领域**: 图像生成  
**关键词**: 视频生成先验, 图像修复, Few-shot学习, 渐进式修复, 漂移校正

## 一句话总结
将图像修复重新定义为渐进式视频生成过程，利用预训练视频模型（Wan2.2-TI2V-5B）的丰富视觉先验，仅用 1,000 个多任务训练样本（不到现有方法的 2%）就实现了多种退化类型的全能修复，超越了在百万级数据上训练的专用架构。

## 研究背景与动机
**领域现状**：图像修复（去噪、去模糊、去雾等）主要依赖任务特定模型，需要大量有监督数据（>百万样本）从头学习修复映射。All-in-One 方法（AirNet、PromptIR、FoundIR）虽然统一了多任务，但仍需海量数据。

**现有痛点**：现有方法与大规模生成模型的发展脱节——视频生成模型（Wan、HunyuanVideo）在海量数据上学到了丰富的结构、语义和动态先验，但这些先验未被利用于低层视觉任务。

**核心矛盾**：视频生成模型学到了强大的视觉世界模型，但修复任务仍在"从头训练"——存在巨大的先验利用缺口。

**切入角度**：观察到修复过程本质上是"从低质到高质的渐进演变"，这与视频帧的时序演变天然对应——可以把修复问题建模为视频生成问题。

**核心idea**：将每个 (LQ, HQ) 对构造为伪时序序列，让视频生成模型学习修复轨迹而非单步回归。

## 方法详解

### 整体框架
V-Bridge 的核心一步是把"图像修复"重新理解成"质量进化的视频"。一张低质图 $\mathbf{I}_{\text{LQ}}$ 不是要一步映射到高质图，而是被当作一段视频的首帧，高质图 $\mathbf{I}_{\text{HQ}}$ 是末帧，中间是一连串质量逐渐变好的过渡帧。这样一来，预训练视频生成模型（Wan2.2-TI2V-5B）那套"理解一帧如何平滑演变到下一帧"的能力就能直接复用，修复变成了一个它天生擅长的条件视频生成问题。整条流水线围绕三件事展开：先把静态的 (LQ, HQ) 配对造成伪时序序列喂给模型，再用分辨率由低到高的课程把视频先验拉到 4K 修复的需求上，最后用一个轻量的漂移校正补回因预训练分辨率受限而丢掉的高频细节。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["静态 (LQ, HQ) 配对"] --> B["伪时序数据构造<br/>线性插值出质量进化序列"]
    B --> C["渐进式课程训练<br/>分辨率由粗到细 SFT Wan2.2"]
    C --> D["基础模型输出<br/>高频仍带系统性偏差"]
    D --> E["漂移校正<br/>把偏差当成新退化再修一次"]
    E --> F["高质修复结果 I(HQ)"]
```

### 关键设计

**1. 伪时序数据构造：把静态修复对变成模型读得懂的"质量进化视频"**

视频模型只会处理"一段连续变化的帧"，而修复数据集给的是孤零零的 (LQ, HQ) 两张图，没有中间过程——直接拿来训练等于让视频模型做单步回归，浪费了它建模时序演变的本事。V-Bridge 用最简单的线性插值把缺失的中间帧补出来：第 $t$ 帧为 $\mathbf{I}_t = (1-\alpha_t)\mathbf{I}_{\text{LQ}} + \alpha_t \mathbf{I}_{\text{HQ}}$，权重 $\alpha_t = t/T$ 从 0 单调增到 1，于是 $T+1$ 帧构成一条质量只增不减的轨迹。模型学的不再是"输入→输出"的跳变，而是"这一帧比上一帧好一点"的连续监督信号，每一步的目标都更接近、更稳定，也正好落在视频先验最擅长的区间里。

**2. 渐进式课程训练：用分辨率由粗到细的课程，把 720p 的视频先验拉到 4K 修复上**

视频模型是在 720p 量级上预训练的，而修复往往要输出 4K，分辨率鸿沟太大，直接在高分辨率上硬训既昂贵又学不动。V-Bridge 定义一条分辨率课程 $\{r_t\}$（$r_1 < r_2 < \cdots < r_T$），每个阶段先把训练样本用 DownUp 降采样到当前分辨率 $v_i^{(t)} = \text{DownUp}(v_i, r_t)$，从低到高逐级训练。低分辨率阶段模型先把全局结构、语义这些"大方向"的修复学会，再到高分辨率阶段补高频细节，coarse-to-fine 地缩小与预训练分辨率的距离。各阶段共用同一个条件视频生成目标：

$$\mathcal{L}(\theta) = \mathbb{E}\big[\ell(f_\theta(\mathbf{I}_0, t), \mathbf{I}_t)\big]$$

本质上就是在预训练视频模型上做条件生成的 SFT，每类退化只取 50 个样本（来自 FoundIR 和 RealCE），全部加起来约 1000 个——不到现有方法 2% 的数据量。

**3. 漂移校正：把分辨率受限留下的系统性偏差，当成一种新退化再修一次**

课程训练缩小了分辨率鸿沟，却没法完全抹平——基础模型的输出在高频细节上仍带着可见偏差。作者的视角很巧：把基础模型的输出 $\hat{x}$ 看成是从一个"低保真分布" $p_\theta^{\text{LR}}(x)$ 里采的样，那这种偏差就等价于一种新型退化，可以再训一个修复器来纠。具体是训练一个辅助校正模型 $g_\phi: p_\theta^{\text{LR}} \to p_{\text{HR}}$，仍沿用伪时序的思路，构造从 $\hat{x}$ 到真值 $x^{\text{HR}}$ 的一段短插值轨迹来学习。因为只补最后一段偏差，轨迹短、开销小，校正模型同样每类 50 个样本、同样以 Wan2.2 为主干。消融里这一步贡献最大：去掉它，低光照 PSNR 从 26.94 直接掉到 19.18（-7.76 dB）。

## 实验关键数据

### 主实验 — FoundIR 测试集（PSNR/SSIM，越高越好）

| 退化类型 | V-Bridge (1K数据) | FoundIR-G (1M数据) | 最佳专用方法 |
|---------|-------------------|---------------------|-------------|
| Blur | 24.92 / 0.781 | 24.34 / 0.786 | 25.31 (DiffUIR) |
| Lowlight | **26.94 / 0.894** | 12.35 / 0.719 | 21.90 (InstructIR) |
| B+N (混合) | **27.31 / 0.847** | 22.53 / 0.765 | 24.44 (DiffUIR) |
| B+J (混合) | **25.33 / 0.802** | 28.33 / 0.849 | 32.99 (InstructIR) |

### 消融实验

| 配置 | 说明 |
|------|------|
| w/o Drift Correction | 低光照 PSNR 从 26.94 降至 19.18（-7.76 dB） |
| 每类 50 → 更少样本 | 性能平缓下降，展现极强 few-shot 能力 |
| 视频先验 vs 图像方法 | 1K 数据的 V-Bridge 在低光/混合退化上超越 1M 数据的 FoundIR |

### 关键发现
- 在低光增强和混合退化上优势最显著（+7.76 dB / +4.78 dB），说明视频先验对复杂退化有更好的泛化
- 在单一简单退化（噪声、JPEG）上略逊于百万级数据训练的方法，但数据效率提升 1000×
- 漂移校正贡献最大，尤其在需要高频细节的任务上（低光PSNR提升近8dB）
- 展现出强大的 OOD 泛化能力——在未见过的退化类型和外部 benchmark 上也能保持竞争性能
- 训练仅需每类 50 个样本（总计 ~1K），而 FoundIR 使用 1M 数据——数据效率提升约 1000 倍
- 渐进式课程训练策略有效弥合了预训练分辨率（720p）与修复目标分辨率之间的差距

## 亮点与洞察
- **范式创新**：首次将图像修复转化为视频生成任务，用"质量进化轨迹"替代"单步回归"——这是一个全新的任务定义，打破了修复领域"输入→输出"的固有范式
- **1000× 数据效率**：1K 样本 vs 1M 样本达到竞争甚至更优性能，证明视频生成模型已隐式编码了强大的修复先验——这些先验是在视频训练中"免费"获得的
- **Chain-of-Frames 思想的延伸**：将 CoF 推理从高层语义推理延伸到低层像素修复，证明了视频模型的通用性
- **渐进式课程的工程价值**：coarse-to-fine 策略不仅提升性能，更是让大模型适配新任务的通用工程方法论

## 局限与展望
- 在噪声去除和 JPEG 压缩等简单退化上不如专用方法，可能因为这些退化不需要复杂的结构先验
- 推理成本高（视频生成模型参数量为 5B），不适合实时应用
- 仅验证了 Wan2.2 作为主干，其他视频模型（如 CogVideoX、HunyuanVideo）未验证
- 线性插值构造中间帧过于简单，可能丢失非线性质量演变的信息
- 伪时序序列的长度 $T$ 对性能的影响未充分消融
- 漂移校正模型使用的也是完整的 Wan2.2-5B，参数量冗余严重——是否可以用轻量模型完成最后的精化？

## 相关工作与启发
- **vs FoundIR**: FoundIR 是当前全能修复 SOTA，但需要 1M 数据；V-Bridge 用 0.1% 数据达到竞争性能，在低光和混合退化上甚至超越
- **vs DiffBIR/DiffUIR**: 这些方法利用 2D 扩散先验，V-Bridge 利用 3D 视频先验——额外的时序建模能力带来更好的全局一致性，尤其是跨退化类型的泛化
- **vs Chain-of-Frames**: CoF 推理主要用于高层语义任务，V-Bridge 首次证明了 CoF 在像素级修复上的可行性
- **vs All-in-One方法（AirNet/PromptIR）**: 这些方法需要退化类型特定的 prompt 或对比学习；V-Bridge 无需退化类型先验
- **启发**：视频基础模型可能是下一代统一视觉模型的基座——不仅用于生成，还可用于理解和修复。这为 foundation model 的应用打开了新维度

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 全新范式——将修复建模为视频生成，视角新颖且实验有力支撑
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种退化和 benchmark，OOD 测试充分，但缺少与更多 all-in-one 方法的对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，框架图直观，但部分实验表格不够整洁
- 价值: ⭐⭐⭐⭐⭐ 开辟了视频先验用于低层视觉的新方向，影响力大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)
- [\[CVPR 2026\] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[CVPR 2026\] Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control](taming_video_models_for_3d_and_4d_generation_via_zero-shot_camera_control.md)
- [\[CVPR 2026\] FontCrafter: High-Fidelity Element-Driven Artistic Font Creation with Visual In-Context Generation](fontcrafter_high-fidelity_element-driven_artistic_font_creation_with_visual_in-c.md)

</div>

<!-- RELATED:END -->
