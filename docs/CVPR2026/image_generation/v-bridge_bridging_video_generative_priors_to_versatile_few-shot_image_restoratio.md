---
title: >-
  [论文解读] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration
description: >-
  [CVPR 2026][图像生成][视频生成先验] 将图像修复重新定义为渐进式视频生成过程，利用预训练视频模型（Wan2.2-TI2V-5B）的丰富视觉先验，仅用 1,000 个多任务训练样本（不到现有方法的 2%）就实现了多种退化类型的全能修复，超越了在百万级数据上训练的专用架构。
tags:
  - CVPR 2026
  - 图像生成
  - 视频生成先验
  - 图像修复
  - Few-shot学习
  - 渐进式修复
  - 漂移校正
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
V-Bridge 将图像修复建模为一条从低质到高质的"质量进化轨迹"。输入低质图像 $\mathbf{I}_{\text{LQ}}$ 作为首帧，高质图像 $\mathbf{I}_{\text{HQ}}$ 作为末帧，中间帧通过线性插值生成。基于 Wan2.2-TI2V-5B 视频生成主干，分三阶段训练：伪时序数据构造 → 渐进式课程训练 → 漂移校正。

### 关键设计

1. **伪时序数据构造**:

    - 功能：将静态 (LQ, HQ) 对转化为长度 $T+1$ 的伪视频序列
    - 核心思路：$\mathbf{I}_t = (1-\alpha_t)\mathbf{I}_{\text{LQ}} + \alpha_t \mathbf{I}_{\text{HQ}}$，$\alpha_t = t/T$，形成单调递增的质量轨迹
    - 设计动机：视频模型本就擅长建模时序一致的连续变化，伪时序序列为模型提供了时间一致的监督信号，比单步回归更稳定

2. **渐进式课程训练（Progressive Curriculum Training）**:

    - 功能：通过多阶段分辨率递增策略弥合视频预训练分辨率（720p）与高分辨率修复需求（4K）的差距
    - 核心思路：定义分辨率课程 $\{r_t\}$（$r_1 < r_2 < \cdots < r_T$），在每个阶段对训练数据做 DownUp 降采样 $v_i^{(t)} = \text{DownUp}(v_i, r_t)$，从低分辨率到高分辨率逐步训练
    - 设计动机：先学全局结构修复，再逐步精化高频细节。直接高分辨率训练计算昂贵且和预训练分辨率差距太大影响学习效率
    - 训练目标统一为：$\mathcal{L}(\theta) = \mathbb{E}[\ell(f_\theta(\mathbf{I}_0, t), \mathbf{I}_t)]$

3. **漂移校正（Drift Correction）**:

    - 功能：修正视频模型因预训练分辨率受限导致的高频细节偏差
    - 核心思路：将基础模型输出 $\hat{x}$ 视为低保真分布 $p_\theta^{\text{LR}}(x)$ 的样本，训练辅助校正模型 $g_\phi: p_\theta^{\text{LR}} \to p_{\text{HR}}$，构造 $\hat{x}$ 到 $x^{\text{HR}}$ 的短轨迹插值序列进行训练
    - 设计动机：课程训练减少了分辨率差距但无法完全消除。漂移校正用轻量的短序列修正弥补剩余偏差，计算开销小。本质上是将分辨率受限导致的系统性偏差建模为一种新型退化

### 损失函数 / 训练策略
- 两个模型都采用 Wan2.2-TI2V-5B 作为主干网络
- 统一训练目标：$\mathcal{L}(\theta) = \mathbb{E}[\ell(f_\theta(\mathbf{I}_0, t), \mathbf{I}_t)]$，本质是条件视频生成的 SFT
- 每类退化任务仅选 50 个样本（来自 FoundIR 和 RealCE），总计 ~1000 训练样本
- 漂移校正模型同样每类用 50 个样本，也使用 Wan2.2 主干

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

## 相关论文

- [AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)
- [Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [Agentic Retoucher for Text-To-Image Generation](agentic_retoucher_for_text-to-image_generation.md)
- [Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control](taming_video_models_for_3d_and_4d_generation_via_zero-shot_camera_control.md)

<!-- RELATED:END -->
