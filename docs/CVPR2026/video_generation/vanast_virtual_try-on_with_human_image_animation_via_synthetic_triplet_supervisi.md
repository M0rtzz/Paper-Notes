---
title: >-
  [论文解读] Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision
description: >-
  [CVPR 2026][视频生成][虚拟试穿] Vanast 提出一种统一框架，通过 Dual Module 架构（HAM + GTM）和三阶段合成数据构建流水线，在单阶段内同时完成服装迁移和人体动画生成，在 Internet 数据集上 PSNR 达到 17.95dB（+5.5dB vs 最佳两阶段方案）…
tags:
  - "CVPR 2026"
  - "视频生成"
  - "虚拟试穿"
  - "人体动画"
  - "合成三元组"
  - "Dual Module"
  - "视频扩散"
---

# Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision

**会议**: CVPR 2026  
**arXiv**: [2604.04934](https://arxiv.org/abs/2604.04934)  
**代码**: [https://hyunsoocha.github.io/vanast/](https://hyunsoocha.github.io/vanast/)  
**领域**: 视频生成  
**关键词**: 虚拟试穿、人体动画、合成三元组、Dual Module、视频扩散

## 一句话总结

Vanast 提出一种统一框架，通过 Dual Module 架构（HAM + GTM）和三阶段合成数据构建流水线，在单阶段内同时完成服装迁移和人体动画生成，在 Internet 数据集上 PSNR 达到 17.95dB（+5.5dB vs 最佳两阶段方案），LPIPS 仅 0.237。

## 研究背景与动机

1. **领域现状**：虚拟试穿（VTON）和人体动画是电商和社交媒体的核心需求。现有方案将两者分两阶段处理——先用 CatVTON/OmniTry 换装生成静态图，再用 StableAnimator 做动画。
2. **现有痛点**：两阶段方法存在严重的误差累积：(1) 身份漂移——动画阶段丢失换装阶段的身份信息；(2) 服装扭曲——动画过程中服装细节变形；(3) 前后不一致——正反面服装外观断裂。
3. **核心矛盾**：单阶段统一模型需要同时学习"换装"和"动画"两种不同性质的变换，但缺乏配对的三元组训练数据（人物+服装+动作序列）。
4. **本文目标**：构建大规模三元组数据集并训练单阶段统一模型。
5. **切入角度**：用合成数据弥补真实三元组数据的稀缺——通过扩散inpainting、视频服装提取、工作室拍摄三种策略构建数据。
6. **核心 idea**：Dual Module 架构在冻结的视频 DiT 骨干上并行添加人体动画模块（HAM）和服装迁移模块（GTM），通过加权残差连接实现统一生成。

## 方法详解

### 整体框架

输入（人物图像 + 服装图像 + 姿态引导视频）→ VAE 编码为隐空间 → 冻结 DiT 骨干 + HAM 处理人体姿态条件 + GTM 处理服装条件 → 加权融合 $h_{l+1} = B^{T2V}_l(h_l) + \alpha \cdot B^{HAM}_l(h_l) + \beta \cdot B^{GTM}_l(h_l)$（$\alpha=\beta=0.5$）→ VAE 解码生成视频。

### 关键设计

1. **三阶段数据构建流水线**

    - 功能：从零构建大规模三元组监督（换装图、服装、视频）
    - 核心思路：Stage 1 通过 FLUX 扩散 inpainting 合成替换服装图像；Stage 2 从wild视频提取服装后用扩散生成对应人物；Stage 3 工作室拍摄多服装视频。最终得到 9135 段视频
    - 设计动机：真实三元组数据极度稀缺——自然场景中同一人穿不同衣服做相同动作的数据几乎不存在

2. **Dual Module 架构（HAM + GTM）**

    - 功能：在冻结骨干上并行注入人体动画和服装迁移两种条件
    - 核心思路：HAM 和 GTM 都是轻量级适配模块，只训练这两个分支而冻结主干 DiT。各自独立处理不同条件后通过加权残差叠加到主干特征流中
    - 设计动机：将两种高度不同的条件信号解耦到独立模块中，避免相互干扰。消融显示 Dual Module 比 Single Module 在 FID 上低 17.8（91.05 vs 108.84）

3. **零样本服装插值**

    - 功能：无需重新训练即可实现两件服装之间的渐变混搭
    - 核心思路：通过 $\gamma$ 加权两个 GTM 分支的输出：$h_{l+1} = ... + \gamma \cdot B^{GTM}_l(h_l; G_A) + (1-\gamma) \cdot B^{GTM}_l(h_l; G_B)$，$\gamma \in [0,1]$ 控制混合比例
    - 设计动机：模块化 GTM 设计自然支持多服装条件的线性插值，无额外训练成本

### 损失函数 / 训练策略

标准扩散去噪损失（v-prediction），仅优化 HAM 和 GTM 参数，DiT 骨干完全冻结。训练数据 9135 段视频（3-10秒），测试 80 样本（Internet）+ 50 样本（ViViD）。

## 实验关键数据

### 主实验

| 方法 | L1↓ | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|------|-----|-------|-------|--------|------|
| CatVTON+StableAnimator | 0.1242 | 14.56 | 0.765 | 0.327 | 132.09 |
| OmniTry+StableAnimator | 0.1227 | 14.53 | 0.767 | 0.318 | 121.04 |
| VACE (1-stage) | 0.1453 | 13.09 | 0.689 | 0.405 | 115.40 |
| **Vanast** | **0.0719** | **17.95** | **0.755** | **0.237** | **91.05** |

### 消融实验

| 配置 | L1↓ | PSNR↑ | FID↓ | VFID↓ | 说明 |
|------|-----|-------|------|-------|------|
| Single Module | 0.1162 | 14.28 | 108.84 | 39.64 | 单模块性能差 |
| Backbone-LoRA | 0.1359 | 13.17 | 120.97 | 42.47 | 微调骨干反而差 |
| w/o SynthHuman | 0.1163 | 14.62 | 110.76 | 38.89 | 合成数据关键 |
| **Full model** | **0.1069** | **14.74** | **104.59** | **35.60** | 完整模型 |

### 关键发现

- Dual Module vs Single Module：FID 从 108.84 降到 91.05，验证了条件解耦的必要性
- 冻结骨干 vs LoRA 微调：冻结更好（FID 91.05 vs 120.97），可能因为 LoRA 破坏了预训练视频先验
- SynthHuman 数据贡献 6 点 FID 提升，合成数据策略有效
- VFID_ResNeXt 仅 0.39（vs 基线 1.69-5.86），时序一致性大幅领先

## 亮点与洞察

- **单阶段统一的工程优雅性**：消除了两阶段流水线的误差累积，一步到位生成换装动画视频
- **合成数据弥补真实数据空白**：三阶段数据构建策略可迁移到其他缺乏配对数据的视频生成任务
- **零样本插值能力**：模块化设计自然获得了服装混搭的零样本能力，商业应用价值极高

## 局限与展望

- 训练数据仅 9135 段视频，服装类型覆盖面有限
- 对不常见服装类型（如连体衣、和服）效果可能退化
- 合成数据的质量瓶颈取决于 FLUX inpainting 和 VLM 的能力
- 后续可扩展到多人场景和配饰（帽子、包等）的统一迁移

## 相关工作与启发

- **vs CatVTON/OmniTry+StableAnimator**: 两阶段方案 FID 121-132，Vanast 91.05。差距主要来自误差累积
- **vs VACE**: 虽然也是单阶段，但 VFID_ResNeXt=5.86 远高于 Vanast 的 0.39，时序一致性差距悬殊

## 评分

- 新颖性: ⭐⭐⭐⭐ Dual Module和合成三元组数据策略有新意
- 实验充分度: ⭐⭐⭐⭐ 两数据集+多基线+消融，但测试规模偏小
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰
- 价值: ⭐⭐⭐⭐ 电商虚拟试穿的直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] The Devil is in the Details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection](the_devil_is_in_the_details_enhancing_video_virtual_try-on_via_keyframe-driven_d.md)
- [\[ICML 2026\] iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance](../../ICML2026/video_generation/itryon_mastering_interactive_video_virtual_try-on_with_spatial-semantic_guidance.md)
- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](../../ICCV2025/video_generation/multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[CVPR 2026\] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)
- [\[CVPR 2026\] LottieGPT: Tokenizing Vector Animation for Autoregressive Generation](lottiegpt_vector_animation_generation.md)

</div>

<!-- RELATED:END -->
