---
title: >-
  [论文解读] NTIRE 2026 The 3rd RAIM Challenge: AI Flash Portrait (Track 3)
description: >-
  [CVPR 2026 (Workshop)][图像恢复][低光照人像] NTIRE 2026第三届RAIM挑战赛AI Flash Portrait赛道：将弱闪光灯低光照人像映射为强闪光灯专业级人像，提供800组真实配对数据（含专业设计师修图GT），采用区域感知客观指标+专家盲评的双重评估体系，118支队伍注册、3187次有效提交。
tags:
  - "CVPR 2026 (Workshop)"
  - "图像恢复"
  - "低光照人像"
  - "闪光灯模拟"
  - "人像复原"
  - "主客观评估"
  - "NTIRE"
---

# NTIRE 2026 The 3rd RAIM Challenge: AI Flash Portrait (Track 3)

**会议**: CVPR 2026 (Workshop)  
**arXiv**: [2604.11230](https://arxiv.org/abs/2604.11230)  
**代码**: [CodaBench](https://www.codabench.org/)  
**领域**: 图像复原 / 低光照人像增强  
**关键词**: 低光照人像, 闪光灯模拟, 人像复原, 主客观评估, NTIRE

## 一句话总结

NTIRE 2026第三届RAIM挑战赛AI Flash Portrait赛道：将弱闪光灯低光照人像映射为强闪光灯专业级人像，提供800组真实配对数据（含专业设计师修图GT），采用区域感知客观指标+专家盲评的双重评估体系，118支队伍注册、3187次有效提交。

## 研究背景与动机

移动端低光照人像拍摄是计算摄影的核心难题。受限于小传感器和光线不足，低光照人像存在严重噪声、色彩失真和细节丢失。现有方法存在四个关键局限：(1) 传统低光增强(LLIE)方法聚焦全局亮度提升，导致肤色失真和面部光影平坦化；(2) 真实退化过程极度复杂，合成数据无法模拟弱到强闪光灯的非线性光照变换；(3) 人脸修复模型局限于局部处理，在低光场景下造成前景-背景的"剪贴"感；(4) 传统客观指标（PSNR/SSIM/LPIPS）无法充分捕捉美学和自然度感知。

本赛道由OPPO Y-Lab、深圳大学、港理工VC-Lab和南开大学联合主办，旨在弥合学术研究与工业应用在低光人像计算摄影方面的鸿沟。

## 方法详解

### 整体框架

本赛道提供全新任务定义：从弱闪光灯低光人像映射到强闪光灯专业级人像，超越传统低光增强，结合物理光照增强和美学渲染。评估采用区域感知指标+专家盲评(3:7加权)。

### 关键设计

1. **区域感知评估体系**：评分公式分离人物区域（用LPIPS和ΔE衡量感知相似度和色差）与背景区域（用PSNR衡量信噪比），加上全局SSIM，防止过度锐化人像或平坦化面部仅为拉高全局PSNR
2. **专家盲评机制**：Top-12队伍的结果随机匿名展示给5位以上资深专家，按面部自然度、人像细节保持、光照真实感、背景清洁度、场景平衡、整体一致性六维度评选Top-3，归一化为80-90分的主观分
3. **高质量真实配对数据**：800组1K分辨率数据，每组包含低光输入、专业设计师修图GT和人物掩码，是该领域罕见的高质量真实配对基准

### 训练策略

- 允许使用任何公开外部数据集和预训练模型
- 三阶段竞赛流程：Phase 1训练(600组)→Phase 2在线验证(100组)→Phase 3最终评测(100组隐藏集)
- 最终评测由组织方在统一硬件上复现运行，严禁分辨率缩放

## 实验关键数据

### 主实验（Phase 2在线评测）

| 排名 | 队伍 | Phase 2分数 | LPIPSperson↓ | ΔEperson↓ | GlobalScore↑ |
|------|------|------------|-------------|-----------|-------------|
| 2 | nunucccb | 86.10 | 0.0266 | 7.19 | 0.784 |
| 4 | SHL | 84.91 | 0.0268 | 6.83 | 0.742 |
| 6 | hezhaokun | 84.88 | 0.0270 | 6.75 | 0.739 |
| 7 | KC110 | 84.33 | 0.0284 | 8.07 | 0.765 |
| 基线 | 组织方 | 82.16 | - | - | - |

### 关键发现

- 竞赛吸引118支队伍注册、3187次有效提交，反映该任务的高关注度
- 人物区域的LPIPS和色差(ΔE)与背景PSNR之间存在明显权衡
- 部分队伍在在线榜单得分高但代码复现偏差大，被取消资格（标记为"-"）
- 主客观评价的相关性有待进一步研究

## 亮点与洞察

- 任务定义新颖：不是简单的"低光增强"，而是要求达到专业修图级的美学效果，弥合学术研究与工业应用的鸿沟
- 评估体系设计精良：区域感知指标防止了常见的评价陷阱（如过度平滑得高PSNR），主客观结合保证了实用性
- 真实配对数据+设计师GT是该领域极具价值的资源
- 本赛道揭示：现有方法在面部美学和背景一致性之间难以兼顾

## 局限与展望

- Phase 3详细结果未在本报告中完整披露（主客观融合排名未列出）
- 专家盲评虽更接近人类感知，但评委数量有限(5人)，可能存在主观偏差
- 当前数据集限于1K分辨率，高分辨率场景(4K)未覆盖
- 未来可扩展到视频低光人像增强、多人场景、以及与生成式模型的结合

## 相关工作与启发

- 传统LLIE方法（RetinexNet等）在人像场景的局限性值得系统性研究
- 区域感知评估方法可推广到其他区域重要性不均匀的复原任务
- 人像美学增强与物理一致性的平衡是一个开放性难题
- 六维主观评价标准（面部自然度、人像细节、光照真实感、背景清洁度、场景平衡、整体一致性）可作为人像处理的通用评估框架

## 竞赛流程详解

| 阶段 | 时间 | 内容 | 数据量 |
|------|------|------|--------|
| Phase 1 | 2026.01.23 | 模型设计，发放训练集+基线 | 600组 |
| Phase 2 | 2026.01.28 | 在线客观评测反馈 | 100组(无GT) |
| Phase 3 | 2026.03.05-12 | 代码提交+统一复现+专家盲评 | 100组(隐藏) |
| 最终排名 | 2026.03.19 | 客观分30%+主观分70% | Top-12 |

### 评价指标详解

- **人物区域**：LPIPS_person（感知相似度）+ ΔE_person（色差），确保面部/皮肤高保真
- **背景区域**：PSNR_bg（信噪比），确保背景不引入噪声
- **全局**：SSIM_global（结构相似性），衡量整体结构一致性
- **主观**：50组图像×12队伍匿名展示，专家选Top-3，统计频次归一化为80-90分

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 3 | 任务定义有创新，评估体系设计精良 |
| 技术深度 | 3 | 竞赛报告，涵盖评估和数据构建细节 |
| 实验充分性 | 4 | 118队参赛，主客观双重评估 |
| 写作质量 | 4 | 竞赛动机和评估方案阐述清晰 |
| 实用价值 | 4 | 高质量真实数据集+工业级评估标准 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images](ntire_2026_raindrop_removal_challenge.md)
- [\[CVPR 2026\] Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement](shadow_removal_cascaded_refinement.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[CVPR 2026\] Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis](unicac_universal_computational_aberration_correction_benchmark.md)
- [\[CVPR 2026\] MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring](motionaware_animatable_gaussian_avatars_deblurring.md)

</div>

<!-- RELATED:END -->
