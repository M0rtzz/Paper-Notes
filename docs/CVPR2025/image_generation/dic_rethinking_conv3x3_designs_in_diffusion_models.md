---
title: >-
  [论文解读] DiC: Rethinking Conv3x3 Designs in Diffusion Models
description: >-
  [CVPR 2025][图像生成][纯卷积扩散] 构建纯 3×3 卷积的扩散模型 DiC，通过沙漏编码解码器架构扩大感受野、稀疏跳跃连接减少冗余、阶段特定条件注入，在 ImageNet 256×256 上以 DiT-XL 的相当 FLOPs 实现 4.7× 吞吐量和更优 FID（3.89 vs 6.24, with CFG）。
tags:
  - CVPR 2025
  - 图像生成
  - 纯卷积扩散
  - 3×3卷积
  - Winograd加速
  - 沙漏架构
  - 高效生成
---

# DiC: Rethinking Conv3x3 Designs in Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2501.00603](https://arxiv.org/abs/2501.00603)  
**代码**: https://github.com/YuchuanTian/DiC  
**领域**: 扩散模型 / 模型压缩  
**关键词**: 纯卷积扩散, 3×3卷积, Winograd加速, 沙漏架构, 高效生成

## 一句话总结
构建纯 3×3 卷积的扩散模型 DiC，通过沙漏编码解码器架构扩大感受野、稀疏跳跃连接减少冗余、阶段特定条件注入，在 ImageNet 256×256 上以 DiT-XL 的相当 FLOPs 实现 4.7× 吞吐量和更优 FID（3.89 vs 6.24, with CFG）。

## 研究背景与动机

**领域现状**：扩散模型主流架构是 Transformer-based（DiT, U-ViT），依赖自注意力做全局交互。但自注意力计算量随分辨率二次增长，限制了高分辨率生成的效率。

**现有痛点**：(1) DiT 的 isotropic（等尺度）架构缺乏多尺度特征，不适合密集预测；(2) 自注意力在低分辨率 latent（32²）上虽可行但仍是主要瓶颈；(3) 3×3 卷积天然支持 Winograd 变换加速（有效 FLOPs 降至 5/9），但直接用 Conv3x3 替换注意力性能很差。

**核心矛盾**：3×3 卷积生感受野小（3×3），但高效且可 Winograd 加速；注意力生全局感受野，但计算昂贵。如何让纯 3×3 卷积也能做好扩散生成？

**本文目标** 设计一种纯 3×3 卷积的扩散模型架构，既保持生成质量又利用 Winograd 实现大幅加速。

**切入角度**：沙漏（编码-解码器）架构通过下采样自然扩大等效感受野（下采样 2× 后 3×3 等效原图 6×6），加上稀疏跳跃连接和条件注入优化。

**核心 idea**：沙漏架构扩大 3×3 卷积等效感受野 + 稀疏跳跃连接 + 阶段条件注入 + Winograd 加速，实现纯卷积的高效扩散模型。

## 方法详解

### 关键设计

1. **沙漏编码-解码器**：多级下/上采样扩大感受野。3×3 在 1/4 分辨率下等效原图 12×12。不同分辨率级别处理不同频率信息

2. **稀疏跳跃连接**：不是每个块都跳跃连接（dense skip），而是仅在编码器和解码器对应的少数层间跳跃。减少特征冗余

3. **阶段特定条件注入**：不同分辨率阶段用不同的条件嵌入 + 中间块条件注入 + AdaLN 风格条件门控。GELU 激活替代 SiLU

4. **Winograd 加速**：3×3 卷积可通过 Winograd 变换将乘法量降至原来的 5/9，在 GPU 上直接加速推理

## 实验关键数据（ImageNet 256×256）

| 模型 | FLOPs | FID↓ | IS↑ | 吞吐量↑ |
|------|-------|------|-----|--------|
| DiT-XL/2 | 118.6G | 20.05 (无CFG) | 66.74 | 66.8 |
| U-ViT-XL | 113.0G | 18.35 | 76.59 | 72.6 |
| **DiC-XL** | **57.2G (Wino.)** | **13.11** | **100.15** | **313.7** |

With CFG=1.5: DiC-XL FID **3.89** / IS **224.20** vs DiT-XL/2: 6.24/150.10。吞吐量 4.7×。

### 关键发现
- 沙漏架构是纯卷积扩散的关键——等尺度 Conv 模型无法工作（消融）
- 稀疏跳跃比密集跳跃更好——过多跳跃引入冗余
- Winograd 有效 FLOPs 从 116G 降到 57G（-51%）

## 评分
- 新颖性: ⭐⭐⭐⭐ 证明纯 3×3 卷积也能做好扩散生成是重要发现
- 实验充分度: ⭐⭐⭐⭐⭐ ImageNet 全面对比+详细消融
- 写作质量: ⭐⭐⭐⭐ 每个设计选择都有消融支撑
- 价值: ⭐⭐⭐⭐⭐ 4.7× 吞吐量+更优 FID 对部署有重大价值

<!-- RELATED:START -->

## 相关论文

- [Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](../../ICCV2025/image_generation/rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)
- [CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](../../CVPR2026/image_generation/ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)
- [EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](../../ICCV2025/image_generation/eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)
- [Rethinking Layered Graphic Design Generation with a Top-Down Approach](../../ICCV2025/image_generation/rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)
- [TeRA: Rethinking Text-guided Realistic 3D Avatar Generation](../../ICCV2025/image_generation/tera_rethinking_text-guided_realistic_3d_avatar_generation.md)

<!-- RELATED:END -->
