---
title: >-
  [论文解读] MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance
description: >-
  [ICCV 2025][图像生成][图像融合] MMAIF 提出统一的多任务、多退化、语言引导图像融合框架，通过实际退化流水线和现代化 DiT 架构在潜在空间操作，同时提供回归和 Flow Matching 两个版本，在各类退化融合任务上超越现有 restoration+fusion 流水线。
tags:
  - ICCV 2025
  - 图像生成
  - 图像融合
  - Transformer
  - 多任务
  - 多退化
  - 语言引导
  - Flow Matching
  - MoE
---

# MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance

**会议**: ICCV 2025  
**arXiv**: [2503.14944](https://arxiv.org/abs/2503.14944)  
**代码**: [GitHub](https://github.com/294coder/MMAIF)  
**领域**: 扩散模型·图像融合  
**关键词**: 图像融合, Diffusion Transformer, 多任务, 多退化, 语言引导, Flow Matching, MoE  

## 一句话总结

MMAIF 提出统一的多任务、多退化、语言引导图像融合框架，通过实际退化流水线和现代化 DiT 架构在潜在空间操作，同时提供回归和 Flow Matching 两个版本，在各类退化融合任务上超越现有 restoration+fusion 流水线。

## 研究背景与动机

图像融合旨在将多模态/多参数图像序列整合为单张输出（如红外与可见光融合 VIF、多曝光融合 MEF、多焦点融合 MFF）。现有方法面临四大问题：

**任务专属模型**：为每种融合任务训练单独网络，VIF 模型无法直接用于 MEF

**忽略真实退化**：训练在干净图像上，面对噪声、模糊、雨雪等退化时失效

**像素空间计算昂贵**：Transformer 的二次复杂度在像素空间难以承受

**缺乏用户交互**：无法通过语言指令引导恢复和融合

传统解决方案是在融合前串联图像恢复网络，但这增加了推理复杂度且恢复后的图像可能导致融合失败。

## 方法详解

### 1. 真实退化流水线

为 VIF、MEF、MFF 三种任务设计专属退化策略：
- **通用退化**：高斯模糊、运动模糊、下采样、高斯噪声、雨、雾、雪
- **VIF 特定**：低曝光、低对比度、红外暗条纹
- **MEF 特定**：低对比度
- **MFF 特定**：低/高曝光

每对图像随机采样 $n \in \{1, 3\}$ 种退化组合，模拟复合退化场景。利用 DepthAnything 估计深度后应用大气散射模型添加更真实的雾效果。

GT 由预训练的 SwinFusion 和 DeFuse 生成；利用 ChatGPT 为每种退化生成 10-20 条交互提示。

### 2. 图像分词器选择

比较三种 VAE（$f=8, z=16$）：

| 分词器 | PSNR | SSIM |
|--------|------|------|
| Flux KL-VAE | 33.41 | 0.9227 |
| Asy. KL-VAE | 33.10 | 0.9201 |
| **Cosmos VAE** | **34.02** | **0.9367** |

选用重建性能最优的 Cosmos VAE。

### 3. 现代化 DiT 架构

对原始 DiT 进行多项改进：

- **MoE GLU**：将 FFN 替换为 4 专家 + 1 共享专家的 MoE，通过 token 路由分配，使用负载均衡损失。相比密集模型，提供更大容量和更低 FLOPs
- **2D RoPE**：替换绝对位置编码，支持更好的分辨率泛化和长度外推
- **逐块绝对位置编码**：在每个 block 前添加可学习 PE，消除变分辨率推理的伪影
- **注意力值残差**：$V^l = (1-\eta) \cdot W^V X + \eta V^{l-1}$，缓解深层网络梯度消失
- **LoRA AdaLN 条件注入**：将条件 MLP 分解为两个小 MLP 以减少参数
- **NAFNet 偏置卷积**：在注意力前加入卷积块，提供归纳偏置以改善模糊退化处理

### 4. 回归与 Flow Matching 双版本

**回归版本**（移除时间步嵌入）：

$$\mathcal{L}_{reg} = \|f_\theta(Z_0^m, Z_1^m, P) - Z_{GT}\|_2^2$$

**Flow Matching 版本**：

$$\mathcal{L}_{flow} = \mathbb{E}\|v_\theta(Z_t, t, Z_0^m, Z_1^m, P) - (GT - X_0)\|_2^2$$

辅助融合损失：$\mathcal{L}_{aux} = \sum_{i=0}^{m-1} \|\tilde{X} - X_i\|_1 + \|\nabla\tilde{X} - \nabla X_i\|_1$

## 实验

### 消融实验

| 组件 | PSNR (VIF) | SSIM (VIF) |
|------|-----------|-----------|
| 基础 DiT | 31.02 | 0.892 |
| + MoE | 31.45 | 0.901 |
| + RoPE | 31.62 | 0.908 |
| + 值残差 | 31.78 | 0.912 |
| + NAFNet 卷积 | **32.15** | **0.921** |

每项改进都带来一致的性能提升，其中 NAFNet 卷积对模糊退化帮助最大。

### 与 Restoration+Fusion 流水线对比

| 方法 | 推理流程 | PSNR | SSIM | 推理时间 |
|------|---------|------|------|---------|
| Restormer+SwinFusion | 两阶段 | 29.87 | 0.874 | 慢 |
| TextIF | 单阶段（像素空间） | 30.45 | 0.889 | 中 |
| **MMAIF-Reg** | **单阶段（潜在空间）** | **32.15** | **0.921** | **快** |

MMAIF 在简化推理流程的同时大幅超越现有方法。

## 亮点与洞察

1. **三合一框架**：同时解决多任务（VIF/MEF/MFF）、多退化和语言引导三个问题
2. **回归+Flow Matching 双版本**：回归版本快速推理，Flow Matching 版本在弱先验退化（雪、雨）上更优
3. **现代化 DiT 改进**的每一项都有理论动机和消融验证
4. **潜在空间操作**大幅降低了 Transformer 的计算开销

## 局限性

- GT 由预训练网络生成而非真实标注，可能引入偏差
- 仅支持两张图像的融合，多图像融合需要扩展
- MoE 增加了模型复杂度和训练不稳定性
- 对比实验中部分基线未在完全相同条件下评估

## 相关工作

- 图像融合：U2Fusion、SwinFusion、PSLPT 等
- 退化图像恢复与融合：TextIF、DRMF、Text-DiFuse
- Diffusion Transformer：DiT、Flux、SD3 等架构

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| **综合** | **4.0** |

<!-- RELATED:START -->

## 相关论文

- [M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization](m2sformer_multi-spectral_and_multi-scale_attention_with_edge-aware_difficulty_gu.md)
- [StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](../../CVPR2025/image_generation/gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)
- [Multi-turn Consistent Image Editing](multi-turn_consistent_image_editing.md)
- [UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer](unicombine_unified_multi-conditional_combination_with_diffusion_transformer.md)

<!-- RELATED:END -->
