---
title: >-
  [论文解读] ZeST: Zero-Shot Material Transfer from a Single Image
description: >-
  [ECCV 2024][3D视觉] 提出ZeST，一种零样本免训练的材质迁移方法，通过IP-Adapter提取材质表示、ControlNet提供几何引导、前景灰度图提供光照线索，三条分支组合实现从单张材质样本图像到目标物体的2D材质迁移。
tags:
  - ECCV 2024
  - 3D视觉
---

# ZeST: Zero-Shot Material Transfer from a Single Image

**会议**: ECCV 2024  
**arXiv**: [2404.06425](https://arxiv.org/abs/2404.06425)  
**代码**: [项目页](https://ttchengab.github.io/zest)  
**领域**: 3D视觉

## 一句话总结

提出ZeST，一种零样本免训练的材质迁移方法，通过IP-Adapter提取材质表示、ControlNet提供几何引导、前景灰度图提供光照线索，三条分支组合实现从单张材质样本图像到目标物体的2D材质迁移。

## 研究背景与动机

### 领域现状

**领域现状**：编辑图像中物体的材质（如大理石变钢铁）在游戏设计、电商等应用中极有价值

### 现有痛点

**现有痛点**：传统方法需要显式3D几何、光照估计和材质参数指定，过程复杂

### 核心矛盾

**核心矛盾**：文本驱动方法难以精确描述材质的纹理细节

### 解决思路

**解决思路**：TextureDreamer等方法需要3-5张材质图像做Dreambooth微调，耗时且不可扩展

### 补充说明

**补充说明**：核心问题**：如何从单张材质样本图像、在无需训练的条件下，将材质迁移到目标图像上的物体

## 方法详解

### 整体框架

三条并行分支输入到Stable Diffusion XL Inpainting：
1. **材质编码分支**：IP-Adapter编码材质样本提取材质隐表示z_M
2. **几何引导分支**：DPT估计深度图→ControlNet提供结构约束
3. **光照引导分支**：制作前景灰度图Iinit→Inpainting模型初始化

$$I_{gen} = \mathcal{S}(z_M, D_I, I_{init}, F)$$

### 关键设计

**材质编码（IP-Adapter）**：
- 利用CLIP图像编码器提取材质样本的特征表示
- 通过cross-attention注入到扩散模型中
- 无需Dreambooth微调，单张图像即可

**几何引导（ControlNet）**：
- Depth-based ControlNet从输入图像的深度图获取结构信息
- 覆盖材质编码z_M中的几何信息，确保生成物体保持原始形状
- IP-Adapter + Img2Img无法保持原始几何（关键发现）

**前景灰度化的光照引导（核心设计选择）**：
- 直接用原图：原物体颜色作为强先验干扰材质颜色（橙色南瓜）
- 随机噪声初始化：丢失光影方向信息
- **前景灰度图（最优）**：移除颜色先验的同时保留光影信息
- $I_{init} = F \odot I_{gray} + (1-F) \odot I$

**实现细节**：
- 深度估计用DPT，前景提取用Rembg
- 基于SDXL Inpainting + 对应版本的ControlNet和IP-Adapter
- 单张A-10 GPU，约15秒生成一张

### 损失函数

不涉及训练，所有过程在预训练模型的推理阶段完成。

## 实验关键数据

### 主实验

合成数据集定量对比（9材质×10网格=90组）：

| 方法 | PSNR↑ | LPIPS↓ | CLIP↑ |
|------|-------|--------|-------|
| IP-Adapter + InstructPix2Pix | 16.92 | 0.096 | 0.745 |
| Dreambooth + Geo/Illum Guidance | 25.46 | 0.053 | 0.893 |
| **ZeST** | **25.82** | **0.046** | **0.899** |

用户研究（真实图像，1-5分）：

| 方法 | 材质保真度↑ | 真实感↑ |
|------|-----------|---------|
| IP-Adapter + InstructPix2Pix | 1.48 | 3.23 |
| Dreambooth + Geo/Illum Guidance | 3.25 | 3.41 |
| **ZeST** | **4.05** | **3.78** |

### 消融实验

光照引导方式对比验证：原图保留物体底色干扰材质色（Setting 1）；随机噪声导致光影方向错误（Setting 2）；前景灰度图最优（Setting 3）。

鲁棒性测试：
- 改变材质样本的光照方向和旋转角度→生成结果高度一致
- 缩放材质样本图像→模型自动调整纹理尺度适配目标物体

### 关键发现

- ZeST在材质保真度用户评分上大幅领先（4.05 vs 3.25），零样本方法胜过微调方法
- 前景灰度化是光照引导的最优选择（核心设计choices的价值）
- ZeST对材质样本的光照/旋转/缩放变化鲁棒
- Dreambooth编码过程会丢失材质信息并引起色偏（尤其在真实场景）
- 可扩展到多物体编辑（配合SAM迭代）和光照感知材质迁移

## 亮点与洞察

- 纯工程但极其精巧的pipeline设计——三条分支各司其职、完全免训练
- 前景灰度化的洞察简单却关键：去色→去颜色先验，保留灰度→保光影
- 作为新问题（2D-to-2D材质迁移）的开创者，提出了合成和真实评估数据集
- 可与Text2Tex等3D纹理方法联合使用，将材质样本驱动的纹理化引入3D

## 局限与展望

- 有时仅在物体最"可能"的区域迁移材质（部分迁移问题）
- 材质样本中包含多种材质时可能混合
- IP-Adapter缺乏区域级材质提取能力
- 扩散模型的隐空间控制有时不可预测

## 评分

- 新颖性：⭐⭐⭐⭐ — 新问题定义 + 巧妙的免训练方案
- 有效性：⭐⭐⭐⭐ — 用户研究显著领先
- 实用性：⭐⭐⭐⭐⭐ — 零样本、免训练、15秒生成
- 推荐度：⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)

<!-- RELATED:END -->
