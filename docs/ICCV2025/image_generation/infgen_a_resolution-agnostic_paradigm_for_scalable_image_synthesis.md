---
title: >-
  [论文解读] InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis
description: >-
  [ICCV 2025][图像生成][任意分辨率生成] 提出InfGen，一种"第二代生成"范式，用一个基于Transformer的生成器替换VAE解码器，从固定大小的latent一步解码出任意分辨率图像，无需修改或重新训练扩散模型，将4K图像生成时间压缩至10秒以内，比现有最快方法UltraPixel提速10倍以上。
tags:
  - ICCV 2025
  - 图像生成
  - 任意分辨率生成
  - 潜在扩散模型
  - VAE解码器替换
  - 隐式神经位置编码
  - 高效4K生成
---

# InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis

**会议**: ICCV 2025  
**arXiv**: [2509.10441](https://arxiv.org/abs/2509.10441)  
**代码**: [GitHub](https://github.com/taohan10200/InfGen)  
**领域**: 图像生成  
**关键词**: 任意分辨率生成, 潜在扩散模型, VAE解码器替换, 隐式神经位置编码, 高效4K生成

## 一句话总结

提出InfGen，一种"第二代生成"范式，用一个基于Transformer的生成器替换VAE解码器，从固定大小的latent一步解码出任意分辨率图像，无需修改或重新训练扩散模型，将4K图像生成时间压缩至10秒以内，比现有最快方法UltraPixel提速10倍以上。

## 研究背景与动机

- **任意分辨率图像生成的需求**：不同设备（手机、4K显示器等）需要一致的视觉体验
- **现有方法的瓶颈**：
    - 扩散模型的计算需求随分辨率二次增长，生成4K图像延迟超过100秒
    - Training-free方法（如ScaleCrafter、FouriScale）通过调整推理过程（膨胀卷积等）实现超分辨率，但与特定网络架构绑定、泛化性差
    - Inf-DiT等方法重新设计了注意力机制但速度仍慢（255秒生成2K图像）
    - UltraPixel需要微调扩散模型本身
- **关键洞察**：
    - 生成模型已经完成了内容生成（第一阶段），第二阶段的解码器只需一次前向传播
    - 增强解码器能力来实现高分辨率图像生成是更高效的路径
    - 将latent到高分辨率图像的映射本质上是超分辨率任务，但需要生成能力来补充信息损失

## 方法详解

### 整体框架

InfGen是一个"二次生成"模型：
1. **第一阶段**：扩散模型生成固定大小的content latent z（如4×64×64）
2. **第二阶段**：InfGen将z解码为任意分辨率的图像

$$f: \text{InfGen}(z, (h, w)) \rightarrow x_{(h,w)}$$

这种范式带来两大优势：（1）高推理速度——避免在高分辨率latent上做多步去噪；（2）即插即用——适用于任何基于相同latent空间的扩散模型。

### 任意分辨率解码器架构

基于传统VAE结构，引入Transformer-based latent生成器：
- 将latent变量z作为**keys和values**
- 创建与目标图像尺寸(h,w)对应的**mask token**作为query，形状为(⌈h/8⌉, ⌈w/8⌉)
- 在多层Transformer块中，mask token通过**交叉注意力**机制与latent keys交互获取信息
- 最终将mask token送入解码器上采样，生成最终的任意分辨率图像

### 隐式神经位置编码（INPE）

解决固定latent与动态大小mask token之间的空间信息对齐问题：

1. **坐标标准化**：将mask token和latent token坐标映射到统一尺度
$$(\hat{x}^m, \hat{y}^m) = (x^m/W^m, y^m/H^m)$$

2. **球面映射**：将标准化的2D坐标转换为单位球面上的3D笛卡尔坐标，利用球面几何捕获复杂空间关系

3. **傅里叶变换+神经网络映射**：通过高频傅里叶特征增强模式捕获能力
$$\gamma(x,y,z) = [\cos(B[x,y,z]^T), \sin(B[x,y,z]^T)]$$
然后通过隐式神经网络生成动态位置编码

### 训练目标

$$L_{AE} = \ell_1(x, \hat{x}) + \lambda_P \mathcal{L}_P(x, \hat{x}) + \lambda_G \mathcal{L}_G(\hat{x})$$

其中 $\ell_1$ 为L1重建损失，$\mathcal{L}_P$ 为LPIPS感知损失，$\mathcal{L}_G$ 为PatchGAN判别器的对抗损失。λ_P和λ_G均设为0.1。

### 无训练分辨率外推

通过迭代方式实现超越训练分辨率的超高分辨率（如4K）生成：
$$L_n = \text{Encoder}(I_{n-1}), \quad I_n = \text{InfGen}(L_n, k_n^s)$$

最终分辨率：$R_f = 512 \cdot \prod_{i=1}^n s_i^h \times 512 \cdot \prod_{i=1}^n s_i^w$

## 实验关键数据

### 主实验：图像Tokenizer重建质量对比

| 方法 | 输入→输出分辨率 | ImageNet rFID↓ | PSNR↑ | SSIM↑ |
|------|----------------|---------------|-------|-------|
| VQGAN | 256²→256² | 1.19 | 23.38 | 0.762 |
| SD-VAE | 256²→256² | 0.74 | 25.68 | 0.820 |
| SDXL-VAE | 256²→256² | 0.68 | 26.04 | 0.834 |
| InfGen | 256²→256² | 1.07 | 24.61 | 0.798 |
| InfGen | 512²→512² | 0.61 | 27.92 | 0.867 |
| SD-VAE | 256²→512² | 1.43 | 24.14 | 0.759 |
| **InfGen** | **256²→512²** | **1.15** | **22.86** | **0.728** |

InfGen在跨分辨率重建（256²→512²）任务上明显优于SD-VAE。

### 对扩散模型的高分辨率提升

| 方法 | 512² FIDp↓ | 1024² FIDp↓ | 2048² FIDp↓ | 3072² FIDp↓ |
|------|-----------|------------|------------|------------|
| DiT-XL/2 | 44.17 | 61.52 | 64.87 | 77.84 |
| InfGen+DiT | 39.81 (↓9.9%) | 41.75 (↓32%) | 56.21 (↓13.4%) | 45.94 (↓41%) |
| SD1.5 | 21.58 | 55.30 | - | - |
| InfGen+SD1.5 | 16.92 (↓21%) | 41.12 (↓26%) | - | - |
| FiTv2 | 42.04 | 66.95 | - | 79.30 |
| InfGen+FiTv2 | 38.77 (↓7.8%) | 61.56 (↓8.1%) | - | 45.72 (↓42%) |

关键发现：InfGen在所有分辨率和模型组合上都有显著提升，3072²分辨率下提升最高达44%。

### 与SOTA高分辨率生成方法对比

| 方法 | 1024² FIDp↓ | 2048² FIDp↓ | 1024² Latency(s) | 2048² Latency(s) |
|------|-----------|-----------|-----------------|-----------------|
| ScaleCrafter | 55.36 | 144.61 | 7 | 97 |
| Inf-DiT | 48.48 | 142.05 | 50 | 255 |
| UltraPixel | 48.37 | 127.26 | 11 | 20 |
| InfGen+SD1.5 | 44.85 | 139.14 | 2.9+0.4 | 2.9+1.9 |
| **InfGen+SDXL** | **35.14** | **96.41** | **5.4+0.4** | **5.4+1.9** |

InfGen+SDXL在FID和速度上均大幅领先，生成2K图像仅需~7秒，比UltraPixel快4倍，比Inf-DiT快35倍。

### 训练设置

- 训练数据：LAION-Aesthetic中分辨率>1024²的500万张图像，以及>2048²的子集
- 两阶段训练：第一阶段512²→1024²（batch=32，500k迭代），第二阶段512²→2048²（batch=8，100k迭代）
- 硬件：8×A100，训练15天
- 优化器：AdamW，学习率从2e-4余弦衰减到1e-5

## 亮点与洞察

1. **范式转换**：将任意分辨率生成问题从"修改扩散模型"转变为"增强解码器"，大幅降低了复杂度和计算成本
2. **即插即用设计**：InfGen不修改VAE编码器，因此可作为现有扩散模型的直接升级补丁，适用于SD1.5、SDXL、DiT、SiT等
3. **隐式神经位置编码**：通过球面映射+傅里叶变换解决了固定latent与动态分辨率之间的空间对齐问题
4. **迭代外推策略**：无训练地将生成能力扩展到任意超高分辨率，每次迭代2倍放大

## 局限性

- InfGen在256²同分辨率重建上略弱于原始VAE（rFID 1.07 vs 0.74），因为其任务更复杂
- 迭代外推过程需要多次编码-解码，每次额外引入信息损失
- 对抗损失的训练可能导致生成内容与原始语义有偏差
- 当前仅在SDXL的VAE latent空间上训练，迁移到其他latent空间需要重新训练

## 相关工作与启发

- **与超分辨率方法的关系**：本质上是一种条件生成式超分辨率，但以latent为条件而非低分辨率图像
- **与VAE改进的关系**：不改进VAE本身，而是在其之上添加一层生成能力
- **启发**：二次生成范式将"分辨率"从扩散模型的负担中解耦出来，使得高分辨率/任意分辨率生成不再需要在diffusion阶段解决

## 评分 ⭐⭐⭐⭐

思路清晰，方法简洁高效。即插即用的设计理念和10倍速度提升使其具有很强的实用价值。隐式神经位置编码设计巧妙。

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion](../../NeurIPS2025/image_generation/scalediff_higher-resolution_image_synthesis_via_efficient_and_model-agnostic_dif.md)
- [\[ICCV 2025\] LIFT: Latent Implicit Functions for Task- and Data-Agnostic Encoding](lift_latent_implicit_functions_for_task-_and_data-agnostic_encoding.md)
- [\[ICCV 2025\] PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)
- [\[ICCV 2025\] DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching](diffumatch_category-agnostic_spectral_diffusion_priors_for_robust_non-rigid_shap.md)
- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)

<!-- RELATED:END -->
