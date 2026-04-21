---
title: >-
  [论文解读] Learning to See in the Extremely Dark
description: >-
  [图像生成] 提出配对到配对的数据合成管线构建极暗场景（低至0.0001 lux）RAW图像增强数据集SIED，并设计基于扩散模型的框架，通过自适应光照校正模块（AICM）和颜色一致性损失实现极低信噪比RAW图像的高质量恢复。
tags:
  - 图像生成
---

# Learning to See in the Extremely Dark

| 信息 | 内容 |
|------|------|
| 会议 | ICCV2025 |
| arXiv | [2506.21132](https://arxiv.org/abs/2506.21132) |
| 代码 | [JianghaiSCU/SIED](https://github.com/JianghaiSCU/SIED) |
| 领域 | 图像增强 / 低光RAW图像恢复 |
| 关键词 | 极暗场景, RAW图像增强, 扩散模型, 数据合成, 光照校正 |

## 一句话总结

提出配对到配对的数据合成管线构建极暗场景（低至0.0001 lux）RAW图像增强数据集SIED，并设计基于扩散模型的框架，通过自适应光照校正模块（AICM）和颜色一致性损失实现极低信噪比RAW图像的高质量恢复。

## 研究背景与动机

### 核心挑战

低光RAW图像增强需要同时解决：全局/局部对比度增强、噪声抑制、细节保留和颜色映射。现有方法主要在中等低光条件下工作（如SID数据集的0.03-5.0 lux），但当环境照度降至0.0001 lux的极暗场景时，面临两个根本性困难：

**数据缺失**：在极暗条件下无法通过延长曝光获取清晰参考图像（会引入残余噪声和运动模糊）

**性能瓶颈**：信噪比极低导致现有方法产生严重的颜色失真、细节模糊和噪声放大

### 现有方法的局限

- **单阶段方法**（SID、DID等）：用单一网络建模去噪+RAW-to-sRGB两种变换，导致域歧义问题
- **多阶段方法**（LDC、MCR等）：虽然解耦了任务，但在极暗条件下仍面临颜色偏差
- **预放大策略**：大多数方法依赖GT曝光值进行预放大，实际应用中不可用
- **数据集问题**：SID、SDSD、SMID等仅提供粗略照度范围，缺乏精确标定

## 方法详解

### 一、SIED数据集构建：配对到配对合成管线

与传统"从正常光合成低光"不同，本文提出"从低光合成更低光"的策略：

**Step 1：光学实验室标定**
- 使用Sony α7RIII和Canon EOS R在专业光学实验室（配备PHOTO-2000μ照度计）中采集三个照度级别的标准RAW图像：
    - 0.01-0.1 lux
    - 0.001-0.01 lux
    - 0.0001-0.001 lux

**Step 2：真实场景配对采集**
- 在多样真实场景中用三脚架+遥控快门采集低光RAW和正常光sRGB配对
- 参考图曝光时间为低光图的20-200倍
- 裁剪至3840×2160分辨率，每个子集1680对

**Step 3：光照对齐**

$$I_{syn} = I_{cap} \times \left(\frac{\text{Expo}(I_{st})}{\text{Expo}(I_{cap})} + \eta\right)$$

通过ISP管线转换到YUV空间，手动调整 $\eta$ 使Y通道光照直方图匹配标准数据。KL散度小于0.06。

**Step 4：噪声添加**
- 在光学实验室拟合Canon和Sony相机的高斯+泊松噪声分布
- 补充暗帧数据库（dark-frame database）处理极低光下非P+G噪声
- ISO范围：0.01-0.1和0.001-0.01 lux为100-20000，0.0001-0.001 lux为100-40000

最终数据集：每个照度级别每个相机子集1,500训练对 + 180评估对。

### 二、基于扩散模型的增强框架

#### 整体流程

1. RAW编码器 $\mathcal{E}_{raw}$ 和 sRGB编码器 $\mathcal{E}_{rgb}$ 将输入映射到潜空间
2. AICM对RAW特征进行自适应光照校正
3. 扩散模型以校正后的RAW特征为引导，重建sRGB特征
4. sRGB解码器输出最终图像

#### 自适应光照校正模块（AICM）

与依赖GT曝光值的传统预放大不同，AICM从低光RAW特征本身估计放大系数：

- 卷积层嵌入 → 级联卷积 + 自适应平均池化 → 逐通道放大系数 $A_{raw} \in \mathbb{R}^{1 \times 1 \times C}$
- 基于Retinex理论的光照校正损失：

$$\mathcal{L}_{icl} = \|\mathbf{L}_{\hat{\mathcal{F}}_{raw}} - \mathbf{L}_{\tilde{\mathcal{F}}_{raw}}\|_1 + \|\mathbf{R}_{\hat{\mathcal{F}}_{raw}} - \mathbf{R}_{\mathcal{F}_{raw}}\|_1$$

确保光照改善的同时保持反射分量一致。

#### 扩散RAW-to-sRGB重建

**前向扩散**：编码sRGB特征 $\mathcal{F}_{rgb}$ 渐进加噪为高斯噪声

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_t$$

**反向扩散**：以校正后RAW特征 $\hat{\mathcal{F}}_{raw}$ 为条件引导，从噪声恢复干净sRGB特征

$$p_\theta(\hat{\mathbf{x}}_{t-1}|\hat{\mathbf{x}}_t, \tilde{\mathbf{x}}) = \mathcal{N}(\hat{\mathbf{x}}_{t-1}; \boldsymbol{\mu}_\theta(\hat{\mathbf{x}}_t, \tilde{\mathbf{x}}, t), \sigma_t^2\mathbf{I})$$

#### 颜色一致性损失

基于颜色直方图的KL散度约束，促进准确颜色映射：

$$\mathcal{L}_{ccl} = \sum_{c \in [0,C)} \mathcal{H}_{\hat{\mathcal{F}}_{rgb}^c} \log\left(\frac{\mathcal{H}_{\hat{\mathcal{F}}_{rgb}^c}}{\mathcal{H}_{\mathcal{F}_{rgb}^c} + \tau}\right)$$

### 三、两阶段训练

- **阶段一**：优化编解码器和AICM，冻结扩散模型。$\mathcal{L}_{stage1} = \mathcal{L}_{con} + \mathcal{L}_{icl}$
- **阶段二**：优化扩散模型，冻结其他模块。$\mathcal{L}_{stage2} = \mathcal{L}_{cdl} + \lambda\mathcal{L}_{ccl}$（$\lambda=0.1$）

## 实验关键数据

### 主实验：SIED数据集（Canon子集）

| 类型 | 方法 | 0.01-0.1 lux PSNR/SSIM/LPIPS | 0.001-0.01 lux | 0.0001-0.001 lux |
|------|------|--------------------------|----------------|-----------------|
| 单阶段 | SID | 20.69/0.811/0.428 | 20.34/0.799/0.450 | 19.28/0.764/0.497 |
| 单阶段 | SGN | 21.79/0.813/0.421 | 21.07/0.800/0.447 | 19.42/0.762/0.514 |
| 多阶段 | DNF | 24.03/0.813/0.456 | 23.47/0.796/0.486 | 21.63/0.769/0.522 |
| 多阶段 | RAWMamba | 22.63/0.791/0.461 | 21.99/0.782/0.482 | 21.05/0.757/0.521 |
| — | **Ours** | **24.85/0.849/0.360** | **24.02/0.839/0.379** | **22.52/0.811/0.435** |

在所有三个照度级别和所有指标上均达到SOTA。0.0001-0.001 lux级别比次优DNF提升 +0.89 dB PSNR。

### 消融实验

| 变体 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| 单阶段训练 | 22.96 (-1.89) | 0.809 (-0.040) | 0.431 (+0.071) |
| 无AICM | 23.23 (-1.62) | 0.839 (-0.010) | 0.378 (+0.018) |
| 固定放大=100 | 23.74 (-1.11) | 0.841 (-0.008) | 0.373 (+0.013) |
| 固定放大=200 | 23.48 (-1.37) | 0.844 (-0.005) | 0.371 (+0.011) |
| 固定放大=300 | 23.18 (-1.67) | 0.838 (-0.011) | 0.382 (+0.022) |
| 无颜色一致性损失 | 24.53 (-0.32) | 0.836 (-0.013) | 0.389 (+0.029) |
| **完整方法** | **24.85** | **0.849** | **0.360** |

关键发现：
- AICM贡献最大（-1.62 dB），自适应放大显著优于固定放大
- 两阶段训练至关重要（-1.89 dB），早期不稳定的编码特征不利于扩散模型学习
- 颜色一致性损失主要提升SSIM和LPIPS，改善颜色准确性

### SID数据集对比

在公开SID数据集（Sony子集）上同样达到SOTA：PSNR 31.20，比DNF和RAWMamba分别提升 +0.58和 +0.58 dB。且本方法无需GT曝光值进行预放大。

## 亮点与洞察

1. **数据合成创新**：配对到配对策略比传统从正常光合成低光更加真实，光学实验室标定确保照度精确
2. **AICM设计巧妙**：免去对GT曝光信息的依赖，实际应用中更实用
3. **扩散模型的独特应用**：利用扩散模型的生成能力和固有去噪特性，天然适合RAW增强
4. **颜色直方图损失**：用分布级操约束替代像素级，更适合颜色映射任务
5. **真实场景泛化**：合成数据训练的模型可直接迁移到真实极暗场景

## 局限性

- 数据合成依赖高成本的光学实验室设备
- 仅支持Sony和Canon两款相机，跨相机泛化待验证
- 扩散模型推理速度较慢（20步采样），实时应用受限
- 极暗场景下PSNR绝对值仍不高（22.52 dB），信息损失不可逆

## 相关工作与启发

- **SID**（CVPR 2018）：RAW低光增强的开创性工作，但照度范围有限
- **DNF**（CVPR 2023）：多阶段方法的代表，本文的主要竞争对手
- **Retinex理论**：用于光照校正损失的设计灵感
- 该数据合成思路可推广到其他极端条件（如水下、雾天）的数据构建

## 评分

⭐⭐⭐⭐ — 数据集贡献突出（填补极暗场景空白），方法设计合理有效，但实验主要在自建数据集上验证。

<!-- RELATED:START -->

## 相关论文

- [Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](regen_learning_compact_video_embedding_with_re-generative_decoder.md)
- [DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)
- [FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)

<!-- RELATED:END -->
