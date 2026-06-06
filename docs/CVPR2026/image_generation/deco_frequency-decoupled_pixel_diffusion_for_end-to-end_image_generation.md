---
title: >-
  [论文解读] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation
description: >-
  [CVPR 2026][图像生成][像素扩散] DeCo 提出频率解耦的像素扩散框架，用轻量像素解码器处理高频细节并让DiT专注低频语义建模，配合频率感知flow matching损失，在ImageNet上达到FID 1.62（256）和2.22（512），缩小了像素扩散与潜空间扩散的差距。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "像素扩散"
  - "频率解耦"
  - "端到端生成"
  - "Transformer"
  - "频率感知损失"
---

# DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation

**会议**: CVPR 2026  
**arXiv**: [2511.19365](https://arxiv.org/abs/2511.19365)  
**代码**: [https://github.com/Zehong-Ma/DeCo](https://github.com/Zehong-Ma/DeCo)  
**领域**: 图像生成  
**关键词**: 像素扩散, 频率解耦, 端到端生成, 扩散Transformer, 频率感知损失

## 一句话总结
DeCo 提出频率解耦的像素扩散框架，用轻量像素解码器处理高频细节并让DiT专注低频语义建模，配合频率感知flow matching损失，在ImageNet上达到FID 1.62（256）和2.22（512），缩小了像素扩散与潜空间扩散的差距。

## 研究背景与动机
1. **领域现状**：潜空间扩散（LDM）是主流范式，但依赖VAE的两阶段流程引入有损重建和分布偏移。像素扩散直接在像素空间端到端建模，避免VAE限制但训练和推理效率低。
2. **现有痛点**：现有像素扩散模型用单个DiT同时建模高频信号和低频语义，高频噪声难以学习且干扰低频语义学习，导致训练缓慢和生成质量不佳。
3. **核心矛盾**：DiT擅长捕捉低频语义但不善处理高频信号，而像素空间同时包含两者。
4. **本文目标**：设计更高效的像素扩散范式，解耦高低频的建模任务。
5. **切入角度**：受"高频信号在高分辨率输入上更容易重建、低频语义在低分辨率上更容易建模"的启发。
6. **核心idea**：DiT处理下采样输入专注低频语义，轻量像素解码器在全分辨率上以DiT输出为条件生成高频细节。

## 方法详解

### 整体框架
输入图像下采样后通过DiT建模低频语义 $x_{\text{low}} = \text{DiT}(\bar{x}_t, t, y)$，再由像素解码器以 $x_{\text{low}}$ 为条件在全分辨率上预测像素速度 $v_\theta(x_t, t, y) = \text{Dec}(x_t, t, x_{\text{low}})$。训练目标结合标准FM损失、频率感知FM损失和REPA对齐损失。

### 关键设计

1. **频率解耦架构**:
    - 功能：将高频和低频建模分配给不同模块
    - 核心思路：DiT从下采样输入建模低频语义；轻量像素解码器是无注意力的线性网络，由N个线性块和投影层组成，直接在全分辨率噪声图像上操作，以DiT输出为条件生成高频细节。多尺度输入策略是关键——像素解码器的高分辨率输入使其天然适合高频建模。
    - 设计动机：DCT能量分析证实DeCo成功将高频成分从DiT转移到像素解码器中，DiT输出的高频能量显著低于baseline。

2. **AdaLN条件交互**:
    - 功能：将DiT的低频语义注入像素解码器
    - 核心思路：将DiT输出上采样到全分辨率，通过MLP生成调制参数 $\alpha, \beta, \gamma$，用AdaLN-Zero方式调制解码器中的dense query：$h_N = h_{N-1} + \alpha \cdot \text{MLP}((1+\gamma) \cdot h_{N-1} + \beta)$。
    - 设计动机：AdaLN提供了比简单相加更有效的条件注入方式，实验证实优于UNet式上采样相加。

3. **频率感知FM损失**:
    - 功能：强调视觉显著频率，抑制不重要的高频成分
    - 核心思路：将预测和GT速度转换到YCbCr色彩空间后做8×8 DCT变换到频率域，使用JPEG量化表的归一化倒数作为自适应权重：量化区间越小的频率越重要。加权后计算频率域MSE：$\mathcal{L}_{\text{FreqFM}} = \mathbb{E}[w\|\mathbb{V}_\theta - \mathbb{V}_t\|^2]$。
    - 设计动机：标准FM损失对所有频率一视同仁，但人眼对不同频率的敏感度差异很大。JPEG量化表编码了关于视觉重要性的鲁棒先验。

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{\text{FM}} + \mathcal{L}_{\text{FreqFM}} + \mathcal{L}_{\text{REPA}}$。50步Euler采样。

## 实验关键数据

### 主实验

| 方法 | 类型 | FID↓ (256) | FID↓ (512) | IS↑ | 说明 |
|------|------|-----------|-----------|-----|------|
| DeCo | 像素扩散 | 1.62 | 2.22 | 294.6 | 像素扩散SOTA |
| DiT-XL/2 | 潜空间 | 2.27 | - | 278.2 | 需要VAE |
| REPA-XL/2 | 潜空间 | 1.42 | - | 305.5 | 当前最优LDM |
| PixelFlow | 像素扩散 | 54.33 | - | 24.67 | 多尺度方法 |
| Baseline | 像素扩散 | 61.10 | - | 16.81 | 未解耦 |

### 消融实验

| 配置 | FID↓ | 说明 |
|------|-----|------|
| DeCo完整 | 31.35 | 200K迭代 |
| w/o FreqFM | 34.12 | 频率损失有效 |
| w/o REPA | 67.55 | REPA对齐很重要 |
| Baseline | 61.10 | 无解耦 |

### 关键发现
- DeCo在400K迭代时达到2.57 FID，比baseline收敛快10倍。
- 频率解耦的关键在于多尺度输入策略和AdaLN交互，缺一不可。
- 像素解码器极其轻量（无注意力），仅增加3%参数却带来巨大收益。
- 文本到图像生成也表现优异：GenEval 0.86，DPG-Bench 81.4。

## 亮点与洞察
- **频率解耦的思路**简洁有力：让不同模块做它最擅长的事。
- **JPEG量化表作为感知先验**是一个优雅的trick，零成本引入了人眼感知知识。
- 像素扩散终于能与潜空间扩散竞争，证明VAE不是必须的。

## 局限与展望
- 在512分辨率上仍略逊于最强的LDM，但差距在缩小。
- 像素解码器的hidden dimension和层数需要调参。
- 未来可探索更强的频率解耦方案或与JiT等并行工作结合。

## 相关工作与启发
- **vs PixelFlow**: 用不同分辨率阶段的级联方式，但每个阶段仍需同时处理所有频率。DeCo在每个时间步内同时解耦。
- **vs DDT**: 在潜空间做单尺度频率解耦，DeCo是像素空间的多尺度方案。

## 评分
- 新颖性: ⭐⭐⭐⭐ 频率解耦思路清晰但不算革命性
- 实验充分度: ⭐⭐⭐⭐⭐ 256/512/T2I多场景验证，消融深入
- 写作质量: ⭐⭐⭐⭐⭐ 分析透彻，图表说服力强
- 价值: ⭐⭐⭐⭐⭐ 推动像素扩散重新成为竞争性方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] PixelDiT: Pixel Diffusion Transformers for Image Generation](pixeldit_pixel_diffusion_transformers_for_image_generation.md)
- [\[ICML 2026\] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](../../ICML2026/image_generation/end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)
- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](../../ICCV2025/image_generation/end-to-end_multi-modal_diffusion_mamba.md)
- [\[CVPR 2026\] Bias at the End of the Score: Demographic Biases in Reward Models for T2I](bias_reward_models_t2i.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)

</div>

<!-- RELATED:END -->
