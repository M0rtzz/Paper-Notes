---
title: "Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation"
conference: "CVPR 2025"
arxiv: "2408.13509"
code: "https://github.com/yinyjin/DualAnoDiff"
project: ""
领域: "图像生成"
关键词: ["异常检测", "小样本生成", "扩散模型", "工业缺陷", "双分支扩散"]
评分: "⭐⭐⭐⭐"
---

# DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation

## 一句话总结

提出 DualAnoDiff，通过双相互关联扩散模型（全局分支生成整体异常图像+异常分支生成局部异常部分）同时生成高质量的异常图像-掩码对，并引入背景补偿模块维持背景和物体形状的一致性，显著提升下游异常检测/定位/分类的性能。

## 研究背景与动机

工业异常检测面临核心挑战：**异常数据极度稀缺**。现有方法局限：
- **无模型方法**(DRAEM、Cut-Paste)：随机裁剪粘贴纹理生成异常，结果不逼真
- **GAN 方法**(SDGAN、DefectGAN)：需要大量异常数据训练，且无法生成异常掩码
- **DFMGAN**：先在正常数据上预训练 StyleGAN2 再迁移到异常域，但生成的异常不够逼真且掩码对齐差
- **AnomalyDiffusion**：基于 textual inversion 分别学习异常外观和掩码位置，导致两个问题：(1) 异常与原图融合不自然；(2) 掩码可能出现在图像背景区域

本文的核心洞察：**同时生成整体异常图像和对应异常部分**，天然保证掩码与异常的高度对齐，而非分别学习再拼合。

## 方法详解

### 整体框架

DualAnoDiff 基于 Stable Diffusion 构建双扩散分支：全局分支 $SD$ 生成完整异常图像 $I$，异常分支 $SD^*$ 生成仅包含异常区域的图像 $I_a = I \times M_a$。两个分支共享时间步、通过自注意力交互模块(SAIM)交换信息，并使用嵌套提示词引导各自生成。背景补偿模块(BCM)注入背景信息维持形状和背景稳定性。

### 关键设计

#### 1. 双相互关联扩散 + 嵌套提示词

- **功能**：同时生成异常图像 $I$ 和对应的异常部分 $I_a$，确保掩码与异常高度对齐
- **核心思路**：冻结 SD 权重，使用两组 LoRA 分别微调全局分支和异常分支。两个分支使用嵌套提示词 $p$: "a **x** with **y**"（全局）和 $p'$: "**y**"（异常），其中 x=vfx, y=sks 是弱先验词。同一时间步添加噪声和去噪，通过 SAIM 共享信息
- **设计动机**：通过同时生成而非分别学习来解决异常-掩码对齐问题。嵌套提示词反映了包含关系，让模型正确分离物体属性和异常属性。使用弱先验词(vfx, sks)比高频词更容易拟合

#### 2. 自注意力交互模块 (SAIM)

- **功能**：在去噪过程中同步两个分支的信息，保持生成的整体图像与局部异常的一致性
- **核心思路**：每个 attention block 后，将两个分支的中间特征 $\varphi_i(z)$ 和 $\varphi_i(z')$ 拼接、重排列后执行共享自注意力计算，再拆分回各分支并添加残差连接。self-attention 后共享位置和细节信息，cross-attention 后共享语义信息
- **设计动机**：直接独立生成两路图像会导致异常位置、形态不一致。SAIM 通过共享注意力实现隐式的空间和语义对齐，无需额外对齐损失

#### 3. 背景补偿模块 (BCM)

- **功能**：注入背景信息和物体掩码形状，防止小样本训练下的物体变形、背景混淆
- **核心思路**：用 U2-Net 分割物体得到背景 $I_b = (1-M_f) \times I$，将背景图像送入 SD 编码得到中间特征。在每层自注意力中，将背景特征通过自适应 MLP 融合到全局分支：$\varphi_i(z) = \varphi_i(z) + \gamma \cdot MLP(\varphi_i(z^b))$，$\gamma$ 初始化为 0.1
- **设计动机**：小样本(~8 张)训练极易导致过拟合，表现为物体形状扭曲、背景颜色污染、物体与背景耦合等问题（如瓶子只生成一半、牙刷出现双柄）。BCM 将背景和物体形状作为显式约束，使模型专注于异常区域的生成

### 损失函数

$$\mathcal{L} = \mathbb{E}_{\mathcal{E}(I),\epsilon,t}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(p), I_b)\|_2^2\right] + \mathbb{E}_{\mathcal{E}(I_a),\epsilon^*,t}\left[\|\epsilon^* - \epsilon_\theta^*(z_t', t, \tau_\theta(p'))\|_2^2\right]$$

## 实验关键数据

### 主实验表 (MVTec AD 生成质量)

| 方法 | IS ↑ (平均) | IC-LPIPS ↑ (平均) |
|------|-----------|------------------|
| CDC | 1.65 | 0.07 |
| Crop-Paste | 1.51 | 0.14 |
| SDGAN | 1.71 | 0.13 |
| DFMGAN | 1.72 | 0.20 |
| AnomalyDiffusion | 1.79 | 0.32 |
| **DualAnoDiff (ours)** | **1.93** | **0.38** |

### 下游异常检测性能

| 方法 | 像素级 AUROC ↑ | 像素级 AP ↑ |
|------|--------------|------------|
| 无异常数据 | 96.8 | 64.1 |
| AnomalyDiffusion | 98.8 | 82.3 |
| **DualAnoDiff (ours)** | **99.1** | **84.5** |

### 消融表

- SAIM 贡献：移除 SAIM 后 IS 从 1.93 降至约 1.75，生成图像与异常部分的一致性显著下降
- BCM 贡献：移除 BCM 后出现明显的物体变形和背景混淆（瓶子不完整、牙刷双柄）
- 嵌套提示词：vfx/sks 弱先验词优于真实类别名/异常名

### 关键发现

- **同时生成 vs 分别生成**：同时生成的异常图像在逼真度、多样性和掩码对齐度上全面优于分别生成(AnomalyDiffusion)
- **掩码获取简便**：由于异常分支生成独立的异常部分，用 SAM/U2-Net 即可轻松获取高精度掩码
- **极少样本即可工作**：平均每类仅 ~8 张异常样本训练，即可生成高质量异常数据

## 亮点与洞察

1. **双流同步生成的设计**：最关键的创新——通过同时生成整体和部分来解决掩码对齐问题，将难处理的"生成+分割"变成优雅的"并行生成+简单分割"
2. **SAIM 的隐式对齐**：无需显式对齐损失，通过共享注意力计算自然实现两个分支的空间-语义一致性
3. **BCM 解决小样本过拟合**：将背景信息作为显式先验注入，有效缓解小样本训练下的各种退化现象
4. **端到端实用**：直接为下游异常检测/定位/分类提供训练数据，AUROC 99.1% 达到 SOTA

## 局限与展望

1. **推理效率**：双分支同时去噪的计算成本约为单分支的 2 倍
2. **依赖 U2-Net 分割**：BCM 和掩码获取都依赖预训练分割模型的质量
3. **仅验证 MVTec AD**：未在更多异常检测数据集(VisA、MPDD)上验证泛化性
4. **固定类别训练**：每个类别需要单独训练，无法零样本泛化到新类别
5. **异常类型受限**：嵌套提示词中的异常描述较简单(单个弱先验词)，对复杂多类异常的建模能力有限

## 相关工作与启发

- **AnomalyDiffusion**：分别学习异常外观和掩码位置，本文通过双流同步生成显著改进对齐问题
- **LayerDiffusion**：将图像分解为多层并行生成的灵感来源
- **DreamBooth/Textual Inversion**：小样本定制化生成的技术基础
- **启发**：在生成任务中，"同时生成多个相关输出" 比 "分别生成再对齐" 更有效且更自然

## 评分

⭐⭐⭐⭐

问题定义精准(小样本异常生成+掩码对齐)，双流同步生成的设计巧妙且有效。下游任务指标达到 SOTA，实用价值高。BCM 对小样本过拟合的解决方案也值得参考。

<!-- RELATED:START -->

## 相关论文

- [HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](../../ICCV2025/image_generation/hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)
- [Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_for_unified_image_generation_and_understanding.md)
- [V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](../../CVPR2026/image_generation/uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)

<!-- RELATED:END -->
