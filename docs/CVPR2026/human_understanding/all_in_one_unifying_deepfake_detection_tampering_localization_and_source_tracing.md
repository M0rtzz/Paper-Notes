---
title: >-
  [论文解读] All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a Robust Landmark-Identity Watermark
description: >-
  [CVPR2026][人体理解][deepfake detection] 提出 LIDMark，首个将 deepfake 检测、篡改区域定位和源追踪统一到单一主动取证框架中的方法——通过嵌入 152 维 Landmark-Identity 水印（136D 面部关键点 + 16D 源 ID），利用内在/外在一致性实现三合一取证，PSNR/SSIM 和检测精度均超越现有方法。
tags:
  - CVPR2026
  - 人体理解
  - deepfake detection
  - watermarking
  - tampering localization
  - source tracing
  - proactive forensics
  - facial landmark
---

# All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a Robust Landmark-Identity Watermark

**会议**: CVPR2026  
**arXiv**: [2602.23523](https://arxiv.org/abs/2602.23523)  
**代码**: [GitHub](https://github.com/vpsg-research/LIDMark)  
**领域**: human_understanding  
**关键词**: deepfake detection, watermarking, tampering localization, source tracing, proactive forensics, facial landmark

## 一句话总结

提出 LIDMark，首个将 deepfake 检测、篡改区域定位和源追踪统一到单一主动取证框架中的方法——通过嵌入 152 维 Landmark-Identity 水印（136D 面部关键点 + 16D 源 ID），利用内在/外在一致性实现三合一取证，PSNR/SSIM 和检测精度均超越现有方法。

## 研究背景与动机

深度伪造（Deepfake）技术快速发展，带来了严重的安全威胁。现有取证方法分为两大类：

**被动取证**：直接从图像中提取伪造痕迹进行检测。问题在于：(a) 仅能做二分类（真/假），无法定位篡改区域或追溯来源；(b) 泛化性差，对未见过的伪造方法性能急剧下降。

**主动取证（水印方法）**：预先在图像中嵌入水印，通过水印的破坏/保留情况进行取证。现有水印方法（如 FaceSigns、MBRS、PIMoG）的局限：
   - 大多仅支持检测，不支持篡改定位
   - 水印容量有限（通常 30 bits），难以同时编码多种信息
   - 检测和定位需要不同的水印设计，难以统一

**核心洞察**：面部关键点（landmarks）天然具备两种互补性质——(1) 对篡改敏感（swap 后关键点分布会变化），适合定位；(2) 身份 ID 需要对伪造鲁棒，适合源追踪。将两者编码为统一水印，可同时解决三大取证任务。

## 核心问题

如何设计一个统一的主动取证框架，在单一水印中同时实现：
- **Deepfake 检测**：判断图像是否经过篡改
- **篡改定位**：精确定位被篡改的面部区域
- **源追踪**：追溯图像的原始来源身份

## 方法详解

### LIDMark 水印设计（152 维）

水印由两部分组成：

- **Landmark 分量（136D）**：68 个面部关键点的 $(x, y)$ 坐标，归一化到 $[0,1]$。这部分对篡改敏感——换脸后，恢复出的水印关键点与实际检测到的关键点不一致
- **Identity 分量（16D）**：源身份 ID 的二进制编码。这部分需要对各种伪造操作鲁棒，用于追溯图像来源

### Encoder：双流融合网络

编码器采用双流架构将水印嵌入图像：

- **图像流**：基于 SEResNet 的特征提取，捕获图像内容特征
- **水印流**：基于 DiffusionNet 的水印处理，将 152 维向量映射为与图像同尺寸的特征图
- **融合机制**：两流特征通过拼接（concatenation）融合，并添加 skip connection 保持图像质量

嵌入过程可表示为：$I_w = E(I, m)$，其中 $I$ 为原始图像，$m = [m_{\text{land}}, m_{\text{id}}]$ 为 152 维水印。

### FHD：因子化头部解码器

解码器采用统一的 Factorized-Head Decoder（FHD）设计：

- **共享 Backbone**：提取水印图像的公共特征表示
- **回归头（Regression Head）**：输出 136 维关键点坐标 $\hat{m}_{\text{land}}$，使用 L1 损失
- **分类头（Classification Head）**：输出 16 维源 ID $\hat{m}_{\text{id}}$，使用 BCE 损失

FHD 比双 decoder 方案更统一且参数更少，共享 backbone 使两个任务互相增益。

### 内在-外在一致性检测

这是实现三合一取证的关键机制：

1. **Intrinsic Landmarks**：FHD 从水印图像中恢复的关键点 $\hat{m}_{\text{land}}$（编码时嵌入的原始关键点）
2. **Extrinsic Landmarks**：使用外部人脸关键点检测器（如 dlib）重新检测当前图像的关键点 $m_{\text{ext}}$

**检测逻辑**：

- **全局检测**：计算 Average Euclidean Distance（AED）—— $\text{AED} = \frac{1}{68}\sum_{i=1}^{68} \| \hat{p}_i - p_i^{\text{ext}} \|_2$。若 AED 超过阈值 $\tau$，判定为伪造
- **区域级定位**：对每个关键点的偏移进行空间分析，偏移大的区域即为篡改区域
- **源追踪**：分类头解码出的 16D ID 标识原始来源

### 两阶段训练策略

- **Stage 1 预训练**：使用常规图像失真（JPEG 压缩、高斯噪声、裁剪等）训练编码器-解码器，建立基础水印嵌入/提取能力
- **Stage 2 微调**：使用深度伪造方法（SimSwap、UniFace、CSCS、StarGAN-v2）生成的伪造图像进行微调，增强对 deepfake 场景的鲁棒性

### 损失函数

总训练损失为：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{image}} + \lambda_2 \mathcal{L}_{\text{land}} + \lambda_3 \mathcal{L}_{\text{id}}$$

其中 $\mathcal{L}_{\text{image}}$ 为图像质量损失（L2 + LPIPS），$\mathcal{L}_{\text{land}}$ 为关键点回归 L1 损失，$\mathcal{L}_{\text{id}}$ 为 ID 分类 BCE 损失。

## 实验关键数据

### 图像质量

| 分辨率 | PSNR ↑ | SSIM ↑ | 水印容量 |
|--------|--------|--------|---------|
| 128×128 | **40.22** | **0.98** | 152 bits |
| 256×256 | **44.31** | **0.99** | 152 bits |
| 基线最佳（MBRS） | 38.76 | 0.97 | 30 bits |

在更高容量（152 bits vs 30 bits）的情况下，LIDMark 的图像质量仍超越所有基线。

### Deepfake 检测性能

| 数据集 | 方法 | AUC ↑ |
|--------|------|-------|
| CelebA-HQ | LIDMark | **最优** |
| LFW | LIDMark | **最优** |

在 CelebA-HQ 和 LFW 两个数据集上，LIDMark 的检测 AUC 均优于现有主动取证方法。

### 篡改定位精度

通过区域级关键点偏移分析，LIDMark 能够生成与换脸区域高度吻合的篡改热力图，IoU 显著优于基于全局水印差异的方法。

### 源追踪准确率

16D ID 在各种伪造攻击后的恢复准确率超过 95%，证明 ID 分量的鲁棒性设计有效。

### 消融实验

| 组件 | PSNR | 检测 AUC | 说明 |
|------|------|----------|------|
| 完整 LIDMark | 40.22 | 最优 | — |
| 去掉 skip connection | 38.5 | 下降 | 图像质量显著下降 |
| 双 decoder 替代 FHD | 39.8 | 相当 | 参数更多，质量略降 |
| 仅 Stage 1 训练 | 40.1 | 下降 | 对 deepfake 不鲁棒 |

## 亮点与洞察

1. **三合一统一框架**：首次将检测、定位、追踪统一到单个水印方案中，不需要为不同任务设计不同的水印
2. **152 维水印设计极其巧妙**：利用面部关键点的天然双重属性（篡改敏感 + 语义丰富），将水印从"信息编码"提升为"语义编码"
3. **内在-外在一致性**是核心创新——将水印恢复问题转化为一致性检验问题，自然实现了从检测到定位的扩展
4. **FHD 因子化解码设计**：共享 backbone + 任务特定头部，比分离式设计更高效且互相增益
5. **高信息容量下的高图像质量**：152 bits 远超现有方法（30 bits），而 PSNR/SSIM 反而更优

## 局限性

1. **主动取证的根本限制**：需要在图像发布前嵌入水印，对已存在的无水印图像无效
2. **分辨率限制**：实验仅在 128×128 和 256×256 上验证，对高分辨率（1024+）的可扩展性不明确
3. **伪造方法覆盖**：微调阶段仅使用 4 种 deepfake 方法，对新型伪造技术（如扩散模型生成）的泛化性需进一步验证
4. **关键点检测器依赖**：外在一致性依赖 dlib 等关键点检测器的精度，检测器失败时框架受影响
5. **对抗性攻击**：未讨论针对水印的对抗性去除攻击的鲁棒性

## 相关工作与启发

- 与 **FaceSigns**（Neekhara et al., 2022）相比：FaceSigns 仅做检测（水印有/无），LIDMark 扩展到定位和追踪
- 与 **MBRS**（Jia et al., 2021）相比：MBRS 水印容量仅 30 bits 且无篡改定位能力，LIDMark 152 bits + 定位
- 与被动方法（**Xception**、**Face X-ray**）相比：被动方法无需预处理但泛化差，LIDMark 牺牲部署便利性换取可靠三合一能力
- **启发**：水印不必是任意比特串——利用领域语义（关键点）编码水印，可以实现远超传统水印的功能。这一思路可推广到其他领域（如医学图像的解剖关键点、遥感图像的地标编码）

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 首个三合一主动取证框架，水印设计巧妙
- **实验充分性**: ⭐⭐⭐⭐ — 多数据集多基线对比充分，但缺乏高分辨率和更多伪造方法验证
- **实用性**: ⭐⭐⭐⭐ — 主动取证场景有明确应用价值，但需预先嵌入水印
- **写作质量**: ⭐⭐⭐⭐ — 框架描述清晰，动机论证合理
- **综合评分**: ⭐⭐⭐⭐ (4.0/5)

<!-- RELATED:START -->

## 相关论文

- [RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)
- [Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)
- [Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties](../../CVPR2025/human_understanding/multi-sensor_object_anomaly_detection_unifying_appearance_geometry_and_internal_.md)
- [Face Time Traveller: Travel Through Ages Without Losing Identity](face_time_traveller_travel_through_ages_without_losing_identity.md)
- [AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)

<!-- RELATED:END -->
