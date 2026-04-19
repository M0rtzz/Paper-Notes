---
title: >-
  [论文解读] GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields
description: >-
  [ECCV 2024][3D视觉][NeRF版权保护] 提出GeometrySticker，将二进制版权信息"贴"在NeRF的**几何组件**（而非颜色组件）上，使得即使NeRF被重着色（recolorization），原始创建者仍能从渲染图像中提取水印来主张所有权。
tags:
  - ECCV 2024
  - 3D视觉
  - NeRF版权保护
  - 数字水印
  - 几何嵌入
  - 重着色鲁棒
  - 隐式表示安全
---

# GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields

**会议**: ECCV 2024  
**arXiv**: [2407.13390](https://arxiv.org/abs/2407.13390)  
**代码**: https://kevinhuangxf.github.io/GeometrySticker (有，项目页)  
**领域**: 3D视觉  
**关键词**: NeRF版权保护, 数字水印, 几何嵌入, 重着色鲁棒, 隐式表示安全

## 一句话总结

提出GeometrySticker，将二进制版权信息"贴"在NeRF的**几何组件**（而非颜色组件）上，使得即使NeRF被重着色（recolorization），原始创建者仍能从渲染图像中提取水印来主张所有权。

## 研究背景与动机

**领域现状**: NeRF重着色技术（CLIP-NeRF、PaletteNeRF、RecolorNeRF等）已非常成熟，可以轻松修改NeRF模型的颜色属性。同时NeRF作为可共享的数字资产日益普及（如NeRFStudio平台）。

**现有痛点**: 恶意用户可以重着色他人的NeRF模型并虚假声称其所有权。现有水印方法如CopyRNeRF将信息嵌入颜色表示中，一旦颜色被修改，水印即失效（bit accuracy降至~50%随机猜测水平）。StegaNeRF依赖完整的几何+颜色表示做信息隐藏，重着色后同样失效。

**核心矛盾**: 重着色修改的是颜色表示，而现有水印方法恰恰依赖颜色表示嵌入信息。需要一种"颜色无关"的水印方案。

**本文目标**: 如何在NeRF模型中嵌入水印，使其在经过各种重着色操作后仍可被提取？

**切入角度**: NeRF由颜色MLP ($\Theta_c$) 和几何MLP ($\Theta_\sigma$) 分别建模，重着色只修改颜色部分，因此将水印嵌入几何部分即可天然抵抗重着色。

**核心 idea**: 在NeRF物体表面的高密度3D点（几何组件）上附加（而非修改）可学习的水印信息，像贴纸一样。

## 方法详解

### 整体框架

GeometrySticker包含三个核心组件：**(1)** 基于可学习Laplace CDF的载体媒介选择；**(2)** 基于MLP的消息贴纸(Message Sticker)；**(3)** 基于VGG16的消息提取器。整体流程：选择物体表面的3D点 → 用MLP将二进制消息转为几何兼容格式 → 通过加法附加到选中点的密度值上 → 从渲染图像中提取水印验证。

### 关键设计

1. **载体媒介选择（Cover Media Selection）**: 不是将水印嵌入全部几何组件（会导致明显伪影），而是只选择**物体表面附近的高密度3D点**。使用Laplace CDF计算每个采样点的重要性值：

$$\psi = \frac{1}{2} + \frac{1}{2} \cdot \text{sign}(\sigma - \mu) \cdot \left(1 - \exp\left(-\frac{|\sigma - \mu|}{\beta}\right)\right)$$

其中 $\mu$ 为几何场的均值，$\beta$ 为**可学习参数**（而非固定阈值），$\psi \in [0,1]$ 表示密度值的累积概率。$\psi$ 接近1的点被选为载体——它们是物体表面的高密度点，仅占少量3D点，确保了嵌入的**隐蔽性(Subtlety)**。使用稀疏损失约束 $\psi$ 趋向0或1的二值分布：

$$\mathcal{L}_{sparse} = \frac{1}{|N_p|} \sum_{\psi_i} [\log(\psi_i) + \log(1 - \psi_i)]$$

关键创新在于 $\beta$ 可学习——自适应调整阈值，避免固定阈值带来的"覆盖太多导致失真"或"覆盖太少导致不可提取"的问题。

2. **消息贴纸（Message Sticker）**: 一个轻量MLP $\Theta_\mathbf{m}$ 将位置编码和二进制消息映射为一维消息嵌入：

$$m = \Theta_\mathbf{m}(\gamma_x(\mathbf{x}), \mathbf{M})$$

然后通过简单**加法**将消息附加到密度值上：

$$\tilde{\sigma} = \sigma + \psi \cdot m$$

这个设计保证了**可扩展性(Scalability)**：不修改NeRF原有结构，不依赖特定架构，可应用于vanilla NeRF、InstantNGP、TensoRF等各种变体。水印化后的密度值参与标准的体积渲染：

$$\tilde{C} = \sum_{i=1}^{N} \exp\left(-\sum_{j=1}^{i-1} \tilde{\sigma}_j \delta_j\right)(1 - \exp(-\tilde{\sigma}_i \delta_i)) \mathbf{c}_i$$

训练时在每个视角重复上述操作，确保水印从任意角度可访问——即**普遍性(Ubiquity)**。

3. **消息提取与验证**: 使用VGG16作为backbone的CNN消息提取器 $D_\chi$，从渲染图像 $\mathbf{I}_w$ 中提取水印 $\hat{\mathbf{M}} = D_\chi(\mathbf{I}_w)$。另外训练一个CNN分类器 $C_\phi$ 判断图像是否含水印。

### 损失函数 / 训练策略

总损失由四部分组成：

$$\mathcal{L}_{total} = \mathcal{L}_{cont} + \mathcal{L}_{msg} + \mathcal{L}_{cls} + \mathcal{L}_{sparse}$$

- **内容损失**: $\mathcal{L}_{cont} = \|\mathbf{I}_w - \mathbf{I}_o\|_2^2$，保证渲染质量不受水印影响
- **消息损失**: $\mathcal{L}_{msg} = \text{BCE}(D_\chi(\mathbf{I}_w), \mathbf{M})$，监督水印提取准确性
- **分类损失**: $\mathcal{L}_{cls} = \text{BCE}(C_\phi(\mathbf{I}_w), C_\phi(\mathbf{I}_u))$，训练水印检测分类器
- **稀疏损失**: $\mathcal{L}_{sparse}$，约束重要性值趋向二值分布

训练时对渲染图像施加高斯噪声、随机旋转、随机裁剪、高斯模糊等扰动增强鲁棒性。训练补丁大小 $400 \times 400$，仅需5000步，45分钟内在单V100上完成。嵌入消息长度48 bits。

### 支持的重着色方案

- **CLIP-based重着色**: 使用文本提示（如"red"）通过CLIP特征匹配损失修改颜色
- **Palette-based重着色**: 分解颜色为调色板基底的线性组合，修改调色板颜色
- **图像级色彩抖动**: 直接修改渲染图像的色调

## 实验关键数据

### 主实验

**重着色后的水印提取准确率（Blender/LLFF数据集，核心表格）**：

| 方法 | PSNR/SSIM↑ | LPIPS↓ | Color-jitter准确率 | CLIP准确率 | Palette准确率 |
|------|------------|--------|-------------------|-----------|--------------|
| HiDDeN+NeRF | 30.80/0.9999 | 0.0167 | 50.13% | 51.08% | 50.91% |
| CopyRNeRF | 29.99/0.9999 | 0.0171 | 51.32% | 49.96% | N.A. |
| StegaNeRF | 31.48/0.9999 | 0.0149 | 54.18% | 52.48% | N.A. |
| **GeometrySticker** | **32.13/0.9999** | **0.0136** | **99.33%** | **99.50%** | **99.40%** |

**标准场景（无重着色）+ 各种图像扰动（Blender）**：

| 方法 | 无扰动 | 噪声 | 旋转 | 裁剪 | 模糊 |
|------|--------|------|------|------|------|
| HiDDeN+NeRF | 50.19% | 49.84% | 50.12% | 50.09% | 50.16% |
| CopyRNeRF | 66.80% | 65.92% | 64.52% | 63.44% | 66.22% |
| StegaNeRF | 100% | 90.21% | 57.17% | 60.30% | 92.88% |
| **GeometrySticker** | **100%** | **99.25%** | **98.87%** | **98.75%** | **99.88%** |

### 消融实验

**载体媒介选择策略消融**：

| 策略 | 效果 | 说明 |
|------|------|------|
| 全部几何点嵌入 | 明显失真 | 低密度空间点被修改导致可见伪影 |
| 固定阈值Laplace CDF | 仍有可察觉失真 | 固定阈值不够灵活 |
| **可学习Laplace CDF** | **几乎不可察觉** | 自适应找到最优阈值 |

**跨架构可扩展性**：

| NeRF架构 | PSNR/SSIM↑ | Bit准确率 |
|----------|------------|----------|
| NeRF | 27.44/0.8759 | 100% |
| InstantNGP | 28.59/0.8868 | 100% |
| TensoRF | 29.18/0.8907 | 100% |

### 关键发现

- 其他方法在重着色后水印准确率降至~50%（相当于随机猜测），GeometrySticker保持>96%
- GeometrySticker不影响重着色效果（水印vs无水印的重着色结果PSNR>32dB）
- 可学习阈值 $\beta$ 是关键——固定阈值方案有明显质量损失
- 对抗攻击分析：如果消息提取器泄露，PGD攻击可以降低准确率；但模型净化攻击若要保持渲染质量则难以有效去除水印

## 亮点与洞察

- **问题定义新颖**: 首次关注NeRF重着色场景下的版权保护问题，定义了一个实际且重要的安全威胁模型
- **"贴纸"式设计精巧**: 不修改原NeRF结构，仅通过加法附加信息，实现了真正的即插即用
- **三大设计原则清晰**: Scalability（跨架构适配）、Subtlety（隐蔽性）、Ubiquity（全视角可访问），每个原则都有对应的技术方案
- **可学习CDF**: 将载体选择从固定阈值升级为可学习参数，既巧妙又实用

## 局限与展望

- 消息提取器泄露时对抗攻击仍可降低准确率，需保持提取器私有
- 仅在NeRF系列上验证，未扩展到3D Gaussian Splatting（作者明确提及为未来工作）
- 对几何编辑（如cage-based变形、运动迁移）的鲁棒性未探索
- 水印嵌入需要额外训练步骤（虽然仅45分钟），未实现零样本嵌入
- 仅验证48-bit消息长度，更大容量的可行性未探索

## 相关工作与启发

- **CopyRNeRF**: 将水印嵌入颜色表示但不耐重着色，GeometrySticker的直接改进对象
- **StegaNeRF**: NeRF隐写术方法，同样不耐重着色
- **HiDDeN**: 经典深度学习图像水印方法，在NeRF场景下完全失效
- **PaletteNeRF/RecolorNeRF**: 代表性重着色方法，验证了GeometrySticker的兼容性
- 启发：对于隐式表示的安全防护，利用"不被攻击操作影响的组件"嵌入信息是通用策略；未来可推广到其他3D表示的版权保护

## 评分

- 新颖性: ⭐⭐⭐⭐ 问题定义新颖，"几何嵌入抵抗颜色修改"的思路直觉清晰且有效
- 实验充分度: ⭐⭐⭐⭐⭐ 多架构×多重着色方案×多扰动类型×对抗攻击/模型净化分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，三大设计原则贯穿全文
- 价值: ⭐⭐⭐⭐ 解决了实际安全问题，但受限于NeRF生态的实际应用范围

<!-- RELATED:START -->

## 相关论文

- [G2fR: Frequency Regularization in Grid-Based Feature Encoding Neural Radiance Fields](g2fr_frequency_regularization_in_grid-based_feature_encoding_neural_radiance_fie.md)
- [BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](benerf_neural_radiance_fields_from_a_single_blurry_image_and_event_stream.md)
- [Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)
- [LaRa: Efficient Large-Baseline Radiance Fields](lara_efficient_large-baseline_radiance_fields.md)
- [Dynamic Neural Radiance Field from Defocused Monocular Video](dynamic_neural_radiance_field_from_defocused_monocular_video.md)

<!-- RELATED:END -->
