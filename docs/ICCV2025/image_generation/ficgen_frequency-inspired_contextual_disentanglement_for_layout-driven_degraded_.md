---
title: >-
  [论文解读] FICGen: Frequency-Inspired Contextual Disentanglement for Layout-driven Degraded Image Generation
description: >-
  [ICCV 2025][图像生成][Layout-to-Image] 提出 FICGen，首次解决退化场景（低光照/水下/遥感/恶劣天气等）Layout-to-Image 生成中的"上下文幻觉困境"，通过可学习双查询机制提取退化场景的高低频原型，经视觉-频率增强注意力注入 latent 扩散空间，并使用实例一致性图 + 空间-频率自适应聚合实现前景-背景解耦，在 5 个退化数据集上全面超越现有 L2I 方法。
tags:
  - ICCV 2025
  - 图像生成
  - Layout-to-Image
  - Frequency Disentanglement
  - Low-light
  - Remote Sensing
  - Underwater
---

# FICGen: Frequency-Inspired Contextual Disentanglement for Layout-driven Degraded Image Generation

**会议**: ICCV 2025  
**arXiv**: [2509.01107](https://arxiv.org/abs/2509.01107)  
**代码**: 无（未提及）  
**领域**: 图像生成 / Layout-to-Image / 退化场景合成  
**关键词**: Layout-to-Image, Degraded Image Generation, Frequency Disentanglement, Low-light, Remote Sensing, Underwater

## 一句话总结
提出 FICGen，首次解决退化场景（低光照/水下/遥感/恶劣天气等）Layout-to-Image 生成中的"上下文幻觉困境"，通过可学习双查询机制提取退化场景的高低频原型，经视觉-频率增强注意力注入 latent 扩散空间，并使用实例一致性图 + 空间-频率自适应聚合实现前景-背景解耦，在 5 个退化数据集上全面超越现有 L2I 方法。

## 研究背景与动机

### 问题背景
退化场景（低光照、水下、遥感、恶劣天气等）的视觉感知任务严重缺乏标注数据。例如 ExDARK 低光照数据集仅有 7,363 张图像，是 COCO 的 1/20。Layout-to-Image（L2I）生成是一种利用布局条件合成训练数据的有前景方案。

### 核心挑战：上下文幻觉困境（Contextual Illusion Dilemma）
现有 L2I 方法在自然场景下表现良好，但应用到退化场景时面临严重问题：
- 遥感目标（如车辆）尺寸小且与周围结构（如桥梁）视觉相似
- 水下物种（如鱼）经常与附近生物（如珊瑚）混为一体
- 这导致生成中出现**物体数量、位置和交互的幻觉**

### 频率视角的分析
自然图像中高频（HF）和低频（LF）分量相对平衡，前景-背景区分明显。退化图像中，**前景实例的高频细节被衰减**，**低频背景主导了整体频率分布**。这解释了为何退化场景中实例容易被"淹没"。

### 动机
从频率视角实现上下文解耦：提取退化场景的高频（实例边界/纹理）和低频（背景颜色/氛围）知识，将其注入扩散生成过程，同时通过实例级掩码实现前景-背景的 latent 空间解耦。

## 方法详解

### 整体框架
FICGen 包含三个核心模块：
1. **频率感知重采样器**（Frequency Perceiver Resamplers）——通过双查询机制提取 HF/LF 频率原型
2. **视觉-频率增强注意力**（Visual-Frequency Enhanced Attention）——将频率知识注入 latent 扩散空间
3. **自适应空间-频率聚合**（Adaptive Spatial-Frequency Aggregation）——混合空间和频率信息重建退化表示

### 频率原型提取
**步骤一：构建频率原型**
从训练集中按类别采样退化实例，提取中间特征图 $\mathbf{X} \in \mathbb{R}^{H \times W}$，经 DFT 变换到频域：

$$\textbf{X}_{\mathcal{F}}(u,v) = \frac{1}{H \times W}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1}\textbf{X}(h,w)e^{-j2\pi(uh+vw)}$$

用二值掩码 $\mathbf{M}_{\mathcal{F}}$ 分离高频/低频区域，经可学习通道权重增强后逆 DFT 回空间域：

$$\textbf{X}^{\uparrow} = \mathcal{F}^{-1}(\textbf{X}_{\mathcal{F}}\textbf{M}_{\mathcal{F}})\cdot\mathbf{W}_{\mathcal{F}}$$

对 HF/LF 特征图做均值池化得到频率原型：$\textbf{p}^{\uparrow} = \{p_i^{\uparrow}\}_{i=1}^N$（实例 HF）和 $p^{\downarrow}$（背景 LF）。

**步骤二：双查询频率重采样器**
受 Perceiver 架构启发，使用两个独立的可学习查询通过 Transformer 块与频率原型交互：

$$\textbf{q}_i^{\uparrow} = \text{HF-Resampler}(\mathcal{Q}^{\uparrow}, \phi_{k1}^r(p_i^{\uparrow}), \phi_{v1}^r(p_i^{\uparrow}))$$
$$\textbf{q}^{\downarrow} = \text{LF-Resampler}(\mathcal{Q}^{\downarrow}, \phi_{k1}^g(p^{\downarrow}), \phi_{v1}^g(p^{\downarrow}))$$

双查询机制同时感知实例边界纹理（HF）和环境氛围颜色（LF）。

### 上下文频率知识迁移

**视觉-频率增强注意力**：将频率感知 token 与布局条件融合后注入扩散 U-Net：
- 实例表示：$\textbf{R}_i = [\textbf{q}_i^{\uparrow}; \textbf{E}_{clip}(l_i); \textbf{E}_{box}(\text{Fourier}(b_i))]$（HF token + 语义 + 位置）
- 背景表示：$\textbf{G} = [\textbf{q}^{\downarrow}; \textbf{E}_{clip}(\mathcal{Y})]$（LF token + 全局描述）

**实例一致性图**解耦前景与背景：

$$\hat{\mathbf{M}}_i(x,y) = \begin{cases} 1, & \text{if } [x,y] \in b_i \\ 0, & \text{otherwise} \end{cases}$$
$$\hat{\mathbf{M}}^g = 1 - \sum_{i=1}^{N}\hat{\mathbf{M}}_i$$

通过掩码约束每个布局只影响对应局部区域，防止属性泄漏。

### 自适应空间-频率聚合
不同于简单求和或纯空间域融合，FICGen 在空间和频率两个域同时聚合退化实例和背景：

$$\textbf{F}^s = \textbf{SAM}([\sum_{i=1}^N \textbf{f}_i^r, \textbf{f}^g]), \quad \textbf{F}^f = \textbf{FAM}([\sum_{i=1}^N \textbf{f}_i^r, \textbf{f}^g])$$

其中 SAM 用标准自注意力捕获空间关系依赖，FAM 用频率注意力强调跨实例的细粒度属性（边界锐度、纹理）。两路输出通过可学习深度卷积 $\zeta$ 融合后做 softmax 加权聚合得到最终退化表示 $\delta^{final}$。

### 训练目标
冻结预训练 LDM 参数，仅训练 FICGen 模块：

$$\min_{\theta'} \mathcal{L}_{FICGen} = \mathbb{E}_{z_0, \epsilon, t, \mathcal{Y}}[\|\epsilon - \mathcal{G}_{\theta,\theta'}(z_t, t, \mathcal{Y}, \mathcal{B}, \mathcal{Q})\|_2^2]$$

## 实验

### 实验设置
- **基础模型**：SDv1.5，只在 8×8 和 16×16 分辨率解码器层部署 FICGen
- **训练**：AdamW，lr=1e-4，300 epochs，8×A100，batch=320
- **评估**：5 个退化数据集——ExDARK（低光照）、RUOD（水下）、DIOR-H（遥感）、DAWN（恶劣天气）、blurred VOC2012（模糊）
- **指标**：FID（保真度）、COCO-style AP（对齐度）、下游检测器 mAP（可训练性）

### 主实验：保真度和对齐度

| 数据集 | 方法 | FID↓ | mAP↑ | AP_50↑ | AP_75↑ |
|--------|------|------|------|--------|--------|
| DIOR-H (遥感) | MIGC | 31.64 | 21.8 | 38.4 | 17.5 |
| DIOR-H | CC-Diff | 30.88 | 23.6 | 42.4 | 21.4 |
| DIOR-H | **FICGen** | 31.25 | **27.6** | **48.7** | **27.6** |
| RUOD (水下) | MIGC | 26.50 | 27.2 | 54.1 | 24.6 |
| RUOD | CC-Diff | 25.21 | 29.7 | 58.4 | 27.9 |
| RUOD | **FICGen** | **25.10** | **37.0** | **68.6** | **36.5** |
| ExDARK (低光照) | MIGC | 45.76 | 32.4 | 63.5 | 29.5 |
| ExDARK | CC-Diff | 44.26 | 35.1 | 65.6 | 34.1 |
| ExDARK | **FICGen** | **42.40** | **42.5** | **73.0** | **45.1** |

FICGen 在所有退化场景下的对齐度（mAP）大幅领先，特别是在 ExDARK 上甚至超越了真实测试集基线（42.5 vs 37.2 mAP），说明生成的退化实例精确遵循布局。

### DIOR-H 遥感对比（多方法）

| 方法 | FID↓ | YOLO mAP↑ | AP_50↑ | AP_75↑ |
|------|------|----------|--------|--------|
| LayoutDiffusion | 45.31 | 20.0 | 37.4 | 19.3 |
| GLIGEN | 41.31 | 25.8 | 44.4 | 27.8 |
| AeroGen | 38.57 | 29.8 | 54.2 | 31.6 |
| CC-Diff | 30.88 | 26.4 | 44.2 | 28.5 |
| **FICGen** | 31.25 | **31.2** | **49.9** | **34.6** |

FICGen 在 YOLO 评估下也取得最优 mAP。

### 下游可训练性（数据增强效果）
FICGen 合成数据用于增强下游检测器训练时：
- 一致性提升 ~2.0 mAP
- 部分类别提升显著：遥感中"airport"类 **+6.0 AP**（32.2→38.1）
- 在 ExDARK 上 Cascade R-CNN 的 mAP 从 37.2 提升至 42.5

### Deformable-DETR 验证

| 数据集 | 方法 | mAP↑ | AP_50↑ | AP_75↑ |
|--------|------|------|--------|--------|
| ExDARK | CC-Diff | 31.3 | 61.8 | 28.8 |
| ExDARK | **FICGen** | **38.5** | **68.5** | **39.5** |
| RUOD | CC-Diff | 29.7 | 57.8 | 28.0 |
| RUOD | **FICGen** | **37.1** | **67.1** | **36.7** |

用更强的 Deformable-DETR 评估时，FICGen 的优势更加明显。

## 亮点与洞察
- **首次系统性地解决退化场景的 L2I 生成**：提出"上下文幻觉困境"的概念并从频率角度给出解决方案
- **频率原型的巧妙设计**：将退化场景的频率特性（HF 实例衰减 + LF 背景主导）显式建模为可学习原型
- **双查询机制的优雅架构**：HF 查询关注实例细节、LF 查询关注环境氛围，分工明确
- **实例一致性图的简单有效**：用二值掩码实现 latent 空间解耦，避免属性泄漏和物体合并
- **五个退化场景的广泛验证**：从严重低光照到轻微模糊，证明方法的通用性
- **实用的数据增强能力**：生成数据可直接提升下游检测器性能

## 局限性
- 频率原型从训练集中采样退化实例构建，依赖训练集的退化模式覆盖度
- 基于 SDv1.5 实现，未探索新一代基础模型的适配
- $\gamma$ 参数（控制 HF 区域大小）需要手动设定
- 对极端退化（如几乎全黑的低光照）的生成鲁棒性未充分验证
- 仅在目标检测下游任务上验证可训练性，分割等其他任务未探索

## 相关工作
- **文本驱动图像合成**：DALL-E、LDM 等扩散/自回归模型
- **布局驱动图像合成**：GLIGEN、LayoutDiffusion、MIGC（多实例控制）、CC-Diff（上下文一致性）
- **退化场景生成**：AeroGen（遥感）是先驱，但受限于语义模糊和布局控制力不足

## 评分
- 新颖性：⭐⭐⭐⭐ — 频率视角切入退化场景 L2I 生成的思路新颖
- 技术深度：⭐⭐⭐⭐ — 双查询 + 频率原型 + 实例解耦的完整设计链
- 实验充分度：⭐⭐⭐⭐⭐ — 5 个数据集、多个检测器、下游训练验证
- 实用价值：⭐⭐⭐⭐ — 退化场景数据增强是实际工程中的刚需

<!-- RELATED:START -->

## 相关论文

- [Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](../../ECCV2024/image_generation/unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation](the_silent_assistant_noisequery_as_implicit_guidance_for_goal-driven_image_gener.md)
- [Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing](dct-shield_a_robust_frequency_domain_defense_against_malicious_image_editing.md)
- [SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)

<!-- RELATED:END -->
