---
title: >-
  [论文解读] CompleteMe: Reference-based Human Image Completion
description: >-
  [ICCV 2025][图像生成][Image Completion] 提出CompleteMe框架，通过双U-Net架构和Region-focused Attention（RFA）Block，利用参考图像中的细粒度人物细节（衣物纹理、纹身等），实现高保真的参考引导人体图像补全。
tags:
  - ICCV 2025
  - 图像生成
  - Image Completion
  - 图像修复
  - Dual U-Net
  - 注意力机制
  - Human Body
---

# CompleteMe: Reference-based Human Image Completion

**会议**: ICCV 2025  
**arXiv**: [2504.20042](https://arxiv.org/abs/2504.20042)  
**代码**: 无  
**领域**: Image Generation / Human Image Completion  
**关键词**: Image Completion, Reference-based Inpainting, Dual U-Net, Attention Mechanism, Human Body

## 一句话总结

提出CompleteMe框架，通过双U-Net架构和Region-focused Attention（RFA）Block，利用参考图像中的细粒度人物细节（衣物纹理、纹身等），实现高保真的参考引导人体图像补全。

## 研究背景与动机

### 领域现状

**领域现状**：人体图像补全是计算机视觉的重要任务，应用于照片编辑、虚拟试穿、动画等领域。现有方法存在两类问题：

**无参考方法的局限**：

### 核心矛盾

**核心矛盾**：LOHC、BrushNet等方法能生成合理的人体形状，但无法恢复个人独有细节（如特定衣物花纹、纹身图案、独特配饰）

### 现有痛点

**现有痛点**：没有参考图像时，这些独特信息无法凭空生成

**参考引导方法的不足**：

### 解决思路

**解决思路**：Paint-by-Example、AnyDoor等主要关注物体级别的插入/补全

### 补充说明

**补充说明**：MimicBrush等方法在源图像和参考图像姿态差异较大时，难以建立准确的对应关系

### 补充说明

**补充说明**：现有方法无法有效捕捉并整合参考图像中的细粒度细节

核心挑战：如何在姿态差异显著的情况下，精确地将参考图像中的局部细节映射到待补全区域。

## 方法详解

### 整体框架

CompleteMe采用**双U-Net架构**：
1. **Reference U-Net ($U_{ref}$)**：从多张参考图像中提取详细视觉特征
2. **Complete U-Net ($U_{comp}$)**：处理遮挡输入，利用参考特征完成补全
3. **CLIP图像编码器**：提供全局语义特征

参考图像按人体部位分割（上衣、下装、头发、面部、鞋等），分别编码后在Complete U-Net中通过RFA Block融合。

### 关键设计

**1. Reference U-Net**

- 从Stable Diffusion 1.5预训练权重初始化
- 在timestep=0处直接编码参考图像（无扩散噪声）
- 为不同人体部位（上身衣物、下身衣物、头发/配饰、面部、鞋）分别提取多尺度空间特征
- 顺序处理各参考图像，确保灵活性

**2. Region-focused Attention (RFA) Block**

这是CompleteMe的核心创新：

- **显式掩码过滤**：用参考掩码遮蔽参考特征中无关区域，生成masked reference features
- **特征拼接**：将masked reference features与输入特征拼接
- **区域聚焦注意力**：

$$\text{RFA}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

其中 $Q = f_{input}$, $K, V = f_{concat}$

- **解耦交叉注意力**：借鉴IP-Adapter，分别对局部参考特征和全局CLIP特征执行交叉注意力，然后求和

**3. 掩码策略**

训练时采用混合掩码策略：
- 50%概率使用随机网格掩码（1-30次）
- 50%概率使用人体形状掩码

### 损失函数/训练策略

- **损失函数**：MSE Loss
- **优化器**：Adam, lr=2×10⁻⁵
- **训练配置**：8×A100, batch size 64, 30K iterations
- **随机丢弃**：所有参考特征以0.2概率随机丢弃；每个参考条件独立以0.2概率丢弃
- **推理**：DDIM 50步, guidance scale 7.5
- **训练数据**：基于DeepFashion-MultiModal构建，4万对训练数据

## 实验关键数据

### 主实验 (表格)

| 方法 | CLIP-I↑ | DINO↑ | DreamSim↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|------|---------|-------|-----------|--------|-------|-------|
| BrushNet | 95.90 | 95.08 | 0.0576 | 0.0600 | 28.58 | 0.9224 |
| LeftRefill | 96.33 | 95.12 | 0.0574 | 0.0598 | 28.87 | 0.9283 |
| MimicBrush | 96.98 | 94.37 | 0.0651 | 0.0694 | 28.36 | 0.9174 |
| **CompleteMe** | **97.18** | **96.29** | **0.0419** | **0.0588** | 28.70 | 0.9239 |

CompleteMe在身份一致性指标（CLIP-I, DINO, DreamSim）上全面领先，DreamSim从0.0574降到0.0419（降低27%）。

### 消融实验 (表格)

| 消融项 | 结果 |
|--------|------|
| 无参考方法 vs 有参考 | 无参考无法恢复个人特征 |
| 无RFA | 难以建立精确对应 |
| 多参考 vs 单参考 | 多部位参考提供更全面信息 |

### 关键发现

1. **显式区域聚焦是关键**：直接在全图上做交叉注意力效果差，显式masking+拼接让模型精确匹配对应区域
2. **多部位分离编码**：将参考图像按身体部位拆分处理，比整图编码更有效
3. **模型灵活性**：推理时可仅用一张参考图，也可加入文本提示
4. **用户研究验证**：大规模用户研究确认CompleteMe的主观优势

## 亮点与洞察

1. **任务定义清晰**：明确区分了"无参考补全"和"有参考补全"两个子问题
2. **RFA设计精巧**：通过显式掩码将注意力引向相关区域，比隐式学习更高效可靠
3. **解耦全局+局部**：CLIP全局语义 + Reference U-Net局部细节的双轨设计
4. **实用的benchmark**：构建了417组包含显著姿态差异的测试集

## 局限与展望

1. **训练数据规模有限**：仅4万对训练数据
2. **基于SD1.5**：基座模型较老，升级到SDXL/SD3可能获得质量提升
3. **姿态极端差异**：当源和参考姿态差异过大时，对应关系仍可能出错
4. **部位解析依赖**：需要预先分割人体部位作为参考输入
5. **仅支持静态图像**：未扩展到视频序列的时序一致补全

## 相关工作与启发

- **MimicBrush**：双扩散U-Net + 自监督视频训练，但在大姿态差异下效果不佳
- **AnyDoor**：零样本物体传送框架，偏向物体级操作
- **IP-Adapter**：解耦交叉注意力被CompleteMe借鉴用于全局/局部特征融合
- **BrushNet**：双分支像素级掩码特征嵌入，但无参考能力
- 启发：**参考引导的图像编辑核心挑战是对应关系建立**，显式区域引导比纯隐式学习更可靠

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 3.5 |
| 技术深度 | 3.5 |
| 实验充分性 | 4 |
| 写作质量 | 3.5 |
| 实用性 | 4 |
| 总评 | 3.5 |

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Free-viewpoint Human Animation with Pose-correlated Reference Selection](../../CVPR2025/image_generation/free-viewpoint_human_animation_with_pose-correlated_reference_selection.md)
- [\[ICCV 2025\] Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences](cycle_consistency_as_reward_learning_image-text_alignment_without_human_preferen.md)
- [\[ICCV 2025\] HPSv3: Towards Wide-Spectrum Human Preference Score](hpsv3_towards_wide-spectrum_human_preference_score.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[CVPR 2026\] HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images](../../CVPR2026/image_generation/hifi-inpaint_towards_high-fidelity_reference-based_inpainting_for_generating_det.md)

<!-- RELATED:END -->
