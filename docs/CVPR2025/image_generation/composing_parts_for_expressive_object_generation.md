---
title: >-
  [论文解读] Composing Parts for Expressive Object Generation
description: >-
  [CVPR 2025][图像生成][部件级控制] 提出 PartComposer，一种无需训练的方法，通过并行"部件扩散"从注意力图中定位对象部件，再用区域扩散为每个部件独立生成用户指定的细粒度属性（颜色、风格、描述），实现部件级可控图像合成。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "部件级控制"
  - "无训练生成"
  - "注意力图分割"
  - "区域扩散"
  - "Rich-Text"
---

# Composing Parts for Expressive Object Generation

**会议**: CVPR 2025  
**arXiv**: [2406.10197](https://arxiv.org/abs/2406.10197)  
**代码**: 无（使用标准 SD 1.5/2.1/XL，无需额外代码）  
**领域**: 扩散模型 / 图像生成  
**关键词**: 部件级控制, 无训练生成, 注意力图分割, 区域扩散, Rich-Text

## 一句话总结
提出 PartComposer，一种无需训练的方法，通过并行"部件扩散"从注意力图中定位对象部件，再用区域扩散为每个部件独立生成用户指定的细粒度属性（颜色、风格、描述），实现部件级可控图像合成。

## 研究背景与动机

**领域现状**：Stable Diffusion 等模型可通过文本 prompt 生成高质量图像，ControlNet/GLIGEN 等提供空间控制（边缘图、框等）。但控制粒度停留在对象级——无法指定对象各部件的属性（如鸟的喙用红色、翅膀用蓝色）。

**现有痛点**：当在 prompt 中添加部件细节时，SD 要么完全忽略（如 "a bird with a red beak and blue wings" 生成普通鸟），要么生成与基础 prompt 完全不同的图像。Rich-Text 方法虽支持局部属性，但操作在对象级而非部件级，会修改整个对象区域。InstructPix2Pix 做全局编辑，无法精确修改单个部件。

**核心矛盾**：需要在不破坏对象整体结构的情况下单独控制各部件的属性，但预训练扩散模型的注意力图在对象级工作良好而在部件级信号很弱。

**本文目标** 如何从预训练扩散模型中零样本地定位对象部件，并基于用户指定的细粒度属性为每个部件生成内容。

**切入角度**：通过并行运行一个"部件扩散"过程（仅在对象区域内去噪，使用部件 token 作为条件），迫使 U-Net 将部件 token 的注意力集中到正确的空间区域，从而获得部件定位掩码。然后用区域扩散为每个部件独立生成并组合。

**核心 idea**：用并行部件扩散在对象掩码区域内去噪来激活部件级注意力定位，再用掩码区域扩散为每个部件生成指定属性并和谐组合。

## 方法详解

### 整体框架
两阶段流程。**部件定位阶段**：先运行基础扩散获取对象掩码 $\mathcal{M}_o$，然后在 $T_{th} \approx T/2$ 后运行并行部件扩散（部件 prompt 条件的 U-Net 仅在对象区域内去噪）。从部件扩散的自注意力图做谱聚类得到 K 个分割图，用交叉注意力的点积协议将分割图分配给部件 token，得到每个部件的掩码 $\mathbf{M}_{\mathbf{p}_i}$。**部件生成阶段**：每个部件运行独立的区域扩散（基于 Rich-Text 接口的属性描述），通过掩码加权组合噪声预测，并与基础生成的背景混合。

### 关键设计

1. **部件扩散定位（Part Diffusion）**

    - 功能：从预训练扩散模型中零样本提取对象部件的空间掩码
    - 核心思路：在对象掩码区域内，用部件 token 列表（如 "beak crown wings"）条件化的 U-Net 输出替代基础扩散输出：$\epsilon_t = \alpha \mathcal{M}_o \odot D(x_t, \hat{\mathbf{p}}, t) + (1-\alpha\mathcal{M}_o) \odot D(x_t, \hat{\mathbf{b}}, t)$。这迫使部件 U-Net 学习在有限区域内去噪各部件，使每个部件 token 的交叉注意力聚焦于正确位置。对自注意力图做谱聚类获得 K 个空间分割，用点积协议（而非平均注意力）将分割分配给部件——点积偏好仅在局部高激活的注意力图，避免噪声干扰
    - 设计动机：标准 SD 的部件 token 注意力非常弱且不精确。通过限制 U-Net 只在对象区域去噪，部件 token 被迫关注具体区域

2. **Rich-Text 接口 + 区域扩散生成**

    - 功能：为每个部件独立生成用户指定的细粒度属性
    - 核心思路：用户通过 Rich-Text 接口（支持 footnote 描述、颜色 RGB 值、风格和大小）指定每个部件的属性。每个部件运行独立的扩散过程，通过掩码组合：$\epsilon_t = \sum_i \mathbf{M}_{\mathbf{p}_i} \odot D(x_t, f(\mathbf{p}_i, \mathbf{a}_i), t)$。背景区域与基础生成混合保持整体结构。颜色属性通过梯度引导精确实现 RGB 值
    - 设计动机：部件级生成需要比文本更精细的控制（如精确 RGB 颜色），Rich-Text 接口提供了自然的多属性指定方式

3. **定位质量保障机制**

    - 功能：防止将部件掩码错误分配给不相关的区域
    - 核心思路：通过交叉注意力的最大值判断部件是否被成功定位：$L(j) = \mathds{1}\{\max(\hat{\mathbf{m}}_j) \geq (1-\delta)/K\}$。未定位的部件保持原始状态，避免错误修改。独立文本嵌入初始化（"A photo of {part} of a {object}"）使部件 token 嵌入更有意义
    - 设计动机：部件注意力图本质上有噪声，宁可不定位也不要错误定位

### 损失函数 / 训练策略
完全无训练方法，仅使用预训练 SD 模型。对真实图像使用 Null-Text Inversion 获取反演 latent。DDIM 50 步，CFG scale 8.5。

## 实验关键数据

### 主实验

| 方法 | LPIPS↓ (定位) | CLIP↑ (一致性) | Aesthetic (美学) |
|------|-------------|-------------|----------------|
| **PartComposer** | **0.168** | **0.201** | 5.66 |
| StableDiffusion | 0.467 | 0.183 | 5.68 |
| InstructPix2Pix | 0.189 | 0.193 | 5.63 |
| Rich-Text | 0.243 | 0.187 | 5.65 |

零样本无监督部件分割（CUB200, FG-NMI/ARI）：

| 方法 | FG-NMI | FG-ARI |
|------|--------|--------|
| SD baseline | 8.0 | 0.6 |
| Rich-Text | 3.1 | 0.3 |
| **PartComposer** | **20.5** | **9.2** |
| Unsup-Parts (有训练) | 46.0 | 21.0 |

### 消融实验

| 配置 | FG-NMI | FG-ARI |
|------|--------|--------|
| 完整 PartComposer | 35.4 | 11.0 |
| 无 Null-Text Inversion | 23.1 | 5.2 |
| 无 Max 定位 | 21.3 | 2.8 |
| 无点积分配 | 23.7 | 5.0 |

### 关键发现
- 部件扩散定位比 SD 基线提升 12.5 FG-NMI，证明在对象区域内去噪确实激活了部件级注意力
- 用户研究（28 人）强烈偏好 PartComposer 的定位和一致性
- 方法可泛化到 SDXL 和跨风格域（莫奈绘画、皮克斯角色等）
- Null-Text Inversion 对真实图像的定位质量至关重要（FG-NMI 从 23.1 提升到 35.4）

## 亮点与洞察
- **部件扩散的核心洞察**巧妙：限制去噪区域迫使部件 token 竞争空间位置，自然产生部件定位。这是一种从扩散模型中提取更细粒度知识的通用策略
- **点积分配协议**优于平均注意力：噪声注意力图中只有局部高激活区域可靠，点积偏好这种模式
- **完全无训练+通用性强**：同一方法在鸟、人、卡通角色、绘画等所有域都可工作

## 局限与展望
- 部件生成质量受限于部件定位——如果定位失败则生成也失败
- 注意力图本质上有噪声，某些部件（尤其是小部件或重叠部件）可能无法正确定位
- 真实图像需要 Null-Text Inversion，增加计算开销
- 仅限于扩散模型有语义理解的部件——非常规部件可能不被识别

## 相关工作与启发
- **vs Rich-Text**: Rich-Text 操作在对象级，修改整个对象区域。PartComposer 进一步分解到部件级，只修改指定部件
- **vs InstructPix2Pix**: 编辑方法做全局修改。PartComposer 通过掩码实现严格局部修改
- **vs ControlNet+掩码**: 需要手动提供部件掩码。PartComposer 自动从扩散模型提取部件掩码

## 评分
- 新颖性: ⭐⭐⭐⭐ 部件扩散定位的核心idea有创意，零样本部件分割是新设置
- 实验充分度: ⭐⭐⭐⭐ 定位评估+生成评估+用户研究+消融+跨域泛化，较全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示丰富
- 价值: ⭐⭐⭐⭐ 为创意设计提供了前所未有的部件级控制能力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)
- [\[CVPR 2026\] ExpPortrait: Expressive Portrait Generation via Personalized Representation](../../CVPR2026/image_generation/expportrait_expressive_portrait_generation_via_personalized_representation.md)
- [\[CVPR 2025\] ObjectMover: Generative Object Movement with Video Prior](objectmover_generative_object_movement_with_video_prior.md)
- [\[CVPR 2025\] BootPlace: Bootstrapped Object Placement with Detection Transformers](bootplace_bootstrapped_object_placement_with_detection_transformers.md)

</div>

<!-- RELATED:END -->
