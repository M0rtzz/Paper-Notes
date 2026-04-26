---
title: >-
  [论文解读] FineCaption: Compositional Image Captioning Focusing on Wherever You Want at Any Granularity
description: >-
  [CVPR 2025][图像分割][区域描述] FineCaption 提出一种支持任意 mask 引用和高分辨率图像输入的视觉语言模型，结合 mask 感知 CLIP 编码器、ConvNeXT 和 SAM 高分辨率编码器，以及新构建的 CompositionCap 数据集，实现了多粒度组合式区域图像描述任务。
tags:
  - CVPR 2025
  - 图像分割
  - 区域描述
  - 视觉语言模型
  - mask引用
  - 高分辨率编码
  - 组合式属性
---

# FineCaption: Compositional Image Captioning Focusing on Wherever You Want at Any Granularity

**会议**: CVPR 2025  
**arXiv**: [2411.15411](https://arxiv.org/abs/2411.15411)  
**代码**: https://hanghuacs.github.io/FineCaption/  
**领域**: 分割 / 多模态VLM  
**关键词**: 区域描述, 视觉语言模型, mask引用, 高分辨率编码, 组合式属性

## 一句话总结

FineCaption 提出一种支持任意 mask 引用和高分辨率图像输入的视觉语言模型，结合 mask 感知 CLIP 编码器、ConvNeXT 和 SAM 高分辨率编码器，以及新构建的 CompositionCap 数据集，实现了多粒度组合式区域图像描述任务。

## 研究背景与动机

**领域现状**：预训练视觉语言模型（VLM）在多模态任务中表现出色，包括图像描述、视觉问答等。近期研究开始探索区域级理解，如 Kosmos-2 和 Shikra 通过边界框坐标实现区域引用，Ferret 和 ViP-LLaVA 使用叠加视觉提示进行自由形式区域引用。

**现有痛点**：现有区域引用方式存在明显缺陷。边界框的 IoU 精度不足（与 mask 的平均 IoU 仅 56.11%），无法精确指代不规则形状区域。叠加在图像上的自由形式视觉提示（如圆圈、箭头）容易被 VLM 误解为图像语义内容的一部分，导致引用失败。此外，大多数模型处理分辨率仅 224×224 到 448×448，无法感知细粒度的组合式属性信息（如材质、纹理、姿态等）。

**核心矛盾**：精确的区域引用需要 mask 级别的输入，但现有 VLM 的视觉编码器不支持 mask 输入；细粒度组合属性描述需要高分辨率感知，但大分辨率带来巨大计算开销。

**本文目标** (1) 如何让 VLM 精确识别任意形状的 mask 引用区域？(2) 如何在保持效率的同时支持高分辨率图像输入？(3) 如何训练模型生成多粒度的组合式区域描述？

**切入角度**：作者从编码器架构和数据集两方面同时入手。模型端引入 Alpha-CLIP 的 mask 感知编码机制并补充高分辨率编码器；数据端构建了包含 18 种组合属性的人工标注高质量数据集 CompositionCap。

**核心 idea**：通过融合 mask 感知低分辨率编码和双高分辨率编码（ConvNeXT + SAM），配合专门的组合式属性数据集，实现精确区域引用下的多粒度图像描述。

## 方法详解

### 整体框架

FineCaption 的输入包括低分辨率图像（336×336）、高分辨率图像（1024×1024）和二值 mask。低分辨率图像和 mask 通过 mask 感知 CLIP 编码器处理，高分辨率图像分别通过 ConvNeXT 和 SAM 编码器提取特征。三路特征经过通道拼接融合、适配器映射后送入大语言模型（LLM）生成文本输出。训练分为三阶段：预训练对齐、mask-图像对齐预训练、全量微调。

### 关键设计

1. **Mask 感知编码 (Mask-Aware Encoding)**:

    - 功能：让 CLIP 视觉编码器能够感知 mask 引用的区域
    - 核心思路：沿用 Alpha-CLIP 的方法，在 CLIP 编码器的 patch embedding 层旁添加一个额外的卷积层 $\text{Conv}_\alpha$ 处理二值 mask。mask embedding 与 patch embedding 相加后送入 CLIP 的 Transformer 编码器：$\mathbf{E}_{\text{seq}} = \text{Flatten}(\mathbf{E}_{\text{patch}} + \mathbf{E}_{\text{mask}})^\top$。这样保留了原始图像语义的同时注入了区域引用信息
    - 设计动机：直接在图像上叠加 mask 会破坏图像语义，通过独立编码 mask 并在 embedding 层融合，既保留了图像完整性又实现了精确区域引用

2. **双高分辨率编码 (Dual High-Resolution Encoding)**:

    - 功能：从 1024×1024 分辨率图像中提取细粒度空间和语义特征
    - 核心思路：同时使用 ConvNeXT 和 SAM encoder 作为高分辨率编码器。ConvNeXT 擅长捕获层次化视觉特征，SAM encoder 擅长捕获空间结构信息。两者在同一高分辨率输入上独立提取特征 $\mathbf{F}_{\text{HR1}}$ 和 $\mathbf{F}_{\text{HR2}}$，互补的特征表示增强了模型对材质、纹理、形状等组合属性的感知
    - 设计动机：单一编码器难以同时捕获所有维度的细节，两个互补编码器的组合可以覆盖更全面的视觉信息

3. **通道融合与适配 (Channel-wise Fusion & Adaptation)**:

    - 功能：将三路编码特征统一映射到 LLM 的嵌入空间
    - 核心思路：将 mask 感知特征上采样到与高分辨率特征相同的空间尺寸后，三路特征沿通道维度拼接：$\mathbf{F}_{\text{fusion}} = [\mathbf{F}'_M; \mathbf{F}_{\text{HR1}}; \mathbf{F}_{\text{HR2}}]$，然后通过适配器模块映射到 LLM 的 word embedding 空间
    - 设计动机：通道拼接是最简单有效的多源特征融合方式，保留了各编码器的完整信息

### 损失函数 / 训练策略

三阶段训练策略：Stage 1 冻结编码器和 LLM，只训练投影层做视觉-语言对齐（使用 LLaVA-Pretrain 数据）；Stage 2 冻结其他部分，只训练 mask 感知编码器的 alpha 通道做 mask-图像对齐（使用 CompositionCap + GranD + RefCOCO 系列数据）；Stage 3 全量微调。损失函数为标准的自回归负对数似然损失。

## 实验关键数据

### 主实验

CompositionCap 数据集统计：
- 14,590 个实体，5,392 张图像，186,490 条属性描述
- 测试集来自 Open Images：1,000 张图，7,215 个 mask 实体，19,326 条属性描述
- 18 种组合属性：类别名、体型、肤色纹理、服装配饰、交互关系、姿态、相对位置、颜色、材质、视角、形状、表情、发型、年龄等

| 任务 | 模型 | ROUGE-L | BLEU-4 |
|------|------|---------|--------|
| 属性感知区域描述 | GPT-4o | - | - |
| 属性感知区域描述 | LLaMA-3.2 | - | - |
| 属性感知区域描述 | FineCaption | **最优** | **最优** |

### 消融实验

| 配置 | 说明 | 效果 |
|------|------|------|
| 仅 CLIP (336x336) | 无 mask、无高分辨率 | 基线 |
| + Alpha-CLIP mask | 加入 mask 感知编码 | 区域定位精度显著提升 |
| + 单高分辨率编码器 | 仅 ConvNeXT 或仅 SAM | 属性感知能力提升 |
| + 双高分辨率编码器 | ConvNeXT + SAM | 进一步提升，特别是材质和纹理属性 |

### 关键发现
- Mask 引用比边界框引用在区域描述准确性上有明显优势，特别是对不规则形状区域
- 高分辨率编码对颜色、材质、纹理等细粒度属性的感知至关重要，336x336 分辨率下这些属性的描述质量显著下降
- 双编码器比单编码器的提升在材质和纹理属性上最为明显，说明两者的特征确实互补
- 现有强模型（包括 GPT-4o）在组合式属性描述上仍有很大改进空间

## 亮点与洞察
- **CompositionCap 数据集的细粒度属性定义**是一个重要贡献——将区域描述从粗粒度的"这是什么"扩展到 18 个维度的"详细描述它的所有属性"，这个任务定义比传统 referring expression 更实用
- **Alpha-CLIP 的 mask 编码方式**非常优雅——通过在 embedding 层加法融合而非图像层叠加，避免了视觉提示被误解为语义内容的问题
- 三阶段训练中 Stage 2 专门做 mask-图像对齐的设计值得注意——这是一种"分步解耦"的训练思路，避免多个目标在早期互相干扰

## 局限与展望
- 三个编码器的推理开销较大，1024×1024 分辨率下 SAM 和 ConvNeXT 的计算成本不低
- mask 需要外部提供（如通过交互式分割工具），模型本身不能自动生成 mask，限制了端到端应用
- CompositionCap 数据集规模相对较小（5K 训练图），泛化到长尾场景可能不足
- 论文未深入分析模型在不同属性类别上的分项表现差异，也未探讨多属性联合描述时的冲突问题
- 未与最新的基于 mask 的 VLM（如 OMG-LLaVA、RegionGPT）在完全对齐的设置下进行公平对比

## 相关工作与启发
- **vs Alpha-CLIP**: Alpha-CLIP 提出了 mask 感知编码的核心技术，本文在此基础上增加高分辨率编码和组合属性数据集，从"能引用区域"到"详细描述区域属性"
- **vs Osprey**: Osprey 也支持 mask 引用和区域稠密描述，但限于低分辨率，本文通过高分辨率编码器增强了细粒度感知
- **vs GLaMM**: GLaMM 支持 mask 引用和区域属性描述，但不支持高分辨率输入；本文通过 1024×1024 编码实现了更精细的属性感知

## 评分
- 新颖性: ⭐⭐⭐ 各组件（mask编码、高分辨率、数据集）单独看不新，但组合起来解决了一个实际问题
- 实验充分度: ⭐⭐⭐⭐ 数据集构建扎实，多基线对比充分，但缺少更多定量消融
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ CompositionCap 数据集和多粒度区域描述任务定义有重要的研究价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity](holmes-vau_towards_long-term_video_anomaly_understanding_at_any_granularity.md)
- [\[NeurIPS 2025\] Panoptic Captioning: An Equivalence Bridge for Image and Text](../../NeurIPS2025/segmentation/panoptic_captioning_an_equivalence_bridge_for_image_and_text.md)
- [\[ICCV 2025\] VSC: Visual Search Compositional Text-to-Image Diffusion Model](../../ICCV2025/segmentation/vsc_visual_search_compositional_text-to-image_diffusion_model.md)
- [\[CVPR 2025\] Segment Any Motion in Videos](segment_any_motion_in_videos.md)
- [\[CVPR 2025\] SAP: Segment Any 4K Panorama](sap_segment_any_4k_panorama.md)

<!-- RELATED:END -->
