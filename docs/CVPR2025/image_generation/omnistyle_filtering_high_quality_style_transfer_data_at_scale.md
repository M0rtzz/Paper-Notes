---
title: >-
  [论文解读] OmniStyle: Filtering High Quality Style Transfer Data at Scale
description: >-
  [CVPR 2025][图像生成][风格迁移] 构建了首个百万级风格迁移配对数据集 OmniStyle-1M（100万 content-style-stylized 三元组，1000种风格），设计 OmniFilter 多维质量过滤框架筛选高质量数据，并基于 DiT 架构训练端到端风格迁移模型 OmniStyle，同时支持指令引导和参考图引导的风格迁移，全面超越现有方法。
tags:
  - CVPR 2025
  - 图像生成
  - 风格迁移
  - 大规模数据集
  - 质量过滤
  - Transformer
  - 端到端训练
---

# OmniStyle: Filtering High Quality Style Transfer Data at Scale

**会议**: CVPR 2025  
**arXiv**: [2505.14028](https://arxiv.org/abs/2505.14028)  
**代码**: https://wangyephd.github.io/projects/cvpr25_omnistyle.html (项目页)  
**领域**: 图像生成  
**关键词**: 风格迁移, 大规模数据集, 质量过滤, Diffusion Transformer, 端到端训练

## 一句话总结

构建了首个百万级风格迁移配对数据集 OmniStyle-1M（100万 content-style-stylized 三元组，1000种风格），设计 OmniFilter 多维质量过滤框架筛选高质量数据，并基于 DiT 架构训练端到端风格迁移模型 OmniStyle，同时支持指令引导和参考图引导的风格迁移，全面超越现有方法。

## 研究背景与动机

**领域现状**：风格迁移经历了从优化方法（Neural Style Transfer）到任意风格迁移（AdaIN 等）再到扩散模型方法的演进。近期扩散模型方法分为 tuning-based（需要为每个风格微调）和 tuning-free（通过注意力注入实现），但都依赖迭代推理，速度慢。

**现有痛点**：(1) 现有方法泛化能力不足，只能处理有限的风格类别；(2) 大多采用非端到端架构（优化-based 或 inversion-based），需要数百甚至上千次迭代，计算成本极高；(3) 缺乏大规模高质量配对数据集，比如 IMAGStyle 只有 14 种风格类别且依赖单一模型生成数据；(4) 训练是无监督的，缺乏对风格化输出的精确控制。

**核心矛盾**：高质量的端到端风格迁移需要大规模配对数据来训练，但配对数据获取成本极高；同时现有的自动生成数据质量参差不齐，缺乏有效的质量控制手段。

**本文目标**：(1) 构建一个足够大、足够多样、质量可控的风格迁移配对数据集；(2) 基于该数据集训练一个快速高效的端到端风格迁移模型。

**切入角度**：既然单个风格迁移模型生成的结果质量不稳定，那就用6个SOTA模型集体生成候选结果，再通过一个多维度的自动评估系统筛选出最佳结果。监督式训练取代了无监督的风格对齐。

**核心 idea**：用"多模型生成+智能过滤"的范式大规模构建高质量配对数据，使端到端监督训练成为可能，从根本上解决风格迁移的效率和质量问题。

## 方法详解

### 整体框架

方法包含三个核心组件：(1) OmniStyle-1M 数据集构建——用 FLUX 生成 2000 张内容图像，从 Style30K 选取 1000 风格图像，通过 6 个 SOTA 风格迁移模型生成百万级三元组；(2) OmniFilter 质量过滤框架——从内容保持、风格一致性、美学吸引力三个维度评估并筛选高质量三元组；(3) OmniStyle 模型——基于 FLUX-dev DiT 架构的端到端风格迁移框架，支持指令引导和参考图引导两种模式。

### 关键设计

1. **OmniStyle-1M 数据集构建流程**:

    - 功能：提供大规模、高多样性的风格迁移配对训练数据
    - 核心思路：内容图像通过 ChatGPT 生成 20 个类别的 2000 条 prompt，再由 FLUX 生成无版权问题的图像。风格图像从 Style30K 精选 1000 种细粒度风格。每张内容图随机选 100 种风格，由 StyleID、ArtFlow、StyleShot、AesPANet、CSGO、CAST 六个模型分别生成风格化结果，最终产出超过 100 万个三元组。每个三元组附带文字描述和指令式 prompt。
    - 设计动机：多模型生成避免了单一模型的偏差和局限性，1000 种风格覆盖了漫画、水墨、油画、矢量插画等极其丰富的类型，20 种内容类别确保了分布均衡。

2. **OmniFilter 多维质量过滤框架**:

    - 功能：从三个维度自动评估风格化图像质量，筛选最佳结果
    - 核心思路：**内容保持评估**——结合 CLIP 图文相似度（语义一致性 $S_{semantic}$）和 DINOv2 嵌入相似度（结构完整性 $S_{structural}$），加权为 $C_{score} = 0.5 \cdot S_{semantic} + 0.5 \cdot S_{structural}$。**风格一致性评估**——用 Style30K 数据通过对比学习微调 CLIP 图像编码器，使同风格图像在特征空间聚集，计算风格一致性分 $S_{score}$。**美学评估**——定义 40 个视觉属性（构图、色彩和谐、光影等），用 InternVL2 生成属性描述，在 AVA + BAID 数据集上训练美学评分模型 $A_{score}$。最终综合得分 $Score = 0.2 \cdot C + 0.6 \cdot S + 0.2 \cdot A$，风格权重最高。每对 content-style 从 6 个模型输出中选最高分，最终筛出 OmniStyle-150K。
    - 设计动机：现有评估方法只关注部分维度（如 style loss 只看风格），OmniFilter 三维联合评估确保既保持内容又对齐风格又保证审美品质。风格一致性权重最高（0.6）是因为风格迁移的核心目标就是风格对齐。

3. **OmniStyle 模型架构**:

    - 功能：基于 DiT 的高效端到端风格迁移
    - 核心思路：基于 FLUX-dev 模型，只微调 Diffusion Transformer 部分，冻结其它组件。**参考图引导模式**：用 VAE 提取风格图和内容图的视觉特征，与噪声 latent 和文本 token 空间拼接后送入 MM-DiT，风格和内容特征使用不同的位置编码。**指令引导模式**：内容特征作为文本 token 处理，用 [img]/[/img] 标签区分。参考图模式则用 [img1]/[img2] 标签区分风格和内容图像。
    - 设计动机：基于 FLUX-dev 这样的强力 DiT 基础模型微调，既继承了其强大的生成能力，又通过大规模配对数据实现了端到端风格迁移，无需迭代优化。

### 损失函数 / 训练策略

在 8×NVIDIA H100 上训练，batch size 1/GPU，学习率 1e-4。对输入风格图像做随机裁剪和水平翻转增强风格学习的鲁棒性。只微调 DiT 部分，VAE 和文本编码器冻结。训练数据来自 OmniFilter 筛选后的 OmniStyle-150K 高质量子集。

## 实验关键数据

### 主实验

| 任务 | 方法 | 内容保持↑ | 风格一致性↑ | 美学分↑ | Style Loss↓ |
|------|------|-----------|-------------|---------|-------------|
| 指令引导 | UltraEdit | 0.4906 | 0.5087 | 5.6351 | 0.3432 |
| 指令引导 | DiffStyler | 0.4816 | 0.5127 | 5.4551 | 0.4256 |
| 指令引导 | **OmniStyle** | **0.5128** | **0.6441** | **5.7512** | **0.2873** |
| 参考图引导 | StyleShot | 0.5410 | 0.7347 | 5.7818 | 0.1500 |
| 参考图引导 | CSGO | 0.5067 | 0.7251 | 5.7712 | 0.3727 |
| 参考图引导 | **OmniStyle** | **0.5450** | **0.7483** | **5.7913** | **0.1086** |

### 消融实验 (用户研究)

| 方法 | 指令引导 Rank 1 | 参考图引导 Rank 1 |
|------|----------------|------------------|
| OmniStyle | **86.90%** | **41.22%** |
| StyleShot | - | 18.63% |
| CSGO | - | 14.70% |
| DiffStyler | 1.19% | - |
| UltraEdit | 5.16% | - |
| OmniGen | 5.16% | - |

### 关键发现

- **指令引导任务**中 OmniStyle 以压倒性优势领先：风格一致性 0.6441 大幅超过第二名 OmniGen 的 0.5487，用户研究中 86.9% 的首选率远超所有对手
- **参考图引导任务**中优势同样明显：Style Loss 0.1086 是所有方法中最低的，用户首选率 41.22% 超过第二名 StyleShot 的 18.63%
- OmniStyle 在四个评估维度上实现了均衡的高性能，不像 CAST（高内容保持但风格一致性差）或 CSGO（高风格一致性但高 style loss）存在偏科
- 数据集质量是关键：OmniFilter 的三维过滤确保了训练数据的高质量，相比直接使用所有数据，过滤后的 150K 子集训练效果更好

## 亮点与洞察

- **"多模型生成+智能过滤"的数据构建范式**：这种思路不局限于风格迁移，可以推广到任何需要高质量配对数据的生成任务。用多个弱模型的集体智慧筛选出高质量训练数据，是一个非常实用的工程方法论。
- **OmniFilter 的三维度评估体系**：将内容保持、风格一致性和美学吸引力解耦评估，比单一的 style loss 或 FID 更全面。特别是用对比学习微调 CLIP 来评估风格一致性，以及用 MLLM 视觉属性描述来评估美学，都是很有创意的方案。
- **端到端取代迭代**：通过大规模配对数据的监督训练，将风格迁移从迭代优化问题转化为前馈推理问题，速度提升巨大。

## 局限与展望

- 数据集构建依赖 6 个现有风格迁移模型的生成质量，如果所有模型在某些极端风格上都失败，则无法获得高质量训练数据
- OmniFilter 的评估标准（特别是风格一致性权重 0.6）是手动设定的，可能不适用于所有风格类型
- 模型基于 FLUX-dev 微调，参数量和推理成本较高（8×H100 训练），实际部署可能需要进一步蒸馏
- 内容图像由 FLUX 生成而非真实照片，可能存在域偏差
- 未来可以考虑与专业艺术家合作标注"完美配对"数据集，提供更客观的评估基准

## 相关工作与启发

- **vs IMAGStyle**: IMAGStyle 只有 14 种风格、依赖单一模型 B-LoRA 生成，规模 21 万。OmniStyle-1M 有 1000 种风格、6 个模型、100 万三元组，多样性和规模完全不在一个量级
- **vs StyleShot/CSGO**: 这些方法在参考图引导任务中是强竞争对手，但它们都是非端到端的。OmniStyle 在风格一致性和 style loss 上都更优
- **vs DiffStyler**: DiffStyler 只支持文本引导且风格理解能力有限，无法处理细粒度风格（如"美式漫画"）。OmniStyle 同时支持文本和图像引导

## 评分

- 新颖性: ⭐⭐⭐⭐ 数据集构建范式新颖，OmniFilter 设计精巧，但模型架构本身是 FLUX 微调
- 实验充分度: ⭐⭐⭐⭐⭐ 定量+定性+用户研究，指令和参考图两种模式全覆盖，对比方法丰富
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表精美，数据集构建流程描述详尽
- 价值: ⭐⭐⭐⭐⭐ 数据集+过滤框架+模型三位一体，对风格迁移社区贡献巨大，150K 高质量配对数据集非常有价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] HSI: A Holistic Style Injector for Arbitrary Style Transfer](hsi_a_holistic_style_injector_for_arbitrary_style_transfer.md)
- [\[CVPR 2025\] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer](samam_style-aware_state_space_model_for_arbitrary_image_style_transfer.md)
- [\[CVPR 2025\] StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements](stylestudio_text-driven_style_transfer_with_selective_control_of_style_elements.md)
- [\[CVPR 2025\] 3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion](3dtopia-xl_scaling_high-quality_3d_asset_generation_via_primitive_diffusion.md)
- [\[CVPR 2025\] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)

<!-- RELATED:END -->
