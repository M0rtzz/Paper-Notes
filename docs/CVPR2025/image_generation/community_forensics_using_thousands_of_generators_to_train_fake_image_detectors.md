---
title: >-
  [论文解读] Community Forensics: Using Thousands of Generators to Train Fake Image Detectors
description: >-
  [CVPR 2025][图像生成][AI生成图像检测] 构建包含 4803 个生成模型、270 万张图像的 Community Forensics 数据集，发现即使架构相似的模型也能通过增加数量显著提升假图检测泛化性，在多个基准上达到最优平均 mAP 0.966。
tags:
  - CVPR 2025
  - 图像生成
  - AI生成图像检测
  - 数据多样性
  - 社区模型
  - 泛化检测
  - 大规模数据集
---

# Community Forensics: Using Thousands of Generators to Train Fake Image Detectors

**会议**: CVPR 2025  
**arXiv**: [2411.04125](https://arxiv.org/abs/2411.04125)  
**代码**: https://jespark.net/projects/2024/community_forensics (数据集)  
**领域**: AI安全 / 扩散模型  
**关键词**: AI生成图像检测, 数据多样性, 社区模型, 泛化检测, 大规模数据集

## 一句话总结
构建包含 4803 个生成模型、270 万张图像的 Community Forensics 数据集，发现即使架构相似的模型也能通过增加数量显著提升假图检测泛化性，在多个基准上达到最优平均 mAP 0.966。

## 研究背景与动机

**领域现状**：AI 生成图像检测面临严峻的泛化挑战——检测器在训练集中的生成模型上表现好，但面对未见模型时急剧下降。每个生成器有独特的架构、损失函数、训练数据和图像处理流程，导致模型特异性的指纹差异大。

**现有痛点**：现有检测数据集的模型多样性严重不足——最大的 RED140 也只有 140 个模型。Wang et al. 仅用 1 个 GAN 训练检测器，严重依赖数据增强超参数，对新模型泛化差。即使 Synthbuster、GenImage 等使用了更多模型，仍不超过 20 个，远不够覆盖野外遇到的生成器多样性。

**核心矛盾**：开源社区（如 Hugging Face）已有数千个生成模型，但检测器训练数据只用了少数几个，导致巨大的训练-测试分布差异。需要大规模收集且系统化的多模型数据集。

**本文目标** 通过构建远超以往规模和多样性的检测数据集，研究模型多样性如何影响检测泛化性。

**切入角度**：Hugging Face 上数千个 text-to-image 模型使用统一的 diffusers 库接口，可以自动化批量下载和采样。虽然大部分是 latent diffusion 变体，但每个模型的微调数据、LoRA 配置、图像处理细节各不相同，集体涵盖了大量细微变化。

**核心 idea**：系统化下载 4763 个开源扩散模型 + 19 个手选模型 + 11 个商用模型构建 270 万图像数据集，证明模型数量的增加（即使架构相似）可以对数线性地提升检测泛化性。

## 方法详解

### 整体框架
数据集分三部分。**系统收集**：按下载量排序从 Hugging Face 下载 4763 个 latent diffusion 模型，每个采样约 403 张图像（用真实数据集的 caption 作为 prompt），总计 190 万张。**手选开源模型**：19 个不同架构的模型（GAN、像素扩散、自回归等），每个平均 40K 张，共 77.4 万张。**商用模型**：11 个（DALL·E、Midjourney、FLUX 等），15K 张用于评测。检测器训练用标准 CNN/ViT 分类器端到端训练。

### 关键设计

1. **大规模自动化模型采样流水线**

    - 功能：系统化从模型社区采集数千个生成模型的图像
    - 核心思路：利用 Hugging Face diffusers 库的统一接口，自动下载模型、提取超参数（步数、guidance scale、pipeline 配置），用来自 LAION/ImageNet/COCO 等真实数据集的 caption 作为 text prompt 采样。图像保存为 PNG（避免 JPEG 压缩偏差）。每个模型只采样几百张（更多无明显收益）。不兼容自动流水线的模型（如像素扩散模型）手动处理并放入测试集
    - 设计动机：模型多样性比每模型图像数更重要（关键发现）。自动化流水线使得收集 4763 个模型变得可行

2. **模型数量-泛化性的对数线性关系**

    - 功能：证明增加训练中的模型数量是提升检测泛化性的核心因素
    - 核心思路：固定总训练图像数，仅改变模型数量（1/10/100/1000 个）。发现检测性能随模型数量对数线性增长，且这种提升不仅在同类型（latent diffusion）模型上有效，在分布外模型（GAN、像素扩散）上改善更大。这说明不同的 latent diffusion 微调版本捕获了不同的生成伪影模式，训练检测器学到了更通用的真伪区分线索
    - 设计动机：此前未有工作系统研究"模型数量"这一多样性维度对检测的影响

3. **跨架构多样性的额外收益**

    - 功能：加入不同架构的模型进一步提升泛化性
    - 核心思路：除 4763 个 latent diffusion 模型外，加入 GAN、自回归、像素扩散等手选模型。实验显示架构多样性带来显著额外提升——分类器不能完全在架构间泛化，每种新架构提供独特的检测信号
    - 设计动机：野外会遇到各种架构的生成器，仅靠 latent diffusion 变体不够

### 损失函数 / 训练策略
标准二分类（真/假）交叉熵损失。使用 CNN（ResNet）和 ViT 两种架构端到端训练。发现简单的端到端训练即可获得强泛化性（无需特殊架构或预训练），与此前工作中需要 CLIP 特征的结论不同。

## 实验关键数据

### 主实验（跨基准平均 mAP）

| 训练数据 | Wang↑ | Ojha↑ | Synthbuster↑ | GenImage↑ | Ours(Comp)↑ | 平均 |
|---------|-------|-------|-------------|----------|------------|------|
| Wang et al. | 0.897 | 0.696 | 0.516 | 0.642 | 0.537 | 0.648 |
| Ojha et al. | 0.939 | 0.957 | 0.620 | 0.797 | 0.592 | 0.760 |
| GenImage | 0.929 | 0.984 | 0.813 | 0.999 | 0.912 | 0.934 |
| **Ours** | **0.964** | **0.991** | **0.904** | **0.990** | **0.971** | **0.966** |
| Ours (High res) | 0.967 | 0.996 | 0.974 | 0.998 | 0.987 | 0.986 |

### 消融实验

| 训练模型数量 | 同架构测试mAP | 跨架构测试mAP |
|-------------|-------------|-------------|
| 1 模型 | ~0.7 | ~0.5 |
| 100 模型 | ~0.85 | ~0.7 |
| 1000 模型 | ~0.92 | ~0.85 |
| **4763 模型** | **~0.97** | **~0.95** |

（性能随模型数量对数线性增长）

### 关键发现
- 即使所有新增模型都是 latent diffusion 变体，检测性能在所有生成器类型上都持续提升，包括 GAN 和像素扩散（最反直觉的发现）
- 端到端 CNN/ViT 训练即可获得强泛化性，不需要 CLIP 特征或特殊架构（与 Ojha et al. 的结论不同）
- 高分辨率输入（不resize）进一步提升 mAP 到 0.986
- Synthbuster 基准最难（mAP 0.904），因为其使用 Dresden 真实图像（无 JPEG 压缩），消除了压缩偏差

## 亮点与洞察
- **"模型数量比模型类型更重要"的发现**高度实用——不需要等新架构出现，持续从社区收集 latent diffusion 变体就能提升检测能力
- **对数线性关系**意味着性能提升有明确的 scaling law，可以预测未来收集更多模型的收益
- **数据集构建方法**本身是一个可复用的流水线——任何使用 diffusers 库的模型都可自动纳入

## 局限与展望
- 数据集严重偏向 latent diffusion（4763 个中几乎全是），其他架构（GAN、自回归）仅 19 个手选模型
- 像素扩散模型不兼容自动化流水线，需手动处理
- 本质是数据驱动方法——需要持续收集新模型来保持检测能力，没有学到架构无关的泛化特征
- 未研究对抗性攻击（如后处理逃避检测）的鲁棒性

## 相关工作与启发
- **vs Wang et al.**: 用 1 个 ProGAN 训练，泛化极差（平均 mAP 0.648 vs 0.966）。证明单模型训练根本不够
- **vs Ojha et al.**: 用 CLIP 特征+线性分类器，4 个模型训练。本文用端到端训练+4803 模型显著更优
- **vs GenImage**: 8 个模型，1.4M 图像。本文模型数量 600×，平均 mAP 从 0.934 提升到 0.966
- **vs RED116/140**: 116-140 个模型但每模型仅 1K 图像。本文在模型数量和总图像数上都大幅领先

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统化大规模收集社区模型的思路新颖，scaling law 发现有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个评测基准+详细 scaling 分析+跨架构消融，极其全面
- 写作质量: ⭐⭐⭐⭐ 数据集构建描述详细，实验分析深入
- 价值: ⭐⭐⭐⭐⭐ 数据集+方法论对 AI 安全领域有重大影响，数据集已公开

<!-- RELATED:START -->

## 相关论文

- [Visual Language Models as Zero-Shot Deepfake Detectors](../../ICML2025/image_generation/visual_language_models_as_zero-shot_deepfake_detectors.md)
- [Flatten Graphs as Sequences: Transformers are Scalable Graph Generators](../../NeurIPS2025/image_generation/flatten_graphs_as_sequences_transformers_are_scalable_graph_generators.md)
- [SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images](../../CVPR2026/image_generation/simlbr_learning_to_detect_fake_images_by_learning_to_detect_real_images.md)
- [TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)
- [Reversing Flow for Image Restoration](reversing_flow_for_image_restoration.md)

<!-- RELATED:END -->
