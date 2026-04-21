---
title: >-
  [论文解读] Image Quality Assessment: From Human to Machine Preference
description: >-
  [CVPR 2025][图像分割][图像质量评估] 本文首次提出面向机器视觉的图像质量评估（IQA for MVS），构建了包含 225 万细粒度标注和 3 万参考/失真图像对的 Machine Preference Database (MPD)，实验证明现有 HVS-centric IQA 指标无法准确表征机器偏好，揭示了人类与机器视觉系统间的根本性差异。
tags:
  - CVPR 2025
  - 图像分割
  - 图像质量评估
  - 机器偏好
  - 多模态大模型
  - 下游任务
  - 数据库构建
---

# Image Quality Assessment: From Human to Machine Preference

**会议**: CVPR 2025  
**arXiv**: [2503.10078](https://arxiv.org/abs/2503.10078)  
**代码**: https://github.com/lcysyzxdxc/MPD  
**领域**: 图像质量评估 / 机器视觉  
**关键词**: 图像质量评估, 机器偏好, 多模态大模型, 下游任务, 数据库构建

## 一句话总结

本文首次提出面向机器视觉的图像质量评估（IQA for MVS），构建了包含 225 万细粒度标注和 3 万参考/失真图像对的 Machine Preference Database (MPD)，实验证明现有 HVS-centric IQA 指标无法准确表征机器偏好，揭示了人类与机器视觉系统间的根本性差异。

## 研究背景与动机

**领域现状**：过去二十年，IQA 领域围绕人类视觉系统（HVS）已发展出成熟的方法体系，包括数百个细粒度数据库和精确建模 HVS 偏好的算法。2023 年起，机器对机器（M2M）连接数首次超过机器对人（M2H），机器已成为图像/视频数据的主要消费者。

**现有痛点**：HVS 和机器视觉系统（MVS）的感知机制存在根本差异——HVS 关注纹理、结构、色彩的相似性，而 MVS 关注下游任务（分割、检测、问答）结果的一致性。人眼感知到的明显质量降级可能完全不影响机器下游任务；反之，人眼几乎不可察觉的微小扰动可能严重破坏机器输出。目前甚至没有一个数据库为图像标注机器的整体偏好分数。

**核心矛盾**：图像处理算法评估指标碎片化——不同压缩/复原算法各自使用不同的下游任务和模型作为验证指标，缺乏统一、全面的机器偏好表示。

**本文目标** (1) 定义机器的主观偏好——哪些下游任务、测试模型和评价指标？ (2) 建立大规模机器偏好标注数据库；(3) 验证现有 HVS-centric IQA 算法能否预测机器偏好。

**切入角度**：参考 ITU 人类主观评价标准，将 15 个 LMM + 15 个专用 CV 模型作为"机器受试者"，在 7 个下游任务上收集机器的 Mean Opinion Score (MOS)。

**核心 idea**：用 30 个机器模型在 7 个下游任务上对 3 万张失真图像评分，构建首个机器偏好数据库 MPD，证明现有 IQA 指标无法准确表征机器偏好。

## 方法详解

### 整体框架

整个工作分为三步：(1) 参考图像收集与失真生成——选取 1000 张高质量参考图（NSI/SCI/AIGI），施加 30 种失真 × 5 级强度生成 3 万张失真图；(2) 多任务标注——用 15 个 LMM 和 15 个专用 CV 模型在 7 个任务上标注参考/失真图对的性能差异；(3) MOS 汇总——将 7 个任务分为 5 个维度，每维度 15 个受试者，归一化后汇总为 (0,5) 的 MOS。

### 关键设计

1. **多任务机器偏好定义框架**:

    - 功能：统一定义"机器认为什么是好图像"
    - 核心思路：涵盖 7 个下游任务，分为 LMM 任务（YoN/MCQ/VQA/CAP）和 CV 任务（SEG/DET/RET）。各任务的评分方式经过精心设计：YoN 用 softmax 概率差 $S_{YoN} = |\sigma(P_{dis}) - \sigma(P_{ref})|$；MCQ 用选项概率向量的余弦距离；VQA 用 CLIP 文本编码器计算语义相似度；CAP 用 BLEU+CIDEr+SPICE 综合评价；SEG/DET 用 IoU；RET 用 Top-1/5/10 准确率之和
    - 设计动机：任何单一任务都不能代表机器的全面偏好。通过覆盖 7 种任务并分组为 5 个评分维度（每维 15 个受试者），模拟了人类 IQA 中多人评分取 MOS 的标准做法

2. **大规模失真图像构建**:

    - 功能：全面覆盖现实通信场景中的各类失真
    - 核心思路：参考图像包含三大类：自然场景图（UGC/PGC）、屏幕内容图（网页/游戏/电影）、AI 生成图（6 种文生图模型）。30 种失真分为 7 大类（Blur、Luminance、Chrominance、Contrast、Noise、Compression、Spatial），每种 5 级强度。为确保公平，人工控制失真强度使人眼感知的质量降级在同一级别上对齐
    - 设计动机：现有 IQA 数据库多专注于单一图像类型（自然图或 AI 生成图），且失真类型有限。MPD 通过全面覆盖图像类型和失真类型，确保了评估的完整性

3. **多粒度分析与验证框架**:

    - 功能：从整体和个体两个层面验证 MPD 的可靠性和发现
    - 核心思路：整体分析各任务评分间的 SRCC 相关性、30 种失真下的 MOS 分布；个体分析每个机器受试者在不同强度和内容类型下的表现差异，以及机器受试者间的一致性（SRCC=0.62）。还对比了人类受试者间的一致性（SRCC=0.76），证明机器偏好的"个体差异"甚至大于人类
    - 设计动机：类比 ITU 人类主观评价标准，需要验证受试者信度和评分分布的合理性

### 损失函数 / 训练策略

MPD 本身是数据库贡献，不涉及模型训练。但文中对 10+ 种现有 IQA 指标（PSNR、SSIM、LPIPS、CLIPIQA、ARNIQA 等）在 MPD 上进行了全面 benchmark，作为验证实验。

## 实验关键数据

### 主实验（IQA 指标预测机器偏好的能力）

| IQA 指标 | 严重失真 SRCC ↑ | 轻微失真 SRCC ↑ | NSI SRCC ↑ | 人类设计 |
|----------|----------------|----------------|------------|---------|
| PSNR | 0.387 | 0.310 | 0.412 | ✗ |
| SSIM | 0.597 | 0.267 | 0.629 | ✗ |
| LPIPS | 0.625 | 0.008 | 0.697 | ✔ |
| ARNIQA | 0.834 | 0.239 | 0.870 | ✗ |
| AHIQ | 0.806 | 0.530 | 0.845 | ✗ |
| TOPIQ-FR | 0.718 | 0.425 | 0.751 | ✔ |

关键发现：(1) 所有指标在严重失真上表现尚可，但在轻微失真上急剧下降（LPIPS 降至 0.008）；(2) 专为 HVS 设计的指标（标记 ✔）反而可能更差（如 LPIPS 在轻微失真上几乎失效）。

### 消融实验（机器偏好特征分析）

| 分析维度 | 关键发现 |
|---------|---------|
| 任务间相关性 | VQA 与 SEG/DET/RET 相关性最低，说明不同任务代表不同偏好维度 |
| 机器受试者一致性 | 机器间 SRCC=0.62 < 人类间 SRCC=0.76，机器偏好差异更大 |
| 失真敏感性 | 机器对 Lens blur 极敏感，对 Mean brighten 几乎不受影响 |
| 强度影响 | Lab saturation 随强度增加质量下降，HSV saturation 则几乎不变 |

### 关键发现

- 现有 IQA 指标对轻微失真下的机器偏好预测极差，这恰恰是实际应用中最常见的场景
- 机器对不同失真类型的敏感性与人类截然不同——四种 Macro Block 失真对人眼几乎相同，但机器对 Block exchange 的感知远差于其他三种
- 微调现有 IQA 算法后在 MPD 上有微小提升，但仍无法充分表征机器偏好，说明需要根本性的新方法
- LMM 的先进程度影响其鲁棒性：InternVL2 在 YoN 上表现优异，而传统 LLaVA1.5 在 CAP 上反而更稳定

## 亮点与洞察

- **问题定义具有前瞻性**：在 M2M 通信超过 M2H 的时代，为机器系统性定义图像质量偏好是一个及时且重要的方向开拓工作。随着更多图像被机器而非人类消费，MVS-centric IQA 将成为核心需求
- **数据库构建方法论完善**：参考 ITU 标准设计机器 MOS 收集流程，覆盖 7 个任务和 30 个模型，既有严谨性又有全面性。这个方法论可推广到视频质量评估
- **揭示的 HVS vs MVS 差异具有实际指导价值**：例如，图像压缩算法如果面向机器消费（如自动驾驶、安防监控），应该针对机器敏感的失真类型优化，而非人眼偏好

## 局限与展望

- 参考图像仅 1000 张，尽管失真图有 3 万张，但图像内容多样性仍可能不足
- LMM 的输出受 temperature 等参数影响，虽然设为 0 但不同模型的输出稳定性存在差异
- 7 个任务的权重如何确定是开放问题——当前等权归一化可能不适合所有应用场景
- 未考虑视频质量评估，而视频是机器消费的主要媒体形式
- 机器模型更新迭代极快，数据库的时效性需要持续维护
- 缺少直接面向机器偏好优化的 IQA 算法提案，仅验证了现有方法的不足

## 相关工作与启发

- **vs LIVE/TID2013/KADID**: 这些经典 IQA 数据库仅标注人类偏好，失真类型有限。MPD 首次引入机器偏好维度，且覆盖 NSI/SCI/AIGI 三类图像
- **vs KonIQ-10K/AGIQA-3K**: 这些 NR-IQA 数据库虽然规模大，但仍基于人类主观评价。MPD 揭示了人类评价与机器性能之间的巨大鸿沟
- **vs CMC-Bench**: CMC-Bench 评估 LMM 的多模态能力，MPD 则反过来——用 LMM 作为评估者来标注图像的机器偏好，视角互补

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次提出 IQA for Machine Vision 的完整框架，问题定义开创性
- 实验充分度: ⭐⭐⭐⭐ 225 万标注、30 个模型、10+ IQA 算法的全面验证，但缺乏新算法提案
- 写作质量: ⭐⭐⭐⭐ 数据库构建流程清晰，但部分公式和表格过于密集
- 价值: ⭐⭐⭐⭐⭐ 数据库和方法论对 IQA 社区具有深远影响，有望推动从 HVS 到 MVS 的范式转换

<!-- RELATED:START -->

## 相关论文

- [QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](../../ICML2025/segmentation/qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)
- [The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)
- [GleSAM: Segment Any-Quality Images with Generative Latent Space Enhancement](segment_any-quality_images_with_generative_latent_space_enhancement.md)
- [ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object](ros-sam_high-quality_interactive_segmentation_for_remote_sensing_moving_object.md)
- [SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories](segagent_exploring_pixel_understanding_capabilities_in_mllms_by_imitating_human_.md)

<!-- RELATED:END -->
