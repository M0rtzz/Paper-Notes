---
title: >-
  [论文解读] OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions
description: >-
  [NeurIPS 2025][图像生成][主体驱动视频定制] OmniVCus 提出了一个前馈式 DiT 框架，通过数据构建流水线 VideoCus-Factory 和两种嵌入机制（Lottery Embedding 和 Temporally Aligned Embedding），实现了多主体、多模态控制条件下的视频定制生成，在身份保持和可控性上显著超越 SOTA。
tags:
  - NeurIPS 2025
  - 图像生成
  - 主体驱动视频定制
  - 多模态控制
  - DiT
  - 多主体生成
  - 前馈生成
---

# OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions

**会议**: NeurIPS 2025  
**arXiv**: [2506.23361](https://arxiv.org/abs/2506.23361)  
**代码**: https://caiyuanhao1998.github.io/project/OmniVCus/ (项目页面)  
**领域**: 扩散模型 / 视频定制生成  
**关键词**: 主体驱动视频定制, 多模态控制, DiT, 多主体生成, 前馈生成

## 一句话总结
OmniVCus 提出了一个前馈式 DiT 框架，通过数据构建流水线 VideoCus-Factory 和两种嵌入机制（Lottery Embedding 和 Temporally Aligned Embedding），实现了多主体、多模态控制条件下的视频定制生成，在身份保持和可控性上显著超越 SOTA。

## 研究背景与动机
主体驱动视频定制（subject-driven video customization）旨在根据用户提供的参考图像，生成包含特定身份主体的视频。现有前馈方法面临三大挑战：

**多主体数据构建困难**：ConceptMaster 仅支持有限类别，Video Alchemist 依赖稀缺的高质量文本-视频对，数据规模有限

**推理时主体数量受限**：训练视频中主体数量有限（通常1-2个），如何在推理时组合更多主体尚未被充分探索

**多模态控制条件缺失**：如何将深度图、分割掩码、相机轨迹、文本编辑指令等控制信号融入主体驱动定制，仍是开放问题

核心 idea：构建大规模多主体训练数据，设计两种特殊的位置嵌入机制，使单一 DiT 模型能灵活组合不同模态的控制信号。

## 方法详解

### 整体框架
OmniVCus 基于 DiT 架构，将文本、图像、视频和控制信号通过 patchify 编码后拼接为一维长 token 序列输入模型。模型可混合训练多种任务：单/双主体定制、深度/掩码到视频、文本到多视角、文本到图像/视频、图像编辑等。

### 关键设计

1. **VideoCus-Factory 数据构建流水线**

    - **视频字幕**：随机从视频中选一帧，用 Kosmos-2 生成描述并检测主体，插入 IMG1/IMG2 等图像标签
    - **主体过滤**：用 SAM-2 跟踪和分割主体，过滤分割失败的情况（如遮挡严重的物体）
    - **数据增强**：对分割出的主体进行随机旋转、缩放、居中、颜色增强，并随机放置背景图。这避免了训练中主体位置/大小/背景泄露，防止 copy-paste 效应
    - **控制信号生成**：同时构建 mask-to-video（主体掩码序列）和 depth-to-video（深度序列）的训练数据对。这些控制数据与主体定制数据不配对，但推理时可灵活组合

2. **Lottery Embedding (LE)**

    - 核心动机：训练视频中主体数量 K 有限，但推理时希望组合 M > K 个主体
    - 设计思路：从 [1, M] 中均匀随机采样 K 个数作为集合 S，排序后作为帧位置嵌入分配给 K 个训练主体
    - 效果：训练时激活了更多帧位置嵌入，使得推理时可以零样本地组合更多主体（如训练2个主体但推理4个）

3. **Temporally Aligned Embedding (TAE)**

    - 核心动机：深度、掩码等控制信号与生成视频时间对齐，应共享相同的时间位置信息
    - 设计思路：对于密集语义控制信号（深度/掩码），用 3D-VAE 编码后与噪声 token 共享相同帧位置嵌入 {M+1, ..., M+N}；仅在噪声 token 上添加时间步嵌入以区分
    - 对于稀疏的相机信号（Plücker 坐标），通过 MLP 映射后直接加到噪声 token 上以减少 token 长度

4. **Image-Video Transfer Mixed (IVTM) 训练**

    - 动机：缺乏主体定制+编辑指令的训练数据
    - 方法：将图像编辑数据与单主体图像/视频定制数据混合训练，通过对齐帧位置嵌入实现编辑效果从图像到视频的迁移
    - 推理时组合编辑指令和主体定制 prompt 即可激活编辑效果

### 损失函数 / 训练策略
- 采用 flow-matching 损失进行联合训练
- 线性插值产生噪声输入：$X^t = tX^1 + (1-t)X^0$
- 模型预测速度场 $V^t = X^1 - X^0$
- 不同任务的训练数据互不配对，某些输入条件在不同样本中自然缺失
- 基于 5B 参数的 T2V DiT 微调 100K 步，batch size 356，64 张 A100

## 实验关键数据

### 主实验

| 方法 | 主体数 | CLIP-T | CLIP-I | DINO-I | 一致性 | 动态度 |
|------|--------|--------|--------|--------|--------|--------|
| VideoBooth | 单 | 0.2541 | 0.5891 | 0.3033 | 0.9593 | 0.4287 |
| DreamVideo | 单 | 0.2799 | 0.6214 | 0.3792 | 0.9609 | 0.4696 |
| Wan2.1-I2V | 单 | 0.2785 | 0.6319 | 0.4203 | 0.9754 | 0.5310 |
| SkyReels | 单 | 0.2820 | 0.6609 | 0.4612 | 0.9797 | 0.5238 |
| **OmniVCus** | **单** | **0.3293** | **0.7154** | **0.5215** | **0.9928** | **0.5541** |
| SkyReels | 多 | 0.2785 | 0.6429 | 0.4107 | 0.9710 | 0.5892 |
| **OmniVCus** | **多** | **0.3264** | **0.6672** | **0.4965** | **0.9908** | **0.6878** |

### 消融实验

| 配置 | CLIP-T | DINO-I | 一致性 | 动态度 |
|------|--------|--------|--------|--------|
| 基线（无过滤/增强） | 0.2175 | 0.2405 | 0.9588 | 0.3759 |
| +主体过滤 | 0.2431 | 0.5053 | 0.9617 | 0.3826 |
| +数据增强 | 0.3293 | 0.5215 | 0.9928 | 0.5541 |

| TAE 消融 | CLIP-T | DINO-I | 一致性 | 动态度 |
|----------|--------|--------|--------|--------|
| Naive 嵌入 | 0.2618 | 0.2947 | 0.9751 | 0.4948 |
| 加到噪声 | 0.1722 | 0.1680 | 0.9319 | 0.5437 |
| **TAE** | **0.3054** | **0.3794** | **0.9909** | **0.4965** |

| LE 消融（多主体） | CLIP-T | DINO-I | 一致性 | 动态度 |
|-------------------|--------|--------|--------|--------|
| 无 LE | 0.2105 | 0.3364 | 0.9702 | 0.6943 |
| **有 LE** | **0.2728** | **0.4163** | **0.9810** | 0.6806 |

### 关键发现
- 数据增强中的随机背景放置极大改善了视频的动态多样性（动态度从 0.38 提升到 0.55）
- TAE 中直接将深度加到噪声会导致模型崩溃，因为精细空间信息被噪声破坏
- LE 使推理时组合3-4个主体成为可能，DINO-I 提升 0.08
- IVTM 训练策略显著优于直接混合训练（CLIP-T: 0.3126 vs 0.2585）

## 亮点与洞察
- VideoCus-Factory 流水线设计巧妙，从无标签原始视频自动构建多主体训练对和控制信号数据
- Lottery Embedding 以简洁的方式实现了训练时少量主体到推理时多主体的泛化，核心是随机激活更多帧位置嵌入
- TAE 的设计灵感来自控制信号与视频帧的时间对齐关系，对密集和稀疏信号采用不同处理策略
- 用户偏好研究（37人）显示 OmniVCus 在身份保持、对齐和质量三方面均大幅领先

## 局限与展望
- 依赖内部大规模视频数据池，复现门槛较高（64 A100, 300M 文生图数据）
- Kosmos-2 和 SAM-2 的检测/分割质量直接影响数据质量
- 主体之间的交互和遮挡关系处理仍有改进空间
- 指令编辑的 hard case（如风格迁移）效果有限

## 相关工作与启发
- 与 ConceptMaster/Video Alchemist 相比，VideoCus-Factory 更通用且数据规模更大
- LE 的思路可以推广到其他需要在推理时扩展能力的场景
- IVTM 的图像-视频迁移训练策略为缺乏配对数据的场景提供了参考

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers](../../ICCV2025/image_generation/freecus_free_lunch_subject-driven_customization_in_diffusion_transformers.md)
- [\[NeurIPS 2025\] Mind-the-Glitch: Visual Correspondence for Detecting Inconsistencies in Subject-Driven Generation](mind-the-glitch_visual_correspondence_for_detecting_inconsistencies_in_subject-d.md)
- [\[NeurIPS 2025\] Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling](track_inpaint_resplat_subject-driven_3d_and_4d_generation_with_progressive_textu.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](../../CVPR2025/image_generation/multi-party_collaborative_attention_control_for_image_customization.md)

<!-- RELATED:END -->
