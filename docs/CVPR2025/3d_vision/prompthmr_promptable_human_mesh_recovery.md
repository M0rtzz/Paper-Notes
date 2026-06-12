---
title: >-
  [论文解读] PromptHMR: Promptable Human Mesh Recovery
description: >-
  [CVPR 2025][3D视觉][人体网格恢复] PromptHMR 提出了一种基于 Transformer 的可提示式人体姿态与形状估计方法，通过空间提示（边界框、分割掩码）和语义提示（语言描述、交互标签）灵活引导全图 3D 人体重建，在多个基准上达到 SOTA 并支持视频版的世界坐标运动估计。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "人体网格恢复"
  - "可提示式估计"
  - "多模态提示"
  - "SMPL-X"
  - "人体交互"
---

# PromptHMR: Promptable Human Mesh Recovery

**会议**: CVPR 2025  
**arXiv**: [2504.06397](https://arxiv.org/abs/2504.06397)  
**代码**: [https://yufu-wang.github.io/phmr-page](https://yufu-wang.github.io/phmr-page)  
**领域**: 3D视觉 / 人体姿态估计  
**关键词**: 人体网格恢复, 可提示式估计, 多模态提示, SMPL-X, 人体交互

## 一句话总结

PromptHMR 提出了一种基于 Transformer 的可提示式人体姿态与形状估计方法，通过空间提示（边界框、分割掩码）和语义提示（语言描述、交互标签）灵活引导全图 3D 人体重建，在多个基准上达到 SOTA 并支持视频版的世界坐标运动估计。

## 研究背景与动机

**领域现状**：3D 人体姿态与形状（HPS）估计经典范式是"像素到参数"——从紧密裁剪的人物图像回归 SMPL 参数。裁剪方法精度高但丢失了场景上下文；全图方法保留上下文但检测和精度往往不佳。

**现有痛点**：(1) 裁剪方法无法利用场景上下文，在遮挡、密集人群、人人交互等场景表现受限；(2) 全图方法面临检测遗漏和精度不足的双重挑战；(3) 最近的基于语言的方法（如 ChatPose）虽然尝试结合视觉语言模型（VLM），但 3D 精度远低于 SOTA；(4) 体型估计在单目视图下存在严重的透视歧义。

**核心矛盾**：VLM 对图像中人物的语义理解强但缺乏 3D 理解，而度量回归器对 3D 人体理解深入但不理解语义。如何让两者互补？

**本文目标**：设计一个可以接受多种"侧信息"（side information）作为提示的统一 HPS 框架，既能处理全图保留场景上下文，又能通过各类提示提升不同场景下的鲁棒性和精度。

**切入角度**：将 HPS 重新定义为一个提示引导的回归问题——给定图像和多种模态的提示（边界框、掩码、文本、交互标签），回归每个被提示人物的 SMPL-X 参数。

**核心 idea**：用空间提示替代传统检测，用语义提示补充视觉信息不足（如体型描述改善体型估计），设计可切换的跨人注意力层处理交互场景，实现灵活、鲁棒、高精度的多人 HPS。

## 方法详解

### 整体框架

输入一张 896×896 的完整图像和每人的提示集合。图像通过 DINOv2 ViT 编码器提取特征（每帧只运行一次，与人数无关）。可选的分割掩码通过卷积编码器降采样后加到图像 token 上。各类提示（边界框、文本描述、交互标签）通过提示编码器映射为统一维度的 token。SMPL-X 解码器将图像特征和提示 token 送入 Transformer 解码器，输出每人的 SMPL-X 参数（朝向、姿态、体型、平移）。

### 关键设计

1. **多模态提示编码器（Multi-modal Prompt Encoder）**:

    - 功能：将不同类型的输入提示统一编码为相同维度的 token 向量
    - 核心思路：边界框通过位置编码 + 学习嵌入生成 2 个 token $T_{bi} \in \mathbb{R}^{2 \times d}$；训练时模拟全身框、人脸框、截断框三种类型并加高斯噪声，推理时自适应接受任意框类型。文本描述（如"高大肌肉男性"）通过 CLIP 文本编码器生成 token $T_{ti} \in \mathbb{R}^d$；训练数据中使用 SHAPY 的体型-属性方法自动生成描述。交互标签 $k_i$ 控制跨人注意力层的开关。缺失的提示用学习到的空 token 替代，训练时随机掩码不同类型的提示
    - 设计动机：统一编码使模型能灵活接受不同组合的输入，随机掩码训练让模型在测试时对任意提示子集都能工作

2. **SMPL-X 解码器与位置回归**:

    - 功能：从图像特征和提示 token 回归每人的 SMPL-X 参数和 3D 位置
    - 核心思路：解码器由 3 个注意力块组成，每块包含自注意力、跨人注意力（可选）和双向交叉注意力。使用分离的 query token $T_{smpl}$ 和 $T_{depth}$ 分别回归姿态/体型和位置。位置不直接回归 $\tau$，而是预测归一化的 2D 平移 $p_{xy}$ 和逆深度 $p_z$，再通过焦距转换：$t_{xy} = p_{xy}/p_z$，$t_z = (1/p_z) \cdot (f/f_c)$
    - 设计动机：位置和姿态的表示空间差异大，分离 token 使位置表示不受 3D 姿态影响；逆深度与人在图像中的大小线性相关，预测更直观稳定

3. **可提示跨人交互层（Cross-Person Interaction）**:

    - 功能：在解码器中建模双人交互，提升紧密接触场景的估计精度
    - 核心思路：实现为带残差连接的流控制。当交互标签 $k_i$ 激活时，两人的 query token 经过额外的自注意力层进行信息交换；否则跳过该层。token 加位置编码区分两人身份，注意力输出通过残差连接融合。训练时使用 CHI3D 和 HI4D 双人交互数据集
    - 设计动机：对所有人都做注意力会在拥挤场景中产生不必要的依赖；通过可切换设计，只在有交互标注时启用，避免数据多样性不足的问题，提升灵活性

### 损失函数 / 训练策略

使用 2D 和 3D 联合损失：$\mathcal{L} = \lambda_1\mathcal{L}_{2D} + \lambda_2\mathcal{L}_{3D} + \lambda_3\mathcal{L}_{SMPL} + \lambda_4\mathcal{L}_V + \lambda_5\mathcal{L}_t$。$\mathcal{L}_{2D}$ 为重投影 2D 关节误差，$\mathcal{L}_{3D}$ 为 3D 关节误差，$\mathcal{L}_{SMPL}$ 为 SMPL 参数误差，$\mathcal{L}_V$ 为顶点误差，$\mathcal{L}_t$ 为归一化平移和逆深度误差。

训练使用 BEDLAM、AGORA、3DPW、COCO、MPII 等标准数据集，加 CHI3D 和 HI4D 交互数据。AdamW 优化器，batch size 96，图像分辨率 896×896，训练 350K 步收敛。

## 实验关键数据

### 主实验（相机空间重建 PA-MPJPE mm）

| 方法 | 3DPW | EMDB | RICH | 类型 |
|------|------|------|------|------|
| CLIFF | 43.0 | 68.3 | 68.1 | 裁剪 |
| HMR2.0a | 44.4 | 61.5 | 60.7 | 裁剪 |
| CameraHMR | 35.1 | 43.3 | 34.0 | 裁剪 |
| Multi-HMR | 45.9 | 50.1 | 46.3 | 全图 |
| **PromptHMR** | **36.6** | **41.0** | **37.3** | 全图 |
| **PromptHMR-Vid** | **35.5** | **40.1** | **37.0** | 视频 |

### 消融实验（体型提示，HBW 数据集）

| 训练 w/ text | 测试 w/ text | Height | Chest | Waist | Hip |
|-------------|-------------|--------|-------|-------|-----|
| ✗ | ✗ | 69 | 51 | 88 | 63 |
| ✓ | ✗ | 69 | 48 | 86 | 60 |
| ✓ | ✓ | **62** | **43** | **76** | **58** |

### 关键发现

- 全图方法 PromptHMR 在 PA-MPJPE 上匹配甚至超越裁剪方法（如 CameraHMR），证明提示式方法可以兼顾场景上下文和精度
- 文本提示仅在训练时使用就能提升体型精度，测试时提供文本进一步改善（Height 误差从 69→62 mm）
- 交互层在不使用 HI4D 训练数据时仍能改善双人度量（Pair-PA-MPJPE 从 87.2→73.0），展现了跨域泛化能力
- 掩码提示在紧密接触场景中优于边界框（边界框歧义，掩码更精确）
- 视频版 PromptHMR-Vid 结合 TRAM 的 metric SLAM 在世界坐标运动估计上达到 SOTA

## 亮点与洞察

- **范式转变**：将 HPS 从"像素到参数"重新定义为"像素 + 提示到参数"，打开了与 VLM 协作的大门
- **多模态协同的实用方案**：不同类型的提示（框、掩码、文本、交互）各有擅长场景，随机掩码训练使模型灵活适配
- **分离位置和姿态**的设计简洁有效，逆深度表示也体现了对单目深度估计文献的良好借鉴

## 局限与展望

- 形状描述和交互提示目前需要手动提供，未来应与 VLM 集成实现自动提示
- 单目回归方法在紧密交互时仍不可避免产生人体互穿
- 只考虑了 body pose，未建模面部和手部参数
- 更多类型的侧信息（动作描述、3D 场景上下文、身体测量值）可能在不同场景提供额外收益

## 相关工作与启发

- 与 SAM 的"可提示分割"理念一脉相承，证明了提示范式在回归任务中也能奏效
- ChatPose 等 VLM 方法方向正确但精度不足，PromptHMR 提供了一种更实用的"VLM + 度量回归器"协作路径
- BUDDI 等优化方法处理交互是后处理方式，PromptHMR 在回归阶段直接建模交互更端到端

## 评分

- **新颖性**: 8/10 — 可提示 HPS 的框架设计新颖，多模态协同简洁有效
- **实验充分度**: 9/10 — 覆盖多个数据集、多种提示类型的详尽消融
- **写作质量**: 9/10 — 动机清晰，实验展示丰富
- **价值**: 9/10 — 对人体感知领域有显著推动，提示范式可推广到其他回归任务

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MEGA: Masked Generative Autoencoder for Human Mesh Recovery](mega_masked_generative_autoencoder_for_human_mesh_recovery.md)
- [\[CVPR 2025\] HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery](heatformer_a_neural_optimizer_for_multiview_human_mesh_recovery.md)
- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](../../ICCV2025/3d_vision/ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [\[ECCV 2024\] Global-to-Pixel Regression for Human Mesh Recovery](../../ECCV2024/3d_vision/global-to-pixel_regression_for_human_mesh_recovery.md)
- [\[ICML 2025\] ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization](../../ICML2025/3d_vision/adhmr_aligning_diffusion-based_human_mesh_recovery_via_direct_preference_optimiz.md)

</div>

<!-- RELATED:END -->
