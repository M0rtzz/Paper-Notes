---
title: >-
  [论文解读] Compass Control: Multi Object Orientation Control for Text-to-Image Generation
description: >-
  [CVPR 2025][图像生成][方向控制] 提出 Compass Control，通过引入轻量级方向编码器预测 compass token 并结合耦合注意力定位（CALL）机制，实现文本到图像扩散模型中多物体的精确3D方向控制，仅需合成数据训练即可泛化到未见类别和多物体场景。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "方向控制"
  - "文本到图像生成"
  - "扩散模型"
  - "多物体场景"
  - "注意力约束"
---

# Compass Control: Multi Object Orientation Control for Text-to-Image Generation

**会议**: CVPR 2025  
**arXiv**: [2504.06752](https://arxiv.org/abs/2504.06752)  
**代码**: 无  
**领域**: 3D视觉 / 可控图像生成  
**关键词**: 方向控制, 文本到图像生成, 扩散模型, 多物体场景, 注意力约束

## 一句话总结

提出 Compass Control，通过引入轻量级方向编码器预测 compass token 并结合耦合注意力定位（CALL）机制，实现文本到图像扩散模型中多物体的精确3D方向控制，仅需合成数据训练即可泛化到未见类别和多物体场景。

## 研究背景与动机

### 领域现状

**领域现状**：现有文本到图像（T2I）扩散模型难以通过文字精确控制物体的3D朝向（如"面向右方"语义模糊），需要反复调整提示词

### 现有痛点

**现有痛点**：已有3D控制方法要么需要密集3D信息（多视图图像、3D包围盒），要么仅限于简单单物体场景

### 核心矛盾

**核心矛盾**：缺乏一种用户友好的接口，允许指定每个物体的方向角度，同时生成多物体场景

### 解决思路

**解决思路**：核心问题：如何在保留 T2I 模型原有生成能力的前提下，实现物体级别的解耦方向控制？

## 方法详解

### 整体框架

Compass Control 在 Stable Diffusion 的文本嵌入空间中引入方向感知的 compass token $\mathbf{c}_n$，由轻量级MLP编码器 $\mathcal{P}$ 根据方向角 $\theta_n$ 预测。Compass token 嵌入提示文本中（如 "A photo of $\mathbf{c}_1$ jeep and $\mathbf{c}_2$ sedan"），经过文本编码器后用于条件化去噪U-Net。同时采用LoRA微调U-Net，在合成数据集上训练。

### 关键设计

1. **Compass Token 方向编码**:
    - 功能：将物体方向角编码为文本嵌入空间中的token
    - 核心思路：用三层MLP（带ReLU）将方向角 $\theta$ 映射到文本编码器输入空间的embedding，前置于对应物体token之前
    - 设计动机：将方向作为物体属性嵌入到文本条件中，保持T2I模型原始接口不变，支持多物体场景

2. **耦合注意力定位（CALL）**:
    - 功能：约束 compass token 和物体token的交叉注意力区域，实现物体方向解耦
    - 核心思路：在训练和推理时，用松弛的2D包围盒生成二值掩码 $m$（框内为0，框外为 $-\infty$），对交叉注意力做掩码操作 $\Psi(\mathbf{c}_n) = \text{softmax}(m + QK(\mathbf{c}_n)^T / \sqrt{d_K})$
    - 设计动机：直接训练时compass token会关注不相关区域，导致方向控制失败；通过CALL将compass token与对应物体token绑定在同一区域，实现解耦控制

3. **分阶段训练策略**:
    - 功能：确保物体在包围盒内生成，为CALL奠定基础
    - 核心思路：先在单物体场景训练学习包围盒遵循，再混合单/双物体场景继续训练
    - 设计动机：单阶段训练时物体不遵守包围盒约束，导致多物体布局中物体泄漏到邻近区域

### 损失函数 / 训练策略

- 标准扩散损失训练编码器 $\mathcal{P}$ 和 LoRA 权重
- 合成数据集：10种3D资产×多种布局方向，在Blender中渲染（1000单物体 + 7900双物体场景）
- ControlNet增强：用Canny边缘图条件化ControlNet生成多样背景，避免过拟合到纯黑背景
- 训练配置：SD v2.1，LoRA rank 4，batch size 4，学习率 $10^{-4}$，25K步，松弛系数 $\lambda = 1.2$

## 实验关键数据

### 主实验

| 方法 | Text Align. ↑ | % Obj. Generated ↑ | Angular Err. ↓ |
|------|--------------|-------------------|----------------|
| ViewNeTI | 22.12 | 0.920 | 0.596 |
| Cont-3D-Words | 29.88 | 0.732 | 0.509 |
| LooseControl | 31.60 | 0.656 | 0.385 |
| **Ours (单物体)** | **32.98** | **0.968** | **0.198** |
| LooseControl (多物体) | 31.73 | 0.778 | 0.372 |
| **Ours (多物体)** | **33.93** | **0.964** | **0.215** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无CALL | 方向控制差，物体纠缠 | compass token 关注不相关区域 |
| 单阶段训练 | 多物体生成数量少 | 物体不遵守包围盒导致泄漏 |
| 无ControlNet增强 | 黑色背景过拟合 | 合成数据缺乏背景多样性 |

### 关键发现

- 仅训练1-2物体场景，可泛化到3-5物体的复杂场景
- 可泛化到训练集未见类别（如婴儿车、船、人类等）
- 与DreamBooth结合仅需约10张无位姿图像即可实现个性化方向控制
- 用户研究（57名参与者）中各维度均优于基线方法

## 亮点与洞察

- 将3D方向控制编码为文本嵌入空间中的属性token，设计优雅且保留了T2I模型原有能力
- CALL机制简单有效：通过注意力掩码同时实现方向绑定和物体解耦
- 仅需10种3D资产的合成数据即可训练出强泛化能力的模型
- 证明了T2I扩散模型内在具有某种3D理解能力

## 局限与展望

- 遮挡或重叠严重时控制失败，可能丢失物体或混合属性
- 单角度方向参数化对非刚体（如人体）过于简化
- 推理时需要提供2D包围盒（虽然可以启发式生成）
- 可扩展到三自由度方向控制和更复杂的3D属性

## 相关工作与启发

- 延续了 ViewNeTI、Continuous 3D Words 等方向控制思路，但提升到多物体场景
- CALL机制与 Attend-and-Excite 等注意力约束工作相关，但专门设计用于物体-方向token绑定
- 个性化扩展沿用 DreamBooth 框架，展示了条件化机制的通用性

## 评分

- 新颖性: ⭐⭐⭐⭐ 方向token结合注意力定位的框架设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 消融完整，用户研究充分，个性化等扩展实验丰富
- 写作质量: ⭐⭐⭐⭐ 动机清晰，图示直观
- 价值: ⭐⭐⭐⭐ 在创意设计和3D内容生成领域有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multitwine: Multi-Object Compositing with Text and Layout Control](multitwine_multi-object_compositing_with_text_and_layout_control.md)
- [\[CVPR 2025\] GPS as a Control Signal for Image Generation](gps_as_a_control_signal_for_image_generation.md)
- [\[CVPR 2025\] MCA-Ctrl: Multi-party Collaborative Attention Control for Image Customization](mca_ctrl_attention_control_customization.md)
- [\[CVPR 2025\] PreciseCam: Precise Camera Control for Text-to-Image Generation](precisecam_precise_camera_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation](mvportrait_text-guided_motion_and_emotion_control_for_multi-view_vivid_portrait_.md)

</div>

<!-- RELATED:END -->
