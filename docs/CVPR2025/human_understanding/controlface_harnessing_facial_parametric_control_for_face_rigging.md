---
title: >-
  [论文解读] ControlFace: Harnessing Facial Parametric Control for Face Rigging
description: >-
  [人体理解] 提出 ControlFace，利用双分支 U-Net（FaceNet + 去噪 U-Net）结合 3DMM 渲染条件，实现无需微调即可灵活编辑人脸姿态、表情和光照，同时精确保留身份和语义细节。
tags:
  - 人体理解
---

# ControlFace: Harnessing Facial Parametric Control for Face Rigging

| 属性 | 值 |
|------|------|
| 会议 | CVPR 2025 |
| arXiv | [2412.01160](https://arxiv.org/abs/2412.01160) |
| 代码 | [项目页面](https://cvlab-kaist.github.io/ControlFace/) |
| 领域 | 人体理解 / 人脸编辑 |
| 关键词 | face rigging, 3DMM, dual-branch U-Net, diffusion model, reference control guidance |

## 一句话总结

提出 ControlFace，利用双分支 U-Net（FaceNet + 去噪 U-Net）结合 3DMM 渲染条件，实现无需微调即可灵活编辑人脸姿态、表情和光照，同时精确保留身份和语义细节。

## 研究背景与动机

### 领域现状

人脸操控（face rigging）是计算机视觉的基础任务，目标是根据用户指定的姿态、表情、光照等控制信号修改人脸图像，同时保持身份一致性。近年来扩散模型在人脸生成方面展现了强大能力，结合 3D 可变形人脸模型（3DMM，如 FLAME）可以实现参数化的显式控制。

### 现有痛点

1. **依赖图像数据集导致的重建训练困境**：现有方法（如 DiffusionRig、CapHuman）在 FFHQ 等单图像数据集上以重建方式训练。为避免模型直接复制参考图而忽略控制信号，它们不得不将参考图压缩为单一向量（如人脸识别特征），这丢失了发型、背景等精细语义信息。
2. **需要个体特定微调**：对于每个新身份，需要额外的微调数据和训练，实用性差。
3. **控制精度与身份保持的矛盾**：编码参考图信息过多则忽视控制信号，编码过少则丢失身份细节。

### 本文目标

如何在**不需要微调**的前提下，同时实现**精细身份保持**（包括发型、背景等）和**精确的参数化控制**（姿态、表情、光照）？

### 切入角度与核心 idea

利用人脸视频数据集构建配对四元组，避免重建训练困境；使用双分支 U-Net 充分编码参考图的丰富表征；提出控制混合模块（CMM）和参考控制引导（RCG）增强控制精度。

## 方法详解

### 整体框架

ControlFace 采用双分支 U-Net 架构：FaceNet 编码参考图的身份和语义细节，去噪 U-Net 负责生成。两者通过增强自注意力层（Augmented Self-Attention）集成。控制信号通过 Face Controller 和 Control Mixer Module 注入，推理时用 Reference Control Guidance 进一步增强控制精度。

### 关键设计 1：双分支 U-Net + 视频数据训练

- **功能**：捕获参考图的完整身份和语义细节，同时不忽略控制信号
- **核心思路**：FaceNet 和去噪 U-Net 共享相同架构（均初始化自 Stable Diffusion v1.5）。FaceNet 编码参考图，其 key 和 value 与去噪 U-Net 的 key/value 拼接后执行增强自注意力
- **设计动机**：在 CelebV-HQ 人脸视频数据集上训练，随机选取同一视频的两帧作为参考图 $X_R$ 和目标图 $X_T$，构建配对四元组 $\{X_R, X_T, D_R, D_T\}$。这避免了重建训练中 $X_R = X_T$ 导致的控制信号被忽视的问题
- **损失函数**：标准扩散模型去噪损失 $\mathcal{L} = \mathbb{E}[\|\epsilon_\theta(z_{T,t}; t, z_R, D_T, D_R) - \epsilon\|^2_2]$

### 关键设计 2：Control Mixer Module (CMM)

- **功能**：编码目标控制 $D_T$ 与参考控制 $D_R$ 之间的关联特征，增强控制对齐
- **核心思路**：两个共享权重的控制编码器（含卷积层和交叉注意力层）分别编码 $D_R$ 和 $D_T$，输出关联嵌入 $E_R$ 和 $E_T$。这些嵌入被加到增强自注意力的 query、key 上，引导模型注意力
- **设计动机**：仅编码目标控制 $D_T$ 无法让模型理解参考图与目标之间的"变化量"，CMM 通过建模二者关联提供变化方向指引
- **修改后的自注意力**：$\text{Aug-Attn}^* = \text{Softmax}\left(\frac{(Q+E_T)[K+E_T, K^{\text{face}}+E_R]^T}{\sqrt{d}}\right)[V, V^{\text{face}}]$

### 关键设计 3：Reference Control Guidance (RCG)

- **功能**：在推理时增强对目标控制信号的遵循度
- **核心思路**：不同于标准 CFG 用空条件作为 null condition，RCG 用参考控制 $D_R$ 替代 null condition：$\hat{\epsilon}_\theta(\cdot, D_T) = \epsilon_\theta(\cdot, D_R) + w(\epsilon_\theta(\cdot, D_T) - \epsilon_\theta(\cdot, D_R))$
- **设计动机**：标准 CFG 的 null condition（如空输入）提供的 grounding 不佳，差值信号noise较大。而 $D_R$ 与 $D_T$ 共享同一身份，差值集中在面部区域，能精确指示需要修改的部分。可视化显示 RCG 的差值在所有时间步上都是干净的面部对齐估计

## 实验关键数据

### 主实验表

**控制精度（DECA 重推理误差 ↓）**：

| 方法 | Light | Shape | Exp. | Pose | Avg. |
|------|-------|-------|------|------|------|
| GIF | 17.04 | 2.29 | 8.16 | 8.17 | 8.91 |
| CapHuman | 15.16 | 2.65 | 6.68 | 19.03 | 10.40 |
| DiffusionRig | 6.31 | 2.11 | 5.58 | 6.26 | 5.06 |
| **ControlFace** | **3.75** | 2.56 | **5.43** | 7.67 | **4.85** |

**图像质量与身份保持**：

| 方法 | ID ↑ | FID ↓ | LPIPS ↓ |
|------|------|-------|---------|
| Arc2Face | 0.7825 | 17.82 | 0.5253 |
| DiffusionRig | 0.2042 | 23.05 | 0.3758 |
| **ControlFace** | 0.7586 | **15.50** | **0.1429** |

### 消融表

| 配置 | Re-Infer. ↓ | ID ↑ | FID ↓ |
|------|-------------|------|-------|
| FaceNet only | 7.13 | 0.8234 | 32.45 |
| +CMM | 5.78 | 0.7520 | 15.35 |
| +CMM+RCG | **4.85** | 0.7586 | 15.50 |

Face Controller（~1M 参数）比 ControlNet（~360M）和 ControlNeXt（~3M）性能更好。

### 关键发现

- 用户研究中 ControlFace 在语义一致性（0.875）和感知质量（0.861）上远超基线
- 即使在动漫风格等 out-of-domain 图像上也能成功操控
- 无需微调即超越需要微调的 DiffusionRig

## 亮点与洞察

1. **视频数据集训练策略精妙**：用配对帧规避重建训练困境，是最核心的 insight
2. **RCG 概念简洁有效**：用参考控制替代 null condition，无需额外训练，即插即用
3. **LPIPS 0.1429 远超所有基线**：说明语义细节（发型/背景）保持极好
4. **轻量级 Face Controller**：仅 ~1M 参数优于 ControlNet 的 ~360M，简洁即是美

## 局限性

1. 依赖 DECA 提取 3DMM 渲染，DECA 的精度限制了控制的上界
2. 仅在 CelebV-HQ 一个视频数据集上训练，身份多样性有限（~15K 人）
3. 目前仅支持 256×256 分辨率，高分辨率场景需要扩展

## 相关工作与启发

- **DiffusionRig**（CVPR 2023）：需要个体微调，编码能力有限，ControlFace 无需微调即超越
- **IP-Adapter / ReferenceNet 系列**：双分支结构思想类似，但 ControlFace 针对人脸任务做了精细设计
- **视频数据集训练思路**：可推广到其他需要配对训练但缺少标注的生成任务

## 评分

⭐⭐⭐⭐ — 方法设计精巧且各组件动机清晰，视频训练策略和 RCG 是亮点，但分辨率受限、3DMM 依赖是遗留问题

<!-- RELATED:START -->

## 相关论文

- [CryptoFace: End-to-End Encrypted Face Recognition](cryptoface_end-to-end_encrypted_face_recognition.md)
- [CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties](multi-sensor_object_anomaly_detection_unifying_appearance_geometry_and_internal_.md)
- [DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations](dualtalk_dual-speaker_interaction_for_3d_talking_head_conversations.md)
- [What's Making That Sound Right Now? Video-centric Audio-Visual Localization](../../ICCV2025/human_understanding/whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)

<!-- RELATED:END -->
