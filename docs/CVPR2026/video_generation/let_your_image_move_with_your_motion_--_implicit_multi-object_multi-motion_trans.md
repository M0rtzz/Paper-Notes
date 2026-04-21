---
title: >-
  [论文解读] Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer
description: >-
  [CVPR 2026][运动迁移] 本文提出 FlexiMMT，首个支持隐式多目标多运动迁移的 I2V 框架，通过运动解耦掩码注意力机制（MDMA）约束 motion/text token 仅影响对应目标区域、差异化掩码提取机制（DMEM）从扩散注意力中推导目标掩码并渐进传播，实现了精确的组合式多目标运动迁移。
tags:
  - CVPR 2026
  - 运动迁移
  - 多目标多运动
  - 注意力掩码
  - 视频扩散模型
  - I2V生成
---

# Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer

**会议**: CVPR 2026  
**arXiv**: [2603.01000](https://arxiv.org/abs/2603.01000)  
**代码**: [项目页](https://ethan-li123.github.io/FlexiMMT_page/)  
**领域**: 视频理解 / 扩散模型  
**关键词**: 运动迁移、多目标多运动、注意力掩码、视频扩散模型、I2V生成

## 一句话总结

本文提出 FlexiMMT，首个支持隐式多目标多运动迁移的 I2V 框架，通过运动解耦掩码注意力机制（MDMA）约束 motion/text token 仅影响对应目标区域、差异化掩码提取机制（DMEM）从扩散注意力中推导目标掩码并渐进传播，实现了精确的组合式多目标运动迁移。

## 研究背景与动机

1. **领域现状**：运动迁移是可控视频生成的重要方向，旨在从参考视频中捕获运动动态并应用到目标主体。现有方法分为显式（pose/光流/轨迹）和隐式（从参考视频编码运动嵌入）两类。隐式方法通过可训练的 motion token 从参考视频中学习运动表示。

2. **现有痛点**：几乎所有现有隐式运动迁移方法都只能处理单目标单运动场景。当场景中存在多个目标且需要不同运动模式时，现有方法无法将不同运动独立分配给不同目标。

3. **核心矛盾**：直接将多组 motion token 注入 3D 全注意力层时，所有 token 之间的交互是全局纠缠的——一个目标的 motion token 会影响其他目标的视频 token，导致运动混淆和错误传递。

4. **本文目标**：如何在 I2V 生成中实现多目标独立运动迁移，让每个目标按照其指定的参考视频运动。

5. **切入角度**：在注意力层面做运动解耦——通过目标特定的掩码约束 motion 和 text token 仅与对应目标的 video token 交互。

6. **核心 idea**：用掩码注意力实现运动解耦，让每个目标只"看到"属于自己的运动信号和文本描述。

## 方法详解

### 整体框架

FlexiMMT基于CogVideoX-5B-I2V扩散模型构建。训练阶段：给定单个参考视频，学习可训练的运动token来编码运动表征，同时在注意力层应用目标掩码约束运动-视频交互。推理阶段：对于包含多个目标的输入图像和多个参考视频，将各自预训练的运动token拼接到文本和视频token中，通过MDMA掩码确保每组运动token只影响对应目标区域，通过RMPM动态传播目标掩码到后续帧。

### 关键设计

1. **运动解耦掩码注意力 (MDMA)**:
    - 功能：在注意力层面显式解耦不同目标的运动信号
    - 核心思路：构建掩码矩阵$\mathcal{M}$，包含Motion-to-X (M2X)和Text-to-X (T2X)两大类子掩码。M2X确保运动token只与对应目标的视频token交互（$\mathcal{M}_{m \to v}$），且不同运动token之间互不干扰（$\mathcal{M}_{m \to m} = \mathbf{0}$）。T2X确保描述运动的文本token也只关注对应目标
    - 设计动机：原始MM-DiT中所有token全局交互导致运动纠缠，掩码机制以最小修改实现了运动的精确隔离

2. **差异化掩码提取机制 (DMEM)**:
    - 功能：为训练和推理阶段分别提供目标掩码
    - 核心思路：训练阶段（单目标）利用文本query $Q_y^k$与视频key $K_v$的注意力图自动提取目标区域，用均值阈值二值化。推理阶段（多目标）使用语义分割模型提取首帧掩码，再通过RMPM传播到后续帧
    - 设计动机：训练时避免外部分割模型的计算开销和训推不一致问题；推理时简单的注意力方法无法区分多个目标（特征纠缠），故需要分割+传播方案

3. **回归式掩码传播 (RMPM)**:
    - 功能：将首帧的目标掩码准确传播到视频所有帧
    - 核心思路：维护一个滑动窗口的锚点集（首帧+近邻帧），通过特征相关性矩阵$\mathcal{C}_l^k$将锚点掩码传播到当前帧。动态RMPM进一步优化：当连续步骤的掩码变化低于阈值$\alpha=5\%$时停止更新，复用稳定掩码
    - 设计动机：扩散模型的运动迁移主要在早期去噪步骤完成，后续掩码变化微小，动态终止可大幅提升推理效率

### 损失函数 / 训练策略

- 采用CogVideoX-5B-I2V的标准噪声预测损失
- 运动token训练2000步，使用AdamW优化器，学习率3e-3，batch size为1
- 视频分辨率$720 \times 480$，每段49帧
- RMPM滑动窗口大小$W=2$
- 在6块NVIDIA A800 GPU上进行实验

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(FlexiMMT) | 最佳baseline | 提升 |
|--------|------|------|----------|------|
| 200对 | Trajectory Fidelity (TF) | 0.577 | 0.488 (Go-with-Flow) | +0.089 |
| 200对 | Flow Fidelity (FF) | 0.723 | 0.648 (Go-with-Flow) | +0.075 |
| 200对 | Appearance Consistency | 0.904 | 0.939 (FlexiAct) | 略低但运动更准 |
| 200对 | 人工评估-运动保真度 | 89.475% | 6.500% (FlexiAct) | 压倒性领先 |
| 200对 | 人工评估-时间一致性 | 83.875% | 11.550% (FlexiAct) | 压倒性领先 |

### 消融实验

| 配置 | TF↑ | FF↑ | 说明 |
|------|---------|---------|------|
| 完整FlexiMMT | 0.577 | 0.723 | 基线 |
| w/o M2X掩码 | 0.381 | 0.618 | TF下降34%，M2X是运动解耦核心 |
| w/o T2X掩码 | 0.461 | 0.665 | TF下降20%，T2X也很重要 |
| w/o 训练阶段掩码 | 0.440 | 0.656 | 无法准确学习参考运动 |
| w/o 推理阶段掩码(DMEM) | 0.373 | 0.602 | 多运动完全纠缠 |
| w/o RMPM | 0.377 | 0.607 | 效果类似w/o推理掩码 |

### 关键发现

- 人工评估中FlexiMMT在运动保真度上获得89.475%的选票，远超所有baselines（第二名FlexiAct仅6.5%）
- AC和TC等CLIP-based指标对静态/弱运动视频有偏好——运动失败的方法反而可能获得高AC/TC分数
- 论文提出的Flow Fidelity (FF)指标比Trajectory Fidelity更全面地衡量运动相似性
- 动态RMPM相比全量RMPM大幅降低推理时间，且性能无损

## 亮点与洞察

- 首个解决隐式多目标多运动迁移问题的框架，填补了重要空白
- MDMA机制思路简洁有效：通过掩码矩阵在attention层面硬性切断跨目标信号，比软约束更可靠
- 训练/推理阶段使用不同的掩码提取策略是一个务实的工程选择——训练时用轻量注意力方法，推理时用分割+传播
- 运动token可以任意重组和交换，实现了真正的组合式运动迁移

## 局限与展望

- 训练阶段每个视频只含单个目标，依赖大量单目标参考视频
- 首帧分割依赖外部语义分割模型（Grounded SAM），引入额外依赖
- RMPM基于特征相关性传播，在遮挡或快速运动导致目标外观剧变时可能失败
- AC和TC指标偏低可能表明生成的运动引入了一定程度的外观漂移

## 相关工作与启发

- **vs FlexiAct**: FlexiAct通过时空注意力特征提取运动，但在多目标场景下运动信号纠缠严重；FlexiMMT通过MDMA在注意力层面显式隔离
- **vs Go-with-the-Flow**: 使用显式光流进行运动控制，需要光流估计器且对目标几何有约束；FlexiMMT是隐式方法，更灵活
- **vs MotionDirector**: T2V场景下的LoRA-based运动解耦，无法处理I2V的外观保持需求
- **启发**: 在注意力层面进行结构化约束（而非纯粹的损失函数约束）是实现解耦控制的有效范式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个隐式多目标多运动I2V框架，问题定义和解决方案均有创新
- 实验充分度: ⭐⭐⭐⭐ 200对评估、自动+人工指标、详细消融，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 公式化表述严谨，整体结构清晰
- 价值: ⭐⭐⭐⭐ 对可控视频生成领域有显著推进，开启多运动迁移新方向

<!-- RELATED:START -->

## 相关论文

- [Anti-I2V: Safeguarding your photos from malicious image-to-video generation](anti-i2v_safeguarding_your_photos_from_malicious_image-to-video_generation.md)
- [SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation](symphomotion_joint_control_of_camera_motion_and_object_dynamics_for_coherent_vid.md)
- [ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](../../CVPR2025/video_generation/conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)
- [Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)
- [MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](moviedrive_urban_scene_synthesis_with_multi-modal_multi-view_video_diffusion_tra.md)

<!-- RELATED:END -->
