---
title: >-
  [论文解读] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability
description: >-
  [图像生成] DynamicID 通过语义激活注意力（SAA）和身份-运动重构器（IMR）两个核心组件，实现了零样本的单/多身份个性化图像生成，同时保持高保真度和灵活的面部可编辑性。
tags:
  - 图像生成
---

# DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability

## 基本信息

- **会议**: ICCV 2025
- **arXiv**: 2503.06505
- **代码**: 未公开
- **领域**: 图像生成 (Image Generation)
- **关键词**: 个性化图像生成, 多ID生成, 人脸编辑, 注意力机制, 扩散模型

## 一句话总结

DynamicID 通过语义激活注意力（SAA）和身份-运动重构器（IMR）两个核心组件，实现了零样本的单/多身份个性化图像生成，同时保持高保真度和灵活的面部可编辑性。

## 研究背景与动机

个性化人物图像生成旨在从参考图像中保持一致身份的同时融入用户指定的文本提示。现有 tuning-free 方法存在两个关键限制：

**多 ID 生成受限**：多数方法针对单身份设计，多身份场景下面临严重的身份混淆（identity blending）问题——不同参考人物的面部特征在生成过程中相互混淆

**面部可编辑性不足**：现有方法未显式解耦身份特征（面部结构、肤质）与运动特征（表情、朝向），导致无法灵活编辑面部属性

已有的缓解策略（如 FastComposer 的局部交叉注意力、InstantFamily 的 masked cross-attention）均建立在为单 ID 设计的框架之上，导致核心功能受损。

## 方法详解

### 整体框架

DynamicID 包含三个核心组件：人脸编码器（提取面部特征）、IMR（特征空间中解耦和重配面部运动与身份）、SAA（将处理后的特征注入 T2I 模型）。采用任务解耦训练范式：

- **Anchoring 阶段**：联合训练 SAA + 人脸编码器（仅需单 ID 数据集）
- **Reconfiguration 阶段**：冻结上述组件，训练 IMR（使用 VariFace-10k 数据集）

### 语义激活注意力 (SAA)

标准交叉注意力中 softmax 的归一化约束迫使每个 query 在所有 key 上分配固定的注意力总量，即使 query 与 key 之间不存在语义关联，这导致：(1) 扰动原始模型行为；(2) 多 ID 场景下的身份混淆。

SAA 引入 query 级激活门控机制：

$$z_{\text{new}} = z + \text{Expand}(w) \odot \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

$$w = \text{Norm}(QK^\top J)$$

其中 $J \in \mathbb{R}^{k \times 1}$ 为全 1 列向量，$\text{Norm}(\cdot)$ 进行 min-max 归一化到 [0,1]。激活权重 $w$ 反映每个 query 与参考面部信息的语义相关程度：

- 面部区域 query → 强激活（接近 1）
- 背景区域 query → 抑制（接近 0）
- 身体区域 query → 适度激活

这带来三个零样本泛化能力：
- **上下文解耦**：非面部区域不受面部信息干扰
- **布局控制**：通过 mask 调节激活权重实现空间布局控制
- **多 ID 个性化**：注入某个角色的面部信息时，抑制其他角色区域的 query

### 身份-运动重构器 (IMR)

IMR 由 DisentangleNet $\phi_1$ 和 EntangleNet $\phi_2$ 组成，在特征空间中解耦和重配身份与运动：

$$\xi_{\text{pred}} = \phi_2(\phi_1(\xi_{\text{src}}, \psi_{\text{src}}), \psi_{\text{tgt}})$$

其中运动特征 $\psi$ 来自面部 prompt（文本编码）和面部关键点（关键点编码器），通过 MLP 融合。

### 损失函数

**Anchoring 阶段**使用标准扩散噪声预测损失：

$$\mathcal{L}_{\text{noise}} = E_{z,t,\xi,\tau,\epsilon} \|\epsilon - \epsilon_\theta(z_t, t, \xi, \tau)\|_2^2$$

**Reconfiguration 阶段**使用双目标损失：

$$\mathcal{L}_{\text{edit}} = \|\xi_{\text{pred}} - \xi_{\text{tgt}}\|_2^2 + \lambda \|\epsilon'_\theta(\xi_{\text{pred}}) - \epsilon'_\theta(\xi_{\text{tgt}})\|_2^2$$

第一项为直接特征匹配，第二项为潜在扩散一致性（确保预测特征在生成模型隐空间中具有相同语义），$\lambda=1$。

### VariFace-10k 数据集

为 IMR 训练构建：包含 10k 个独立身份，每个身份 35 张不同面部图像（涵盖不同表情、姿态、光照）。

## 实验关键数据

### 主实验：单 ID 个性化量化对比

| Method | Arch. | CLIP-T ↑ | FaceSim ↑ | Expr ↑ | Pose ↑ |
|---|---|---|---|---|---|
| PuLID-FLUX | FLUX | 0.237 | 0.667 | 0.181 | 0.273 |
| PhotoMaker-v2 | SDXL | 0.238 | 0.592 | 0.243 | 0.869 |
| InstantID | SDXL | 0.233 | 0.723 | 0.151 | 0.264 |
| IPA-FaceID-Plus | SD1.5 | 0.236 | 0.712 | 0.156 | 0.266 |
| MasterWeaver | SD1.5 | 0.237 | 0.651 | 0.189 | 0.278 |
| **DynamicID (Ours)** | SD1.5 | **0.239** | 0.671 | **0.456** | **0.878** |

DynamicID 在 Expr (+0.213) 和 Pose (+0.009 vs PhotoMaker) 上大幅领先，证明其面部可编辑性远超现有方法。FaceSim 略低于 InstantID/IPA-FaceID，但后两者高 FaceSim 来自直接面部复制（低 Expr/Pose 证实）。

### 消融实验：SAA 与 IMR 的贡献

| Method | CLIP-T ↑ | FaceSim ↑ | Expr ↑ | Pose ↑ |
|---|---|---|---|---|
| Ours w/o SAA | 0.224 | 0.682 | 0.422 | 0.862 |
| Ours w/o IMR | 0.228 | 0.712 | 0.161 | 0.253 |
| **Ours (Full)** | **0.239** | 0.671 | **0.456** | **0.878** |

- 移除 SAA → CLIP-T 大幅下降（0.224 vs 0.239），说明 SAA 保护了模型原始文本编辑能力
- 移除 IMR → Expr/Pose 急剧退化（0.161/0.253 vs 0.456/0.878），FaceSim 反而升高（直接复制参考人脸）

### 多 ID 个性化对比

| Method | CLIP-T ↑ | FaceSim ↑ | Expr ↑ | Pose ↑ |
|---|---|---|---|---|
| FastComposer | 0.233 | 0.594 | 0.144 | 0.256 |
| UniPortrait | 0.235 | 0.718 | 0.149 | 0.268 |
| StoryMaker | 0.219 | 0.678 | 0.147 | 0.296 |
| **DynamicID** | **0.237** | 0.664 | **0.431** | **0.867** |

多 ID 场景下 DynamicID 仍保持显著优势，Expr 和 Pose 远超所有基线。

## 亮点与洞察

1. **SAA 的本质洞察**：标准交叉注意力的 softmax 归一化是导致模型行为被扰动和身份混淆的根源，query 级激活门控是一个优雅的解决方案
2. **零样本多 ID 泛化**：无需多 ID 训练数据即可实现多身份生成，仅通过 SAA 的激活权重操控
3. **任务解耦训练**：将联合训练拆为两阶段，降低数据需求（Anchoring 仅需单 ID 数据，IMR 仅需同一人的多张图片）
4. **特征空间操作**：IMR 在 latent 空间而非像素空间进行身份-运动解耦，计算高效且泛化性好

## 局限性

- 基于 SD1.5 构建，生成质量受限于基模型能力，未适配 SDXL/FLUX 等更强模型
- VariFace-10k 数据集的多样性可能不足以覆盖所有面部变化
- 三个及以上 ID 场景的布局控制需要手动指定 bounding box
- 推理采用 50 步 DDIM，效率有待优化

## 相关工作与启发

- **InstantID** 使用 ControlNet 实现高保真但缺乏表情编辑能力
- **PhotoMaker** 在文本嵌入空间融合面部特征，保持了部分编辑能力但牺牲了保真度
- **IP-Adapter** 的解耦交叉注意力是注意力注入的先驱，但粒度粗糙
- SAA 的激活门控思想可推广到其他条件注入场景（如风格迁移、姿态控制等）

## 评分

⭐⭐⭐⭐ — 方法设计精巧，尤其是 SAA 的 query 级激活门控机制优雅地解决了多 ID 和模型行为保持问题。面部可编辑性指标远超 SOTA。但基于 SD1.5 限制了上限。

<!-- RELATED:START -->

## 相关论文

- [MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation](mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)
- [DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](../../CVPR2025/image_generation/dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [Multi-party Collaborative Attention Control for Image Customization](../../CVPR2025/image_generation/multi-party_collaborative_attention_control_for_image_customization.md)
- [MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)
- [Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)

<!-- RELATED:END -->
