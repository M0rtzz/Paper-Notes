---
title: >-
  [论文解读] Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention
description: >-
  [ICCV 2025][图像生成][去偏见] 提出 Entanglement-Free Attention（EFA），一种推理时应用的去偏见方法，通过修改跨注意力机制将目标属性（如性别、种族）注入人物区域，同时保持非目标属性（如背景、物品）不变，在消除生成偏见的同时避免引入新的不公平关联。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "去偏见"
  - "属性解耦"
  - "交叉注意力"
  - "扩散模型"
  - "公平性"
---

# Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention

**会议**: ICCV 2025  
**arXiv**: [2506.13298](https://arxiv.org/abs/2506.13298)  
**代码**: 无  
**领域**: 文本到图像生成/公平性  
**关键词**: 去偏见, 属性解耦, 交叉注意力, 扩散模型, 公平性

## 一句话总结

提出 Entanglement-Free Attention（EFA），一种推理时应用的去偏见方法，通过修改跨注意力机制将目标属性（如性别、种族）注入人物区域，同时保持非目标属性（如背景、物品）不变，在消除生成偏见的同时避免引入新的不公平关联。

## 研究背景与动机

扩散模型（如 Stable Diffusion）在文本到图像生成中常表现出社会偏见——例如生成"nurse"时偏向女性、生成"CEO"时偏向白人男性。现有去偏见方法虽能调整目标属性分布，但存在关键问题：**属性纠缠（Attribute Entanglement）**。

本文通过一个生动的例子说明问题严重性：当用 prompt "face of successful man with his house" 并应用去偏见方法使种族多样化时，现有方法不仅改变了人物种族，还改变了房屋的材质和建筑风格——使黑人角色配低质量住宅，白人角色配高档住宅，**反而加剧了社会经济地位的刻板印象**。

现有方法的具体问题：

**微调方法**（UCE、Finetuning Diffusion）：在小数据集上微调导致生成质量和多样性下降，且不可避免地修改模型参数

**推理时引导方法**（Interpret Diffusion、Concept Algebra）：虽不修改模型参数，但仍然无法避免对背景等非目标属性的意外修改

本文的核心洞察：偏见消除应该**仅限于语义相关的人物区域**，非目标属性（如场景、物品、背景）不应该因去偏见而改变。

## 方法详解

### 整体框架

EFA 是一个轻量级的注意力增强模块，插入到 UNet 的特定交叉注意力层中。对于每种目标偏见 $C$（如性别），针对每个目标属性 $a_i$（如 female、male）分别训练一个 EFA 模块。推理时随机等概率采样一个属性，应用对应的 EFA 模块。

训练数据构造：用原始模型配合显式描述目标属性的 prompt 生成图像，用 Grounded SAM2 提取人物分割掩码，构成 $(\text{图像}, \text{掩码})$ 数据集。

### 关键设计

1. **注意力值预测器（Attention Predictor, AP）**：EFA 的核心组件是一个轻量级 AP 模块（3层卷积 + 2层 SiLU 激活），接收中间特征 $\mathbf{z}_t$ 并预测额外的注意力值。这些注意力值与原始交叉注意力的注意力值拼接后经 softmax，从而控制目标属性值向量 $V_{a_i}$ 的注入强度：

$$\text{EFA}_C^{a_i}(\mathbf{z}_t) = \text{softmax}\left(\left[\frac{QK^\top}{\sqrt{d}}, \text{AP}_{a_i}(\mathbf{z}_t)\right]\right) [V, V_{a_i}]$$

其中 $V_{a_i} = \pi(p_{a_i}) W_v$ 是目标属性文本的值向量。通过 softmax 的竞争机制，原始值向量和属性值向量的权重自动调节。

2. **双场景训练策略**：EFA 在两种场景下训练以学习自适应增强强度：

    - **场景1（目标属性已存在）**：输入包含目标属性 $a$ 的 prompt $p_a$ 和对应的噪声图像。此时 EFA 不应做任何修改，通过 L1 正则化将 AP 输出的注意力值推向零：
    $\mathcal{L}_{reg}^{trg} = \|W \odot A_a(\mathbf{z}_t)\|_1, \quad W = \mathbf{1}$

    - **场景2（反事实属性）**：输入包含反事实属性 $a^{cf}$ 的 prompt $p_a^{cf}$（如目标为 female 时输入 male），EFA 应在人物区域增强目标属性。使用掩码约束的重建损失：
    $\mathcal{L}_{recon} = \mathbb{E}[\|M \odot (\epsilon - \epsilon_\theta(\mathbf{x}_t, t, p_a^{cf}))\|_2^2]$

   同时在非人物区域施加正则化防止影响扩散到背景：
    $\mathcal{L}_{reg}^{cf} = \|\mathbf{1} - \bar{M}) \odot A_a(\mathbf{z}_t^{cf})\|_1$

3. **共享骨干的多属性 AP**：同一偏见类型下的所有 AP 模块共享卷积骨干，仅输出通道不同。这使得多个属性的 EFA 可以联合高效训练。

4. **推理时无需掩码**：训练时使用分割掩码作为监督信号，但推理时完全不需要掩码。AP 已经学会在语义相关区域应用属性增强，实现全自动的去偏见生成。

### 损失函数 / 训练策略

总损失为：
$$\mathcal{L}_{tot} = \mathcal{L}_{recon} + \lambda_1 \mathcal{L}_{reg}^{trg} + \lambda_2 \mathcal{L}_{reg}^{cf}$$

- EFA 应用于 UNet 上采样模块中输入分辨率为 16×16 的层（低分辨率特征捕获高层语义信息）
- 仅更新 AP 模块参数，冻结扩散模型所有原始参数
- 基于 Stable Diffusion v1.5 实验
- 支持多偏见类型的扩展（如同时处理性别×种族偏见）

## 实验关键数据

### 主实验

WinoBias 36 种职业评估（$\mathcal{T}_{basic}$ / $\mathcal{T}_{complex}$ 两种 prompt 模板）：

**性别偏见消除**：

| 方法 | DR↓ (basic) | PSNR↑ (basic) | LPIPS↓ (basic) | DINO↑ (basic) |
|------|------------|---------------|----------------|---------------|
| Original SD | 0.71 | - | - | - |
| UCE | 0.34 | 21.04 | 0.1374 | 0.757 |
| Interpret Diff. | 0.26 | 17.18 | 0.2290 | 0.616 |
| Finetuning Diff. | 0.48 | 22.62 | 0.1166 | 0.814 |
| **EFA (Ours)** | **0.06** | **32.52** | **0.0411** | **0.916** |

**种族偏见消除**：

| 方法 | DR↓ (basic) | PSNR↑ (basic) | LPIPS↓ (basic) | DINO↑ (basic) |
|------|------------|---------------|----------------|---------------|
| Original SD | 0.60 | - | - | - |
| UCE | 0.27 | 21.55 | 0.1261 | 0.787 |
| Interpret Diff. | 0.16 | 16.87 | 0.2416 | 0.584 |
| **EFA (Ours)** | **0.04** | **30.93** | **0.0353** | **0.938** |

模型保持能力（COCO-no-person 数据集）：

| 方法 | FID↓ | CLIP-T↑ |
|------|------|---------|
| Original SD | - | 26.17 |
| UCE | 11.65 | 25.17 |
| Interpret Diff. | 15.78 | 24.80 |
| Finetuning Diff. | 1.92 | 25.79 |
| **EFA (Ours)** | **0.23** | **26.03** |

### 消融实验

| 配置 | 说明 |
|------|------|
| 应用层选择 | 16×16 分辨率层最优，更高分辨率层影响细节保持 |
| 掩码使用 | 无掩码训练导致非目标属性保持下降 |
| 反事实训练 | 去掉反事实场景导致去偏见效果大幅下降 |
| 正则化权重 | $\lambda_1$, $\lambda_2$ 平衡偏见消除和属性保持 |
| 多偏见扩展 | 性别×种族(8类)同时去偏，DR从0.56降至0.03 |

### 关键发现

- EFA 的 DR 指标远优于所有基线（性别偏见：0.06 vs 次优 0.26；种族偏见：0.04 vs 次优 0.16）
- 非目标属性保持能力领先巨大：PSNR 32.52 vs 次优 22.62（+10 dB！），DINO 0.916 vs 次优 0.814
- 模型保持能力最强：FID 仅 0.23（意味着非人物图像几乎不受影响），而 UCE 为 11.65
- 推理成本极低：仅在选定层增加轻量级 AP 模块

## 亮点与洞察

- **问题定义精准**：首次明确提出"属性纠缠"问题，揭示现有去偏见方法可能引入新偏见
- **设计优雅**：通过拼接额外注意力值到 softmax，利用注意力竞争机制自然实现属性增强/抑制
- **双场景训练**：让 AP 学会"该增强时增强、不该动时不动"的自适应能力
- **实用性强**：推理无需掩码、不修改模型参数、支持多偏见类型、用户可通过调整采样概率控制属性分布
- **定量领先幅度惊人**：非目标属性保持指标碾压所有基线方法

## 局限与展望

- 当前仅聚焦于人物为中心的偏见（gender、race），对物类或场景级偏见尚未验证
- 依赖 CLIP 分类器评估属性分布，CLIP 本身可能存在分类偏差
- 二元性别设定是实验评估的简化，不反映性别多样性
- 人物分割掩码质量影响训练效果，极端遮挡场景可能出问题
- 对于非预定义的属性偏见（如社会经济地位），仍需要先验知识定义目标属性集合
- 仅在 SD v1.5 上验证，对更新架构（如 SDXL、SD3）的适用性待验证

## 相关工作与启发

- **UCE** [Gandikota et al., 2023]：通过修改交叉注意力权重去偏，但改变了模型参数
- **Interpret Diffusion** [Li & Parihar et al., 2024]：通过 h-space 操控去偏，但导致背景变化
- **Semantic Guidance** [Friedrich et al., 2023]：需要预注册概念，不够灵活
- **Grounded SAM2** [Liu et al., 2024]：提供人物分割掩码用于训练
- 启发：在注意力机制中做"加法"（拼接而非替换）比"减法"（擦除/修改权重）更安全

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 属性纠缠问题的提出极具洞察力，EFA 设计简洁有效
- **实验充分度**: ⭐⭐⭐⭐⭐ 三维度评估(偏见/属性保持/模型保持)+两种prompt+多偏见组合
- **写作质量**: ⭐⭐⭐⭐⭐ 动机图(Fig.1)一目了然，问题阐述极为清晰
- **价值**: ⭐⭐⭐⭐⭐ 实际可部署的去偏方案，解决了长期被忽视的属性纠缠问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering](larender_training-free_occlusion_control_in_image_generation_via_latent_renderin.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)
- [\[ICCV 2025\] VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)
- [\[ICML 2025\] Visual Generation Without Guidance](../../ICML2025/image_generation/visual_generation_without_guidance.md)
- [\[CVPR 2025\] Learning Flow Fields in Attention for Controllable Person Image Generation](../../CVPR2025/image_generation/learning_flow_fields_in_attention_for_controllable_person_image_generation.md)

</div>

<!-- RELATED:END -->
