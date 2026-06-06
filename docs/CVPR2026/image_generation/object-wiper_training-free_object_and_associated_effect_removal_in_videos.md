---
title: >-
  [论文解读] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos
description: >-
  [CVPR 2026][图像生成][视频物体移除] 提出 Object-WIPER，首个无训练的视频物体及其关联效应（阴影、反射、镜像等）移除框架，利用 DiT 中的文本-视觉交叉注意力和视觉自注意力定位关联效应区域，通过前景重初始化和注意力缩放实现干净移除…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "视频物体移除"
  - "关联效应"
  - "训练免费"
  - "注意力机制"
  - "扩散模型"
---

# Object-WIPER: Training-Free Object and Associated Effect Removal in Videos

**会议**: CVPR 2026  
**arXiv**: [2601.06391](https://arxiv.org/abs/2601.06391)  
**代码**: 即将发布  
**领域**: 图像生成 / 视频编辑  
**关键词**: 视频物体移除, 关联效应, 训练免费, 注意力机制, 扩散模型

## 一句话总结
提出 Object-WIPER，首个无训练的视频物体及其关联效应（阴影、反射、镜像等）移除框架，利用 DiT 中的文本-视觉交叉注意力和视觉自注意力定位关联效应区域，通过前景重初始化和注意力缩放实现干净移除，并提出 TokSim 指标和 WIPER-Bench 真实世界基准。

## 研究背景与动机

**领域现状**：视频物体移除是影视制作和隐私保护的关键技术。经典方法（PatchMatch/图割）和学习方法（Propainter）专注填充物体区域，完全忽视关联效应（阴影/反射）。近期扩散方法（VACE/Videopainter）也保留关联效应。

**现有痛点**：(a) 几乎所有现有方法保留阴影/反射导致视觉伪影；(b) ROSE 能处理关联效应但需大量合成数据训练；(c) Omnimatte-Zero 从用户 mask 扩展关联区域但依赖外部点追踪模型（TAP-Net），在快速运动/透明物体下失败，且扩展策略次优。

**核心矛盾**：物体移除不等于区域填充——必须同时移除物体的"视觉痕迹"（阴影、反射、镜像等）才算干净移除。

**本文目标**：无训练地同时移除物体及其所有关联视觉效应。

**切入角度**：利用 MMDiT 中文本-视觉共享嵌入空间直接定位关联效应，不依赖外部模型。

**核心 idea**：交叉注意力定位关联效应种子 → 自注意力精修 → 前景重初始化 + 注意力缩放 → 自适应时序 mask。

## 方法详解

### 整体框架
输入为 RGB 视频 $\mathcal{I}_k$、物体 mask $\mathbf{M}^{obj}$、描述物体和效应的文本提示 $\{P_s, P_T\}$。三步处理：(1) 关联效应定位；(2) 反转获得结构化噪声并保存背景值；(3) 前景重初始化后去噪生成干净视频。

### 关键设计

1. **关联效应定位**:

    - 功能：从视频中找出物体关联效应（阴影、反射等）的空间位置
    - 核心思路：**两步法**——Step 1: 从 $T\to I$ 交叉注意力中提取与物体/效应文本 token 高相关的视觉 token：$\bar{\mathbf{A}}^{\tilde{T}\to I} = \text{Mean}(\text{Softmax}(\frac{\mathbf{Q}_{\tilde{T}}\cdot\mathbf{K}_I^\top}{\sqrt{d}}))$，Otsu 阈值得到提议 mask $m^{PRO}$。Step 2: 用视觉自注意力 $\mathbf{A}^{I\to I}$ 计算每个 token 对 $m^{PRO}$ 的响应比，阈值化得到最终 mask $\mathbf{M}^{AE}$
    - 设计动机：(a) 仅用物体 mask 扩展（Omnimatte-Zero 做法）会遗漏弱激活区域；(b) 交叉注意力提供语义定位但不完整（内部有孔洞）；(c) 自注意力精修可以填补孔洞——如果隶属同一物体则必有高自注意力
    - 与 Omnimatte-Zero 的区别：不依赖外部点追踪模型，利用 DiT 内在注意力更鲁棒

2. **时步自适应 Masking**:

    - 功能：解决固定 mask 在噪声空间中覆盖不足的问题
    - 核心思路：在反转过程中计算物体响应分数 $RS_p(j) = \frac{\sum_{y\in\mathbf{M}^{obj}(j)}A_{p,y}^{I\to I}}{\sum_{x\in\mathcal{I}(j)}A_{p,x}^{I\to I}}$，随时步增加物体"存在感"扩散，阈值化得到自适应 mask $\hat{M}_t^{obj}$
    - 设计动机：反转到噪声分布的过程中，自注意力使物体影响持续扩散，固定 mask 无法完全覆盖

3. **注意力缩放（Attention Scaling）**:

    - 反转时：缩小背景对前景的注意力 $\tilde{\mathbf{A}}^{bg\to obj} = \text{Softmax}(\frac{\mathbf{Q}_I^{bg}\cdot(c\mathbf{K}_I^{obj})^\top}{\sqrt{d}})$，$c<1$
    - 去噪时：放大前景对背景的注意力 $\tilde{\mathbf{A}}^{obj\to bg} = \text{Softmax}(\frac{\mathbf{Q}_I^{obj}\cdot(b\mathbf{K}_I^{bg})^\top}{\sqrt{d}})$，$b>1$
    - 设计动机：反转时减少背景受前景"污染"，去噪时让前景（已重初始化）主动从背景获取语义

4. **前景重初始化**:

    - 功能：在反转结果的前景区域替换为高斯噪声
    - 核心思路：$\tilde{\mathbf{Z}}_1 = \mathbf{Z}_1\odot(1-\mathbf{M}^{obj}\cup\mathbf{M}^{AE}) + \varepsilon\odot(\mathbf{M}^{obj}\cup\mathbf{M}^{AE})$
    - 设计动机：消除物体及其效应的任何残留先验

5. **TokSim 指标**:

    - 核心思路：$\text{TokSim} = 100\cdot\frac{1}{F}\sum_z\sum_i \lambda_z^k\cdot(1-\eta_z^k)\cdot\tau_z^k$，其中 $\lambda$ 奖励时序一致、$\eta$ 惩罚物体残留、$\tau$ 奖励前景-背景融合

### 损失函数 / 训练策略
- 完全无训练，基于预训练 T2V DiT
- 推理时仅需注意力操控和值复制

## 实验关键数据

### 主实验

| 方法 | 训练 | DAVIS TokSim↑ | WIPER TokSim↑ | DAVIS BG-PSNR↑ | DAVIS Text-align↑ |
|------|------|--------------|---------------|----------------|-------------------|
| Propainter | ✓ | 28.24 | 20.99 | 34.01 | 26.18 |
| ROSE | ✓ | 29.36 | 30.02 | 26.97 | 26.13 |
| VACE | ✓ | 15.86 | 11.53 | 24.48 | 24.01 |
| Gen-Prop | ✓ | 30.52 | - | 24.27 | 25.89 |
| KV-Edit-Video | ✗ | 28.68 | 23.26 | 25.78 | 25.21 |
| Attentive-Eraser | ✗ | 30.82 | 25.28 | 28.07 | 26.31 |
| **Object-WIPER** | **✗** | **32.80** | **33.09** | 23.02 | **26.63** |

### 消融实验

| 配置 | TokSim↑ | BG-PSNR↑ | Text-align↑ |
|------|---------|----------|-------------|
| Full Object-WIPER | 32.80 | 23.02 | 26.63 |
| w/o 注意力缩放 | 32.97 | 21.92 | 26.42 |
| w/o 自适应 mask | 32.10 | 22.73 | 26.44 |
| w/o 重初始化 | 30.36 | 23.47 | 25.92 |
| w/o $\mathbf{M}^{AE}$ | 32.18 | 23.10 | 26.17 |

### 关键发现
- Object-WIPER 无训练即在 TokSim 上赶超所有训练方法（包括专门训练关联效应的 ROSE）
- TokSim 比 BG-PSNR 区分力强得多：VAE 重建（不移除物体）BG-PSNR 34.05 但 TokSim 仅 0.32
- 重初始化是最关键组件（去掉后 TokSim 降 2.44）
- 关联效应 mask $\mathbf{M}^{AE}$ 对 WIPER-Bench 至关重要——只有加上才能移除阴影/反射
- 自适应 mask 在快速运动场景（如高速行驶的车）中必不可少

## 亮点与洞察
- **MMDiT 内在注意力做关联效应定位**：完全不依赖外部模型，利用文本-视觉共享空间的语义关联精准定位。这个技巧可迁移到任何 MMDiT-based 编辑任务
- **TokSim 指标设计**精巧：同时度量移除完整性、时序一致性和背景融合，暴露了现有指标的根本缺陷
- **WIPER-Bench**是首个包含镜像、透明物体、多关联效应等真实场景的物体移除基准

## 局限与展望
- BG-PSNR 不如训练方法（因为背景也被扩散模型重新生成）
- 依赖文本描述物体和效应类型，自动化程度有限
- 视频分辨率受预训练模型限制
- 仅处理动态物体，静态物体移除未讨论

## 相关工作与启发
- **vs Omnimatte-Zero**: 不依赖 TAP-Net 点追踪，定位策略更完善
- **vs ROSE/Gen-Prop**: 训练型方法需大量合成数据，Object-WIPER 零成本
- **vs KV-Edit**: KV-Edit 针对图像设计，简单扩展到视频效果差

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在 DiT 内解决关联效应定位和移除，TokSim 指标重要
- 实验充分度: ⭐⭐⭐⭐ 两数据集+新基准+新指标+消融完整
- 写作质量: ⭐⭐⭐⭐ 问题定义和方法层层递进
- 价值: ⭐⭐⭐⭐⭐ WIPER-Bench + TokSim 对社区有持久价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)
- [\[ICML 2026\] AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression](../../ICML2026/image_generation/adaeraser_training-free_object_removal_via_adaptive_attention_suppression.md)
- [\[CVPR 2026\] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)
- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](vihoi_human-object_interaction_synthesis_with_visual_priors.md)

</div>

<!-- RELATED:END -->
