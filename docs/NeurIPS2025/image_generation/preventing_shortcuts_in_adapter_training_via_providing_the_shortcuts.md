---
title: >-
  [论文解读] Preventing Shortcuts in Adapter Training via Providing the Shortcuts
description: >-
  [NeurIPS 2025][图像生成][Adapter训练] 提出Shortcut-Rerouted Adapter Training，通过在adapter训练过程中主动提供confounding因素的专用通路（如LoRA吸收分布偏移、ControlNet吸收姿态/表情），使adapter只学习目标属性（如身份），推理时移除辅助模块即可获得去纠缠的适配器。
tags:
  - NeurIPS 2025
  - 图像生成
  - Adapter训练
  - 快捷学习
  - 去纠缠
  - 个性化生成
  - 身份保持
---

# Preventing Shortcuts in Adapter Training via Providing the Shortcuts

**会议**: NeurIPS 2025  
**arXiv**: [2510.20887](https://arxiv.org/abs/2510.20887)  
**代码**: [项目主页](https://snap-research.github.io/shortcut-rerouting/)  
**领域**: 扩散模型 / 图像生成  
**关键词**: Adapter训练, 快捷学习, 去纠缠, 个性化生成, 身份保持

## 一句话总结

提出Shortcut-Rerouted Adapter Training，通过在adapter训练过程中主动提供confounding因素的专用通路（如LoRA吸收分布偏移、ControlNet吸收姿态/表情），使adapter只学习目标属性（如身份），推理时移除辅助模块即可获得去纠缠的适配器。

## 研究背景与动机

Adapter（如LoRA、IP-Adapter）已成为扩展大规模T2I模型能力的核心机制。但adapter训练面临一个根本性挑战：

**快捷学习问题**：Adapter通常通过单图重建目标训练。然而一张图片编码了身份、姿态、表情、光照、背景等所有属性。重建损失不区分目标属性和干扰属性，会驱使adapter编码所有因素——即学到"快捷方式"。

**具体表现**：一个本该只注入人物身份的adapter，会同时复制参考图的表情、头部姿态，导致生成结果忽视文本提示中指定的表情和姿态。此外，finetune数据集与基础模型之间的分布差异也会被adapter吸收，产生背景退化、人体解剖失真等伪影。

**现有方法的不足**：InfU、PuLID等方法各有优势但仍然存在身份保真度下降或表情/姿态泄露问题。背景遮挡等简单手段只能处理部分confounding因素。

核心洞察异常优雅：**要阻止adapter学习不需要的快捷方式，最有效的方法是在训练时主动提供这些快捷方式**。

## 方法详解

### 整体框架

将adapter训练形式化为概率框架。观测图像 $X \sim p(X|T,C)$，其中 $T$ 是目标因素（如身份），$C$ 是confounding因素（如姿态、分布偏移）。标准训练最小化 $\mathbb{E}[\mathcal{L}(G(\mathcal{A}(X)), X)]$，隐式鼓励 $\mathcal{A}(X)$ 同时编码 $T$ 和 $C$。

Shortcut-Rerouted训练将生成过程修改为：

$$\hat{X} = G(\mathcal{A}(X), \mathcal{S}_C(C))$$

其中 $\mathcal{S}_C$ 是辅助模块，直接向生成器提供confounding因素 $C$。训练目标变为：

$$\mathbb{E}[\mathcal{L}(G(\mathcal{A}(X), \mathcal{S}_C(C)), X)]$$

由于 $C$ 已被 $\mathcal{S}_C$ "解释掉"，$\mathcal{A}(X)$ 不再有动机编码它们。推理时移除 $\mathcal{S}_C$，恢复干净的adapter。

### 关键设计

1. **SR-LoRA：吸收分布偏移**：Finetune数据集（如工作室人像）与基础模型（如Flux）的训练分布存在显著差异。先在finetune数据集上预训练一个LoRA模块，使其吸收数据集特有的风格、光照和低层特征。训练adapter时冻结该LoRA，使身份编码器 $\mathcal{A}$ 成为唯一活跃模块，专注于学习身份表示。推理时移除LoRA，使adapter能跨域泛化。

2. **SR-CN：吸收姿态和表情泄露**：使用预训练的冻结ControlNet模块 $\mathcal{S}_{CN}$ 处理训练图像的姿态和表情映射（通过姿态估计和关键点检测提取）。训练目标为：

$$\mathcal{L}(G(\mathcal{A}(T), \mathcal{S}_{CN}(C_{CN})), X)$$

ControlNet承担了姿态/表情重建的责任，adapter只需聚焦身份注入。这使得adapter在各种姿态和表情下都具有鲁棒性。

3. **组合使用**：SR-LoRA和SR-CN可以组合使用，甚至可以加入背景adapter（SR-BG）来防止光照泄露。SR-LoRA+CN+BG能同时对齐姿态和背景，真正只注入目标身份。

### 损失函数 / 训练策略

基于FLUX.1 Dev模型（DiT骨干 + Conditional Flow Matching目标）。使用openai/clip-vit-large-patch14进行身份编码。训练配置：8×A100 GPU，AdamW优化器（lr=5e-5），全局batch size 32，训练250K迭代。推理标准化配置：IP scale 1.0，CFG 3.5，28步，1024×1024分辨率。所有新增投影和融合层以零权重初始化，保持与预训练兼容。

## 实验关键数据

### 主实验——"Face" Adapters

| 方法 | LLM Id.↑ | FaceNet Id.↑ | LLM Expr.↑ | EMOCA Expr.↑ | Head Pose↓ | Prior(LPIPS)↓ |
|------|----------|-------------|-----------|-------------|-----------|-------------|
| InfU | 3.382 | 0.740 | 3.766 | 0.542 | 17.714 | 0.449 |
| PuLID | 4.283 | 0.774 | 3.590 | 0.489 | 17.535 | 0.458 |
| IPA (baseline) | 4.793 | 0.715 | 3.071 | 0.347 | 16.120 | 0.480 |
| **SR-LoRA IPA** | 4.719 | 0.671 | 3.429 | 0.458 | 13.270 | 0.433 |
| **SR-CN IPA** | **4.794** | 0.712 | **3.693** | **0.580** | **12.676** | **0.394** |

### 消融实验——"Body" Adapters

| 方法 | LLM Id.↑ | FaceNet Id.↑ | LLM Expr.↑ | Head Pose↓ | Body Pose↓ | Prior(LPIPS)↓ |
|------|----------|-------------|-----------|-----------|-----------|-------------|
| InstantX | 2.993 | 0.353 | 3.474 | 25.97 | 186.75 | 0.508 |
| IPA (baseline) | 4.599 | 0.573 | 3.300 | 20.70 | 167.40 | 0.457 |
| **SR-CN IPA** | **4.651** | **0.586** | **3.526** | **18.05** | **137.69** | **0.413** |

### 关键发现

- **Head Pose控制显著改善**：SR-CN将头部姿态误差从16.12°降至12.68°（Face）、从20.70°降至18.05°（Body），说明ControlNet有效吸收了姿态快捷方式
- **先验保持大幅提升**：Prior LPIPS从0.480降至0.394，说明SR模块有效缓解了adapter对模型先验的破坏
- **表情可控性恢复**：LLM Expr.从3.071提升到3.693，EMOCA从0.347提升到0.580，adapter不再复制参考图表情，能响应文本指令
- **身份保真度基本维持**：FaceNet Id.仅有轻微下降（0.715→0.712），说明去纠缠没有牺牲核心身份信息
- **组合SR模块效果互补**：SR-LoRA解决质量退化，SR-CN保持姿态先验，SR-BG抑制光照泄露

## 亮点与洞察

- **"以其人之道还治其人之身"的设计哲学**：不与快捷学习对抗，而是主动提供快捷通路让adapter没有动机学习干扰因素。这种反直觉但优雅的方案揭示了一个通用原则
- **模块化和可组合性**：SR模块可灵活组合（LoRA、ControlNet、背景adapter），每个模块吸收特定confounding因素，实现职责分解
- **推理零开销**：辅助模块仅在训练时使用，推理时完全移除，不增加任何计算成本
- **跨设置泛化**：从Face到Body设置均有效，说明方法的通用性

## 局限与展望

- 当前仅在IP-Adapter上验证，应用于更强baseline（如InfU框架）可能获得更好整体性能
- 仅验证了编码器类adapter，LoRA作为主adapter时的表现待探索（如风格LoRA去除布局偏移）
- 个性化增强带来的deepfake风险需配合水印和使用限制
- SR-CN依赖预训练ControlNet的质量和覆盖能力
- 未深入分析在极端姿态差异或罕见表情下的表现

## 相关工作与启发

- Shortcut Rerouting的核心思想——通过显式建模干扰因素来阻止目标模型学习它们——可推广到风格迁移、视频编辑等多种生成任务
- 与因果推理中"调整门"（adjustment for confounders）的思想异曲同工，为生成模型的纠缠问题提供了新的方法论视角
- SR-LoRA处理分布偏移的策略对所有adapter微调场景（尤其是域自适应）都有参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 核心idea简洁而深刻，"提供快捷方式以消除快捷方式"是一个具有启发性的通用原则
- **实验充分度**: ⭐⭐⭐⭐ Face和Body两个设置均有对比，指标全面（身份、表情、姿态、先验），但缺少更多baseline组合消融
- **写作质量**: ⭐⭐⭐⭐⭐ 问题形式化清晰，从数学框架到实例化自然流畅，叙事引人入胜
- **价值**: ⭐⭐⭐⭐ 方法简单易实现，对个性化生成社区有直接实用价值，通用原则可启发更广泛的研究方向

<!-- RELATED:START -->

## 相关论文

- [UNIC-Adapter: Unified Image-Instruction Adapter with Multi-modal Transformer for Image Generation](../../CVPR2025/image_generation/unic-adapter_unified_image-instruction_adapter_with_multi-modal_transformer_for_.md)
- [Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](../../ICCV2025/image_generation/trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)
- [Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter](../../ICLR2026/image_generation/mod-adapter_tuning-free_and_versatile_multi-concept_personalization_via_modulati.md)
- [Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](../../ICCV2025/image_generation/meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)
- [Training-Free Constrained Generation with Stable Diffusion Models](training-free_constrained_generation_with_stable_diffusion_models.md)

<!-- RELATED:END -->
