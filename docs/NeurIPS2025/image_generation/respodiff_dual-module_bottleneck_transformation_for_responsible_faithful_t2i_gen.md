---
title: >-
  [论文解读] RespoDiff: Dual-Module Bottleneck Transformation for Responsible & Faithful T2I Generation
description: >-
  [NeurIPS 2025][图像生成][负责任生成] 提出RespoDiff框架，在扩散模型UNet的瓶颈层引入双模块可学习变换——负责任概念对齐模块(RAM)和语义对齐模块(SAM)，通过分数匹配目标实现公平和安全的文本到图像生成，同时保持图像质量和语义忠实度。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "负责任生成"
  - "公平性"
  - "安全性"
  - "瓶颈变换"
  - "分数匹配"
---

# RespoDiff: Dual-Module Bottleneck Transformation for Responsible & Faithful T2I Generation

**会议**: NeurIPS 2025  
**arXiv**: [2509.15257](https://arxiv.org/abs/2509.15257)  
**代码**: [项目页面](https://vssilpa.github.io/respodiff_project_page)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 负责任生成, 公平性, 安全性, 瓶颈变换, 分数匹配

## 一句话总结

提出RespoDiff框架，在扩散模型UNet的瓶颈层引入双模块可学习变换——负责任概念对齐模块(RAM)和语义对齐模块(SAM)，通过分数匹配目标实现公平和安全的文本到图像生成，同时保持图像质量和语义忠实度。

## 研究背景与动机

Stable Diffusion、SDXL、FLUX等T2I模型虽然生成质量优秀，但存在严重的社会偏见问题：
- **性别偏见**：输入"a doctor"主要生成男性形象
- **种族偏见**：生成结果偏向特定肤色
- **安全风险**：可能生成暴力、裸露等不当内容

**现有方法的痛点**：

**提示修改方法**（去除有害词汇、提示调优）：能力有限，无法精确控制

**模型微调方法**（概念擦除、权重微调）：可能损害模型原始性能，需要针对每个提示重新训练

**分类器引导方法**：不需额外训练但控制力不够精细

**潜向量注入方法**（SDisc等）：在瓶颈空间操作中缺乏对中性去噪潜变量的显式引用，导致控制精度不够

**核心矛盾**：现有方法在提升公平性/安全性时往往牺牲语义忠实度和图像质量。如何同时实现负责任生成且保持生成质量是核心挑战。

**本文切入角度**：在UNet瓶颈层（被证实是语义潜空间）引入**双路变换**：一路负责将生成引导向目标概念（如"女性"），另一路负责保持与原始扩散轨迹的一致性，两者互相制衡。

## 方法详解

### 整体框架

UNet被分解为编码器 $e: \mathcal{Z} \times \mathcal{Y} \to \mathcal{H}$ 和解码器 $g: \mathcal{H} \times \mathcal{Y} \to \mathcal{Z}$，在瓶颈表示 $\boldsymbol{h}_{neu} \in \mathcal{H}$ 上施加双模块变换：

$$\hat{f}(y_{neu}) = g(\mathcal{T}_\theta^{resp,s}(\boldsymbol{h}_{neu}) + \mathcal{T}_\theta^{sem,s}(\boldsymbol{h}_{neu}))$$

### 关键设计

1. **负责任概念对齐模块 (RAM, $\mathcal{T}_\theta^{resp,s}$)**：

    - **目标**：修改中性提示(如"a person")的瓶颈表示，使其扩散轨迹对齐到目标概念(如"a woman")
    - **分数匹配损失**：在随机选取的时间步 $t$，先通过带RAM的扩散模型逆扩散得到中性去噪潜变量 $\boldsymbol{z}_{t,neu}$，然后：
    $\mathcal{L}_{resp} = \mathbb{E}_{\boldsymbol{z}_{t,neu}} \left[\|\epsilon_{f_{resp}}(\boldsymbol{z}_{t,neu}, y_{neu}) - \epsilon_f(\boldsymbol{z}_{t,neu}, y_{tar}^s)\|_2^2\right]$
    - **关键创新**：以中性去噪潜变量作为稳定锚点，通过对比中性和目标概念的UNet预测来提取精确的方向性指导
    - 此阶段只更新 $\mathcal{T}_\theta^{resp,s}$，不涉及完整的双模块变换

2. **语义对齐模块 (SAM, $\mathcal{T}_\theta^{sem,s}$)**：

    - **目标**：防止RAM的变换造成过度偏转，保持与原始扩散模型路径的语义一致性
    - **分数匹配损失**：
    $\mathcal{L}_{sem} = \mathbb{E}_{\boldsymbol{z}_{t,neu}} \left[\|\epsilon_{\hat{f}}(\boldsymbol{z}_{t,neu}, y_{neu}) - \epsilon_f(\boldsymbol{z}_{t,neu}, y_{neu})\|_2^2\right]$
    - 此阶段使用完整的双模块变换 $\hat{f}$，但只更新 $\mathcal{T}_\theta^{sem,s}$
    - 损失以 $\lambda$ 加权（默认 $\lambda=0.5$）

3. **交替训练策略**：

    - 两步交替：先更新RAM（只用 $\mathcal{L}_{resp}$），再更新SAM（只用 $\lambda \mathcal{L}_{sem}$）
    - 不通过逆扩散过程反向传播（减少计算开销）
    - 随着训练迭代，中性去噪潜变量逐步同时对齐目标概念和原始扩散过程

4. **推理时的灵活应用**：

    - **公平性**：为每个敏感概念 $s \in \mathcal{S}_c$ 学习变换，推理时随机选择以实现均匀分布
    - **安全性**：将负面概念（暴力、裸露）作为负提示学习"反暴力""反色情"变换，推理时聚合所有安全变换

### 损失函数 / 训练策略

- 变换实现为常量函数线性加到瓶颈激活上
- 公平性训练：5000迭代，batch=1
- 安全性训练：1500迭代，batch=1
- 使用中性提示 "a person"（公平）或 "a scene"（安全）训练，**无需特定职业或场景数据**
- 整个训练过程UNet权重冻结，只更新轻量变换模块

## 实验关键数据

### 主实验 — 性别公平性 (SD v1.4, Winobias 36职业)

| 方法 | DevRat↓ | WinoAlign↑ | FID(30K)↓ | CLIP(30K)↑ |
|------|---------|-----------|-----------|-----------|
| SD (原版) | 0.68 | 27.51 | 14.09 | 31.33 |
| SDisc | 0.17 | 26.61 | 23.59 | 29.94 |
| FDF | 0.40 | 23.90 | 15.22 | 30.63 |
| BAct | 0.57 | 27.67 | 17.07 | 30.54 |
| **RespoDiff** | **0.14** | **27.30** | **14.91** | **30.67** |

### 种族公平性 (SD v1.4)

| 方法 | DevRat↓ | WinoAlign↑ | FID(30K)↓ | CLIP(30K)↑ |
|------|---------|-----------|-----------|-----------|
| SD (原版) | 0.56 | 27.51 | 14.09 | 31.33 |
| SDisc | 0.23 | 26.80 | 17.47 | 30.27 |
| **RespoDiff** | **0.16** | **27.53** | **12.82** | **31.02** |

### 安全生成 (I2P基准, SD v1.4)

| 方法 | I2P不当↓ | FID(30K)↓ | CLIP(30K)↑ |
|------|---------|-----------|-----------|
| SD | 0.27 | 14.09 | 31.33 |
| SLD | 0.20 | 18.76 | 29.75 |
| ESD | 0.32 | 13.68 | 30.43 |
| **RespoDiff** | **0.16** | **17.89** | **31.10** |

### 消融实验

| 配置 | DevRat↓ | WinoAlign↑ | FID↓ | CLIP↑ |
|------|---------|-----------|------|-------|
| 仅RAM | 0.12 | 26.12 | 15.63 | 29.93 |
| RAM+SAM (完整) | 0.14 | 27.30 | 14.91 | 30.67 |
| 共享模块 | 0.16 | 26.12 | 15.63 | 29.93 |
| $\lambda=0$ (无SAM) | 0.12 | 26.12 | 15.63 | 29.93 |
| $\lambda=0.5$ (默认) | 0.14 | 27.30 | 14.91 | 30.67 |
| $\lambda=4$ | 0.29 | 27.53 | 14.17 | 31.24 |

### 关键发现

- RespoDiff在不当内容过滤上超越SLD约20%，同时在图文对齐上也更优
- 仅用"a person"训练即可泛化到36种具体职业，无需职业特定数据
- 在SDXL上的迁移：性别DevRat从0.72降至0.26，种族从0.57降至0.23
- 分离双模块优于共享模块：各自专注于自己的优化目标更高效
- $\lambda=0.5$ 是拐点：更小过度偏向目标概念，更大过度保守

## 亮点与洞察

- **双模块解耦设计**：RAM和SAM各司其职，避免了公平性和质量之间的零和博弈
- **中性去噪潜变量作为锚点**：比间接从目标图像重建更精确的方向性控制
- **无需职业/场景特定数据**：泛化能力强，只从通用中性提示学习即可覆盖多样场景
- **可扩展到SDXL等大模型**：轻量模块设计使得适配大模型成本很低
- **安全性与公平性可组合**：模块化设计允许独立学习、推理时灵活组合

## 局限与展望

- 公平性和安全性概念需要预先定义——无法自动发现新兴的偏见类别
- 依赖预定义的中性提示——虽然实验验证了替代提示的鲁棒性，但自动选择仍是开放问题
- 变换以常量函数实现——更复杂的结构可能带来更细致的控制
- 交叉偏见（如性别×种族）需要组合现有模块，但未系统评估

## 相关工作与启发

- 与SDisc同在瓶颈空间操作，但引入双模块+显式轨迹对齐实现更好平衡
- 分数匹配目标将轨迹对齐问题巧妙转化为噪声预测匹配，理论优美
- 模块化设计思想可扩展到其他概念控制场景（不限于公平/安全）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 双模块分数匹配框架是对瓶颈空间操作方法的有力改进
- **实验充分度**: ⭐⭐⭐⭐⭐ — Winobias、I2P基准全面覆盖，包含SDXL迁移和详尽消融
- **写作质量**: ⭐⭐⭐⭐ — 框架描述清晰，但符号较多，初读需一定耐心
- **价值**: ⭐⭐⭐⭐ — 在负责任AI生成方向贡献显著，实际部署价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](../../CVPR2025/image_generation/dual_diffusion_for_unified_image_generation_and_understanding.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)
- [\[NeurIPS 2025\] Breaking AR's Sampling Bottleneck: Provable Acceleration via Diffusion Language Models](breaking_ars_sampling_bottleneck_provable_acceleration_via_d.md)
- [\[ICML 2026\] MIRO: 多奖励条件预训练同时提升 T2I 质量与效率](../../ICML2026/image_generation/miro_multi-reward_conditioned_pretraining_improves_t2i_quality_and_efficiency.md)
- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](../../CVPR2025/image_generation/dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)

</div>

<!-- RELATED:END -->
