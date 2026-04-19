---
title: >-
  [论文解读] SMooDi: Stylized Motion Diffusion Model
description: >-
  [ECCV 2024][图像生成] 提出SMooDi——首个将预训练文本-动作模型适配为风格化动作生成的扩散模型，通过风格适配器和双重风格引导（无分类器引导+基于分类器引导）实现内容文本与风格动作序列驱动的多样化风格动作生成。
tags:
  - ECCV 2024
  - 图像生成
---

# SMooDi: Stylized Motion Diffusion Model

**会议**: ECCV 2024  
**arXiv**: [2407.12783](https://arxiv.org/abs/2407.12783)  
**领域**: 图像生成

## 一句话总结

提出SMooDi——首个将预训练文本-动作模型适配为风格化动作生成的扩散模型，通过风格适配器和双重风格引导（无分类器引导+基于分类器引导）实现内容文本与风格动作序列驱动的多样化风格动作生成。

## 研究背景与动机

- 人体动作由内容（走路、挥手等）和风格（老年人、开心、愤怒等）两个维度组成
- **文本驱动动作生成**（MDM、MLD等）已取得显著进展，但主要关注内容，缺乏风格控制
- **动作风格迁移**方法可以将风格从一个序列转移到另一个序列，但需要已有的内容动作序列作为输入，限制了灵活性
- 两者**简单串联**（先生成+再迁移）存在三个问题：
  1. 效率低：需要逐个处理每个序列
  2. 误差累积：风格迁移模型在不完美的生成动作上性能下降
  3. 数据受限：风格迁移方法依赖特定风格数据集中有限的动作内容

## 方法详解

### 整体框架

SMooDi在预训练的MLD（Motion Latent Diffusion）模型基础上构建，包含两个核心模块：
1. **风格适配器**（Style Adaptor）：通过残差特征注入风格条件
2. **风格引导**（Style Guidance）：无分类器+基于分类器的双重引导

在去噪步骤 $t$ 中，模型接收内容文本 $\mathbf{c}$、风格动作 $\mathbf{s}$ 和噪声潜在变量 $\mathbf{z}_t$ 作为输入，预测噪声 $\epsilon_t$。

### 关键设计

**1. 风格适配器**

- 训练一个MLD中Transformer Encoder的可训练副本
- 独立的风格编码器提取风格动作序列的嵌入
- 风格适配器与MLD之间通过**零初始化线性层**连接（借鉴ControlNet设计）
- 训练过程中，适配器逐渐学习风格约束并将修正特征应用到MLD的对应层

**2. 无分类器风格引导**

将条件引导分解为内容和风格两个独立分量：

$$\epsilon_\theta(\mathbf{z}_t, t, \mathbf{c}, \mathbf{s}) = \epsilon_\theta(\mathbf{z}_t, t, \emptyset, \emptyset) + w_c(\epsilon_c - \epsilon_\emptyset) + w_s(\epsilon_{cs} - \epsilon_c)$$

其中 $w_c$ 和 $w_s$ 分别控制内容和风格引导的强度，可以灵活平衡内容保持和风格反映。

**3. 基于分类器的风格引导**

设计解析函数 $G(\mathbf{z}_t, t, \mathbf{s})$，计算生成动作与参考风格在风格嵌入空间的L1距离：

$$G(\mathbf{z}_t, t, \mathbf{s}) = |f(\hat{\mathbf{x}}_0) - f(\mathbf{s})|$$

利用其梯度引导生成动作接近目标风格，风格特征提取器通过在100STYLE数据集上训练风格分类器获取。

### 损失函数

总训练损失包含三项：

$$\mathcal{L}_{all} = \mathcal{L}_{std} + \lambda_{pr}\mathcal{L}_{pr} + \lambda_{cyc}\mathcal{L}_{cyc}$$

- $\mathcal{L}_{std}$：标准去噪损失，在100STYLE数据集上计算
- $\mathcal{L}_{pr}$：**内容先验保持损失**——从HumanML3D采样计算，防止"内容遗忘"
- $\mathcal{L}_{cyc}$：**循环先验保持损失**——交换两个数据集的内容和风格后重建原始序列，鼓励风格-内容解耦

## 实验关键数据

### 主实验

风格化文本动作生成任务的对比（HumanML3D内容 + 100STYLE风格）：

| 方法 | FID↓ | Foot Skating↓ | MM Dist↓ | R-precision↑ | Diversity→ | SRA(%)↑ |
|------|------|--------------|---------|-------------|-----------|--------|
| MLD+Motion Puzzle | 6.127 | 0.185 | 6.467 | 0.290 | 6.476 | 63.769 |
| MLD+Aberman et al. | 3.309 | 0.347 | 5.983 | 0.406 | 8.816 | 54.367 |
| ChatGPT+MLD | 0.614 | 0.131 | 4.313 | 0.605 | 8.836 | 4.819 |
| **SMooDi** | **1.609** | **0.124** | **4.477** | **0.571** | **9.235** | **72.418** |

ChatGPT+MLD的SRA仅4.82%，证明MLD即使通过文本描述风格也无法实现风格化生成。

### 消融实验

各模块对性能的贡献：

| 配置 | FID↓ | Foot Skating↓ | MM Dist↓ | R-precision↑ | Diversity→ | SRA(%)↑ |
|------|------|--------------|---------|-------------|-----------|--------|
| 完整模型 | 1.609 | 0.124 | 4.477 | 0.571 | 9.235 | 72.418 |
| w/o $L_{cyc}$ | 2.046 | 0.136 | 4.465 | 0.569 | 8.869 | 64.866 |
| w/o $L_{pr}+L_{cyc}$ | 5.996 | 0.166 | 6.098 | 0.335 | 7.456 | 81.841 |
| w/o 分类器引导 | 1.050 | 0.111 | 4.085 | 0.630 | 9.445 | 20.245 |
| w/o 适配器 | 2.984 | 0.123 | 4.526 | 0.550 | 8.372 | 69.952 |

**关键观察**：
- 移除 $L_{pr}+L_{cyc}$ 后SRA达到81.84%但FID暴涨至5.996 → 严重"内容遗忘"，动作全变成风格数据集的locomotion
- 移除分类器引导后SRA从72.4%骤降至20.2% → 分类器引导对风格反映至关重要
- 适配器和分类器引导互补：适配器确定基本风格方向，分类器引导精细调整

### 关键发现

1. **双重引导互补**：无分类器引导捕获风格相关的粗特征，分类器引导提供精确的风格控制，两者缺一不可
2. **内容保持损失关键**：没有先验保持损失会导致严重的内容遗忘，单一模型无法同时支持多样化内容和风格
3. **动作风格迁移作为下游任务**：通过DDIM-Inversion可以将内容动作序列转换为噪声潜在表示，无需额外优化即可支持风格迁移
4. 用户研究中SMooDi在真实性、风格反映、内容保持三个维度均获得更多用户偏好

## 亮点与洞察

- 首次将预训练的文本-动作扩散模型适配为风格化生成，设计思路清晰且有效
- 循环先验保持损失的设计非常巧妙，通过交换两个数据集的风格-内容来鼓励解耦
- 将风格引导分解为分类器自由和分类器引导两部分，可以灵活调节内容与风格的平衡
- 单一模型支持100种风格×多样化内容，无需为每种风格单独微调（对比之前方法需要per-style tuning）

## 局限性

- 基于分类器的风格引导依赖100STYLE数据集训练的风格分类器，当内容文本与locomotion差异较大时效果可能下降
- 100STYLE数据集仅包含locomotion相关运动，限制了可学习的风格类型
- 定量评估中SRA指标依赖预训练分类器，不一定完全反映视觉感知的风格质量
- 需要预训练的MLD模型和额外的风格数据集，训练流程相对复杂

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首个适配预训练动作扩散模型进行风格化生成，双重引导机制新颖
- **实用性**: ⭐⭐⭐⭐ — 单模型多风格、支持动作风格迁移作为下游任务
- **实验充分性**: ⭐⭐⭐⭐ — 定量+定性+用户研究+消融实验完整，消融分析深入
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰、图示直观、motivation阐述充分

<!-- RELATED:START -->

## 相关论文

- [Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [MotionChain: Conversational Motion Controllers via Multimodal Prompts](motionchain_conversational_motion_controllers_via_multimodal_prompts.md)

<!-- RELATED:END -->
