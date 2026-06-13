---
title: >-
  [论文解读] SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction
description: >-
  [CVPR 2025][图像生成][多模态视频预测] 提出 SyncVP 多模态视频预测框架，使用双分支扩散模型通过高效时空跨模态注意力同步生成 RGB+深度未来帧，创新地使用共享噪声和跨模态引导训练策略，在 Cityscapes 上达到 SOTA 且支持部分模态输入。 1. 领域现状：视频预测是自动驾驶、天气预报等决策系统…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "多模态视频预测"
  - "联合扩散"
  - "跨模态注意力"
  - "噪声共享"
  - "深度预测"
---

# SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction

**会议**: CVPR 2025  
**arXiv**: [2503.18933](https://arxiv.org/abs/2503.18933)  
**代码**: [https://SyncVp.github.io/](https://SyncVp.github.io/)  
**领域**: 图像生成  
**关键词**: 多模态视频预测, 联合扩散, 跨模态注意力, 噪声共享, 深度预测

## 一句话总结
提出 SyncVP 多模态视频预测框架，使用双分支扩散模型通过高效时空跨模态注意力同步生成 RGB+深度未来帧，创新地使用共享噪声和跨模态引导训练策略，在 Cityscapes 上达到 SOTA 且支持部分模态输入。

## 研究背景与动机
1. **领域现状**：视频预测是自动驾驶、天气预报等决策系统的关键任务。现有方法主要预测单模态（RGB）未来帧，扩散模型已成为该领域的主导范式。
2. **现有痛点**：仅预测 RGB 帧无法完全捕获真实世界动态的复杂性。多模态信息（深度、语义）可以提供互补线索，但现有多模态生成工作（LDM3D、MM-Diffusion）未针对视频预测设计，且简单拼接模态效果差。
3. **核心矛盾**：RGB 和深度等模态特性差异巨大（RGB 含丰富的外观细节，深度含几何结构），单一网络难以同时学习两者的复杂度。
4. **本文目标**：构建可扩展的多模态视频预测框架，支持双模态和部分模态输入。
5. **切入角度**：用两个预训练的模态特定扩散模型作为初始化，通过轻量跨模态注意力建立模态间信息交换。
6. **核心 idea**：双分支扩散 + 时空分离跨模态注意力 + 共享噪声 + 跨模态引导训练。

## 方法详解

### 整体框架
SyncVP 包含两个分支（RGB 和深度），每个分支是基于 PVDM 架构的小型 UNet。两分支通过**时空跨模态注意力 (STCA)** 模块在最深层连接。训练分两步：先独立预训练各模态扩散模型，再用共享噪声联合微调。推理时以 8 帧为单位自回归生成。

### 关键设计

1. **时空分离跨模态注意力 (STCA)**

    - 功能：高效地在 RGB 和深度特征之间交换时空信息
    - 核心思路：PVDM 的 latent 分为空间向量 $z^s$、高度-时间向量 $z^h$、宽度-时间向量 $z^w$。STCA 不在整个 latent 上做完整 cross-attention（计算开销大），而是分别对三个向量对做跨模态注意力，且共享一个注意力矩阵 $A = Q_R Q_D^\top / \sqrt{d_k}$：RGB 用 $\text{Softmax}(A) \cdot V_D$ 获取深度信息，深度用 $\text{Softmax}(A^\top) \cdot V_R$ 获取 RGB 信息。在 64×64 分辨率下仅需完整 CA 的 37% 计算量。
    - 设计动机：全尺寸 cross-attention 在视频序列上计算开销过大；时空分离利用了 PVDM latent 的结构特性

2. **共享噪声 (Shared Noise)**

    - 功能：确保多模态扩散过程同步，加速训练收敛
    - 核心思路：在前向扩散过程中，两个模态使用完全相同的噪声样本 $\epsilon$。即对 RGB 和深度的 latent 施加相同的 $\epsilon \sim \mathcal{N}(0, I)$，使得两个去噪网络学习相同的"噪声→数据"逆变换。两个分支的损失函数中目标噪声相同：$\mathcal{L}_M = \|\epsilon - \epsilon_{\theta_R}(\cdot)\|_2^2 + \|\epsilon - \epsilon_{\theta_D}(\cdot)\|_2^2$。
    - 设计动机：两个模态描述的是同一物理场景，共享噪声强制学习一致的逆过程，实验中显著改善了收敛速度和条件生成质量

3. **跨模态引导训练 (Cross-Modality Guidance)**

    - 功能：使模型支持只有部分模态输入时也能预测完整多模态输出
    - 核心思路：类比 classifier-free guidance，训练时以 50% 概率输入双模态条件，各 25% 概率只输入 RGB 或只输入深度（另一个用零替代）。同时学习三个分布：$p(r_x, d_x | r_c, d_c)$、$p(r_x, d_x | 0, d_c)$、$p(r_x, d_x | r_c, 0)$。
    - 设计动机：现实场景中可能只有部分传感器可用（如自动驾驶中深度传感器故障），模型需要鲁棒处理部分输入

### 损失函数 / 训练策略
- 各模态独立预训练：标准 DDPM 损失
- 联合微调：$\mathcal{L}_M = \mathbb{E}[\|\epsilon - \epsilon_{\theta_R}(\cdot)\|_2^2 + \|\epsilon - \epsilon_{\theta_D}(\cdot)\|_2^2]$
- DDPM 1000 步训练，DDIM 100 步推理
- 每个分支仅 58M 参数（PVDM-L 的 11%）

## 实验关键数据

### 主实验（Cityscapes, 2→28）

| 方法 | FVD↓ | SSIM↑ | LPIPS↓ |
|------|------|-------|--------|
| ExtDM-K4 | 121.3 | 0.745 | 108 |
| STDiff | 107.31 | 0.658 | 136.26 |
| SyncVP (w/o depth) | 97.31 | 0.652 | 161.1 |
| **SyncVP** | **84** | 0.649 | 159.7 |

### 消融实验

| 配置 | FVD↓ | 说明 |
|------|------|------|
| Full SyncVP | 84 | 完整模型 |
| 简单拼接模态 | ~130 | 朴素方案失效 |
| w/o 共享噪声 | ~110 | 收敛慢、一致性差 |
| 普通 CA 替代 STCA | ~95 | 效率低且效果略差 |
| w/o 跨模态引导 | ~90 | 无法处理部分输入 |

### 关键发现
- 共享噪声是最关键的设计——去掉后 FVD 恶化 ~30%
- 多模态信息（深度）确实改善了 RGB 预测质量（FVD 从 97 降至 84）
- STCA 比普通 CA 计算效率高 50% 且效果更好
- 在 SYNTHIA（语义）和 ERA5-Land（气候数据）上验证了框架对其他模态的泛化性
- 即使只有单模态输入，SyncVP 仍超越单模态 SOTA

## 亮点与洞察
- **共享噪声**是一个简单但深刻的洞察——同一场景的不同模态表征应该从相同的"混沌状态"出发恢复，大幅简化了学习难度
- **时空分离跨模态注意力**利用了 PVDM latent 的结构，在降低计算复杂度的同时保持效果
- **跨模态引导**巧妙地借鉴了 classifier-free guidance 的思路用于模态缺失场景

## 局限与展望
- 仅验证了两种模态的组合，三模态以上的扩展性待测试
- 自回归生成长视频时可能累积误差
- 当前仅在小分辨率（64×64/128×128）上验证
- 未来可探索与大规模视频扩散模型的结合

## 相关工作与启发
- **vs PVDM**: 单模态视频预测基线，SyncVP 在其基础上添加跨模态分支，即使不用深度也更好
- **vs LDM3D**: 图像域 RGB+D 联合生成（简单拼接通道），不适用于视频预测
- **vs MM-Diffusion**: 音视频联合生成，用全连接 cross-attention；SyncVP 的 STCA 更高效

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个多模态视频预测扩散框架，共享噪声策略巧妙
- 实验充分度: ⭐⭐⭐⭐ 三个数据集+多模态类型+详尽消融
- 写作质量: ⭐⭐⭐⭐ 框架清晰，消融设计合理
- 价值: ⭐⭐⭐⭐ 开辟多模态视频预测新方向，框架设计通用可扩展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](pqpp_a_joint_benchmark_for_text-to-image_prompt_and_query_performance_prediction.md)
- [\[CVPR 2025\] DiffSensei: Bridging Multi-Modal LLMs and Diffusion Models for Customized Manga Generation](diffsensei_bridging_multi-modal_llms_and_diffusion_models_for_customized_manga_g.md)
- [\[CVPR 2025\] Generative Modeling of Class Probability for Multi-Modal Representation Learning](generative_modeling_of_class_probability_for_multi_modal_representation_learning.md)
- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)
- [\[CVPR 2025\] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)

</div>

<!-- RELATED:END -->
