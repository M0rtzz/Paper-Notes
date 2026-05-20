---
title: >-
  [论文解读] Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders
description: >-
  [CVPR 2025][模型压缩][视线目标估计] 提出 Gaze-LLE，一个基于冻结 DINOv2 编码器的极简视线目标估计框架——仅用 ~2.8M 可训练参数（比先前方法少 1-2 个数量级）、无需辅助深度/姿态模型、无需独立头部编码器…
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "视线目标估计"
  - "DINOv2"
  - "基础模型"
  - "位置提示"
  - "轻量解码器"
---

# Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders

**会议**: CVPR 2025  
**arXiv**: [2412.09586](https://arxiv.org/abs/2412.09586)  
**代码**: [github.com/fkryan/gazelle](https://github.com/fkryan/gazelle)  
**领域**: 模型压缩 / 视线估计  
**关键词**: 视线目标估计, DINOv2, 基础模型, 位置提示, 轻量解码器

## 一句话总结

提出 Gaze-LLE，一个基于冻结 DINOv2 编码器的极简视线目标估计框架——仅用 ~2.8M 可训练参数（比先前方法少 1-2 个数量级）、无需辅助深度/姿态模型、无需独立头部编码器，通过人物位置提示 + 轻量 transformer 解码器即在 GazeFollow/VideoAttentionTarget 等基准上达到 SOTA（AUC 0.958）。

## 研究背景与动机

**领域现状**：视线目标估计预测人在场景中看向哪里，是理解人类行为的关键环节。先前方法都采用多分支架构：独立的头部编码器 + 场景编码器，加上深度/姿态/目标检测等辅助模型，特征融合复杂。

**现有痛点**：（1）多分支架构训练复杂，需要精心设计融合机制和多任务损失；（2）可训练参数量大（通常 30-100M+）；（3）收敛慢——通常需要数十 GPU 小时。DINOv2 等基础模型在深度估计等密集预测任务上表现优异，但直接替换到现有视线架构中反而性能下降。

**核心矛盾**：DINOv2 特征强大，但现有视线架构无法有效利用——因为多分支设计要求将头部位置信息在编码前注入（RGB + head channel），冻结 DINOv2 时无法适应这种输入格式。

**切入角度**：三个关键设计决策——（1）头部位置在编码器之后注入而非之前；（2）用 transformer 而非 CNN 解码，获得全局信息传播；（3）取消独立头部分支，因为 DINOv2 已编码足够的头部朝向信息。

**核心 idea**：冻结 DINOv2 + 位置提示式轻量解码器 = 极简且 SOTA 的视线估计。

## 方法详解

### 整体框架

输入 RGB 图像经冻结 DINOv2 编码器提取场景 token 特征图（$d_\mathcal{F} \times H \times W$），线性投影到 $d_\text{model}$。在特征图上的头部位置添加可学习位置嵌入（head prompting），然后送入 3 层 transformer 编码器更新特征。最后上采样解码为热图 + 可选的 in/out 分类。

### 关键设计

1. **Head Position Prompting（头部位置提示）**：将头部 bounding box 下采样为二值 mask $M$，在对应位置的 token 上加上可学习嵌入 $p_\text{head}$：$S = x_\mathcal{F} + (M * p_\text{head})$。关键是在编码器**之后**注入（late integration），使冻结编码器的特征不受头部信息干扰

2. **轻量 Transformer 解码器**：仅 3 层标准 transformer encoder layer + 2D 正弦位置编码，利用 self-attention 的全局信息传播能力，使距离头部较远的视线目标也能被捕捉（CNN 解码器因感受野限制失败）

3. **无需头部分支（No Head Branch）**：实验证明使用 transformer 解码器时，额外的头部裁剪分支几乎不带来提升（AUC 0.954 vs 0.953），因为 DINOv2 的全局特征已编码了头部朝向信息，transformer 的全局注意力能自动提取

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_\text{hm} + \lambda \mathcal{L}_\text{in/out}$$

$\mathcal{L}_\text{hm}$ 是像素级 BCE 损失，GT 为高斯热图（$\sigma=3$）。$\mathcal{L}_\text{in/out}$ 是 BCE 分类损失。DINOv2 backbone 完全冻结，仅训练 2.8M 参数的解码器。训练时间 < 1.5 GPU 小时即达 SOTA。

## 实验关键数据

| 方法 | 可训练参数 | 输入 | GazeFollow AUC↑ | Avg L2↓ | Min L2↓ |
|------|-----------|------|-----------------|---------|---------|
| Chong et al. | ~61M | I | 0.921 | 0.137 | 0.077 |
| Gupta et al. | 35M | I+D+P | 0.943 | 0.114 | 0.056 |
| Tafasca et al. | 105M | I | 0.944 | 0.113 | 0.057 |
| **Gaze-LLE (ViT-B)** | **2.8M** | **I** | **0.956** | **0.104** | **0.045** |
| **Gaze-LLE (ViT-L)** | **2.9M** | **I** | **0.958** | **0.099** | **0.041** |

### 设计决策消融

| 头部注入位置 | 解码器 | 分支 | AUC | Avg L2 |
|-------------|--------|------|-----|--------|
| 编码器前(early) | CNN | H+S | 0.854 | 0.254 |
| 编码器前(early) | Transformer | H+S | 0.904 | 0.178 |
| 编码器后(late) | CNN | H+S | 0.932 | 0.155 |
| 编码器后(late) | Transformer | H+S | 0.954 | 0.113 |
| **编码器后(late)** | **Transformer** | **仅 S** | **0.953** | **0.114** |

### 关键发现

- DINOv2 直接替换到现有架构中性能**下降**（AUC 0.921→0.908），必须配合新解码器设计
- Late head integration 比 early integration 提升 AUC ~0.05——冻结编码器的关键决策
- Transformer 解码器 vs CNN 解码器：AUC 0.953 vs 0.916（无头部分支时差距更大）
- 跨数据集零样本泛化能力强——无需微调即在 ChildPlay 和 GOO-Real 上表现良好

## 亮点与洞察

- **极致简化**——将多分支复杂架构简化为单一编码器+轻量解码器，参数量减少 10-40 倍
- **训练效率惊人**——< 1.5 GPU 小时即达 SOTA，先前方法需要数十小时
- **head prompting 的启示**——冻结基础模型时，任务条件信息应在特征提取后注入
- **首次证明**视线估计可以不需要深度/姿态等辅助信号，DINOv2 已隐式编码这些信息

## 局限与展望

- 依赖预训练 DINOv2 的质量——更大更强的基础模型会进一步提升
- 仅处理单人视线，多人场景需要多次前向解码
- 视频场景仅逐帧处理，未利用时序信息
- Head bounding box 由外部检测器提供（不是端到端的）
- 对极端遮挡或低分辨率头部的鲁棒性未充分探讨

### VideoAttentionTarget 结果

| 方法 | AUC↑ | L2↓ | AP in/out↑ |
|------|------|-----|--------|
| Chong et al. | 0.860 | 0.134 | 0.853 |
| Miao et al. | 0.917 | 0.109 | 0.908 |
| **Gaze-LLE (ViT-L)** | **0.937** | **0.103** | **0.903** |

## 相关工作

- **多分支视线估计**：Recasens, Chong, Fang, Gupta, Tafasca 等——场景+头部+辅助模型
- **视觉基础模型**：DINOv2, CLIP——大规模自监督预训练
- **密集预测与基础模型**：Depth Anything 等证明冻结模型可做密集预测
- **社会注视分析**：共同注意力、互视等——可受益于共享场景表征

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次证明冻结基础模型+轻量解码器可完成视线估计
- 实验充分度: ⭐⭐⭐⭐⭐ 设计空间分析极其详尽，4个benchmark
- 写作质量: ⭐⭐⭐⭐⭐ 设计决策的分析和叙述非常清晰
- 价值: ⭐⭐⭐⭐⭐ 为视线估计指明了基础模型时代的正确范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)
- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](../../NeurIPS2025/model_compression/c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](mambaic_state_space_models_for_high-performance_learned_image_compression.md)

</div>

<!-- RELATED:END -->
