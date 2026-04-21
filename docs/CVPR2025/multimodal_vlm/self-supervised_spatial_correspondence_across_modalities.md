---
title: >-
  [论文解读] Self-Supervised Spatial Correspondence Across Modalities
description: >-
  [CVPR 2025][多模态][跨模态像素对应] 将对比随机游走（CRW）框架扩展到跨模态像素级对应问题，通过同时学习模态内和模态间的循环一致性特征表示，在无需配对标注的情况下实现 RGB-Depth、RGB-Thermal、Photo-Sketch 等跨模态密集匹配，显著超越现有方法。
tags:
  - CVPR 2025
  - 多模态
  - 跨模态像素对应
  - 对比随机游走
  - 自监督学习
  - RGB-深度匹配
  - RGB-热成像匹配
---

# Self-Supervised Spatial Correspondence Across Modalities

**会议**: CVPR 2025  
**arXiv**: [2506.03148](https://arxiv.org/abs/2506.03148)  
**代码**: https://ayshrv.com/cmrw (有)  
**领域**: 多模态VLM  
**关键词**: 跨模态像素对应、对比随机游走、自监督学习、RGB-深度匹配、RGB-热成像匹配

## 一句话总结

将对比随机游走（CRW）框架扩展到跨模态像素级对应问题，通过同时学习模态内和模态间的循环一致性特征表示，在无需配对标注的情况下实现 RGB-Depth、RGB-Thermal、Photo-Sketch 等跨模态密集匹配，显著超越现有方法。

## 研究背景与动机

不同传感器（RGB、深度、热成像）拍摄同一场景时，像素存储的信息完全不同（亮度 vs 深度 vs 温度），因此跨模态像素对应是一个极具挑战性的问题。传统方法面临几个困难：

1. **光度一致性假设失效**：深度图和 RGB 图的像素值含义完全不同，无法使用光度损失
2. **跨模态翻译困难**：先将一种模态翻译到另一种（如用 CycleGAN）再匹配，但翻译本身就是个难题（单目深度估计）
3. **需要配对数据**：现有多模态学习方法通常需要空间对齐的多传感器配对数据

核心观察：自监督追踪方法（如 CRW）通过循环一致性学习对应关系——一个点在视频中"走"出去再"走"回来应该回到原位。作者将这个思路从时间维度扩展到模态维度。然而直接在跨模态上优化循环一致性会遇到收敛困难（由于模态间巨大的域差异），因此需要辅助的模态内随机游走来稳定训练。

## 方法详解

### 整体框架

构建一个图，节点是两种模态图像中的 patch，边连接跨模态的 patch 对。训练一个全局匹配Transformer，为随机游走分配转移概率。随机游走者从一种模态出发，经过另一种模态，再返回。通过最大化返回概率（循环一致性）来学习跨模态对应关系。

### 关键设计

1. **跨模态匹配Transformer（Cross-modal Matching Transformer）**:
    - 功能：为不同模态的图像 patch 对生成匹配特征
    - 核心思路：每种模态使用独立的视觉编码器提取特征 $\phi(I_t^m) \in \mathbb{R}^{\frac{H}{c} \times \frac{W}{c} \times d}$（$c=4$ 下采样），特征拼接 2D 位置编码后送入共享 Transformer（6层 self-attention + cross-attention + FFN），输出关联特征 $F_t^{m_1}$ 和 $F_{t+k}^{m_2}$。转移矩阵 $A_{t,t+k}^{m_1,m_2} = \text{softmax}(F_t^{m_1}(F_{t+k}^{m_2})^\top / \tau)$，其中 $\tau = \sqrt{d}$
    - 设计动机：共享 Transformer 使模型能同时处理模态内（$m_1 = m_2$）和模态间（$m_1 \neq m_2$）的匹配，无需模态特定假设。使用独立编码器避免强制不同模态共享低层特征

2. **跨模态+模态内联合随机游走（Joint Cross-modal & Intra-modal CRW）**:
    - 功能：解决直接跨模态优化收敛困难的问题
    - 核心思路：跨模态损失使用回文序列 $\{I_t^{m_1}, I_{t+k}^{m_2}, I_t^{m_1}\}$，损失为 $\mathcal{L}_{\text{cross-crw}} = \mathcal{L}_{\text{CE}}(A_{t,t+k}^{m_1,m_2} A_{t+k,t}^{m_2,m_1}, T_f^b(I))$。模态内损失使用数据增强的裁剪对 $\{I_{\text{ori}}^{m_i}, I_{\text{aug}}^{m_i}, I_{\text{ori}}^{m_i}\}$，损失为 $\mathcal{L}_{\text{intra-crw}} = \sum_{i=1}^{2} \mathcal{L}_{\text{CE}}(A_{\text{ori,aug}}^{m_i} A_{\text{aug,ori}}^{m_i}, T_f^b(I))$
    - 设计动机：仅用跨模态损失训练时会陷入局部最优（因初始随机特征在不同模态间可能产生任意对齐，如 RGB 亮区与深度远区高值碰巧对齐）。模态内 CRW 提供了"脚手架"——先学会在同一模态内匹配，再扩展到跨模态

3. **边缘感知平滑损失（Edge-Aware Smoothness Loss）**:
    - 功能：鼓励预测流场的空间连贯性
    - 核心思路：$\mathcal{L}_{\text{smooth}} = \mathbb{E}_p \sum_{d \in \{x,y\}} \exp(-\lambda_c I_d(p)) |\frac{\partial^2 \mathbf{f}_{s,t}(p)}{\partial d^2}|$，惩罚流场的二阶导数，但在图像梯度大的边缘处降低惩罚权重
    - 设计动机：仅在源图像为 RGB 时应用（因为 RGB 的视觉边缘是可靠的感知分组线索）。对于深度/热成像作为源时，边缘的视觉含义不同，不适合使用此约束

### 损失函数 / 训练策略

总损失为三项之和：$\mathcal{L}_{\text{cross-crw}} + \mathcal{L}_{\text{intra-crw}} + \lambda_s \mathcal{L}_{\text{smooth}}$

采用三阶段训练策略：
1. **Stage 1**：仅模态内 CRW（RGB-RGB 和 Depth-Depth / Thermal-Thermal）
2. **Stage 2**：加入跨模态 CRW（RGB-Depth 和 Depth-RGB / RGB-Thermal 和 Thermal-RGB）
3. **Stage 3**：加入平滑损失

对于语义匹配任务（photo-sketch, cross-style），使用 DINOv2 作为共享编码器并微调，利用预训练的语义先验。

## 实验关键数据

### 主实验（几何对应）

| 数据集 | 方向 | 本文 | RAFT(监督) | GMFlow(监督) | ARFlow(SSL) |
|--------|------|------|-----------|-------------|-------------|
| NYU Depth | RGB→D | 33.5 | 7.9 | 12.7 | 7.5 |
| NYU Depth | D→RGB | 34.3 | 1.3 | 12.5 | 7.4 |
| Thermal-IM | RGB→T | 41.8 | 5.6 | 3.8 | 12.5 |
| Thermal-IM | T→RGB | 47.9 | 0.9 | 2.6 | 13.2 |
| KAIST | RGB→T | 35.2 | 29.2 | 23.1 | 31.0 |
| KAIST | T→RGB | 34.1 | 7.4 | 22.3 | 30.4 |

### 消融实验

| 配置 | NYU RGB→D | NYU D→RGB | Thermal RGB→T | Thermal T→RGB |
|------|-----------|-----------|---------------|---------------|
| 仅模态内 CRW | 2.5 | 2.2 | 4.9 | 6.2 |
| 仅跨模态 CRW | 5.6 | 4.5 | 6.2 | 8.3 |
| 模态内+跨模态（无平滑） | 19.1 | 21.1 | 30.2 | 38.5 |
| 全部（+平滑） | 33.5 | 34.3 | 41.8 | 47.9 |

### 语义匹配（Photo-Sketch, PSC6K）

| 方法 | PCK-5 | PCK-10 |
|------|-------|--------|
| DINOv2+NN | 11.48 | 31.66 |
| SD-DINO | 33.10 | 70.50 |
| PSCNet (SOTA, 监督) | 57.92 | 84.72 |
| 本文 (DINO+full) | 53.61 | 82.20 |

### 关键发现

- **跨模态匹配上大幅超越所有基线**：NYU Depth RGB→D 达到 33.5 vs 之前最好的 12.7（GMFlow），提升超过 2.6 倍
- **模态内预训练是关键**：仅跨模态 CRW 训练不稳定（5.6 vs 33.5），模态内预训练提供了至关重要的初始化
- **平滑损失贡献显著**：从 19.1 提升到 33.5（+14.4），边缘感知平滑对跨模态匹配的正则化作用巨大
- **在语义匹配上与专门方法竞争**：Photo-Sketch 上接近监督 SOTA（53.61 vs 57.92 PCK-5），cross-style 上超越所有方法包括监督的 GeoAwareSC

## 亮点与洞察

- **极简却强大**：不依赖任何手工设计的光度一致性度量，纯粹通过循环一致性自监督学习，适用于任意模态对
- **三阶段训练策略**的渐进式设计非常合理：先让模型学会"看同类模态"，再学"跨模态看"，最后加空间约束
- **数据效率高**：不需要空间/时间对齐的配对数据，只需来自同一场景的不同时刻、不同模态的帧
- **通用性强**：同一框架无需修改即可处理 4 种不同的跨模态匹配任务

## 局限与展望

- 仅验证了 4 种模态对，其他模态（如 SAR、MRI、超声波）可能缺乏遮挡轮廓等匹配线索
- Cross-style 匹配中偶尔出现左右对称混淆（如动物左右肢体）
- 当前所有数据集都包含 RGB 模态，纯非 RGB 模态间的匹配未探索
- 评估基准需要手动标注关键点或依赖追踪器生成伪标签，评估的可靠性有局限

## 相关工作与启发

- CRW 从时间追踪到跨模态匹配的自然扩展是一个优雅的概念迁移
- 与音频-视觉对应（patch-level）不同，本文聚焦于像素级密集对应
- 启发：该思路可进一步扩展到 3D 点云与 2D 图像的稠密匹配、跨模态 SLAM 等应用

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 CRW 扩展到跨模态是自然但有效的创新，模态内 CRW 辅助训练是关键技术贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 4 种任务、全面的消融、新基准构建（cross-style），基线丰富
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，公式简洁，图解直观
- 价值: ⭐⭐⭐⭐ 为跨模态对应提供了通用的自监督解决方案，可直接应用于多传感器融合场景

<!-- RELATED:START -->

## 相关论文

- [A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks](a_closed-form_solution_for_debiasing_vision-language_models_with_utility_guarant.md)
- [BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models](stealthy_backdoor_attack_in_self-supervised_learning_vision_encoders_for_large_v.md)
- [Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](../../ECCV2024/multimodal_vlm/self-adapting_large_visual-language_models_to_edge_devices_across_visual_modalit.md)
- [Vision-Language Models Struggle to Align Entities across Modalities](../../ACL2025/multimodal_vlm/vision-language_models_struggle_to_align_entities_across_modalities.md)
- [It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](its_a_blind_match_towards_vision-language_correspondence_without_parallel_data.md)

<!-- RELATED:END -->
