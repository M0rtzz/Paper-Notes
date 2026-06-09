---
title: >-
  [论文解读] Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)
description: >-
  [CVPR 2025][模型压缩][参数高效微调] 提出 FPET（Faster Parameter-Efficient Tuning），在参数高效微调（PET）中引入即插即用的 token 冗余压缩模块——在 ViT 中间层用可微的二分匹配策略合并约一半的 token…
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "参数高效微调"
  - "Token合并"
  - "推理加速"
  - "可微匹配"
  - "直通估计器"
---

# Faster Parameter-Efficient Tuning with Token Redundancy Reduction (FPET)

**会议**: CVPR 2025  
**arXiv**: [2503.20282](https://arxiv.org/abs/2503.20282)  
**代码**: [github.com/kyk120/fpet](https://github.com/kyk120/fpet)  
**领域**: 模型压缩 / 参数高效微调  
**关键词**: 参数高效微调, Token合并, 推理加速, 可微匹配, 直通估计器

## 一句话总结

提出 FPET（Faster Parameter-Efficient Tuning），在参数高效微调（PET）中引入即插即用的 token 冗余压缩模块——在 ViT 中间层用可微的二分匹配策略合并约一半的 token，实现比原始 backbone 更快 20% 的推理速度、减少约 40% GPU显存、且精度与 SOTA PET 方法持平。

## 研究背景与动机

**领域现状**：参数高效微调（PET）方法（LoRA、AdaptFormer、VPT 等）通过只训练少量参数实现存储高效。部分方法（RepAdapter、SSF）可做到不增加推理延迟。但所有这些方法都继承了预训练大模型本身的推理延迟和计算量。

**现有痛点**：现有 PET 方法在推理效率上存在瓶颈——要么引入额外模块增加推理开销（如 Adapter），要么只能维持原始推理速度不变（如 LoRA 的参数融合）。对于 Web 平台和边缘设备等需要低延迟的场景不够实用。

**核心矛盾**：参数效率 ≠ 计算效率。PET 解决了存储问题但没解决速度问题。而 token 剪枝/合并方法（如 ToMe）用于全量训练场景，直接用于 PET 会因非可微性和 Adapter 影响不充分导致次优合并。

**切入角度**：将 token 冗余压缩与 PET 结合——在 ViT 中间层一次性合并 ~50% 的 token，并设计可微的匹配策略使合并过程可被端到端优化。

**核心 idea**：PET + 单层可微 token 合并 = 推理速度突破原始 backbone 的天花板。

## 方法详解

### 整体框架

基于标准 ViT-B/16（12 层），在第 6 层插入一个 token 合并模块：前 6 层保持原始 196 个 token 全量计算（保证 Adapter 影响充分体现），在第 6 层合并掉 98 个 token，后 6 层仅处理约 98 个 token，从而大幅减少后半部分的计算量。

### 关键设计

1. **棋盘格划分**：不同于 ToMe 的奇偶交替条纹划分，采用棋盘格（checkerboard）模式将 token 分为 A/B 两组，使每个 B 组 token 可以与上下左右四个相邻 token 合并（条纹只能左右合并），提升匹配覆盖率

2. **可微二分匹配**：用 Adapter 先精炼 self-attention 的 key 特征 $\mathbf{K'} = \mathbf{K} + s \cdot \text{ReLU}(\mathbf{K}\mathbf{W}_{down})\mathbf{W}_{up}$，然后计算精炼后的相似矩阵。关键点是用 sigmoid + 阈值 0.5 代替 max/top-k 操作构造硬匹配矩阵，并通过直通估计器（STE）传递梯度：$\tilde{\mathbf{C}}_{AB} = \hat{\mathbf{C}}_{AB} + \text{const}(\mathbf{C}_{AB} - \hat{\mathbf{C}}_{AB})$

3. **单层合并策略**：只在第 6 层合并一次（而非 ToMe 每层合并少量），因为早期层 Adapter 影响未充分体现、相似度不可靠；单次合并还避免了重复计算相似矩阵的开销，节省显存

### 损失函数 / 训练策略

与标准 PET 相同的分类交叉熵损失，token 合并过程通过 STE 自然地被端到端优化。梯度传播到 key 精炼 Adapter 但**不再继续传播**到 backbone（避免 token 间的 push-pull 效应干扰特征学习）。训练 100 epoch，AdamW 优化器。

## 实验关键数据

| 方法 | Acc (%) | 推理时间 (ms) | FLOPs (G) | 显存 (GB) |
|------|---------|-------------|-----------|----------|
| Full Fine-tuning | 68.9 | 2.62 | 17.6 | 11.9 |
| VPT-Deep | 72.0 | 2.79 (+6.5%) | 18.5 | 9.8 |
| LoRA | 75.7 | 2.62 (+0%) | 17.6 | 8.4 |
| AdaptFormer | 76.2 | 2.68 (+1.5%) | 17.6 | 7.6 |
| Bi-AdaptFormer | 77.0 | 2.77 (+5.7%) | 17.7 | 7.6 |
| **FPET+LoRA** | **75.6** | **2.10 (-19.8%)** | **13.3** | **7.1** |
| **FPET+AdaptFormer** | **76.2** | **2.12 (-19.1%)** | **13.5** | **6.2** |
| **FPET+Bi-AdaptFormer** | **77.0** | **2.17 (-17.2%)** | **13.5** | **6.2** |

### 关键发现

- FPET 是唯一能让推理速度**快于**原始 backbone 的 PET 方法（-19.8%）
- 与 5 种 SOTA PET 方法结合（即插即用），精度几乎无损（77.0% vs 77.0%）
- FLOPs 降低约 24%，训练显存降低约 40-48%
- 棋盘格划分 vs 条纹划分：棋盘格提供更好的合并覆盖，精度更高
- 合并位置第 6 层最优——太早合并精度下降，太晚效率增益不足

### 三组分类任务详细表现

| 类别 | FPET+Bi-AdaptFormer | Bi-AdaptFormer | 差异 |
|------|-------------------|----------------|------|
| Natural（7 任务） | ~77% | ~77% | ≈0% |
| Specialized（4 任务） | ~82% | ~82% | ≈0% |
| Structured（8 任务） | ~72% | ~72% | ≈0% |
| **推理速度** | **2.17ms** | **2.77ms** | **-21.7%** |

在 Natural/Specialized/Structured 三组中精度均持平，但推理全面加速。

## 亮点与洞察

- **打破PET推理速度天花板**——首次实现 PET 后推理比原始模型更快
- **即插即用设计**——可直接叠加到任何现有 PET 方法上（LoRA/AdaptFormer/Bi-LoRA等）
- **可微 token 合并的工程巧思**——STE + 梯度截断的组合既保证可学习性，又避免了反向传播对特征的干扰

## 局限与展望

- 仅在 VTAB-1K（每任务 1000 样本）上验证，未测试大规模数据集如 ImageNet
- 固定在第 6 层合并 50%——不同任务可能需要自适应的合并位置和比例
- 仅适用于 ViT 架构，对 CNN 或混合架构不直接适用
- 合并操作对细粒度空间任务（如检测、分割）的影响未探讨

## 相关工作

- **PET 方法**：LoRA, AdaptFormer, VPT, SSF, RepAdapter, Bi-LoRA 等
- **Token 压缩**：ToMe（二分软匹配）、DiffRate（学习阈值）——非可微限制
- **效率增强 PET**：SynQT（解耦学习）、Pruned RepAdapter（结构剪枝）——精度损失

## 评分

- 新颖性: ⭐⭐⭐⭐ 可微token合并+PET的首次有效结合
- 实验充分度: ⭐⭐⭐⭐ 5种PET方法×19个数据集的全面验证
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表信息密度高
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高，即插即用加速PET推理

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [\[CVPR 2025\] Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation](parameter_efficient_mamba_tuning_via_projector-targeted_diagonal-centric_linear_.md)
- [\[ICCV 2025\] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](../../ICCV2025/model_compression/tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](../../ACL2025/model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)

</div>

<!-- RELATED:END -->
