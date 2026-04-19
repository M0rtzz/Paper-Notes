---
title: >-
  [论文解读] Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers
description: >-
  [ICLR 2026][图像分割][Transformer] 本文提出 Jumbo 方法：将 ViT 的 CLS token 扩展为 $J$ 倍宽度，在注意力前拆分为 $J$ 个与 patch 等宽的 token 参与自注意力，注意力后重新拼接并经过专用的宽 FFN 处理——以极低的计算开销显著增加全局建模容量，使 plain ViT 在高速推理场景下超越专用高效架构（EfficientViT、SHViT、MobileNetV4），同时保留 ViT 的所有架构优势。
tags:
  - ICLR 2026
  - 图像分割
  - Transformer
  - CLS Token
  - 高效架构
  - Registers
  - 时间序列
  - ImageNet
---

# Thicker and Quicker: A Jumbo Token for Fast Plain Vision Transformers

**会议**: ICLR 2026  
**arXiv**: [2502.15021](https://arxiv.org/abs/2502.15021)  
**代码**: 无  
**领域**: segmentation  
**关键词**: Vision Transformer, CLS Token, 高效架构, Registers, 时间序列, ImageNet

## 一句话总结

本文提出 Jumbo 方法：将 ViT 的 CLS token 扩展为 $J$ 倍宽度，在注意力前拆分为 $J$ 个与 patch 等宽的 token 参与自注意力，注意力后重新拼接并经过专用的宽 FFN 处理——以极低的计算开销显著增加全局建模容量，使 plain ViT 在高速推理场景下超越专用高效架构（EfficientViT、SHViT、MobileNetV4），同时保留 ViT 的所有架构优势。

## 研究背景与动机

Vision Transformer 具有简洁、灵活和高效的优势：支持 token dropping（关键于 SOTA 自监督学习算法）、轻松适配多模态数据、可直接受益于 FlashAttention 等内核优化。但在高速推理（tiny/nano 尺寸）场景下，plain ViT 的性能远不如 EfficientViT、SHViT 等专用高效架构。

**核心症结**：在标准设置（224×224 图像、16×16 patch）下，196 个 patch token 加 1 个 CLS token 意味着仅 1/197 的模型容量用于全局信息聚合——显然不够。Darcet et al. (2024) 发现 ViT 会"劫持"部分 patch token 作为 pseudo-CLS token，并提出 Registers 作为额外全局 token 来缓解这一问题。

但 Registers 的局限在于：全局 token 之间仅通过注意力交互——注意力本质上是信息搬运机制（加权线性组合），表达能力有限。缺乏的是 FFN 提供的非线性函数建模能力。

**关键洞察**：将全局 token 拼接后送入一个宽 FFN，就能用非线性函数处理全局特征，而成本几乎可忽略——因为只处理一个 token。

## 方法详解

### 整体框架

Jumbo 在标准 ViT 基础上做了最小改动：(1) CLS token 宽度扩展为 $J \cdot D$；(2) 注意力前拆分、注意力后拼接；(3) 使用独立的宽 FFN 处理 Jumbo CLS token。整个方法保持 attention-only、non-hierarchical 的 plain ViT 形态。

### 关键设计

1. **Jumbo CLS Token 的创建与处理**：

    - 初始化一个宽度为 $J \cdot D$ 的可学习 CLS token $\mathbf{x}_{\text{Jumbo}} \in \mathbb{R}^{J \cdot D}$
    - **注意力前**：沿特征维度拆分为 $J$ 个宽度 $D$ 的 token，与 patch token 拼接形成长度 $(N+J)$ 的序列
    - **注意力中**：标准多头自注意力处理所有 $(N+J)$ 个等宽 token
    - **注意力后**：从序列中提取 $J$ 个 Jumbo 分片，沿通道维度重新拼接为 $\mathbb{R}^{1 \times J \cdot D}$
    - **专用 FFN**：宽度为 $J \cdot D$ 的独立 FFN 处理重组后的 Jumbo token；patch token 则由共享的标准 FFN 处理
    - 最后一层的 patch FFN 被丢弃（因为分类头直接从 Jumbo token 投射）

2. **为什么计算开销极低（设计动机）**：

    - ViT 层的计算量几乎完全由 patch 数 $N$ 和 patch 宽度 $D$ 决定
    - 添加 $J=6$ 个额外 token 对注意力的 FLOP 影响微乎其微（$(N+J)$ vs $N$，而 $N=196$）
    - 宽 FFN 虽然参数更多，但只处理单个 token——FLOP 贡献可忽略

3. **两个核心假说**：

    - **假说 1**：patch width 越窄（模型越小），Jumbo 增益越大——因为窄网络对全局容量的"饥渴"更严重
    - **假说 2**：输出维度越高（任务越复杂），Jumbo 增益越大——需要更多宽度来存储和推理更多概念

4. **从 Vision 到 Time Series 的迁移**：

    - 将 Jumbo 应用于 PatchTST 架构：将 1D 时间序列 patch 化后添加 Jumbo CLS token
    - 无需任何架构修改即可迁移，展示了 plain transformer 的通用性

### 损失函数 / 训练策略

- **ImageNet-1K**：function matching (知识蒸馏)，128×128 px 训练 400 epoch + 224×224 px 微调 20 epoch
- 教师：DeiT-III base (85.7%) 和 large (87.0%)
- AdamW 优化器，学习率 {1e-3, 3e-3}，1024 batch size
- 数据增强：mixup $\alpha=0.8$, cutmix $\alpha=1$, 3-Augment / AutoAugment
- **ImageNet-21K**：直接训练 50 epoch，使用 token dropping（90% 线性降到 10%）节省训练成本
- **时间序列**：PatchTST 框架，12 种超参组合网格搜索（4 个学习率 × 3 个 dropout）

## 实验关键数据

### 主实验——ImageNet-1K 高速场景

| 模型 | 大小 | 吞吐量 (imgs/s) | ImageNet-Val Top-1 | ImageNet-v2 Top-1 |
|------|------|-----------|----------------|-------------|
| ViT+Registers | nano (D=128) | 105.9K | 53.6 | 42.4 |
| ViT+Jumbo | nano (D=128) | 101.7K | **68.8** (+15.2) | **55.1** |
| ViT+Registers | tiny (D=192) | 52.2K | 68.8 | 55.9 |
| ViT+Jumbo | tiny (D=192) | 56.5K | **73.0** (+4.2) | **59.4** |
| EfficientViT-B1 | — | 38.7K | 72.8 | 60.4 |
| SHViT-S3 | — | 70.4K | 71.2 | 58.6 |
| MobileNetV4-conv-medium | — | 54.5K | 73.3 | 60.6 |

* Jumbo nano 超过 Registers tiny 的速度，且精度相当

### ImageNet-21K（10450 类）

| 模型 | 尺寸 | 吞吐量 | Top-1 |
|------|------|-------|-------|
| ViT+Registers | small | 8.4K | 41.48 |
| **ViT+Jumbo** | small | 8.0K | **44.95** (+3.4) |
| ViT+Registers | base | 3.6K | 46.31 |
| **ViT+Jumbo** | base | 3.2K | **47.28** (+1.0) |

### 时间序列分类（PatchTST 框架）

| 变体 | 单变量 Best Rank | 多变量 Best Rank |
|------|-----------|-----------|
| PatchTST | 2.0 | 2.1 |
| PatchTST+Registers | 2.5 | 2.1 |
| **PatchTST+Jumbo** | **1.5** | **1.6** |

### 消融实验

| Patch Width | Jumbo $J$ | Inner FFN mult | Throughput (128px) | IN-Val Top-1 |
|------------|-------|-------------|-----------|-------------|
| 192 | 2 | 2 | 71.6K | 70.0 |
| 192 | 4 | 4 | 64.9K | 72.2 |
| 192 | 6 | 4 | 56.5K | **73.0** |
| 384 | 6 | 4 | 19.5K | **78.3** |

### 关键发现

- **假说 1 得到验证**：Jumbo 增益随 patch width 递减而递增——nano (+13.5%) > tiny (+3.2%) > small (~0%)
- **假说 2 得到验证**：在 ImageNet-21K (10450 类) 上，即使 ViT-small 也能获得 +3.4% 提升，而 ImageNet-1K (1000 类) 上 small 无增益
- **参数量问题及解决方案**：$J=6$ 的 ViT-base Jumbo 参数量从 25.7M 增至 55M，但通过层间权重共享可降至 88.3M→88.8M（+LoRA），精度几乎不变
- **plain ViT 首次在高速场景追平/超越专用高效架构**：ViT+Jumbo 是首个 attention-only + non-hierarchical 就能竞争的架构

## 亮点与洞察

- 极其简洁的改动带来显著效果——两次 split、两次 concat 加一个独立 FFN，增加的代码不超过 10 行
- 保留了 plain ViT 的所有优势：token dropping（兼容 MAE, I-JEPA 等 SSL）、多模态支持、FlashAttention 兼容
- "宽度不足是窄 ViT 的核心瓶颈"这一洞察非常有价值——Registers 只增加了全局容量的线性部分，Jumbo 补充了非线性部分
- 实验设计的公平性值得称道：所有模型使用相同训练流程（同一教师、同一超参网格），避免了"配方差异 vs 架构差异"的混淆
- 从图像无缝迁移到时间序列的示范表明 Jumbo 是通用方法而非视觉特化

## 局限与展望

- ViT-small 在 ImageNet-1K 上无增益——方法并非万灵药
- 参数量增加可观（$J=6$ 时约 2 倍），虽有层间共享+LoRA 解决方案但增加了实现复杂度
- $J=6$ 的 ViT-base 在 ImageNet-21K 上因显存限制只能用 $J=3$
- 未测试大规模预训练（CLIP、DINOv2 等）和目标检测/分割下游任务
- 仅用知识蒸馏训练，未验证从头训练（不用教师）时 Jumbo 的效果
- 与 Hiera 等分层架构缺乏直接对比（虽然设计哲学不同）

## 相关工作与启发

- Darcet et al. (ICLR 2024) ViT+Registers 是直接前置工作——Jumbo 可视为"带非线性的超级 Register"
- Hiera (Ryali et al., 2023) 从另一个方向简化分层 ViT——用 pooling 替代卷积
- PatchTST (Nie et al., 2023) 展示了 plain transformer 在时间序列中的竞争力
- 启发：CLIP 等视觉-语言框架的 ViT 编码器需要建模 50K+ 词汇，Jumbo 在这种高输出维度场景有望获得更大收益

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [Revisiting \[CLS\] and Patch Token Interaction in Vision Transformers](revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)
- [MPM: Mutual Pair Merging for Efficient Vision Transformers](../../CVPR2026/segmentation/mpm_mutual_pair_merging_for_efficient_vision_transformers.md)
- [Locality-Attending Vision Transformer](locality-attending_vision_transformer.md)
- [Vision Transformers with Self-Distilled Registers](../../NeurIPS2025/segmentation/vision_transformers_with_self-distilled_registers.md)
- [LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](../../ICCV2025/segmentation/legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)

<!-- RELATED:END -->
