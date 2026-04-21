---
title: >-
  [论文解读] Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective
description: >-
  [AAAI 2026][图像生成][多模态虚假信息检测] 提出 RetSimd，通过将文本分段并用文本转图像模型生成一系列增补图像来"重放完整故事"，配合图神经网络融合多图像关系，显著提升了图像模态对虚假信息检测的贡献，在三个基准数据集上一致性地改进了五种 SOTA 方法的性能。
tags:
  - AAAI 2026
  - 图像生成
  - 多模态虚假信息检测
  - 文本转图像
  - 图神经网络
  - 模态贡献
  - 信息增益
---

# Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective

**会议**: AAAI 2026  
**arXiv**: [2511.06284](https://arxiv.org/abs/2511.06284)  
**代码**: https://github.com/wangbing1416/RETSIMD  
**领域**: Image Generation / Multimodal  
**关键词**: 多模态虚假信息检测, 文本转图像, 图神经网络, 模态贡献, 信息增益

## 一句话总结
提出 RetSimd，通过将文本分段并用文本转图像模型生成一系列增补图像来"重放完整故事"，配合图神经网络融合多图像关系，显著提升了图像模态对虚假信息检测的贡献，在三个基准数据集上一致性地改进了五种 SOTA 方法的性能。

## 研究背景与动机

多模态虚假信息检测（MMD）旨在判断社交媒体帖子（包含文本和图像）中的信息是否为虚假信息。当前主流方法通常假设文本和图像模态同等重要，主要通过融合两种模态特征或学习模态间语义不一致性来进行检测。

然而，作者提出了一个关键观察：**文本模态比图像模态信息量更大**。原因在于文本通常描述了事件的完整故事，而配图往往只展示了故事中的部分场景（如一条关于飓风的新闻，文本描述了整个灾难过程，但图片可能只拍到了一栋坍塌的房屋）。

为验证这一论点，作者设计了系统的初步消融实验：
- 对五种 SOTA 方法分别测试 full / text-only / image-only / text-replaced / image-replaced 五个变体
- 结果表明：移除图像（text-only）导致的性能下降**远小于**移除文本（image-only），甚至 image-only 变体在某些情况下几乎无效
- 进一步用信息论（信息增益）量化了两个模态的贡献度差异

核心矛盾：图像模态信息不足，拉低了多模态检测效果。切入角度：为图像模态"补课"——生成更多图像来完整呈现文本描述的故事。

## 方法详解

### 整体框架
RetSimd 包含四个核心模块：
1. 特征编码器（BERT + ResNet34）
2. 文本转图像生成器（基于 Stable Diffusion）
3. 多模态融合网络（图神经网络）
4. 真实性分类器

训练过程交替优化生成器和检测器。

### 关键设计

1. **文本分段与图像增补生成**:

    - 功能：将文本 $\mathbf{x}^t$ 按固定长度滑动窗口分为 $K$ 个片段，每个片段送入预训练的 Stable Diffusion 模型生成对应图像
    - 核心思路：每个文本片段描述故事的一部分场景，生成的图像序列可以"重放"整个故事
    - 设计动机：弥补原始单张图像只能展示部分场景的不足，使图像模态包含与文本同等丰富的信息

2. **生成器的信息论正则化**:

    - 功能：设计两个互信息目标函数来微调生成器
    - **文本-图像互信息 $\mathcal{R}_{MTI}$**：利用同一文本的不同片段作为天然的正负样本，约束生成图像与对应文本片段的语义一致性，并引入自适应权重 $\xi_{jm}$ 考虑片段间的时序关系
    - **图像-标签互信息 $\mathcal{R}_{MIL}$**：最大化生成图像序列对真实性标签的信息增益 $\mathscr{G}(y_i|\{\mathbf{x}_{ij}^g\})$
    - 同时用 LAION-2B 数据集持续后训练生成器以保持图像质量
    - 设计动机：确保增补图像既语义正确又对虚假信息检测有实际帮助

3. **基于图结构的多模态融合**:

    - 功能：将原始图像和 $K$ 张增补图像作为节点构建图结构，用 GNN 学习融合特征
    - 三种启发式边关系：
        - **中心关系**：所有增补图像与原始图像相连（原始图像展示最核心场景）
        - **时序关系**：按文本片段顺序连接相邻增补图像
        - **语义关系**：用交叉注意力计算图像间语义相似度，相似的图像相连
    - 融合后的图像特征再与文本特征通过交叉注意力生成最终特征
    - 设计动机：捕捉增补图像间的潜在关系，比简单拼接更有效

### 损失函数 / 训练策略
- 生成器目标：$\mathcal{L}_{GEN} = \mathcal{L}_{T2I} + \alpha_1 \mathcal{R}_{MTI} + \alpha_2 \mathcal{R}_{MIL}$
- 检测器目标：$\mathcal{L}_{DET} = \mathcal{L}_{VC} + \beta \mathcal{R}_{CA}$（交叉熵 + 交叉注意力正则）
- 交替优化：先固定检测器训练生成器，再固定生成器训练检测器

## 实验关键数据

### 主实验

在 GossipCop 数据集上，RetSimd 一致性地提升了所有 5 种基线方法：

| 基线方法 | 原始 Acc | +RetSimd Acc | 提升Δ | 图像信息增益 |
|----------|----------|-------------|-------|-------------|
| ResNet+BERT | 87.17 | **88.13*** | +1.21 | 0.0349→0.0301 |
| R&B+SAFE | 87.14 | **88.30*** | +1.29 | 0.0325→0.0287 |
| R&B+MCAN | 87.29 | **88.22*** | +1.12 | 0.0200→0.0129 |
| R&B+CAFE | 87.16 | **88.38*** | +1.09 | 0.0439→0.0402 |
| R&B+BMR | 87.32 | **88.42*** | +1.02 | 0.0458→0.0409 |
| R&B+GAMED | 87.03 | **88.30*** | +1.81 | 0.0350→(改善) |

*标注 * 表示 p-value < 0.05 显著性检验通过*

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full RetSimd | 88.42 Acc | 完整方法最优 |
| 移除 $\mathcal{R}_{MTI}$ | 性能下降 | 文本-图像对齐对生成质量关键 |
| 移除 $\mathcal{R}_{MIL}$ | 性能下降 | 图像-标签信息增益确保图像对检测有用 |
| 固定长度分段 vs 语义分段 | 固定长度最优 | 简单策略反而更稳定 |
| 不同 $K$ 值 (片段数) | $K=4$ 或 $K=5$ 最优 | 太少信息不够，太多引入噪声 |
| 移除 GNN 图融合 | 性能下降 | 图结构关系对融合效果重要 |

### 关键发现
- RetSimd 不仅提升了最终检测准确率，还**降低了图像模态的信息增益差距**（即图像变得"更有用"了）
- 在 Weibo 和 Twitter 数据集上也观察到一致的跨数据集改善
- 增补图像的数量 $K$ 存在最优值，过多的图像可能引入噪声
- 所有改进都通过了统计显著性检验（p < 0.05）

## 亮点与洞察
- **问题定义新颖**：首次系统性地用信息论量化了 MMD 中两个模态的贡献度不均衡
- **以生成增强检测**：用 text-to-image 模型不是为了生成内容，而是为了增强检测模型的输入
- **即插即用**：RetSimd 是一个框架级方法，可以叠加到任意 MMD 基线上
- 三种图关系（中心/时序/语义）的设计合理，抓住了图像间的核心联系
- 贡献度指标的提出为理解多模态系统提供了分析工具

## 局限与展望
- 生成器基于 Stable Diffusion，推理开销大（每条样本需生成 $K$ 张图像）
- 图像生成质量受限于 SD 的零样本能力，对新闻特定场景可能不够准确
- 仅关注了二分类（真/假），未考虑更细粒度的虚假类型
- 固定长度分段策略简单但可能切断语义单元
- 未讨论对抗攻击下的鲁棒性

## 相关工作与启发
- "模态贡献不均衡"这一观察具有普适性，在其他多模态任务中可能同样存在
- 用生成模型增强判别模型的思路值得借鉴，特别是当某个模态信息不足时
- 信息论视角的贡献度分析方法可以扩展到其他需要理解模态交互的场景
- 图结构融合多图像的方法可推广到视频理解等需要整合多帧信息的任务

## 评分
- 新颖性: ⭐⭐⭐⭐ — 角度新颖（图像模态贡献不足 + 生成式增强），但技术组件较常规
- 实验充分度: ⭐⭐⭐⭐⭐ — 3 个数据集、5 种基线、信息论分析、充分的消融
- 写作质量: ⭐⭐⭐⭐ — 动机论证充分，初步实验设计精巧
- 价值: ⭐⭐⭐⭐ — 即插即用框架有实用价值，信息论分析有启发性

<!-- RELATED:START -->

## 相关论文

- [Infinite-Story: A Training-Free Consistent Text-to-Image Generation](infinite-story_a_training-free_consistent_text-to-image_gene.md)
- [When One Modality Rules Them All: Backdoor Modality Collapse in Multimodal Diffusion Models](../../ICLR2026/image_generation/when_one_modality_rules_them_all_backdoor_modality_collapse_in_multimodal_diffus.md)
- [TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective](tsgdiff_rethinking_synthetic_time_series_generation_from_a_pure_graph_perspectiv.md)
- [Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](../../CVPR2026/image_generation/enhancing_image_aesthetics_with_dualconditioned_di.md)
- [MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](../../CVPR2026/image_generation/micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)

<!-- RELATED:END -->
