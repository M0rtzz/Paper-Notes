---
title: >-
  [论文解读] Token Bottleneck: One Token to Remember Dynamics
description: >-
  [NeurIPS 2025][视频理解][自监督学习] 提出Token Bottleneck（ToBo），一种自监督视觉表征学习流水线，通过将参考场景压缩为单个瓶颈token、并利用该token与极少量目标场景patch来预测后续场景，使视觉骨干网络同时学会保守编码场景信息和捕获时间动态变化。
tags:
  - NeurIPS 2025
  - 视频理解
  - 自监督学习
  - 视觉表征
  - 机器人操控
  - token瓶颈
  - 连续场景理解
---

# Token Bottleneck: One Token to Remember Dynamics

**会议**: NeurIPS 2025  
**arXiv**: [2507.06543](https://arxiv.org/abs/2507.06543)  
**代码**: [GitHub](https://github.com/naver-ai/tobo)  
**领域**: 视频理解  
**关键词**: 自监督学习, 视觉表征, 机器人操控, token瓶颈, 连续场景理解

## 一句话总结

提出Token Bottleneck（ToBo），一种自监督视觉表征学习流水线，通过将参考场景压缩为单个瓶颈token、并利用该token与极少量目标场景patch来预测后续场景，使视觉骨干网络同时学会保守编码场景信息和捕获时间动态变化。

## 研究背景与动机

### 连续场景理解的核心需求

在视觉追踪、机器人操控等连续场景理解任务中，视觉骨干网络需要两个能力：(1) 保守地编码当前观察到的场景状态；(2) 捕获连续场景间的时间动态转变。然而现有自监督方法在这两方面都有明显不足。

### 静态场景SSL的局限

MAE等基于静态图像的自监督方法通过掩码预测学会了良好的局部化能力，但从未被优化来比较连续帧，无法建模时间动态。最近的研究还发现其难以学习更广泛的上下文，导致全局理解受限。

### 动态场景SSL的局限

SiamMAE等方法尝试在MAE框架中引入跨帧对应学习，将参考帧的patch传播到其在目标帧中的对应位置。然而作者发现这类方法的改进非常有限，在某些机器人操控任务上甚至不如MAE。

**核心问题分析**：SiamMAE的训练目标虽然建立了patch级别的对应关系，但忽视了从整体角度理解这些匹配意味着什么。换言之，**识别时间变化本身不够，任务还需要能够无损地总结观察到的场景**。

### 组合式架构的效率问题

RSP通过组合掩码自编码、全局表示对齐和目标场景重建三个目标来实现全面能力，但计算开销超过竞争方法的2倍以上（32.5 vs 13.0-15.9 GFLOPs），性能/成本比不理想。

## 方法详解

### 整体框架

ToBo包含两个步骤：**压缩步骤**（Squeeze）将参考场景压缩为单个瓶颈token，**重建步骤**（Reconstruction）利用瓶颈token和极少量目标场景patch来预测目标场景。

### 关键设计

#### 1. 瓶颈token的压缩机制

给定参考场景 $\mathbf{x}^t$ 和目标场景 $\mathbf{x}^{t+k}$（时间间隔 $k$），两者patchify为 $N$ 个非重叠patch。

参考场景的所有patch送入编码器 $f_\theta$，得到空间表示 $\{\mathbf{u}_i^t\}$。编码过程中的CLS token输出作为**瓶颈token** $\mathbf{u}_{tobo}$，它被引导去紧凑地总结整个参考场景。

**设计精髓**：通过信息瓶颈——将整个场景压缩到单个token——迫使编码器必须保留最本质的信息，而非分散在多个token中。消融实验证实，瓶颈token数量为1时性能最佳（1 token均值61.1% vs 2 token 41.8% vs 4 token 36.1%）。

#### 2. 极高掩码率的目标场景重建

目标场景 $\mathbf{x}^{t+k}$ 以**极高掩码率 $r=0.9$** 进行掩码，仅保留约10%的patch作为提示。这些极少量的target patch经相同编码器处理后，与瓶颈token拼接，填充掩码token后送入解码器 $d_\phi$ 预测缺失patch。

**关键约束**：由于目标场景信息极度稀缺，解码器被迫高度依赖瓶颈token来完成重建。这带来两个学习效果：
- 瓶颈token必须保留参考场景的关键信息（否则无法重建目标）
- 这些信息必须以能识别时间动态的方式编码（与target hints交织时需理解场景变化）

训练损失为余弦距离：

$$\mathcal{L}_{\text{ToBo}} = \sum_{i \in \mathcal{M}} d(\hat{\mathbf{x}}_i^{t+k}, \mathbf{x}_i^{t+k})$$

#### 3. 纯自注意力解码器

与SiamMAE等使用交叉注意力层学习时间对应不同，ToBo的解码器仅使用自注意力层和MLP层。这确保解码器只关注给定信息（瓶颈token + 少量target patch）进行推理，不引入额外的跨注意力机制。

解码器由8个Transformer块组成，计算成本仅为15.9 GFLOPs，远低于RSP的32.5 GFLOPs。

### 训练策略

- 数据集：Kinetics-400，训练400 epoch（含2×重复采样即实际200 epoch）
- 架构：ViT-S/16（21.7M参数）
- 帧采样：30 FPS，时间间隔4-96帧
- 优化器：AdamW，batch size 1536
- 辅助目标：额外的孪生掩码自编码损失用于增强patch级对应学习

## 实验关键数据

### 机器人操控主实验（Franka Kitchen）

| 任务 | MAE | SiamMAE | RSP | CropMAE | **ToBo** | 提升 |
|------|-----|---------|-----|---------|---------|------|
| Knob1 on | 12.0 | 16.8 | 31.0 | 31.5 | **57.0** | +25.5 |
| Light on | 24.3 | 36.5 | 44.5 | 54.0 | **82.0** | +28.0 |
| Sdoor open | 71.5 | 68.0 | 82.5 | 77.0 | **95.0** | +12.5 |
| Ldoor open | 12.8 | 17.3 | 28.8 | 25.5 | **51.0** | +22.2 |
| Micro open | 10.0 | 13.5 | 30.3 | 32.5 | **55.0** | +22.5 |

### 真实机器人操控

| 方法 | Cabinet Opening | Drawer Closing | Cup Stacking |
|------|----------------|----------------|-------------|
| SiamMAE | 20.0 | 55.0 | 50.0 |
| RSP | 25.0 | 65.0 | 55.0 |
| CropMAE | 0.0 | 25.0 | 20.0 |
| **ToBo** | **65.0** | **75.0** | **80.0** |

### 消融实验

| 配置 | Knob1 on | Light on | Sdoor open | Ldoor open | Micro open | 均值 |
|------|----------|----------|------------|------------|------------|------|
| 1个瓶颈token | **46.7** | **78.7** | **95.3** | **47.3** | **37.3** | **61.1** |
| 2个 | 31.0 | 54.0 | 74.0 | 26.0 | 24.0 | 41.8 |
| 4个 | 28.0 | 24.3 | 78.0 | 28.0 | 22.0 | 36.1 |
| 8个 | 10.0 | 20.0 | 56.0 | 26.0 | 9.3 | 24.3 |

| 掩码率 | Knob1 on | Light on | Sdoor open | Ldoor open | Micro open |
|--------|----------|----------|------------|------------|------------|
| 0.50 | 14.0 | 24.0 | 70.0 | 16.0 | 14.0 |
| 0.75 | 26.0 | 60.0 | 79.0 | 28.0 | 22.0 |
| **0.90** | **46.7** | **78.7** | **95.3** | **47.3** | **37.3** |
| 0.95 | 34.0 | 66.0 | 86.0 | 38.0 | 26.0 |

### 关键发现

1. **单token最优**：瓶颈token数量从1增到8，性能持续下降（61.1% → 24.3%），证实极致压缩才能强制保留关键信息
2. **极高掩码率关键**：0.9是最优点，0.5和0.75下性能大幅下降，0.95过度剪枝也有退化，验证了"极度稀缺→强制依赖瓶颈"的设计逻辑
3. **真实环境泛化**：在Cabinet Opening任务上ToBo达65%成功率，而其他方法最高仅25%
4. **与视觉-语言模型的对比**：ToBo（21.7M参数，0.2B帧）超过CLIP（149M参数，12.8B帧）、DINOv2、SigLIP等，在每个Kitchen任务上至少领先13%
5. **可扩展性**：ViT-B/16和ViT-L/16上ToBo仍保持与基线的大幅领先

## 亮点与洞察

1. **信息瓶颈设计极简而高效**：单个token编码整个场景的思路违反直觉，但正是这种极致约束迫使表征保留最本质信息
2. **动机分析透彻**：清晰展示了MAE（无时间建模）→SiamMAE（有时间对应但无整体总结）→ToBo（压缩+时间推理一体化）的递进逻辑
3. **性能/成本比出色**：15.9 GFLOPs的训练计算量远低于RSP（32.5），但性能大幅领先
4. **实际部署验证**：真实物理机器人上的评估提供了强有力的实用性证据
5. **推理无额外开销**：推理时所有模型使用相同骨干和输入分辨率，FLOPs完全相同

## 局限与展望

- 多帧源输入的朴素扩展效果不佳（46.9% vs 61.1%），需要专门的多帧瓶颈设计
- 当前仅用于机器人操控和视频标签传播，未在更广泛的视频理解任务上验证
- 下游任务使用冻结骨干+简单MLP策略头，更复杂的策略网络是否受益相同有待验证
- 对训练数据的视频质量和时间间隔选择有一定敏感性（最佳间隔96帧）
- 瓶颈token的可解释性尚未深入分析

## 相关工作与启发

- **静态图像SSL**: MAE, SimMIM, DINO, SimCLR, MoCo v3
- **动态场景SSL**: SiamMAE, RSP, CropMAE
- **机器人表征学习**: VC-1 (MAE+Ego4D), MVP, R3M, Voltron, Theia
- **启发**: 信息瓶颈思想可推广至其他需要紧凑状态表示的sequential decision-making任务；极高掩码率策略可扩展到跨模态（如语音-视觉）的对应学习

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 单token瓶颈+极高掩码率的设计思路新颖且反直觉
- 实验充分度: ⭐⭐⭐⭐⭐ — 模拟+真实机器人+视频传播+多尺度+多消融的全面验证
- 写作质量: ⭐⭐⭐⭐⭐ — 动机推导清晰，逐层递进的baseline分析令人信服
- 价值: ⭐⭐⭐⭐⭐ — 小模型小数据超越大规模VLM，对机器人视觉表征学习有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Video-KTR: 通过关键 Token 归因增强视频推理](../../ICLR2026/video_understanding/video-ktr_reinforcing_video_reasoning_via_key_token_attribution.md)
- [\[CVPR 2025\] DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models](../../CVPR2025/video_understanding/divprune_diversity-based_visual_token_pruning_for_large_multimodal_models.md)
- [\[NeurIPS 2025\] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](../../ICCV2025/video_understanding/aim_adaptive_inference_of_multi_modal_llms_via_token_merging_and_pruning.md)
- [\[CVPR 2025\] Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks](../../CVPR2025/video_understanding/omni-rgpt_unifying_image_and_video_region-level_understanding_via_token_marks.md)

<!-- RELATED:END -->
