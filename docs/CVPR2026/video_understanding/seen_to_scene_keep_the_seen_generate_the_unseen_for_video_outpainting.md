---
title: >-
  [论文解读] Seen-to-Scene: Keep the Seen, Generate the Unseen for Video Outpainting
description: >-
  [CVPR 2026][视频理解][video outpainting] 提出 Seen-to-Scene，统一传播式和生成式范式的视频外推框架，通过参考帧引导的潜空间传播与视频扩散模型结合，在零样本推理中实现了超越需要输入特定适配的先进方法的时空一致性和视觉保真度。
tags:
  - "CVPR 2026"
  - "视频理解"
  - "video outpainting"
  - "propagation"
  - "扩散模型"
  - "光流"
  - "temporal coherence"
---

# Seen-to-Scene: Keep the Seen, Generate the Unseen for Video Outpainting

**会议**: CVPR 2026  
**arXiv**: [2604.14648](https://arxiv.org/abs/2604.14648)  
**代码**: [github.com/InSeokJeon/Seen_to_Scene](https://github.com/InSeokJeon/Seen_to_Scene)  
**领域**: 视频理解/生成  
**关键词**: video outpainting, propagation, diffusion model, optical flow, temporal coherence

## 一句话总结

提出 Seen-to-Scene，统一传播式和生成式范式的视频外推框架，通过参考帧引导的潜空间传播与视频扩散模型结合，在零样本推理中实现了超越需要输入特定适配的先进方法的时空一致性和视觉保真度。

## 研究背景与动机

视频外推需在保持空间保真度和时间一致性的同时扩展帧边界外的内容。现有方法两极分化：传播式方法依赖光流传播已知内容但计算成本高且无法合成不可见区域；生成式方法利用扩散模型强大的生成能力但隐式时间建模导致帧间不一致，有限空间线索导致幻觉内容。将两种范式的互补优势统一到单一框架是开放挑战。

## 方法详解

### 整体框架

给定输入视频，选择参考帧并估计光流，流补全网络补全外推区域的流场。输入帧编码到潜空间后，利用补全流进行参考帧引导的潜空间传播。轻量精修模块缓解传播伪影后，传播潜码作为条件送入 3D U-Net 扩散去噪过程。VAE 解码器重建最终帧。

### 关键设计

1. **参考帧引导潜空间传播**: 基于帧间结构相关度（SSIM 结构分量）在滑动窗口中选择内容丰富的参考帧。传播在潜空间而非像素空间进行，避免逐帧密集传播的高计算开销。通过累积参考帧间流场实现直接长距离传播。

2. **流补全网络域适应**: 首次分析视频外推中的流补全域差距——视频修补训练的流补全网络在大面积外推区域表现不佳。通过在端到端管线中联合微调预训练流补全网络适应外推域。

3. **潜码精修模块**: 轻量化模块通过预测残差采样偏移和自适应调制权重选择性调整不确定区域的潜码对齐。双向对齐融合减少对流场的过度依赖和局部错位。

### 损失函数 / 训练策略

微调预训练视频扩散模型（AnimateDiff），冻结卷积和空间注意力层保留空间先验，仅训练时间 Transformer 块。传播潜码与真实潜码沿通道维拼接作为去噪输入。仅用 100K 视频样本（YouTube-VOS）训练。

## 实验关键数据

### 主实验

| 方法 | 适配方式 | DAVIS PSNR↑ | DAVIS FVD↓ | YT-VOS LPIPS↓ |
|------|---------|------------|-----------|--------------|
| M3DDM | zero-shot | 低 | 高 | 高 |
| MOTIA | one-shot | 中 | 中 | 中 |
| Follow-Canvas | zero-shot | 中 | 中 | 中 |
| **Seen-to-Scene** | **zero-shot** | **21.95** | **218.8** | **最优** |

在 DAVIS 和 YouTube-VOS 上所有指标均优于先前方法，包括需要输入特定适配的 one-shot 方法。

### 消融实验

- 流补全域适应对外推区域的流场质量至关重要
- 参考帧选择策略优于固定间隔采样
- 潜码精修模块显著减少了传播伪影

### 关键发现

- 传播潜码为扩散过程提供的空间线索显著减少了内容幻觉
- 仅用 100K 公开数据即可实现强泛化的零样本外推
- 潜空间传播比像素空间传播效率高出数量级

## 亮点与洞察

- 传播+生成的统一框架自然结合了两种范式的优势
- 零样本性能超越 one-shot 方法，消除了输入特定适配的部署障碍
- 流补全域差距的首次系统分析填补了文献空白

## 局限与展望

- 依赖光流估计的准确性，极端运动场景可能退化
- 推理时需要计算参考帧链和流场的额外开销
- 非常大的扩展比（如 0.66 遮罩率）下仍存在质量挑战

## 相关工作与启发

- 传播-生成统一范式可应用到视频修补、视频外推等相关任务
- 参考帧选择策略对视频处理流水线有通用参考价值
- 域适应流补全的思路可推广到其他跨域迁移场景

## 评分

7/10 — 统一框架设计出色，实验全面，以零样本方式超越 one-shot 方法有说服力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)
- [\[CVPR 2026\] SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](savax_egotoexo_imitation_error_detection_via_scene.md)
- [\[CVPR 2025\] HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation](../../CVPR2025/video_understanding/hyperglm_hypergraph_for_video_scene_graph_generation_and_anticipation.md)
- [\[ACL 2026\] Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](../../ACL2026/video_understanding/response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)
- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](../../CVPR2025/video_understanding/egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)

</div>

<!-- RELATED:END -->
