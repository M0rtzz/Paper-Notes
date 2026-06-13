---
title: >-
  [论文解读] Seen-to-Scene: Keep the Seen, Generate the Unseen for Video Outpainting
description: >-
  [CVPR 2026][视频理解][video outpainting] 提出 Seen-to-Scene，统一传播式和生成式范式的视频外推框架，通过参考帧引导的潜空间传播与视频扩散模型结合，在零样本推理中实现了超越需要输入特定适配的先进方法的时空一致性和视觉保真度。 视频外推需在保持空间保真度和时间一致性的同时扩展帧边界外…
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

视频外推要在扩展帧边界外内容的同时保住空间保真和时间一致，而现有两条路各有短板：传播式靠光流搬已知内容、但算得贵又造不出不可见区域；生成式靠扩散模型能「无中生有」、但隐式时间建模导致帧间不一致、空间线索少又爱幻觉。Seen-to-Scene 把两条路拼成一个统一框架——用传播把「看得见的」稳稳搬过去，用扩散把「看不见的」生成出来。

具体流转：给定输入视频，先选参考帧并用 RAFT 估计光流，由流补全网络把外推区域的流场补全；输入帧经 VAE 编码进潜空间后，用补全的流做参考帧引导的潜空间传播；潜码精修模块缓解传播伪影后，传播潜码作为条件送进 3D U-Net 的扩散去噪过程；最后 VAE 解码器重建出帧。下面按数据流向（流补全 → 潜空间传播 → 潜码精修）拆解三个关键设计。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入视频<br/>RAFT 估计光流"] --> B["流补全网络的域适应<br/>端到端微调，补全大面积外推区流场"]
    B --> C["VAE 编码进潜空间"]
    C --> D["参考帧引导的潜空间传播<br/>SSIM 选参考帧 + 累积流场长距离搬运"]
    D --> E["潜码精修模块<br/>残差采样偏移 + 双向对齐融合"]
    E --> F["3D U-Net 扩散去噪<br/>传播潜码作条件，生成不可见区"]
    F --> G["VAE 解码 → 输出视频"]
```

### 关键设计

**1. 流补全网络的域适应：补外推区的流，不能直接拿修补任务的网络来用**

传播要靠光流把已知内容搬到外推区，而流场本身得先补全。本文首次系统指出视频外推里存在流补全的域差距——用视频修补（inpainting）训练的流补全网络（FCNet），面对大面积外推区域时表现不佳：它只在局部小遮罩里估运动，离源区越远预测的流越不稳，动态物体边缘还会「流溢出」。解法是把预训练流补全网络放进端到端管线里联合微调，让它适应外推域，从而给下游传播提供更可靠的流场。

**2. 参考帧引导的潜空间传播：在潜空间里把「看得见的」长距离搬过去**

逐帧在像素空间密集传播又慢、又只能搬相邻帧（代价随分辨率和帧数飙升）。Seen-to-Scene 改在潜空间做传播：先按帧间结构相关度（灰度图上的 SSIM 结构分量）在滑动时间窗里挑内容丰富的参考帧，再沿累积的参考帧间流场实现直接的长距离传播，把已知内容稳定地铺到外推区域，省去了逐帧顺序合成流场。潜空间传播比像素空间高出数量级的效率，传过来的潜码又给后续扩散提供了可靠的空间线索，显著压住了内容幻觉。

**3. 潜码精修模块：把传播带来的错位和伪影轻量地修回来**

潜空间传播难免有伪影和局部错位，又因传播直接接进生成过程，若全盘信任流场会把误差带进扩散。这个轻量模块以传播潜码、原始潜码、外推掩码和传播掩码为输入，靠外观线索（而非运动信息）对不确定区域预测残差采样偏移和自适应调制权重，有选择地调整潜码对齐；对齐双向进行，再用一层浅卷积融合两个方向的潜码，从而减少对流场的过度依赖、在送入扩散前先把传播潜码理顺。

### 损失函数 / 训练策略

微调预训练视频扩散模型（AnimateDiff），冻结卷积和空间注意力层以保留空间先验，只训时间 Transformer 块；传播潜码与真实潜码沿通道维拼接作为去噪输入。仅用 100K 视频样本（YouTube-VOS）训练。

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
