---
title: >-
  [论文解读] Revisiting Model Stitching in the Foundation Model Era
description: >-
  [CVPR 2025][多模态][model stitching] 系统研究异构 Vision Foundation Model（如 CLIP、DINOv2、SigLIP 2）之间的 stitchability，发现用 Final Feature Matching 预训练 stitch layer 可实现可靠拼接，并提出 VFM Stitch Tree 架构实现多 VFM 的高效共享。
tags:
  - CVPR 2025
  - 多模态
  - model stitching
  - vision foundation model
  - VFM
  - representation alignment
---

# Revisiting Model Stitching in the Foundation Model Era

**会议**: CVPR 2025  
**arXiv**: [2603.12433](https://arxiv.org/abs/2603.12433)  
**代码**: 待确认  
**领域**: multimodal_vlm  
**关键词**: model stitching, vision foundation model, VFM, representation alignment, multimodal LLM

## 一句话总结

系统研究异构 Vision Foundation Model（如 CLIP、DINOv2、SigLIP 2）之间的 stitchability，发现用 Final Feature Matching 预训练 stitch layer 可实现可靠拼接，并提出 VFM Stitch Tree 架构实现多 VFM 的高效共享。

## 研究背景与动机

**领域现状**: VFM 已成为视觉任务的默认 backbone，不同 VFM（CLIP、DINOv2、SigLIP 2 等）在不同任务上各有所长，现代多模态系统越来越多地同时使用多个 VFM。

**现有痛点**: 同时部署 $k$ 个 VFM 需要 $k\times$ 的计算和内存开销；不清楚这些异构 VFM 的内部表示是否兼容可复用。

**核心矛盾**: 先前 model stitching 研究仅在同数据集小模型上验证（如 ResNet-18 on CIFAR-10），VFM 时代下训练目标、数据、模态混合完全不同的大模型是否仍可拼接未知。

**本文目标**: 异构 VFM 是否可拼接？如何正确训练 stitch layer？是否能从拼接中获得超越单模型的性能？

**切入角度**: 设计系统化的实验协议（stitch point × stitch layer family × training loss × downstream task），揭示现有方法的失败模式并提出新方案。

**核心 idea**: 用 Final Feature Matching 匹配 target 模型倒数第二层特征来初始化 stitch layer，使异构 VFM 可靠拼接并融合互补知识。

## 方法详解

### 整体框架

给定 source VFM $f_\theta$ 和 target VFM $f_\phi$，在第 $n$ 层插入 stitch layer $S$，构建拼接模型：
$$F(x) = T_\phi^N \circ S \circ R_\theta^n(x)$$
其中 $R_\theta^n$ 取 source 前 $n$ 层特征，$T_\phi^N$ 取 target 后 $N-n$ 层，仅 $S$ 可训练。

### 关键设计

**1. Layer Feature Matching (LFM) 的失败分析**
- **功能**: 训练 stitch layer 最小化拼接点处的特征差异 $\|S(R_\theta^n(x)) - R_\phi^n(x)\|_2^2$。
- **核心思路**: 虽然 layer 特征距离很低（$10^{-3}$ 量级），但最终特征距离很高，尤其在浅层拼接时。
- **设计动机**: 小的中间层 mismatch 会被冻结的后续层累积放大，导致输出特征严重偏离。

**2. Final Feature Matching (FFM)**
- **功能**: 训练 stitch layer 匹配 target 模型最终层（pre-logit）的 patch 特征。
- **核心思路**: $\mathcal{L}_{\text{FFM}} = \frac{1}{M}\sum_{i=1}^{M} \|T_\phi^N(S(R_\theta^n(x_i))) - T_\phi^N(R_\phi^n(x_i))\|_2^2$，注意 FFM 是 label-free 的。
- **设计动机**: 直接约束最终输出，消除误差累积问题；意外发现 FFM 同时在拼接点处保持了低特征距离，说明最终层监督能隐式诱导中间层对齐。

**3. 两阶段训练策略**
- **功能**: (i) FFM 预训练 stitch layer → (ii) 用下游 task loss 微调。
- **核心思路**: 解决 Task Loss Training (TLT) 在浅层拼接时的梯度消失问题——梯度需穿过大量冻结层才能到达 stitch layer，FFM 预训练提供好的初始化绕过这一优化困难。
- **设计动机**: TLT 在 DINOv2→SigLIP2 第 2 层仅 25.1% accuracy，FFM 初始化后提升至 51.7%。

**4. Self-Stitch 基线设计**
- **功能**: 在同一模型内插入相同 stitch layer（如 DINOv2→DINOv2），排除 stitch layer 容量带来的伪提升。
- **核心思路**: 如果跨模型拼接优于同模型自拼接，说明确实发生了互补知识融合。
- **设计动机**: 严格的对照实验设计，隔离了 stitch layer 容量 vs. 真正的知识融合。

### 损失函数 / 训练策略

- Stage 1: $\mathcal{L}_{\text{FFM}}$（label-free，可预提取特征离线训练）
- Stage 2: $\mathcal{L}_{\text{task}}$（cross-entropy 分类或 mIoU 分割）
- Stitch layer 默认为 2 层 MLP + ReLU

## 实验关键数据

### 主实验

fMoW 分类精度（%），两阶段 FFM+TLT:

| 拼接方向 | Layer 2 | Layer 6 | Layer 10 | Layer 14 | Layer 18 | Layer 22 |
|---|---|---|---|---|---|---|
| DINOv2→SigLIP2 | 51.7 | 55.8 | 59.3 | 68.0 | 72.0 | 71.8 |
| SigLIP2→DINOv2 | 53.8 | 53.8 | 61.9 | 69.6 | 70.4 | 72.2 |

多数据集验证（Layer 6/14/22）:

| 方向 | fMoW | iNaturalist | Aircraft | ADE20K(mIoU) |
|---|---|---|---|---|
| DINOv2→DINOv2 (self) | 41.5/59.7/69.9 | 56.9/81.5/91.2 | 37.8/79.3/91.2 | 35.4/50.9 |
| SigLIP2→SigLIP2 (self) | 50.5/62.0/68.9 | 71.2/88.5/87.3 | 67.9/88.1/89.3 | 44.5/50.5 |
| DINOv2→SigLIP2 | **55.8/68.0/71.8** | **75.9/89.1/92.8** | **77.8/87.6/92.4** | **44.9/51.2** |
| SigLIP2→DINOv2 | **53.8/69.6/72.2** | **86.3/88.9/91.9** | **80.7/89.0/91.0** | **49.0/51.4** |

### 消融实验

Stitch layer 选择（fMoW, DINOv2→SigLIP2, Layer 22）:

| Stitch Layer | Acc(%) |
|---|---|
| Linear | 69.6 |
| MLP | **71.8** |
| LoRA | 67.3 |

FFM 初始化 vs 无初始化（TLT, DINOv2→SigLIP2）:

| Layer | 无 FFM | 有 FFM |
|---|---|---|
| 2 | 25.1 | **51.7** |
| 6 | 39.4 | **55.8** |
| 22 | 68.6 | **71.8** |

### 关键发现

1. **跨模型拼接稳定优于自拼接**: 在所有数据集和任务上，拼接模型比 self-stitch 基线高 +0.7% ~ +5.5%，证明了互补知识融合的存在。
2. **FFM 初始化是关键**: 浅层拼接时 TLT 严重失败（25.1%），FFM 预训练完全恢复并超越 linear probing。
3. **弱 source 会拖累**: 当 CLIP（较弱 VFM）作为 source 时，拼接模型无法匹配强 target 的性能；但作为 target 时性能良好。
4. **MLP 优于 LoRA**: 可能因为适度的 mismatch 反而有助互补信息融合。
5. **VFM Stitch Tree 实用性**: VST-22（4.3% 额外资源）恢复 45% 的双 VFM 增益；VST-14（39% 额外资源）恢复 84%。

## 亮点与洞察

- 将 model stitching 从诊断工具升级为实用工具，证明异构 VFM 可靠拼接并融合互补知识
- FFM 的发现非常深入：最终层监督能隐式诱导中间层对齐，且是 label-free 的
- Self-stitch 基线设计严谨，有效排除了容量伪提升的解释
- VFM Stitch Tree 为多 VFM 部署提供了连续的性能-效率权衡 knob

## 局限与展望

- VFM Stitch Tree 仅在 VQAv2 和 MME 上验证，更广泛的多模态评估待确认
- 仅考虑相同架构（ViT）的 VFM，不同架构间的拼接未探索
- FFM 需要运行 target 模型的完整前向传播，训练效率有提升空间
- 未探索动态选择拼接点或根据输入自适应使用不同 VFM 分支

## 相关工作与启发

- Bansal et al. (2021) 提出 model stitching 的"Anna Karenina 假说"：成功模型学到相似表示；本文在 VFM 层面验证并扩展了该结论
- SN-Net 设计可拼接的网络族用于弹性推理；本文是 post-hoc 拼接独立训练的异构 VFM
- 启发：VFM 的浅层可能编码预训练特定特征而深层更具迁移性，可用于指导高效推理架构设计

## 评分

- **新颖性**: ⭐⭐⭐⭐ FFM + 两阶段训练的发现有深度，VFM Stitch Tree 有实用价值
- **实验充分度**: ⭐⭐⭐⭐⭐ 极其系统——多数据集、多任务、多 VFM、多 stitch layer 类型，self-stitch 对照设计严谨
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，failure mode 分析透彻，实验设计层层递进
- **价值**: ⭐⭐⭐⭐ 对 VFM 表示理解和多 VFM 高效部署都有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Evaluating Model Perception of Color Illusions in Photorealistic Scenes](evaluating_model_perception_of_color_illusions_in_photorealistic_scenes.md)
- [\[CVPR 2025\] EgoLM: Multi-Modal Language Model of Egocentric Motions](egolm_multi-modal_language_model_of_egocentric_motions.md)
- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](collm_a_large_language_model_for_composed_image_retrieval.md)
- [\[CVPR 2025\] Vision-Language Model IP Protection via Prompt-based Learning](vision-language_model_ip_protection_via_prompt-based_learning.md)
- [\[CVPR 2025\] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector](rethinking_vision-language_model_in_face_forensics_multi-modal_interpretable_for.md)

<!-- RELATED:END -->
