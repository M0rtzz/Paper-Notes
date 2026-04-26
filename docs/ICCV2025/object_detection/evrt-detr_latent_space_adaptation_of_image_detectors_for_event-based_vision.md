---
title: >-
  [论文解读] EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision
description: >-
  [ICCV 2025][目标检测][事件相机] 提出I2EvDet框架，通过在冻结的RT-DETR检测器的潜空间中插入轻量级RNN时序模块，以最小的架构修改将主流图像检测器适配为事件相机视频检测模型，在Gen1和1Mpx基准上分别取得+2.3和+1.4 mAP的SOTA。
tags:
  - ICCV 2025
  - 目标检测
  - 事件相机
  - RT-DETR
  - 潜空间适配
  - 时序建模
---

# EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision

**会议**: ICCV 2025  
**arXiv**: [2412.02890](https://arxiv.org/abs/2412.02890)  
**代码**: https://github.com/realtime-intelligence/evrt-detr  
**领域**: Object Detection / Event-based Vision  
**关键词**: 事件相机, 目标检测, RT-DETR, 潜空间适配, 时序建模

## 一句话总结

提出I2EvDet框架，通过在冻结的RT-DETR检测器的潜空间中插入轻量级RNN时序模块，以最小的架构修改将主流图像检测器适配为事件相机视频检测模型，在Gen1和1Mpx基准上分别取得+2.3和+1.4 mAP的SOTA。

## 研究背景与动机

事件相机（EBC）是一种生物启发视觉传感器，相比传统帧式相机具有低功耗（约10 mW）、高时间分辨率（μs级）和高动态范围（>100 dB）的优势，在自动驾驶、机器人等领域广泛应用。然而，事件数据的稀疏异步特性给传统视觉方法带来了根本性挑战。

现有EBC目标检测方法主要沿两条路线发展：（1）设计复杂的事件数据表示（如Time Surface、ERGO-12）；（2）构建专用架构处理事件数据的时序特性（如RVT、S5-ViT）。这两条路线都假设EBC数据与传统视觉存在本质不兼容性，导致研究路径与主流计算机视觉日益分离。

本文提出一个截然不同的视角：利用主流目标检测器强大的特征提取能力，通过针对性适配而非全新架构设计来处理事件数据，从而弥合两个领域之间的鸿沟。

## 方法详解

### 整体框架

I2EvDet是一个通用的两阶段适配框架：第一阶段在简单的事件帧表示上直接训练图像检测器；第二阶段冻结检测器参数，在其编码器潜空间中插入轻量级RNN模块捕获时序动态。该框架适用于任何具有清晰特征提取/检测分离的检测模型（如YOLO系列、DETR系列）。

### 关键设计

1. **堆叠2D直方图事件表示**: 将事件流按 $T_{\text{frame}}=50$ ms的固定时间窗口划分为帧，每帧进一步细分为10个 $T_{\text{bin}}=5$ ms的间隔。构建中间堆叠直方图 $S(p, t_i, y, x)$，形状为 $(2, 10, H, W)$，合并极性和时间维度后得到 $(20, H, W)$ 的类图像表示。这种表示虽然简单，但为主流检测器提供了直接兼容的输入格式。

2. **潜空间时序适配模块（I2EvDet核心）**: 在冻结的RT-DETR编码器输出的多尺度特征 $\{\mathcal{E}_3, \mathcal{E}_4, \mathcal{E}_5\}$ 上，插入三个并行的ConvLSTM时序记忆模块 $\{\mathbf{R}_3, \mathbf{R}_4, \mathbf{R}_5\}$。每个模块通过残差连接与对应特征图交互:

   $$\mathcal{E}_i^{t,proj} = W_i^{down} \cdot \mathcal{E}_i^t$$
   $$(\mathcal{O}_i^{t,proj}, \mathcal{M}_i^t) = \mathbf{R}_i(\mathcal{E}_i^{t,proj}, \mathcal{M}_i^{t-1})$$
   $$\tilde{\mathcal{E}}_i^t = \mathcal{E}_i^t + \alpha_i \cdot W_i^{up} \cdot \mathcal{O}_i^{t,proj}$$

   其中 $W_i^{down}$ 和 $W_i^{up}$ 为投影矩阵，$\alpha_i$ 为可学习缩放因子（受ReZero技术启发）。这种设计在保持原始空间表示完整性的同时融入时序上下文。

3. **RNN vs Transformer的理论选择**: 事件相机面临独特挑战——静止物体不会产生事件因而对相机"不可见"，检测需要对历史事件的持久记忆。Transformer的固定上下文窗口限制了时序记忆长度，而RNN通过循环状态理论上具有无界记忆容量，特别适合在事件流中维持物体的存在性。

### 损失函数 / 训练策略

两阶段训练：第一阶段使用Adam优化器和EMA权重策略在单帧事件图像上标准训练RT-DETR；第二阶段冻结所有第一阶段参数，仅训练新插入的RNN模块。时序训练时在随机和顺序clip上混合训练，对顺序clip保持RNN记忆连续性，随机clip则重置状态。Gen1用21帧clip，1Mpx用10帧clip。

## 实验关键数据

### 主实验

| 模型 | Gen1 mAP(%) | 1Mpx mAP(%) | 参数量(M) | 推理时间(ms) |
|------|------------|-------------|----------|-------------|
| RVT-B | 47.2 | 47.4 | 18.5 | 10.2/11.9 |
| ERGO-12 | 50.4 | 40.6 | 59.6 | 69.9/100.0 |
| S5-ViT-B | 47.4 | 47.2 | 17.5 | 8.2/9.6 |
| SAST-CB | 48.2 | 48.7 | 18.9 | -/57.5 |
| RT-DETR-B(Stage1) | 47.6 | 45.2 | 42.8 | 10.5/14.9 |
| **EvRT-DETR-B** | **52.7** | **50.1** | 57.1 | 12.7/18.8 |
| **EvRT-DETR-T** | 52.3 | 49.9 | 34.4 | 8.4/12.5 |

### 消融实验

| 时序模块配置 $(x,y,z)$ | mAP(%) | mAP₅₀(%) | 说明 |
|----------------------|--------|----------|------|
| $(0,1,1)$ 无 $\mathcal{E}_3$ | 51.0 | 80.7 | 低层特征缺失影响最大 |
| $(1,0,1)$ | 52.2 | 81.3 | - |
| $(1,1,0)$ | 52.4 | 81.7 | - |
| $(1,1,1)$ 完整 | **52.7** | **82.0** | 多尺度时序适配最优 |

| 隐藏维度M | mAP(%) | 可训练参数(M) | 说明 |
|----------|--------|-------------|------|
| 64 (4x压缩) | 52.1 | 2.3 | 仅5.4%参数增加，已超所有先前SOTA |
| 128 | 52.5 | 5.4 | - |
| 256 (默认) | 52.7 | 14.4 | - |
| 512 | 52.9 | 42.9 | 收益递减 |

| 检测器 | 原始mAP(%) | I2EvDet适配后mAP(%) | 提升 |
|--------|----------|-------------------|------|
| RT-DETR-T | 46.0 | 52.3 | +6.3 |
| RT-DETR-B | 47.6 | 52.7 | +5.1 |
| YOLOX-T | 36.0 | 42.4 | +6.4 |
| YOLOX-X | 43.4 | 47.8 | +4.4 |

### 关键发现

- 仅用简单的堆叠直方图表示训练RT-DETR，就已与专用EBC方法性能相当，颠覆了"事件数据需要专用架构"的认知
- 低层特征（$\mathcal{E}_3$）从时序适配中获益最大，移除该层模块性能下降最显著
- 时序适配对静止物体检测至关重要：运动时RT-DETR和EvRT-DETR都表现良好，但当物体静止时只有EvRT-DETR能保持检测
- I2EvDet框架在不同架构（RT-DETR, YOLOX）上一致提升4.4-6.4 mAP，验证了框架通用性

## 亮点与洞察

- 研究思路非常优雅：与其为事件相机设计全新架构，不如适配已经被充分优化的主流检测器
- 类LoRA的参数高效适配（M=64仅需2.3M额外参数）就已超越所有先前方法，暗示事件相机领域可能一直在"过度工程化"
- 选择RNN而非Transformer作为时序模块的理论动机充分——事件相机的"静止即不可见"特性需要无界记忆

## 局限与展望

- 推理时间随时序模块增加，在极端实时场景下可能成为瓶颈
- 仅在自动驾驶场景的两个数据集上验证，其他事件相机应用场景（如高速运动捕捉）未涉及
- ConvLSTM的选择较为保守，可探索更先进的状态空间模型（如Mamba）替代

## 相关工作与启发

- 潜空间适配的思想与LoRA在NLP中的成功类似，验证了"冻结+轻量插件"范式的跨领域有效性
- I2EvDet框架可推广到其他时序视觉任务（如视频目标检测、光流估计等）
- 为事件相机领域提供了"站在巨人肩上"的新研究范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 主流检测器→事件相机的适配思路新颖，潜空间RNN插入方案优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 两大标准数据集SOTA，跨架构验证，参数效率分析完善
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述严谨，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ 为事件相机视觉开辟了基于主流架构适配的全新研究方向

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[ICCV 2025\] Sim-DETR: Unlock DETR for Temporal Sentence Grounding](sim-detr_unlock_detr_for_temporal_sentence_grounding.md)
- [\[ICCV 2025\] DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion](distil_data-free_inversion_of_suspicious_trojan_inputs_via_latent_diffusion.md)
- [\[NeurIPS 2025\] FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies](../../NeurIPS2025/object_detection/flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)
- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)

<!-- RELATED:END -->
