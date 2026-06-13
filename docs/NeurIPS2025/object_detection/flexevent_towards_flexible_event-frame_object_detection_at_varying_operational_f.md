---
title: >-
  [论文解读] FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies
description: >-
  [NeurIPS 2025][目标检测][事件相机] 提出 FlexEvent 框架，通过自适应事件-图像融合模块 FlexFuse 和频率自适应微调机制 FlexTune，实现事件相机在不同操作频率下的灵活目标检测，在 20Hz 到 180Hz 范围内保持鲁棒性能，显著超越现有方法。 事件相机以微秒级时间分辨率和异步工作方…
tags:
  - "NeurIPS 2025"
  - "目标检测"
  - "事件相机"
  - "event-frame fusion"
  - "frequency adaptation"
  - "自训练"
---

# FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies

**会议**: NeurIPS 2025  
**arXiv**: [2412.06708](https://arxiv.org/abs/2412.06708)  
**作者**: Dongyue Lu, Lingdong Kong, Gim Hee Lee, Camille Simon Chane, Wei Tsang Ooi (NUS, CNRS, CY Cergy Paris University)
**代码**: [flexevent.github.io](https://flexevent.github.io)  
**领域**: 目标检测 / 事件相机 / 多模态融合  
**关键词**: 事件相机, 目标检测, event-frame fusion, frequency adaptation, 自训练

## 一句话总结

提出 FlexEvent 框架，通过自适应事件-图像融合模块 FlexFuse 和频率自适应微调机制 FlexTune，实现事件相机在不同操作频率下的灵活目标检测，在 20Hz 到 180Hz 范围内保持鲁棒性能，显著超越现有方法。

## 研究背景与动机

事件相机以微秒级时间分辨率和异步工作方式，在动态环境中具有独特优势。然而现有事件检测器存在两个核心限制：

**固定频率范式**：大多数方法将事件数据与低频帧率对齐，采用固定时间间隔处理事件流，忽略了高频事件流中丰富的时间细节。当需要在动态环境中进行高频检测时，性能急剧下降。

**语义信息不足**：纯事件方法缺乏 RGB 帧提供的空间和语义信息；而现有的事件-帧融合方法虽有改善，但在不同操作频率的适应性上仍然不足。

核心挑战在于：高频事件数据的标注极其昂贵（需要大量人工），且现有融合方法无法有效平衡不同模态在不同频率下的贡献。例如经典的 RVT 检测器在操作频率从 20Hz 提升到更高时，性能显著下降。

## 方法详解

### 整体框架

FlexEvent 由两个关键组件构成：

- **FlexFuse**：自适应事件-帧融合模块，将高频事件数据与 RGB 帧的丰富语义信息动态融合
- **FlexTune**：频率自适应微调机制，通过生成频率调整标签实现跨频率泛化

### 事件数据表示

事件相机在像素 $(x,y)$ 处检测到对数亮度变化超过阈值 $C$ 时产生事件 $e=(x,y,t,p)$，其中 $p \in \{-1,1\}$ 为极性。事件流被预处理为 4D 张量 $E(p,\tau,x,y)$，维度为 $[2, T, H, W]$，通过时间离散化将连续事件映射到 $T$ 个时间片中，便于后续卷积处理。

### FlexFuse：自适应事件-帧融合

**动态事件聚合**：给定频率 $a$ 的标注数据和对应帧数据，将时间间隔 $\Delta t^a$ 划分为 $b/a$ 个子区间（$b > a$），从中随机采样一个高频事件集合 $\mathbf{E}^b$。这种策略在训练时引入毫秒级时间抖动作为隐式时间增强，增强对实际同步噪声的鲁棒性。

**特征提取**：采用双分支架构：
- 事件分支 $\phi_E(\cdot)$：基于 RVT 提取事件特征
- 帧分支 $\phi_F(\cdot)$：基于 ResNet-50 提取 RGB 特征

两个分支均为四阶段结构，在每个尺度 $i$ 提取事件特征 ${}^{(i)}\mathbf{h}_E^a, {}^{(i)}\mathbf{h}_E^b$ 和帧特征 ${}^{(i)}\mathbf{h}_F$。

**自适应门控融合**：在每个尺度 $i$，先拼接事件和帧特征 ${}^{(i)}\mathbf{h}_{\text{shared}}^a = [{}^{(i)}\mathbf{h}_E^a,\ {}^{(i)}\mathbf{h}_F]$，然后通过带噪声的门控函数计算自适应软权重：

$$[\alpha, \beta] = \text{Softmax}((\mathbf{h}_{\text{shared}} \cdot \mathbf{W}) + \sigma \cdot \epsilon)$$

其中 $\mathbf{W}$ 是可训练权重矩阵，$\sigma$ 是学习到的标准差控制噪声扰动幅度，$\epsilon \sim \mathcal{N}(0,1)$ 为高斯噪声。融合特征通过逐元素加权得到：

$$\mathbf{h}_{\text{fuse}}^a = \alpha \odot \mathbf{h}_E^a + \beta \odot \mathbf{h}_F$$

最终将不同频率的融合特征相加：$\mathbf{h}_{\text{fuse}} = \mathbf{h}_{\text{fuse}}^a + \mathbf{h}_{\text{fuse}}^b$，多尺度特征级联后送入检测头。

**正则化**：引入变异系数惩罚项防止模型过拟合到单一模态：

$$\mathcal{L}_{\text{fuse}} = \mathcal{L}_{\text{det}} + \lambda \left(\frac{\text{Var}(\alpha)}{(\mathbb{E}[\alpha])^2} + \frac{\text{Var}(\beta)}{(\mathbb{E}[\beta])^2}\right)$$

### FlexTune：频率自适应微调

FlexTune 分为两个主要阶段：

**阶段1：低频稀疏训练**：在高频率 $b$ 下训练，但仅使用最后一个事件（对应标注时间戳），使模型在利用低频标签的同时捕获高频时间信息。

**阶段2：跨频率传播**：包含三个步骤：

1. **高频引导**（High-Frequency Bootstrapping）：用预训练模型对完整高频事件集生成伪标签 $\tilde{\mathbf{y}}$。

2. **时间一致性校准**（Temporal Consistency Calibration）：
    - 双向事件增强：正向和反向处理事件流以增强召回
    - 置信度感知过滤：应用 NMS 和低置信度阈值 $\tau$ 消除重复并保留高潜力检测
    - 轨迹修剪：通过 IoU 跟踪关联跨帧检测，修剪短轨迹以抑制瞬态噪声

3. **循环自训练**（Cyclic Self-Training）：迭代训练，总损失函数为：

$$\mathcal{L}_{\text{tune}} = \mathcal{L}_{\text{GT}} + \beta \sum \mathcal{L}_{\text{det}}(\tilde{\mathbf{y}}, \hat{\mathbf{y}})$$

## 实验与结果

### 实验设置

在三个大规模数据集上验证：
- **DSEC-Det**：78,344帧，60个序列，8个类别（主要基准）
- **DSEC-Detection**：52,727帧，41个序列，3个类别
- **DSEC-MOD**：13,314帧，16个序列，1个类别

训练100K迭代，batch size=8，序列长度=11，学习率1e-4，两块 A5000 GPU 约一天完成训练。

### 主要结果

| 数据集 | 指标 | 前最优 | FlexEvent | 提升 |
|--------|------|--------|-----------|------|
| DSEC-Det | mAP | 41.9 (DAGr-50) | **57.4** | +15.5% |
| DSEC-Detection | Avg mAP | 38.0 (CAFR) | **47.4** | +9.4% |
| DSEC-MOD | Avg mAP | 29.0 (RENet) | **36.9** | +7.9% |

在 DSEC-Det 上的完整指标：mAP 57.4、AP50 78.2、AP75 66.6、APS 51.7、APM 64.9、APL 83.7，全面超越所有基线方法。

### 高频泛化能力

FlexEvent 在频率变化时的鲁棒性极为突出：

| 频率 | 20Hz | 36Hz | 45Hz | 60Hz | 90Hz | 180Hz | 平均 |
|------|------|------|------|------|------|-------|------|
| 无 FlexFuse/FlexTune | 53.2 | 52.0 | 49.4 | 45.9 | 38.8 | 22.9 | 43.7 |
| 完整 FlexEvent | **57.4** | **60.1** | **59.5** | **58.8** | **56.5** | **50.9** | **57.2** |

- 从 20Hz 到 90Hz 仅损失 ~1.5% 性能（保持 96.2%）
- 在 180Hz 极端条件下仍达 50.9% mAP（基线仅 22.9%）

### 推理效率

| 方法 | 参数量 | 20Hz | 90Hz | 180Hz |
|------|--------|------|------|-------|
| RVT | 18.5M | 9.20ms | 7.19ms | 6.77ms |
| DAGr-50 | 34.6M | 73.35ms | 45.29ms | 43.89ms |
| FlexEvent | 45.4M | 14.27ms | 12.47ms | 12.37ms |

虽然参数量较大，但推理速度与 SAST 相当，远快于 DAGr。FlexTune 为离线操作，不引入运行时开销。

### 消融分析

- **FlexFuse 贡献**：仅添加帧信息使平均 mAP 从 43.7% 提升到 56.4%，高频增益更显著
- **FlexTune 贡献**：180Hz 下 mAP 从 22.9% 提升到 30.4%（无 FlexFuse 时）；与 FlexFuse 联合使用从 49.2% 提升到 50.9%
- **融合策略对比**：自适应门控优于简单 Add、Concat 和 Vanilla Attention
- **插值标签 vs FlexTune**：线性插值标签在快速出现/消失的物体上表现不佳，FlexTune 通过时间一致性校准生成更准确的伪标签

## 亮点与洞察

- **频率灵活性**：首次明确解决事件相机在不同操作频率下的检测问题，从 20Hz 到 180Hz 均保持高精度，在实际部署中具有重要意义——无需为不同场景训练多个模型
- **优雅的融合设计**：带噪声的自适应门控机制简洁有效，通过学习到的软权重动态平衡事件和帧模态的贡献，变异系数正则化防止模态坍塌
- **伪标签的质量保证**：FlexTune 中的时间一致性校准（双向增强 + 轨迹修剪）确保高频伪标签的可靠性，避免自训练中常见的噪声累积问题
- **实用性强**：FlexTune 为离线步骤，不增加推理开销；整体框架在两块 A5000 上一天训练完成，具有良好的可复现性

## 局限与展望

- **RGB 帧依赖**：在极端光照（如完全黑暗）下 RGB 帧质量下降，融合可能反而引入噪声；纯事件模式的 fallback 机制值得探索
- **伪标签上界**：FlexTune 的高频伪标签质量受限于教师模型在低频上的初始性能，存在天然的性能上界
- **类别范围有限**：DSEC 数据集主要包含驾驶场景（车辆、行人），在更多样的目标类别和场景（室内、工业）上的泛化性未验证
- **计算开销**：45.4M 参数量在嵌入式平台上的部署挑战未讨论，实际自动驾驶应用需要更轻量的变体
- **时间一致性假设**：轨迹修剪假设物体运动较为平滑，在极端场景（突然出现的遮挡、急刹车）下可能失效

## 相关工作与启发

- **RVT (CVPR'23)**：基于 Transformer 的事件检测器，是事件分支的基础架构，但固定频率范式限制了高频性能
- **DAGr (Nature'24)**：最新的事件-帧融合方法，以图注意力网络为核心，本文在 DSEC-Det 上超越其 +15.5% mAP
- **CAFR (ECCV'24)**：跨注意力融合方法，本文在 DSEC-Detection 上超越其 +9.4%
- **LEOD (CVPR'24)**：标签高效的事件检测先驱，但未解决高频泛化
- **SSM (CVPR'24)**：基于状态空间模型的频率适应方法，但纯事件模式在高频下难以检测静态物体

## 评分 ⭐

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention](../../CVPR2025/object_detection/efficient_event-based_object_detection_a_hybrid_neural_network_with_spatial_and_.md)
- [\[CVPR 2025\] Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset](../../CVPR2025/object_detection/object_detection_using_event_camera_a_moe_heat_conduction_based_detector_and_a_n.md)
- [\[ICCV 2025\] EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision](../../ICCV2025/object_detection/evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)
- [\[NeurIPS 2025\] Test-Time Adaptive Object Detection with Foundation Model](test-time_adaptive_object_detection_with_foundation_model.md)
- [\[NeurIPS 2025\] ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)

</div>

<!-- RELATED:END -->
