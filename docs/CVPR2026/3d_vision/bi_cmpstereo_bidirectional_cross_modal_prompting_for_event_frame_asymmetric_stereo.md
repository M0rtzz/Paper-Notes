---
title: >-
  [论文解读] Bi-CMPStereo: Bidirectional Cross-Modal Prompting for Event-Frame Asymmetric Stereo
description: >-
  [CVPR 2026][3D视觉][event camera] 提出 Bi-CMPStereo，一种双向跨模态提示框架，交替将事件和帧设为目标域进行立体规范化约束和跨域嵌入适配，同时利用两个方向的代价体实现鲁棒的事件-帧非对称立体匹配。
tags:
  - "CVPR 2026"
  - "3D视觉"
  - "event camera"
  - "stereo matching"
  - "跨模态"
  - "asymmetric stereo"
  - "深度估计"
---

# Bi-CMPStereo: Bidirectional Cross-Modal Prompting for Event-Frame Asymmetric Stereo

**会议**: CVPR 2026  
**arXiv**: [2604.15312](https://arxiv.org/abs/2604.15312)  
**代码**: [github.com/xnh97/Bi-CMPStereo](https://github.com/xnh97/Bi-CMPStereo)  
**领域**: 3D视觉  
**关键词**: event camera, stereo matching, cross-modal, asymmetric stereo, depth estimation

## 一句话总结

提出 Bi-CMPStereo，一种双向跨模态提示框架，交替将事件和帧设为目标域进行立体规范化约束和跨域嵌入适配，同时利用两个方向的代价体实现鲁棒的事件-帧非对称立体匹配。

## 研究背景与动机

事件相机的高时间分辨率和高动态范围与帧相机的丰富上下文信息互补，使事件-帧非对称立体在高速运动和极端光照下很有前景。然而模态差距严重：现有方法要么通过域级对齐（统一表示+Siamese 特征提取）要么通过特征级对齐（独立编码器+共享嵌入）来缓解，但都可能边缘化域特有的判别性线索。关键挑战是学习表达性表示而不进行信息损失的边缘化。

## 方法详解

### 整体框架

交替将非对称立体输入设为目标域和源域。CMPStereo 在目标域的规范空间中学习对齐的立体表示。实例化两个互补配置（evCMPStereo 以事件为目标，imgCMPStereo 以图像为目标），Bi-CMPStereo 同时利用两方向的代价体实现鲁棒视差估计。

### 关键设计

1. **立体规范化约束 (SCC)**: 正则化网络从两种模态中学习目标域判别性特征，在目标域规范空间中实现高保真跨模态对齐。确保源域提取的特征也具有目标域的区分性表达。

2. **跨域嵌入适配器 (CDEA)**: 对源域中弱编码的目标域线索进行强化。通过轻量适配器在特征级别完成源到目标的初步适应，再由域特定编码器进一步提取。

3. **层次化视觉变换 (HVT)**: 从帧图像中提取上下文特征时采用 HVT 避免捷径学习，增强泛化性。级联 ConvGRU 迭代精修视差。双向代价体融合互补信息实现最终鲁棒估计。

### 损失函数 / 训练策略

迭代精修的视差损失，双向代价体各自产出视差后融合。SCC 约束作为正则化项在训练中施加。

## 实验关键数据

### 主实验

在 DSEC 和 MVSEC 基准上评估：

| 基准 | 指标 | 先前SOTA | Bi-CMPStereo |
|------|------|---------|-------------|
| DSEC | 各项指标 | 基线 | **显著超越** |
| MVSEC | 各项指标 | 基线 | **显著超越** |

在准确性和泛化性上均显著超越 SOTA。

### 消融实验

- 双向框架优于任一单向配置
- SCC 约束对跨模态特征质量提升关键
- CDEA 有效补充了源域中缺失的目标域线索

### 关键发现

- 交替设置目标域有效避免了信息边缘化
- 双向代价体提供了互补的匹配置信度
- 保留域特有线索比追求统一表示更有效

## 亮点与洞察

- "交替目标域"的双向设计理念新颖——不是寻找折中的共同空间，而是两个方向各自充分利用
- SCC 将立体匹配的几何约束与跨模态对齐结合
- HVT 防止模型走捷径直接用帧信息绕过事件信息

## 局限与展望

- 双向框架意味着两倍的计算开销
- 事件表示的选择（event concentration）可能不是最优的
- 仅在两个立体基准上验证

## 相关工作与启发

- 交替目标域框架可推广到其他跨模态融合任务
- SCC 的域提示思路借鉴了 NLP 中的 prompting
- 对事件-帧融合的系统性方案为神经形态传感器应用提供参考

## 评分

7/10 — 方法设计系统完整，实验改进显著，是非对称立体领域的有力推进。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis](affordgrasp_cross-modal_diffusion_for_affordance-aware_grasp_synthesis.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[CVPR 2026\] EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors](eventhub_data_factory_for_generalizable_event-based_stereo_networks_without_acti.md)
- [\[CVPR 2026\] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](cmhanet_a_cross-modal_hybrid_attention_network_for_point_cloud_registration.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](lite_any_stereo_efficient_zero-shot_stereo_matching.md)

</div>

<!-- RELATED:END -->
