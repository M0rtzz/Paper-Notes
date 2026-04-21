---
title: >-
  [论文解读] DeDelayed: Deleting Remote Inference Delay via On-Device Correction
description: >-
  [CVPR 2026][图像分割][协同推理] 提出 DeDelayed 端云协同推理框架，将轻量本地图像模型与延迟感知的云端时序预测视频模型结合，通过时序预测训练补偿网络延迟，在 100ms 延迟下比纯本地推理提升 6.4 mIoU、比纯远程推理提升 9.8 mIoU。
tags:
  - CVPR 2026
  - 图像分割
  - 协同推理
  - 实时视频分割
  - 延迟补偿
  - 时序预测
  - 端云协作
---

# DeDelayed: Deleting Remote Inference Delay via On-Device Correction

**会议**: CVPR 2026  
**arXiv**: [2510.13714](https://arxiv.org/abs/2510.13714)  
**代码**: [github.com/InterDigitalInc/dedelayed](https://github.com/InterDigitalInc/dedelayed) (有)  
**领域**: Segmentation  
**关键词**: 协同推理, 实时视频分割, 延迟补偿, 时序预测, 端云协作

## 一句话总结
提出 DeDelayed 端云协同推理框架，将轻量本地图像模型与延迟感知的云端时序预测视频模型结合，通过时序预测训练补偿网络延迟，在 100ms 延迟下比纯本地推理提升 6.4 mIoU、比纯远程推理提升 9.8 mIoU。

## 研究背景与动机
**领域现状**: 最强大的视频理解模型计算量太大无法在资源受限的终端设备上运行；而将推理卸载到云端会引入通信延迟，导致预测过时。

**现有痛点**: (1) 现有分割计算方法将所有本地计算资源用于单一推理管线，一旦云端不可用则无法 fallback；(2) 未考虑延迟对预测准确性的影响；(3) 使用降低时空分辨率的方式控制计算成本。

**核心矛盾**: 云端模型精度高但有延迟，本地模型实时但精度低——如何取两者之长？

**本文目标**: 设计一个实时推理系统，结合延迟的高质量远程特征和实时的本地低分辨率特征。

**切入角度**: 训练远程模型预测**未来帧**的特征，让延迟的远程输出在到达时仍然有用。

**核心 idea**: $\hat{y}_t = f_{\text{local}}(x_t, z_{t-\tau})$，本地模型处理当前帧，远程模型预测未来帧特征，两者通过逐元素加法融合。

## 方法详解

### 整体框架
- **远程模型**: 2D ViT (EfficientViT-L1) 对每帧提特征 → 时序拼接 K=4 帧 → 加延迟嵌入 → 3D ViT 编码器 → 自适应池化 + 通道瓶颈（DR-AE）→ 下行传输
- **本地模型**: CNN2D + CoAt2D 处理低分辨率当前帧，融合远程特征后做分割
- **融合方式**: 远程特征经空间池化对齐后**逐元素加到**本地 CNN2D 输出上

### 关键设计
1. **时序预测训练 (Temporally Predictive Training)**: 训练时对远程模型输入施加人工延迟 $D$ 帧（0-5帧均匀采样），但监督信号来自未来帧的标签。加入可学习的"延迟嵌入"（类似位置编码），使模型行为能根据实际延迟自适应调整。设计动机：网络延迟不可避免，让模型学会预测未来从而主动补偿延迟。

2. **与本地 Fallback 完全集成**: 远程特征通过逐元素加法融合到本地模型中间层——如果远程输出缺失，本地模型仍能独立工作。设计动机：硬实时应用中必须有完整的本地 fallback。加法融合的优势在于：远程信号为零时模型行为完全等价于纯本地推理。

3. **混合分辨率推理 (Mixed-Resolution Inference)**: 本地模型处理低分辨率帧（704×480），远程模型处理高分辨率帧（720p）。设计动机：在终端设备上以捕获分辨率运行任何模型都不现实，但云端 GPU 可以处理高分辨率视频。远程提供语义理解能力，本地提供空间定位能力。

### 损失函数 / 训练策略
- 多阶段训练：远程和本地模型先分别在 ImageNet→Cityscapes→BDD100K 上预训练，最后联合微调
- 联合训练时使用 per-pixel 交叉熵损失，Adan 优化器 + warmup-stable-decay 学习率调度
- 训练延迟 $\tau$ 从 0-5 帧均匀采样（0-167ms @30fps）

## 实验关键数据

### 主实验（BDD100K 语义分割 mIoU）

| 推理配置 | 0ms | 33ms | 67ms | 100ms | 167ms |
|---------|------|------|------|-------|-------|
| Local only | 0.601 | 0.601 | 0.601 | 0.601 | 0.601 |
| Remote image | 0.655 | 0.616 | 0.567 | 0.530 | 0.525 |
| Remote predictive | 0.655 | 0.649 | 0.644 | 0.637 | 0.624 |
| **DeDelayed** | **0.670** | **0.668** | **0.666** | **0.665** | **0.668** |

### 消融实验

| 配置 | mIoU @167ms | 说明 |
|------|------------|------|
| Local only | 0.601 | 不受延迟影响但精度低 |
| Remote image | 0.525 | 延迟严重损害精度 |
| Remote predictive | 0.624 | 时序预测大幅缓解 |
| **DeDelayed (full)** | **0.668** | 几乎消除延迟影响 |

### 关键发现
- 常规远程推理在延迟超过 67ms 后性能低于本地推理
- DeDelayed 在 167ms 延迟下比本地推理高 6.7 mIoU，相当于使用 10 倍大模型
- 远程模型的激活图可视化显示：远程提供准确的对象分类，本地提供精确的空间定位
- 延迟嵌入使单个模型适应 0-167ms 的动态延迟范围

## 亮点与洞察
- **Fallback-first 设计**: 远程信息作为"辅助信号"而非必需依赖，确保硬实时安全性
- **延迟嵌入 ≈ 可学习的运动补偿**: 通过条件化于延迟量，模型学习不同程度的运动预测
- **混合分辨率的互补性**: 远程高分辨率识别远处小目标（如远处行人），本地低分辨率提供精确空间校准
- 逐元素加法虽然简单，但行为定义良好，远程缺失时优雅退化

## 局限与展望
- 仅在分割任务上验证，未测试检测或其他密集预测任务
- 使用伪标签训练（因 BDD100K 缺少逐帧标注），真实标签下效果可能更好
- 上行视频压缩引入的失真未被显式建模
- 超过 167ms 的高延迟场景未测试
- 未探索多个远程模型或分层级融合
- 本地模型在极低功耗设备（<5W）上的可行性有待验证
- 下行带宽受限时 DR-AE 的压缩效率有待优化
- 异构传感器（如 LiDAR + 相机）的融合场景未覆盖

## 相关工作与启发
- 与 split computing 的区别: FCM 将本地计算全部用于上行管线，无 fallback
- 与 Knowledge Boosting 相比: 后者需要为每个固定延迟训练单独模型
- 延迟嵌入设计可推广到其他异步信息融合场景
- 自适应模型流 (Adaptive Model Streaming) 流式更新权重，与 DeDelayed 的特征融合正交

## 技术细节补充
- **远程模型**: EfficientViT-L1 (2D ViT, patch 8×8) → 时序拼接 K=4 帧 → 3D ViT + 延迟嵌入
- **本地模型**: CNN2D + CoAt2D，最大分辨率 704×480
- **DR-AE**: 自适应空间池化 + 通道瓶颈，匹配本地分辨率并压缩下行带宽
- **上行压缩**: 720p 30fps 以 1-10 Mbps 传输（5G 蜂窝网络）
- **目标延迟**: 本地和远程均为 33ms (单帧@30fps)
- **伪标签**: 验证集用 DepthAnything，训练集用 EoMT
- **优化器**: Adan + warmup-stable-decay + 梯度裁剪 + LLRD
- **关键洞察**: 远程模型激活图可视化显示其准确区分和分类对象（如远处行人），本地模型则提供精确位置校正
- **数据集**: BDD100K 包含 70K 训练视频，30fps 城市驾驶场景
- **评估**: 使用 Cityscapes 19 类语义分割标准
- **5G 适用性**: 设计参数匹配 5G 蜂窝网络能力（上行 1-10 Mbps）

## 评分
- 新颖性: ⭐⭐⭐⭐ 时序预测+延迟嵌入的端云协同框架设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 多种配置对比详尽，但仅一个数据集一个任务
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，系统图直观
- 价值: ⭐⭐⭐⭐⭐ 直接面向实际部署场景，工程价值极高

<!-- RELATED:START -->

## 相关论文

- [A²LC: Active and Automated Label Correction for Semantic Segmentation](../../AAAI2026/segmentation/a2lc_active_and_automated_label_correction_for_semantic_segm.md)
- [EdgeTAM: On-Device Track Anything Model](../../CVPR2025/segmentation/edgetam_on-device_track_anything_model.md)
- [Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)
- [SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)
- [RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportionaware_dynamic_adaptive_sali.md)

<!-- RELATED:END -->
