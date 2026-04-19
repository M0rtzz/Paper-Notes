---
title: >-
  [论文解读] Seeing the Unseen: Zooming in the Dark with Event Cameras
description: >-
  [AAAI 2026][图像恢复][低光视频超分] 提出首个事件驱动低光视频超分（LVSR）框架 RetinexEVSR，通过 Retinex 启发的双向融合策略（RBF）——先用光照图引导事件特征去噪（IEE），再用增强后的事件特征恢复反射率细节（ERE），在 SDSD 基准上实现 2.95dB 增益且运行时间减少 65%。
tags:
  - AAAI 2026
  - 图像恢复
  - 低光视频超分
  - 事件相机
  - Retinex分解
  - 跨模态融合
  - 双向增强
---

# Seeing the Unseen: Zooming in the Dark with Event Cameras

**会议**: AAAI 2026  
**arXiv**: [2601.02206](https://arxiv.org/abs/2601.02206)  
**代码**: [RetinexEVSR](https://github.com/DachunKai/RetinexEVSR)  
**领域**: 低光视频超分辨率  
**关键词**: 低光视频超分, 事件相机, Retinex分解, 跨模态融合, 双向增强

## 一句话总结

提出首个事件驱动低光视频超分（LVSR）框架 RetinexEVSR，通过 Retinex 启发的双向融合策略（RBF）——先用光照图引导事件特征去噪（IEE），再用增强后的事件特征恢复反射率细节（ERE），在 SDSD 基准上实现 2.95dB 增益且运行时间减少 65%。

## 研究背景与动机

**领域现状**：视频超分辨率（VSR）在正常光照下已取得良好效果。然而，低光条件（如夜间监控、夜景摄影）下 VSR 面临严峻挑战——输入帧噪声大、对比度低、纹理细节严重缺失。

**现有痛点**：
(1) **级联策略失效**——先增强后超分会传播并放大像素误差；先超分后增强会放大噪声和弱化纹理；
(2) **单阶段方法不足**——直接学习低光 LR 到正常光 HR 的映射（如 DP3DF）仍存在严重伪影、结构失真和光照不准；
(3) **事件信号融合困难**——低光下 RGB 帧和事件数据都严重退化（噪声、时间拖尾、空间非稳态分布），直接融合会引入伪影。

**核心矛盾**：事件相机提供高动态范围（120dB）和高频边缘信息，是低光条件的理想补充源，但两个模态都退化的情况下如何有效提取和融合有用信息是核心难题。

**本文切入角度**：借助 Retinex 分解（光照+反射率），利用平滑低噪的光照图引导事件去噪，再用增强后的事件特征恢复反射率细节，形成双向互利的融合策略。

## 方法详解

### 整体框架

RetinexEVSR 接收低分辨率图像序列 $\{X_t^{LR}\}_{t=1}^T$ 和对应事件数据 $\{\mathcal{E}_t^{LR}\}_{t=1}^T$，输出超分辨率正常光照序列 $\{Y_t^{SR}\}_{t=1}^T$。每帧先进行 Retinex 分解获取光照图和反射率图，从反射率估计光流进行时间对齐，然后通过 IEE 和 ERE 模块进行双向增强，最后联合光照、增强反射率和精炼事件特征进行上采样重建。

### 关键设计

1. **光照引导事件增强模块（IEE）**

    - **功能**：利用 Retinex 分解得到的光照图渐进式去除事件特征中的低光伪影
    - **核心思路**：光照图提供平滑、低噪声的全局亮度信息，通过多尺度融合从粗到细地引导事件特征精炼。在每个时间步，将光照特征与事件特征在多个空间尺度上进行交互，逐层消除噪声和拖尾效应
    - **设计动机**：两个退化模态都不可靠时，光照分量因本身平滑特性噪声最少，可充当可靠的引导信号。相比直接融合退化的 RGB 和事件数据，先净化事件特征能避免噪声传播

2. **事件引导反射率增强模块（ERE）**

    - **功能**：利用经 IEE 增强后的事件特征恢复反射率图中缺失的高频细节
    - **核心思路**：采用动态注意力机制将事件的高频边缘信息注入反射率特征流，通过多尺度融合实现细粒度纹理恢复
    - **设计动机**：反射率分量保留了场景固有内容但在低分辨率下细节不足，而事件相机特有的"运动边缘"信息正好可以弥补这一缺陷。经过 IEE 增强后的事件特征质量更高，能更有效地注入信息

3. **Retinex 启发双向融合策略（RBF）**

    - **功能**：统一 IEE 和 ERE 的融合流程，形成"光照→事件→反射率"的双向互利链路
    - **核心思路**：区别于先前方法直接融合两个退化模态，RBF 引入 Retinex 分解作为中间桥梁：光照引导事件精炼，精炼后的事件再引导反射率恢复
    - **设计动机**：这种级联增强避免了退化信号之间的直接对抗，确保每一步融合都有高质量的引导信号

### 损失函数 / 训练策略

使用 Charbonnier Loss 和感知损失的组合进行端到端训练。在 SDSD、SDE 和 RELED 三个数据集上训练和评估，涵盖合成数据和真实世界数据。

## 实验关键数据

### 主实验——SDSD 数据集 4× LVSR

| 方法 | 类型 | PSNR↑(in) | PSNR↑(out) | SSIM↑(in) | LPIPS↓(in) |
|------|------|----------|-----------|----------|-----------|
| Retinexformer+IART | 增强→超分 | 27.09 | 19.84 | 0.8331 | 0.3938 |
| IART+Retinexformer | 超分→增强 | 25.30 | 24.07 | 0.8500 | 0.3541 |
| EvTexture | 事件超分 | 27.33 | 24.20 | 0.8776 | 0.3286 |
| FMA-Net | 联合方法 | 27.53 | 23.93 | 0.8680 | 0.3300 |
| **RetinexEVSR** | **本文** | **30.28** | **25.15** | **0.8932** | **0.3149** |

### RELED 真实数据集对比

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Params(M) | FLOPs(G) | Runtime(ms) |
|------|-------|-------|--------|-----------|----------|-------------|
| EvTexture | 28.07 | 0.8604 | 0.4837 | 8.90 | 1141.1 | 126.9 |
| FMA-Net | 27.61 | 0.8611 | 0.4633 | 9.62 | 1941.3 | 596.3 |
| **RetinexEVSR** | **28.92** | **0.8707** | **0.4612** | **8.07** | **159.1** | **44.5** |

### 关键发现

- RetinexEVSR 在 SDSD-in 上比次优方法高 2.95dB，同时运行时间仅 44.5ms（比 EvTexture 快 65%，比 FMA-Net 快 93%）
- 模型参数量（8.07M）和 FLOPs（159.1G）均为所有方法最低，效率优势显著
- 在极暗（-6.7EV）和严重运动模糊条件下仍能恢复清晰可辨的文字和纹理
- 级联策略（Type I-IV）性能普遍差于端到端方法（Type V-VI），验证了级联误差传播问题

## 亮点与洞察

1. **RBF 策略设计精巧**：通过 Retinex 分解中的光照分量作为桥梁，化解了两个退化模态直接融合的困境
2. **效率-性能帕累托最优**：参数量和 FLOPs 最小，性能最优，同时运行时间最短
3. **系统性基线对比**：覆盖 6 种策略类型（增强→超分、超分→增强、事件增强→超分、事件超分→增强、纯超分、联合方法），对比极为全面
4. **真实世界验证**：在 RELED 真实低光事件数据集上同样最优，证明方法的实用性

## 局限与展望

1. 依赖事件相机硬件，限制了应用场景的通用性
2. Retinex 分解的准确度会影响后续模块效果，在极端退化下分解质量可能不稳定
3. 仅验证了 4× 超分，更高放大倍率（8×、16×）的效果未知
4. 时间对齐依赖光流估计，大位移场景下可能失效

## 相关工作与启发

- Retinex 分解作为模态间融合的桥梁是一个通用设计范式，可推广至其他退化信号融合任务（如雨天超分、水下增强）
- 事件相机的高动态范围特性与低光视觉的协同还有更多探索空间（如低光语义分割、低光目标检测）
- "先净化辅助模态再融合"的策略对多传感器融合领域具有普遍参考价值

## 评分

⭐⭐⭐⭐

- **新颖性** ⭐⭐⭐⭐：首个事件驱动 LVSR 方案，RBF 双向融合策略有新意
- **实验充分度** ⭐⭐⭐⭐⭐：三个数据集、六类对比策略、合成+真实数据验证
- **写作质量** ⭐⭐⭐⭐：动机清晰，策略对比图直观易懂
- **价值** ⭐⭐⭐⭐：为低光视频恢复提供了高效且高性能的新范式

<!-- RELATED:START -->

## 相关论文

- [SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](../../CVPR2026/video_generation/seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [Seeing the Wind from a Falling Leaf](../../NeurIPS2025/video_generation/seeing_the_wind_from_a_falling_leaf.md)
- [SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](../../CVPR2026/video_generation/switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [Chain of Event-Centric Causal Thought for Physically Plausible Video Generation](../../CVPR2026/video_generation/chain_of_event-centric_causal_thought_for_physically_plausible_video_generation.md)
- [Mind the Time: Temporally-Controlled Multi-Event Video Generation](../../CVPR2025/video_generation/mind_the_time_temporally-controlled_multi-event_video_generation.md)

<!-- RELATED:END -->
