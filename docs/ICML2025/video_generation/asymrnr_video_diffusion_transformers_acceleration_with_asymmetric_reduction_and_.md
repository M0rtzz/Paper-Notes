---
title: >-
  [论文解读] AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration
description: >-
  [ICML 2025][视频生成][视频生成加速] 提出 AsymRnR——一种免训练的视频 DiT 加速方法，基于注意力中不同组件（Q/K/V）、不同层、不同去噪步骤的冗余程度不同的观察，非对称地削减 token 以实现无损加速。
tags:
  - "ICML 2025"
  - "视频生成"
  - "视频生成加速"
  - "Transformer"
  - "token削减"
  - "非对称策略"
  - "training-free"
---

# AsymRnR: Video Diffusion Transformers Acceleration with Asymmetric Reduction and Restoration

**会议**: ICML 2025  
**arXiv**: [2412.11706](https://arxiv.org/abs/2412.11706)  
**代码**: [https://github.com/wenhao728/AsymRnR](https://github.com/wenhao728/AsymRnR)  
**领域**: 图像/视频恢复  
**关键词**: 视频生成加速, 扩散 Transformer, token削减, 非对称策略, training-free

## 一句话总结
提出 AsymRnR——一种免训练的视频 DiT 加速方法，基于注意力中不同组件（Q/K/V）、不同层、不同去噪步骤的冗余程度不同的观察，非对称地削减 token 以实现无损加速。

## 研究背景与动机

**领域现状**：视频 Diffusion Transformer（DiT）如 HunyuanVideo、CogVideoX 生成质量优秀但计算成本极高（生成几秒视频需要数分钟甚至数小时）。

**现有痛点**：(a) 蒸馏方法需要大量训练; (b) 特征缓存方法针对特定架构; (c) Token Merging（ToMe）均匀削减所有组件导致视频出现扭曲和像素化。

**核心矛盾**：不同组件对 token 削减的敏感度差异巨大——但现有方法一视同仁。

**本文目标**：如何考虑组件的不同敏感度来高效削减 token？

**切入角度**：实验发现三条关键规律：(a) Q 扰动比 K/V 扰动影响大; (b) 浅层扰动比深层影响大; (c) 早期去噪步骤影响语义，晚期影响细节。

**核心 idea**：非对称削减——对 K/V 可大幅削减而保留全部 Q，不同层和步骤使用不同削减比例。

## 方法详解

### 整体框架
在注意力计算前：
1. 保留全部 Query token
2. 对 Key/Value 按相似性合并冗余 token
3. 用非对称调度器根据层深度和去噪步骤调整削减比例
4. 注意力计算后恢复到原始 token 数量

### 关键设计

1. **非对称 Q-KV 削减**:

    - 功能：仅削减 K/V 的 token，保留全部 Q
    - 核心思路：Q 的 token 直接影响输出每个位置的表示，K/V 提供上下文可容忍冗余
    - 设计动机：随机丢弃 30% Q token 导致严重降质，而丢弃 30% K/V 几乎无影响

2. **自适应削减调度**:

    - 功能：根据层深度和去噪步骤动态调整 K/V 削减比例
    - 核心思路：浅层削减少（对质量敏感），深层削减多（冗余更高）；早期步骤削减少（影响语义），晚期步骤可多削减
    - 设计动机：符合 Liebig 最小定律——系统质量由最敏感组件决定

3. **匹配缓存**:

    - 功能：跨去噪步骤复用 token 匹配结果
    - 核心思路：相邻去噪步骤的 token 相似度变化缓慢，可复用匹配
    - 设计动机：减少匹配计算本身的开销

### 损失函数 / 训练策略
- 完全免训练，即插即用
- 适用于任何视频 DiT 架构

## 实验关键数据

### 主实验

| 模型 | 方法 | 加速比 | VBench↑ |
|------|------|--------|---------|
| HunyuanVideo | 基线 | 1.0× | 83.2 |
| | ToMe | 1.4× | 79.8 (-3.4) |
| | **AsymRnR** | **1.45×** | **83.5** (+0.3) |
| CogVideoX | 基线 | 1.0× | 81.5 |
| | **AsymRnR** | **1.38×** | **81.8** (+0.3) |

### 消融实验

| 配置 | VBench | 加速 | 说明 |
|------|--------|------|------|
| 对称削减（Q+KV 同比例） | 79.8 | 1.4× | 扭曲严重 |
| **非对称（仅 KV）** | **83.5** | **1.45×** | 无损甚至提升 |
| 均匀调度 | 82.1 | 1.4× | 不适应层/步差异 |
| 自适应调度 | **83.5** | **1.45×** | 更精细的控制 |

### 关键发现
- 非对称策略不仅无损还可能提升质量（削减冗余 K/V 起到了正则化效果）
- 自适应调度比均匀调度提升 +1.4 VBench 分
- 跨 4 种 SOTA 视频 DiT 模型一致有效

## 亮点与洞察
- **Q 和 KV 角色不对称**的发现具有普适性——不仅适用于视频 DiT，可能对所有 Transformer 架构的推理加速有启发
- 免训练+模型无关使方法极其实用
- 质量"提升"暗示原始模型中存在 KV 冗余的"噪声"

## 局限与展望
- 加速比目前约 1.4-1.5×，不如蒸馏方法的加速幅度
- 匹配缓存的复用频率需要手动调节
- 未讨论与步骤蒸馏等方法的组合

## 相关工作与启发
- **vs ToMe**: 对称削减导致降质，AsymRnR 非对称策略无损
- **vs 特征缓存**: 架构相关，AsymRnR 架构无关
- 对 Transformer 推理优化有广泛启发

## 评分
- 新颖性: ⭐⭐⭐⭐ 非对称 token 削减视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 4 种 SOTA 模型，VBench 评估
- 写作质量: ⭐⭐⭐⭐ 分析透彻，可视化清晰
- 价值: ⭐⭐⭐⭐⭐ 实用的免训练视频生成加速方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers](riflex_a_free_lunch_for_length_extrapolation_in_video_diffusion_transformers.md)
- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](../../CVPR2025/video_generation/video_motion_transfer_with_diffusion_transformers.md)
- [\[ICCV 2025\] MagicMirror: ID-Preserved Video Generation in Video Diffusion Transformers](../../ICCV2025/video_generation/magicmirror_id-preserved_video_generation_in_video_diffusion_transformers.md)
- [\[CVPR 2025\] Towards Precise Scaling Laws for Video Diffusion Transformers](../../CVPR2025/video_generation/towards_precise_scaling_laws_for_video_diffusion_transformers.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](../../ICCV2025/video_generation/decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)

</div>

<!-- RELATED:END -->
