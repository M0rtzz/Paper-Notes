---
title: >-
  [论文解读] Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models
description: >-
  [NeurIPS 2025][文本到视频生成] 对开源文本到视频 (T2V) 模型进行系统性延迟和能耗分析，建立了基于 FLOP 的计算分析模型预测 WAN2.1 的缩放规律（空间/时间维度二次缩放、去噪步数线性缩放），并在 7 个 T2V 模型上提供跨模型能耗基准。
tags:
  - NeurIPS 2025
  - 文本到视频生成
  - 能耗分析
  - 延迟基准测试
  - 扩散模型
  - 可持续AI
---

# Video Killed the Energy Budget: Characterizing the Latency and Power Regimes of Open Text-to-Video Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.19222](https://arxiv.org/abs/2509.19222)  
**代码**: [GitHub](https://github.com/anonymized)  
**领域**: 图像修复 / 可持续计算  
**关键词**: 文本到视频生成, 能耗分析, 延迟基准测试, 扩散模型, 可持续AI

## 一句话总结

对开源文本到视频 (T2V) 模型进行系统性延迟和能耗分析，建立了基于 FLOP 的计算分析模型预测 WAN2.1 的缩放规律（空间/时间维度二次缩放、去噪步数线性缩放），并在 7 个 T2V 模型上提供跨模型能耗基准。

## 研究背景与动机

1. **领域现状**: T2V 生成技术快速发展，从 Sora、Veo 等闭源系统到 WAN2.1、CogVideoX 等开源模型。这些模型从研究原型向实际应用过渡，开始被用于创意工具和视频合成 API。

2. **现有痛点**: 大多数 T2V 评估仅关注感知质量（FID、运动平滑度等），忽略了延迟和能效。生成几秒连贯视频需要数十步去噪、高分辨率和数百帧，导致巨大能耗。已有研究仅限于 Open-Sora 单个模型的 2 秒 240p 视频。

3. **核心矛盾**: T2V 模型质量持续提升，但计算成本和环境影响缺乏系统性理解——无法在质量和可持续性之间做出明智权衡。

4. **本文目标**: 系统量化 T2V 模型的延迟和能耗随关键参数（分辨率、帧数、去噪步数）的缩放规律。

5. **切入角度**: 将 T2V 推理建模为计算受限 (compute-bound) 过程，分解各操作符的 FLOP，推导解析缩放定律并实验验证。

6. **核心 idea**: T2V 推理被 DiT 的自注意力主导——延迟和能耗随空间/时间维度二次增长、随去噪步数线性增长，模型间能耗差距可达 3000 倍。

## 方法详解

### 整体框架

两阶段方法：
1. **理论分析**: 以 WAN2.1-T2V-1.3B 为参考，分解推理 FLOP 并推导缩放定律
2. **实验验证**: 微观基准验证缩放预测 + 7 个模型的跨模型比较

### 关键设计

**1. 分操作符 FLOP 分解**

- **功能**: 精确预测 T2V 推理的计算成本
- **核心思路**: WAN2.1 推理分解为五个组件：
    - **一次性**: 文本编码器 (T5)、VAE 解码器
    - **每步重复** (×$g \cdot S$): DiT 自注意力 ($N(8\ell d^2 + 4\ell^2 d)$)、交叉注意力 ($N(4\ell d^2 + 4md^2 + 4\ell md)$)、MLP ($N(4f\ell d^2)$)、时间步 MLP
  
  其中 DiT token 长度 $\ell = (1+T/4) \cdot H/16 \cdot W/16$ 随分辨率和帧数线性增长。
- **设计动机**: 知道每个操作的 FLOP 组成才能预测不同配置下的成本，指导高效部署。

**2. 计算受限延迟模型**

- **功能**: 用理论 FLOP 预测实际延迟和能耗
- **核心思路**: 在 H100 上分析后确认主要操作（自注意力、MLP）为计算受限（非内存受限）。定义经验效率 $\mu = F_{total}/(D_{measured} \cdot \Theta_{peak})$，通过回归得到 $\mu \approx 0.456$（$R^2 = 0.998$），然后 $D_{total} \approx F_{total}/(\mu \cdot \Theta_{peak})$，$E_{total} \approx P_{max} \cdot D_{total}$。
- **设计动机**: 计算受限假设使得延迟与 FLOP 成正比，大大简化了预测。

**3. 缩放规律推导**

- **功能**: 明确 T2V 的三种缩放行为
- **核心思路**: 
    - **空间维度 $(H,W)$**: $\ell \propto HW$ → 自注意力 $\propto \ell^2 \propto (HW)^2$ → **二次缩放**
    - **时间维度 $T$**: $\ell \propto T$ → 自注意力 $\propto T^2$ → **二次缩放**
    - **去噪步数 $S$**: 每步相同操作 → **线性缩放**
    - 辅助组件（文本编码器、时间步 MLP）贡献微小
- **设计动机**: 理解哪个维度的增长最"贵"是优化部署的基础。

### 损失函数 / 训练策略

（基准测试工作，无训练过程。硬件：NVIDIA H100 SXM 80GB，每个配置 2 轮 warmup + 5 轮测量。能耗通过 CodeCarbon 测量。）

## 实验关键数据

### 主实验

**缩放规律验证（WAN2.1-T2V-1.3B）**

| 缩放维度 | 理论预测 | 验证结果 | 能耗平均误差 | 延迟平均误差 |
|---------|---------|---------|------------|------------|
| 空间分辨率 | 二次 | ✓ 二次 | 11.6% | 14.0% |
| 帧数 | 二次 | ✓ 二次 | 6.6% | 10.5% |
| 去噪步数 | 线性 | ✓ 完美线性 | **1.9%** | **1.9%** |

**跨模型能耗基准（默认设置，单视频生成）**

| 模型 | 延迟 (s) | GPU 能耗 (Wh) | 分辨率 | 帧数 | 步数 |
|------|---------|-------------|--------|------|------|
| AnimateDiff | **0.68** | **0.115** | 512² | 16 | 4 |
| LTX-Video | 9.7 | 3.16 | 512×704 | 121 | 40 |
| CogVideoX-2b | 50.6 | 8.3 | 480×720 | 49 | 50 |
| CogVideoX-5b | 124 | 21.6 | 480×720 | 49 | 50 |
| Mochi-1-preview | 263 | 44.7 | 480×848 | 84 | 64 |
| WAN2.1-1.3B | 410 | 78.8 | 720×1280 | 81 | 50 |
| WAN2.1-14B | **1875** | **359.7** | 720×1280 | 81 | 50 |

### 消融实验

| 组件 | FLOP 占比 | 主导因素 |
|------|----------|---------|
| DiT 自注意力 | ~60-70% | **主导** |
| DiT MLP | ~20-25% | 次要 |
| DiT 交叉注意力 | ~5-10% | 小 |
| VAE 解码器 | <5% | 可忽略 |
| 文本编码器 | 一次性 | 可忽略 |

### 关键发现

- **模型间差距达 3000 倍**: AnimateDiff (0.14 Wh) vs WAN2.1-14B (415 Wh)
- 能耗+延迟的主要驱动因素：模型大小 > 分辨率 > 帧数 > 步数
- GPU 占总能耗的 **80-90%**，CPU 和 RAM 贡献微小
- WAN2.1-14B 生成单个 5 秒 720p 视频消耗 360 Wh GPU 能量——相当于运行冰箱约 1 小时
- 去噪步数是最"便宜"的优化维度（线性缩放），应优先考虑减步数策略

## 亮点与洞察

- 首次建立 T2V 模型的系统性能耗基准，填补了该领域可持续性评估空白
- FLOP 分析模型与实测的高吻合度（$R^2 = 0.998$）验证了计算受限假设
- 明确指出优化方向优先级：减少步数（线性）>> 降分辨率/帧数（二次但影响质量大）
- 跨模型对比揭示了巨大的效率差异，为模型选择提供实用参考

## 局限与展望

- 未评估感知质量——无法建立质量-能耗的 Pareto 前沿
- 仅在 H100 单卡上测试，多卡并行和不同硬件的行为未研究
- 未考虑推理优化技术（如 diffusion caching、量化、蒸馏）对能耗的影响
- 效率 $\mu$ 为经验标定的固定值，不同模型可能不同
- 仅覆盖了 UNet/DiT 架构的模型，其他架构（如自回归视频生成）未分析

## 相关工作与启发

- 延续 Strubell et al. (2019)、Luccioni et al. (2024) 的 ML 能耗研究传统
- Li et al. (2024) 对 Open-Sora 的初步研究是直接前驱，本文大幅扩展了模型和配置覆盖范围
- 启发：随着 T2V 大规模部署，应将能效作为模型评估的标准维度之一

## 评分

- **新颖性**: ⭐⭐⭐ 方法论并不新颖（FLOP 分析+基准测试），但系统性和覆盖范围是首次
- **实验充分度**: ⭐⭐⭐⭐⭐ 理论-实验对照完整，三维度缩放验证 + 7 模型对比
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，理论推导详细，表格和图表信息量大
- **价值**: ⭐⭐⭐⭐ 为 T2V 可持续部署提供了重要参考数据和优化指南

<!-- RELATED:START -->

## 相关论文

- [Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [DisMo: Disentangled Motion Representations for Open-World Motion Transfer](dismo_disentangled_motion_representations_for_openworld_moti.md)
- [VideoDirector: Precise Video Editing via Text-to-Video Models](../../CVPR2025/video_generation/videodirector_precise_video_editing_via_text-to-video_models.md)
- [Multi-subject Open-set Personalization in Video Generation](../../CVPR2025/video_generation/multi-subject_open-set_personalization_in_video_generation.md)
- [VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)

<!-- RELATED:END -->
