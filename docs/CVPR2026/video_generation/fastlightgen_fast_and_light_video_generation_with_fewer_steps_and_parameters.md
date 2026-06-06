---
title: >-
  [论文解读] FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters
description: >-
  [CVPR 2026][视频生成][视频生成加速] FastLightGen 提出三阶段蒸馏算法，首次实现采样步数与模型大小的联合蒸馏，通过识别冗余层、动态概率剪枝和 well-guided teacher guidance 分布匹配…
tags:
  - "CVPR 2026"
  - "视频生成"
  - "视频生成加速"
  - "步数蒸馏"
  - "模型剪枝"
  - "分布匹配"
  - "DiT压缩"
---

# FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters

**会议**: CVPR 2026  
**arXiv**: [2603.01685](https://arxiv.org/abs/2603.01685)  
**代码**: 无  
**领域**: 视频生成  
**关键词**: 视频生成加速, 步数蒸馏, 模型剪枝, 分布匹配, DiT压缩

## 一句话总结

FastLightGen 提出三阶段蒸馏算法，首次实现采样步数与模型大小的联合蒸馏，通过识别冗余层、动态概率剪枝和 well-guided teacher guidance 分布匹配，将 HunyuanVideo/WanX 压缩为 4 步 30% 参数剪枝的轻量生成器，实现约 35 倍加速且性能超越教师模型。

## 研究背景与动机

**领域现状**：大规模视频生成模型（HunyuanVideo、WanX）基于 DiT，130亿+参数，多步去噪。5秒视频在 H100 上约需 20 分钟。

**核心问题**：
   - 现有加速要么减步数（LCM/DMD）要么减参数（F3-Pruning/ICMD），无联合优化
   - 极端步数蒸馏（1-2步）性能急剧下降
   - 联合蒸馏可在同性能下更大加速（4步50%参数=50x vs 纯步数3步=33.3x）

**本文方案**：三阶段——识别冗余层、动态概率剪枝、well-guided teacher guidance 分布匹配

## 方法详解

### 整体框架

三阶段蒸馏管道：Stage I 识别不重要块，Stage II 动态概率剪枝训练，Stage III 精细分布匹配。

### 关键设计

#### 1. Stage I: 识别不重要模型块

逐一跳过每个 DiT block，用 Tweedie 公式估计 ELBO 评估重要性。发现 **U 型模式**：初始层和末尾层最关键，中间层冗余。HunyuanVideo 的 double block 比 single block 更关键。

#### 2. Stage II: 动态概率剪枝训练

不重要层按伯努利分布（p=0.5）随机跳过，构建参数共享的 unpruned/pruned 模型：

- 蒸馏损失：pruned 输出对齐 unpruned 输出（stop gradient）
- 关键发现：alpha=1（完全去除 GT 监督、仅用蒸馏）效果最佳，"软"监督优于"硬"GT
- 输出能在不同深度配置下都表现良好的鲁棒模型

#### 3. Stage III: 精细分布匹配

基于 DMD2 框架引入 well-guided teacher guidance：

- Real DiT 同时使用 pruned 和 unpruned 输出
- beta_1（inter CFG）控制文本引导强度，beta_2（intra CFG）控制 unpruned 对 pruned 引导
- 从均匀分布采样 CFG 增强鲁棒性
- 避免过弱（无效）和过强（学生跟不上）教师的两种失败模式

### 损失函数 / 训练策略

- Stage II: 16卡 H100, lr=1e-5, 4000 iter, ~64 GPU days
- Stage III: lr=5e-7, 1000 iter, ~16 GPU days
- 最优配置：(alpha, beta_1, beta_2) = (1, 3.5, 0.25) for WanX
- 不宜过长训练（运动过剧烈/颜色过饱和）

## 实验关键数据

### 主实验

**VBench-I2V 对比（WanX-TI2V, 表2）**：

| 方法 | motion smooth | dynamic deg | aesthetic | imaging | average | time |
|------|-------------|------------|-----------|---------|---------|------|
| Euler (teacher) | 0.982 | 0.461 | 0.653 | 0.711 | 0.790 | 885s |
| DMD2 | 0.977 | 0.160 | 0.583 | 0.683 | 0.716 | 35.4s |
| LCM | 0.979 | 0.003 | 0.570 | 0.665 | 0.684 | 35.4s |
| MagicDistillation | 0.980 | 0.561 | 0.634 | 0.701 | 0.798 | 35.4s |
| **FastLightGen** | **0.983** | 0.500 | **0.656** | **0.717** | 0.794 | **28.3s** |

**与开源VDM对比（表1）**：

| 方法 | average |
|------|---------|
| CogVideoX-I2V | 0.759 |
| SVD-XT-1.0 | 0.789 |
| WanX-TI2V (teacher) | 0.790 |
| **FastLightGen** | **0.794** |

### 消融实验

**蒸馏权重消融（表4）**：

| distill weight alpha | average |
|---------------------|---------|
| 0.0 | 0.780 |
| 0.5 | 0.780 |
| 0.7 | 0.788 |
| **1.0** | **0.791** |

**Intra CFG 消融（表5, beta_1=3.5）**：

| beta_2 | dynamic deg | average |
|--------|------------|---------|
| 0.00 | 0.459 | 0.812 |
| **0.25** | 0.500 | **0.820** |
| 0.75 | 1.000 | 0.848 (有抖动) |

### 关键发现

- 4步+30%剪枝（保留70%参数）最优性价比，约 35.71x 加速
- 联合蒸馏同性能下比单维度更快（50x vs 33.3x）
- alpha=1 纯蒸馏是 Stage II 最佳
- aesthetic 和 imaging quality 超越教师模型

## 亮点与洞察

1. **联合蒸馏范式**：首次证明步数+大小联合蒸馏优于单维度
2. **Well-guided teacher**：inter/intra CFG 独立控制两个正交维度
3. **动态概率剪枝**：单模型适应不同深度
4. **U 型重要性**：VDM 初末层最关键的普适发现

## 局限与展望

1. 仅验证 TI2V 任务
2. 训练成本高（~80 GPU days）
3. beta_2 大时运动异常
4. 仅 block 级剪枝
5. 数据质量敏感

## 相关工作与启发

- **DMD2**：分布匹配蒸馏基础
- **MagicDistillation**：强步数蒸馏基线
- **ICMD**：视频大小蒸馏先驱
- **启发**："过强教师反而有害"值得更多验证

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 联合蒸馏+well-guided teacher |
| 技术深度 | 4 | 三阶段精细设计 |
| 实验完整性 | 4.5 | 多模型多指标充分消融 |
| 写作质量 | 4 | 图表清晰 |
| 实用价值 | 4.5 | 35x 加速意义重大 |
| 总分 | 4.2 | |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LightMover: Generative Light Movement with Color and Intensity Controls](lightmover_generative_light_movement_with_color_and_intensity_controls.md)
- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](../../ICML2026/video_generation/light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[NeurIPS 2025\] MagCache: Fast Video Generation with Magnitude-Aware Cache](../../NeurIPS2025/video_generation/magcache_fast_video_generation_with_magnitudeaware_cache.md)
- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](../../ICCV2025/video_generation/generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[CVPR 2025\] From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](../../CVPR2025/video_generation/from_slow_bidirectional_to_fast_autoregressive_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
