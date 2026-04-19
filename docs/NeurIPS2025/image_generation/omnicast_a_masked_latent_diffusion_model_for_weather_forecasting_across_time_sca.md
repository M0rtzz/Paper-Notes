---
title: >-
  [论文解读] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales
description: >-
  [NeurIPS 2025][图像生成][天气预报] 提出 OmniCast，一种结合掩码生成框架和潜在扩散模型的天气预报方法，通过联合生成未来天气序列（而非自回归迭代）来缓解误差累积，在次季节至季节（S2S）尺度达到 SOTA 性能，同时在中期预报上保持竞争力且推理速度快 10-20 倍。
tags:
  - NeurIPS 2025
  - 图像生成
  - 天气预报
  - 潜在扩散模型
  - 掩码生成建模
  - 次季节预报
  - VAE
---

# OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales

**会议**: NeurIPS 2025  
**arXiv**: [2510.18707](https://arxiv.org/abs/2510.18707)  
**代码**: [GitHub](https://github.com/tung-nd/omnicast)  
**领域**: 天气预报、生成模型  
**关键词**: 天气预报, 潜在扩散模型, 掩码生成建模, 次季节预报, VAE

## 一句话总结

提出 OmniCast，一种结合掩码生成框架和潜在扩散模型的天气预报方法，通过联合生成未来天气序列（而非自回归迭代）来缓解误差累积，在次季节至季节（S2S）尺度达到 SOTA 性能，同时在中期预报上保持竞争力且推理速度快 10-20 倍。

## 研究背景与动机

天气预报跨越不同时间尺度：中期（~2 周）已被深度学习方法显著推进（PanguWeather、GraphCast 等超越了 IFS 数值预报系统），但**次季节至季节（S2S, 2-6 周）**预报仍是巨大挑战。

S2S 预报难在三点：
- **误差累积**：现有方法多为自回归设计，短时间步迭代预测中误差逐步放大；多步微调在 S2S 的长序列上计算成本过高
- **初始条件 vs 边界条件**：短期预报主要依赖初始条件，S2S 还需考虑边界条件（如海温、土壤湿度），自回归短步训练无法捕捉这些长期驱动因素
- **不确定性量化**：S2S 预报本质上需要概率化方法，集合系统维度受限于计算成本

## 方法详解

### 整体框架

OmniCast 采用两阶段训练：

1. **第一阶段 — VAE 编码器**：将原始天气数据 $X \in \mathbb{R}^{V \times H \times W}$（$V$ 个物理变量）压缩到连续低维潜在空间的 token 图 $h \times w$，每帧独立编码
2. **第二阶段 — 掩码生成 Transformer**：在潜在空间中建模未来 token 序列的条件分布 $p(\mathbf{x} | \mathbf{c})$，使用扩散头处理连续 token

### 关键设计

1. **连续 VAE 替代离散 VQ-VAE**：天气数据有上百个物理变量，离散化压缩比高达 ~3938 倍导致严重重建误差。连续 VAE 使用 $D=16$ 维向量，压缩比仅 100 倍，大幅降低信息损失。空间下采样 16 倍（$128 \times 256 \to 8 \times 16$）。

2. **掩码生成建模**：训练时随机掩码一部分未来 token，任务是根据条件 token（初始状态）和可见 token 恢复被掩码的 token。采样掩码率 $\gamma \sim \mathcal{U}[0.5, 1.0]$。推理时从全掩码出发，按余弦调度逐步揭示 token。这种联合生成方式避免了自回归的误差累积，并允许模型捕捉跨时空的长程依赖。

3. **逐 token 扩散头**：Transformer backbone 为每个位置输出条件向量 $z_i$，一个小型 MLP 扩散网络以 $z_i$ 为条件估计 token 分布。采用 6 层残差块（宽度 2048），AdaLN 整合扩散步嵌入。训练时 1000 步线性噪声调度，推理时重采样至 100 步。关键效率优势：backbone 仅需一次前向传播，扩散步只走轻量 MLP。

4. **辅助确定性目标**：对前 10 帧施加 MSE 损失（指数衰减权重），因为 10 天内天气动力学仍具确定性。超过此范围天气变得混沌，强加 MSE 反而有害。总目标：$\mathcal{L} = \mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{deter}}$。

### 实现细节

- Transformer：MAE 编码器-解码器架构，各 16 层 × 16 头，隐藏维度 1024
- 训练：32 × A100 GPU，4 天（远少于 Gencast 的 32 TPUv5e × 5 天）
- S2S 设置：$T=44$ 步（1-44 天），$\tau=1.3$，50 成员集合
- 中期设置：2 步预测（12h 间隔），自回归采样，$\tau=1.0$

## 实验关键数据

### S2S 预报确定性指标（ERA5, 1.4°分辨率, 2022 测试年）

| 方法 | 类型 | T850 RMSE 趋势 | Z500 RMSE 趋势 | 偏差表现 |
|------|------|--------------|--------------|---------|
| PanguWeather | DL（自回归） | 短期好，>15天急剧恶化 | 同左 | 偏差较大 |
| GraphCast | DL（自回归） | 短期好，>10天恶化 | 同左 | 偏差较大 |
| ECMWF-ENS | 数值集合 | 全程较优 | 全程较优 | 偏差中等 |
| **OmniCast** | DL（掩码扩散） | 短期略差，>10天匹配 ECMWF | 同左 | **偏差最低，近零** |

OmniCast 在 >10 天的长 lead time 上性能追平或超越 ECMWF-ENS，是唯一能在全程保持近零偏差的方法。

### S2S 物理一致性指标

| 方法 | 谱散度 (SDIV) | 谱残差 (SRES) |
|------|-------------|-------------|
| PanguWeather | 差（频谱失真严重） | 差 |
| GraphCast | 差 | 差 |
| ECMWF-ENS | 中等 | 中等 |
| **OmniCast** | **最优**（频谱保持最佳） | **最优** |

OmniCast 在物理一致性上显著优于其他 DL 方法，甚至常优于所有数值基线。

### 推理效率对比

| 方法 | 硬件 | 15 天预报耗时 (0.25°) | 15 天预报耗时 (1.0°) |
|------|------|---------------------|---------------------|
| Gencast | TPUv5 | 480 秒 | 224 秒 |
| **OmniCast** | A100 | **29 秒** | **11 秒** |
| IFS-ENS | CPU 集群 | ~小时级 | - |

OmniCast 推理速度比 Gencast 快 **10-20 倍**（且使用更弱的硬件）。

### 消融实验

| 消融项 | 短期 RMSE | S2S RMSE | CRPS | SSR |
|--------|----------|---------|------|-----|
| 完整 OmniCast | 中 | **最优** | **最优** | **最优** |
| 无 MSE 目标 | 差（短期明显退化） | 差 | 差 | 接近 |
| MSE 全帧 | 好（短期好） | 差（强加确定性有害） | 差 | 差 |
| 短序列训练 ($T<44$) | **最优**（短期） | 差（误差累积） | 差 | 差 |
| 自回归解掩码 | 接近 | 接近 | 接近 | 差（欠散） |
| $\tau=1.0$ | 接近 | 接近 | 接近 | 差（欠散） |
| $\tau=1.3$ | 接近 | **最优** | **最优** | **最优** |

### 关键发现

- **全序列训练是 S2S 性能的关键**：短序列训练虽然中期更好，但 S2S 上因自回归误差累积而大幅退化
- **MSE 仅对前 10 帧有益**：与天气系统 ~10 天的确定性预测极限一致
- **随机解掩码顺序优于自回归和帧级别顺序**：引入额外随机性产生更多样的集合成员，SSR 更接近 1
- **扩散温度 $\tau=1.3$ 是最佳平衡**：过低导致欠散，过高则偏离均值预测
- **100 年稳定滚动预测**：OmniCast 可生成长达 100 年的稳定模拟输出

## 亮点与洞察

- 将计算机视觉中的掩码生成建模（MaskGIT）和视频生成（MAR）成功迁移到天气预报领域
- 连续 VAE + 扩散头的组合巧妙解决了天气数据高维度（100+ 变量）的编码难题
- 低维潜在空间 + 轻量扩散 MLP 的架构设计实现了数量级的推理加速
- 同一模型统一中期和 S2S 两个时间尺度，在两者上均有竞争力
- 近零偏差和优异的频谱保持是物理科学应用中至关重要的特性

## 局限与展望

- 短期和中期预报仍不如专门优化的自回归方法（如 Gencast），存在中短期-长期性能权衡
- S2S 实验使用较低分辨率（1.4°），是否能在高分辨率保持优势需验证
- VAE 重建质量与 Transformer 建模能力之间的权衡未充分研究
- 未能与 Gencast 和 NeuralGCM 在 S2S 上直接对比（对方计算需求过大）
- 时间维度未做压缩（预实验显示无明显收益），未来可进一步探索

## 相关工作与启发

- FourCastNet、GraphCast、Stormer 等自回归方法在中期取得突破，但 S2S 受限于误差累积
- ChaosBench 作为 S2S 标准基准提供了系统性比较框架
- 掩码生成模型（MaskGIT、MAR）在图像/视频生成中的成功为天气预报提供了跨领域方法论
- Gencast 的扩散方法在推理效率上的限制恰好是 OmniCast 的主要改进点

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 掩码潜在扩散架构统一中期/S2S 预报是全新思路
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖中期/S2S、确定性/概率/物理指标、全面消融、效率对比
- **写作质量**: ⭐⭐⭐⭐⭐ — 动机清晰，方法推导完整，图表丰富
- **价值**: ⭐⭐⭐⭐⭐ — 对气象和 AI for Science 社区有重要贡献，开源代码和模型

<!-- RELATED:START -->

## 相关论文

- [Graph-based Neural Space Weather Forecasting](graph-based_neural_space_weather_forecasting.md)
- [Probabilistic Weather Forecasting with Deterministic Guidance-Based Diffusion Model](../../ECCV2024/image_generation/probabilistic_weather_forecasting_with_deterministic_guidance-based_diffusion_mo.md)
- [Latent Diffusion Models with Masked AutoEncoders](../../ICCV2025/image_generation/latent_diffusion_models_with_masked_autoencoders.md)
- [StableGuard: Towards Unified Copyright Protection and Tamper Localization in Latent Diffusion Models](stableguard_towards_unified_copyright_protection_and_tamper_localization_in_late.md)
- [MGE-LDM: Joint Latent Diffusion for Simultaneous Music Generation and Source Extraction](mge-ldm_joint_latent_diffusion_for_simultaneous_music_generation_and_source_extr.md)

<!-- RELATED:END -->
