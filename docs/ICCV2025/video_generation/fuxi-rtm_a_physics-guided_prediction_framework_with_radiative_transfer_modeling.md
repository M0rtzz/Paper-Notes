---
title: >-
  [论文解读] FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling
description: >-
  [ICCV 2025][视频生成][物理引导深度学习] 提出 FuXi-RTM，首个将深度学习辐射传输模型 (DLRTM) 作为可微物理正则化器集成到天气预报框架中的混合物理引导体系，在 88.51% 的变量-预报时效组合上超越无约束基线。 当前深度学习天气预报模型（如 FuXi、Pangu-Weather 等）虽然在预报精…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "物理引导深度学习"
  - "辐射传输建模"
  - "天气预报"
  - "时空序列预测"
  - "物理一致性"
---

# FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling

**会议**: ICCV 2025  
**arXiv**: [2503.19940](https://arxiv.org/abs/2503.19940)  
**代码**: 无  
**领域**: 气象预测 / Weather Forecasting  
**关键词**: 物理引导深度学习, 辐射传输建模, 天气预报, 时空序列预测, 物理一致性

## 一句话总结

提出 FuXi-RTM，首个将深度学习辐射传输模型 (DLRTM) 作为可微物理正则化器集成到天气预报框架中的混合物理引导体系，在 88.51% 的变量-预报时效组合上超越无约束基线。

## 研究背景与动机

当前深度学习天气预报模型（如 FuXi、Pangu-Weather 等）虽然在预报精度上已超越 ECMWF 的 HRES，但存在一个根本缺陷：**缺乏显式物理约束**，可能产生非物理输出（如负湿度），这在辐射过程建模方面尤为突出。

辐射传输是地球天气和气候系统的主要能量驱动力，调控温度梯度、大气压力模式和风环流。传统 NWP 模型通过参数化方案模拟辐射传输，但计算成本极高。

本文的关键洞察：**可以用预训练的深度学习辐射传输替代模型 (DLRTM) 作为冻结的可微正则化器**，在训练时为预报模型提供物理约束信号，无需额外训练即可将辐射传输能力嵌入天气预报框架。

## 方法详解

### 整体框架

FuXi-RTM 采用 encoder-processor-decoder 范式，由两个核心组件组成：
- **FuXi-base**：可训练的主预报模型（1.1B 参数，30 层 Swin Transformer V2），输入连续两帧大气状态预测下一时步
- **DLRTM**：预训练且冻结的深度学习辐射传输替代模型，基于 Bi-LSTM 架构，处理大气柱数据输出辐射通量

训练时，FuXi-base 的输出经 DLRTM 生成辐射通量，与 RRTMG 传统模型计算的真值对比产生物理约束损失。

### 关键设计

1. **DLRTM 辐射传输替代模型**:

    - 架构：3 层 Bi-LSTM（正向96维 + 反向128维），逐列（grid point）独立处理
    - 输入：$\mathbf{Y}_t \in \mathbb{R}^{1 \times 71 \times H \times W}$，包含 13 个气压层 × 11 个变量（5 个高空变量 + 6 个单层变量）
    - 输出：4 种辐射通量（SWUFLX, SWDFLX, LWUFLX, LWDFLX）× 13 层
    - **Ghost Level 动态掩码**：处理地形高度差异，当 $P_{level} > P_{surface}$ 时掩码为 0，排除非物理层
    - 设计动机：DLRTM 独立处理每个大气柱，天然支持全球并行计算，将 RRTMG 的 22 分钟（8 CPU）降至 ~3 秒（1 H100 GPU）

2. **Sunlit Region-Centered (SRC) 采样策略**:

    - 问题：全球随机采样会包含大量无阳光区域（对短波辐射无意义）
    - 方案：动态选择一个受阳光照射的位置，取其周围 250×250 网格作为损失计算区域
    - 设计动机：集中在太阳辐射交互最显著的区域计算梯度，保留局部上下文关系，减少无关背景信号干扰

3. **物理引导训练策略**:

    - FuXi-RTM 训练时 DLRTM 参数冻结，仅更新 FuXi-base
    - 聚焦地表短波上行/下行通量（SWUFLX/SWDFLX）的约束
    - 通过反向传播，地表辐射约束隐式传递垂直大气交互信息
    - 设计动机：地表辐射约束本身已包含垂直大气柱信息，多层显式监督反而会造成信息冗余

### 损失函数 / 训练策略

总损失：$L_{total} = L_{forecast} + L_{reg}$

- **预报损失**：纬度加权 Charbonnier L1 损失
$$L_{forecast} = \frac{1}{C \times H \times W} \sum_c \sum_i \sum_j \alpha_i \sqrt{(\hat{X}_{c,i,j} - X_{c,i,j})^2 + \epsilon^2}$$
$\alpha_i = H \times \frac{\cos\Phi_i}{\sum_i \cos\Phi_i}$ 为纬度权重因子

- **物理正则化损失**：在 SRC 采样区域计算
$$L_{reg} = \frac{1}{R' \times H' \times W'} \sum_r \sum_i \sum_j \alpha_i (\lambda \sqrt{(\hat{Y}^{DLRTM} - Y^{DLRTM})^2 + \epsilon^2})$$
$\lambda = 10^{-3}$ 平衡物理约束与直接预报

- 训练配置：4×H100 GPU，60,000 iterations，AdamW（$\beta_1=0.9, \beta_2=0.95$），lr=2.5e-4

## 实验关键数据

### 主实验

在 2018-2022 年 5 年测试集上评估（每日 00/12 UTC 初始化，6 小时间隔至 10 天）：

| 指标 | FuXi-RTM vs FuXi-base |
|------|----------------------|
| 总体优势比例 | 3320 组合中 88.51% FuXi-RTM 更优 |
| 云覆盖 (CC) 优势比例 | 95.38% |
| 比湿 (Q) 优势比例 | 93.46% |
| CLWC (高层) 改进 | nRMSE 差异 > 2% |
| 地表反照率 (FAL) | 全 10 天预报 nRMSE 改进 > 7% |
| 辐射通量 | 100% 时效组合 FuXi-RTM 更优 |
| ISSRD 改进 | 接近 100% |
| DLRTM 加速比 | 22min (8 CPU) → 3s (1 GPU) |

### 消融实验

| 模型变体 | Q50 | Q500 | CLWC500 | CC150 | TCC | TTR | TP | FAL |
|---------|-----|------|---------|-------|-----|-----|-----|-----|
| FuXi-RTM-Random | 0.1857 | 0.7459 | 0.0226 | 0.1725 | 0.3193 | 159.25 | 2.4327 | 0.02515 |
| FuXi-RTM-13level | 0.1697 | 0.7409 | 0.0226 | 0.1732 | 0.3184 | 159.23 | 2.4418 | 0.02313 |
| FuXi-RTM-13levelSW | 0.1762 | 0.7409 | 0.0224 | 0.1714 | 0.3183 | 158.42 | 2.4200 | 0.02335 |
| FuXi-RTM-GSW | 0.1668 | 0.7567 | 0.0226 | 0.1735 | 0.3207 | 160.95 | 2.4564 | 0.02263 |
| FuXi-RTM-ISSRD | 0.1741 | 0.7441 | 0.0226 | 0.1719 | 0.3191 | 159.14 | 2.4299 | 0.02379 |
| **FuXi-RTM** | **0.1546** | **0.7388** | **0.0223** | **0.1705** | **0.3179** | **158.13** | **2.4092** | **0.02300** |
| FuXi-base | 0.1735 | 0.7453 | 0.0226 | 0.1720 | 0.3196 | 159.14 | 2.4127 | 0.02553 |

### 关键发现

- **SRC 优于随机采样**：在阳光区域集中计算梯度显著优于全球随机采样，空间连贯的梯度计算增强了特征学习
- **地表约束优于全层约束**：优化所有气压层（13level）反而性能下降，地表辐射约束通过反向传播隐式捕捉垂直交互
- **约束基础物理量优于导出量**：直接约束 SW 通量优于约束 ISSRD 或 GSW 等导出指标
- **能量守恒验证**：10 天以上预报中 FuXi-RTM 展现更好的全球大气总能量守恒特性

## 亮点与洞察

- **首个显式集成物理过程建模的 DL 天气预报框架**：不同于之前用 ODE 求解器处理原始方程的方法
- **冻结 DLRTM 作为可微正则化器**：无需额外训练即可扩展辐射传输能力，计算效率极高
- **地表约束的信息完备性**：一个看似简化的设计（仅约束地表 SW 通量）实际包含了完整的物理信息
- **与视频生成的类比**：天气预报本质上是多通道时空序列预测，但要求跨数十个物理关联变量的一致性

## 局限与展望

- 当前排除了风分量 (u, v)，扩展后可能进一步提升性能但计算需求增加
- 仅集成了辐射传输过程，未涵盖对流、行星边界层、云微物理等其他重要物理过程
- 某些变量（如 1000hPa Q）在短期预报中初期性能低于基线，需要 1.25 天才能超越
- SRC 采样策略仅适用于短波，长波辐射处理可能需要不同策略

## 相关工作与启发

- FuXi 系列模型的自回归预报范式为时空序列预测提供了成熟框架
- Yao et al. 发现 Bi-LSTM 在辐射建模上效果最佳，为 DLRTM 设计奠定基础
- 将 NWP 参数化方案替换为 DL 替代模型的思路可推广到其他物理过程

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将可微物理过程建模集成到 DL 天气预报中，开创性工作
- **实验充分度**: ⭐⭐⭐⭐⭐ 5 年测试集、3320 组合评估、详尽消融和物理守恒验证
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，物理背景和技术细节解释充分
- **价值**: ⭐⭐⭐⭐⭐ 为下一代物理一致的天气预报系统铺平道路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation](../../CVPR2025/video_generation/phyt2v_llm-guided_iterative_self-refinement_for_physics-grounded_text-to-video_g.md)
- [\[ICCV 2025\] OCK: Unsupervised Dynamic Video Prediction with Object-Centric Kinematics](ock_unsupervised_dynamic_video_prediction_with_object-centric_kinematics.md)
- [\[ICCV 2025\] VSRM: A Robust Mamba-Based Framework for Video Super-Resolution](vsrm_a_robust_mamba-based_framework_for_video_super-resolution.md)
- [\[CVPR 2025\] Unified Dense Prediction of Video Diffusion](../../CVPR2025/video_generation/unified_dense_prediction_of_video_diffusion.md)
- [\[CVPR 2026\] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](../../CVPR2026/video_generation/phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)

</div>

<!-- RELATED:END -->
