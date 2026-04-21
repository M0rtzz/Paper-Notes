---
title: >-
  [论文解读] Coded-E2LF: Coded Aperture Light Field Imaging from Events
description: >-
  [CVPR2026][light field imaging] 首次证明仅用 event camera（无需传统 intensity 图像）即可重建像素级精度的 4D 光场，提出 Coded-E2LF 系统：通过编码光圈序列触发 events 并累积为 event images，利用全黑 pattern 建立 event-based 与 intensity-based coded aperture imaging 的数学等价性，结合端到端 deep optics 训练实现 8×8 视点光场重建。
tags:
  - CVPR2026
  - light field imaging
  - event camera
  - coded aperture
  - deep optics
  - end-to-end optimization
  - black-first coding sequence
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Coded-E2LF: Coded Aperture Light Field Imaging from Events

**会议**: CVPR2026  
**arXiv**: [2602.22620](https://arxiv.org/abs/2602.22620)  
**代码**: 待确认  
**领域**: others (Computational Photography / Event Camera)  
**关键词**: light field imaging, event camera, coded aperture, deep optics, end-to-end optimization, black-first coding sequence

## 一句话总结

首次证明仅用 event camera（无需传统 intensity 图像）即可重建像素级精度的 4D 光场，提出 Coded-E2LF 系统：通过编码光圈序列触发 events 并累积为 event images，利用全黑 pattern 建立 event-based 与 intensity-based coded aperture imaging 的数学等价性，结合端到端 deep optics 训练实现 8×8 视点光场重建。

## 研究背景与动机

**光场成像的价值与局限**：4D 光场记录了场景中光线的空间和角度信息，可用于数字重聚焦、深度估计、视点合成等应用。传统光场相机（如 Lytro）使用微透镜阵列，空间分辨率与角度分辨率之间存在固有的分辨率折中

**编码光圈方法的进展**：coded aperture 通过在镜头光圈上施加已知编码 pattern，将角度信息编码到单张 2D 图像中，后端计算重建光场。这避免了微透镜的分辨率损失，但重建质量依赖于编码设计和解码算法

**传统编码光圈的限制**：基于 intensity 相机的编码光圈成像需要多次曝光（每次使用不同 pattern），受限于相机读出速度和场景动态——多次曝光间的物体运动会导致伪影

**Event camera 的独特优势**：event camera 异步地检测像素级亮度变化，具有微秒级时间分辨率、高动态范围 (120+ dB)、低功耗等特性。当 coded aperture pattern 切换时，即使场景完全静态，pattern 变化本身就会触发 events

**未被探索的结合**：event camera + coded aperture 的组合尚无先例——event camera 天然适合检测 pattern 切换引起的亮度变化，理论上可以极快速度完成多 pattern 采集，但 event 数据的非线性对数响应使得传统 coded aperture 理论不直接适用

## 核心问题

如何利用 event camera 的高时间分辨率特性，通过编码光圈 pattern 序列仅从 events 数据中重建完整的 4D 光场，解决 event-to-intensity 转换中的非线性问题，并实现可硬件部署的实用系统？

## 方法详解

### 系统概述

Coded-E2LF 系统由三部分组成：**(1)** 硬件层——可编程光圈 + event camera；**(2)** 编码理论——black pattern 等价性定理 + BF 编码序列；**(3)** 网络层——AcqNet (学习编码 pattern) + RecNet (重建光场) 端到端训练。

### 编码光圈 + Event Camera 成像模型

- **编码过程**：$N$ 个编码 pattern $\{a^{(n)}\}_{n=1}^{N}$ 依次施加于光圈，每个 pattern $a^{(n)} \in \{0, 1\}^{u \times v}$（$u \times v$ 为角度分辨率，如 $8 \times 8$），控制对应子光圈的开关
- **静态场景假设**：场景在 pattern 序列切换期间保持静态（约 20ms），pattern 切换是唯一触发 events 的亮度变化来源
- **Event 累积**：pattern 从 $a^{(n-1)}$ 切换到 $a^{(n)}$ 时触发的 events 可累积为 event image：
  $$E^{(n-1,n)}(x) = \log I^{(n)}(x) - \log I^{(n-1)}(x)$$
  其中 $I^{(n)}(x) = \sum_{s,t} a^{(n)}(s,t) \cdot L(x, s, t)$ 是 pattern $a^{(n)}$ 下的强度图像，$L(x,s,t)$ 是待重建的光场

### 关键理论：Black Pattern 的作用

- **核心定理 (Eq. 8)**：若编码序列中包含一个全黑 pattern $a^{(n_B)} = \mathbf{0}$（即光圈完全关闭），则有：
  $$E^{(n_B, n)}(x) = \log I^{(n)}(x) - \log I^{(n_B)}(x) = \log I^{(n)}(x) + C$$
  因为 $I^{(n_B)} = 0$ 时需特殊处理——实际中 event camera 有暗电流基底 $I_{\text{dark}}$，使得 $\log I^{(n_B)}$ 为常数 $C$
- **等价性**：上式说明包含黑 pattern 的 event image 与 intensity-based coded aperture image 仅差一个全局常数，因此传统编码光圈理论的解码方法可以直接适用
- **Pattern 置换近似不变性**：在黑 pattern 参与下，不同 pattern 顺序生成的 event images 近似等价（因为黑 pattern 提供了统一的参考基准），简化了编码设计

### Black-First Coding Sequence (BF)

- **设计**：将黑 pattern 固定为序列第一个（$a^{(1)} = \mathbf{0}$），后续 $N-1$ 个 pattern 依次施加
- **优势**：
    - 从第一个黑 pattern 到各后续 pattern 的 event images $\{E^{(1,n)}\}_{n=2}^{N}$ 直接对应 intensity-based 测量
    - 大幅减少 event 数量——相比任意序列，BF 避免了相邻非零 pattern 之间的冗余 events
    - $N-1$ 个 event images 即可重建 $u \times v$ 视点的完整光场
- **实测效率**：20ms 左右即可完成整个编码序列的采集

### Reference-Aware Event Generation (RA)

- **动机**：event camera 的对数响应和阈值机制使得简单的 event 累积存在误差
- **方法**：显式追踪参考强度 $I_{\text{ref}}$，准确模拟 event 生成过程：
  $$e_k = \begin{cases} +1 & \text{if } \log I(x_k, t_k) - \log I_{\text{ref}}(x_k) \geq C_{\text{pos}} \\ -1 & \text{if } \log I(x_k, t_k) - \log I_{\text{ref}}(x_k) \leq -C_{\text{neg}} \end{cases}$$
  每触发一次 event，$I_{\text{ref}}$ 随之更新
- **在训练中**：RA 作为可微分的 event 生成模拟器，使编码 pattern 的优化梯度可以准确回传

### 端到端 Deep Optics 训练

- **AcqNet（学习编码 pattern）**：输入随机初始化的连续 pattern $\tilde{a}^{(n)} \in [0,1]^{u \times v}$，训练收敛后二值化为 $a^{(n)} \in \{0,1\}^{u \times v}$
- **RecNet（重建光场）**：接收 $N-1$ 个 event images，输出完整光场 $\hat{L} \in \mathbb{R}^{H \times W \times u \times v}$
    - 架构：CNN-based encoder-decoder，spatial 和 angular 维度分别处理后融合
- **损失函数**：$\mathcal{L} = \mathcal{L}_{\text{recon}}(\hat{L}, L_{\text{GT}}) + \gamma \cdot \mathcal{L}_{\text{binary}}$
    - $\mathcal{L}_{\text{recon}}$：光场重建的 L1 + SSIM 损失
    - $\mathcal{L}_{\text{binary}}$：鼓励 pattern 趋于二值的正则化
- **训练流程**：前向——AcqNet 生成 pattern → RA 模拟 events → RecNet 重建光场；反向——梯度穿过整个 pipeline 联合优化编码和解码

## 实验

### 实验设置

- **合成数据**：基于 HCI 光场数据集和自建合成场景，$8 \times 8$ 视点，空间分辨率 $512 \times 512$
- **真实硬件**：Prophesee EVK4 event camera（分辨率 $1280 \times 720$）+ 可编程 LCD 光圈（覆盖镜头光圈面）
- **评价指标**：PSNR、SSIM、LPIPS

### 合成数据结果

| 方法 | #Patterns | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|-----------|--------|--------|---------|
| Intensity-based coded aperture | 9 | 34.2 | 0.952 | 0.041 |
| Naive event accumulation | 9 | 28.7 | 0.891 | 0.098 |
| Coded-E2LF (random patterns) | 9 | 33.5 | 0.945 | 0.048 |
| **Coded-E2LF (learned, BF)** | **9** | **35.1** | **0.961** | **0.035** |

- 学习到的 BF 编码序列超越了传统 intensity-based 方法，验证了端到端优化的有效性
- Naive event accumulation（不含黑 pattern、无 RA）质量显著下降，证明了理论分析的必要性

### 真实硬件验证

- 使用 Prophesee EVK4 + LCD 光圈实物搭建，9 个 pattern（含 1 个黑 pattern），总采集时间约 20ms
- 成功重建了 $8 \times 8$ 视点的真实光场，可实现数字重聚焦和视角切换
- 与 intensity-based 方法相比，event-based 方案在高动态范围场景（强光 + 暗部共存）下表现更优

### 消融实验

| 配置 | PSNR |
|------|------|
| 无黑 pattern (任意 N 个非零 pattern) | 29.4 |
| 有黑 pattern + 随机位置 | 33.8 |
| 有黑 pattern + BF (固定首位) | **35.1** |
| BF + 无 RA | 33.2 |
| BF + RA (完整) | **35.1** |

- 黑 pattern 是性能跳跃的关键（+4.4 dB）
- BF 序列比随机放置黑 pattern 进一步提升 1.3 dB
- RA 模块贡献 1.9 dB，准确的 event 生成建模不可忽略

## 亮点

- **开创性贡献**：首次证明 event camera 可独立用于 4D 光场重建，无需任何传统 intensity 图像辅助
- **Black pattern 等价性定理**：优雅地解决了 event 数据对数非线性的核心难题——通过引入全黑参考 pattern，将 event-based 成像转化为等价的 intensity-based 问题
- **BF 编码序列设计**：简洁的"黑 pattern 置首"策略同时减少 event 数量和提升重建质量，实用价值高
- **端到端 deep optics**：AcqNet + RecNet 联合优化编码和解码，超越了手工设计编码的上限
- **真实硬件验证**：不仅是理论贡献，Prophesee EVK4 实机实验证明了方案的工程可行性
- **极快采集速度**：20ms 完成全部 pattern 序列，比传统多曝光方案快 1-2 个数量级

## 局限性

- 静态场景假设限制了应用范围——20ms 内的场景运动仍会引入伪影，动态场景需额外运动补偿
- LCD 光圈的切换速度（约 2ms/pattern）是采集速度的瓶颈，换用 DMD（微秒级切换）可进一步加速
- 当前 $8 \times 8$ 角度分辨率需 9 次 pattern 切换，更高角度分辨率将线性增加采集时间
- Event camera 的暗电流和噪声在低光照场景下可能降低 event image 质量
- RecNet 的 CNN 架构对极高空间分辨率（如 4K）的可扩展性有待验证
- 仅验证了静态室内场景，室外/长距离/大基线场景未涉及

## 相关工作

- **传统光场相机**：Lytro、RayTrix（微透镜阵列）— 空间/角度分辨率折中严重
- **Coded aperture 光场**：Veeraraghavan et al. (2007)、Marwah et al. (2013) — 基于 intensity 相机的编码光圈，多次曝光受动态场景限制
- **Deep optics**：Sitzmann et al. (2018)、Chang & Wetzstein (2019) — 端到端优化光学编码 + 计算解码，但均基于 intensity 相机
- **Event camera 3D 重建**：E2VID、ESIM、EventNeRF — event 用于深度估计或 NeRF，但未做完整光场重建
- **Event-based HDR**：Han et al. (2020)、Rebecq et al. (2019) — 利用 event camera 高动态范围优势，与本文互补
- **Compressive light field**：Kamal et al. (2016) — 压缩感知框架重建光场，本文的端到端方法性能更优

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将 event camera 引入编码光圈光场成像，black pattern 等价性定理具有理论原创性
- 实验充分度: ⭐⭐⭐⭐ — 合成 + 真实硬件验证 + 消融完整，但真实场景多样性有限
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，从物理模型到系统设计逻辑通顺
- 价值: ⭐⭐⭐⭐⭐ — 开辟了 event-based 计算光场成像新方向，理论贡献与工程实践俱全
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Revisiting the Predictability of Performative, Social Events](../../ICML2025/others/revisiting_the_predictability_of_performative_social_events.md)
- [Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Cameras](../../AAAI2026/others/spike_imaging_velocimetry_dense_motion_estimation_of_fluids_using_spike_cameras.md)
- [Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sen.md)
- [POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction](polishing_the_sky_widefield_and_highdynamic_range.md)
- [Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion](../../AAAI2026/others/towards_temporal_fusion_beyond_the_field_of_view_for_camera-based_semantic_scene.md)

<!-- RELATED:END -->
