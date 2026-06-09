---
title: >-
  [论文解读] Eliminating Warping Shakes for Unsupervised Online Video Stitching
description: >-
  [ECCV 2024][LLM评测][视频拼接] 定义了视频拼接中的"warping shake"新问题（图像拼接扩展到视频时非重叠区域的时域抖动），提出StabStitch首个无监督在线视频拼接框架，通过拼接轨迹生成与平滑同时实现视频拼接和稳定，达到实时28.2ms/帧。
tags:
  - "ECCV 2024"
  - "LLM评测"
  - "视频拼接"
  - "视频稳定"
  - "warping shake"
  - "无监督学习"
  - "在线处理"
---

# Eliminating Warping Shakes for Unsupervised Online Video Stitching

**会议**: ECCV 2024  
**arXiv**: [2403.06378](https://arxiv.org/abs/2403.06378)  
**代码**: 有 ([https://github.com/nie-lang/StabStitch](https://github.com/nie-lang/StabStitch))  
**领域**: LLM评测  
**关键词**: 视频拼接, 视频稳定, warping shake, 无监督学习, 在线处理

## 一句话总结

定义了视频拼接中的"warping shake"新问题（图像拼接扩展到视频时非重叠区域的时域抖动），提出StabStitch首个无监督在线视频拼接框架，通过拼接轨迹生成与平滑同时实现视频拼接和稳定，达到实时28.2ms/帧。

## 研究背景与动机

视频拼接技术用于将不同视角的有限视野视频合成全景/宽视野画面，广泛应用于自动驾驶、视频监控和虚拟现实。本文聚焦于最常见且最具挑战性的手持相机视频拼接场景。

**核心问题——Warping Shake（变形抖动）**：

作者发现了一个被忽视的关键问题：当直接将图像拼接算法逐帧应用于视频时，尽管每一帧的拼接结果看起来自然合理，但在**非重叠区域**的时域相邻帧之间会出现明显的内容抖动。关键点在于：

**抖动不来自源视频**：得益于现代视频稳定技术（硬件+软件），手持相机拍摄的源视频通常是稳定的。

**抖动来自拼接变形**：图像拼接方法（如LPC、UDIS++）各自独立地估计每帧的空间变形，虽然每帧都保持了良好的结构，但帧间变形在时域上不连续，导致变形后的非重叠区域内容抖动。

**定义**：warping shake = 不论源视频是否稳定，由时域不平滑的空间变形引起的非重叠区域时域不稳定性。

**现有视频拼接方案的局限**：

- **假设过时**：Nie et al.等先前方法假设手持相机视频必然有严重抖动，对每个源视频都需单独稳定，不符合当今便携设备普遍内置防抖的现实
- **优化复杂**：联合优化拼接和稳定需建立复杂的非线性求解系统，需多轮迭代分步优化不同参数，推理速度极慢
- **鲁棒性差**：复杂的优化过程对输入视频质量要求苛刻（需足够多、准确且均匀分布的匹配点），实际应用中系统脆弱
- **离线处理**：所有现有方法都需要完整视频输入，无法在线实时处理

**StabStitch的思路**：既然源视频本就稳定，问题简化为——**消除拼接变形引入的抖动**，而非传统的同时稳定源视频和拼接。通过借鉴视频稳定中相机轨迹的概念，推导出视频拼接中的"拼接轨迹"，对其进行平滑以实现同时拼接和稳定。

## 方法详解

### 整体框架

StabStitch包含三个核心组件：

1. 时域变形模型（Temporal Warp Model）：估计同一视频相邻帧间的运动
2. 空间变形模型（Spatial Warp Model）：估计不同视角间的空间对齐
3. 变形平滑模型（Warp Smoothing Model）：平滑拼接轨迹，消除warping shake

### 关键设计

1. **时域变形模型（TNet）**：用CNN估计相邻帧间的网格控制点运动 $m^T(t)$。

    - 基于UDIS++架构简化，使用ResNet18替代ResNet50，用局部相关层（cost volume）替代全局相关层（因为相邻帧运动通常较小）
    - 优化目标：$\mathcal{L}^{tmp} = \mathcal{L}_{alignment} + \lambda^{tmp} \mathcal{L}_{distortion}$
    - **设计动机**：使用CNN替代传统特征点方法估计运动，在低光照、低纹理等困难场景中更鲁棒。

2. **Stitch-Meshflow（拼接轨迹推导）**：创新性地将空间变形和时域变形结合，推导出拼接轨迹。

    - 相机轨迹（视频稳定中）定义为时域运动的累积：$C_i(t) = m_i(1) + m_i(2) + \cdots + m_i(t)$
    - 空间变形模型（SNet）估计参考帧和目标帧间的网格运动 $m^S$，额外添加运动一致性约束：
    $\mathcal{L}_{consis.} = \frac{1}{(U+1)(V+1)} \sum_{i} \|m_i(t) - m_i(t-1) - \mu^{spt}\|_2$
    - **拼接运动推导**：通过TPS变换将时域运动映射到变形后空间，得到拼接运动：
    $s(t) = \mathcal{T}_{M^{Rig} \to M^S(t-1)}(M^T(t)) - M^S(t)$
    - 拼接轨迹为拼接运动的累积：$S_i(t) = s_i(1) + s_i(2) + \cdots + s_i(t)$
    - **设计动机**：要对齐变形视频中第 $t$ 帧与第 $t-1$ 帧，不能直接用源视频的时域运动，需要考虑第 $t-1$ 帧经过空间变形后的坐标变换。通过TPS将时域运动"投射"到变形后的空间，再与当前帧的实际空间变形做差，就得到变形视频中每帧之间的残差运动。

3. **变形平滑模型（SmoothNet）**：3D卷积网络，输入拼接轨迹序列、空间网格序列和重叠区域掩码，输出平滑增量 $\Delta$。

    - 平滑后的轨迹：$\hat{S} = S + \Delta$
    - 平滑后的空间网格：$\hat{M}^S = M^S - \Delta$
    - **四项优化目标的综合设计**：
        - **数据项** $\mathcal{L}_{data} = \|(\hat{S}-S)(\alpha \cdot OP + 1)\|_2$：平滑轨迹不应偏离原始轨迹太远，对重叠区域加权更大（乘以 $\alpha \cdot OP + 1$），保护对齐质量
        - **平滑项** $\mathcal{L}_{smoothness}$：约束轨迹在任意时刻的位置应位于前后时刻的中点，隐式满足无突然旋转和一致平移幅度
        - **空间一致性项** $\mathcal{L}_{space} = \frac{1}{N} \sum_t \mathcal{L}_{distortion}(\hat{M}^S(t))$：确保所有 $(U+1) \times (V+1)$ 个轨迹的变化保持一致，避免独立优化导致的空间畸变
        - **在线协作项** $\mathcal{L}_{online} = \frac{1}{N-1} \sum_t \|\hat{S}^{(\xi)}(t) - \hat{S}^{(\xi+1)}(t-1)\|_2$：确保相邻滑动窗口中重叠帧的平滑轨迹一致

### 损失函数 / 训练策略

三个模型分别独立训练：

**时域变形**：$\mathcal{L}^{tmp} = \mathcal{L}_{alignment} + \lambda^{tmp} \mathcal{L}_{distortion}$（$\lambda^{tmp}=5$）

**空间变形**：$\mathcal{L}^{spt} = \mathcal{L}_{alignment} + \lambda^{spt} \mathcal{L}_{distortion} + \omega^{spt} \mathcal{L}_{consis.}$（$\lambda^{spt}=10, \omega^{spt}=0.1$）

**变形平滑**：$\mathcal{L}^{smooth} = \mathcal{L}_{data} + \lambda^{smooth} \mathcal{L}_{smoothness} + \omega^{smooth} \mathcal{L}_{space}$（$\lambda^{smooth}=50, \omega^{smooth}=10$），在线模式额外加 $\mathcal{L}_{online}$。

控制点分辨率 $(6+1) \times (8+1)$，滑动窗口长度7帧。

## 实验关键数据

### 主实验

与图像拼接方法在StabStitch-D数据集上对比（PSNR/SSIM）：

| 方法 | Regular | Low-Light | Low-Texture | Fast-Moving | Average |
|------|---------|-----------|-------------|-------------|---------|
| LPC | 24.22/0.812 | 失败 | 失败 | 23.88/0.813 | - |
| UDIS++ | 23.19/0.785 | 31.09/0.936 | 29.98/0.906 | 21.56/0.756 | 27.19/0.859 |
| UDIS++* (重训) | 24.63/0.829 | 34.26/0.957 | 32.81/0.920 | **24.78/0.819** | 29.78/0.891 |
| **StabStitch** | **24.64/0.832** | **34.51/0.958** | **33.63/0.927** | 23.36/0.787 | **29.89/0.890** |

与视频拼接方法用户偏好研究（排除Nie et al.失败案例后）：

| StabStitch偏好 | Nie et al.偏好 | 无偏好 |
|--------------|--------------|--------|
| **30.47%** | 6.25% | 63.28% |

### 消融实验

| 配置 | Alignment↑ | Distortion↓ | Stability↓ |
|------|-----------|-------------|------------|
| Basic Stitching | 30.67/0.902 | 0.784 | 81.57 |
| + $\mathcal{L}_{consis.}$ | **30.75/0.903** | 0.804 | 60.32 |
| + $\mathcal{L}_{consis.}$ + Warp Smoothing | 29.89/0.890 | **0.674** | **48.74** |

推理速度分析（RTX 4090Ti）：

| SNet | TNet | 轨迹生成 | SmoothNet | Warping | Blending | 总计 |
|------|------|---------|-----------|---------|----------|------|
| 11.5ms | 10ms | 1.1ms | 1ms | 4.4ms | 0.2ms | **28.2ms** |

### 关键发现

- **warping shake确实存在且严重**：即使输入视频稳定，基础图像拼接的稳定性得分高达81.57，StabStitch降至48.74
- **运动一致性约束有效**：$\mathcal{L}_{consis.}$ 使稳定性从81.57降到60.32，同时略微提升对齐质量
- **牺牲少量对齐换取大量稳定**：完整StabStitch对齐性能仅下降2.8%（30.75→29.89），但稳定性提升40.2%
- **场景鲁棒性极强**：Nie et al.在20对测试视频中有10对失败（低光照和低纹理），StabStitch全部成功
- **实时性**：28.2ms/帧 vs Nie et al.的40+分钟/26秒视频，速度提升数万倍

## 亮点与洞察

- 精准定义了"warping shake"这一新问题——看似稳定的输入经过拼接后产生不稳定，这个现象在工程实践中很容易遇到但之前未被系统研究
- Stitch-Meshflow的推导是关键技术贡献：巧妙地将空间变形和时域变形整合为拼接轨迹，使得视频稳定的成熟技术可直接应用于视频拼接
- 3D卷积平滑网络设计简洁高效（仅3层），突出了无监督学习scheme的有效性
- 在线模式仅需1帧延迟，通过滑动窗口和在线协作约束实现

## 局限与展望

- 以稳定输入为前提——如果源视频本身就有大幅抖动（如极端运动），需要先单独稳定
- Fast-Moving场景的对齐性能不如UDIS++重训版本（23.36 vs 24.78），因为平滑会牺牲部分对齐
- 数据集规模相对有限（100+视频对），更大规模数据可能带来进一步提升
- 仅支持两个视频的拼接，未扩展到多视频全景拼接

## 相关工作与启发

- MeshFlow的轨迹表示被创新性地扩展到拼接场景，从camera path到stitching path的推广思路值得借鉴
- UDIS++的空间变形网络被高效改造（ResNet50→18，全局相关→局部相关），兼顾性能和效率
- "先拼接后稳定"vs"先稳定后拼接"的传统二选一被统一框架一步解决
- 构建的StabStitch-D数据集可作为视频拼接领域的标准评测基准

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次定义warping shake问题，Stitch-Meshflow推导新颖
- **实验充分度**: ⭐⭐⭐⭐ 多场景评测+用户研究+速度分析+消融完整
- **写作质量**: ⭐⭐⭐⭐⭐ 问题定义清晰，公式推导严谨，图示说明力强
- **价值**: ⭐⭐⭐⭐⭐ 高实用价值——实时在线视频拼接系统，开源代码和数据集

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] SIGMA: Sinkhorn-Guided Masked Video Modeling](sigma_sinkhorn-guided_masked_video_modeling.md)
- [\[ECCV 2024\] Deep Cost Ray Fusion for Sparse Depth Video Completion](deep_cost_ray_fusion_for_sparse_depth_video_completion.md)
- [\[ECCV 2024\] Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)
- [\[ECCV 2024\] ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization](colormnet_a_memory-based_deep_spatial-temporal_feature_propagation_network_for_v.md)
- [\[NeurIPS 2025\] Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator](../../NeurIPS2025/llm_evaluation/your_pre-trained_llm_is_secretly_an_unsupervised_confidence_calibrator.md)

</div>

<!-- RELATED:END -->
