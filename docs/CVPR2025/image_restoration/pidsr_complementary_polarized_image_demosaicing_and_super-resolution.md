---
title: >-
  [论文解读] PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution
description: >-
  [CVPR 2025][图像恢复][偏振图像去马赛克] PIDSR 提出了一个将偏振图像去马赛克（PID）和超分辨率（PISR）联合互补优化的框架，通过两阶段循环管线（空间-物理相干重建 + 偏振感知分辨率增强）和 Stokes 辅助网络，从 CPFA 原始图像直接获得高质量高分辨率偏振图像，显著减少了 DoP 和 AoP 的误差。
tags:
  - CVPR 2025
  - 图像恢复
  - 偏振图像去马赛克
  - 超分辨率
  - Stokes参数
  - 两阶段循环
  - 偏振度与偏振角
---

# PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution

**会议**: CVPR 2025  
**arXiv**: [2504.07758](https://arxiv.org/abs/2504.07758)  
**代码**: https://github.com/PRIS-CV/PIDSR  
**领域**: 图像复原 / 偏振成像  
**关键词**: 偏振图像去马赛克, 超分辨率, Stokes参数, 两阶段循环, 偏振度与偏振角

## 一句话总结

PIDSR 提出了一个将偏振图像去马赛克（PID）和超分辨率（PISR）联合互补优化的框架，通过两阶段循环管线（空间-物理相干重建 + 偏振感知分辨率增强）和 Stokes 辅助网络，从 CPFA 原始图像直接获得高质量高分辨率偏振图像，显著减少了 DoP 和 AoP 的误差。

## 研究背景与动机

**领域现状**：偏振相机通过焦平面分割技术（DoFP）在单次曝光中捕获四个偏振方向（0°、45°、90°、135°）的图像，为偏振视觉任务（如形状估计、反射去除、去雾、HDR）提供了便利。但相机的直接输出是 CPFA（彩色偏振滤波阵列）原始图像，每个像素只有一个颜色通道+一个偏振方向的信息，需要去马赛克才能重建完整的偏振图像。

**现有痛点**：(1) 去马赛克不可避免地引入伪影（artifacts），且由于 DoP 和 AoP 与偏振图像之间的非线性关系（$p = \sqrt{S_1^2+S_2^2}/S_0$, $\theta = \frac{1}{2}\arctan(S_2/S_1)$），去马赛克误差在 DoP/AoP 上被显著放大；(2) 偏振相机的分辨率受硬件限制远低于普通 RGB 相机；(3) 现有 PID 方法无法提升分辨率，PISR 方法假设输入没有去马赛克伪影但实际上总有——串行执行 PID→PISR 导致误差累积。

**核心矛盾**：PID 和 PISR 被视为独立任务串行处理，但实际上它们是互补的——更高分辨率可以减轻去马赛克伪影（验证实验表明误差率随分辨率增加而降低），而更少的去马赛克伪影可以提升超分效果。串行流水线无法利用这种互补性。

**本文目标**：设计一个联合框架 $\mathcal{D}^{\uparrow}$，从 CPFA 原始图像直接同时输出去马赛克结果和高分辨率偏振图像，且 DoP/AoP 比串行方法更准确。

**切入角度**：作者发现 CPFA 原始图像可以近似转换为 4 张半分辨率全彩偏振图像（分离偏振方向后做简单 RGB 去马赛克），这使得 PID 可以分解为"空间不连续修复"和"分辨率提升"两个子问题，与 PISR 的子问题形成统一结构。

**核心 idea**：将 PID 和 PISR 统一为循环结构，每轮迭代包含空间-物理相干重建（intra-resolution）和分辨率增强（cross-resolution）两个阶段，通过 Stokes 参数注入物理先验，循环执行 n 轮实现 $2^n$ 倍超分。

## 方法详解

### 整体框架

输入 CPFA 原始图像 $R$ 首先转换为 4 张半分辨率全彩偏振图像 $R_{\alpha_{1,2,3,4}}$（按偏振方向分离像素 + 双线性 RGB 去马赛克）。然后交替通过两个阶段：Stage 1（$f$, 空间-物理相干重建器）修复空间不连续和物理相关性，Stage 2（$g$, 偏振感知分辨率增强器）做 2× 超分。第一轮循环完成去马赛克得到全分辨率图像；后续每轮循环再做一次 2× 超分。$n$ 轮循环实现 $k = 2^n$ 倍超分。

### 关键设计

1. **两阶段循环管线（Recurrent PIDSR Pipeline）**:

    - 功能：将去马赛克和超分统一为可循环执行的结构，使两者互补优化
    - 核心思路：关键观察是 CPFA 原始图像可以得到 4 张半分辨率偏振图像，因此 PID 等价于"修复空间不连续"（intra-resolution）+"2× 分辨率提升"（cross-resolution）。同样 PISR 可分解为"恢复物理相关性"（intra）+"分辨率提升"（cross）。将 intra 和 cross 解耦后统一为两个阶段，形成负反馈循环：去马赛克的改善减少了超分的输入误差，超分的改善减少了去马赛克的伪影
    - 设计动机：朴素串行的 $\mathcal{D}$ 后接 $\uparrow$ 有两个致命问题：(1) 两阶段独立，误差单向累积无负反馈；(2) 无法利用互补性。循环结构解决了这两个问题

2. **Stokes 特征注入（SFI）块**:

    - 功能：将偏振物理先验（Stokes 参数 $S_1, S_2$）显式注入网络特征中，保持偏振性质
    - 核心思路：SFI 块包含两个分支——输入特征分支（带 MDTA 注意力模块）和 Stokes 特征分支，两个分支的处理结果相乘生成偏置来调整输入特征。嵌入修改后的 U-Net 中，替换标准卷积块。Stokes 参数包含高频物理信息（$S_1$ 描述水平/垂直偏振差异，$S_2$ 描述 45°/135° 差异），与图像特征的低频结构形成互补
    - 设计动机：如果直接 concat 图像特征和 Stokes 特征，两者的 domain gap 太大——图像特征主要是低频结构，Stokes 特征主要是高频信息。SFI 的乘法调制方式能更好地桥接 domain gap，类似 FiLM

3. **偏振感知分辨率增强器（Stage g）**:

    - 功能：在保持偏振性质的前提下进行 2× 超分辨率
    - 核心思路：直接复用 Stage f 编码器最粗层特征（不再对中间结果提取特征，避免重复计算），通过另一个解码器+Stokes 特征注入头（$\mathcal{F}_s^g$ 处理修复后更准确的 Stokes 参数 $S_{1,2}^b$）进行上采样。输出特征经过特征精修块 $\mathcal{A}^g$ 和上采样块 $\mathcal{U}$ 重建残差
    - 设计动机：Stage f 已修复了空间不连续和物理相关性，此时的 Stokes 参数更可靠，可以更好地指导超分过程。跳过重复的特征提取也提升了效率

### 损失函数 / 训练策略

总损失 $L = \lambda_1 L_{img} + \lambda_2 L_{Stokes} + \lambda_3 L_{pol}$，$\lambda_1 = 1.0, \lambda_2 = 10.0, \lambda_3 = 10.0$。

- **图像损失** $L_{img}$：$L_1(I_{\alpha_1}+I_{\alpha_3}, I_{\alpha_2}+I_{\alpha_4})$（利用偏振恒等式约束）+ 梯度损失
- **Stokes 损失** $L_{Stokes}$：$S_0$ 的梯度损失 + $S_{1,2}$ 的 L1 损失
- **偏振损失** $L_{pol}$：DoP 和 AoP 的 L1 损失

使用 Mitsuba 3 渲染的合成数据集训练，Adam 优化器，学习率 0.005，训练 100 epochs，NVIDIA A800 GPU。两个阶段同时训练。

## 实验关键数据

### 主实验

| 方法 | 去马赛克 $S_0$ PSNR↑ | DoP PSNR↑ | AoP MAE↓ | 2×SR $S_0^{HR}$ PSNR↑ | 2×SR DoP PSNR↑ |
|------|---------------------|-----------|----------|----------------------|----------------|
| Polanalyser | 33.28 | 26.68 | 17.87 | - | - |
| IGRI2 | 35.50 | 27.78 | 16.58 | - | - |
| TCPDNet | 38.65 | 32.26 | 13.19 | - | - |
| **PIDSR (demosaic)** | **40.24** | **33.33** | **12.24** | - | - |
| PSRNet (2×) | - | - | - | 36.46 | 32.01 |
| CPSRNet (2×) | - | - | - | 33.60 | 24.14 |
| **PIDSR (2×)** | - | - | - | **37.44** | **32.97** |

### 消融实验

| 配置 | $S_0$ PSNR↑ | DoP PSNR↑ | AoP MAE↓ |
|------|-------------|-----------|----------|
| Sequential $\mathcal{D}$ and $\uparrow$ | 32.32 | 23.78 | 19.29 |
| Single-stage pipeline | 34.61 | 27.95 | 38.22 |
| Without SFI blocks | 37.18 | 32.73 | 13.12 |
| Ours (demosaic) → PSRNet | 40.24 | 33.33 | 12.24 |
| TCPDNet → ours (SR only) | 38.65 | 32.26 | 13.19 |
| **Complete PIDSR** | **40.24** | **33.33** | **12.24** |

### 关键发现

- 去马赛克上 PIDSR 比最强基线 TCPDNet 提升了 1.59 dB（$S_0$）和 1.07 dB（DoP），AoP 误差降低 0.95°
- 串行方法（Sequential $\mathcal{D}$ and $\uparrow$）表现最差，证明误差累积问题严重
- 单阶段管线的 AoP MAE 暴涨到 38.22°——因为空间不连续未修复就直接超分，物理相关性被破坏
- 去掉 SFI 块后 DoP 掉 0.6 dB，说明 Stokes 物理先验注入确实帮助了偏振性质保持
- 互补效果验证：用 PIDSR 做去马赛克再用 PSRNet 做 SR ≈ 用完整 PIDSR 做联合优化——说明 PIDSR 的去马赛克质量已经很高；但反过来用 TCPDNet 去马赛克再用 PIDSR 做 SR 则弱于完整 PIDSR——说明 SR 阶段确实从更好的去马赛克中受益
- 在真实数据上 PIDSR 的 AoP 和 DoP 没有明显的锯齿状伪影，其他方法有

## 亮点与洞察

- **互补性的理论和实验双重验证**：不只是说 PID 和 PISR 互补，而是通过控制变量实验（不同分辨率下的去马赛克误差率）定量证明了高分辨率减少去马赛克伪影，再通过消融证明联合优化优于串行。这种论证方式非常扎实
- **CPFA → 半分辨率偏振图像的等价变换**：这个洞察是整个方法框架的基础——将 PID 重新定义为"空间修复 + 2× 超分"，从而自然地与 PISR 统一。这是一个优雅的问题重构
- **Stokes 参数作为物理先验注入**：利用偏振成像特有的物理约束（如 $I_{\alpha_1}+I_{\alpha_3} = I_{\alpha_2}+I_{\alpha_4}$）作为损失正则化，并通过 SFI 块注入网络。这种 physics-informed 的设计比纯数据驱动更可靠

## 局限与展望

- 仅处理单帧 CPFA 原始图像，不支持偏振视频序列
- 不能处理非偏振的 CFA 原始图像（因为需要 Stokes 参数输入）
- 合成训练数据与真实偏振相机数据之间可能存在 domain gap
- 循环多轮（如 4× 超分需要 2 轮）会增加推理时间
- 未探索更大超分倍数（如 8×）或与其他低级视觉任务（去噪、HDR）的联合

## 相关工作与启发

- **vs TCPDNet**：当前最强的偏振去马赛克方法，使用 CNN 直接从 CPFA 重建偏振图像。PIDSR 通过联合超分获得了更好的去马赛克结果，说明任务间互补确实有效
- **vs PSRNet/CPSRNet**：现有偏振超分方法假设输入无伪影，导致在真实场景（有去马赛克伪影）中效果打折。PIDSR 通过联合优化从根本上解决了这个假设不成立的问题
- **vs 传统 RGB 去马赛克+超分**：偏振场景比 RGB 更复杂（12 个通道 vs 3 个通道，非线性 DoP/AoP 计算），不能直接用 RGB 方法。PIDSR 的 Stokes 先验注入是偏振成像独有的设计

## 评分

- 新颖性: ⭐⭐⭐⭐ PID 和 PISR 互补的观察是新的，两阶段循环管线设计优雅
- 实验充分度: ⭐⭐⭐⭐ 合成和真实数据，全面消融，下游任务验证
- 写作质量: ⭐⭐⭐⭐ 问题建模和动机推导非常清晰
- 价值: ⭐⭐⭐⭐ 对偏振成像社区很有价值，互补优化的思路也可泛化

<!-- RELATED:START -->

## 相关论文

- [Augmenting Perceptual Super-Resolution via Image Quality Predictors](augmenting_perceptual_super-resolution_via_image_quality_predictors.md)
- [Progressive Focused Transformer for Single Image Super-Resolution](progressive_focused_transformer_for_single_image_super-resolution.md)
- [AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](adversarial_diffusion_compression_for_real-world_image_super-resolution.md)
- [QMambaBSR: Burst Image Super-Resolution with Query State Space Model](qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)
- [FIPER: Factorized Features for Robust Image Super-Resolution and Compression](../../NeurIPS2025/image_restoration/fiper_factorized_features_for_robust_image_super-resolution_and_compression.md)

<!-- RELATED:END -->
