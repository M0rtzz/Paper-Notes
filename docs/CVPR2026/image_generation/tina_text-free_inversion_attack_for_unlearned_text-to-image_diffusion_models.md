---
title: >-
  [论文解读] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models
description: >-
  [图像生成] 提出 TINA（Text-free INversion Attack），通过在 null-text 条件下优化 DDIM 反演找到精确的初始噪声，绕过所有基于文本的概念擦除防御，证明当前擦除方法仅切断了文本-图像映射而未真正删除模型内部的视觉知识。
tags:
  - 图像生成
---

# TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models

## 基本信息

| 项目 | 内容 |
|------|------|
| 会议 | CVPR 2026 |
| arXiv | [2603.17828](https://arxiv.org/abs/2603.17828) |
| 代码 | [GitHub](https://github.com/qianlong0502/TINA) |
| 领域 | 图像生成 / AI安全 / 概念擦除攻击 |
| 关键词 | Concept Erasure, Machine Unlearning, DDIM Inversion, Text-to-Image Diffusion, Adversarial Attack |

## 一句话总结

提出 TINA（Text-free INversion Attack），通过在 null-text 条件下优化 DDIM 反演找到精确的初始噪声，绕过所有基于文本的概念擦除防御，证明当前擦除方法仅切断了文本-图像映射而未真正删除模型内部的视觉知识。

## 研究背景与动机

当前文本到图像扩散模型（如 Stable Diffusion）的概念擦除（Concept Erasure）领域存在一个根本性盲区：所有擦除方法和对抗攻击都围绕**文本条件通路**展开。擦除方法（ESD、UCE、AdvUnlearn 等）通过切断文本提示与目标概念的映射来实现"遗忘"；攻击方法（P4D、UDA、CCE 等）则试图找到替代文本/嵌入来重新激活概念。

这种"text-centric co-evolution"带来了一个致命假设：**切断文本-图像链接 = 删除视觉知识**。作者认为这是错误的——即使文本路径被封堵，模型参数空间中与被擦除概念对应的**视觉知识仍然存在**。为验证这一假说，需要一种完全绕过文本条件的攻击方式。

核心假说：即使文本-图像映射被移除，被擦除概念的**确定性生成路径**仍然存在于模型中，可以在完全无文本条件下被重新发现。

## 方法详解

### 整体框架

TINA 的攻击流程分为两个阶段：

1. **Text-Free Inversion（无文本反演）**：给定被擦除模型 $\epsilon_\theta$ 和目标图像 $x$（代表被擦除概念），在 null-text 条件 $c_\text{null}$ 下，通过优化找到精确的初始噪声 $z_T^*$，使其能确定性地生成目标图像。

2. **Deterministic Concept Regeneration（确定性概念再生）**：将优化得到的 $z_T^*$ 输入同一被擦除模型 $\epsilon_\theta$，在 $c_\text{null}$ 条件下执行标准 DDIM 采样，确定性地再生被擦除概念。

关键洞察：整个过程**完全不涉及文本条件**，因此彻底绕过了所有基于文本的擦除防御。

### 前置知识：DDIM 采样与反演

**DDIM 采样**是确定性的——给定模型 $\theta$、条件 $c$ 和初始噪声 $z_T$，生成的 $z_0$ 是唯一确定的。采样更新公式为：

$$z_{t-1} = \sqrt{\alpha_{t-1}} \hat{z}_0(z_t) + \sqrt{1-\alpha_{t-1}} \cdot \epsilon_\theta(z_t, t, c)$$

其中 $\hat{z}_0(z_t) = \frac{z_t - \sqrt{1-\alpha_t} \epsilon_\theta(z_t, t, c)}{\sqrt{\alpha_t}}$ 是对干净潜变量的预测。

**标准 DDIM 反演**试图从 $z_0$ 反推 $z_T$。理想的精确反演关系为：

$$z_t = C_1(t) z_{t-1} + C_2(t) \cdot \epsilon_\theta(z_t, t, c)$$

其中 $C_1(t) = \frac{\sqrt{\alpha_t}}{\sqrt{\alpha_{t-1}}}$，$C_2(t) = \sqrt{1-\alpha_t} - \sqrt{\frac{\alpha_t(1-\alpha_{t-1})}{\alpha_{t-1}}}$。

但这里 $z_t$ 出现在等式两边，存在循环依赖。标准做法用近似 $\epsilon_\theta(z_{t-1}, t-1, c)$ 代替 $\epsilon_\theta(z_t, t, c)$，导致**累积误差**。

### 标准反演为何失败

作者分析了两种朴素方案的失败原因：

- **文本引导反演**（使用被擦除概念的提示词）：被擦除模型主动抵抗该文本条件，导致完全失败。这恰好说明文本擦除防御在文本路径上是有效的。

- **Null-text 反演**（使用空文本条件）：由于标准 DDIM 反演的近似公式严重依赖有意义的文本条件来引导噪声预测，在无文本引导时，每步的微小误差快速累积，最终 $\hat{z}_T$ 偏离真实 $z_T^*$，无法高保真还原目标概念。

### 核心创新：基于优化的无文本反演

TINA 将反演问题重新定义为**定点优化问题**。不使用近似公式，而是直接从 DDIM 采样方程推导精确的反演关系：

$$z_t = f_\theta^*(z_t, z_{t-1}, t, c) = C_1(t) z_{t-1} + C_2(t) \cdot \epsilon_\theta(z_t, t, c)$$

这建立了一个**自一致性约束**：真实生成轨迹上的每个 $z_t$ 都必须是映射 $f_\theta^*$ 的不动点。

在每个时间步 $t$，给定已计算的 $z_{t-1}$，TINA 通过最小化以下损失函数来优化 $z_t$：

$$\mathcal{L}_t(z_t) = \| f_\theta^*(z_t, z_{t-1}, t, c_\text{null}) - z_t \|_2^2$$

优化过程：
1. 用标准 DDIM 反演在 $c_\text{null}$ 下计算初始估计 $\tilde{z}_t$
2. 以 $\tilde{z}_t$ 为起点，执行 $K$ 步梯度下降精炼 $z_t$
3. 对 $t = 1, \dots, T$ 依次执行，最终得到精确的 $z_T^*$

### 算法伪代码

| 步骤 | 操作 |
|------|------|
| 输入 | 被擦除模型 $\epsilon_\theta$, 目标图像潜变量 $z_0$, 步数 $T$, 空文本 $c_\text{null}$, 优化轮数 $K$, 学习率 $\eta$ |
| for $t = 1$ to $T$ | 用标准反演在 $c_\text{null}$ 下从 $z_{t-1}$ 计算初始 $\tilde{z}_t$，设 $z_t \leftarrow \tilde{z}_t$ |
| for $k = 1$ to $K$ | 计算 $\mathcal{L}_t$, 梯度下降 $z_t \leftarrow z_t - \eta \nabla_{z_t} \mathcal{L}_t$ |
| 输出 | 优化后的初始噪声 $z_T^* \leftarrow z_T$ |

### 实现细节

- 基础模型：Stable Diffusion v1.4
- 调度器：Linear Multistep Scheduler，$T = 50$ 步，CFG = 7.5
- 内循环优化：$K = 25$ 轮，AdamW 优化器，$\eta = 0.001$
- 硬件：单张 NVIDIA A100 GPU

## 实验

### 主实验：裸体概念擦除攻击成功率

| 攻击方法 | ESD | FMN | UCE | MACE | RECE | AdvUnlearn | SalUn | STEREO |
|----------|-----|-----|-----|------|------|------------|-------|--------|
| MMA | 13.1 | 67.0 | 32.6 | 6.0 | 22.8 | 1.7 | 1.7 | 5.5 |
| P4D | 69.0 | 97.9 | 76.1 | 75.4 | 66.2 | 18.3 | 15.5 | 24.7 |
| UDA | 76.1 | 97.9 | 78.9 | 81.7 | 63.4 | 23.2 | 13.4 | 25.4 |
| RAB | 50.5 | 97.9 | 29.5 | 6.3 | 10.5 | 2.1 | 0.0 | 8.4 |
| CCE | 74.7 | 55.0 | 49.3 | 50.0 | 66.9 | 76.8 | 2.8 | 16.9 |
| **TINA** | **82.4** | **97.9** | **82.4** | **93.0** | **80.3** | **78.9** | **71.1** | **81.0** |

关键发现：TINA 在所有 8 种防御上均取得最高 ASR。特别是对 AdvUnlearn（78.9%）、SalUn（71.1%）、STEREO（81.0%）等鲁棒防御，文本攻击几乎失效（UDA 仅 23.2%/13.4%/25.4%），而 TINA 依然保持高攻击率。

### 风格擦除攻击成功率（Van Gogh）

| 攻击方法 | ESD | FMN | AC | MACE | SPM | RECE | AdvUnlearn | STEREO |
|----------|-----|-----|-----|------|-----|------|------------|--------|
| P4D | 30.0 | 54.0 | 68.0 | 42.0 | 78.0 | 62.0 | 0.0 | 0.0 |
| UDA | 32.0 | 56.0 | 77.0 | 56.0 | 88.0 | 64.0 | 2.0 | 0.0 |
| CCE | 8.0 | 18.0 | 14.0 | 26.0 | 36.0 | 40.0 | 44.0 | 4.0 |
| **TINA** | **70.0** | **72.0** | **74.0** | **72.0** | **80.0** | **74.0** | **70.0** | **44.0** |

### 物体擦除攻击成功率（Tench 类别）

| 攻击方法 | ESD | EraseDiff | SalUn | Scissorhands | STEREO |
|----------|-----|-----------|-------|--------------|--------|
| P4D | 32.0 | 8.0 | 18.0 | 6.0 | 0.0 |
| UDA | 46.0 | 2.0 | 12.0 | 6.0 | 2.0 |
| CCE | 40.0 | 34.0 | 58.0 | 0.0 | 2.0 |
| **TINA** | **70.0** | **68.0** | **72.0** | **78.0** | **72.0** |

### 消融实验

| 方法 | ASR (%) | 说明 |
|------|---------|------|
| Standard Inv.（文本引导标准反演） | 30 | 擦除方法主动对抗文本条件 |
| TINA-Less（优化步数不足） | 46 | 误差纠正不充分 |
| **TINA**（完整优化 $K=25$） | **70** | 充分优化达到自一致 |

TINA-Less → TINA 的 24% ASR 提升证明充分的优化迭代对精确追踪生成轨迹至关重要。

### 与 DDIM 重建方法对比（EasyInv）

| 方法 | ESD | EraseDiff | SalUn | Scissorhands | STEREO |
|------|-----|-----------|-------|--------------|--------|
| EasyInv | 24.0 | 26.0 | 30.0 | 34.0 | 24.0 |
| **TINA** | **70.0** | **68.0** | **72.0** | **78.0** | **72.0** |

通用 DDIM 重建方法在概念恢复任务上远不如 TINA 的专用优化方案。

### 关键发现

1. **文本擦除 ≠ 视觉知识删除**：TINA 在全部三类任务（裸体/风格/物体）上均高效绕过所有 12 种擦除防御，证明被擦除概念的视觉知识仍保留在模型参数中。
2. **鲁棒防御对 TINA 无效**：AdvUnlearn 和 STEREO 等对抗训练强化的防御能有效阻止文本攻击，但对 TINA 几乎不构成障碍。
3. **潜变量嵌入分析**（t-SNE）：优化后的噪声 $z_T^*$ 本身在噪声空间无法区分概念，但其在 UNet mid_block 的激活清晰地按概念聚类，证明模型内部的概念特异性视觉知识被精确激活。
4. **架构通用性**：TINA 在 DiT 架构（PixArt-XL-2）上同样有效，说明该漏洞不限于 UNet。

## 亮点

- **范式突破**：首次从视觉角度质疑概念擦除的有效性，揭示"text-centric"范式的根本性缺陷
- **方法精巧**：将 DDIM 反演中的近似误差问题转化为定点优化问题，无需额外模型或文本信息
- **实验全面**：覆盖 12 种擦除方法 × 5 种基线攻击 × 3 类概念任务，论证充分
- **安全警示**：为 AI 安全社区提供关键预警，推动向操作内部视觉表示的擦除范式转变

## 局限

- 需要目标概念的**参考图像**作为反演起点，不是完全零样本的攻击
- 对 STEREO 在风格擦除任务上 ASR 仅为 44%，说明对抗训练可部分扰动内部视觉表示
- 攻击计算开销较大（每个时间步需 $K=25$ 轮优化迭代，共 $T \times K = 1250$ 次前向传播）
- 仅在 SD v1.4 上全面评估，未覆盖 SDXL 等更大规模模型
- 论文主要诊断问题但未提出对应的防御方案

## 评分

⭐⭐⭐⭐ — 概念擦除领域的重要范式警醒工作。通过优雅的无文本反演攻击揭示了当前擦除方法的根本性不足，实验设计严谨全面。但攻击需要参考图像、未提供防御方案是遗憾。

<!-- RELATED:START -->

## 相关论文

- [When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)
- [TAUE: Training-free Noise Transplant and Cultivation Diffusion Model](taue_training-free_noise_transplant_and_cultivation_diffusion_model.md)
- [TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)

<!-- RELATED:END -->
