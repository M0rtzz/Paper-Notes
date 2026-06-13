---
title: >-
  [论文解读] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models
description: >-
  [CVPR 2026][图像生成][Concept Erasure] 提出 TINA（Text-free INversion Attack），通过在 null-text 条件下优化 DDIM 反演找到精确的初始噪声，绕过所有基于文本的概念擦除防御，证明当前擦除方法仅切断了文本-图像映射而未真正删除模型内部的视觉知识。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "Concept Erasure"
  - "Machine Unlearning"
  - "DDIM Inversion"
  - "扩散模型"
  - "Adversarial Attack"
---

# TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models

## 基本信息

**会议**: CVPR 2026  
**arXiv**: [2603.17828](https://arxiv.org/abs/2603.17828)  
**代码**: [GitHub](https://github.com/qianlong0502/TINA)  
**领域**: 图像生成 / AI安全 / 概念擦除攻击  
**关键词**: Concept Erasure, Machine Unlearning, DDIM Inversion, Text-to-Image Diffusion, Adversarial Attack  

## 一句话总结

提出 TINA（Text-free INversion Attack），通过在 null-text 条件下优化 DDIM 反演找到精确的初始噪声，绕过所有基于文本的概念擦除防御，证明当前擦除方法仅切断了文本-图像映射而未真正删除模型内部的视觉知识。

## 研究背景与动机

当前文本到图像扩散模型（如 Stable Diffusion）的概念擦除（Concept Erasure）领域存在一个根本性盲区：所有擦除方法和对抗攻击都围绕**文本条件通路**展开。擦除方法（ESD、UCE、AdvUnlearn 等）通过切断文本提示与目标概念的映射来实现"遗忘"；攻击方法（P4D、UDA、CCE 等）则试图找到替代文本/嵌入来重新激活概念。

这种"text-centric co-evolution"带来了一个致命假设：**切断文本-图像链接 = 删除视觉知识**。作者认为这是错误的——即使文本路径被封堵，模型参数空间中与被擦除概念对应的**视觉知识仍然存在**。为验证这一假说，需要一种完全绕过文本条件的攻击方式。

核心假说：即使文本-图像映射被移除，被擦除概念的**确定性生成路径**仍然存在于模型中，可以在完全无文本条件下被重新发现。

## 方法详解

### 整体框架

TINA 想证明一件事：当前的概念擦除只是切断了文本到图像的映射，并没真把视觉知识从模型里删掉。为绕开所有“基于文本”的防御，它设计了一条完全不碰文本条件的攻击：第一阶段 **无文本反演**，给定被擦除模型 $\epsilon_\theta$ 和一张代表被擦概念的目标图像 $x$，在 null-text 条件 $c_\text{null}$ 下优化出能确定性生成该图的初始噪声 $z_T^*$；第二阶段 **确定性概念再生**，把 $z_T^*$ 喂回同一个被擦除模型、仍在 $c_\text{null}$ 下跑标准 DDIM 采样，就能把被擦概念重新生成出来。整个过程不含任何文本条件，因此所有文本路径上的防御都被跳过。基础模型为 SD v1.4，$T=50$ 步、CFG=7.5。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入：被擦除模型 ε_θ + 目标概念图像 x<br/>（全程 null-text 条件，不碰文本）"] --> B["两种朴素反演为何失败<br/>（设计动机）"]
    B -->|"文本引导反演"| B1["被擦模型主动抵抗文本路径 → 失败"]
    B -->|"null-text 标准反演"| B2["近似公式逐步累积误差 → z_T 偏离真值"]
    B1 --> C["定点优化反演<br/>自一致性损失约束，K=25 步梯度精炼"]
    B2 --> C
    C --> D["精确初始噪声 z_T*"]
    D --> E["确定性概念再生<br/>null-text 下跑标准 DDIM 采样"]
    E --> F["输出：被擦概念被重新生成<br/>（绕过所有基于文本的防御）"]
```

### 关键设计

**1. 标准无文本反演为什么失败：误差在 null-text 下快速累积**

DDIM 采样是确定性的——给定模型、条件 $c$ 和初始噪声 $z_T$，输出 $z_0$ 唯一确定，更新式为
$$z_{t-1} = \sqrt{\alpha_{t-1}} \hat{z}_0(z_t) + \sqrt{1-\alpha_{t-1}} \cdot \epsilon_\theta(z_t, t, c)$$
其中 $\hat{z}_0(z_t) = \frac{z_t - \sqrt{1-\alpha_t} \epsilon_\theta(z_t, t, c)}{\sqrt{\alpha_t}}$。理想的精确反演关系为 $z_t = C_1(t) z_{t-1} + C_2(t) \cdot \epsilon_\theta(z_t, t, c)$（$C_1(t) = \frac{\sqrt{\alpha_t}}{\sqrt{\alpha_{t-1}}}$，$C_2(t) = \sqrt{1-\alpha_t} - \sqrt{\frac{\alpha_t(1-\alpha_{t-1})}{\alpha_{t-1}}}$），但 $z_t$ 同时出现在等式两边，标准做法用 $\epsilon_\theta(z_{t-1}, t-1, c)$ 近似 $\epsilon_\theta(z_t, t, c)$ 就埋下累积误差。两种朴素方案都行不通：用被擦概念的提示词做**文本引导反演**会被模型主动抵抗、彻底失败（恰好说明文本防御在文本路径上确实有效）；而 **null-text 反演** 失去了文本引导，近似公式每步的微小误差迅速累积，$\hat{z}_T$ 偏离真实 $z_T^*$，还原不出概念。

**2. 把反演变成定点优化：用自一致性约束精确追踪轨迹**

TINA 不再用近似公式，而是直接把精确反演关系当成一个不动点约束：真实轨迹上的每个 $z_t$ 都必须满足
$$z_t = f_\theta^*(z_t, z_{t-1}, t, c) = C_1(t) z_{t-1} + C_2(t) \cdot \epsilon_\theta(z_t, t, c)$$
于是在每个时间步把求 $z_t$ 变成最小化自一致性损失
$$\mathcal{L}_t(z_t) = \| f_\theta^*(z_t, z_{t-1}, t, c_\text{null}) - z_t \|_2^2$$
具体做法是：先用标准 DDIM 反演在 $c_\text{null}$ 下算个初始估计 $\tilde{z}_t$，以它为起点做 $K$ 步梯度下降精炼 $z_t$，对 $t=1,\dots,T$ 依次推进，最终得到精确的 $z_T^*$。优化内循环取 $K=25$ 轮、AdamW、$\eta=0.001$；消融显示优化不足（TINA-Less）ASR 只有 46%，充分优化到自一致后升到 70%，可见精确追踪轨迹靠的就是这些迭代。

**3. 确定性概念再生：null-text 采样还原被擦概念**

拿到 $z_T^*$ 后，再生阶段什么花样都不用——把它输入同一个被擦除模型，在 $c_\text{null}$ 下跑一遍标准 DDIM 采样即可确定性地重建被擦概念。t-SNE 分析显示 $z_T^*$ 本身在噪声空间分不出概念，但它在 UNet mid_block 的激活会按概念清晰聚类，说明模型内部的概念特异性视觉知识被精确激活了——这正是“文本擦除 ≠ 视觉知识删除”的直接证据。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)
- [\[AAAI 2026\] Copyright Infringement Detection in Text-to-Image Diffusion Models via Differential Privacy](../../AAAI2026/image_generation/copyright_infringement_detection_in_text-to-image_diffusion_models_via_different.md)

</div>

<!-- RELATED:END -->
