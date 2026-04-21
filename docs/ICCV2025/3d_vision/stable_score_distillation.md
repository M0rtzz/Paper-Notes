---
title: >-
  [论文解读] Stable Score Distillation
description: >-
  [ICCV 2025][3D视觉][Score Distillation] 提出 Stable Score Distillation (SSD)，通过单分类器跨提示词引导和 null-text 分支的跨轨迹正则化，实现更稳定精准的文本引导 2D/3D 编辑，在保持源内容结构的同时提升编辑对齐度。
tags:
  - ICCV 2025
  - 3D视觉
  - Score Distillation
  - 3D场景编辑
  - 2D图像编辑
  - 扩散模型
  - Classifier-Free Guidance
  - NeRF
  - 3DGS
---

# Stable Score Distillation

**会议**: ICCV 2025  
**arXiv**: [2507.09168](https://arxiv.org/abs/2507.09168)  
**代码**: [https://github.com/Alex-Zhu1/SSD](https://github.com/Alex-Zhu1/SSD)  
**领域**: 3D视觉 / 文本引导编辑  
**关键词**: Score Distillation, 3D场景编辑, 2D图像编辑, 扩散模型, Classifier-Free Guidance, NeRF, 3DGS

## 一句话总结

提出 Stable Score Distillation (SSD)，通过单分类器跨提示词引导和 null-text 分支的跨轨迹正则化，实现更稳定精准的文本引导 2D/3D 编辑，在保持源内容结构的同时提升编辑对齐度。

## 研究背景与动机

文本引导的图像/3D编辑依赖扩散模型的先验知识，但现有 score distillation 方法存在明显缺陷：

**SDS 的局限**：Score Distillation Sampling 在编辑任务中会引入全局优化干扰，导致非编辑区域模糊和伪影，因为它是面向整个 prompt 全局优化的。

**DDS 的不足**：Delta Denoising Score 通过引入源分支消除模型偏差，但缺乏对源内容结构的显式保护，导致非编辑区域也被修改（如编辑人物面部时衣服也被改变）。

**CSD 的问题**：Classifier Score Distillation 使用双分类器获取跨提示词编辑方向，但同样缺少源结构保持机制，导致结构变形和伪影。

**编辑强度不足**：DDS 类方法在风格编辑等场景中容易出现编辑力度不够，最终几乎未产生变化。

作者的两个核心观察：

- **跨提示词（Cross-prompt）**：只需单个分类器即可提供从源提示词到目标提示词的编辑方向，无需双分类器的复杂结构。
- **跨轨迹（Cross-trajectory）**：通过将编辑方向与源内容结构对齐，可确保优化过程稳定，避免结构突变。

## 方法详解

### 整体框架

SSD 的设计围绕三个核心组件展开，最终损失为三项之和：

$$L_{\text{final}} = L_{\text{ssd}} + L_{\text{align}} + L_{\text{ID}}$$

其中 $L_{\text{ssd}}$ 是核心蒸馏损失，$L_{\text{align}}$ 是提示词增强项，$L_{\text{ID}}$ 是源 latent 正则化项。

### 关键设计一：Stable Score Distillation 核心公式

不同于 DDS 使用辅助源分支，SSD 利用 CFG 公式构建跨提示词编辑方向，并引入 null-text 分支进行正则化：

$$L_{\text{ssd}} = \epsilon_\phi(z_t, \hat{y}) + s(\epsilon_\phi(z_t, y) - \epsilon_\phi(z_t, \hat{y})) - \epsilon_\phi(\hat{z}_t, \varnothing)$$

该公式可分解为两项：

$$L_{\text{ssd}} = \underbrace{w_p(\epsilon_\phi(z_t, y) - \epsilon_\phi(z_t, \hat{y}))}_{\text{cross-prompt}} + \underbrace{w_t(\epsilon_\phi(z_t, \hat{y}) - \epsilon_\phi(\hat{z}_t, \varnothing))}_{\text{cross-trajectory}}$$

- **Cross-prompt 项**：用单一分类器衡量当前 latent 在目标提示词 $y$ 与源提示词 $\hat{y}$ 下的预测差异，提供平滑的纹理过渡方向。
- **Cross-trajectory 项**：衡量当前 latent 在源提示词下的预测与源 latent 在 null-text 下的预测之间的距离，约束结构不发生突变，是 SSD 与 CSD 的关键区别。当 $w_t=0$ 时退化为 CSD，结构无法保持。

### 关键设计二：提示词增强分支

DDS 系列方法在风格编辑时编辑力度不足，SSD 添加目标提示词增强项：

$$L_{\text{align}} = w_e(\epsilon_\phi(z_t, y) - \epsilon_\phi(z_t, \varnothing))$$

该项为标准的 CFG 分类器方向，直接增强目标提示词的引导力度。$w_e$ 控制增强强度，过大会导致过饱和，需要与 cross-trajectory 同步调节。

### 关键设计三：源 Latent 正则化

在 3DGS 编辑中，latent 空间损失可能导致局部梯度爆炸（出现亮斑），因此引入 ID 正则化：

$$L_{\text{ID}} = w(t) \cdot (x_t - \hat{x}_t)$$

其中 $w(t)$ 为随迭代递减的权重。与 PDS 使用无噪 $\hat{x}_0$ 不同，SSD 使用带噪 $\hat{x}_t$ 避免梯度爆炸。

### 与 InstructPix2Pix 的联系

作者发现 SSD 的设计与 IP2P 的单步反向采样公式有结构性对应：IP2P 公式中的中间项对应 cross-trajectory 正则化，最后一项对应 cross-prompt 项。这意味着在 IP2P 模型上应用 DDS 损失只需编辑分支，无需源分支。

## 实验关键数据

### 主实验一：3D 场景编辑

| 方法 | CLIP Sim ↑ | Sim Dire ↑ | User Study ↑ |
|------|-----------|-----------|-------------|
| IN2N | 0.1676 | 0.0707 | 14.54% |
| DDS | 0.1780 | 0.0401 | 5.45% |
| GS-Editor | 0.1758 | 0.0429 | 14.54% |
| DGE | 0.1758 | 0.0563 | 23.63% |
| **SSD (Ours)** | **0.1846** | **0.0773** | **41.81%** |

- 在 IN2N、LLFF、Mip-NeRF360 等数据集上测试，6 个场景 10 个提示词。
- 用户研究 55 人参与，SSD 获得 41.81% 最高支持率，远超其他方法。
- CLIP Sim 和 Sim Dire 均取得最优。

### 主实验二：2D 图像编辑（PIE-Bench，700 张图，9 种编辑类型）

| 方法 | Distance↓ | LPIPS↓ | MSE↓ | CLIP↑ |
|------|----------|--------|------|-------|
| DDIM + P2P | 69.43 | 208.80 | 219.88 | 25.01 |
| DDS | 14.74 | 50.58 | 45.09 | 25.86 |
| DDS + CDS | 7.15 | 33.14 | 25.29 | 24.96 |
| **Ours** | 28.13 | 82.43 | 86.64 | **26.94** |
| **Ours + CDS** | **6.90** | **32.15** | **24.21** | 25.12 |

- SSD 在 CLIP Similarity 上取得最优（26.94），证明提示词增强分支的有效性。
- 结合 CDS 后，在结构保持指标上全面最优（Distance 6.90, LPIPS 32.15, MSE 24.21）。
- 纯 SSD 模式以较高的结构变化换取显著更好的编辑效果，尤其在风格编辑上。

### 消融实验关键发现

| 组件 | 效果 |
|------|------|
| Cross-trajectory ($w_t=0$) | 退化为 CSD，结构无法保持，出现饱和和伪影 |
| Prompt-enhancement ($w_e$) | 对风格编辑至关重要，去掉后编辑力度显著不足 |
| ID 正则化 | 抑制 3DGS 中局部梯度爆炸（亮斑），但权重过大会限制编辑属性 |
| 收敛速度 | NeRF 约 3000 次迭代，3DGS 约 1500 次迭代（结合非递增 timestep 采样） |

## 亮点与洞察

1. **简洁的框架设计**：相比 DDS 的双分支和 CSD 的双分类器，SSD 只需单分类器 + null-text 分支，结构更简洁，却同时解决了稳定性和编辑力度两个问题。
2. **Cross-trajectory 正则化的巧妙性**：通过对比同一 latent 在源提示词下的预测与源 latent 在无条件下的预测，隐式约束结构变化，而非显式添加像素级重建损失。
3. **与 IP2P 的理论联系**：揭示了 SSD 与 InstructPix2Pix 之间的结构对应关系，为理解 IP2P 的工作机制提供了新视角。
4. **即插即用兼容性**：可直接集成到现有 DDS-based 编辑流水线（NeRF 编辑、2D 编辑），无需 LoRA 或微调，与 CDS 等方法互补组合效果更佳。
5. **用户研究压倒性优势**：3D 编辑的用户研究中获得 41.81% 投票，几乎是第二名 DGE（23.63%）的两倍。

## 局限性

1. **优化速度**：作为基于优化的方法，编辑过程需要数千次迭代，相比一步法或少步法（如 TurboEdit、SD-Turbo）仍然较慢。
2. **结构保持与编辑力度的权衡**：PIE-Bench 上纯 SSD 的结构距离指标（28.13）高于 DDS（14.74），表明强编辑必然伴随更大的结构变化。
3. **ID 正则化的权衡**：权重过大会抑制编辑属性（如角色胸前的蜘蛛标志被影响），需要手动调参。
4. **超参数较多**：$w_p$、$w_t$、$w_e$ 以及 ID 正则化权重 $w(t)$ 需要根据具体场景调节。

## 相关工作与启发

- **DDS (Delta Denoising Score)**：通过源分支消除偏差但不保结构，SSD 用 cross-trajectory 替代了源分支的角色。
- **CSD (Classifier Score Distillation)**：双分类器提供编辑方向但缺乏结构约束，SSD 的 $w_t=0$ 情况即退化为 CSD。
- **NFSD (Noise-Free Score Distillation)**：分解 CFG score 揭示分类器是编辑方向的核心驱动力，启发了 SSD 的单分类器设计。
- **PDS (Posterior Distillation Sampling)**：匹配随机 latent 进行后验蒸馏，SSD 简化了其 ID 保持策略（用带噪 latent 替代无噪 latent）。
- **DreamCatalyst**：基于递减 timestep 采样扩展 PDS，SSD 也采用非递增 timestep 采样加速收敛。
- **启发**：CFG 公式本身就是跨分布引导的天然工具，将其从"条件 vs 无条件"推广到"目标 vs 源"是一个优雅的泛化思路，值得在其他蒸馏场景中探索。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 单分类器 + null-text 分支的设计简洁高效，cross-trajectory 正则化是有意义的新贡献
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 3D（NeRF/3DGS）和 2D（PIE-Bench）编辑，有用户研究和消融实验，但缺少定量消融表格
- **写作质量**: ⭐⭐⭐ — 公式推导清晰但 LaTeX 符号使用偶有不一致，部分语法有小问题
- **价值**: ⭐⭐⭐⭐ — 对 score distillation 编辑领域有实际推进，框架兼容性好，即插即用

<!-- RELATED:START -->

## 相关论文

- [Advancing Text-to-3D Generation with Linearized Lookahead Variational Score Distillation](advancing_text-to-3d_generation_with_linearized_lookahead_variational_score_dist.md)
- [Stable-SCore: A Stable Registration-Based Framework for 3D Shape Correspondence](../../CVPR2025/3d_vision/stable-score_a_stable_registration-based_framework_for_3d_shape_correspondence.md)
- [Identity Preserving 3D Head Stylization with Multiview Score Distillation](identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection](g2sf_geometry-guided_score_fusion_for_multimodal_industrial_anomaly_detection.md)

<!-- RELATED:END -->
