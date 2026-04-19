---
title: >-
  [论文解读] DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution
description: >-
  [NeurIPS 2025][LLM对齐][图像超分辨率] 提出 DP²O-SR 框架，利用扩散模型固有的随机性生成多样化超分辨率输出，通过混合感知奖励构建偏好对，并设计层次化偏好优化（HPO）策略自适应加权训练对，在无需人工标注的前提下显著提升真实世界图像超分辨率的感知质量。
tags:
  - NeurIPS 2025
  - LLM对齐
  - 图像超分辨率
  - 偏好优化
  - 扩散模型
  - perceptual quality
  - DPO
---

# DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution

**会议**: NeurIPS 2025  
**arXiv**: [2510.18851](https://arxiv.org/abs/2510.18851)  
**代码**: [github.com/cswry/DP2O-SR](https://github.com/cswry/DP2O-SR)  
**领域**: llm_alignment  
**关键词**: 图像超分辨率, 偏好优化, 扩散模型, perceptual quality, DPO

## 一句话总结

提出 DP²O-SR 框架，利用扩散模型固有的随机性生成多样化超分辨率输出，通过混合感知奖励构建偏好对，并设计层次化偏好优化（HPO）策略自适应加权训练对，在无需人工标注的前提下显著提升真实世界图像超分辨率的感知质量。

## 研究背景与动机

真实世界图像超分辨率（Real-ISR）旨在从低分辨率输入重建高分辨率图像。传统方法侧重像素级精度（PSNR/SSIM），但常产生过度平滑的结果。近年来基于预训练 Text-to-Image（T2I）扩散模型（如 Stable Diffusion、FLUX）的方法展现了合成丰富细节的强大能力，但存在一个核心问题：**扩散模型的固有随机性**——不同噪声输入会导致感知质量差异显著的输出。

现有方法通常将这种随机性视为缺陷，试图通过稳定化生成过程或训练单步模型来消除它。而本文的核心洞察截然不同：**将随机性视为高质量监督信号的来源**。不同噪声种子产生的输出形成了感知质量的多样化分布，可以被利用来构建偏好对进行优化。

此外，现有 DPO 方法（如 Diff-DPO）仅从不同模型产出中构建单一的"最优 vs 最差"偏好对，监督信号有限。而且，缺乏针对 Real-ISR 任务精心设计的感知奖励和偏好数据策展策略。

## 方法详解

### 整体框架

DP²O-SR 框架包含三个核心组件：

1. **多样化 ISR 样本生成**：冻结参考模型 $\pi_{\text{ref}}$，对同一 LR 输入使用不同噪声种子采样 $M$ 个 SR 候选输出
2. **感知奖励排序与偏好对构建**：用混合 IQA 奖励对候选进行排序，选取 top-$N$ 和 bottom-$N$ 构成 $N^2$ 个偏好对
3. **层次化偏好优化（HPO）**：自适应加权训练对，聚焦最有信息量的比较

### 关键设计

#### 1. 混合感知奖励设计

奖励信号结合了**全参考（FR）指标**和**无参考（NR）指标**：

- **FR 集合 $\mathcal{FR}$**：LPIPS、TOPIQ-FR、AFINE-FR — 促进结构保真度，抑制幻觉内容
- **NR 集合 $\mathcal{NR}$**：MANIQA、MUSIQ、CLIPIQA+、TOPIQ-NR、AFINE-NR、Q-Align — 鼓励真实感和美学一致性

对每个候选 $I_m$ 的每个指标 $\phi$，先计算原始分数 $s_m^\phi$，方向对齐后做 min-max 归一化：

$$\bar{s}_m^\phi = \frac{s_m^\phi - s_{\min}^\phi}{s_{\max}^\phi - s_{\min}^\phi}$$

最终奖励等权融合 FR 和 NR：

$$R_m = \frac{0.5}{|\mathcal{FR}|}\sum_{\phi \in \mathcal{FR}} \bar{s}_m^\phi + \frac{0.5}{|\mathcal{NR}|}\sum_{\phi \in \mathcal{NR}} \bar{s}_m^\phi$$

关键发现：仅用 FR 奖励会过度平滑；仅用 NR 奖励会产生幻觉细节；混合奖励既保持结构一致性，又提升真实感。

#### 2. 偏好对策展策略

与 Diff-DPO 仅构建单一"最优 vs 最差"对不同，本文从同一模型采样 $M$ 个输出，选取 top-$N$ 和 bottom-$N$ 构成 $N^2$ 个偏好对。引入两个关键控制参数：

- **采样数 $M$**：增大 $M$ 提高感知多样性和训练稳定性，但收益递减
- **选择比 $N/M$**：小 $N/M$ 产生更强奖励差异（高对比度），大 $N/M$ 增加覆盖和多样性

关键发现（架构敏感性）：
- **小模型 C-SD2（0.8B）**：最优 $N/M = 1/4$，需要更多冗余和平滑梯度
- **大模型 C-FLUX（12B）**：最优 $N/M = 1/16$，能有效从高对比度信号中学习

#### 3. 层次化偏好优化（HPO）

在两个层级自适应加权：

**组内权重**（聚焦大奖励差异的对）：
$$w_{\text{intra}}(\mathbf{x}_0^w, \mathbf{x}_0^l) = |R_w - R_l| + (1 - \mu_{\text{gap}})$$

其中 $\mu_{\text{gap}}$ 是组内所有对的平均奖励差。

**组间权重**（优先感知多样性高的 LR 输入组）：
$$w_{\text{inter}}(g) = \sigma_g + (1 - \mu_\sigma)$$

其中 $\sigma_g$ 是组内奖励标准差，$\mu_\sigma$ 是所有组的平均标准差。

### 损失函数 / 训练策略

最终损失基于 Diff-DPO，加入层次化权重：

$$\mathcal{L}_{HPO} = \sum_{(\mathbf{x}_0^w, \mathbf{x}_0^l)} w \cdot \ell(\mathbf{x}_0^w, \mathbf{x}_0^l; \theta)$$

其中 $w = w_{\text{intra}} \cdot w_{\text{inter}}$，$\ell(\cdot)$ 是 Diff-DPO 损失。

训练配置：batch size 1024，学习率 $2 \times 10^{-5}$，$\beta = 5000$，训练 1000 步，8×A800 GPU。C-SD2 选 $N=8, M=32$；C-FLUX 选 $N=4, M=64$。

## 实验关键数据

### 主实验

在 Syn-Test 和 RealSR 两个基准上评估，横跨 14 个 IQA 指标（4 类）。

**Syn-Test 上的关键提升**（DP²O-SR vs 基线）：

| 指标 | C-SD2 | DP²O-SR(SD2) | C-FLUX | DP²O-SR(FLUX) |
|------|-------|-------------|--------|--------------|
| MANIQA↑ | 0.6684 | **0.7165** | 0.6857 | **0.7199** |
| CLIPIQA+↑ | 0.7595 | **0.8124** | 0.7473 | **0.7993** |
| QALIGN↑ | 4.2481 | **4.5526** | 4.4266 | **4.7060** |
| VQ-R1↑ | 4.43 | **4.57** | 4.53 | **4.65** |

**RealSR 上**（域外泛化，包含标准差统计）：

| 指标 | C-SD2 | DP²O-SR(SD2) | C-FLUX | DP²O-SR(FLUX) |
|------|-------|-------------|--------|--------------|
| MANIQA↑ | 0.664±0.019 | **0.705±0.012** | 0.665±0.025 | **0.694±0.013** |
| MUSIQ↑ | 70.34±1.79 | **73.24±0.81** | 69.70±2.15 | **72.78±0.93** |
| QALIGN↑ | 3.630±0.187 | **4.017±0.117** | 3.654±0.231 | **4.143±0.113** |

DP²O-SR 不仅提升感知质量，还显著**降低了输出方差**（标准差大幅缩小），说明模型变得更稳定。

### 消融实验

**采样数 $M$ 的影响**：增大 $M$ 持续提升性能但收益递减。$M=64$ 时接近饱和。

**选择比 $N/M$ 的影响**：
- C-SD2 在 $N/M = 1/4$ 最优，过低会出现 reward collapse
- C-FLUX 在 $N/M = 1/16$ 最优，在 $N/M = 1/2$ 时性能下降

**HPO 的有效性**：层次化权重相比均匀权重的 Diff-DPO 显著提升训练效率和最终感知质量。

### 关键发现

1. **感知-失真权衡**：感知质量提升伴随 PSNR/SSIM 的轻微下降，符合经典理论
2. **泛化性强**：在未参与训练的 VQ-R1、NIMA 等指标上同样表现优异
3. **架构通用**：在扩散模型（SD2, UNet）和 flow 模型（FLUX, DiT）上均有效
4. **快速收敛**：500 步即可超越 SeeSR、OSEDiff 等强基线
5. **鲁棒性提升**：DP²O-SR 显著改善 Worst@M，最差输出也达到高质量

## 亮点与洞察

1. **将随机性转化为优势**：逆转传统观点，将扩散模型随机性视为偏好学习的信号源，非常巧妙
2. **混合奖励设计兼顾保真与感知**：FR 和 NR 指标的等权平衡避免了单一指标的偏向
3. **HPO 的两级自适应权重**：组内关注困难对，组间关注信息量大的输入，设计合理
4. **架构感知的超参选择**：揭示了模型容量与最优偏好对策展策略之间的关系规律
5. **无需人工标注**：完全自动化的偏好数据构建，可扩展性极强

## 局限与展望

1. **离线候选生成成本极高**：30K 图像 × 64 采样需 168-432 GPU 小时，IQA 标注还需额外 72 小时
2. **PSNR/SSIM 有所下降**：虽符合理论预期，但在某些保真度要求高的应用场景可能不适用
3. **奖励指标选择的鲁棒性**：FR 和 NR 的等权分配是否最优？不同应用场景可能需要调整
4. **仅验证了 ControlNet 架构**：能否推广到其他 Real-ISR 架构（如端到端训练的模型）待验证
5. **潜在的 reward hacking**：长期训练是否会导致模型过度优化特定 IQA 指标

## 相关工作与启发

- **Diff-DPO**：本文的直接基线，仅构建单一偏好对，本文通过多采样+排序做了有效扩展
- **RLHF → DPO**：借鉴 LLM 领域的偏好对齐思路到视觉生成，跨领域迁移
- **感知-失真权衡理论**：理论支撑了感知质量提升与像素精度下降的合理性
- **启发**：偏好对齐思想可扩展到其他视觉生成任务（视频超分、图像修复、图像生成），混合奖励设计思路值得借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将扩散随机性转化为偏好监督信号，HPO 设计有创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 14 个指标、两个架构、多个消融实验，非常充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，逻辑推进自然
- 价值: ⭐⭐⭐⭐ — 为视觉生成的偏好对齐提供了系统性框架

<!-- RELATED:START -->

## 相关论文

- [Rethinking Direct Preference Optimization in Diffusion Models](rethinking_direct_preference_optimization_in_diffusion_models.md)
- [PolyJuice Makes It Real: Black-Box, Universal Red Teaming for Synthetic Image Detectors](polyjuice_makes_it_real_black-box_universal_red_teaming_for_synthetic_image_dete.md)
- [Boost Your Human Image Generation Model via Direct Preference Optimization](../../CVPR2025/llm_alignment/boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/llm_alignment/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)

<!-- RELATED:END -->
