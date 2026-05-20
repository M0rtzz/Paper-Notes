---
title: >-
  [论文解读] Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation
description: >-
  [NeurIPS 2025][3D视觉][文本到3D生成] 从理论上证明SDS是Schrödinger Bridge的特例，并基于此提出TraCe框架——在当前渲染和文本条件目标之间构建显式扩散桥，通过LoRA微调学习桥轨迹的score dynamics，在低CFG值下实现高质量text-to-3D生成。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "文本到3D生成"
  - "Score Distillation Sampling"
  - "Schrödinger Bridge"
  - "扩散模型"
  - "3D高斯溅射"
---

# Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation

**会议**: NeurIPS 2025  
**arXiv**: [2511.05609](https://arxiv.org/abs/2511.05609)  
**代码**: [GitHub](https://github.com/emmaleee789/TraCe.git)  
**领域**: 3D视觉  
**关键词**: 文本到3D生成, Score Distillation Sampling, Schrödinger Bridge, 扩散模型, 3D高斯溅射

## 一句话总结

从理论上证明SDS是Schrödinger Bridge的特例，并基于此提出TraCe框架——在当前渲染和文本条件目标之间构建显式扩散桥，通过LoRA微调学习桥轨迹的score dynamics，在低CFG值下实现高质量text-to-3D生成。

## 研究背景与动机

Text-to-3D生成的主流范式是利用预训练T2I扩散模型通过Score Distillation Sampling（SDS）优化3D表示。然而SDS存在两个核心问题：

**需要高CFG值**（通常100）才能获得强文本对齐——但高CFG导致**过饱和**和**过平滑**等视觉伪影

**SDS的梯度信号本质上是噪声的**——来自扩散模型的score估计并不保证是3D优化的最优方向

VSD等改进方法尝试在低CFG（如7.5）下工作，但在3DGS等表示上效果不佳。核心矛盾：现有方法都在**匹配扩散模型预测的梯度方向**，但这些score来自为2D图像生成训练的模型，与3D生成任务存在domain gap。

本文的关键洞察是：SDS可以从Schrödinger Bridge的角度理解——SDS实际上使用的是一个特殊的（一端为高斯噪声的）Schrödinger Bridge的反向过程。基于这个理论联系，作者提出构建一个**从当前渲染到目标图像的直接传输轨迹**（而非从噪声出发），使优化路径更稳定、更直接。

## 方法详解

### 整体框架

TraCe的每步优化：渲染当前3D模型 → 用预训练扩散模型估计目标图像$x_0^{pred}$ → 在渲染和目标之间构建Schrödinger Bridge → 采样中间状态$x_t$ → LoRA模型预测噪声 → 计算梯度更新3D参数$\theta$。

### 关键设计

1. **SDS作为Schrödinger Bridge的特例（理论贡献）**

   Schrödinger Bridge求解两个任意分布$P_A$和$P_B$之间的最可能随机演化。当$P_B \approx \mathcal{N}(0,I)$（高斯噪声）且前向Schrödinger因子$\Psi(x,t) \approx 1$时，由Nelson对偶性$\Psi \cdot \hat{\Psi} = q$得到$\hat{\Psi}(X_t,t) \approx p(X_t,t)$——反向SDE的score函数$-\nabla_{X_t}\log\hat{\Psi}$退化为标准扩散模型学到的score $s_\psi(X_t,t)$。因此SDS正是利用了这个特殊Schrödinger Bridge的反向score。

2. **构建显式扩散桥**

   关键步骤：
    - **目标端点**$X_0 \leftarrow x_0^{pred}$：对当前渲染用预训练模型**一步去噪**估计理想目标图像
    - **源端点**$X_1 \leftarrow x_{rndr}$：当前3D模型的渲染图
    - **中间采样**：基于可解析的桥后验$q(x_t|x_0^{pred}, x_{rndr}) = \mathcal{N}(x_t; \mu_t, \Sigma_t I)$采样：
    $\mu_t = \gamma_t x_0^{pred} + (1-\gamma_t) x_{rndr}, \quad \gamma_t = \frac{\bar{\sigma}_t^2}{\sigma_t^2 + \bar{\sigma}_t^2}$

3. **LoRA自适应学习桥轨迹**

   用LoRA微调T2I扩散模型$\epsilon_\phi$学习沿桥轨迹的score dynamics。最终梯度：
    $\nabla_\theta \mathcal{L}_{TraCe}(\theta) = \mathbb{E}_{\epsilon,t,c}\left[w(t)\left(\epsilon_\phi(x_t,t,y,c) - \frac{x_t - x_{rndr}}{\sigma_t}\right)\frac{\partial x_{rndr}}{\partial\theta}\right]$

   与SDS对比：SDS中噪声目标是$\epsilon_{noise}$（随机高斯噪声），TraCe中则是$\frac{x_t - x_{rndr}}{\sigma_t}$（桥轨迹的精确噪声），更稳定准确。

4. **渐进$t$采样策略**

   训练过程中$t$从0.5逐渐退火到0.02，逐步将Schrödinger Bridge插值的焦点从宽泛状态移向接近目标$x_0^{pred}$的状态，辅助渐进细化渲染结果。

### 损失函数 / 训练策略

- 使用Stable Diffusion作为T2I backbone
- 3D表示采用3D高斯溅射
- CFG值设为20（远低于SDS的100）
- LoRA参数在每步优化中持续更新
- 平均训练时间14分钟，VRAM 18741MiB

## 实验关键数据

### 主实验：83个Dreamfusion提示的定量比较

| 方法 | CLIP-L/14↑ | GPTEval3D↑ | ImageReward↑ | 时间 | VRAM |
|------|-----------|------------|-------------|------|------|
| SDS | 68.61 | 1018.09 | -0.43 | 10min | 18147M |
| VSD | 67.27 | 1007.49 | -0.53 | 17min | 26473M |
| CSD | 68.03 | 983.04 | -0.67 | 11min | 19804M |
| ISM | 69.01 | 1012.37 | -0.39 | 20min | 10151M |
| SDI | 63.04 | 971.98 | -0.83 | 10min | 16011M |
| **TraCe** | **69.26** | **1028.03** | **-0.29** | 14min | 18741M |

### 消融实验：LoRA + 渐进t采样

| 配置 | ImageReward↑ |
|------|-------------|
| 两者都关 | -0.4488 |
| 仅渐进t采样 | -0.3389 |
| 仅LoRA | -0.4020 |
| **两者都开（完整）** | **-0.2486** |

### 关键发现

- TraCe在**所有**ViT backbone的CLIP Score上都取得最高分
- CFG=15-20时TraCe已达到高质量，无需高CFG——解决了SDS的核心痛点
- LoRA和渐进t采样有**协同效应**——两者同时使用的提升远大于各自单独贡献
- VSD和CSD在3DGS上效果不佳（Figure 4可视化），而TraCe表现一致

## 亮点与洞察

1. **理论贡献扎实**：SDS ↔ Schrödinger Bridge的联系提供了新的优化视角，不仅是解释性的，而且指导了TraCe的设计
2. 中间状态$x_t$是在当前渲染和目标之间的**有意义的插值**（而非纯噪声），梯度信号更稳定
3. 框架的通用性——任何distillation-based的text-to-3D方法都可能从TraCe的桥轨迹思想中获益

## 局限与展望

- $x_0^{pred}$的质量依赖预训练模型的一步去噪能力——对复杂/罕见提示可能估计不准
- 仍然依赖SDS式的逐视图优化范式，缺乏3D先验
- 14分钟的训练时间虽合理，但对实时应用仍嫌慢
- 论文未讨论多视图一致性问题（如Janus问题是否缓解）

## 相关工作与启发

- Schrödinger Bridge在生成模型中的应用范围可进一步扩大，如图像编辑、视频生成
- LoRA在桥轨迹上的微调思路可推广到其他score distillation场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ SDS↔Schrödinger Bridge的理论联系是首创，基于此的方法设计自然优雅
- 实验充分度: ⭐⭐⭐⭐ 83个提示+6种方法对比+消融+CFG分析，较全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰，方法动机链条完整
- 价值: ⭐⭐⭐⭐ 提供了text-to-3D优化的新理论视角和实用改进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](../../ICCV2025/3d_vision/segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[ECCV 2024\] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](../../ECCV2024/3d_vision/dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)
- [\[ECCV 2024\] DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting](../../ECCV2024/3d_vision/dreamscene360_unconstrained_text-to-3d_scene_generation_with_panoramic_gaussian_.md)
- [\[CVPR 2025\] On Denoising Walking Videos for Gait Recognition](../../CVPR2025/3d_vision/on_denoising_walking_videos_for_gait_recognition.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
