---
title: >-
  [论文解读] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation
description: >-
  [ICCV 2025][一致性蒸馏] 本文分析一致性模型蒸馏中高/低噪声水平的优化冲突，提出参数高效的双专家一致性模型（DCM），语义专家负责布局和运动、细节专家负责精细细节，配合时序一致性损失和GAN+特征匹配损失，在HunyuanVideo（13B）上实现4步采样接近50步基线质量。
tags:
  - ICCV 2025
  - 一致性蒸馏
  - 视频生成加速
  - 双专家模型
  - 时序一致性损失
  - GAN
---

# Dual-Expert Consistency Model for Efficient and High-Quality Video Generation

**会议**: ICCV 2025  
**arXiv**: [2506.03123](https://arxiv.org/abs/2506.03123)  
**代码**: [GitHub](https://github.com/Vchitect/DCM)  
**领域**: image_generation  
**关键词**: 一致性蒸馏, 视频生成加速, 双专家模型, 时序一致性损失, GAN蒸馏

## 一句话总结
本文分析一致性模型蒸馏中高/低噪声水平的优化冲突，提出参数高效的双专家一致性模型（DCM），语义专家负责布局和运动、细节专家负责精细细节，配合时序一致性损失和GAN+特征匹配损失，在HunyuanVideo（13B）上实现4步采样接近50步基线质量。

## 研究背景与动机

1. **领域现状**: 一致性蒸馏（Consistency Distillation）是主流的扩散模型加速方法，训练学生模型将ODE轨迹上任意点映射到相同终点。LCM、PCM等方法已有广泛应用。
2. **现有痛点**: 直接在视频扩散模型上应用一致性蒸馏会导致严重的时序一致性退化和外观细节丢失。高噪声样本主要学习语义布局/运动，低噪声样本主要精化细节，但两者的梯度和损失贡献差异显著，联合优化导致次优解。
3. **核心矛盾**: 单一学生模型容量有限，同时学习语义布局合成和精细细节生成引入优化干扰。可视化显示蒸馏过程中高/低噪声样本的损失和梯度范数存在显著差异。
4. **本文目标**: 如何解耦语义学习和细节学习两个阶段的优化，同时保持参数效率？
5. **切入角度**: 训练两个专家去噪器分别负责ODE轨迹的两段子轨迹，验证组合优于单一模型后，通过参数差异分析设计参数共享方案。
6. **核心 idea**: 将ODE轨迹分为语义和细节两段，用语义专家+LoRA细节专家的参数高效方案实现解耦蒸馏。

## 方法详解

### 整体框架
将教师模型的50步ODE轨迹以$t_\kappa$（$\kappa=37$）为界分为两段：语义合成阶段$\{x_{t_i}\}_{i=\kappa}^N$和细节精化阶段$\{x_{t_j}\}_{j=0}^\kappa$。分两阶段训练：先训练语义专家SemE在高噪声子轨迹上，再冻结SemE并加LoRA训练细节专家DetE在低噪声子轨迹上。推理时根据采样阶段动态切换专家。

### 关键设计

1. **参数高效双专家蒸馏**:
    - 功能: 以最少额外参数解耦语义和细节学习
    - 核心思路: 分析两个独立训练的专家参数差异，发现主要差异在 (1) 包含时间步的embedding层$\Psi$ 和 (2) 注意力层中的线性层$\Lambda$。因此阶段1训练SemE（全参数，在$[t_\kappa, t_N]$）；阶段2以SemE初始化，冻结主体，仅添加新的时间步依赖embedding层$\Psi$和注意力块LoRA $\Lambda^\dagger$，在$[t_0, t_\kappa]$上训练。
    - 设计动机: 直接训练两个完整模型增加一倍参数和显存，参数差异分析显示大部分权重共享即可。LoRA ($\Lambda^\dagger$) 仅微调注意力线性层足以捕获细节阶段的差异。

2. **时序一致性损失（Temporal Coherence Loss）**:
    - 功能: 增强语义专家SemE生成视频的运动一致性
    - 核心思路: 让SemE关注不同帧对应位置的变化和运动。定义帧间差分一致性: $\mathcal{L}_{TC} = \|(x_{l:L}^{t_\kappa} - x_{0:L-l}^{t_\kappa}) - (\hat{x}_{l:L}^{t_\kappa} - \hat{x}_{0:L-l}^{t_\kappa})\|_2^2$，其中$x_{l:L}^{t_\kappa}$表示从第$l$帧到第$L$帧的视频潜变量。
    - 设计动机: 语义阶段主要建立视频的运动和空间布局，时序一致性损失显式鼓励SemE保持帧间一致的运动和空间关系。

3. **GAN + 特征匹配损失**:
    - 功能: 增强细节专家DetE的细粒度合成质量
    - 核心思路: 使用冻结教师模型作为特征提取骨干$\Omega$，对学生和EMA学生的输出分别前向扩散得到fake/real样本，提取中间特征计算GAN损失和特征匹配损失: $\mathcal{L}_{FM} = \|\Omega(x_{fake}) - \Omega(x_{real})\|_2^2$。判别头$f_D$和DetE交替优化。
    - 设计动机: 分布匹配蒸馏中GAN损失对高质量细节合成效果已被验证。特征匹配损失稳定GAN训练，增强中间特征监督。

### 损失函数 / 训练策略
- SemE: 一致性损失$\mathcal{L}_{SemE}$ + 时序一致性损失$\mathcal{L}_{TC}$
- DetE: 一致性损失$\mathcal{L}_{DetE}$ + GAN损失$\mathcal{L}_G$ + 特征匹配损失$\mathcal{L}_{FM}$ + 判别器损失$\mathcal{L}_D$
- 在24张A100上训练，HunyuanVideo各专家1000 iteration，学习率1e-6 / 5e-6

## 实验关键数据

### 主实验

| 方法 | 步数 | 延迟(秒) | VBench Total | Quality | Semantic |
|--------|------|------|----------|------|------|
| HunyuanVideo (Teacher) | 50 | 1504.5 | 83.87 | 85.00 | 79.34 |
| LCM | 4 | 120.68 | 80.33 | 80.83 | 78.32 |
| PCM | 4 | 120.89 | 80.93 | 81.94 | 76.90 |
| **DCM (本文)** | 4 | 121.52 | **83.83** | **85.12** | **78.67** |
| **DCM (本文)** | 8 | 244.72 | **83.86** | 85.00 | **79.32** |

用户偏好研究: DCM vs LCM: 82.67%偏好DCM; DCM vs PCM: 77.33%偏好DCM

### 消融实验

| 配置 | VBench Total | Quality | Semantic |
|------|---------|------|------|
| VCM (基线一致性模型) | 80.30 | 80.74 | 78.36 |
| + OD (轨迹解耦) | 83.08 | 84.20 | 78.59 |
| + OD + PE (参数高效) | 83.03 | 84.16 | 78.53 |
| + OD + PE + TC (时序一致性) | 83.42 | 84.63 | — |
| Full DCM | **83.83** | **85.12** | **78.67** |

### 关键发现
- 轨迹解耦（OD）是最大的提升来源，证实了高/低噪声优化冲突的核心假设
- 参数高效方案（PE）只损失0.05 VBench分数，但大量减少参数
- SemE+DetE组合显著优于VCM在语义和细节两方面的表现
- 4步DCM几乎恢复50步教师的质量（83.83 vs 83.87），12.4×加速
- CogVideoX上也一致有效: 4步DCM (79.99) vs CogVideoX 50步 (80.59)

## 亮点与洞察
- 对蒸馏训练动力学的深入分析：可视化高/低噪声样本的损失和梯度差异，为双专家设计提供了坚实动机
- 参数差异分析得出的精准结论：差异主要在embedding层和注意力线性层，启发了LoRA方案
- 首次在HunyuanVideo（13B参数）级别成功实现一致性蒸馏
- 专家特定的优化目标设计（TC Loss for SemE, GAN for DetE）体现了对不同阶段学习需求的深刻理解

## 局限与展望
- 分界点$t_\kappa$的选择（$\kappa=37$）是启发式的，可探索自适应分界
- 双专家推理时需要两套embedding层和LoRA，虽然参数少但增加了推理复杂度
- 仅在4/8步上评估，未探索更极端的1-2步蒸馏
- GAN损失稳定性在更大模型上的可扩展性有待验证

## 相关工作与启发
- **vs LCM**: 单一模型一致性蒸馏无法处理高/低噪声优化冲突
- **vs PCM**: PCM将轨迹分段但仍用单一模型学习所有段
- **vs Hyper-SD**: 集成了轨迹分段一致性蒸馏和DMD，但未针对视频场景解耦
- **vs Seaweed-APT**: 对真实数据做一步对抗后微调，但仅2秒视频

## 评分
- 新颖性: ⭐⭐⭐⭐ 从训练动力学分析出发设计双专家方案，参数高效设计有洞察力
- 实验充分度: ⭐⭐⭐⭐⭐ HunyuanVideo+CogVideoX两个骨干，VBench+用户研究+全面消融
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，从观察到假设到验证到设计层层递进
- 价值: ⭐⭐⭐⭐⭐ 首次实现130亿参数视频模型的高质量4步蒸馏，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [OSV: One Step is Enough for High-Quality Image to Video Generation](../../CVPR2025/video_generation/osv_one_step_is_enough_for_high-quality_image_to_video_generation.md)
- [DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation](dh-facevid-1k_a_large-scale_high-quality_dataset_for_face_video_generation.md)
- [Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation](../../NeurIPS2025/video_generation/foresight_adaptive_layer_reuse_for_accelerated_and_highquali.md)
- [MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance](../../ICML2025/video_generation/mimicmotion_high-quality_human_motion_video_generation_with_confidence-aware_pos.md)
- [Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing](../../ICML2025/video_generation/ca2-vdm_efficient_autoregressive_video_diffusion_model_with_causal_generation_an.md)

<!-- RELATED:END -->
