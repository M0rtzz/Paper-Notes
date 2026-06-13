---
title: >-
  [论文解读] TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting
description: >-
  [ECCV 2024][3D视觉][说话人头部合成] 提出TalkingGaussian，基于3D高斯溅射的形变驱动说话人头部合成框架，通过对持久性高斯基元施加平滑形变表示面部运动，并分解面部和口腔内部区域以解决运动不一致问题。 现有NeRF-based说话头方法（如RAD-NeRF、ER-NeRF）通过直接修改点的颜色和密…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "说话人头部合成"
  - "3D高斯溅射"
  - "形变场"
  - "面部-口腔分解"
  - "音频驱动"
---

# TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2404.15264](https://arxiv.org/abs/2404.15264)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 说话人头部合成, 3D高斯溅射, 形变场, 面部-口腔分解, 音频驱动

## 一句话总结

提出TalkingGaussian，基于3D高斯溅射的形变驱动说话人头部合成框架，通过对持久性高斯基元施加平滑形变表示面部运动，并分解面部和口腔内部区域以解决运动不一致问题。

## 研究背景与动机

现有NeRF-based说话头方法（如RAD-NeRF、ER-NeRF）通过直接修改点的颜色和密度来表示面部运动，但面部相邻区域可能呈现显著不同的颜色和结构变化，连续平滑的神经场难以拟合这种跳跃的外观变化，导致口部模糊、眼睑透明等严重畸变。形变是更平滑、连续的运动表示方式，但之前的形变方法缺乏精细的点控制能力。

## 方法详解

### 整体框架

TalkingGaussian包含：(1) Persistent Gaussian Field——保持不变外观和稳定几何的静态头部结构；(2) Grid-based Motion Field——基于tri-plane hash编码的运动场，预测条件驱动的逐点形变；(3) Face-Mouth Decomposition——将模型分为面部分支和口腔内部分支。

### 关键设计

**形变范式**: 每个高斯基元保持颜色$f$和不透明度$\alpha$不变，仅通过形变$\delta_i = \{\Delta\mu_i, \Delta s_i, \Delta q_i\}$改变位置、缩放和旋转，形变后参数$\theta_D = \{\mu+\Delta\mu, s+\Delta s, q+\Delta q, \alpha, f\}$。

**增量采样策略**: 解决形变学习中梯度消失的问题。使用滑动窗口按面部动作度量$m_j$渐进采样训练帧：$m_j \in [B_{lower}+k \times T, B_{upper}+k \times T]$，从嘴巴闭合逐步到张开，从眼睛睁开到闭合。

**面部-口腔分解**: 唇部和口腔内部空间接近但运动不一致，用单一运动场难以准确表示。使用语义掩码将两个区域分离，面部分支接受音频$\mathbf{a}$和上脸表情$\mathbf{e}$条件，口腔分支仅接受音频条件且只预测平移。最终通过前后遮挡关系融合：$\mathcal{C}_{head} = \mathcal{C}_{face} \times \mathcal{A}_{face} + \mathcal{C}_{mouth} \times (1-\mathcal{A}_{face})$。

### 损失函数

- **静态初始化**: $\mathcal{L}_C = \mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$
- **运动学习**: $\mathcal{L}_D = \mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}$（使用形变后参数渲染）
- **融合微调**: $\mathcal{L}_F = \mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM} + \gamma\mathcal{L}_{LPIPS}$（$\lambda=0.2, \gamma=0.5$）

## 实验关键数据

### 自重建设置

4个人物视频（Macron/Lieu/Obama/May）平均结果：

| 方法 | PSNR↑ | LPIPS↓ | SSIM↑ | LMD↓ | Sync-C↑ | 训练时间 | FPS |
|------|-------|--------|-------|------|---------|---------|-----|
| AD-NeRF | 31.87 | 0.0942 | 0.877 | 2.791 | 5.353 | 18.7h | 0.11 |
| RAD-NeRF | 33.07 | 0.0530 | 0.887 | 2.761 | 5.052 | 5.3h | 28.7 |
| ER-NeRF | 32.83 | 0.0289 | 0.889 | 2.676 | 5.295 | 2.1h | 31.2 |
| ER-NeRF+e | 33.14 | 0.0271 | 0.902 | 2.623 | 5.754 | - | - |
| **Ours** | **33.61** | **0.0259** | **0.910** | **2.586** | **6.516** | **0.5h** | **108** |

### 唇同步设置

跨域音频驱动的Sync-C/Sync-E评估（Audio A驱动Obama）：

| 方法 | Obama Sync-E↓ | Obama Sync-C↑ | May Sync-E↓ | May Sync-C↑ |
|------|---------------|---------------|-------------|-------------|
| AD-NeRF | 9.742 | 5.195 | 9.517 | 4.757 |
| ER-NeRF | - | - | - | - |
| **TalkingGaussian** | 最优 | 最优 | 最优 | 最优 |

### 关键发现

- 训练仅需0.5小时 vs ER-NeRF的2.1小时，渲染108 FPS vs 31.2 FPS，效率提升巨大
- LPIPS从0.0289降至0.0259，说明形变范式产生更清晰锐利的面部细节
- Sync-C从5.295提升至6.516（GT为7.584），唇音同步显著改善

## 亮点与洞察

1. **形变 vs 外观修改**的对比分析很有说服力——形变空间更平滑连续，避免了颜色跳变导致的畸变
2. Face-Mouth分解抓住了一个被忽略的核心矛盾：唇部外侧和口腔内部运动可能完全不同
3. 增量采样策略解决了形变学习中的梯度消失问题，思路值得借鉴

## 局限性

- 依赖预训练的面部解析模型获取口腔掩码
- 仅支持单人场景的person-specific训练
- 极端头部姿态变化的泛化能力有限

## 相关工作与启发

将3DGS应用于动态说话头的思路，后续被GaussianTalker等工作进一步发展。面部-口腔分解的思路可推广到全身动画的区域分解。

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实用性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](s3d-nerf_single-shot_speech-driven_neural_radiance_field_for_high_fidelity_talki.md)
- [\[ECCV 2024\] SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)
- [\[ECCV 2024\] HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting](headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)
- [\[CVPR 2026\] EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization](../../CVPR2026/3d_vision/emotag_emotion-aware_talking_head_synthesis_on_gaussian_splatting_with_few-shot_.md)
- [\[CVPR 2025\] MGGTalk: Monocular and Generalizable Gaussian Talking Head Animation](../../CVPR2025/3d_vision/monocular_and_generalizable_gaussian_talking_head_animation.md)

</div>

<!-- RELATED:END -->
