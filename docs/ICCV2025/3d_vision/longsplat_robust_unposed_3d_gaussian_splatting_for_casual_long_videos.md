---
title: >-
  [论文解读] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos
description: >-
  [3D视觉] LongSplat 针对无相机位姿的随拍长视频场景，提出增量联合优化框架同时优化相机位姿和 3DGS，设计基于 MASt3R 先验的鲁棒位姿估计模块和自适应八叉树锚点形成机制，解决位姿漂移、几何初始化不准和内存限制问题。
tags:
  - 3D视觉
---

# LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2508.14041](https://arxiv.org/abs/2508.14041)
- **代码**: [项目页面](https://linjohnss.github.io/longsplat/)
- **领域**: 3D视觉
- **关键词**: 3D Gaussian Splatting, 无位姿长视频重建, 八叉树锚点, 增量联合优化, 位姿估计

## 一句话总结
LongSplat 针对无相机位姿的随拍长视频场景，提出增量联合优化框架同时优化相机位姿和 3DGS，设计基于 MASt3R 先验的鲁棒位姿估计模块和自适应八叉树锚点形成机制，解决位姿漂移、几何初始化不准和内存限制问题。

## 研究背景与动机

随拍长视频是 3D 内容的重要来源，但给新视角合成带来独特挑战：

**位姿估计困难**：COLMAP 在随拍视频中经常失败；MASt3R 等基础模型在长序列中积累误差和漂移

**内存限制**：CF-3DGS 等 COLMAP-free 方法在大规模场景中遭遇 OOM

**复杂轨迹**：LocalRF 在不规则相机运动下产生碎片化重建

**缺乏全局一致性**：增量式方法容易陷入局部最优

## 方法详解

### 整体框架

LongSplat 采用完全增量的管线：
1. 初始化：MASt3R 全局对齐点云 → 八叉树锚点 3DGS
2. 全局优化：联合优化所有位姿和 Gaussians
3. 逐帧插入：PnP 位姿估计 + 光度精炼 + 锚点扩展
4. 局部-全局交替优化

### 关键设计一：八叉树锚点形成（Octree Anchor Formation）

不同于 Scaffold-GS 的固定分辨率体素，LongSplat 根据点云密度自适应细分：

$$\epsilon_{l+1} = \frac{1}{2}\epsilon_l$$

密度低于 $\tau_{\text{prune}}$ 的体素被剪枝，高于 $\tau_{\text{split}}$ 的体素递归细分至最大层级 $L$。锚点的空间尺度与体素大小成正比：$s_v \propto \epsilon_v$。新锚点与现有锚点重叠检查可防止重复。

### 关键设计二：鲁棒位姿估计模块

对每个新帧 $t$：

1. **PnP 初始化**：利用 MASt3R 的 2D 对应关系及之前帧的渲染深度反投影得到 2D-3D 对应，通过 PnP+RANSAC 估算位姿

$$X_i = D_{t-1}(x_i) \cdot K^{-1}\tilde{x}_i$$

2. **光度精炼**：最小化渲染图像与实际帧的差异

$$\mathcal{L}_{\text{photo}} = \sum_{p \in \Omega}\|I_t(p) - \hat{I}_t(p)\|^2$$

3. **深度尺度校正**：对齐 MASt3R 深度与渲染深度的尺度

$$\hat{s}_t = \frac{\langle D_{t-1}, D_t^{\text{align}}\rangle}{\langle D_t^{\text{align}}, D_t^{\text{align}}\rangle}$$

4. **遮挡感知扩展**：通过前向 warp 检测新暴露区域，反投影为新 3D 点并转为八叉树锚点

5. **回退机制**：PnP 失败时触发全局重优化后重试

### 关键设计三：可见性自适应局部窗口

通过锚点可见性的 IoU 度量帧间共视关系：

$$\text{IoU}(t, t') = \frac{|\mathcal{V}(t) \cap \mathcal{V}(t')|}{|\mathcal{V}(t) \cup \mathcal{V}(t')|}$$

低于阈值 $\tau$ 的帧被排除在局部优化窗口外，确保 Gaussians 获得平衡的多视角监督。

### 总损失

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{photo}} + \lambda_{\text{depth}}\mathcal{L}_{\text{depth}} + \lambda_{\text{reprojection}}\mathcal{L}_{\text{reprojection}}$$

## 实验

### 主实验：Free 数据集

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 备注 |
|------|-------|-------|--------|------|
| COLMAP + Scaffold-GS | 失败 | - | - | 位姿估计失败 |
| CF-3DGS | - | - | - | OOM |
| LocalRF | 较低 | 较低 | 较高 | 碎片化重建 |
| MASt3R + Scaffold-GS | 中等 | 中等 | 中等 | 位姿不准 |
| **LongSplat** | **最高** | **最高** | **最低** | 鲁棒 |

LongSplat 在所有场景上一致性地超越基线。

### 消融实验：关键组件

| 八叉树锚点 | 位姿精炼 | 增量优化 | PSNR↑ |
|---------|---------|---------|-------|
| ✗ (固定体素) | ✗ | ✗ | 基线 |
| ✓ | ✗ | ✗ | 提升 + 节省内存 |
| ✓ | ✓ | ✗ | 显著提升 |
| ✓ | ✓ | ✓ | **最高** |

八叉树锚点在保持质量的同时显著降低内存使用；位姿精炼是性能提升的最大贡献者。

### 关键发现
- COLMAP 在 14/19 个 Free 场景中完全失败，而 LongSplat 全部成功
- 八叉树相比固定体素内存减少 30-50%
- PnP 回退机制在约 5-10% 帧上被触发，显著提升鲁棒性
- 在 Tanks and Temples 和 Hike 数据集上同样全面领先

## 亮点与洞察

1. **端到端无位姿重建**：不依赖 COLMAP 或精确相机标定
2. **增量式设计**：逐帧处理，内存可控，适合长序列
3. **MASt3R 作为软先验**：而非硬约束，通过联合优化逐步修正
4. **鲁棒性机制完善**：PnP 回退 + 光度精炼 + 尺度校正组合确保稳定性

## 局限性
- 初始帧的 MASt3R 估计质量对整个重建影响较大
- 极端快速运动或纯旋转场景中 PnP 可能不稳定
- 全局优化随帧数增长变慢
- 相机内参未知时依赖 MASt3R 估计的焦距

## 相关工作
- CF-3DGS：渐进式无位姿 3DGS 优化
- LocalRF：局部化辐射场构建
- MASt3R / DUSt3R：3D 基础模型
- Scaffold-GS / Octree-GS：锚点/八叉树 3DGS

## 评分
- **创新性**: ⭐⭐⭐⭐ — 八叉树锚点+增量联合优化+PnP回退的完整系统
- **实用性**: ⭐⭐⭐⭐⭐ — 直接处理手机随拍视频，极具实用价值
- **实验完整度**: ⭐⭐⭐⭐ — 多数据集验证+消融完整
- **写作质量**: ⭐⭐⭐⭐ — 系统描述清晰，管线图直观

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [\[ICCV 2025\] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)

<!-- RELATED:END -->
