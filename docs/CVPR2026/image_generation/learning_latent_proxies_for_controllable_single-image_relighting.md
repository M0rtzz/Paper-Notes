---
title: >-
  [论文解读] Learning Latent Proxies for Controllable Single-Image Relighting
description: >-
  [CVPR 2026][图像生成][单图重光照] 提出 LightCtrl，一个基于扩散模型的单图重光照框架，通过小样本潜在代理编码器（few-shot latent proxy）提供轻量材质-几何先验、光照感知掩码引导空间选择性去噪、DPO 后训练增强物理一致性，实现对光照方向/强度/色温的精确连续控制，在合成和真实场景上均优于现有方法。
tags:
  - CVPR 2026
  - 图像生成
  - 单图重光照
  - PBR 先验
  - 潜在代理编码器
  - DPO 后训练
  - 光照感知掩码
---

# Learning Latent Proxies for Controllable Single-Image Relighting

**会议**: CVPR 2026  
**arXiv**: [2603.15555](https://arxiv.org/abs/2603.15555)  
**代码**: 无  
**领域**: 图像重光照 / 扩散模型  
**关键词**: 单图重光照, PBR 先验, 潜在代理编码器, DPO 后训练, 光照感知掩码

## 一句话总结

提出 LightCtrl，一个基于扩散模型的单图重光照框架，通过小样本潜在代理编码器（few-shot latent proxy）提供轻量材质-几何先验、光照感知掩码引导空间选择性去噪、DPO 后训练增强物理一致性，实现对光照方向/强度/色温的精确连续控制，在合成和真实场景上均优于现有方法。

## 研究背景与动机

单图重光照是一个严重欠约束问题：阴影、高光和漫反射依赖不可观测的几何和材质，且光照的微小变化可导致外观的大幅非线性变化。现有方法存在明确的能力边界：

**Intrinsic/G-buffer 方法**（如 Neural LightRig）需要密集 PBR 监督，脆弱且成本高

**纯潜空间方法**（如 LBM）缺乏物理基础，方向/强度控制不可靠

**端到端方法**（如 IC-Light）在肖像上效果好但缺乏几何感知，难泛化到复杂场景

**关键洞察**：精确重光照不需要完整的 intrinsic 分解；稀疏但物理有意义的线索——指示**哪里**光照应变化、**材质如何**响应——就足以引导扩散模型。这催生了轻量 proxy + mask 的设计思路。

## 方法详解

### 整体框架

LightCtrl 基于 Stable Diffusion 骨干：
- **输入**：源图像 $x_s^{\ell_s}$ + 相对光照编码 $\Delta\ell$（方向/强度/色温差异）
- **输出**：目标光照下的重光照结果 $\hat{x}_s^{\ell_t} = f_\theta(x_s^{\ell_s}, \Delta\ell)$
- **条件注入**：appearance token $t_{\mathrm{img}}$、lighting token $t_{\mathrm{light}}$、physics proxy token $t_{\mathrm{phys}}$

扩散损失加权：$\mathcal{L}_{\mathrm{diff}} = \|W \odot (\epsilon - \epsilon_\theta(z_t, t \mid t_{\mathrm{img}}, t_{\mathrm{light}}, t_{\mathrm{phys}}))\|_2^2$

### 关键设计

1. **Few-shot Latent Proxy Conditioning**

   轻量编码器-解码器 $E_\phi$ 从源图预测紧凑潜在代理 $\hat{\mathcal{B}} = \{a, n, r, m\} \in \mathbb{R}^{H \times W \times 8}$（albedo、法线、粗糙度、金属度）。仅在少量样本上使用 PBR 监督训练：

   $$\mathcal{L}_{\text{proxy}} = \lambda_a\|a-\hat{a}\|_1 + \lambda_n(1-\langle n, \hat{n}\rangle) + \lambda_r\|r-\hat{r}\|_1 + \lambda_m \mathrm{BCE}(m, \hat{m})$$

   Proxy maps 经空间池化+投射为条件 token $t_{\text{proxy}} = f_{\text{proj}}(E_\phi(x_s^{\ell_s})) \in \mathbb{R}^{1 \times 768}$ 注入去噪器。设计动机：不追求精确 intrinsic 重建，只需"够用"的材质-几何暗示来约束去噪轨迹。

2. **Lighting-Aware Mask Prediction**

   光照变化通常仅影响少量像素（阴影边界、高光区域）。基于源-目标对的线性亮度差异导出软 ground-truth mask：

   $$M_{\mathrm{gt}} = \mathcal{N}\left(\alpha|\log Y_t - \log Y_s| + (1-\alpha)D_{\mathrm{robust}}(Y_s, Y_t)\right)$$

   训练时无法访问目标图，因此轻量预测器 $M_\theta = m_\theta(x_s^{\ell_s}, \Delta\ell)$ 从源图+光照变化推断 mask（BCE+Dice loss 监督）。Mask 转化为空间权重图 $W$ 调制噪声重建损失，引导去噪器关注光照敏感区域。

3. **DPO Post-training for Latent Encoder**

   为补偿 PBR 监督的稀疏性，冻结主扩散骨干，对 PBR 编码器 $E_\phi$ 进行 DPO 风格后训练：GT PBR maps 为正样本 $y_{\text{pos}}$，当前编码器输出为负样本 $y_{\text{neg}}$，物理奖励 $\Delta r = r(y_{\text{pos}}) - r(y_{\text{neg}})$ 聚合 L1/角度/BCE 度量，冻结参考编码器提供稳定似然估计。DPO 目标增加高奖励预测的似然，显著改善代理的物理一致性。

### 损失函数 / 训练策略

- 主干在 ScaLight 上全量微调学习泛化光传输先验
- Proxy 分支小样本训练，DPO 后训练增强稳定性
- 最终扩散目标使用 lighting-aware 空间加权
- 构建 **ScaLight** 数据集：30万+可控3D物体、100万+渲染图像，系统变化光照方向/强度/色温，配有完整相机-灯光元数据

## 实验关键数据

### 主实验

ScaLight 测试集，三类光照变化（色温/方向/强度）：

| 方法 | 条件类型 | Temp RMSE↓/PSNR↑ | Pos RMSE↓/PSNR↑ | Energy RMSE↓/PSNR↑ |
|------|---------|-----------------|-----------------|-------------------|
| IC-Light | text | 0.397/8.21 | 0.375/8.65 | 0.380/8.63 |
| LBM | image | 0.064/27.8 | 0.084/23.1 | 0.073/25.3 |
| LumiNet | image | 0.172/15.8 | 0.146/17.8 | 0.164/16.2 |
| **Ours (full)** | **Light Info** | **0.053/30.2** | **0.074/25.6** | **0.083/27.1** |

场景级（MIIW）评测：

| 方法 | RMSE↓ | SSIM↑ | PSNR↑ |
|------|-------|-------|-------|
| IC-Light | 0.413 | 0.337 | 7.94 |
| LumiNet | 0.139 | 0.904 | 17.20 |
| **Ours** | 0.167 | 0.655 | **18.30** |

用户偏好研究（N=35）：场景级 55.73%，物体级 **81.45%**。

### 消融实验

| 配置 | Temp RMSE↓ | Pos PSNR↑ | Energy PSNR↑ |
|------|-----------|-----------|-------------|
| w/o proxy | 0.062 | 22.4 | 18.0 |
| w/o mask | 0.073 | 20.5 | 23.2 |
| w/o DPO | 0.114 | 19.8 | 17.5 |
| **Full** | **0.053** | **25.6** | **27.1** |

### 关键发现

- DPO 后训练对所有光照变化类型的提升最显著（移除后 RMSE 翻倍），是体系中最关键的组件
- Lighting-aware mask 对方向变化特别重要（阴影边界精确性）
- 用户偏好率在物体级达 81.45%，远超 IC-Light(11.45%) 和 LumiNet(4.3%)

## 亮点与洞察

- **"中间路线"哲学**：不追求完整 intrinsic 分解，也不放弃物理基础，用稀疏物理线索约束扩散
- **DPO 引入 PBR 质量优化**：将 RLHF 范式引入 intrinsic 估计是新颖的跨领域应用
- **ScaLight 大规模数据集**：30万物体+系统光照参数变化，填补了可控物体级重光照数据的空白

## 局限与展望

- 场景级性能仍与物体级有差距，复杂全局光传输（长距离阴影投射）是薄弱环节
- 高频几何和强高光区域易被过度平滑，proxy 缺少足够高频约束
- 训练主要在合成数据，真实场景泛化依赖 fine-tuning

## 相关工作与启发

- **IC-Light**：端到端扩散重光照，在肖像上强但缺物理建模，本文补充了物理先验
- **Neural LightRig**：密集 G-buffer 管线，本文用小样本 proxy 替代
- **LBM**：潜空间光照插值，物理基础弱，本文通过 proxy+mask 增强可控性

## 评分

- **新颖性**: ★★★★☆ — Latent proxy + DPO post-training 的组合新颖
- **技术深度**: ★★★★☆ — 三模块互补设计清晰，消融充分验证各组件贡献
- **实验充分度**: ★★★★★ — 合成/真实/用户研究/消融全面，ScaLight 数据集有持久价值
- **实用性**: ★★★★☆ — 连续光照控制实用性强，但复杂场景仍需改进

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [ScribbleLight: Single Image Indoor Relighting with Scribbles](../../CVPR2025/image_generation/scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](../../ECCV2024/image_generation/learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal](learning_latent_transmission_and_glare_maps_for_lens_veiling_glare_removal.md)
- [Self-Corrected Image Generation with Explainable Latent Rewards](self-corrected_image_generation_with_explainable_latent_rewards.md)
- [RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)

<!-- RELATED:END -->
