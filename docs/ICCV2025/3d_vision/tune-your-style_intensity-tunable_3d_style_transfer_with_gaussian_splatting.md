---
title: >-
  [论文解读] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting
description: >-
  [3D视觉] 提出 Tune-Your-Style，首个强度可调的 3D 风格迁移范式，通过 Gaussian 神经元显式建模风格强度并参数化可学习 style tuner，配合两阶段优化策略，实现用户自由调节风格注入的程度。
tags:
  - 3D视觉
---

# Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2602.00618](https://arxiv.org/abs/2602.00618)
- **作者**: Yian Zhao, Rushi Ye, Ruochong Zheng, Zesen Cheng, Chaoran Feng, Jiashu Yang, Pengchong Qiao, Chang Liu, Jie Chen
- **机构**: 北京大学深圳研究生院, 清华大学, 大连理工大学
- **代码**: [项目主页](https://zhao-yian.github.io/TuneStyle)
- **领域**: 3D视觉 / 3D风格迁移
- **关键词**: 3D Gaussian Splatting, 风格迁移, 强度可调, 扩散模型先验, 跨视图一致性

## 一句话总结
提出 Tune-Your-Style，首个强度可调的 3D 风格迁移范式，通过 Gaussian 神经元显式建模风格强度并参数化可学习 style tuner，配合两阶段优化策略，实现用户自由调节风格注入的程度。

## 研究背景与动机

### 问题定义
3D 风格迁移旨在将参考风格图像的艺术效果迁移到 3D 场景中，同时保持内容一致性和多视角一致性。核心挑战在于**平衡内容保持与风格注入**。

### 现有方法局限
当前主流方法（StyleGaussian、G-Style、InstantStyleGaussian 等）均采用**固定输出范式**：
- 每次训练只能产出一个固定平衡点的风格化结果
- 不同用户对内容-风格平衡的需求不同，但无法灵活调整
- 若用户觉得风格过强或过弱，只能重新训练模型或调参

### 核心创新思路
引入**强度可调范式**：用户训练一次后，可通过一个 style tuner（范围 0%~100%）自由调节风格注入强度，无需重新训练。关键在于如何显式建模"风格强度"这一概念。

## 方法详解

### 整体框架
Tune-Your-Style 包含两个核心组件：
1. **Intensity-tunable Style Injection (ISI)**：显式建模风格强度 + 可学习 style tuner
2. **Tunable Stylization Guidance (TSG)**：扩散模型生成多视图一致的风格化引导 + 两阶段优化

### 1. 强度可调风格注入 (ISI)

**Gaussian 神经元建模风格强度**：为每个 Gaussian 原语分配一个可学习神经元，预测其所有属性（位置、缩放、旋转、透明度、颜色）的偏移量：

$$\mathcal{G}(S_k, \Theta) = \{\Delta_k \boldsymbol{\mu}_i, \Delta_k \boldsymbol{S}_i, \Delta_k \boldsymbol{R}_i, \Delta_k \sigma_i, \Delta_k \boldsymbol{c}_i\}_{i=1}^{N}$$

风格化场景为原始场景 + 偏移量：$\hat{\Theta}_k = \Theta + \mathcal{G}(S_k, \Theta)$

**Style Tuner 参数化**：引入阶梯函数 $\mathcal{H}(\beta)$ 将连续的风格调节信号量化为 $Z=10$ 个离散级别，并建立到可学习嵌入的双射映射 $\mathcal{V}_\beta = f(\mathcal{H}(\beta))$：

$$\hat{\Theta}_k^{\beta} = \Theta + \mathcal{V}_\beta \odot \mathcal{G}(S_k, \Theta)$$

当 $\beta = 0\%$ 时，$\mathcal{V}_\beta$ 压制偏移量实现零风格；$\beta = 100\%$ 时保持完整偏移。

**3D Gaussian 过滤器**：计算每个 Gaussian 在所有训练视角中的重要性得分 $\psi_i$，过滤掉最不重要的 50%，避免冗余 Gaussian 在风格化渲染中产生伪影：

$$\psi_i = \sum^{C} \sum^{H \times W} \kappa(\Theta_i) \cdot \sigma_i \cdot \prod_{j=1}^{i-1}(1-\sigma_j)$$

### 2. 可调风格化引导 (TSG)

**扩散模型风格化引导**：使用 IP-Adapter-SDXL 作为 2D 扩散模型，对渲染视图进行风格迁移生成风格化视图 $\mathcal{I}_v^k$。

**跨视图风格对齐**：解决扩散模型生成的多视角风格化结果缺乏 3D 一致性的问题：
- 随机选择一个锚定视图，将其特征注入其他视图的 self-attention 层
- 进行跨视图特征匹配实现内容校准（将锚定视图的内容特征反投影到 3D 再投影到目标视角）
- 使用 mutual self-attention：

$$\mathcal{A}_{v_c}^{t} = \text{softmax}\left(\frac{Q_{v_c}^{t} \cdot ([K_{v_a \rightarrow v_c}^{t}, K_{v_c}^{t}])^{T}}{\sqrt{d}}\right) \cdot [V_{v_a \rightarrow v_c}^{t}, V_{v_f}^{t}]$$

**两阶段优化**：

**第一阶段（full-style guidance，2000步）**：
- 仅优化 Gaussian 神经元 $\mathcal{G}$ 和 full offset 嵌入 $\mathcal{V}_{full}$
- 损失：$\mathcal{L}_{full}^{s_1} = \mathcal{L}_1(\mathcal{I}_v^{\tilde{t}_1}, \mathcal{I}_v^k) + \mathcal{L}_{lpips}(\mathcal{I}_v^{\tilde{t}_1}, \mathcal{I}_v^k)$

**第二阶段（tunable guidance，2000步）**：
- 冻结神经元和 full offset 嵌入，仅优化其他级别的嵌入
- 随机采样中间 $\beta$ 值，加权混合零风格引导和全风格引导：

$$\mathcal{L}_{tunable} = (1-\beta_{\tilde{t}_2}) \cdot \mathcal{L}_{zero} + \beta_{\tilde{t}_2} \cdot \mathcal{L}_{full}^{s_2}$$

### 损失函数
最终训练包含 L1 损失和 LPIPS 感知损失的加权组合，分别用于零风格引导（与原始渲染比较）和全风格引导（与风格化视图比较）。

## 实验关键数据

### 主实验：3DGS 风格迁移方法对比

| 方法 | 短程一致性↓ (LPIPS/RMSE) | 长程一致性↓ (LPIPS/RMSE) | CLIP_S↑ | CLIP_Sdir↑ | 用户研究↑ |
|------|---------------------------|---------------------------|---------|-----------|----------|
| StyleGaussian | 0.067 / 0.070 | 0.126 / 0.108 | 0.2134 | 0.2223 | 2.79±0.16 |
| G-Style | 0.044 / 0.059 | 0.093 / 0.096 | 0.2406 | 0.2391 | 3.10±0.40 |
| InstantStyleGaussian | 0.053 / 0.062 | 0.108 / 0.113 | 0.2204 | 0.2160 | 2.06±0.22 |
| **Ours** | **0.033 / 0.035** | **0.062 / 0.067** | **0.2619** | **0.2881** | **3.97±0.13** |

- 多视角一致性（短程和长程）全面最优，远优于其他方法
- CLIP 相似度和方向相似度均最高，表明风格保真度最好
- 用户研究得分 3.97/5.0，显著领先其他方法

### 消融实验

**两阶段优化消融**：
- 去除两阶段优化（同时训练 zero-style 和 full-style）→ 风格化质量严重下降（过度风格化），且 style tuner 失去调节能力
- 原因：零风格引导和全风格引导的随机组合产生不稳定监督信号

**跨视图风格对齐消融**：

| 配置 | 效果 |
|------|------|
| 原始扩散生成 | 视觉良好但缺乏 3D 一致性 |
| + 特征注入 | 近锚点视角一致性好，远距离视角内容扭曲 |
| + 内容校准 | 远距离视角内容和布局保持完好，风格纹理一致 |

### 关键发现
1. **强度可调确实可行**：style tuner 从 0% 到 100% 可以平滑控制风格注入强度
2. **多风格组合**：结合 SAM 分割，可对场景不同区域施加不同风格并独立调节强度
3. **两阶段训练至关重要**：先稳定学习 full-style 偏移量，再学习中间级别嵌入
4. **训练效率**：单 V100 GPU 上每个场景约 20 分钟完成训练

## 亮点与洞察

1. **范式创新**：首次提出强度可调的 3D 风格迁移范式，从"固定输出"到"连续可调"是一个质的飞跃
2. **显式建模风格强度**：通过 Gaussian 神经元预测属性偏移 + 阶梯函数量化 + 可学习嵌入，三层设计实现优雅的强度控制
3. **跨视图风格对齐**：基于深度反投影-重投影的内容校准方案，有效解决扩散生成结果的多视角不一致问题
4. **全属性偏移**：不仅修改颜色，还预测位置/缩放/旋转/透明度偏移，可同时迁移几何和外观风格

## 局限性

- 量化级别 $Z=10$ 是手动设定的，可能不适合所有场景
- 扩散模型生成风格化引导增加前处理时间
- Gaussian 过滤器移除 50% 原语可能在某些场景中导致信息丢失
- 暂不支持局部分层编辑的精细粒度控制

## 相关工作与启发

- **vs StyleGaussian/G-Style**：从 VGG 特征对齐转向扩散先验引导，避免耗时的编码器-解码器训练
- **vs SDS/IDU**：提出的跨视图风格对齐比 SDS 损失和迭代数据更新更适合处理精细风格纹理
- **启发**：Gaussian 神经元 + style tuner 的设计思路可扩展到其他 3D 属性编辑任务（如光照、材质）
- **多风格组合**能力对游戏和影视行业有直接应用价值

## 评分 ⭐⭐⭐⭐
创新性强（首个强度可调的 3D 风格迁移范式），实验全面，定量和用户研究均大幅领先。跨视图风格对齐设计精巧。但部分设计选择（量化级别、过滤比例）缺乏充分论证。

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[ICCV 2025\] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)

<!-- RELATED:END -->
