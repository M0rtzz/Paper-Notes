---
title: >-
  [论文解读] Surface-Based Visibility-Guided Uncertainty for Continuous Active 3D Neural Reconstruction
description: >-
  [AAAI 2026][3D视觉][主动3D重建] 提出基于表面的可见性场(SBV)，通过SDF推导的表面置信度和体素网格更新机制在连续主动学习过程中准确估计不确定性的可见性，指导Next-Best View选择，在DTU/Blender/TanksAndTemples/BlendedMVS四个基准上图像渲染质量提升最高11.6%。
tags:
  - AAAI 2026
  - 3D视觉
  - 主动3D重建
  - 神经隐式表面
  - 不确定性估计
  - Next-Best View
  - 可见性推理
---

# Surface-Based Visibility-Guided Uncertainty for Continuous Active 3D Neural Reconstruction

**会议**: AAAI 2026  
**arXiv**: [2405.02568](https://arxiv.org/abs/2405.02568)  
**代码**: https://github.com/hskAlena/Surface-Based-Visibility  
**领域**: 3D视觉  
**关键词**: 主动3D重建, 神经隐式表面, 不确定性估计, Next-Best View, 可见性推理

## 一句话总结
提出基于表面的可见性场(SBV)，通过SDF推导的表面置信度和体素网格更新机制在连续主动学习过程中准确估计不确定性的可见性，指导Next-Best View选择，在DTU/Blender/TanksAndTemples/BlendedMVS四个基准上图像渲染质量提升最高11.6%。

## 研究背景与动机
**领域现状**：主动3D神经重建通过评估场景不确定性来选择Next-Best View (NBV)，减少数据采集和计算成本。现有方法用体密度分布的方差（ActiveNeRF）、颜色分布的方差、熵（ActiveRMAP）或Fisher信息（FisherRF）来量化不确定性。

**现有痛点**：(1) 基于体密度的可见性估计在模型未充分收敛（underfitted）时严重不准确——低密度区域的高不确定性被体渲染过程忽略，导致信息量大的候选视角被排除。(2) 现有考虑可见性的方法（NVF）只能在训练子阶段完全收敛后才估计可见性，无法适用于连续主动学习场景。

**核心矛盾**：体渲染（volume rendering）在训练早期阶段无法可靠地确定不确定区域的可见性——低密度浮标（floaters）和欠拟合表面的不确定性被系统性地低估。但连续主动学习要求实时评估信息增益以进行view selection。

**本文目标** 在连续主动学习过程中（模型持续训练同时选择新视角），如何准确估计不同区域不确定性的可见性，即区分"表面上的不确定性"和"空旷区域的不确定性"？

**切入角度**：用NeuS的SDF值直接推导表面置信度（SDF符号变化检测），绕开体密度的不可靠性。通过体素网格存储和鲁棒更新表面置信度，实现快速且稳定的可见性推理。

**核心 idea**：不依赖体密度而是用SDF符号变化检测表面，通过体素网格鲁棒更新表面置信度，在连续训练过程中准确判断不确定性是否位于可见表面上

## 方法详解

### 整体框架
基于NeuS神经隐式表面网络，扩展颜色预测为高斯分布 $c(\mathbf{r}(t)) \sim \mathcal{N}(\bar{c}, \beta^2)$，其中方差 $\beta^2$ 为渲染不确定性。同时维护一个体素网格存储两类信息：(1) 每个体素的不确定性（颜色方差 $\beta^2$）；(2) 每个体素的表面置信度（SDF符号检测得分）。信息增益计算基于表面引导的可见性：射线穿过表面体素时只计算表面体素的熵，射线未遇到表面时累加所有穿过体素的熵。

### 关键设计

1. **渲染不确定性估计**:

    - 功能：为场景中每个3D点估计颜色的不确定性
    - 核心思路：将NeuS网络扩展为 $F_\Theta: (\mathbf{x}, \mathbf{d}) \rightarrow (g(\mathbf{x}), \bar{c}, \beta^2)$，颜色方差 $\beta^2$ 不依赖视角方向。不确定性损失为负对数似然：$\mathcal{L}_u = \frac{1}{M}\sum_i (\frac{\|\bar{\mathcal{C}}(\mathbf{r}_i) - C(\mathbf{r}_i)\|^2}{2\mathcal{B}^2(\mathbf{r}_i)} + \frac{\log \mathcal{B}^2(\mathbf{r}_i)}{2})$
    - 设计动机：通过高斯分布建模颜色，方差自然反映了该点的重建质量——高方差意味着多视角下颜色不一致或训练不充分

2. **基于SDF的表面置信度 (Surface Confidence)**:

    - 功能：在每个体素中检测是否存在表面，并鲁棒地更新置信度
    - 核心思路：沿射线在体素中心及前后各采样一个点（间距为NeuS的步长 $1/s$），若前后两点SDF值符号相反（乘积为负），则判定存在表面。置信度为二值（1=有表面，0=无表面），更新策略为 $\max(\text{previous} \times 0.95, \text{current})$，阈值0.8以上视为表面体素。衰减率0.95补偿了采样噪声
    - 设计动机：体密度在训练早期极不可靠（欠拟合区域密度低），但SDF的符号变化是更稳定的表面指示器。体素网格的渐进式更新使表面估计在训练过程中持续改善

3. **SBV引导的信息增益计算 (SBV-Guided IG)**:

    - 功能：根据表面可见性区分不同类型的射线，准确计算候选视角的信息增益
    - 核心思路：颜色熵 $H(c) = \frac{1}{2}\log(2\pi\beta^2) + \frac{1}{2}$。对于穿过表面体素的射线，IG只取表面体素的熵；对于未遇到表面的射线（空旷区域或模糊区域），IG累加所有穿过体素的熵。公式为 $G_s(v) = \frac{1}{N}\sum_{r \in \mathcal{R}_v}\sum_{x \in \tilde{\mathcal{X}}_r} H(c(x))$，其中 $\tilde{\mathcal{X}}_r = \mathcal{X}_r \cap \mathcal{S}$（射线遇到表面时）或 $\tilde{\mathcal{X}}_r = \mathcal{X}_r$（未遇到表面时）
    - 设计动机：表面上的不确定性直接影响渲染质量，应被优先关注。空旷区域虽不确定性可能高但对输出影响小——SBV通过表面置信度自然实现了这种区分

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_s + \omega\mathcal{L}_u$，其中 $\mathcal{L}_s$ 包含颜色L1损失和Eikonal正则化，$\omega$ 平衡两者。两阶段多视角选择策略：每轮选择2或10个NBV，使用距离感知策略保证多样性——候选视角需满足与所有已选视角距离超过阈值 $\tau$（初始1.732，衰减0.95），在满足条件的候选中选IG最高者。训练使用频率正则化防止少图过拟合，warm-up阶段稳定早期训练。

## 实验关键数据

### 主实验（四大基准数据集）

| 方法 | DTU-PSNR↑ | Blender-PSNR↑ | TNT-PSNR↑ | BlendedMVS-PSNR↑ |
|------|-----------|--------------|-----------|-----------------|
| Random | 27.69 | 16.24 | 17.17 | 26.21 |
| FVS | 27.09 | 20.07 | 17.94 | 25.49 |
| Entropy | 24.21 | 15.41 | 16.70 | 25.72 |
| ActiveNeRF* | 26.30 | 19.25 | 18.62 | 26.57 |
| FisherRF* | 27.78 | 20.48 | 18.44 | 25.71 |
| **Ours (SBV)** | **28.19** | **21.22** | **20.49** | **26.80** |

### 消融实验与新数据集

| 配置/方法 | DTU-PSNR↑ | DTU-Chamfer↓ | ImBView-PSNR↑ |
|-----------|-----------|-------------|--------------|
| SBV (有表面引导) | **28.19** | **2.002** | **32.23** |
| 无表面引导 (所有体素累加) | 27.18 | 2.251 | - |
| Random | 27.69 | 2.920 | 27.88 |
| ActiveNeRF* | 26.30 | 2.395 | 29.60 |
| FisherRF* | 27.78 | 3.476 | 27.77 |

### 关键发现
- SBV在四大基准上全面超越所有对比方法，尤其在TanksAndTemples大规模场景上PSNR领先第二名1.87（20.49 vs 18.62），提升11.6%
- 表面置信度引导是关键：去掉表面引导后DTU上PSNR从28.19降至27.18，Chamfer距离从2.002增至2.251
- 在ImBView不均衡视角数据集上，SBV比FisherRF*高出4.46 PSNR（32.23 vs 27.77），因为FisherRF偏向选择端部视角，SBV能选出覆盖各角度的多样化视角
- NBV选择速度：SBV仅需0.8秒，显著快于ActiveNeRF*（10.8秒）和FisherRF*（13.5秒），与基于体素的Entropy方法（0.5秒）效率相当
- 网格重建方面SBV同样最优（DTU Chamfer 2.002 vs 第二名ActiveNeRF* 2.395）

## 亮点与洞察
- **SDF符号变化检测表面**：绕开了体密度在训练早期不可靠的根本问题。SDF的零交叉点天然标记表面位置，即使网络未收敛也能提供有用信号。这个思路可迁移到任何需要在训练中间状态估计几何的任务
- **体素网格的鲁棒更新策略**：$\max(\text{previous} \times 0.95, \text{current})$ 这种"慢衰减+即时更新"设计确保了表面置信度单调趋于稳定，不会因训练波动而频繁翻转——简单但有效
- **视角多样性保证**：距离阈值衰减策略 $\tau \times 0.95$ 优雅地平衡了信息增益和视角分布——优先选远距离高IG视角，无符合条件时逐步放宽约束
- **ImBView数据集设计**：通过刻意制造视角不均衡（75%正常/12.5%高角/12.5%低角），清晰暴露了不同NBV策略的偏好偏差

## 局限与展望
- **未考虑视角间位移成本**：在机器人主动重建中，相邻视角间的移动成本是关键约束，本文未涉及
- **基础架构限制**：使用NeuS而非NeuS2/Neuralangelo，因为多分辨率哈希编码的局部性问题影响Eikonal loss和表面精度
- **仅支持前景物体重建**：需要object mask，不适用于室内场景或需要深度信息的环境
- **背景噪声**：SBV IG可视化中存在NerfAcc网格采样导致的持续背景噪声模式

## 相关工作与启发
- **vs NVF**: NVF在子训练阶段完全收敛后才估计可见性引导的不确定性，采用分阶段训练-评估-选择范式，无法用于连续主动学习。SBV通过体素网格解耦表面置信度更新，实现训练过程中的在线可见性估计
- **vs ActiveNeRF**: ActiveNeRF基于颜色方差但不考虑可见性，在DTU的水果场景中无法区分表面和非表面的不确定性，因此无法选到覆盖复杂遮挡区域的视角
- **vs FisherRF**: FisherRF用Fisher信息矩阵量化信息增益，计算较慢（13.5s vs SBV的0.8s）。在ImBView上暴露出严重视角选择偏差（偏好端部视角），SBV的表面引导更均衡
- **vs Entropy方法**: Entropy方法用占用概率的熵但不考虑表面遮挡，导致遮挡后方的模糊区域也被计入IG，在DTU上网格重建Chamfer距离3.644远劣于SBV的2.002
- **体素方法启发**：传统体素占用概率方法（Isler et al.）可稳定更新但不适配神经表示，SBV用SDF替代占用概率实现了两全

## 评分
- 新颖性: ⭐⭐⭐⭐ 用SDF符号变化替代体密度来检测表面的思路简洁有效，但整体框架是对已有方法的组合改进
- 实验充分度: ⭐⭐⭐⭐⭐ 四大基准+自建数据集，五种对比方法，1/2/4/10-image多种设置全覆盖，时间对比也有
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰（Fig.2的体密度可视化分析很有说服力），方法表述严谨，附录很充实
- 价值: ⭐⭐⭐⭐ 解决了连续主动学习中可见性估计的实际痛点，在多个基准上显著提升，NBV选择速度快（0.8s）

<!-- 笔记写于 2026-04-10，基于 paper_cache/AAAI2026/2405.02568.txt 全文缓存 -->

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction](../../ICLR2026/3d_vision/peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[AAAI 2026\] Domain Generalized Stereo Matching with Uncertainty-guided Data Augmentation](domain_generalized_stereo_matching_with_uncertainty-guided_data_augmentation.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[CVPR 2026\] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](../../CVPR2026/3d_vision/neural_gabor_splatting.md)

<!-- RELATED:END -->
