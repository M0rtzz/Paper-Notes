---
title: >-
  [论文解读] Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?
description: >-
  [AAAI2026][3D视觉][3D Gaussian Splatting] 首次系统性地揭示了 3DGS 水印框架的脆弱性，提出 GSPure 框架通过视角感知权重累积和几何特征聚类精准分离并去除水印相关的 Gaussian 原语，在水印 PSNR 最高降低 16.34dB 的同时保持原始场景损失不足 1dB。
tags:
  - AAAI2026
  - 3D视觉
  - 3D Gaussian Splatting
  - watermark purification
  - copyright protection
  - HDBSCAN clustering
---

# Can Protective Watermarking Safeguard the Copyright of 3D Gaussian Splatting?

**会议**: AAAI2026  
**arXiv**: [2511.22262](https://arxiv.org/abs/2511.22262)  
**代码**: 无  
**领域**: 3d_vision  
**关键词**: 3D Gaussian Splatting, watermark purification, copyright protection, HDBSCAN clustering

## 一句话总结

首次系统性地揭示了 3DGS 水印框架的脆弱性，提出 GSPure 框架通过视角感知权重累积和几何特征聚类精准分离并去除水印相关的 Gaussian 原语，在水印 PSNR 最高降低 16.34dB 的同时保持原始场景损失不足 1dB。

## 背景与动机

### 领域现状

**领域现状**：3D Gaussian Splatting (3DGS) 已成为高效表示和渲染 3D 场景的主流方法，通过各向异性高斯原语实现高保真视觉效果和计算效率。由于训练成本高，3DGS 模型本身具有重要的数字资产价值，因此涌现了一系列水印保护方案（如 GS-Hider、Splats in Splats、SecureGS）用于所有权验证和溯源追踪。

然而，现有工作仅关注**如何嵌入水印**，却缺乏对水印鲁棒性的系统性攻击评估。传统 2D 水印去除方法（裁剪、旋转、神经网络擦除等）无法直接迁移到 3DGS 场景，因为 3DGS 水印是嵌入到模型的几何和光度属性中，而非仅在渲染图像表面。这一空白促使作者首次从攻击者视角系统性地审视 3DGS 水印的安全性。

### 核心矛盾

**核心矛盾**：**本文目标**：**现有 3DGS 水印方案是否真正安全？** 作者发现答案是否定的。核心观察是：水印相关的 Gaussian 原语在渲染原始场景时的贡献模式与场景原语存在显著差异——水印原语在多视角渲染中通常表现出不一致或特定视角的行为，其渲染贡献权重显著低于场景原语。这种差异使得精准分离成为可能。

## 方法详解

GSPure 框架包含三个核心模块：

### 1. 视角感知高斯权重累积 (View-Aware Gaussian Weight Accumulation)

对每个 Gaussian 原语 $\mathcal{G}_k$，计算其在多个渲染视角下的贡献权重。具体而言：

- 定义射线-高斯交叉能量 $\mathcal{E}(\mathcal{G}_k, \mathbf{o}_v, \mathbf{r}_v)$，衡量给定视角 $v$ 下射线与高斯的交叉程度
- 考虑遮挡效应，通过 alpha-blending 机制计算视角依赖贡献 $\omega_{k,v}$，即透过前序高斯的累积透射率
- 在 $N$ 个视角上取平均得到视角累积权重 $\omega_k = \frac{1}{N}\sum_{v=1}^{N}\omega_{k,v}$

关键直觉：水印原语因其不一致的渲染行为，$\omega_k$ 值通常较低；场景原语的贡献则跨视角保持一致且较高。

### 2. 几何精确特征聚类 (Geometrically Accurate Feature Clustering)

仅依赖权重无法完美分离（视角不连续性和场景-水印耦合问题），因此构建联合特征：

- 将位置 $\mathbf{p}_k$、不透明度 $\alpha_k$ 和累积权重 $\omega_k$ 分别标准化后拼接为高维特征向量 $\mathbf{F}_k$
- 采用自适应密度聚类 HDBSCAN 进行分簇，自动发现水印相关的 Gaussian 聚类和噪声点

### 3. 自适应剪枝 (Adaptive Pruning)

设计双层级剪枝策略：

- **簇级剪枝**：计算每个簇的平均权重 $\widetilde{\Omega}(C_i)$，若低于阈值 $\bar{\omega}/\tau_c$ 则整簇移除
- **噪声点级剪枝**：对未归入任何簇的噪声点，若单点权重 $\omega_k < \bar{\omega}/\tau_n$ 则单独移除
- 阈值因子 $\tau_c$ 和 $\tau_n$ 根据全局平均权重动态调整，兼顾水印去除效果和场景保真

## 实验关键数据

在 Mip-NeRF 360 数据集上，对三种主流 3DGS 水印方法进行评估：


### 主实验

| 攻击方法 | Splats in Splats Score | GS-Hider Score | SecureGS Score |
|---------|----------------------|----------------|----------------|
| Random Pruning | -0.88 | 0.68 | -1.59 |
| Feature Scaling | -9.38 | -1.42 | 0.39 |
| Gaussian Noise | -7.15 | 0.11 | -0.55 |
| GOF | 2.24 | -0.93 | 0.64 |
| **GSPure** | **15.21** | **10.16** | **5.03** |

- Score 定义：$\Delta PSNR_{message} - \Delta PSNR_{scene}$，越高越好（水印去除多、场景损失少）
- GSPure 在所有三种水印框架上 Score 均大幅领先第二名（分别领先 12.97、9.48、4.39）
- 水印 PSNR 最大降幅达 16.34dB，场景保真损失普遍 <1dB
- 消融实验证实：权重累积 + 不透明度 + 聚类三者联合效果最优，缺少任何一个都会导致性能下降或场景损伤加剧

## 亮点与洞察

1. **问题定义新颖**：首次从攻击者角度系统评估 3DGS 水印安全性，揭示了该领域被忽视的脆弱性
2. **方法设计精巧**：利用水印原语与场景原语在多视角渲染贡献上的本质差异，无需已知水印方案的任何先验信息
3. **通用性强**：对三种不同技术路线的水印方案（SH 加密、译码器隐藏、锚点设计）均有效
4. **可视化有说服力**：点云聚类可视化清晰展示了水印原语的空间聚集特性，直观验证了方法的可靠性

## 局限与展望

1. **阈值需手动调整**：$\tau_c$ 和 $\tau_n$ 对不同水印方法需要不同设置（如 SecureGS 用 (2,3) 而非默认 (4,4)），自动化程度有待提高
2. **SecureGS 上效果相对较弱**：基于 Scaffold-GS 锚点设计的水印更难去除，Score 仅 5.03（vs. Splats in Splats 的 15.21）
3. **仅评估场景隐藏类水印**：未涉及渲染图像级水印或其他保护机制
4. **缺乏对抗性水印的讨论**：若水印方案针对性地对抗 GSPure 的聚类策略进行强化，当前方法的有效性可能降低
5. **计算开销未详细分析**：修改了 3DGS 的 CUDA 实现来计算渲染贡献权重，但未报告额外时间成本

## 相关工作与启发

| 方法 | 类型 | 针对 3DGS | 保持场景质量 | 水印去除效果 |
|------|------|----------|------------|------------|
| 随机剪枝 | 朴素基线 | 部分适用 | 中等 | 差 |
| 特征缩放 | 2D 迁移 | 部分适用 | 差 | 差 |
| 高斯噪声注入 | 2D 迁移 | 部分适用 | 中等 | 差 |
| GOF 表面提取 | 几何方法 | 适用 | 很差 | 中等 |
| **GSPure** | **3D 专用** | **专门设计** | **好** | **优秀** |

GOF 虽能去除部分水印，但会对原始场景造成不可逆损伤（Scene PSNR 降幅可达 10+dB）；GSPure 则在两者之间实现了最佳平衡。

## 相关工作与启发

- **安全与攻击的共演化**：本工作以"攻击"视角为 3DGS 水印保护提出了重要的安全性基准，推动更鲁棒水印方案的发展
- **视角一致性作为信号**：利用多视角渲染一致性分离信息的思路可推广到其他 3DGS 编辑/分析任务
- **聚类+剪枝范式**：HDBSCAN 在高斯原语层面的应用展示了密度聚类在非结构化 3D 点云中的潜力
- **与 NeRF 水印研究的连接**：3DGS 水印攻击的研究范式可借鉴 NeRF 领域的对抗鲁棒性工作

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首次系统性攻击 3DGS 水印，开创新方向)
- 实验充分度: ⭐⭐⭐⭐ (三种水印方法 × 九个场景 × 五种基线，消融完整，但缺少计算成本分析)
- 写作质量: ⭐⭐⭐⭐ (结构清晰、动机充分，可视化优秀)
- 价值: ⭐⭐⭐⭐⭐ (对 3DGS 版权保护领域具有重要警示意义)

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting](../../CVPR2025/3d_vision/guardsplat_efficient_and_robust_watermarking_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] Protecting NeRFs' Copyright via Plug-And-Play Watermarking Base Model](../../ECCV2024/3d_vision/protecting_nerfsapos_copyright_via_plug-and-play_watermarking_base_model.md)
- [\[CVPR 2025\] 3D-GSW: 3D Gaussian Splatting for Robust Watermarking](../../CVPR2025/3d_vision/3d-gsw_3d_gaussian_splatting_for_robust_watermarking.md)
- [\[CVPR 2026\] Where, What, Why: Toward Explainable 3D-GS Watermarking](../../CVPR2026/3d_vision/where_what_why_toward_explainable_3d-gs_watermarking.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)

<!-- RELATED:END -->
