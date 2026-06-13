---
title: >-
  [论文解读] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation
description: >-
  [AAAI 2026][医学图像][心脏MRI重建] 提出 MoCo-INR，首次将隐式神经表示（INR）引入运动补偿（MoCo）框架，通过无监督方式实现心脏 MRI 的高质量动态重建，在超高加速因子（20x Cartesian / 69x Non-Cartesian）下显著优于现有无监督方法。 1. 领域现状： 心脏磁共振…
tags:
  - "AAAI 2026"
  - "医学图像"
  - "心脏MRI重建"
  - "运动补偿"
  - "隐式神经表示"
  - "无监督学习"
  - "欠采样重建"
---

# Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation

**会议**: AAAI 2026  
**arXiv**: [2511.11436](https://arxiv.org/abs/2511.11436)  
**代码**: [MoCo-INR](https://github.com/MeijiTian/MoCo-INR)  
**领域**: 医学图像  
**关键词**: 心脏MRI重建, 运动补偿, 隐式神经表示, 无监督学习, 欠采样重建

## 一句话总结

提出 MoCo-INR，首次将隐式神经表示（INR）引入运动补偿（MoCo）框架，通过无监督方式实现心脏 MRI 的高质量动态重建，在超高加速因子（20x Cartesian / 69x Non-Cartesian）下显著优于现有无监督方法。

## 研究背景与动机

1. **领域现状**: 心脏磁共振（CMR）成像是评估心脏形态和功能的重要工具，但长采集时间使得动态心脏运动的准确成像非常困难。从欠采样 k-t 空间数据重建无伪影的动态 MR 图像是一个高度不适定的逆问题。
2. **现有痛点**: 现有方法分两类——(a) 监督学习的运动补偿方法（如 Pan et al. 2024）效果好但依赖全采样 cine CMR 数据（需屏气采集），限制了临床实用性和泛化性；(b) INR 方法（如 ST-INR）可以无监督重建，但收敛慢、难以表示高频细节，且哈希编码的离散特征表示损害了 INR 的连续性。
3. **核心矛盾**: 无监督方法的连续先验不足以应对极端欠采样的逆问题，而运动补偿的显式建模能有效利用时空冗余但现有实现依赖监督训练和离散插值。
4. **本文目标**: 在无需训练数据的情况下，实现精确的心脏运动分解和高保真 CMR 重建。
5. **切入角度**: 将 INR 的连续表示能力与运动补偿的显式运动建模相结合——用两个 INR 网络分别连续参数化位移向量场（DVF）和共享的标准图像。
6. **核心 idea**: 用连续的 INR 函数替代离散矩阵来参数化运动补偿中的变形场和标准图像，结合粗到细哈希编码和 CNN 解码器实现稳定优化和高频细节恢复。

## 方法详解

### 整体框架

MoCo-INR 将动态 CMR 序列分解为两个连续函数：(1) 变形网络 $\mathcal{F}_\Phi$ 以时空坐标 $(p,t)$ 为输入预测 DVF $u_t(p)$；(2) 标准网络 $\mathcal{G}_\Psi$ 以变形后的空间坐标 $\tilde{p}$ 为输入预测标准图像的复数值。通过可微的正向模型将预测的 CMR 图像映射到 k 空间，与采集数据比较进行联合优化。

### 关键设计

1. **连续 MoCo 表示（Continuous DVF + Canonical INR）**:
    - 功能：将传统离散矩阵的运动补偿表示替换为连续函数
    - 核心思路：DVF 定义为 $f: (p,t) \in \mathbb{R}^3 \mapsto u_t(p) = (\Delta x, \Delta y) \in \mathbb{R}^2$，标准图像定义为 $g: \tilde{p} \in \mathbb{R}^2 \mapsto x_{cano}(\tilde{p}) = a(\tilde{p}) + jb(\tilde{p}) \in \mathbb{C}$。重建过程：$\tilde{p} = p + u_t(p)$，$\hat{x}_t(\tilde{p}) = \mathcal{G}_\Psi(\tilde{p})$
    - 设计动机：INR 对低频连续信号有学习偏置（spectral bias），天然适合表示光滑的运动场；连续表示避免了离散插值导致的高频细节丢失

2. **Coarse-to-Fine Hash Encoding（粗到细哈希编码）**:
    - 功能：稳定 DVF 估计，避免过拟合到高频伪影
    - 核心思路：哈希编码将坐标映射为多分辨率特征 $\gamma(p) = \gamma_1(p) \oplus ... \oplus \gamma_L(p)$。训练过程中先学习低频特征（$\gamma_1$，全局运动），冻结后再逐步优化高频特征（$\gamma_2, \gamma_3, ...$，精细运动细节）
    - 设计动机：全局结构对运动校正更关键，先学低频再学高频可以避免错误的高频运动估计干扰全局运动捕获

3. **CNN-based Decoder（CNN解码器）**:
    - 功能：替代传统 MLP 解码器，增强空间连续性和抗过拟合能力
    - 核心思路：用三层 CNN（64 个 3×3 卷积核，前两层带非线性激活）替换 MLP，利用 CNN 对局部结构的归纳偏置来更好地逼近连续函数 $f$ 和 $g$
    - 设计动机：MLP 的逐体素映射难以捕获图像的空间连续性；哈希编码引入的强拟合能力可能导致对欠采样数据过拟合产生高频伪影，CNN 的局部感受野天然抑制这类伪影

### 损失函数 / 训练策略

总损失函数：
$$\mathcal{L} = \underbrace{\|\hat{y}_t - y_t\|_1}_{\mathcal{L}_{DC}} + \mathcal{L}_{DVF}$$

其中数据一致性损失 $\mathcal{L}_{DC}$ 最小化预测与采集 k 空间数据的 L1 距离；DVF 正则化：
$$\mathcal{L}_{DVF} = \|u_t\|_1 + \|\nabla u_t\|_1 + \|\nabla^2 u_t\|_1$$

包含 DVF 幅值稀疏性、一阶梯度平滑性和二阶梯度平滑性三项约束，确保合理的变形估计。

## 实验关键数据

### 主实验

| 采样模式 | 加速因子 | 指标 | MoCo-INR | ST-INR(L&S) | TDDIP | 提升 |
|--------|------|------|------|----------|------|------|
| VISTA (Cartesian) | 12x | PSNR/SSIM | 42.25/0.971 | 41.35/0.972 | 38.05/0.943 | +0.90 dB |
| VISTA | 20x | PSNR/SSIM | **39.53/0.957** | 36.26/0.937 | 36.58/0.929 | **+3.27 dB** |
| GA Radial | 26x | PSNR/SSIM | **40.33/0.960** | 38.85/0.956 | 34.10/0.895 | +1.48 dB |
| GA Radial | 69x | PSNR/SSIM | **37.75/0.940** | 33.92/0.910 | 33.62/0.883 | **+3.83 dB** |

在超高加速因子（20x, 69x）下优势最为显著，所有比较均达到统计显著性 (p<0.001)。

### 消融实验

| 配置 | PSNR | SSIM | 说明 |
|------|---------|---------|------|
| w/o $\mathcal{L}_{DVF}$ | 34.42 | 0.895 | DVF 估计错误，性能大幅下降 |
| w/o Coarse2fine | 35.51 | 0.926 | DVF 估计基本合理但静态区域异常 |
| Full (MoCo-INR) | **37.75** | **0.940** | 两个组件协同最优 |
| MLP decoder vs CNN decoder | MLP更低 | 更多伪影 | CNN 更稳定，高频伪影更少 |

### 关键发现

- 运行时间优势显著：在 GA Radial 前瞻性研究中 MoCo-INR 仅需 3.4 分钟 vs ST-INR(L&S) 6.7 分钟 vs TDDIP 19.3 分钟
- 学习到的 DVF 在舒张期和收缩期均展示了与心脏生物力学一致的运动模式（心肌放松/收缩）
- 在前瞻性自由呼吸 CMR 数据上也展示了优异表现，验证了临床实用性

## 亮点与洞察

- **INR + MoCo 的首创性结合**：连续表示与显式运动建模的结合非常自然——INR 的连续性天然适合表示光滑的运动场，而运动补偿将时间维度的变化分解为全局标准图像 + 帧间变形，大幅降低了问题难度
- **粗到细策略的直觉**：先学全局运动再学局部细节，这与人类理解运动的方式一致，也符合 INR 的频率偏置特性
- **CNN 替代 MLP 的洞察**：在高度欠采样的逆问题中，过强的拟合能力反而有害，CNN 的局部归纳偏置起到了隐式正则化的效果
- **真正的无监督**：不需要任何训练数据，直接从单个切片的欠采样数据优化，摆脱了对全采样数据和屏气采集的依赖

## 局限与展望

- 目前仅支持 2D+t 重建，高分辨率 3D+t 重建是未来方向
- 运动补偿假设所有帧共享相同的标准图像，在动态对比增强（DCE）MRI 等存在对比度变化的场景下不适用
- 对于极端运动（如心律不齐）的鲁棒性有待验证
- 粗到细调度的超参数（各阶段的训练步数）需要手动设置

## 相关工作与启发

- **vs ST-INR (L&S) (Feng et al. 2025)**: ST-INR 用哈希编码+低秩稀疏约束做无监督 INR 重建，但缺乏显式运动建模，在超高加速因子下表现明显不如 MoCo-INR
- **vs 监督 MoCo 方法 (Pan et al. 2024)**: 监督方法在采样模式变化时泛化性差，且需要全采样数据训练；MoCo-INR 完全无监督，适应性更强
- **vs TDDIP (Yoo et al. 2021)**: DIP 方法无法充分捕获时间动态，心脏解剖在不同相位几乎拟合一致

## 评分

- 新颖性: ⭐⭐⭐⭐ INR+MoCo 的组合是自然但有效的创新，粗到细哈希编码和 CNN 解码器是有洞察力的设计
- 实验充分度: ⭐⭐⭐⭐⭐ 回顾性+前瞻性实验、多种采样模式、多个加速因子、完整消融分析、DVF 可视化、运行时间对比
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图表丰富，公式准确
- 价值: ⭐⭐⭐⭐⭐ 无监督+快速收敛+超高加速因子，对临床实时心脏MRI有重要实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT](unsupervised_multi-parameter_inverse_solving_for_reducing_ring_artifacts_in_3d_x.md)
- [\[AAAI 2026\] Multivariate Gaussian Representation Learning for Medical Action Evaluation](multivariate_gaussian_representation_learning_for_medical_action_evaluation.md)
- [\[AAAI 2026\] GROVER: Graph-guided Representation of Omics and Vision with Expert Regulation for Cancer Survival Prediction](grover_graph-guided_representation_of_omics_and_vision_with_expert_regulation_fo.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](../../CVPR2026/medical_imaging/uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)

</div>

<!-- RELATED:END -->
