---
title: >-
  [论文解读] TAlignDiff: Automatic Tooth Alignment assisted by Diffusion-based Transformation Learning
description: >-
  [AAAI 2026][医学图像][牙齿排列] 提出TAlignDiff框架，将基于点云的几何约束回归网络（PRN）与扩散模型辅助的变换矩阵去噪模块（DTMD）统一为一个联合训练框架，通过双向反馈机制在小样本临床数据上实现了优于现有方法的自动牙齿排列效果。
tags:
  - "AAAI 2026"
  - "医学图像"
  - "牙齿排列"
  - "正畸治疗"
  - "扩散模型"
  - "变换矩阵"
  - "点云回归"
---

# TAlignDiff: Automatic Tooth Alignment assisted by Diffusion-based Transformation Learning

**会议**: AAAI 2026  
**arXiv**: [2508.04565](https://arxiv.org/abs/2508.04565)  
**代码**: 无（论文中提到接收后公开）  
**领域**: 医学图像 / 3D视觉  
**关键词**: 牙齿排列, 正畸治疗, 扩散模型, 变换矩阵, 点云回归

## 一句话总结

提出TAlignDiff框架，将基于点云的几何约束回归网络（PRN）与扩散模型辅助的变换矩阵去噪模块（DTMD）统一为一个联合训练框架，通过双向反馈机制在小样本临床数据上实现了优于现有方法的自动牙齿排列效果。

## 研究背景与动机

**领域现状**：自动牙齿排列是正畸治疗的核心任务——需要预测每颗牙齿从错颌状态到正常咬合状态的最优运动（旋转+平移）。当前深度学习方法主要将3D牙齿模型编码为点云，通过回归网络预测6自由度变换矩阵（$4 \times 4$的旋转+平移矩阵），使用逐点几何损失（如重建误差）进行监督。代表方法包括TANet（图卷积特征传播）、TAPoseNet（多尺度GCN）等。

**现有痛点**：现有方法仅使用点对点的几何约束来监督变换矩阵预测，但变换矩阵本身具有内在的分布特征——有效旋转/平移的合理范围、相邻牙齿之间的相关性等。确定性的几何约束无法捕获这些分布特征，导致预测的变换矩阵可能落在不合理的范围内。

**核心矛盾**：已有的TADPM方法虽然首次引入扩散模型来学习变换矩阵分布，但它直接以高维点云和mesh特征作为扩散条件，计算复杂度高且对大数据集依赖严重，在小样本临床场景下表现不佳。

**本文目标** 如何在小样本临床数据上同时利用显式几何约束和隐式分布建模？具体子问题：（1）如何设计轻量级的扩散模块来学习变换矩阵分布？（2）如何让几何回归和扩散建模之间形成双向反馈？

**切入角度**：不对原始高维几何特征做扩散，而是只对变换矩阵本身做扩散——将扩散的输入维度从高维点云特征降到 $32 \times 16$ 的变换矩阵空间，大幅降低扩散模型的学习难度和数据需求。

**核心 idea**：用轻量级扩散模型对变换矩阵空间进行噪声估计，通过对比预测矩阵和真实矩阵的噪声差异来隐式约束回归网络输出符合真实分布。

## 方法详解

### 整体框架

TAlignDiff包含两个模块：（1）PRN（点云回归网络）：以错颌牙齿点云为输入，提取全局和局部特征，回归32颗牙齿的变换矩阵；（2）DTMD（扩散变换矩阵去噪模块）：对变换矩阵进行前向加噪和反向去噪，学习临床数据中变换矩阵的潜在分布。两个模块通过联合训练策略和对比去噪损失实现双向反馈。推理时仅使用PRN，DTMD不参与，保证推理效率。

### 关键设计

1. **点云回归网络（PRN）**:

    - 功能：从3D牙齿点云预测32颗牙齿的变换矩阵，每个矩阵包含 $3 \times 3$ 旋转矩阵和 $3 \times 1$ 位移向量
    - 核心思路：使用两个PointNet编码器 $\epsilon_g$ 和 $\epsilon_l$ 分别提取全局特征（整个牙列）和局部特征（单颗牙齿），拼接后通过三层MLP解码器（通道 $[512, 256, 16]$）回归变换矩阵：$T^* = \phi(\epsilon_g(P_{in}) \oplus \epsilon_l(P_{in}))$。PointNet编码器由三层1D卷积组成（通道 $[64, 128, 1024]$）
    - 设计动机：PointNet直接处理原始点云，用对称函数高效提取层次特征。全局-局部双编码器设计让模型既能感知整体牙列的排列关系，又能捕获单颗牙齿的几何细节

2. **扩散变换矩阵去噪模块（DTMD）**:

    - 功能：作为训练辅助模块，学习临床数据中变换矩阵的潜在分布，通过噪声估计间接约束PRN的输出
    - 核心思路：将目标变换矩阵 $M_0$（reshape后的 $M_{gt}$）通过前向扩散链逐步加噪：$q(M_t | M_0) = \mathcal{N}(M_t | \sqrt{\gamma_t} M_0, (1-\gamma_t)I)$。扩散模型 $\epsilon_{\theta_d}$ 学习预测噪声。关键创新是对比去噪损失：$L_{denoi} = \mathbb{E}[\|\epsilon_{\theta_d}(M_{gt}^t, t) - \epsilon_{\theta_d}(M_{pre}^t, t)\|_1]$，即对真实矩阵和预测矩阵在同一时间步加入相同噪声后，比较它们的噪声估计——如果预测矩阵越接近真实分布，两者的噪声估计差异越小
    - 设计动机：直接以变换矩阵（而非高维点云特征）作为扩散输入，将输入维度从数千维降到 $32 \times 16$，显著降低数据需求。DTMD只参与训练不参与推理，不影响推理效率

3. **联合训练策略（Joint Training Strategy）**:

    - 功能：通过分阶段训练实现PRN和DTMD的协同优化
    - 核心思路：前200个epoch，PRN和DTMD联合训练（四个损失一起优化）；后200个epoch，固定DTMD参数，仅训练PRN，利用预训练好的DTMD（通过 $L_{denoi}$）继续优化PRN的输出。总损失为 $L_{total} = L_{rec} + \lambda_1 L_{center} + \lambda_2 L_{denoi} + \lambda_3 L_{diffusion}$
    - 设计动机：第一阶段让DTMD学会变换矩阵的分布特征，第二阶段冻结DTMD让PRN专注利用分布先验优化自身输出。这种"先学分布再用分布"的策略避免了两个模块训练不稳定的问题

### 损失函数 / 训练策略

四个损失函数：（1）点云重建损失 $L_{rec} = \frac{1}{N}\sum\|T^* \cdot P_{in} - T \cdot P_{in}\|_1$；（2）牙齿质心偏移损失 $L_{center} = \frac{1}{M}\sum\|C_{predict} - C_{target}\|_1$，约束牙齿集体位移方向；（3）扩散训练损失 $L_{diffusion}$，标准的噪声预测MSE；（4）对比去噪损失 $L_{denoi}$，PRN输出与真实矩阵的噪声估计差异。最优权重：$\lambda_1=0.1, \lambda_2=0.01, \lambda_3=0.1$。数据增强包括多齿随机旋转（5~10颗）和单齿随机平移。

## 实验关键数据

### 主实验

| 方法 | 数据集 | TRE (mm) ↓ | AAE (mm) ↓ |
|------|--------|-----------|-----------|
| PointNet++ | Test | 0.791±0.927 | 0.717±0.833 |
| PointMLP | Test | 0.819±0.935 | 0.743±0.844 |
| TADPM | Test | 0.890±0.963 | 0.821±0.883 |
| PSTN | Test | 0.779±0.917 | 0.705±0.821 |
| **TAlignDiff (Ours)** | Test | **0.725±0.834** | **0.646±0.734** |

### 消融实验

| 配置 ($\lambda_1, \lambda_2, \lambda_3$) | TRE (Test) | AAE (Test) | 说明 |
|------|---------|---------|------|
| (0, 0, 0) 基线 | 0.784±0.927 | 0.711±0.831 | 仅重建损失 |
| (0.1, 0, 0) | 0.748±0.873 | 0.670±0.778 | 加质心损失 |
| (0.1, 0.01, 0.1) 最佳 | **0.725±0.834** | **0.646±0.734** | 完整模型 |
| (0.2, 0.01, 0.1) | 0.766±0.864 | 0.692±0.766 | 质心权重过大 |
| (0.1, 0.05, 0.1) | 0.739±0.850 | 0.663±0.755 | 去噪权重过大 |

### 关键发现

- DTMD模块的加入使TRE从0.784降到0.725（-7.5%），AAE从0.711降到0.646（-9.1%），验证了扩散分布建模的有效性
- 联合训练策略优于分步训练（variant），后者的DTMD和PRN分别独立训练缺乏双向反馈
- TADPM在该小样本数据集上表现最差（TRE 0.890），因为其直接以高维特征驱动扩散，数据需求远大于TAlignDiff
- 3D散点图可视化显示TAlignDiff预测的变换矩阵更加聚类，说明模型稳定性提升
- 深覆颌畸形（deep overbite）等典型困难病例中，TAlignDiff的mesh重建结果最接近目标

## 亮点与洞察

- **将扩散的输入空间从高维几何特征降到变换矩阵空间**是最实用的设计决策。这让扩散模型只需学习 $32 \times 16$ 维空间的分布，而非数千维的点云特征分布，在124例患者的小数据集上即可训练
- **对比去噪损失（contrastive denoising loss）**的设计很巧妙：不是让扩散模型直接修正预测，而是通过比较预测矩阵和真实矩阵在同一噪声水平下的噪声估计差异，间接传递分布信息给PRN。这种"噪声空间的对比学习"思路可推广到其他回归任务
- **推理时不使用DTMD**——扩散模型只在训练中提供正则化/先验约束，推理时零额外开销，非常实用

## 局限与展望

- 数据集极小（124例患者，74:20:30划分），模型的泛化能力未经大规模验证
- 仅使用PointNet作为特征提取器，更强的点云编码器（如PointNet++、Point Transformer）可能进一步提升
- 未建模正畸治疗的时序阶段性——实际正畸是分步进行的（如隐适美每期的矫治器），当前方法直接预测最终状态
- 缺乏与临床专家判断的对比评估，TRE/AAE等定量指标未必完全反映临床可接受性

## 相关工作与启发

- **vs TADPM (Lei et al. 2024)**：TADPM直接以点云和mesh的高维特征条件化扩散过程，在小样本数据上表现最差（TRE 0.890 vs 0.725）。TAlignDiff通过降维到变换矩阵空间大幅减少数据依赖
- **vs TANet/TAPoseNet**：这些方法用GCN/多尺度图网络捕获牙齿关系，但仅使用几何约束。TAlignDiff额外引入分布约束，互补性强
- **vs PSTN (Li et al. 2020)**：PSTN用空间变换网络直接预测变换，性能接近TAlignDiff（TRE 0.779 vs 0.725），但缺乏分布先验的正则化效果

## 评分

- 新颖性: ⭐⭐⭐⭐ 将扩散模型用于变换矩阵分布学习的思路新颖，对比去噪损失设计有创意
- 实验充分度: ⭐⭐⭐ 数据集太小（124例），缺乏大规模验证和临床评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细，但可视化结果可以更丰富
- 价值: ⭐⭐⭐⭐ 对正畸AI有直接临床价值，小样本扩散辅助回归的思路可迁移到其他医学场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LaVCa: LLM-assisted Visual Cortex Captioning](../../ICLR2026/medical_imaging/lavca_llm-assisted_visual_cortex_captioning.md)
- [\[AAAI 2026\] WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images](wdt-md_wavelet_diffusion_transformers_for_microaneurysm_detection_in_fundus_imag.md)
- [\[AAAI 2026\] SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization](spa_achieving_consensus_in_llm_alignment_via_self-priority_optimization.md)
- [\[AAAI 2026\] Divide, Conquer and Unite: Hierarchical Style-Recalibrated Prototype Alignment for Federated Medical Segmentation](divide_conquer_and_unite_hierarchical_style-recalibrated_prototype_alignment_for.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](../../CVPR2026/medical_imaging/interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)

</div>

<!-- RELATED:END -->
