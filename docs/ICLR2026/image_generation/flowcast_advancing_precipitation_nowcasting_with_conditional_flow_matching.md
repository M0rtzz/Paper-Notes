---
title: >-
  [论文解读] FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching
description: >-
  [ICLR 2026][图像生成][Conditional Flow Matching] 首次将条件流匹配(CFM)作为端到端概率生成模型应用于降水临近预报，在压缩潜空间中学习噪声到数据的直接映射，以更少的采样步数超越扩散模型的预测精度和概率性能。
tags:
  - ICLR 2026
  - 图像生成
  - Conditional Flow Matching
  - 降水临近预报
  - 概率预测
  - 潜空间生成
  - 时空预测
---

# FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching

**会议**: ICLR 2026  
**arXiv**: [2511.09731](https://arxiv.org/abs/2511.09731)  
**代码**: [GitHub](https://github.com/b-rbmp/FlowCast)  
**领域**: 扩散模型/气象预测  
**关键词**: Conditional Flow Matching, 降水临近预报, 概率预测, 潜空间生成, 时空预测

## 一句话总结
首次将条件流匹配(CFM)作为端到端概率生成模型应用于降水临近预报，在压缩潜空间中学习噪声到数据的直接映射，以更少的采样步数超越扩散模型的预测精度和概率性能。

## 研究背景与动机

**领域现状**：降水临近预报（nowcasting）对防洪和决策至关重要。深度学习方法已从RNN/Transformer的确定性预测发展到扩散模型的概率预测。PreDiff、LDCast等潜空间扩散模型是当前SOTA，CasCast用确定性+扩散混合方法表现最好。

**现有痛点**：确定性模型用MSE优化导致预测模糊，无法表达不确定性；扩散模型需要数百步迭代去噪，计算开销大，不满足时间敏感场景（如洪水预警）对快速集合预测的需求。

**核心矛盾**：预测精度与计算效率的矛盾——扩散模型精度高但推理慢，确定性模型快但模糊。需要一种既快又准的概率预测方法。

**本文目标**：能否用CFM替代扩散模型，在保持甚至超越预测精度的同时，大幅减少采样步数？

**切入角度**：CFM的直线ODE先验比扩散模型的弯曲概率流路径更适合时空预测——雷达反射率分布虽然多模态，但时间一致性强，线性插值提供了更稳定的先验。

**核心 idea**：CFM在潜空间中学到的直线传输路径天然契合时空数据的连续性，实现少步高质量概率预报。

## 方法详解

### 整体框架
两阶段流程：(1) 训练VAE将雷达帧压缩到低维潜空间；(2) 在潜空间中训练基于Cuboid Attention U-Net的CFM模型。输入13帧历史雷达观测（65分钟），输出12帧未来预报（60分钟），可采样 $N$ 个成员形成概率集合。

### 关键设计

1. **Frame-wise VAE**:

    - 功能：将单帧雷达图从高维像素空间压缩到低维潜表示
    - 核心思路：层次编码器-解码器，含残差块和自注意力，用L1重建+KL散度+PatchGAN对抗损失训练
    - 设计动机：降低生成模型的计算维度，与潜空间扩散模型思路一致

2. **Independent CFM (I-CFM) 训练**:

    - 功能：在潜空间中训练向量场 $v_\theta$，学习从高斯噪声到雷达潜表示的映射
    - 核心思路：概率路径 $p_t(x_t|x_0,x_1) = \mathcal{N}((1-t)x_0 + tx_1, \sigma^2 I)$，目标向量场 $u_t = x_1 - x_0$，训练损失 $\mathcal{L} = \|v_\theta(Z_t, t, Z_{\text{past}}) - u_t\|^2$。关键是 $\sigma > 0$ 提供正则化。
    - 设计动机：相比rectified flows（$\sigma \to 0$），非零 $\sigma$ "加厚"训练轨迹，对高维数据更稳定。CFM的直线ODE轨迹比扩散模型的弯曲路径更适合少步采样。

3. **FlowCast U-Net架构**:

    - 功能：基于Earthformer的Cuboid Attention层构建时空U-Net，以流时间 $t$ 为条件
    - 核心思路：编码器-解码器结构，核心构建块为Cuboid Attention（在3D立方体内做局部自注意力），时间步 $t$ 的嵌入注入每层
    - 设计动机：Cuboid Attention高效处理局部时空动态，U-Net层次结构共享全局信息

### 损失函数 / 训练策略
- VAE: L1重建 + KL散度(权重1e-4) + PatchGAN对抗损失
- CFM: 均方误差回归向量场，AdamW lr=1e-4
- 采样: Euler求解器，可变步数（5/10/20/50/100）

## 实验关键数据

### 主实验
SEVIR数据集（美国雷达），8成员集合预测：

| 模型 | 类型 | CSI-M↑ | FSS-M↑ | CRPS↓ | NFE |
|------|------|--------|--------|-------|-----|
| Earthformer | 确定性 | 基线 | 基线 | 较高 | 1 |
| PreDiff | 扩散 | 次优 | 次优 | 次优 | 250 |
| CasCast | 混合 | 优 | 优 | 优 | 250 |
| FlowCast(50步) | CFM | **最优** | **最优** | **最优** | 50 |
| FlowCast(20步) | CFM | 接近最优 | 接近最优 | 接近最优 | 20 |

### 消融实验：CFM vs 扩散目标（相同架构）
| 配置 | CSI-M↑ | CRPS↓ | 说明 |
|------|--------|-------|------|
| CFM 50步 | 最优 | 最优 | 完整方案 |
| 扩散 50步 | 下降 | 下降 | 相同架构换扩散目标 |
| CFM 20步 | 仍优 | 仍优 | 少步仍保持高性能 |
| 扩散 20步 | 显著下降 | 显著下降 | 步数减少性能急剧衰退 |

### 关键发现
- FlowCast用50步就超越了需要250步的PreDiff和CasCast，计算效率提升5倍
- 关键消融证明：在完全相同架构下，CFM目标比扩散目标更准确且对步数更鲁棒
- 在ARSO本地数据集上同样验证了结论，说明方法不依赖特定数据集
- CFM在少步(20步)时仍保持高性能，扩散模型则急剧退化

## 亮点与洞察
- **CFM vs 扩散的直接对比**：在相同架构下消融CFM和扩散目标，是该领域首个严格公平对比，证明CFM在时空预测中的优势不仅来自架构而是来自训练目标本身。
- **直线轨迹的归纳偏置**：对气象时空数据的独到洞察——雷达反射率虽多模态但时间连续性强，CFM的线性插值路径比扩散的弯曲路径更匹配这种特性。
- **端到端概率模型**：与CasCast需要确定性基底+扩散细化不同，FlowCast直接做噪声到数据的完整概率建模，更简洁。

## 局限与展望
- 仅验证了5分钟/1km分辨率，未探索更高分辨率或更长预报时效
- VAE单独训练后冻结，联合训练可能提升上界
- 集合成员数固定为8，未分析最优集合大小
- 仅对比了Euler求解器，更高阶求解器（如RK45）可能进一步减少步数

## 相关工作与启发
- **vs PreDiff/LDCast**: 都是潜空间生成模型，但FlowCast用CFM替代扩散，步数从250降到50
- **vs CasCast**: CasCast需要两个模型（确定性+扩散），FlowCast单个模型端到端更简洁
- **vs Feng等人的rectified flow方法**: 他们仅用RF做确定性预测的细化，FlowCast是完整的概率生成模型

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将CFM作为端到端概率模型用于降水预报，消融设计严谨
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、多个指标、CFM vs 扩散消融、步数敏感性分析
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述清晰，实验设计科学，代码开源
- 价值: ⭐⭐⭐⭐ 对气象预测领域有直接影响，证明CFM是扩散的强替代方案

<!-- RELATED:START -->

## 相关论文

- [FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching](flowcast_trajectory_forecasting_for_scalable_zero-cost_speculative_flow_matching.md)
- [Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [HazeMatching: Dehazing Light Microscopy Images with Guided Conditional Flow Matching](../../CVPR2026/image_generation/hazematching_dehazing_light_microscopy_images_with_guided_conditional_flow_match.md)
- [Improving Posterior Inference of Galaxy Properties with Image-Based Conditional Flow Matching](../../NeurIPS2025/image_generation/improving_posterior_inference_of_galaxy_properties_with_image-based_conditional_.md)
- [Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)

<!-- RELATED:END -->
