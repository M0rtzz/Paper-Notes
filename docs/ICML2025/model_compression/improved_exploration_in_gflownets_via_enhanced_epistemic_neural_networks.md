---
title: >-
  [论文解读] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks
description: >-
  [ICML 2025][模型压缩][GFlowNets] 将 Epistemic Neural Networks (ENN/epinet) 集成到 GFlowNets 中实现不确定性驱动的探索，提出 ENN-GFN-Enhanced 算法，在 HyperGrid 和序列生成任务上显著改善模式发现效率和分布学习质量。
tags:
  - ICML 2025
  - 模型压缩
  - GFlowNets
  - 认知不确定性
  - Epistemic Neural Networks
  - Thompson采样
  - 探索
---

# Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks

**会议**: ICML 2025  
**arXiv**: [2506.16313](https://arxiv.org/abs/2506.16313)  
**代码**: 无  
**领域**: Generative Models / GFlowNets  
**关键词**: GFlowNets, 认知不确定性, Epistemic Neural Networks, Thompson采样, 探索

## 一句话总结
将 Epistemic Neural Networks (ENN/epinet) 集成到 GFlowNets 中实现不确定性驱动的探索，提出 ENN-GFN-Enhanced 算法，在 HyperGrid 和序列生成任务上显著改善模式发现效率和分布学习质量。

## 研究背景与动机
**领域现状**: GFlowNets 是一类生成式模型，通过序列化构建对象来采样与奖励成正比的样本，在分子设计等科学发现中有重要应用。

**现有痛点**: (a) GFlowNets 训练中容易发生模式坍塌 (mode collapse)，被早期发现的模式吸引；(b) 探索策略 (on-policy, ε-noisy) 不够高效；(c) 现有 Thompson Sampling (TS-GFN) 用 ensemble 近似后验，计算开销大且联合预测质量有限。

**核心矛盾**: GFlowNets 的性能高度依赖采样轨迹的质量，但有效探索需要感知认知不确定性 (epistemic uncertainty)——即"知道自己不知道什么"。

**本文切入**: 用 ENN (epinet) 替代 ensemble 来获得更高效的联合预测和不确定性量化。

**核心 idea**: 在 GFlowNet 的策略网络上附加轻量级 epinet 模块，通过 epistemic index 采样实现隐式 Thompson Sampling。

## 方法详解

### 整体框架
输入：GFlowNet + 奖励函数 → 在策略网络后附加 epinet → 采样 epistemic index $z \sim P_Z$ → 生成不确定性感知的策略 → 采样轨迹 → 用 TB loss 更新 → 迭代改善分布学习。

### 关键设计

1. **ENN 与 Epinet**:

    - ENN 输出额外依赖 epistemic index $z$：$f_\theta(x,z) = \mu_\zeta(x) + \sigma_\eta(\text{sg}[\phi_\zeta(x)], z)$
    - $\mu_\zeta(x)$：base network 输出
    - $\sigma_\eta$：learnable epinet（轻量 MLP）+ 固定 prior function $\sigma_P$
    - sg 表示 stop-gradient，防止 epinet 影响 base network 的特征学习
    - 联合预测：$\hat{P}^{\text{ENN}}(y_{1:\tau}) = \int_z P_Z(dz) \prod_t \text{softmax}(f_\theta(x_t, z))_{y_t}$
    - 设计动机：ENN 以很小的计算开销实现校准的认知不确定性估计

2. **ENN-GFN (基础版)**:

    - 将 epinet 附加到 GFlowNet 的前向策略网络
    - 每个轨迹采样一组 $z$，用加权和方式组合 prior network 的输出
    - 设计动机：直接将 ENN 框架应用于 GFlowNet

3. **ENN-GFN-Enhanced (增强版)**:

    - 关键区别：不用加权和，而是随机选取一个 prior ensemble member
    - 类似 TS-GFN 的维护近似后验策略的方式
    - 但不确定性估计来自 epinet 而非 ensemble
    - 设计动机：结合 Thompson Sampling 的探索优势和 ENN 的高效不确定性估计

### 损失函数 / 训练策略
- Trajectory Balance (TB) Loss: $\mathcal{L}_{\text{TB}}(\tau;\theta) = (\log \frac{Z_\theta \prod P_F(s_{t+1}|s_t)}{R(s_n) \prod P_B(s_t|s_{t+1})})^2$
- 同时更新 base network 和 epinet 参数

## 实验关键数据

### 主实验 (2D HyperGrid, 8×8, $R_0=10^{-4}$)
| 算法 | $L_1$ 距离 (↓) | 4个模式发现 | 说明 |
|------|---------------|------------|------|
| Default-GFN | 较高 | 部分 | 仅发现起始角附近模式 |
| TS-GFN | 较高 | 部分 | ensemble 带来改善但不足 |
| ENN-GFN | **低** | ✓ 全部 | epinet 提供更好不确定性 |
| ENN-GFN-Enhanced | **最低** | ✓ 全部 | 结合 TS 策略效果最佳 |

### 消融/对比实验
| 配置 | 4D Grid ($R_0=10^{-4}, H=8$) | 说明 |
|------|---------------------------|------|
| ENN-GFN-Enhanced | 最低 $L_1$ | 所有模式快速发现 |
| ENN-GFN | 接近最优 | 略逊于 Enhanced |
| TS-GFN | 中等 | ensemble 有效但效率低 |
| Default-GFN | 最高 | 探索不足 |

| 环境 (2D, $R_0=10^{-5}$) | DB-GFN | ENN-GFN | ENN-GFN-Enhanced |
|--------------------------|--------|---------|------------------|
| 64×64 ($L_1 \times 10^{-5}$) | 基线 | 改善 | **最优** |
| 128×128 ($L_1 \times 10^{-5}$) | 基线 | 改善 | **最优** |

### Valid Bit Sequences (Transformer)
| 序列长度 | With Epinet | Without Epinet |
|----------|-------------|----------------|
| 2N=12 | **260** | 207 |
| 2N=14 | **216** | 189 |
| 2N=16 | **135** | 120 |

### 关键发现
- ENN-GFN-Enhanced 在所有环境中一致表现最优
- 在更大/更稀疏的环境中优势更加明显
- epinet 为 Transformer 架构也带来了显著多样性提升
- ENN-GFN 在小环境表现好但在 16×16 grid 上可能退化

## 亮点与洞察
- **轻量级不确定性**：epinet 仅在最后几层添加小 MLP，计算开销远小于 ensemble
- **联合预测的重要性**：不确定性驱动的探索需要联合预测而非边际预测
- **Enhanced 版本的妙处**：随机选择 ensemble member 而非加权和，更好地模拟 Thompson Sampling

## 局限与展望
- 仅在 toy 环境 (HyperGrid, Bit Sequences) 验证，缺乏分子设计等实际应用
- ENN-GFN 在较大环境中表现退化的原因需进一步分析
- 与 GAFN、RND 等其他探索增强方法的比较不够全面

## 相关工作与启发
- Thompson Sampling for GFlowNets (Rector-Brooks et al. 2023) 是直接前驱
- ENN (Osband et al. 2023) 提供了核心技术组件
- Random Network Distillation 是另一种探索增强方法
- 启发：好的不确定性估计是高效探索的基础

## 评分
- 新颖性: ⭐⭐⭐⭐ ENN+GFlowNet 的结合和 Enhanced 变体是新贡献
- 实验充分度: ⭐⭐⭐ 环境偏简单，缺乏实际应用场景
- 写作质量: ⭐⭐⭐⭐ 背景介绍充分，方法清晰
- 价值: ⭐⭐⭐⭐ 为 GFlowNet 探索问题提供了轻量有效的方案

<!-- RELATED:START -->

## 相关论文

- [FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression](fgfp_a_fractional_gaussian_filter_and_pruning_for_deep_neural_networks_compressi.md)
- [An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](an_efficient_matrix_multiplication_algorithm_for_accelerating_inference_in_binar.md)
- [Adaptive Width Neural Networks](../../ICLR2026/model_compression/adaptive_width_neural_networks.md)
- [A Recovery Guarantee for Sparse Neural Networks](../../ICLR2026/model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)
- [QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks](../../NeurIPS2025/model_compression/quadenhancer_leveraging_quadratic_transformations_to_enhance_deep_neural_network.md)

<!-- RELATED:END -->
