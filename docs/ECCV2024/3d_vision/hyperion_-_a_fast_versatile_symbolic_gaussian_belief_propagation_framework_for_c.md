---
title: >-
  [论文解读] Hyperion: A Fast, Versatile Symbolic Gaussian Belief Propagation Framework for Continuous-Time SLAM
description: >-
  [ECCV 2024][3D视觉] 提出首个连续时间高斯置信传播SLAM框架，基于SymForce自动生成超高效B/Z样条实现（2.43-110x加速），支持分布式多智能体推理
tags:
  - ECCV 2024
  - SLAM
  - 高斯置信传播
  - 连续时间
  - B样条
---

# Hyperion: A Fast, Versatile Symbolic Gaussian Belief Propagation Framework for Continuous-Time SLAM

**会议**: ECCV 2024  
**arXiv**: [2407.07074](https://arxiv.org/abs/2407.07074)  
**代码**: https://github.com/VIS4ROB-lab/hyperion  
**领域**: 3D视觉 / 机器人  
**关键词**: 连续时间SLAM, 高斯置信传播, B样条, 分布式优化, 符号计算

## 一句话总结

本文提出Hyperion，一个基于SymForce符号计算框架自动生成超高效B/Z样条实现的连续时间高斯置信传播（GBP）SLAM框架，在运动跟踪和定位场景中达到与传统NLLS求解器（Ceres）相当的精度，同时天然支持分布式多智能体推理。

## 研究背景与动机

**领域现状**：同时定位与建图（SLAM）是机器人感知的核心问题。传统离散时间SLAM估计离散位姿，需要严格同步传感器时间戳。连续时间SLAM（CTSLAM）使用连续时间运动参数化（通常是B样条），能在任意时刻插值位姿，天然支持滚动快门相机、事件相机和IMU等异步多传感器融合。

**现有痛点**：(1) CTSLAM方法计算复杂度高——连续时间参数化涉及复杂的样条求值和解析Jacobian计算，现有手工实现效率不足；(2) 几乎所有SLAM系统（离散和连续时间）都采用集中式非线性最小二乘（NLLS）优化，严格限制在单智能体场景；(3) 手工推导和实现样条相关的代价函数及导数容易出错且耗时。

**核心矛盾**：CTSLAM的优势（异步融合、连续运动估计）被其高计算复杂度和集中式架构所制约，阻碍了在实际多智能体场景中的部署。需要：(1) 更快的连续时间运动参数化实现；(2) 一种天然分布式的优化范式来替代集中式NLLS。

**本文目标** (1) 如何大幅加速B/Z样条的计算？(2) 如何构建一个分布式、异步的连续时间SLAM优化框架？

**切入角度**：利用SymForce符号计算框架自动生成超高效的C++样条代码（消除手工推导），同时引入高斯置信传播（GBP）作为分布式概率推理范式——GBP通过因子图上的消息传递实现分布式异步推理，天然适合多智能体SLAM。

**核心 idea**：结合SymForce自动代码生成实现最快的B/Z样条运算，与高斯置信传播的分布式推理范式结合，构建首个连续时间GBP SLAM框架。

## 方法详解

### 整体框架

Hyperion的核心是一个连续时间因子图的GBP求解器。因子图中，变量节点为B/Z样条的控制点（基），因子节点为传感器测量约束（绝对位姿或视觉重投影）。GBP通过在节点和因子之间交替传播消息来迭代求解最优运动估计。框架支持批处理和滑窗优化，支持多线程并行和dropout更新策略。

### 关键设计

1. **基于SymForce的超高效样条实现**:
    - 功能：自动生成解析的、高度优化的B/Z样条评估和Jacobian计算代码
    - 核心思路：将Sommer等人的递归样条公式与SymForce符号代码生成框架结合。在符号层面定义样条的位姿、速度、加速度评估，SymForce自动进行表达式简化、共同子表达式消除，生成最优的C++实现。同时自动生成解析Jacobian而非依赖自动微分
    - 设计动机：手工实现的B样条（Sommer et al.）在Jacobian计算上依赖自动微分，效率低下。SymForce自动生成的实现在位姿/速度/加速度评估上加速2.43x-110.31x

2. **连续时间GBP求解器**:
    - 功能：在因子图上通过消息传递实现分布式概率推理
    - 核心思路：GBP包含节点更新和因子更新两步交替进行。节点通过汇聚所有相邻因子发来的因子-节点消息来更新其置信度 $B(n_j) = \mathcal{N}^{-1}(\eta_{n_j}, \Lambda_{n_j})$。因子通过收集相邻节点的节点-因子消息，结合残差的线性化信息，计算并发送因子-节点消息。对于Lie群（如旋转），采用混合高斯表示（MGR）处理切空间的坐标变换
    - 设计动机：GBP天然分布式和异步——每个节点和因子可以独立更新，无需全局求解器。这使得多智能体SLAM成为可能，每个智能体仅需处理本地因子图并与邻居交换消息

3. **鲁棒性和灵活性扩展**:
    - 功能：处理异常值并支持灵活的更新策略
    - 核心思路：引入鲁棒损失函数 $\rho$ 修改残差和Jacobian；实现同步更新（所有节点/因子每轮都更新，模拟NLLS）和dropout更新（节点/因子按概率 $d_n$, $d_f$ 随机更新）；引入步长参数 $\alpha$ 控制收敛速度；将非优化参数标记为常量节点减少边际化计算
    - 设计动机：loopy图中GBP收敛性需要dropout策略保障稳定性；常量节点优化减少了 $\mathcal{O}(n^3)$ 的Cholesky分解计算量

### 损失函数 / 训练策略

GBP的优化目标等价于最小化加权残差平方和 $\Theta^* = \arg\min_\Theta \sum_s \sum_t \frac{1}{2}\|\bar{r}(t, \theta_s)\|^2$，与传统NLLS完全等价。步长 $\alpha_{n_j} = \alpha_{f_i} = 0.7$，样条基间隔0.1秒。支持Huber等鲁棒核函数。实验使用同步更新策略进行公平对比。

## 实验关键数据

### 主实验

**绝对位姿设置下的误差对比（RMSE，不同扰动水平）：**

| 求解器 | ±0.01 (R/t) | ±0.05 | ±0.10 | ±0.50 | ±1.00 |
|--------|-------------|-------|-------|-------|-------|
| Hyperion R[rad] | 5.2e-6 | 5.2e-6 | 5.2e-6 | 5.9e-6 | 5.3e-6 |
| Ceres R[rad] | 5.2e-6 | 5.2e-6 | 5.2e-6 | 5.2e-6 | 5.2e-6 |
| Hyperion t[m] | 5.8e-6 | 5.8e-6 | 5.9e-6 | 6.0e-6 | 1.3e-5 |
| Ceres t[m] | 5.9e-6 | 5.9e-6 | 5.9e-6 | 5.9e-6 | 5.9e-6 |

**B样条实现速度对比（vs Sommer et al.）：**

| 操作 | 阶数 | 含Jacobian | 加速比 |
|------|------|-----------|--------|
| SE(3) B-Spline求值 | 6 | 否 | **2.43x** |
| SE(3) B-Spline求值 | 6 | 是 | **22.71x** |
| SU(2) Z-Spline求值 | 6 | 否 | **4.47x** |
| SU(2) Z-Spline求值 | 6 | 是 | **110.31x** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Dropout 0% | ~8次迭代收敛 | 基线（同步更新） |
| Dropout 10% | ~10次迭代 | 轻微减速 |
| Dropout 30% | ~14次迭代 | 中等减速 |
| Dropout 60% | ~20次迭代 | 仍收敛到同一解 |
| B-Spline vs Z-Spline | Z-Spline更快收敛 | Z-Spline的插值性质提供更好的初始约束 |
| GBP vs Ceres迭代次数 | GBP多2-4次 | 分布式推理的固有代价 |

### 关键发现

- Hyperion和Ceres在所有扰动和噪声水平下收敛到相同的解——GBP的正确性得到验证
- SymForce自动生成的代码在含Jacobian场景下提速最显著（22-110x），因为消除了自动微分
- Z-Spline通常比B-Spline收敛更快，因为插值样条的基点直接位于运动轨迹上
- Dropout更新虽然增加迭代次数，但保持最终解的一致性，适合不可靠通信场景
- 在定位设置下（含landmark重投影），视觉因素引入更多环路使GBP对dropout更不敏感
- 单核性能GBP约为Ceres的6-7.5倍慢，但GBP天然支持并行和分布式，可弥补差距

## 亮点与洞察

- **SymForce自动化的工程价值巨大**：消除了CTSLAM中最繁琐的手工推导和实现环节，同时获得超越手工优化的性能
- **GBP为多智能体SLAM开辟新范式**：天然分布式、异步、支持不完美通信
- **与Ceres解等价**：验证了GBP在非平凡SLAM问题上的正确性和实用性
- **开源贡献**：提供了可与Ceres和SymForce无缝集成的因子库

## 局限与展望

- 单核性能仍逊于高度优化的Ceres（6-7.5x），需更多工程优化
- 在loopy图（如完整SLAM系统）中数值不稳定性可能导致收敛困难
- 目前仅在模拟和简单实际场景中验证，未在完整的多智能体SLAM系统中部署
- 协方差估计增加了额外计算开销（Ceres不显式估计协方差）
- 可利用协方差信息实现自适应节点更新（仅更新不确定性高的节点）进一步加速

## 相关工作与启发

- **Ceres Solver**：传统集中式NLLS优化器的标杆，Hyperion的对标基准
- **Murai et al. (Robot Web)**：GBP在分布式SLAM中的先驱工作，提出MGR处理Lie群
- **SymForce**：符号代码生成框架，本文关键基础设施
- **Sommer et al.**：手工优化的B样条实现，本文的对比基准
- 启发：符号计算+自动代码生成思路可推广到其他需要高效解析导数的机器人问题

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将GBP与连续时间SLAM结合，SymForce加速是有价值的工程贡献
- 实验充分度: ⭐⭐⭐ 以模拟实验为主，真实场景验证较少
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨完整，但密度较高
- 价值: ⭐⭐⭐⭐ 为分布式多智能体CTSLAM奠定理论和工程基础

<!-- RELATED:START -->

## 相关论文

- [VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)
- [SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [I²-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM](i2-slam_inverting_imaging_process_for_robust_photorealistic_dense_slam.md)
- [CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field](cg-slam_efficient_dense_rgb-d_slam_in_a_consistent_uncertainty-aware_3d_gaussian.md)
- [GaussReg: Fast 3D Registration with Gaussian Splatting](gaussreg_fast_3d_registration_with_gaussian_splatting.md)

<!-- RELATED:END -->
