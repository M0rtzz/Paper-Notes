---
title: >-
  [论文解读] 2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching
description: >-
  [CVPR 2026][图像生成][扩散模型] 提出2ndMatch微调框架，通过对齐剪枝模型与原始模型的二阶Jacobian矩阵 $J^\top J$（灵感来自有限时间Lyapunov指数），匹配两者对输入扰动的时间敏感性，从而显著缩小剪枝扩散模型与原始模型的生成质量差距。
tags:
  - CVPR 2026
  - 图像生成
  - 扩散模型
  - 模型剪枝
  - Jacobian匹配
  - 有限时间Lyapunov指数
  - 知识蒸馏
---

# 2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching

**会议**: CVPR 2026  
**arXiv**: [2506.05398](https://arxiv.org/abs/2506.05398)  
**代码**: 无  
**领域**: 扩散模型 / 模型压缩  
**关键词**: 扩散模型, 模型剪枝, Jacobian匹配, 有限时间Lyapunov指数, 知识蒸馏

## 一句话总结
提出2ndMatch微调框架，通过对齐剪枝模型与原始模型的二阶Jacobian矩阵 $J^\top J$（灵感来自有限时间Lyapunov指数），匹配两者对输入扰动的时间敏感性，从而显著缩小剪枝扩散模型与原始模型的生成质量差距。

## 研究背景与动机

**领域现状**：扩散模型在图像生成中效果出色，但推理时需要数百次去噪步骤，计算开销巨大。模型剪枝是减少每步计算量的有效策略。

**现有痛点**：剪枝后的微调通常复用原始去噪分数匹配（DSM）目标，对容量减小的剪枝模型来说不足够。现有知识蒸馏对齐输出或中间特征，但忽视了模型的**敏感性**——即分数函数对输入扰动的响应。一阶Jacobian匹配对扩散模型基本等价于KD（因输入本身就有噪声扰动），且无法捕捉跨时间步的扰动传播。

**核心矛盾**：剪枝模型容量减小→对扰动的敏感性与原始模型偏离→去噪轨迹漂移→生成质量下降。需要一种方法约束剪枝模型保持与原始模型相同的时间动力学行为。

**切入角度**：将扩散模型视为离散时间动力系统，从有限时间Lyapunov指数（FTLE）理论出发，FTLE量化了微小扰动在有限时间内的放大/收缩率。

**核心idea**：对齐剪枝模型和原始模型的 $J^\top J$（二阶Jacobian度量），通过随机投影 $v^\top J^\top J v$ 高效估计方向性膨胀率，实现可扩展的二阶Jacobian匹配。

## 方法详解

### 整体框架
混合微调目标：$\mathcal{L}_{total} = \lambda_{NP}\mathcal{L}_{NP} + \lambda_{KD}\mathcal{L}_{KD} + \lambda_{Jac}\mathcal{L}_{2nd\text{-}Jac}$，三个互补组件分别处理噪声预测、输出对齐和时间敏感性匹配。

### 关键设计

1. **噪声预测（Noise Prediction）**:

    - 功能：标准DDPM目标，预测前向过程添加的噪声
    - 核心思路：$\mathcal{L}_{NP} = \mathbb{E}_{\tilde{x},t,\epsilon}[\|s(\tilde{x},t;\theta) - \epsilon\|_2^2]$
    - 设计动机：基础监督信号，但对容量减小的剪枝模型来说单独使用不够

2. **知识蒸馏（Knowledge Distillation）**:

    - 功能：对齐剪枝模型与原始模型的输出
    - 核心思路：$\mathcal{L}_{KD} = \mathbb{E}_{\tilde{x},t}[\|s(\tilde{x},t;\theta) - s_\mathcal{D}(\tilde{x},t;\theta_\mathcal{D})\|_2^2]$
    - 设计动机：提供比噪声更平滑的监督目标，加速收敛

3. **二阶Jacobian匹配（核心创新）**:

    - 功能：对齐剪枝模型和原始模型的局部敏感性
    - 核心思路：FTLE理论表明扰动的放大程度由 $\|v_1\| \approx \sqrt{v_0^\top J^\top J v_0}$ 决定。直接计算全Jacobian不可行，通过随机投影 $v \sim \mathcal{N}(0,I)$ 估计方向性膨胀率：
    $\mathcal{L}_{2nd\text{-}Jac} = \mathbb{E}_{\tilde{x},t,v}\left[(\|J\hat{v}\|_2^2 - \|J_\mathcal{D}\hat{v}\|_2^2)^2\right]$
      其中 $\hat{v} = v/\|v\|$，$J\hat{v}$ 通过Jacobian-向量积（JVP）高效计算，无需形成完整Jacobian矩阵
    - 设计动机：一阶Jacobian匹配在有噪声输入下等价于KD（Taylor展开证明），无额外收益。二阶匹配捕捉扰动跨时间步的传播行为，更匹配动力系统稳定性

### 为什么一阶Jacobian匹配无效？
论文给出Taylor展开证明：$\|s(x') - s_\mathcal{D}(x')\|_2^2 = \|s(x) - s_\mathcal{D}(x)\|_2^2 + \sigma^2\|J - J_\mathcal{D}\|_F^2 + \mathcal{O}(\sigma^4)$。在噪声输入下，输出对齐已隐式包含一阶Jacobian匹配，显式添加只增加计算开销。

### 损失函数 / 训练策略
- 架构无关：适用于U-Net和Transformer两种扩散模型架构
- 剪枝方法无关：与Diff-Pruning、BK-SDM等多种剪枝方法兼容
- 使用PyTorch的JVP功能高效计算 $J\hat{v}$

## 实验关键数据

### 主实验（LSUN + ImageNet 256×256，U-Net模型）

| 数据集 | 方法 | 参数量 | MACs | FID↓ | rFID↓ |
|--------|------|--------|------|------|-------|
| LSUN-Church | DDPM（原始） | 113.7M | 248.7G | 10.58 | - |
| | Diff-Pruning | 63.2M | 138.8G | 13.90 | 4.09 |
| | **2ndM (Ours)** | 63.2M | 138.8G | **11.25** | **2.08** |
| LSUN-Bedroom | DDPM（原始） | 113.7M | 248.7G | 6.62 | - |
| | Diff-Pruning | 63.2M | 138.8G | 17.90 | 7.62 |
| | **2ndM (Ours)** | 63.2M | 138.8G | **9.68** | **2.16** |
| ImageNet | LDM-4（原始） | 400.9M | 99.8G | 3.60 | - |
| | Diff-Pruning | 175.8M | 43.2G | 10.23 | 9.28 |
| | **2ndM (Ours)** | 175.8M | 43.2G | **5.68** | **4.11** |

Stable Diffusion (COCO 512×512)：Base+2ndM FID从15.76降至13.84，Small+2ndM从16.98降至16.17。

### 消融实验（CIFAR-10）

| 配置 | FID↓ | FTLE |
|------|------|------|
| NP only | 5.29 | 0.413 |
| NP + KD | 5.05 | 0.418 |
| NP + KD + 1st JM | 5.14 | - |
| NP + KD + 2ndM (Ours) | **4.58** | - |
| Dense（原始） | 4.19 | - |

### 关键发现
- **一阶Jacobian匹配无效**：加入一阶JM后FID反而从5.05升至5.14，验证了理论分析
- **二阶匹配至关重要**：加入2ndM将FID从5.05大幅降至4.58，且FTLE更接近原始模型，证明时间敏感性对齐的有效性
- LSUN-Bedroom上FID改进46%（17.90→9.68），ImageNet上rFID改进55%
- Transformer模型上同样有效：U-ViT在CIFAR-10上FID从4.63降至4.05

## 亮点与洞察
- **动力系统视角的创新**：将扩散模型的微调问题重新表述为动力系统稳定性问题，用FTLE理论指导损失函数设计，这个视角对理解扩散模型的训练和生成过程有深刻启发
- **Taylor展开的优雅证明**：严格证明了一阶Jacobian匹配在扩散模型中的冗余性，为模型压缩中的损失设计提供理论指导
- **随机投影的实用性**：通过随机方向估计 $v^\top J^\top J v$ 绕过了高维Jacobian计算的瓶颈，使方法可扩展到大规模模型（Stable Diffusion 1.04B参数）

## 局限与展望
- 当前使用步级（step-wise）匹配近似多步Jacobian传播，对长程时间依赖的捕捉能力有限
- 随机投影的效率与估计精度之间的trade-off未充分探讨
- 仅在图像生成上验证，视频/3D等更复杂的扩散模型应用有待探索
- 可将FTLE思想扩展到蒸馏（非剪枝）场景、或用于指导采样调度器设计

## 相关工作与启发
- **vs Diff-Pruning**: Diff-Pruning仅用DSM微调剪枝模型，2ndM在此基础上加入敏感性对齐，同参数量下FID显著改善
- **vs DeepCache**: DeepCache通过缓存中间特征加速但不减参数，和剪枝方法互补
- **vs BK-SDM**: BK-SDM为Stable Diffusion设计的剪枝方法，2ndM可直接叠加使用提升效果

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ FTLE理论引入模型压缩领域，二阶Jacobian匹配的formulation优雅且有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖U-Net和Transformer架构、5个数据集、多种剪枝方法、充分消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，motivation清晰，实验设计系统
- 价值: ⭐⭐⭐⭐ 通用微调框架，但仅限模型剪枝场景
