---
title: >-
  [论文解读] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training
description: >-
  [CVPR 2026][稀疏训练] 提出 ZO-SAM，在 SAM 的扰动步骤中用零阶梯度估计替代反向传播，将 SAM 的计算开销从 2 次反传减少为 1 次，首次让 SAM 在稀疏训练中变得实用，在 CIFAR-10/100 和 ImageNet-1K 上一致提升所有主流稀疏训练方法 0.38%-2.54%…
tags:
  - "CVPR 2026"
  - "稀疏训练"
  - "SAM"
  - "零阶优化"
  - "梯度方差"
  - "平坦极小值"
---

# ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training

**会议**: CVPR 2026  
**arXiv**: [2603.13115](https://arxiv.org/abs/2603.13115)  
**代码**: 无  
**领域**: 其他  
**关键词**: 稀疏训练, SAM, 零阶优化, 梯度方差, 平坦极小值

## 一句话总结

提出 ZO-SAM，在 SAM 的扰动步骤中用零阶梯度估计替代反向传播，将 SAM 的计算开销从 2 次反传减少为 1 次，首次让 SAM 在稀疏训练中变得实用，在 CIFAR-10/100 和 ImageNet-1K 上一致提升所有主流稀疏训练方法 0.38%-2.54%。

## 研究背景与动机

**领域现状**：稀疏神经网络通过保持少量活跃权重大幅降低参数量和计算成本。主流方法分静态（LTH, SNIP, GraSP）和动态（SET, DSR, RigL, MEST）两类。

**现有痛点**：
   - 稀疏训练中梯度信号混乱嘈杂——大量权重被剪枝后，剩余参数承担不成比例的负担，梯度方差随稀疏度增加急剧增大
   - 高稀疏度导致损失面变窄变陡，优化轨迹低效迂回
   - SAM 可引导模型到平坦极小值来缓解这些问题，但其**双重反传的计算开销**正好违背了稀疏训练节约计算的初衷

**核心矛盾**：SAM 的泛化收益 vs 双倍计算开销，在稀疏训练（本身就是为省计算）的场景下矛盾尤为突出

**切入角度**：SAM 的扰动步骤对梯度精度要求不高（只需确定扰动方向），可用粗糙的零阶估计替代精确梯度

**核心 idea**：在 SAM 扰动步用零阶随机梯度估计（RGE），更新步保留一阶精确梯度，将反传次数从 2 减为 1

## 方法详解

### 整体框架

ZO-SAM 保持 SAM 的两步结构，但修改了第一步：

1. **扰动步（零阶）**：用 RGE 估计梯度方向，无需反向传播
   $$\epsilon = \rho \frac{\hat{\nabla}\mathcal{L}(\theta)}{\|\hat{\nabla}\mathcal{L}(\theta)\|}$$
2. **更新步（一阶）**：在扰动点处用精确一阶梯度更新参数
   $$\theta \leftarrow \theta - \eta \nabla\mathcal{L}(\theta^*(\epsilon))$$

### 关键设计

1. **随机梯度估计 (RGE) 替代反传**:

    - 功能：在扰动步用前向传播估计梯度方向，消除第一次反传
    - 核心公式：
    $\hat{\nabla}\mathcal{L}(\theta) = \frac{1}{m}\sum_{i=1}^m \frac{\mathcal{L}(\theta + \delta u_i) - \mathcal{L}(\theta - \delta u_i)}{2\delta} u_i$
      其中 $u_i \sim \mathcal{N}(0, I)$，$\delta$ 为小步长，$m \ll d$ 为采样数
    - 设计动机：扰动步的目标是找到"大致的最差方向"，不需要精确梯度；RGE 只需 $2m$ 次前向传播（$m$ 很小），远低于一次完整反传
    - 为什么选 RGE 而非 CGE：CGE 需要 $d$ 次评估（$d$ 是参数维度，数百万级），不可行；RGE 的随机方向采样还提供更平滑的景观探索

2. **一阶精确更新保留**:

    - 功能：在扰动后的参数点 $\theta^*(\epsilon) = \theta + \epsilon$ 处用标准反传计算精确梯度
    - 设计动机：参数更新步需要高精度梯度以保证训练稳定性和收敛性，这一步不能用近似

3. **与稀疏训练方法的兼容性**:

    - 功能：ZO-SAM 作为优化器可即插即用地替换 SGD，与任何稀疏训练方法组合
    - 验证了与 7 种方法的组合：LTH, SNIP, GraSP（静态） + SET, DSR, RigL, MEST（动态）
    - 设计动机：通用优化框架，不改变稀疏结构搜索逻辑

### 损失函数 / 训练策略

使用标准分类损失（交叉熵），ZO-SAM 仅改变优化器。超参数 $\rho$（邻域大小）继承 SAM 默认值，$\delta$（零阶步长）和 $m$（采样数）为新增超参。

## 实验关键数据

### 主实验 — ResNet-32 on CIFAR-10/100（90%/95%/98% 稀疏度）

| 方法 | CIFAR-10 90% | CIFAR-10 98% | CIFAR-100 90% | CIFAR-100 98% |
|------|-------------|-------------|--------------|--------------|
| RigL | 93.07 | 89.00 | 70.34 | 64.07 |
| **RigL+ZO-SAM** | **93.66**(+0.59) | **90.61**(+1.61) | **72.88**(+2.54) | **65.17**(+1.10) |
| MEST | 92.56 | 89.22 | 70.44 | 64.59 |
| **MEST+ZO-SAM** | **93.50**(+0.94) | **91.53**(+2.31) | **72.20**(+1.76) | **66.01**(+1.42) |

### Transformer on ImageNet-1K

| 模型 | 稀疏度 | 方法 | Accuracy(%) | 提升 |
|------|--------|------|-------------|------|
| DeiT-Small | 70% | RigL | 77.99 | - |
| DeiT-Small | 70% | **RigL+ZO-SAM** | **79.16** | +1.17 |
| DeiT-Tiny | 50% | SViTE | 70.18 | - |
| DeiT-Tiny | 50% | **SNIP+ZO-SAM** | **71.32** | +1.14 |

### 收敛速度对比

| 方法 | 达到90%精度的epoch数（CIFAR-10, sp=0.9） |
|------|--------------------------------------|
| SGD | 104 |
| ESAM | 75 |
| LookSAM(k=5) | 79 |
| GSAM | 84 |
| **ZO-SAM** | **70** |

### 关键发现
- **稀疏度越高，ZO-SAM 收益越大**：98% 稀疏度下提升最显著（MEST+ZO-SAM 在 CIFAR-10 上 +2.31%），因为高稀疏度梯度方差问题更严重
- ZO-SAM 使损失面从窄深盆地变为宽浅盆地（可视化验证）
- 梯度方差显著降低：90% 稀疏度下 ZO-SAM 的梯度方差约为 SGD 的 1/3
- 收敛速度比 SGD 快约 30 个 epoch，与 ESAM 等高效 SAM 变体相当
- 在 Transformer（DeiT）上同样有效，不局限于 CNN
- 在 CIFAR-10-C 分布偏移测试中表现更鲁棒

## 亮点与洞察
- **精确诊断稀疏训练的核心问题**：不是笼统说"稀疏训练难"，而是精确定位到"梯度方差大"这个具体原因，然后针对性解决。
- **零阶-一阶的混合策略非常巧妙**：扰动步不需要精确方向（用零阶），更新步需要精确梯度（用一阶），精准分配计算资源。这种"在不需要精度的地方省计算"的思路值得学习。
- **即插即用的通用性**：7 种稀疏训练方法 × 3 种稀疏度 × 2 个数据集全面提升，无需修改稀疏方法本身。
- **SAM 在稀疏训练中的首次实用化**：之前 SAM 的双倍开销使其在稀疏训练中不实际，ZO-SAM 消除了这个障碍。

## 局限与展望
- RGE 引入的近似噪声在极高维度下可能累积，大模型上需更多验证
- 零阶采样数 $m$ 的选择缺乏自适应策略，目前需手动调优
- 仅验证了 $\ell_\infty$ 稀疏训练，结构化稀疏（如通道剪枝）未涉及
- 在完整 ImageNet-1K 上仅测了 DeiT-Tiny/Small，更大模型（ViT-Large 等）的效果未知
- 与更高效的 SAM 变体（ESAM, LookSAM）的组合未探索——是否可以"ZO-ESAM"进一步加速？

## 评分
- 新颖性: ⭐⭐⭐⭐ 零阶+SAM 的组合虽然简单，但选择性地应用于扰动步的设计决策体现了深入思考
- 实验充分度: ⭐⭐⭐⭐⭐ 7 种方法 × 3 稀疏度 × 多数据集/架构，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 动机分析（梯度方差、损失面可视化）做得好
- 价值: ⭐⭐⭐⭐ 让 SAM 在稀疏训练中真正可用，实际工程价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Order Matters: 3D Shape Generation from Sequential VR Sketches](order_matters_3d_shape_generation_from_sequential_vr_sketches.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](vit3_unlocking_test_time_training_in_vision.md)
- [\[NeurIPS 2025\] Sharpness-Aware Minimization with Z-Score Gradient Filtering](../../NeurIPS2025/others/sharpness-aware_minimization_with_z-score_gradient_filtering.md)
- [\[CVPR 2026\] FEAT: Federated Geometry-Aware Correction for Exemplar Replay under Continual Dynamic Heterogeneity](feat_federated_geometry_aware_correction_for_exemplar_replay_under_continual_dynamic_heterogeneity.md)
- [\[CVPR 2026\] Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model](rethinking_snn_online_training_and_deployment_grad.md)

</div>

<!-- RELATED:END -->
