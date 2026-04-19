---
title: >-
  [论文解读] ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration
description: >-
  [AAAI 2026][图像生成][Transformer] 提出 ProCache，一个免训练的动态特征缓存框架：通过约束感知的非均匀缓存模式搜索和选择性计算策略，在 DiT-XL/2 上实现 2.90 倍加速、PixArt-α 上实现 1.96 倍加速，且图像质量几乎无损，显著优于现有缓存方法。
tags:
  - AAAI 2026
  - 图像生成
  - Transformer
  - 特征缓存
  - 免训练推理
  - 动态调度
  - Token选择
---

# ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration

**会议**: AAAI 2026  
**arXiv**: [2512.17298](https://arxiv.org/abs/2512.17298)  
**代码**: [https://github.com/macovaseas/ProCache](https://github.com/macovaseas/ProCache)  
**领域**: 图像生成  
**关键词**: 扩散Transformer加速, 特征缓存, 免训练推理, 动态调度, Token选择

## 一句话总结

提出 ProCache，一个免训练的动态特征缓存框架：通过约束感知的非均匀缓存模式搜索和选择性计算策略，在 DiT-XL/2 上实现 2.90 倍加速、PixArt-α 上实现 1.96 倍加速，且图像质量几乎无损，显著优于现有缓存方法。

## 研究背景与动机

### 领域现状

Diffusion Transformers (DiTs) 在生成建模中取得了 SOTA 表现，但其计算开销极大，严重限制了实时部署。加速 DiT 的方法包括：
- **需要训练的方法**：剪枝、量化、知识蒸馏——需要额外训练成本，在高加速比下性能不佳。
- **免训练的方法**：特征缓存——利用去噪步骤间的时序冗余，缓存已计算的特征供后续步骤复用，即插即用。

### 现有痛点

现有基于缓存的方法存在两个关键限制：

**均匀缓存间隔与非均匀时序动态不匹配**：所有方法都采用固定间隔（如每 $N$ 步全量计算一次），但 DiT 特征的变化在去噪过程中是非均匀的——早中期特征变化缓慢，后期特征变化急剧（近似指数增长趋势）。均匀策略在稳定阶段浪费了计算，在快速变化阶段又引入过大误差。

**朴素特征复用导致误差累积**：随着缓存间隔增大，特征相似性指数衰减，直接复用陈旧特征导致严重的质量退化。现有方法（如 Δ-DiT、FORA）对复用特征不做任何修正，在高加速比下质量下降明显。

### 核心发现

作者通过定量分析揭示了两个关键观察：

**误差累积是深度依赖的**：深层 block（如 Block 25-28）的特征误差远大于浅层（如 Block 1-4），且随步骤推进逐步增长。

**输出变化是时间依赖的**：早中期阶段特征变化平缓，后期急剧增大，呈近似指数上升趋势。

### 本文切入角度

**核心 idea**：既然 DiT 的特征演化是非均匀的、误差传播是深度依赖的，那么缓存策略也应该是非均匀的、计算修正也应该是有选择性的。通过离线搜索最优缓存模式 + 在缓存步骤中选择性刷新深层 block 的高重要性 token，实现质量保持下的最大加速。

## 方法详解

### 整体框架

ProCache 由两个核心组件构成：

1. **约束感知缓存模式搜索（Constraint-Aware Caching Pattern Search）**：离线搜索非均匀的最优激活调度
2. **选择性计算（Selective Computation）**：在缓存步骤中轻量修正深层 block 的重要 token

### 关键设计

#### 1. **约束感知缓存模式搜索**

**功能**：将均匀间隔缓存替换为针对模型时序特性定制的非均匀激活模式。

**核心思路**：将缓存模式表示为二进制序列 $\mathbf{s} = [s_1, s_2, \ldots, s_T] \in \{0,1\}^T$，其中 $s_t = 1$ 表示在步骤 $t$ 执行计算，$s_t = 0$ 表示复用缓存。定义三个约束：

1. **预算约束**：激活步骤总数不超过预算 $B$：$M = \sum_{t=1}^T s_t \leq B$
2. **单调约束**：复用间隔随时间非递增（后期应更频繁计算）：$v_{i+1} \leq v_i$
3. **有界约束**：每个间隔在可行范围内：$v^{\min} \leq v_i \leq v^{\max}$

其中复用间隔为 $v_i = t_{i+1} - t_i - 1$，表示两次计算之间的连续缓存步骤数。

**搜索流程**：
- 从约束空间 $\mathcal{C}$ 中采样 $K$ 个候选模式（默认 $K=5$）
- 在小规模代表性数据集上评估每个候选（如 FID）
- 选择最优模式 $\mathbf{s}^*$
- 全程免训练，单 GPU 不到 1 小时完成

**设计动机**：单调约束直接来自于图 3 的实证观察——特征变化在后期急剧增大，缓存间隔应单调缩小以匹配这一动态。有界约束避免了过长间隔导致的误差累积和过短间隔带来的效率损失。

#### 2. **选择性计算策略**

**功能**：在缓存步骤中以极小开销（约 3% 额外延迟）执行部分计算来抑制误差累积。

**核心思路**：在每个连续零块的偶数位置处执行选择性更新。更新时仅计算**深层 block 子集**和**高重要性 token 子集**，其余保持缓存。

$$\mathbf{x}_i^{(l)} = \begin{cases} f^{(l)}(\mathbf{x}_i^{(l-1)}), & \text{if } l \in \mathcal{U}^{\text{cmpt}} \text{ and } i \in \mathcal{I}^{\text{cmpt}} \\ \text{Cache}^{(l)}(\mathbf{x}_i), & \text{otherwise} \end{cases}$$

**层选择——聚焦深层 block**：分析发现误差主要集中在深层（高层语义精炼层），浅层时序稳定性强、偏差极小。取最深的 $D = r \times L$ 层进行选择性计算（默认 $r = 25\%$）。

**Token 选择——优先高重要性 token**：用注意力模块输出的 L2 范数作为重要性度量：$T(\mathbf{x}_i) = \|\mathbf{v}_i\|_2$，取 top-$p\%$ token（7-30%）进行重新计算。

$$\mathcal{I}^{\text{cmpt}} = \{i \mid \text{rank}(T(\mathbf{x}_i)) \leq p\% \times N\}$$

注意：自注意力模块不使用 token 选择（因为每个 token 关注所有其他 token，部分计算容易传播误差），仅在交叉注意力和 FFN 中使用。

### 损失函数 / 训练策略

ProCache 是完全免训练的方法，无需任何损失函数或训练流程。其核心优势在于即插即用：
- 离线搜索仅需少量推理评估
- 与现有采样器（DDIM、DPM-Solver++、Rectified Flow）正交兼容

## 实验关键数据

### 主实验

**DiT-XL/2 ImageNet 类条件生成**（50K 张，256×256）：

| 方法 | 延迟(s) ↓ | FLOPs(T) ↓ | 加速比 ↑ | FID ↓ | sFID ↓ | Precision ↑ | IS ↑ |
|------|----------|-----------|---------|------|-------|------------|------|
| DDIM-50步 | 4.549 | 23.74 | 1.00× | 2.43 | 4.40 | 0.80 | 241.25 |
| Δ-DiT (N=3) | 2.572 | 16.46 | 1.47× | 3.75 | 5.70 | 0.77 | 207.57 |
| FORA (N=3) | 2.191 | 8.59 | 2.76× | 3.88 | 6.43 | 0.79 | 229.02 |
| ToCa (N=3) | 2.087 | 10.23 | 2.32× | 3.04 | 5.14 | 0.79 | 230.70 |
| **ProCache** | **1.725** | **8.18** | **2.90×** | **2.96** | **4.93** | **0.80** | **232.85** |

**PixArt-α COCO30K 文本生成图像**：

| 方法 | 延迟(s) ↓ | 加速比 ↑ | FID ↓ | CLIP ↑ |
|------|----------|---------|------|--------|
| DPM-Solver++ 20步 | 2.142 | 1.00× | 28.12 | 16.29 |
| FORA (N=3) | 1.301 | 2.79× | 29.84 | 16.42 |
| ToCa | 1.473 | 1.77× | 28.02 | 16.43 |
| **ProCache** | **1.215** | **1.96×** | **27.66** | **16.45** |

**FLUX.1-dev/schnell PartiPrompts**：ProCache 在 FLUX.1-dev 上达 1.54× 加速（Image Reward 1.207 vs. 基线 1.202），FLUX.1-schnell 上达 1.56× 加速（Image Reward 1.138 vs. 基线 1.133），均优于 FORA 和 ToCa。

### 消融实验

| 配置 | 加速比 | FID ↓ | sFID ↓ | 说明 |
|------|-------|------|-------|------|
| 默认均匀 (B=13) | 2.93× | 4.75 | 8.43 | 均匀间隔基线 |
| + 搜索模式 | 2.93× | 3.15 | 5.12 | 零开销提升 |
| + 选择性计算 | 2.90× | 3.28 | 5.95 | 仅选择性计算 |
| **ProCache** | **2.90×** | **2.96** | **4.93** | 两者结合最优 |

深层 block 比例消融（DiT-XL/2）：

| 比例 r | FLOPs(T) ↓ | FID ↓ | IS ↑ |
|--------|-----------|------|------|
| 50% | 9.138 | 45.32 | 184.18 |
| 75% | 8.594 | 45.31 | 183.61 |
| 90% | 8.344 | 45.38 | 182.46 |

75% 为最佳平衡点；主实验用 25% 以最大化加速。

### 关键发现

1. 搜索到的非均匀模式在不改变速度的情况下将 FID 从 4.75 降至 3.15（改善 33.7%），证明模式设计本身意义重大。
2. ProCache 在 FID 上比 FORA 改善近 30%，同时加速比还更高（2.90× vs. 2.76×）。
3. 选择性计算仅引入约 3% 额外延迟（因为仅计算小部分缓存步、25% 的层、7-30% 的 token）。
4. 搜索过程具有鲁棒性：5 次独立运行的 FID 和 IS 变异极小，K=5 的最小采样预算即已足够。
5. 在超过 4.53× 的极端加速比下，ProCache 相比先前方法将质量退化减少了 56.2%。

## 亮点与洞察

1. **从实证观察直接驱动方法设计**：图 1 和图 3 的分析精准揭示了误差累积和特征演化的非均匀性，三个约束条件与观察完美对应。
2. **搜索空间设计精巧**：单调约束和有界约束将指数级搜索空间压缩到可行规模，K=5 即可找到好的模式。
3. **粒度合理的选择性计算**：层级聚焦深层、token 级聚焦高重要性，且不对自注意力做 token 裁剪避免误差传播——设计考虑细致。
4. **广泛兼容性**：适用于 DiT-XL/2、PixArt-α、FLUX.1-dev/schnell 等不同架构和采样器。

## 局限与展望

1. 离线搜索阶段虽快（<1 小时），但每个模型/采样器/步数组合都需要重新搜索。
2. Token 选择策略基于注意力输出范数，这个重要性指标的理论基础有限，可能存在更优的度量。
3. 自注意力模块不做 token 选择是保守策略，若能设计更安全的部分更新方案可进一步加速。
4. 仅验证了图像生成任务，视频生成中的时间维度带来的额外复杂性未涉及。
5. 搜索约束中的 $v^{\min}$、$v^{\max}$ 是手动设置的超参数，自适应确定可能更优。

## 相关工作与启发

- **FORA** (Selvaraju et al.)：统一均匀缓存，ProCache 证明了非均匀策略的压倒性优势。
- **ToCa** (Zou et al.)：引入 token 级缓存但仍用均匀间隔，ProCache 在模式搜索维度进一步推进。
- **Δ-DiT** (Chen et al.)：MLP 共享策略，加速比有限。
- **DeepCache/Faster Diffusion**：U-Net 架构特有的缓存方法，不适用于 DiT。
- 本文的约束采样搜索思路可推广到其他需要自适应调度的场景（如推理步骤调度、动态精度调度等）。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 约束感知搜索+选择性计算的组合有创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 4 个模型、多维度指标、充分消融、鲁棒性分析
- 写作质量: ⭐⭐⭐⭐⭐ — 分析驱动设计的逻辑链条清晰,图表信息量大
- 价值: ⭐⭐⭐⭐ — 免训练加速的实用方法，但增量感略强

<!-- RELATED:START -->

## 相关论文

- [Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation](../../NeurIPS2025/image_generation/predictive_feature_caching_for_training-free_acceleration_of_molecular_geometry_.md)
- [Playmate2: Training-Free Multi-Character Audio-Driven Animation via Diffusion Transformer with Reward Feedback](playmate2_training-free_multi-character_audio-driven_animation_via_diffusion_tra.md)
- [Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](../../CVPR2026/image_generation/adaptive_spectral_feature_forecasting_for_diffusion_sampling_acceleration.md)
- [LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](../../ICCV2025/image_generation/lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration](../../CVPR2026/image_generation/lesa_learnable_stage-aware_predictors_for_diffusion_model_acceleration.md)

<!-- RELATED:END -->
