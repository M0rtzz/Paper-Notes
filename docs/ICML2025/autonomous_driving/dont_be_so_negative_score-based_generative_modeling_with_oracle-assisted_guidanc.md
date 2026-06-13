---
title: >-
  [论文解读] Don't be so Negative! Score-based Generative Modeling with Oracle-assisted Guidance
description: >-
  [ICML 2025][自动驾驶][扩散模型] 提出 Gen-neG 方法，通过迭代地在扩散模型的合成数据上训练贝叶斯最优分类器并用其引导采样，将生成分布从约束违规区域引导至正支撑域。关键创新在于正确处理类先验概率的重要性采样，交通场景生成中碰撞+越界率从 29.3% 降至 5.6%。 领域现状：Score-based 扩散…
tags:
  - "ICML 2025"
  - "自动驾驶"
  - "扩散模型"
  - "classifier guidance"
  - "constrained generation"
  - "oracle"
  - "collision avoidance"
  - "negative examples"
---

# Don't be so Negative! Score-based Generative Modeling with Oracle-assisted Guidance

**会议**: ICML 2025  
**arXiv**: [2307.16463](https://arxiv.org/abs/2307.16463)  
**代码**: [GitHub](https://github.com/plai-group/gen-neg)  
**领域**: 自动驾驶  
**关键词**: diffusion model, classifier guidance, constrained generation, oracle, collision avoidance, negative examples

## 一句话总结

提出 Gen-neG 方法，通过迭代地在扩散模型的合成数据上训练贝叶斯最优分类器并用其引导采样，将生成分布从约束违规区域引导至正支撑域。关键创新在于正确处理类先验概率的重要性采样，交通场景生成中碰撞+越界率从 29.3% 降至 5.6%。

## 研究背景与动机

**领域现状**：Score-based 扩散模型在图像生成、运动合成、场景生成等任务上取得巨大成功，但在约束域应用中，生成的样本有一定比例违反物理或安全约束。例如，自动驾驶场景生成中车辆可能重叠（碰撞）或偏离道路；人体运动生成中关节可能穿透地面。

**现有痛点**：最简单的解决方案是拒绝采样——用 oracle（如碰撞检测器）过滤生成结果。但在实时系统中（如自动驾驶路径规划），违规率为 $\epsilon$ 时，以 $1-\delta$ 概率获得至少一个合法样本需要 $\log\delta / \log\epsilon$ 个并行采样器（例如 $\epsilon=0.5, \delta=10^{-9}$ 时需要约 30 个），计算代价可能无法承受。

**核心矛盾**：扩散模型通过正则化（有限网络容量、有限扩散步数）实现泛化，但泛化的代价是在约束违反区域也分配了概率质量。我们需要控制泛化的"方向"——在合法区域内泛化，但不泛化到非法区域。

**本文目标** 给定训练数据集（全部为合法样本）+ oracle 函数 $\mathcal{O}: \mathcal{X} \to \{0, 1\}$（标记样本是否合法），训练一个扩散模型使其：(1) 最大化训练数据的似然；(2) 在非法区域分配零概率质量。

**切入角度**：作者发现 classifier guidance 的一个此前被忽略的必要条件——分类器必须是针对当前扩散模型分布的贝叶斯最优分类器，且训练时需要正确处理正负样本的类先验概率 $\alpha$ 和 $1-\alpha$。如果在平衡数据集上训练而不做重要性采样修正，分类器的决策边界会偏移，导致引导方向错误。

**核心 idea**：从扩散模型生成合成数据，用 oracle 标注正负样本，通过重要性采样训练贝叶斯最优分类器进行 classifier guidance，迭代 stacking 分类器逐步消除违规。

## 方法详解

### 整体框架

Gen-neG 分为两个阶段。阶段一：在原始训练数据上按标准方式训练基线扩散模型。阶段二（迭代）：从当前模型采样 → oracle 标注 → 平衡数据集 + 重要性采样训练分类器 → classifier guidance 形成新模型 → 重复。可选地，多个 stacked 分类器可蒸馏为单一模型以降低推理成本。

### 关键设计

1. **贝叶斯最优分类器与重要性采样修正**:

    - 功能：训练一个针对当前模型分布 $p_\theta$ 的时间依赖二元分类器 $C_\phi(\mathbf{x}_t; t)$，用于 classifier guidance
    - 核心思路：贝叶斯最优分类器的目标为交叉熵损失 $\mathcal{L}_\phi^{\text{CE}} = -\mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_t}[\mathcal{O}(\mathbf{x}_0) \log C_\phi(\mathbf{x}_t; t) + (1-\mathcal{O}(\mathbf{x}_0)) \log(1 - C_\phi(\mathbf{x}_t; t))]$。关键问题：当基线模型已经较好时，负样本（违规）远少于正样本，导致严重类别不平衡。Gen-neG 通过采样平衡数据集并在损失中用类先验 $\alpha = p_\theta(y=1)$ 做重要性加权来修正：正样本权重乘 $\alpha$，负样本乘 $(1-\alpha)$。最终损失为 $\hat{\mathcal{L}}_\phi^{\text{cls}} = \frac{1}{N}\sum_{\mathbf{x}_0 \in \hat{\mathcal{D}}^+} \alpha \cdot [-\log C_\phi] + \frac{1}{N}\sum_{\mathbf{x}_0 \in \hat{\mathcal{D}}^-} (1-\alpha) \cdot [-\log(1-C_\phi)]$
    - 设计动机：不做重要性采样修正时，分类器在平衡数据集上收敛到 $C^*(x) = p(y=1|x; \text{balanced})$ 而非真实的 $p_\theta(y=1|x)$——这两者的决策边界不同，后者才能正确引导扩散模型。实验证明不修正 $\alpha$ 会导致 ELBO 显著恶化

2. **分类器引导与迭代 Stacking**:

    - 功能：通过分类器梯度修正扩散过程的 score function，引导采样远离违规区域
    - 核心思路：修正后的 score 为 $s_{\theta,\phi}(\mathbf{x}_t; t) = s_\theta(\mathbf{x}_t; t) + \nabla_{\mathbf{x}_t} \log C_\phi(\mathbf{x}_t; t)$。理论保证（Theorem 3.1）：当 $C_\phi$ 是贝叶斯最优分类器时，$s_{\theta,\phi}$ 精确等于正类条件分布 $p_\theta(\mathbf{x}_t | y=1)$ 的 score。推论（Corollary 3.2）：引导后模型在违规区域概率质量为零，且对任何合法数据集的似然不低于原模型。实际中分类器不完美，因此违规可能残余——Gen-neG 通过迭代 stacking 解决：以引导后模型为新基线，重新生成→标注→训练分类器→引导
    - 设计动机：单次引导可能无法完全消除违规（分类器有近似误差），迭代 stacking 每轮针对当前残余违规训练新分类器，逐步收紧正支撑域的边界

3. **模型蒸馏**:

    - 功能：将 stacked 的多个分类器蒸馏为单一扩散模型，降低推理时的计算开销
    - 核心思路：训练学生模型 $s_\psi^{\text{dtl}}$ 最小化与教师（基线模型 + 所有 stacked 分类器）的 score 差异：$\mathcal{L}_\psi^{\text{dtl}} = \mathbb{E}_{\mathbf{x}_0, t}[\gamma_t \|s_{\theta, \mathbf{\Phi}}(\mathbf{x}_t; t) - s_\psi^{\text{dtl}}(\mathbf{x}_t; t)\|^2]$。蒸馏后每步只需一次前向传播，而非 1+K 次
    - 设计动机：K 轮 stacking 后推理成本线性增长到 K+1 次网络评估，蒸馏消除了这一开销。实验显示蒸馏模型有时甚至超越教师（知识蒸馏效应）

### 损失函数

Gen-neG 涉及两个优化目标：(1) 基线扩散模型的标准 score matching 损失 $\mathcal{L}_\theta^{\text{DM}} = \mathbb{E}[\gamma_t \|s_\theta(\mathbf{x}_t; t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0)\|^2]$；(2) 分类器的重要性加权交叉熵损失 $\hat{\mathcal{L}}_\phi^{\text{cls}}$。两者交替优化——先训练基线模型至收敛，再在每轮 stacking 中训练分类器。

## 实验关键数据

### 交通场景生成（12 辆车联合采样）

| 方法 | 碰撞率(%)↓ | 越界率(%)↓ | 总违规率(%)↓ | r-ELBO(×10⁻²)↑ |
|------|----------|----------|------------|----------------|
| Baseline DM | 28.3±0.70 | 1.3±0.14 | 29.3±0.64 | -27.5±0.01 |
| Normalizing Flow | 91.2±0.27 | 13.1±0.48 | 91.9±0.25 | — |
| Time-indep. classifier | 20.7±0.59 | 0.9±0.09 | 21.4±0.63 | -244±30.4 |
| w/o IS (消融) | 14.6±0.49 | 0.8±0.13 | 15.2±0.50 | -28.0±0.01 |
| Gen-neG (iter 1) | 16.4±0.50 | 0.9±0.12 | 17.2±0.44 | -27.7±0.01 |
| Gen-neG (iter 2) | 11.6±0.65 | 0.6±0.10 | 12.2±0.60 | -28.0±0.01 |
| **Gen-neG (distill iter2)** | **5.1±0.24** | **0.5±0.09** | **5.6±0.20** | -27.0±0.01 |

### 人体运动生成（地面穿透避免）

| 方法 | 违规率(%)↓ | 每步违规率(%)↓ | r-ELBO(×10⁻²)↑ | FID↓ | KID(×10⁻³)↓ |
|------|----------|-------------|----------------|------|------------|
| MDM (baseline) | 27.66±0.77 | 7.84±0.27 | -1.06±0.02 | 0.445±0.040 | 8.27±2.14 |
| **Gen-neG** | 24.25±0.35 | 6.12±0.19 | **-1.01±0.03** | **0.414±0.030** | **6.99±0.78** |
| w/o IS (消融) | **22.85±0.18** | **5.47±0.18** | -1.13±0.06 | 0.415±0.030 | 8.40±1.83 |

### 关键发现

1. 蒸馏版 Gen-neG 将交通场景总违规率从 29.3% 降至 5.6%（降低 81%），同时 r-ELBO 从 -27.5 改善至 -27.0——分布质量不降反升
2. 不使用重要性采样修正 (w/o IS) 虽然违规率可能更低，但 ELBO 显著恶化——说明分布被不当扭曲，正区域的密度估计变差
3. 时间无关分类器虽降低碰撞率至 20.7%，但 ELBO 灾难性恶化至 -244——证明时间依赖性是必要的
4. 蒸馏版有时超越教师（iter 2 的 12.2%→蒸馏 5.6%），可能是知识蒸馏的正则化效应
5. 使用 Gen-neG 可将自动驾驶仿真的 GPU 成本平均降低 57%（更少的拒绝采样循环）

## 亮点与洞察

1. **揭示了 classifier guidance 的一个被忽略的必要条件**：类先验概率必须正确匹配当前模型的正负比例。这一看似简单的发现有深刻影响——不修正会导致引导方向系统性偏移
2. **理论保证优雅**：贝叶斯最优分类器引导后精确等于正条件分布的采样——零违规且似然不降（Theorem 3.1 + Corollary 3.2）
3. **Oracle 作为免费约束信号**：碰撞检测、地面穿透等物理约束的检查函数在实践中几乎免费可得
4. **蒸馏压缩多轮分类器为单一模型**：解决了 stacking 的线性推理开销问题，使方法在实时系统中可部署

## 局限性

- 迭代 stacking 训练多个分类器增加了总训练时间
- 当前方法只用合成数据训练后续分类器，未回到真实训练集——长期迭代可能导致分布漂移
- 运动生成实验中改善较小（27.66%→24.25%），可能与 MDM 预测 $x_0$ 而非 score 的架构有关
- Oracle 函数需要可微或可近似——不可微约束需要额外处理

## 相关工作与启发

- 与 Dhariwal & Nichol (2021) 的标准 classifier guidance 不同，Gen-neG 的分类器在合成数据上训练且使用 IS 修正——适用于训练集中无负样本的场景
- Diffusion bridges (Liu et al., 2023) 可实现理论上的零违规，但要求约束有解析形式——Gen-neG 通过 oracle 黑盒适用于更广泛的约束类型
- 启发：oracle-guided 框架可推广到分子设计（满足物理性质约束）、蛋白质折叠（满足结构约束）等领域

## 评分

⭐⭐⭐⭐ 方法简洁，理论清晰，揭示了 classifier guidance 中被忽略的重要细节。交通场景实验令人印象深刻（29.3%→5.6%）。运动生成实验改善有限是扣分项。整体是约束生成领域的重要贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration](geometry-to-image_synthesis-driven_generative_point_cloud_registration.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](../../ICCV2025/autonomous_driving/sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[CVPR 2025\] SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](../../CVPR2025/autonomous_driving/scenediffuser_city-scale_traffic_simulation_via_a_generative_world_model.md)
- [\[CVPR 2025\] Generative Gaussian Splatting for Unbounded 3D City Generation](../../CVPR2025/autonomous_driving/generative_gaussian_splatting_for_unbounded_3d_city_generation.md)
- [\[ICCV 2025\] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](../../ICCV2025/autonomous_driving/leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)

</div>

<!-- RELATED:END -->
