---
title: >-
  [论文解读] Fine-Tuning Diffusion Models via Intermediate Distribution Shaping
description: >-
  [ICLR 2026][医学图像][扩散模型微调] 统一拒绝采样微调方法为GRAFT框架并证明其隐式执行KL正则化奖励最大化，进而提出P-GRAFT在中间去噪步骤做分布整形（偏差-方差权衡更优），以及Inverse Noise Correction无需奖励即可改进流模型质量，在T2I上VQAScore提升8.81%。
tags:
  - ICLR 2026
  - 医学图像
  - 扩散模型微调
  - 拒绝采样
  - KL正则
  - 中间分布
  - 逆噪声校正
---

# Fine-Tuning Diffusion Models via Intermediate Distribution Shaping

**会议**: ICLR 2026  
**arXiv**: [2510.02692](https://arxiv.org/abs/2510.02692)  
**代码**: 无  
**领域**: 扩散模型/微调  
**关键词**: 扩散模型微调, 拒绝采样, KL正则, 中间分布, 逆噪声校正

## 一句话总结
统一拒绝采样微调方法为GRAFT框架并证明其隐式执行KL正则化奖励最大化，进而提出P-GRAFT在中间去噪步骤做分布整形（偏差-方差权衡更优），以及Inverse Noise Correction无需奖励即可改进流模型质量，在T2I上VQAScore提升8.81%。

## 研究背景与动机

**领域现状**：扩散模型微调常用PPO+KL正则化，但扩散模型的边际似然不可计算，导致KL项要么被忽略（不稳定）要么用轨迹KL近似（次优+初始值函数偏差）。

**现有痛点**：(1) 边际KL不可计算→PPO需要放松近似；(2) 拒绝采样方法（RAFT/BoN）虽实用但理论联系不清楚；(3) 只对最终数据分布做整形，没有利用扩散模型中间步骤的结构。

**核心矛盾**：扩散模型需要KL正则化来稳定微调，但边际KL不可计算。

**切入角度**：证明拒绝采样隐式实现了边际KL约束（尽管似然不可计算），然后利用扩散的多步结构在中间分布做整形。

**核心 idea**：拒绝采样=隐式KL正则化 → 在中间去噪步做拒绝采样 → 更优的偏差-方差权衡。

## 方法详解

### 整体框架
两个贡献：(1) P-GRAFT：在中间步骤 $t$ 做拒绝采样微调；(2) Inverse Noise Correction：反转流模型学习更优初始噪声分布，无需奖励。

### 关键设计

1. **GRAFT统一框架**:

    - 功能：统一经典拒绝采样、Best-of-N、Top-K等为广义拒绝采样(GRS)
    - 核心思路：Lemma 2.3证明GRS接受的样本分布是KL正则化奖励最大化的解 $p^{\text{RL}}(x) \propto \exp(\hat{r}(x)/\alpha)\bar{p}(x)$，奖励被重整形
    - 设计动机：扩散模型的边际KL不可计算，但GRS隐式实现了它

2. **P-GRAFT (Partial-GRAFT)**:

    - 功能：不对最终样本做拒绝采样，而对中间去噪状态 $X_t$ 做
    - 核心思路：Lemma 3.2证明P-GRS整形了中间分布 $\bar{p}_t$ 而非最终分布。微调模型只负责 $T \to t$ 的去噪，$t \to 0$ 用原始模型。偏差-方差权衡：$t$ 大→奖励方差大但学习问题简单（score函数简单），$t$ 小→奖励精确但学习困难。
    - 设计动机：选择合适的中间时间 $t$ 可以兼顾两方面

3. **Inverse Noise Correction**:

    - 功能：反转流模型从数据到噪声的映射，学习更优的初始噪声分布
    - 核心思路：用adapter学习噪声空间中的修正，无需显式奖励函数
    - 设计动机：流模型的可逆性使得推断初始噪声分布成为可能

### 损失函数 / 训练策略
- P-GRAFT：生成M个轨迹→用GRS在中间步骤选择→在接受的样本上SFT微调 $T \to t$ 的部分
- Inverse Noise Correction：参数高效的adapter微调

## 实验关键数据

### 主实验
Stable Diffusion v2 T2I微调：

| 方法 | VQAScore | 相对基线提升 | 说明 |
|------|----------|------------|------|
| SD v2 (基线) | 基线 | — | 未微调 |
| Policy Gradient | 中 | 中 | PPO类方法 |
| GRAFT (最终步) | 好 | 好 | 标准拒绝采样 |
| **P-GRAFT** | **最好** | **+8.81%** | 中间步拒绝采样 |
| SDXL-Base | 对比 | — | 更大模型 |

### 多任务验证

| 任务 | 方法 | 效果 |
|------|------|------|
| 布局生成 | P-GRAFT | 显著提升 |
| 分子生成 | P-GRAFT + 去重 | 提升+多样性保持 |
| 无条件图像生成 | Inverse Noise Correction | FID改善 + FLOPs降低 |

### 关键发现
- P-GRAFT在T2I上超越policy gradient方法（PPO）和标准GRAFT
- 假设检验证实：较小 $t$ 的中间状态 $X_t$ 携带更多关于最终奖励的信息（方差分析）
- 分子生成中GRS的去重变体有效防止模式坍缩，重整形后的奖励自动包含多样性项
- Inverse Noise Correction在不需要奖励的情况下改善FID，且降低了每图FLOPs

## 亮点与洞察
- **GRS=隐式KL正则化**：这个理论结果解决了扩散模型微调中一个基本的技术难题。边际KL不可计算→不需要计算，拒绝采样隐式实现了它。
- **中间分布整形的偏差-方差视角**：不只是"在哪个步骤做"的工程选择，而有明确的数学原理支撑——选择使偏差和方差乘积最小的 $t$。
- **分子生成的去重GRS**：重整形后的奖励 $\hat{r}$ 自动包含多样性惩罚——log(1/N_copies)，优雅地防止模式坍缩。

## 局限与展望
- P-GRAFT中最优中间时间 $t$ 需要实验搜索
- Inverse Noise Correction仅适用于流模型（需要可逆性）
- 生成M个完整轨迹的计算开销较大
- 理论依赖"良好训练的去噪器"假设

## 相关工作与启发
- **vs PPO/DPPO**: 避免了KL计算困难，隐式KL约束更稳定
- **vs RAFT/RSO**: GRAFT提供了统一视角，P-GRAFT利用了扩散结构进一步优化
- **vs DPO for diffusion**: DPO用偏好迁移KL，P-GRAFT用拒绝采样更直接

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ GRAFT统一理论+P-GRAFT的偏差方差分析都是重要贡献
- 实验充分度: ⭐⭐⭐⭐ T2I/布局/分子/无条件生成全覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 理论与实践结合紧密，数学推导清晰
- 价值: ⭐⭐⭐⭐⭐ 对扩散模型微调范式有重要理论和实用影响

<!-- RELATED:START -->

## 相关论文

- [Thompson Sampling via Fine-Tuning of LLMs](thompson_sampling_via_fine-tuning_of_llms.md)
- [Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](../../CVPR2026/medical_imaging/accelerating_stroke_mri_with_diffusion_probabilist.md)
- [Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](../../CVPR2026/medical_imaging/towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](../../AAAI2026/medical_imaging/g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)

<!-- RELATED:END -->
