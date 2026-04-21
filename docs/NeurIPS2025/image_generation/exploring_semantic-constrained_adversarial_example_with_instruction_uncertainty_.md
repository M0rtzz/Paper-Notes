---
title: >-
  [论文解读] Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction
description: >-
  [NeurIPS 2025][图像生成][对抗样本] 提出多维度指令不确定性缩减框架 InSUR，通过 ResAdv-DDIM 采样器稳定对抗优化方向、上下文编码的攻击场景约束、以及基于 WordNet 的语义抽象评估，首次实现了从自然语言指令生成 2D/3D 语义约束对抗样本（SemanticAE）。
tags:
  - NeurIPS 2025
  - 图像生成
  - 对抗样本
  - 语义约束
  - 扩散模型
  - 3D对抗
  - 迁移攻击
---

# Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction

**会议**: NeurIPS 2025  
**arXiv**: [2510.22981](https://arxiv.org/abs/2510.22981)  
**代码**: 未公开  
**领域**: image_generation  
**关键词**: 对抗样本, 语义约束, 扩散模型, 3D对抗, 迁移攻击

## 一句话总结

提出多维度指令不确定性缩减框架 InSUR，通过 ResAdv-DDIM 采样器稳定对抗优化方向、上下文编码的攻击场景约束、以及基于 WordNet 的语义抽象评估，首次实现了从自然语言指令生成 2D/3D 语义约束对抗样本（SemanticAE）。

## 背景与动机

传统对抗样本研究集中在已有数据周围寻找小扰动，而直接从自然语言指令生成对抗样本（SemanticAE）是一个新兴但未被充分探索的方向。给定语义描述，目标是生成语义正确但无法被深度学习模型正确识别的数据。现有方法（AdvDiff、SD-NAE、VENOM 等）存在三方面局限：
1. **指代多样性**导致多步扩散模型中语言引导不一致，对抗优化不稳定
2. **描述不完整性**导致攻击场景适应性差
3. **语义边界模糊**使得评估 SemanticAE 生成器困难

## 核心问题

如何从不确定的人类自然语言指令出发，生成可迁移、自适应、有效的语义约束对抗样本？

## 方法详解

### 问题定义

$$\text{find } x_{\text{adv}} \in \mathcal{S}(\text{Text}) \quad \text{s.t.} \quad \mathcal{M}(x_{\text{adv}}) \in A_{\text{Text}}$$

其中 $\mathcal{S}(\text{Text})$ 是符合语义约束的数据集合，$\mathcal{M}$ 是目标模型（黑盒），$A_{\text{Text}}$ 是与语义不同的错误输出集合。

### 模块 1：ResAdv-DDIM 采样器（解决指代多样性）

核心思想：在每步去噪时利用 DDIM 的残差快捷预测 $x_0$ 的粗略估计 $g_\theta(x_t)$，而非直接用 $\nabla_{x_0}\mathcal{L}_{\text{ATK}}$ 近似 $\nabla_{x_t}\mathcal{L}_{\text{ATK}}$。

$$g_\theta(x_t) = \underbrace{f_{\theta,\Delta T_1} \circ f_{\theta,\Delta T_2} \circ \cdots \circ f_{\theta,\Delta T_k}}_{k \text{ 步, } k \ll T/\Delta T}(x_t)$$

$$x_{t-\Delta T} = f_{\theta,\Delta T}\left(\arg\max_{x_t} \mathcal{L}_{\text{ATK}}(\mathcal{M}(g_\theta(x_t)))\right)$$

语义约束通过轨迹偏差上界保证：

$$\|\text{Denoise}_{\text{DDIM}}(x_{t_s-\Delta T}) - \text{Denoise}_{\text{Adv}}(x_{t_s-\Delta T})\|_2 < \epsilon$$

自适应攻击优化使用早停机制：当估计的攻击失败概率低于阈值 $\xi_1 = 0.1$ 或 $\xi_2 = 0.01$ 时终止。

### 模块 2：上下文编码的攻击场景约束

**2D 生成**：通过 guidance masking 重分配条件/无条件引导：

$$\epsilon_\theta(x_t, t) = (1-M) \cdot \epsilon_{\theta,\text{Unconditional}}(x_t, t) + M \cdot \epsilon_{\theta,\text{Conditional}}(x_t, t, \text{Text})$$

**3D 生成**（首次实现）：将 ResAdv-DDIM 与 Gaussian Splatting 渲染器结合：

$$g_\theta(z_t, \mathbf{pos}, \text{Camera}) = \text{Renderer}_{\text{GS}}(\mathcal{D}_{\text{GS}}(f_{\theta,\Delta T_1} \circ \cdots \circ f_{\theta,\Delta T_k}(z_t, \mathbf{pos}), \mathbf{pos}), \text{Camera})$$

通过 EoT 方法对未知相机位姿进行梯度累积优化。

### 模块 3：语义抽象评估增强

基于 WordNet 构建层次化标签分类体系，定义抽象层级的逃逸攻击任务：

$$\text{Text} = \text{"Realistic image of [AbstractedLabel], specifically, [label]"}$$

$$A_{\text{Text}} = \{\text{label}_{\text{Adv}} \mid \text{AbstractedLabel} \notin \mathbf{Ancestors}(\text{label}_{\text{Adv}})\}$$

提出相对攻击成功率 $ASR_{\text{Relative}}$ 和配对语义差异指标 $\text{SemanticDiff}_\mathcal{S}$，通过同时生成非对抗示例 $x_{\text{exemplar}}$ 验证语义一致性。

## 实验关键数据

### 2D SemanticAE（$\epsilon = 2.5$，平均 ASR 跨目标模型）

| 代理模型 | 方法 | Acc.↓ | ASR↑ | CLIP-Q↑ | LPIPS↓ |
|---------|------|-------|------|---------|--------|
| ResNet50 | MI-FGSM | 33.4% | 41.5% | 0.548 | 0.201 |
| ResNet50 | SD-NAE | 37.1% | 47.4% | 0.841 | 0.457 |
| ResNet50 | VENOM | 34.5% | 34.4% | 0.795 | 0.023 |
| ResNet50 | **InSUR** | **15.1%** | **62.0%** | 0.815 | 0.031 |
| ViT-B | VENOM | 30.5% | 40.6% | 0.796 | 0.021 |
| ViT-B | **InSUR** | **10.9%** | **69.7%** | 0.815 | 0.038 |

- 在所有代理+任务设置中，InSUR 平均 ASR 至少提升 **1.19×**，最小 ASR 提升 **1.08×**
- ViT-B 代理模型上 ASR 达到 69.7%，远超 VENOM 的 40.6%

### 抽象标签逃逸任务

| 代理模型 | 方法 | Acc.↓ | ASR↑ | CLIP-Q↑ |
|---------|------|-------|------|---------|
| ResNet50 | VENOM | 51.0% | 34.9% | 0.779 |
| ResNet50 | **InSUR** | **35.2%** | **47.9%** | **0.808** |
| ViT-B | VENOM | 46.3% | 40.3% | 0.780 |
| ViT-B | **InSUR** | **28.7%** | **55.4%** | **0.814** |

## 亮点

- ⭐ 首次实现从自然语言指令的无参考 3D 语义对抗样本生成
- ⭐ ResAdv-DDIM 通过残差快捷预测解决多步扩散模型中的对抗方向不一致问题
- ⭐ 基于 WordNet 的评估体系为 SemanticAE 提供了合理的语义边界定义
- 系统性地将指令不确定性分解为三个维度并逐一突破
- 在语义保持（低 LPIPS）和攻击效果（高 ASR）之间取得优秀的 Pareto 前沿

## 局限与展望

- $\epsilon$ 参数需手动调节，不同场景可能需要不同值
- 3D 生成依赖 Trellis 框架，换用其他 3D 生成模型的泛化性未验证
- 评估指标回避了 FID/IS（担心被对抗攻击），但缺少其他生成质量评估
- ResAdv-DDIM 的残差步数 $k \in \{1,2,3,4\}$ 选择策略未深入讨论
- 生成时间较 VENOM（3.09s）慢（7.26s），虽然仍比 SD-NAE（24.43s）快

## 与相关工作的对比

| 方法 | 生成形式 | 迁移攻击 | 3D 支持 | 语义约束方式 |
|------|---------|---------|---------|------------|
| AdvDiff | 扰动型 | 弱 | ✗ | 隐式 |
| SD-NAE | 生成型 | 中 | ✗ | 端到端优化 |
| VENOM | 生成型 | 中 | ✗ | 采样过程修改 |
| **InSUR** | **生成型** | **强** | **✓** | **多维不确定性缩减** |

## 启发与关联

- ResAdv-DDIM 的残差预测思想可借鉴到其他扩散模型控制任务
- 语义抽象评估方法可推广为通用的语义一致性评测框架
- Guidance masking 的分区策略可应用于可控图像编辑（背景/前景分离引导）
- 3D 对抗样本生成对自动驾驶安全评估有直接应用价值

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (SemanticAE 概念化 + 首个 3D 实现)
- 实验充分度: ⭐⭐⭐⭐ (多代理、多任务、2D/3D 全面实验)
- 写作质量: ⭐⭐⭐ (内容密集，符号系统复杂，可读性一般)
- 价值: ⭐⭐⭐⭐ (为 AI 安全评估提供新工具)

<!-- RELATED:START -->

## 相关论文

- [Composition and Alignment of Diffusion Models using Constrained Learning](composition_and_alignment_of_diffusion_models_using_constrai.md)
- [Constrained Discrete Diffusion](constrained_discrete_diffusion.md)
- [Training-Free Constrained Generation with Stable Diffusion Models](training-free_constrained_generation_with_stable_diffusion_models.md)
- [Epistemic Uncertainty for Generated Image Detection](epistemic_uncertainty_for_generated_image_detection.md)
- [Instant Adversarial Purification with Adversarial Consistency Distillation](../../CVPR2025/image_generation/instant_adversarial_purification_with_adversarial_consistency_distillation.md)

<!-- RELATED:END -->
