---
title: >-
  [论文解读] Enhancing Diffusion Model Guidance through Calibration and Regularization
description: >-
  [NeurIPS 2025 (SPIGM Workshop)][图像生成][classifier guidance] 针对分类器引导扩散模型中分类器过度自信导致梯度消失的问题，提出两类互补方案：(1) Smooth ECE 校准损失微调分类器，FID 改善 ~3%；(2) 基于 f-散度的正则化采样引导（RKL/FKL/JS），无需重训练即在 ImageNet 128×128 上达到 FID 2.13。
tags:
  - NeurIPS 2025 (SPIGM Workshop)
  - 图像生成
  - classifier guidance
  - 扩散模型
  - f-divergence
  - calibration
  - conditional generation
---

# Enhancing Diffusion Model Guidance through Calibration and Regularization

**会议**: NeurIPS 2025 (SPIGM Workshop)  
**arXiv**: [2511.05844](https://arxiv.org/abs/2511.05844)  
**代码**: [ajavid34/guided-info-diffusion](https://github.com/ajavid34/guided-info-diffusion)  
**领域**: image_generation  
**关键词**: classifier guidance, diffusion model, f-divergence, calibration, conditional generation

## 一句话总结

针对分类器引导扩散模型中分类器过度自信导致梯度消失的问题，提出两类互补方案：(1) Smooth ECE 校准损失微调分类器，FID 改善 ~3%；(2) 基于 f-散度的正则化采样引导（RKL/FKL/JS），无需重训练即在 ImageNet 128×128 上达到 FID 2.13。

## 背景与动机

分类器引导扩散模型（Classifier-Guided DDPM）通过外部分类器的梯度 $\nabla_x \log p(y|x)$ 引导反向扩散过程走向目标类别，是条件图像生成的核心技术。然而存在一个关键缺陷：

**梯度消失问题**：分类器在去噪早期阶段就对部分生成的图像给出过高置信度（接近 one-hot 分布），导致 $\nabla_x \log p(y|x) \to 0$。此后的去噪步骤实际上退化为无条件生成，严重损害条件生成质量。

现有解决方案（如熵约束训练）需要从头训练分类器，无法应用于现成（off-the-shelf）的分类器。

## 核心问题

如何在不重训练扩散模型和分类器的前提下，缓解分类器引导中的梯度消失问题，提升条件生成的质量和多样性？

## 方法详解

### 1. Smooth ECE 校准损失（需微调）

定义可微的 Huber 型校准损失：

$$\mathcal{L}_{\text{ECE}} = \frac{1}{n} \sum_{b=1}^{B} \sum_{i: \hat{p}^{(i)} \in \mathcal{B}_b} \sqrt{(\hat{p}^{(i)} - a^{(i)})^2 + \beta}$$

其中 $\hat{p}^{(i)} = \max_y p_\phi(y|x^{(i)})$ 是预测置信度，$a^{(i)} = \mathbb{I}[\hat{y}^{(i)} = y^{(i)}]$ 是正确性指示。$\beta > 0$ 保证可微。

该损失可作为正则项在微调时使用，改善分类器的校准性，进而改善引导质量。

### 2. f-散度正则化采样（无需重训练）

核心思想：用分类器输出的类别分布 $p(\cdot|x)$ 和目标分布 $q_y(\cdot)$ 之间的 f-散度作为正则项，防止分布过早坍缩。引导得分定义为：

$$\mathcal{S}_\mathcal{D}(x, y) = \log p_{\tau_1, \tau_2}(y|x) - \alpha D_f(q_y(\cdot) \| p(\cdot|x))$$

其中 $q_y(i) = (1-\epsilon)\frac{1}{N} + \epsilon \mathbb{I}_{i=y}$（带平滑的目标分布），$\tau_1, \tau_2$ 为联合和边际温度。

梯度的一般形式为：

$$\nabla_x \mathcal{S}_\mathcal{D} = \tau_1 \nabla_x f_y(x) - \tau_2 \sum_i p_{\tau_2}(i|x) \nabla_x f_i(x) - \alpha \sum_i w_f(q_y(i), p(i|x)) g_i(x)$$

其中 $w_f(q, p) = p f'(p/q)$，$g_i(x) = \nabla_x f_i(x) - \sum_j p(j|x) \nabla_x f_j(x)$。

### 3. 三种 f-散度实例化

#### Reverse KL（模式覆盖）

$f(t) = -\log(t)$，权重 $w_f(q,p) = -q$。梯度分解为：

$$-\nabla_x D_{\text{KL}}(q_y \| p) = \underbrace{\sum_i q_y(i) \nabla_x f_i(x)}_{\text{目标方向}} - \underbrace{\sum_j p(j|x) \nabla_x f_j(x)}_{\text{当前方向}}$$

Reverse KL 的**模式覆盖**特性保证模型在 $q_y$ 有支撑的所有地方保持非零概率，防止模式丢失。高斯混合分析下，引导力由增强的目标方向 $(\tau_1 + \alpha\epsilon)(\mu_y - x)$ 和多样性方向 $\alpha\frac{1-\epsilon}{K}\sum_{k \neq y}(\mu_k - x)$ 组成。

#### Forward KL（模式寻找）

$f(t) = t\log(t)$，权重中包含 $\log(p(i|x)/q_y(i))$ 项。强烈惩罚 $p$ 在 $q_y$ 支撑外的概率质量，产生更锐利但多样性更低的样本。**精确度最高，召回率最低**。

#### Jensen-Shannon（平衡引导）

通过隐式混合分布 $m = \frac{1}{2}(q_y + p)$ 在模式覆盖和模式寻找之间取平衡。权重 $(q_y(i) - p(i|x))/m(i)$ 有界且在 $q_y \approx p$ 时趋于零，提供平滑的梯度动态。实验中表现最优。

### 4. 倾斜采样

利用批次内信息调整引导权重：

$$\mathcal{S}_{\text{tilted}}(t; x, y) = \frac{1}{t} \log\left(\frac{1}{N}\sum_{i \in [N]} e^{t \log p_{\tau_1, \tau_2}(y|x)}\right)$$

$t > 0$ 强调高置信度样本（提升质量），$t < 0$ 强调低置信度样本（提升多样性）。

## 实验关键数据

### Smooth ECE 微调效果（10K ImageNet 128×128）

| 方法 | FID↓ | Precision↑ | Recall↑ |
|------|------|-----------|---------|
| 标准微调分类器 | 6.15 | 0.77 | **0.68** |
| **+Smooth ECE** | **5.94** | **0.79** | 0.66 |

FID 改善 ~3%，仅需少量微调。

### 采样引导对比（10K 样本，ResNet-50）

| 方法 | FID↓ | Precision↑ | Recall↑ |
|------|------|-----------|---------|
| 基线 (ma2023) | 5.34 | **0.78** | 0.67 |
| 倾斜采样 (t=-0.2) | 5.28 | 0.77 | 0.68 |
| 熵正则化 | 5.30 | 0.77 | **0.69** |
| **RKL 引导** | **5.12** | **0.78** | 0.68 |

### 与 SOTA 对比（50K ImageNet 128×128）

| 方法 | 分类器 | FID↓ | Precision↑ | Recall↑ |
|------|--------|------|-----------|---------|
| Dhariwal et al. | 微调 | 2.97 | 0.78 | 0.59 |
| 熵感知分类器 | 专用 | 2.68 | 0.80 | 0.56 |
| Classifier-free | - | 2.43 | - | - |
| ma2023 | ResNet-50 | 2.37 | 0.77 | 0.60 |
| FKL (ours) | ResNet-101 | 2.17 | **0.80** | 0.59 |
| RKL (ours) | ResNet-101 | 2.14 | 0.79 | 0.59 |
| **JS (ours)** | ResNet-101 | **2.13** | 0.79 | 0.60 |

JS 散度取得最优 FID 2.13，无需重训练扩散模型或分类器。

### 三种散度的特性排序

| 散度 | FID | Precision | Recall | 特性 |
|------|-----|-----------|--------|------|
| FKL | 中 | 最高 | 最低 | 模式寻找，高锐利度 |
| RKL | 中上 | 中 | 中 | 模式覆盖，保持多样性 |
| **JS** | **最优** | 中 | **最高** | 平衡覆盖与寻找 |

## 亮点

1. **即插即用**：f-散度引导方法可直接用于现成分类器和扩散模型，零重训练成本
2. **严谨的理论分析**：对三种 f-散度给出了完整的梯度推导（Proposition 2）和高斯混合场景下的闭式分析（Proposition 3），数学基础扎实
3. **新颖的洞察**：JS 散度优于 RKL 和 FKL 挑战了"模式覆盖(RKL)对生成最优"的传统认知
4. **Smooth ECE 损失**简单且有效，仅需少量微调即可改善校准性和 FID

## 局限与展望

1. 作为 **Workshop paper**，实验规模有限——仅在 ImageNet 128×128 上评估，未测试更高分辨率
2. f-散度引导的超参数（$\alpha, \epsilon, \tau_1, \tau_2$）需要调节，论文未充分讨论敏感性
3. 倾斜采样的改进幅度最小（FID 5.34 → 5.28），实用性有限
4. 未与更新的条件生成方法（如 DiT + classifier-free guidance）对比
5. Smooth ECE 微调和 f-散度采样未联合评估

## 与相关工作的对比

| 维度 | Dhariwal et al. | 熵约束训练 | ma2023 | 本文 |
|------|----------------|-----------|--------|------|
| 需要重训练分类器 | ✓ | ✓ | ✗ | **✗** |
| 需要重训练扩散模型 | ✗ | ✗ | ✗ | **✗** |
| 理论分析 | 无 | 有限 | 能量视角 | **f-散度框架** |
| FID (ResNet-101) | 2.97 | 2.68 | 2.19 | **2.13** |
| 多样性保持 | 弱 | 中 | 中 | **强（JS/RKL）** |

## 启发与关联

1. **分类器置信度 ≠ 引导强度**：过度自信的分类器反而提供更弱的引导，这一矛盾揭示了校准性对条件生成的重要性
2. **f-散度族提供了一个正则化设计空间**：不同散度对应不同的 precision-recall 权衡，可根据应用需求选择
3. **JS 散度的"对称性惩罚"机制**：通过混合分布 $m$ 提供自适应校正，当预测偏离目标时加强，对齐时减弱——这一机制可能在其他引导场景中同样有效
4. 倾斜采样利用批次信息进行自适应调整的思路，与集成方法有潜在联系

## 评分

- 新颖性: ⭐⭐⭐⭐ — f-散度用于扩散引导正则化的框架新颖，理论贡献扎实
- 实验充分度: ⭐⭐⭐ — Workshop 论文规模，仅 128×128 分辨率，缺少大规模验证
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰，Proposition 链条完整，直觉解释到位
- 价值: ⭐⭐⭐⭐ — 零成本即插即用的引导改进具有极强实用性

<!-- RELATED:START -->

## 相关论文

- [Increasing the Utility of Synthetic Images through Chamfer Guidance](increasing_the_utility_of_synthetic_images_through_chamfer_guidance.md)
- [Beyond One-Hot Labels: Semantic Mixing for Model Calibration](../../ICML2025/image_generation/beyond_one-hot_labels_semantic_mixing_for_model_calibration.md)
- [Why Diffusion Models Don't Memorize: The Role of Implicit Regularization](why_diffusion_models_dont_memorize_the_role_of_implicit_regularization.md)
- [Generative Model Inversion Through the Lens of the Manifold Hypothesis](generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)

<!-- RELATED:END -->
