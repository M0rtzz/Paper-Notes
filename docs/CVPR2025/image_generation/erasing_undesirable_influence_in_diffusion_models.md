---
title: >-
  [论文解读] Erasing Undesirable Influence in Diffusion Models (EraseDiff)
description: >-
  [CVPR 2025][图像生成][机器遗忘] 本文提出EraseDiff，将扩散模型的数据遗忘问题形式化为基于价值函数的约束优化问题，通过自然的一阶算法同时优化保留性能和擦除效果，在DDPM/Stable Diffusion上比SA快11倍、比SalUn快2倍，同时在保留-遗忘权衡上取得Pareto最优。
tags:
  - CVPR 2025
  - 图像生成
  - 机器遗忘
  - 扩散模型
  - 约束优化
  - 价值函数
  - NSFW内容擦除
---

# Erasing Undesirable Influence in Diffusion Models (EraseDiff)

**会议**: CVPR 2025  
**arXiv**: [2401.05779](https://arxiv.org/abs/2401.05779)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 机器遗忘, 扩散模型, 约束优化, 价值函数, NSFW内容擦除

## 一句话总结
本文提出EraseDiff，将扩散模型的数据遗忘问题形式化为基于价值函数的约束优化问题，通过自然的一阶算法同时优化保留性能和擦除效果，在DDPM/Stable Diffusion上比SA快11倍、比SalUn快2倍，同时在保留-遗忘权衡上取得Pareto最优。

## 研究背景与动机

1. **领域现状**：扩散模型能生成高质量图像但存在严重风险——可能记忆并重生训练集中的个人图像、生成NSFW内容。GDPR和CCPA等法规赋予用户"被遗忘权"，要求从训练后的模型中删除特定数据的影响。

2. **现有痛点**：从头重训成本极高（SD训练需256个A100、150K GPU小时）。现有遗忘方法面临核心困境——擦除目标和保留目标的梯度方向冲突，简单加权组合（$\mathcal{L}_r + \lambda \mathcal{L}_f$）无法平衡两者。SA引入EWC等技术但计算FIM代价大；SalUn关注显著参数但仍用简单加权。

3. **核心矛盾**：擦除损失驱动模型远离遗忘数据的去噪轨迹，保留损失驱动模型维持剩余数据的去噪能力，二者梯度方向经常对立——当对擦除有利的更新方向恰好损害保留性能时，简单多目标优化（MOO）会产生振荡。

4. **本文目标**：如何在扩散模型中高效地擦除不期望的数据影响（类别遗忘/概念擦除），同时最大程度保留模型在剩余数据上的生成能力？

5. **切入角度**：受优化基元学习（MAML等）启发，先将问题建模为双层优化（内层擦除、外层保留），再通过价值函数重新形式化为单层约束优化，得到一个有闭式解的一阶更新规则——更新方向同时下降两个目标函数。

6. **核心 idea**：用价值函数约束将双目标转换为约束优化，推导出同时满足擦除和保留的最优更新方向。

## 方法详解

### 整体框架
给定预训练扩散模型参数 $\theta_0$、遗忘数据 $\mathcal{D}_f$ 和保留数据 $\mathcal{D}_r$。对遗忘数据用mismatched label的噪声预测替代真实噪声（$\epsilon_f = \epsilon_\theta(x_t|c_m), c_m \neq c$），使模型无法为遗忘类生成有意义的图像。对保留数据用标准去噪目标维持性能。通过约束优化找到兼顾两者的更新方向。

### 关键设计

1. **约束优化形式化 (Value Function Formulation)**:

    - 功能：将互相冲突的擦除-保留双目标统一为有优雅解的单层优化问题
    - 核心思路：首先定义双层优化——外层最小化保留损失 $\mathcal{L}_r(\theta; \mathcal{D}_r)$，内层约束 $\theta$ 也是遗忘损失 $\mathcal{L}_f$ 的最小化解。然后用价值函数重写为约束优化：$\min_\theta \mathcal{L}_r$ s.t. $g(\theta) = \mathcal{L}_f(\theta) - \min_\phi \mathcal{L}_f(\phi) \le 0$。约束 $g(\theta)$ 衡量当前参数离遗忘最优解的差距。进一步将更新向量 $\delta_t$ 的求解形式化为QP问题：找最接近保留梯度的方向，同时满足约束梯度方向有正投影（确保约束值下降）。
    - 设计动机：简单加权（MOO）中 $\lambda$ 固定，无法根据两个目标的相对状态动态调整。约束优化自然地引入了 $\lambda_t$ 的自适应机制——当两个梯度已经一致时 $\lambda_t = 0$（不需额外调整），冲突时 $\lambda_t > 0$（投射保留梯度使其不违反擦除约束）。

2. **最优更新规则 (Theorem 3.1)**:

    - 功能：为约束优化问题提供闭式解的一阶更新公式
    - 核心思路：最优更新方向 $\delta^* = \nabla_\theta \mathcal{L}_r + \lambda_t \nabla_\theta g$，其中 $\lambda_t = \max\{0, \frac{a_t - \nabla g^\top \nabla \mathcal{L}_r}{\|\nabla g\|^2}\}$。当保留梯度和擦除梯度方向一致（$\nabla g^\top \nabla \mathcal{L}_r > a_t$）时，$\lambda_t = 0$，直接沿保留梯度更新；当冲突时，$\lambda_t > 0$，将更新方向修正以保证擦除约束值也在下降。实践中通过K步内循环梯度下降近似 $\min_\phi \mathcal{L}_f(\phi)$。
    - 设计动机：是约束优化的KKT条件的直接推导结果。$a_t = \eta \|\nabla g\|^2$ 的选择保证了约束值的下降速度与其梯度大小成正比，避免过度/不足擦除。定理3.2进一步证明了该算法收敛到Pareto最优解。

3. **遗忘目标设计 (Forgetting Objective)**:

    - 功能：引导扩散模型无法为遗忘数据生成有意义的结果
    - 核心思路：对遗忘数据的噪声预测目标，将真实噪声 $\epsilon$ 替换为错误条件下的噪声预测 $\epsilon_f = \epsilon_\theta(x_t | c_m)$，其中 $c_m \neq c$ 是随机标签（EraseDiff$_{\text{rl}}$）或均匀噪声（EraseDiff$_{\text{noise}}$）。这使得模型在被要求生成遗忘类图像时，实际跟随的去噪轨迹指向其他类别或随机噪声，无法产生遗忘类的有意义图像。
    - 设计动机：直接用梯度上升（NegGrad）会导致模型"过度遗忘"——不仅忘掉目标类还破坏其他类的生成能力。用错误条件引导则更温和——只是让模型"混淆"遗忘类的去噪方向，不会全面破坏模型。

### 损失函数 / 训练策略
保留损失：$\mathcal{L}_r = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t|c)\|^2], \, (x_0, c) \sim \mathcal{D}_r$。遗忘损失：$\mathcal{L}_f = \mathbb{E}[\|\epsilon_f - \epsilon_\theta(x_t|c)\|^2], \, (x_0, c) \sim \mathcal{D}_f$。外循环T步更新 $\theta$，每步内循环K步近似价值函数。实践中K通常取较小值即可（5-10步）。

## 实验关键数据

### 主实验（CIFAR-10 DDPM 遗忘"airplane"类）

| 方法 | FID↓ | Precision↑ | Recall↑ | $P_\psi(y=c_f|x_f)$↓ |
|------|------|-----------|---------|---------------------|
| Unscrubbed | 9.63 | 0.40 | 0.79 | 0.97 |
| Finetune | 8.21 | 0.43 | 0.77 | 0.96 |
| NegGrad | 76.73 | 0.08 | 0.61 | 0.61 |
| SA | 8.19 | 0.43 | 0.75 | **0.06** |
| SalUn | 9.16 | 0.41 | 0.76 | **0.07** |
| **EraseDiff$_\text{noise}$** | **7.61** | **0.43** | 0.72 | 0.22 |
| **EraseDiff$_\text{rl}$** | 8.66 | **0.43** | **0.77** | 0.24 |

### Stable Diffusion 擦除nudity概念

| 方法 | FID↓ | CLIP↑ | NudeNet检出数↓ |
|------|------|-------|-------------|
| ESD | 15.76 | 30.33 | ~150 |
| SA | 25.58 | 31.03 | ~180 |
| SalUn | 25.06 | 28.91 | - |
| **EraseDiff** | **17.01** | **30.58** | **最低** |

### 关键发现
- NegGrad过度遗忘（FID飙升到76.73，Precision仅0.08），验证了简单梯度反转不可行
- Finetune和BlindSpot遗忘不足（$P_\psi$ 仍接近unscrubbed），传统分类遗忘方法不适用于生成模型
- EraseDiff在保留性能（FID、Precision、Recall）上全面优于或持平SA和SalUn
- EraseDiff$_\text{noise}$ 在保留性能上最优（FID 7.61），遗忘效果稍弱（$P_\psi$ 0.22）；EraseDiff$_\text{rl}$ 在多样性上更好（Recall 0.77）
- 在nudity擦除中，EraseDiff的FID（17.01）远优于SA（25.58）和SalUn（25.06），说明保留能力更强
- 速度上比SA快11倍、比SalUn快2倍
- 梯度余弦相似度分析验证了EraseDiff有效避免了梯度冲突——更新向量同时与保留和擦除梯度正相关

## 亮点与洞察
- **约束优化视角**：将遗忘问题从"两个损失加权"提升为"在擦除约束下最小化保留损失"，数学上更优雅，且 $\lambda_t$ 自然自适应。这个框架可迁移到任何需要平衡两个冲突目标的fine-tuning场景。
- **Pareto最优性证明**：定理3.2证明算法收敛到Pareto最优，是少有的机器遗忘方法能提供理论保证的工作。
- **梯度冲突可视化**：图2的余弦相似度分析清楚展示了MOO和EraseDiff的本质区别——MOO中保留和擦除梯度交替主导（振荡），EraseDiff两者同时正相关（协调）。

## 局限与展望
- 遗忘效果（$P_\psi$ 0.22-0.24）不如SA（0.06）和SalUn（0.07），在严格要求"完全遗忘"的场景下可能不够
- 内循环K步梯度下降增加了计算开销，K的选择需要调优
- 仅在SD v1.4上验证，未测试SDXL等更大模型
- 遗忘数据的替代标签 $c_m$ 的选择策略（随机标签vs噪声）对结果有影响，缺乏系统性研究
- 未讨论遗忘的不可逆性——攻击者是否能通过微调恢复被遗忘的能力

## 相关工作与启发
- **vs SA (Selective Amnesia)**: SA用EWC+生成重放来保留性能，计算FIM开销大且需要生成保留数据；EraseDiff直接在梯度更新层面解决冲突，更高效
- **vs SalUn**: SalUn通过识别遗忘数据的显著参数来选择性更新，方法论上关注"更新哪些参数"；EraseDiff关注"更新方向是什么"，两者正交可能可以互补
- **vs ESD**: ESD直接修改cross-attention权重擦除概念，只能用于text-conditional模型；EraseDiff适用于条件和无条件扩散模型

## 评分
- 新颖性: ⭐⭐⭐⭐ 约束优化形式化和价值函数方法在扩散遗忘中是新的，但整体算法框架源自已有优化理论
- 实验充分度: ⭐⭐⭐⭐ DDPM+SD双验证、类遗忘+概念擦除、速度比较、梯度分析，较全面
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，但符号和公式较多，入门门槛高
- 价值: ⭐⭐⭐⭐ 为扩散模型遗忘提供了有理论保证的高效方法，对AI安全和隐私保护有实际意义

<!-- RELATED:START -->

## 相关论文

- [DMin: Scalable Training Data Influence Estimation for Diffusion Models](../../CVPR2026/image_generation/dmin_scalable_training_data_influence_estimation_for_diffusion_models.md)
- [MixerMDM: Learnable Composition of Human Motion Diffusion Models](mixermdm_learnable_composition_of_human_motion_diffusion_models.md)
- [UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models](uibdiffusion_universal_imperceptible_backdoor_attack_for_diffusion_models.md)
- [The Art of Deception: Color Visual Illusions and Diffusion Models](the_art_of_deception_color_visual_illusions_and_diffusion_models.md)
- [RAD: Region-Aware Diffusion Models for Image Inpainting](rad_region-aware_diffusion_models_for_image_inpainting.md)

<!-- RELATED:END -->
