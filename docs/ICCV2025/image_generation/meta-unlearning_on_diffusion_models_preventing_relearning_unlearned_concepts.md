---
title: >-
  [论文解读] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts
description: >-
  [ICCV 2025][图像生成][机器遗忘] 本文提出了扩散模型的元遗忘（Meta-Unlearning）框架，在标准遗忘目标之外增加一个元目标，使得模型在被恶意微调时，与遗忘概念相关的良性知识会自毁，从而阻止已遗忘概念的重新学习，该框架兼容大多数现有遗忘方法且仅需添加一个简单的元目标。 领域现状：随着 Stable Di…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "机器遗忘"
  - "扩散模型"
  - "恶意微调防御"
  - "元学习"
  - "概念擦除"
---

# Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts

**会议**: ICCV 2025  
**arXiv**: [2410.12777](https://arxiv.org/abs/2410.12777)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 机器遗忘, 扩散模型, 恶意微调防御, 元学习, 概念擦除

## 一句话总结

本文提出了扩散模型的元遗忘（Meta-Unlearning）框架，在标准遗忘目标之外增加一个元目标，使得模型在被恶意微调时，与遗忘概念相关的良性知识会自毁，从而阻止已遗忘概念的重新学习，该框架兼容大多数现有遗忘方法且仅需添加一个简单的元目标。

## 研究背景与动机

**领域现状**：随着 Stable Diffusion 等大规模扩散模型的发展，机器遗忘（Machine Unlearning）成为防止模型生成有害/侵权内容的重要手段。ESD、SDD、UCE、RECE 等方法可以有效地从预训练模型中"遗忘"特定概念

**现有痛点**：研究发现，即使模型被正确遗忘后公开发布，恶意用户仍可通过少量微调使模型重新学习已遗忘的概念。甚至在不相关的良性数据上微调也可能部分恢复遗忘的能力

**核心矛盾**：遗忘方法只修改了模型对特定概念的输出行为，但模型中保留的某些良性概念（如"皮肤"）与遗忘概念（如"裸体"）存在关联，为恶意微调提供了"桥梁"

**本文目标**：如何使遗忘后的模型在被恶意微调时能抵抗概念重学习？

**切入角度**：受元学习（MAML）启发——在训练时模拟攻击者的微调过程，提前在参数空间中布局，使得沿微调方向优化时良性知识自毁

**核心 idea**：在遗忘训练中模拟恶意微调过程，优化一个元目标使遗忘/保留集梯度夹角 > 90°，微调降低遗忘集 loss 时自动增加保留集 loss

## 方法详解

### 整体框架

Meta-Unlearning 框架由两部分组成：(1) 标准遗忘目标 $\mathcal{L}_{unlearn}$（可以是 ESD/SDD/UCE/RECE 中的任何一种）确保模型遗忘指定概念；(2) 元目标 $\mathcal{L}_{meta}$ 确保模型在被恶意微调后仍无法重新学习遗忘概念。整个过程在 Algorithm 1 中描述为一个两层优化。

### 关键设计

1. **元目标设计（Meta Objective）**:

    - 功能：使恶意微调后的模型无法重学遗忘概念，同时触发相关良性知识自毁
    - 核心思路：
    $\mathcal{L}_{meta}(\theta^{FT}) = -\mathcal{L}_{DM}(\theta^{FT}; \mathcal{D}_{FT}) - \zeta[\mathcal{L}_{DM}(\theta^{FT}; \mathcal{D}_{retain}) - \mathcal{L}_{DM}(\theta; \mathcal{D}_{retain})]$
      其中 $\theta^{FT} = \theta - \tau \nabla_\theta \mathcal{L}_{FT}(\theta; \mathcal{D}_{FT})$ 是模拟恶意微调后的参数
    - 一阶近似展开后，关键项为：(1) $\|\nabla_\theta \mathcal{L}_{DM}(\theta; \mathcal{D}_{FT})\|_2^2$ — 最小化此项减小梯度范数，延缓重学习速度；(2) $\nabla_\theta \mathcal{L}_{DM}(\theta; \mathcal{D}_{FT})^\top \nabla_\theta \mathcal{L}_{DM}(\theta; \mathcal{D}_{retain})$ — 最小化此项使两个梯度夹角 > 90°，实现"微调降低遗忘集 loss → 保留集 loss 自动升高"的自毁机制
    - 设计动机：传统遗忘只考虑"发布时"的安全性，忽略了"发布后"的攻击风险。元目标通过提前"埋雷"使微调路径上的参数更新对模型有害

2. **兼容性设计（与现有方法的集成）**:

    - 功能：使元遗忘框架能无缝集成到基于优化（ESD/SDD）和基于闭式解（UCE/RECE）的遗忘方法中
    - 核心思路：对于需要优化的方法（ESD/SDD），梯度 $g$ 同时包含 $\mathcal{L}_{unlearn}$ 和 $\mathcal{L}_{meta}$ 的贡献；对于有闭式解的方法（UCE/RECE），先获得闭式解 $\theta^{UN}$，再从 $\theta^{UN}$ 出发优化元目标
    - 实现：自动微分即可计算元梯度，无需复杂实现

3. **两层优化结构（Bilevel Optimization）**:

    - 外层循环（N 步，lr=$\omega$）：更新模型参数以同时满足遗忘+元目标
    - 内层循环（M 步，lr=$\tau$）：模拟恶意微调过程，从当前参数出发更新 M 步
    - 权重因子 $\gamma_1$、$\gamma_2$ 分别调控遗忘目标和元目标的贡献

### 损失函数 / 训练策略

恶意微调数据集 $\mathcal{D}_{FT} \subset \mathcal{D}_{forget}$ 每步从遗忘集采样。使用 FLUX.1 模型生成三类微调数据集用于评估：HRM-s（单一有害 prompt）、HRM-m（多个有害 prompt）、CLEAN（良性 prompt）。在 SD-v1-4 和 SDXL 上验证。

## 实验关键数据

### 主实验

**SD-v1-4 裸体评分（Tab.1，Tab.2 中选取关键数据）**:

| 方法 | 类型 | FT前 | FT HRM-m 50步 | FT HRM-m 100步 | FT CLEAN 100步 |
|------|------|------|------|------|------|
| SD-1.4 原始 | - | 97.18 | - | - | - |
| ESD-u-1 | Unlearn | 6.34 | 19.01 | 21.83 | 13.38 |
| ESD-u-1 | **Meta-Unlearn** | **0.00** | **8.45** | **13.38** | **2.11** |
| ESD-u-3 | Unlearn | 3.52 | 26.76 | 38.73 | 4.93 |
| ESD-u-3 | **Meta-Unlearn** | **0.00** | **3.52** | **19.01** | **2.82** |
| SDD | Unlearn | 1.41 | 33.10 | 57.04 | 16.20 |
| SDD | **Meta-Unlearn** | **0.00** | **20.42** | **45.07** | **5.63** |
| UCE | Unlearn | 16.90 | 36.62 | 44.37 | 25.35 |
| UCE | **Meta-Unlearn** | **1.41** | **24.65** | **28.17** | **5.63** |
| RECE | Unlearn | 4.93 | 16.20 | 19.72 | 9.86 |
| RECE | **Meta-Unlearn** | **4.23** | **7.04** | **10.56** | **5.63** |

**生成质量评估（Tab.2）**:

| 方法 | FID | CLIP Score |
|------|-----|-----------|
| 原始 SD | 16.71 | 31.09 |
| ESD-u-1 Unlearn | 16.01 | 30.32 |
| ESD-u-1 Meta-Unlearn | 16.98 | 30.20 |
| UCE Unlearn | 17.59 | 31.01 |
| UCE Meta-Unlearn | 19.20 | 31.25 |

Meta-Unlearn 在 FID/CLIP 上与对应 Unlearn 方法接近，不会显著影响良性生成质量。

### 消融实验

**SDXL 版权移除（Fig.2 定性结果）**:

| 场景 | 配置 | 效果 |
|------|------|------|
| SpongeBob/Snoopy | ESD-u-1 Unlearn + FT 100步 | 重新生成版权角色 |
| SpongeBob/Snoopy | **ESD-u-1 Meta + FT 100步** | **仍无法生成版权角色** |
| Thomas Kinkade/Kelly McKernan 风格 | ESD-u-1 Unlearn + FT 100步 | 重新学会风格 |
| Thomas Kinkade/Kelly McKernan 风格 | **ESD-u-1 Meta + FT 100步** | **仍无法复现风格** |

**梯度夹角验证（Fig.6）**:

- 在元遗忘训练过程中，$\nabla_\theta \mathcal{L}_{DM}(\theta; \mathcal{D}_{FT})^\top \nabla_\theta \mathcal{L}_{DM}(\theta; \mathcal{D}_{retain})$ 的回归线呈下降趋势，证实梯度夹角逐渐超过 90°，实现自毁机制

### 关键发现

- **所有六种遗忘方法在加入元目标后，抗微调能力均大幅提升**。裸体评分在恶意微调后降低约 30-60%
- **良性微调（CLEAN 数据）后也更安全**：传统遗忘方法在 CLEAN 微调后仍可能产生有害内容（裸体评分 9-25%），元遗忘后仅 2-6%
- **版权/风格保护同样有效**：SpongeBob、Snoopy、Thomas Kinkade 等概念的元遗忘在 100 步微调后仍不可恢复
- **生成质量几乎无损**：FID 和 CLIP Score 差异很小，良性 prompt 生成质量保持

## 亮点与洞察

- **元学习思想在安全领域的巧妙应用** — 将 MAML 的"学会学习"转化为"学会不让别人学"，核心数学基础是控制梯度方向使其互相对抗。这个 idea 简洁优雅且通用性强
- **自毁机制的梯度解释** — 一阶近似揭示了元目标的两个作用：减小梯度范数（减速）+ 旋转梯度方向（自毁），提供了清晰的理论直觉
- **即插即用的设计** — 仅需在任何遗忘方法基础上添加几行元目标代码，不改变原方法的工作流程，实用性极高

## 局限与展望

- 元训练需要模拟微调过程（内层循环），带来额外计算开销（二阶梯度）
- 仅在最多 300 步微调下评估，更长时间微调是否仍能抵抗有待验证
- RECE 方法的元遗忘改进幅度相对较小，说明闭式解方法的元优化空间有限
- 未讨论针对知道元遗忘机制的自适应攻击者（如先恢复梯度对齐）
- 元目标中的超参数（$\zeta$、$\gamma_1$、$\gamma_2$、内步数 M）需要调节

## 相关工作与启发

- **vs ESD/SDD**: 标准遗忘方法在微调 100 步后裸体评分可达 20-57%，说明遗忘并不持久；Meta-Unlearn 将这些数字降低到 10-45%
- **vs AdvUnlearn**: AdvUnlearn 通过对抗训练增强鲁棒性，但仍可被微调攻破；Meta-Unlearn 从根本上改变了参数空间的梯度景观
- **vs RECE**: RECE 通过迭代构造新擦除嵌入，本身已是最强遗忘方法之一（遗忘前裸体评分仅 4.93%），Meta-Unlearn 在其基础上进一步降至 4.23%

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 元学习+机器遗忘的结合非常新颖，自毁机制的梯度分析elegant
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 6 种遗忘基线、2 个基础模型、3 种攻击场景、版权/风格/安全三类任务
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，Fig.1 的直觉解释非常好
- 价值: ⭐⭐⭐⭐⭐ 解决了机器遗忘领域的核心实际问题，通用性强，有部署价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] When Are Concepts Erased From Diffusion Models?](../../NeurIPS2025/image_generation/when_are_concepts_erased_from_diffusion_models.md)
- [\[CVPR 2025\] Memories of Forgotten Concepts](../../CVPR2025/image_generation/memories_of_forgotten_concepts.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[NeurIPS 2025\] Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning](../../NeurIPS2025/image_generation/towards_resilient_safety-driven_unlearning_for_diffusion_models_against_downstre.md)
- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](../../NeurIPS2025/image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)

</div>

<!-- RELATED:END -->
