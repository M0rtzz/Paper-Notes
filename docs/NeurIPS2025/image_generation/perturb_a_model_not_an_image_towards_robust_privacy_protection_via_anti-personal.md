---
title: >-
  [论文解读] Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][反个性化扩散模型] 提出Anti-Personalized Diffusion Model (APDM)，首次将隐私保护从数据级（图像扰动）转移到模型级（参数更新），通过Direct Protective Optimization损失和Learning to Protect双路径优化策略，鲁棒地阻止扩散模型对特定主体的个性化，同时保持模型对其他主体的生成和个性化能力。
tags:
  - NeurIPS 2025
  - 图像生成
  - 反个性化扩散模型
  - 隐私保护
  - DPO损失
  - 模型级防御
  - 双路径优化
---

# Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.01307](https://arxiv.org/abs/2511.01307)  
**代码**: [GitHub](https://github.com/KU-VGI/APDM)  
**领域**: 扩散模型 / 隐私保护  
**关键词**: 反个性化扩散模型, 隐私保护, DPO损失, 模型级防御, 双路径优化

## 一句话总结

提出Anti-Personalized Diffusion Model (APDM)，首次将隐私保护从数据级（图像扰动）转移到模型级（参数更新），通过Direct Protective Optimization损失和Learning to Protect双路径优化策略，鲁棒地阻止扩散模型对特定主体的个性化，同时保持模型对其他主体的生成和个性化能力。

## 研究背景与动机

扩散模型的个性化技术（如DreamBooth、Custom Diffusion）允许用户用少量图像微调模型来生成特定人物或物体的图像。这带来了严重的隐私风险——恶意用户可能生成未授权的内容。

现有的数据投毒（data-poisoning）保护方法存在四个关键缺陷：

**不切实际的假设**：要求对所有个人图像（包括已分享的、新创建的）施加扰动，这在实践中不可能做到

**易被绕过**：即使图像被扰动，攻击者只需混入少量干净图像或施加简单变换（翻转、模糊）就能削弱保护效果

**用户负担重**：数据投毒需要技术专长，普通用户难以操作

**不适合服务商**：GDPR等隐私法规要求服务提供商承担隐私保护义务，但用户级的数据投毒方法天然不适合提供商侧部署

这些问题指向一个根本性的方向转变：从保护数据到保护模型。然而，简单地将现有数据级保护的损失函数应用到模型参数上是不可行的。

## 方法详解

### 整体框架

APDM通过直接更新预训练扩散模型的参数 $\theta \to \hat{\theta}$，使其在被攻击者再次个性化时无法成功生成目标主体。框架包含两个核心组件：Direct Protective Optimization (DPO) 损失和 Learning to Protect (L2P) 优化策略。

### 关键设计

1. **朴素方法的收敛不可能性分析**

   直觉上的保护损失是 $\mathcal{L}_{adv} = -\mathcal{L}_{simple}^{per} + \mathcal{L}_{ppl}$（最大化个性化损失 + 保持先验生成），但作者证明这**必然不收敛**：
   
   **Proposition 1**：$\mathcal{L}_{adv}$ 收敛到局部最小值的必要条件是 $\nabla_\theta \mathcal{L}_{simple}^{per}$ 和 $\nabla_\theta \mathcal{L}_{ppl}$ 方向一致。
   
   然而通过一阶Taylor展开分析得出矛盾要求：
   $$|\nabla_\theta \mathcal{L}_{simple}^{per}| < |\nabla_\theta \mathcal{L}_{ppl}| \text{ 且 } |\nabla_\theta \mathcal{L}_{ppl}| < |\nabla_\theta \mathcal{L}_{simple}^{per}|$$
   
   **Theorem 1**：两个梯度范数不可能同时满足彼此小于对方，因此 $\mathcal{L}_{adv}$ 无法有效收敛。这一不可能性结果直接驱动了新损失函数的设计。

2. **Direct Protective Optimization (DPO) 损失**

   受偏好优化启发，为每个待保护图像 $x_0^-$（负样本）配对一个通用类别生成的正样本 $x_0^+$，基于Bradley-Terry模型构建偏好概率：
   $$p(x_0^+ > x_0^-) = \sigma(r(x_0^+) - r(x_0^-))$$
   
   DPO损失定义为：
   $$\mathcal{L}_{DPO} = -\mathbb{E} \log \sigma(-\beta(r^+ - r^-))$$
   
   其中 $r^+ = \|\epsilon_\theta(x_t^+, t, c) - \epsilon\|_2^2 - \|\epsilon_\phi(x_t^+, t, c) - \epsilon\|_2^2$，$r^-$ 类似。$\phi$ 为冻结的预训练模型，$\beta$ 控制偏离程度。
   
   最终保护目标：$\mathcal{L}_{protect} = \mathcal{L}_{DPO} + \mathcal{L}_{ppl}$。

   动机：DPO显式引导模型学习"鼓励什么"和"抑制什么"，避免了朴素方法的梯度冲突问题。

3. **Learning to Protect (L2P) 双路径优化**

   个性化是迭代过程，静态保护不够鲁棒。L2P通过交替执行两条路径模拟未来个性化轨迹并自适应增强保护：
   
   - **个性化路径**：从当前保护状态 $\theta_j$ 模拟 $N_{per}$ 步个性化：$\theta'_{i+1} = \theta'_i - \gamma_{per}\nabla_{\theta'_i}\mathcal{L}_{per}$
   - **保护路径**：在每个中间状态 $\theta'_i$ 计算保护梯度 $\nabla_i = \nabla_{\theta'_i}\mathcal{L}_{protect}$，累积求和后更新保护模型：$\theta_{j+1} = \theta_j - \gamma_{protect}\sum_{i=1}^{N_{per}}\nabla_i$
   
   重复 $N_{protect}$ 次得到最终受保护模型 $\hat{\theta}$。动机：L2P使保护"预见"个性化的演变方向，实现轨迹感知的自适应防御。

### 损失函数 / 训练策略

- 优化器: AdamW, 学习率 $\gamma_{per} = \gamma_{protect} = 5\text{e-6}$
- DPO超参数: $\beta = 1$
- L2P参数: $N_{per} = 20$, $N_{protect} = 800$
- 推理: PNDM调度器, 20步
- 基于Stable Diffusion 1.5, 单张A6000 GPU约9小时

## 实验关键数据

### 主实验

保护性能（DreamBooth个性化，DINO↓表示与原始主体越不像越好，BRISQUE↑表示生成质量越差越好）：

| 方法 | 干净图像数 | DINO↓ (avg) | BRISQUE↑ (avg) |
|------|-----------|-------------|----------------|
| DreamBooth (无保护) | N | 0.6525 | 16.80 |
| AdvDM | 0 | 0.4999 | 24.06 |
| AdvDM | N-1 | 0.5596 | 23.83 |
| SimAC | 0 | 0.4411 | 27.69 |
| SimAC | N-1 | 0.6181 | 20.67 |
| **APDM** | **N** | **0.1167** | **50.50** |

APDM的DINO仅为0.1167（vs 最优基线0.4411），即使所有输入图像均为干净图像。

### 消融实验

| 配置 | DINO↓ (person) | DINO↓ (dog) | BRISQUE↑ (person) | BRISQUE↑ (dog) |
|------|---------------|-------------|-------------------|----------------|
| 无图像配对 | 0.2770 | 0.3487 | 27.32 | 29.87 |
| 有图像配对 | **0.1375** | **0.0959** | **40.25** | **60.74** |
| 无L2P | 0.4454 | 0.3689 | 24.70 | 30.62 |
| 有L2P | **0.1375** | **0.0959** | **40.25** | **60.74** |
| $N_{per}=5$ | 0.3371 | 0.1923 | 37.89 | 39.48 |
| $N_{per}=20$ | **0.1375** | **0.0959** | **40.25** | **60.74** |

### 关键发现

- APDM的保护效果远超所有数据投毒方法，DINO降低至0.1167（基线最优0.4411），提升约74%
- 数据投毒方法在仅1张干净图像混入时保护效果大幅降低（如SimAC从0.4411升至0.5181），而APDM完全不受数据混入影响
- L2P对性能至关重要：无L2P时DINO从0.1375劣化至0.4454
- 保护模型仍能良好地为其他主体（cat、sneaker、glasses）个性化，DINO平均0.6334 vs 原始DreamBooth的0.5991
- 生成质量保持：FID 28.85 vs 原始SD的25.98，CLIP 0.2853 vs 0.2878

## 亮点与洞察

- 从"保护数据"到"保护模型"的范式转变是概念性的重大突破，根本性解决了数据投毒的各种实际局限
- 朴素方法不收敛的理论证明为新损失函数设计提供了严格的数学基础
- L2P的双路径优化思路优雅地将对抗博弈"内化"到训练过程中，类似meta-learning的思想
- 保护与保留能力的平衡做得很好：阻止特定主体个性化的同时，模型的通用生成和其他主体个性化能力几乎不受影响

## 局限与展望

- 保护过程需要约9 GPU小时，成本较高
- 每个待保护主体需要独立执行保护流程
- 目前仅在SD 1.5和2.1上验证，更新的模型（SDXL、FLUX）待验证
- 对于预训练数据中已存在的主体信息，模型级保护的效果可能受限

## 相关工作与启发

- Diffusion-DPO为DPO损失的设计提供了直接灵感
- L2P的双路径优化与MAML等元学习方法有异曲同工之妙
- 模型级防御思路可扩展到其他生成模型的隐私保护场景

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次提出模型级反个性化框架，"不可能性"定理和L2P优化方案都有显著创新
- **实验充分度**: ⭐⭐⭐⭐ 多种主体、多种场景对比充分，消融完整，但缺少更多模型的验证
- **写作质量**: ⭐⭐⭐⭐⭐ 动机→理论分析→方法→实验的故事线非常流畅
- **价值**: ⭐⭐⭐⭐⭐ 解决了隐私保护中的核心实际痛点，对服务提供商合规有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [Anti-Tamper Protection for Unauthorized Individual Image Generation](../../ICCV2025/image_generation/anti-tamper_protection_for_unauthorized_individual_image_generation.md)
- [ObCLIP: Oblivious Cloud-Device Hybrid Image Generation with Privacy Preservation](obclip_oblivious_cloud-device_hybrid_image_generation_with_privacy_preservation.md)
- [Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models](where_and_how_to_perturb_on_the_design_of_perturbation_guidance_in_diffusion_and.md)
- [Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation](vicinity-guided_discriminative_latent_diffusion_for_privacy-preserving_domain_ad.md)
- [Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models](../../CVPR2025/image_generation/nearly_zero-cost_protection_against_mimicry_by_personalized_diffusion_models.md)

<!-- RELATED:END -->
