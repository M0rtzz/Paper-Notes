---
title: >-
  [论文解读] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers
description: >-
  [ICCV 2025][图像生成][端到端训练] 提出 REPA-E，首个成功实现 VAE 与潜在扩散模型端到端联调的训练方案，通过表征对齐(REPA)损失而非扩散损失来更新 VAE，训练速度提升 17-45 倍并达到 ImageNet 256 新 SOTA（FID 1.12）。
tags:
  - ICCV 2025
  - 图像生成
  - 端到端训练
  - VAE
  - 潜在扩散模型
  - 表征对齐
  - 训练加速
---

# REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers

**会议**: ICCV 2025  
**arXiv**: [2504.10483](https://arxiv.org/abs/2504.10483)  
**代码**: https://end2end-diffusion.github.io  
**领域**: 图像生成 / 扩散模型  
**关键词**: 端到端训练, VAE, 潜在扩散模型, 表征对齐, 训练加速

## 一句话总结
提出 REPA-E，首个成功实现 VAE 与潜在扩散模型端到端联调的训练方案，通过表征对齐(REPA)损失而非扩散损失来更新 VAE，训练速度提升 17-45 倍并达到 ImageNet 256 新 SOTA（FID 1.12）。

## 研究背景与动机

**领域现状**：潜在扩散模型（LDM）采用两阶段训练——先训练 VAE 再固定 VAE 训练扩散模型。REPA 通过将扩散模型中间表征与 DINO 等预训练特征对齐来加速训练。

**现有痛点**：(1) 两阶段训练意味着 VAE 的潜在空间未针对生成任务优化——VAE 为重建而训练，不一定是扩散模型的最佳输入空间；(2) 不同 VAE 存在不同问题：SD-VAE 潜在空间有高频噪声，而自训练的 IN-VAE 过度平滑；(3) 直接用扩散损失反传到 VAE 会导致潜在空间坍塌。

**核心矛盾**：深度学习的经验告诉我们端到端训练通常更优，但 LDM 中直接端到端训练会让扩散损失"hack"潜在空间变简单（方便去噪但损害生成质量），导致性能下降。

**本文目标**：找到一种有效的端到端训练方案，联合优化 VAE 和扩散模型以实现最优生成性能。

**切入角度**：分析发现 REPA 的表征对齐分数（CKNNA）与生成质量强相关，且其上限被 VAE 特征瓶颈制约——如果能通过端到端训练改善 VAE 特征，就能突破这个瓶颈。

**核心 idea**：不用扩散损失而用 REPA 损失来更新 VAE——REPA 损失鼓励 VAE 潜在空间与扩散模型特征共同向预训练视觉表征对齐，既避免了潜在空间坍塌，又自适应地改善了 VAE 的潜在空间结构。

## 方法详解

### 整体框架
总损失 $\mathcal{L} = \mathcal{L}_{\text{DIFF}}(\theta) + \lambda \mathcal{L}_{\text{REPA}}(\theta, \phi, \omega) + \eta \mathcal{L}_{\text{REG}}(\phi)$。扩散损失 $\mathcal{L}_{\text{DIFF}}$ 仅更新扩散模型参数 $\theta$（stop-gradient 阻止反传到 VAE）；REPA 损失同时更新扩散模型 $\theta$ 和 VAE $\phi$；VAE 正则损失保持重建能力。

### 关键设计

1. **REPA 损失端到端反传**:

    - 功能：通过表征对齐损失联合优化 VAE 和扩散模型
    - 核心思路：$\mathcal{L}_{\text{REPA}}(\theta, \phi, \omega) = -\mathbb{E}[\frac{1}{N}\sum_n \text{sim}(y^{[n]}, h_\omega(h_t^{[n]}))]$，其中 $y$ 是 DINO-v2 特征，$h_t$ 是扩散 Transformer 的中间隐状态。REPA 损失通过扩散模型反传到 VAE，鼓励 VAE 产生更有利于表征对齐的潜在空间
    - 设计动机：直接用扩散损失更新 VAE 会坍塌（鼓励简单化潜在空间），但 REPA 损失鼓励的是与预训练视觉特征对齐——这不会简化潜在空间，反而会改善其结构

2. **Batch-Norm 层做潜在空间归一化**:

    - 功能：在 VAE 和扩散模型之间提供可微的动态归一化
    - 核心思路：传统 LDM 用预计算的全局统计量归一化 VAE 输出。端到端训练中 VAE 持续更新导致统计量失效。用 BN 层的指数移动平均替代全局统计量，无需每步重新计算
    - 设计动机：VAE 参数更新后潜在空间分布变化，固定的归一化常数不再有效。BN 层提供了轻量的自适应归一化

3. **扩散损失 Stop-Gradient**:

    - 功能：防止扩散损失损害 VAE 潜在空间
    - 核心思路：扩散损失 $\mathcal{L}_{\text{DIFF}}$ 仅用于更新扩散模型参数 $\theta$，通过 stop-gradient 阻断对 VAE 参数 $\phi$ 的梯度
    - 设计动机：实验和分析表明扩散损失会鼓励低方差的简单潜在空间（更易去噪但生成质量差），必须阻断

### 损失函数 / 训练策略
三部分损失：(1) 扩散损失→只更新LDM；(2) REPA损失→同时更新LDM和VAE；(3) VAE正则（重建+KL+GAN+LPIPS）→保持VAE重建能力。

## 实验关键数据

### 主实验

| 方法 | 训练步数 | gFID↓ | 加速比 |
|------|---------|-------|-------|
| Vanilla SiT | 1.4M | 8.61 | 1× |
| REPA | 4M | 5.90 | ~1× |
| **REPA-E (400K)** | 400K | **4.07** | **17× vs REPA, 45× vs vanilla** |
| REPA-E (最终) | - | **1.12** (w/ CFG) | SOTA |

### 消融实验

| 配置 | gFID | 说明 |
|------|------|------|
| REPA-E (完整) | 最优 | REPA 损失端到端 |
| 用扩散损失端到端 | 性能下降 | 潜在空间坍塌 |
| 无 BN 层 | 不稳定 | 归一化失效 |
| 无 VAE 正则 | 重建退化 | 需要保持 VAE 重建能力 |
| 不同 VAE (SD-VAE/IN-VAE) | 均有显著提升 | 泛化性好 |

### 关键发现
- 端到端训练自适应改善了 VAE 潜在空间：SD-VAE 的高频噪声被平滑，IN-VAE 的过度平滑被增加细节——同一种方法针对不同问题自动调整
- CKNNA 对齐分数与 gFID 强相关（相关系数 > 0.9），验证了用对齐分数作为生成质量代理的合理性
- 端到端训练后的 VAE 可以作为原始 VAE 的直接替换，在其他训练设置和模型架构中也能提升生成性能

## 亮点与洞察
- **反直觉发现**：扩散损失不能用于端到端训练但 REPA 损失可以——这揭示了两种损失对潜在空间结构的对立影响，是深刻的理论洞察
- **17-45 倍训练加速**是非常实际的贡献，显著降低了大规模扩散模型训练的成本
- 端到端训练的 VAE 本身也变好了，可以作为改进的 tokenizer 独立使用

## 局限与展望
- 仅在 ImageNet 256×256 上验证，更高分辨率和更大数据集有待确认
- 需要额外训练 VAE 的正则化项（GAN判别器等），增加了一定复杂度
- 目前仅与 SiT 架构配合验证，与 DiT 等其他架构的兼容性待探索
- REPA 依赖 DINO-v2 等预训练特征，对齐质量受限于该模型

## 相关工作与启发
- **vs REPA (Yu et al.)**: REPA 只对齐扩散模型特征，不更新 VAE；REPA-E 通过端到端训练同时优化两者
- **vs LSGM (Vahdat et al.)**: LSGM 用变分下界+熵项防止坍塌但收敛慢；REPA-E 用 REPA 损失更高效
- **vs VA-VAE/DC-AE**: 这些工作优化 VAE 架构但仍是两阶段训练；REPA-E 首次实现真正的端到端

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次成功实现 VAE+LDM 端到端训练，发现扩散损失vs REPA损失对潜在空间的对立影响
- 实验充分度: ⭐⭐⭐⭐⭐ 多种 VAE、多种模型规模、训练速度+最终性能+VAE质量三方面验证
- 写作质量: ⭐⭐⭐⭐⭐ 三个关键洞察层层递进，PCA 可视化非常直观
- 价值: ⭐⭐⭐⭐⭐ FID 1.12 SOTA + 45 倍加速，对扩散模型训练有范式性影响

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](end-to-end_multi-modal_diffusion_mamba.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [\[ICCV 2025\] InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis](infgen_a_resolution-agnostic_paradigm_for_scalable_image_synthesis.md)
- [\[ICCV 2025\] Unlocking the Potential of Diffusion Priors in Blind Face Restoration](unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)

<!-- RELATED:END -->
