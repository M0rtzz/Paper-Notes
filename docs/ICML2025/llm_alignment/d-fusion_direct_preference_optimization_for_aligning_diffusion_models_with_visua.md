---
title: >-
  [论文解读] D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples
description: >-
  [ICML 2025][LLM对齐][扩散模型] 本文提出 D-Fusion 方法，通过 mask 引导的自注意力融合（Self-Attention Fusion）构建视觉一致的偏好数据对并保留去噪轨迹，解决了 DPO 训练扩散模型时因视觉不一致导致效果受限的问题…
tags:
  - "ICML 2025"
  - "LLM对齐"
  - "扩散模型"
  - "DPO"
  - "视觉一致性"
  - "自注意力融合"
  - "文本-图像对齐"
---

# D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples

**会议**: ICML 2025  
**arXiv**: [2505.22002](https://arxiv.org/abs/2505.22002)  
**代码**: [https://github.com/hu-zijing/D-Fusion](https://github.com/hu-zijing/D-Fusion)  
**领域**: LLM对齐/RLHF  
**关键词**: 扩散模型, DPO, 视觉一致性, 自注意力融合, 文本-图像对齐

## 一句话总结
本文提出 D-Fusion 方法，通过 mask 引导的自注意力融合（Self-Attention Fusion）构建视觉一致的偏好数据对并保留去噪轨迹，解决了 DPO 训练扩散模型时因视觉不一致导致效果受限的问题，在多种 RL 算法和 prompt 类型上显著提升了 prompt-image 对齐质量。

## 研究背景与动机
**领域现状**：扩散模型在文本到图像生成中取得了显著成功，但生成图像与文本 prompt 之间的对齐问题（misalignment）仍然严重，限制了实际应用。

**现有痛点**：近期研究将 DPO 引入扩散模型来增强对齐，但效果有限。核心原因在于 DPO 训练数据中存在**视觉不一致**问题：从不同噪声去噪得到的 high-preference 和 low-preference 图像在结构、风格、外观上差异巨大，模型难以分辨哪些因素与对齐正相关。

**核心矛盾**：在语言模型的 RLHF 中，可以对 token 级别做精细编辑得到一致的训练对；但在扩散模型中，手动编辑操作在像素级别进行，会**丢失逐时间步的去噪轨迹**，导致编辑后的图像无法用于 RL 训练。

**本文要解决的问题**：如何生成既**视觉一致**又**保留去噪轨迹**的 RL 可训练图像对？

**切入角度**：利用扩散模型 U-Net 中的自注意力机制，在去噪过程中逐步进行注意力融合，既保证生成图像与原始低偏好图的视觉一致性，又自然地保留了完整的去噪轨迹。

**核心 idea**：通过 cross-attention mask 定位对齐相关区域，在自注意力层将高偏好样本的对齐信息融合到低偏好样本中，生成可直接用于 DPO 训练的视觉一致样本。

## 方法详解

### 整体框架
D-Fusion 分为两个阶段：（1）采样阶段，通过 mask 引导的自注意力融合生成与 base image 视觉一致、与 reference image 同样对齐的 target image；（2）训练阶段，收集融合过程中的中间状态形成去噪轨迹，供 DPO/DDPO/DPOK 等 RL 算法训练。

### 关键设计

1. **Cross-Attention Mask 提取**:

    - 功能：从 reference image（高偏好图像）的去噪过程中自动提取与对齐相关的区域 mask
    - 核心思路：利用 cross-attention map 中 prompt 关键词对应的注意力分布来定位图像中与对齐相关的目标区域
    - 设计动机：手动标注 mask 成本高且不可扩展，cross-attention map 天然反映了 prompt 中各词与图像区域的对应关系，可自动化提取

2. **Self-Attention Fusion（自注意力融合）**:

    - 功能：在去噪过程的每个时间步，将 reference image 的自注意力特征在 mask 区域内融合到 base image 中
    - 核心思路：self-attention 控制图像的结构和风格，在对齐相关区域替换 self-attention 特征可以传递对齐信息，同时在非 mask 区域保持 base image 的原有外观
    - 设计动机：与直接像素编辑不同，self-attention fusion 是在去噪过程中逐步进行的，因此自然保留了完整的去噪轨迹
    - 与之前方法的区别：Prompt-to-Prompt 等方法是在不同 prompt 间转换图像，而 D-Fusion 是在同一 prompt 下将对齐信息从一张图迁移到另一张图

3. **去噪轨迹保留与 RL 训练**:

    - 功能：收集融合过程中每个时间步的中间噪声状态，组合形成 target image 的完整去噪轨迹
    - 核心思路：由于融合是逐时间步进行的，每一步的 (state, action) 对自然形成了 MDP 轨迹
    - 设计动机：DPO/PPO 等 RL 算法需要访问去噪轨迹来计算策略梯度，手动编辑得到的图像缺乏这些轨迹信息

### 损失函数 / 训练策略
- 采用标准的 Diffusion-DPO 损失函数，以 base image 为 low-preference、target image 为 high-preference 进行训练
- D-Fusion 作为数据构建方法，兼容 DPO、DDPO、DPOK 等多种 RL 算法
- DPO 训练中的偏好对由 (base image, target image) 构成，通过 CLIP 等评估器确定偏好顺序
- 在 DDPO 和 DPOK 中，target image 的去噪轨迹直接作为策略优化的正样本轨迹
- 训练时使用共享随机噪声确保 base image 和 target image 的视觉一致性
- 融合操作仅在采样阶段应用，训练阶段使用标准 RL 算法，不增加额外的训练成本

## 实验关键数据

### 主实验

| Prompt 类型 | 指标 | SD + DPO | SD + D-Fusion(DPO) | 提升 |
|------------|------|----------|-------------------|------|
| 物体行为 | CLIP Score | 较低 | 显著提升 | 明显 |
| 物体属性 | CLIP Score | 较低 | 显著提升 | 明显 |
| 位置关系 | CLIP Score | 较低 | 显著提升 | 明显 |

### 不同 RL 算法兼容性

| RL 算法 | 无 D-Fusion | 有 D-Fusion | 说明 |
|---------|------------|------------|------|
| DPO | 基准 | 提升 | 所有 prompt 类型均有效 |
| DDPO | 基准 | 提升 | 兼容策略梯度方法 |
| DPOK | 基准 | 提升 | 兼容混合方法 |

### 关键发现
- 视觉一致的训练对相比传统随机采样的训练对，显著提升了 DPO 在扩散模型上的效果
- D-Fusion 生成的 target image 不仅与 base image 视觉一致，而且与 reference image 具有同等对齐质量
- 方法对三种 prompt 类型（行为、属性、位置关系）均有效，说明了方法的通用性
- D-Fusion 可以与多种 RL 算法无缝结合，不限于 DPO
- 消融实验表明 mask 引导对融合质量至关重要——无 mask 的全图融合会破坏视觉一致性
- 融合时间步的选择也影响效果：早期时间步融合更多影响全局结构，后期影响细节

## 亮点与洞察
- **首次明确指出**扩散模型 DPO 训练中视觉不一致的核心问题，为该领域研究提供了新的视角
- 巧妙利用 self-attention 特性实现了"既融合对齐信息，又保留去噪轨迹"的两难目标
- 方法通用性强，可作为数据增强模块插入到任何 RL-based 扩散模型微调流程中
- 从语言模型 RLHF 的 token-level 精细化训练获得启发，类比创立了扩散模型的"fine-grained"一致性训练

## 局限与展望
- 论文主要在 Stable Diffusion 上验证，尚未推广到 SDXL、DiT 等更先进的扩散架构
- Mask 提取依赖于 cross-attention map 的质量，对某些复杂 prompt 可能不够精确
- 自注意力融合的计算开销比普通采样更大，可能影响训练效率
- 缓存文件较短（118行），论文中的具体数值实验细节未完全获取
- 未来可以探索在 attention 以外的模块进行融合，如 ResNet blocks
- 可探索自适应 mask 策略，根据 prompt 复杂度动态调整融合区域大小

## 相关工作与启发
- 与 Prompt-to-Prompt、Plug-and-Play 等注意力控制方法有技术关联，但目标不同（本文做对齐，它们做编辑）
- 受语言模型 RLHF 中句子级→token 级迁移的启发，类似思想迁移到图像的像素级→注意力级
- 为扩散模型对齐研究开辟了"数据一致性"这一新方向
- 与 Imagic、InstructPix2Pix 等图像编辑方法的区别在于：D-Fusion 保留了完整去噪轨迹
- 对偏好学习中数据质量的重要性提供了新的实证支持

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization](adhmr_aligning_diffusion-based_human_mesh_recovery_via_direct_preference_optimiz.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[ICML 2025\] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](smoothed_preference_optimization_via_renoise_inversion_for_aligning_diffusion_mo.md)
- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/llm_alignment/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](../../AAAI2026/llm_alignment/margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)

</div>

<!-- RELATED:END -->
