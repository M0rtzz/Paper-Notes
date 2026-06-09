---
title: >-
  [论文解读] Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models
description: >-
  [NeurIPS 2025][图像生成][注意力扰动引导] 提出 HeadHunter 框架和 SoftPAG 方法，将扩散模型中的注意力扰动粒度从层级细化到单个注意力头级别，首次发现不同注意力头控制不同视觉概念（结构、风格、纹理等），实现了更精准且可组合的生成引导。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "注意力扰动引导"
  - "注意力头"
  - "Transformer"
  - "细粒度控制"
  - "风格迁移"
---

# Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.10978](https://arxiv.org/abs/2506.10978)  
**代码**: [项目页面](https://cvlab-kaist.github.io/HeadHunter/)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 注意力扰动引导, 注意力头, 扩散 Transformer, 细粒度控制, 风格迁移

## 一句话总结

提出 HeadHunter 框架和 SoftPAG 方法，将扩散模型中的注意力扰动粒度从层级细化到单个注意力头级别，首次发现不同注意力头控制不同视觉概念（结构、风格、纹理等），实现了更精准且可组合的生成引导。

## 研究背景与动机

Classifier-Free Guidance (CFG) 是扩散模型生成质量的关键，但有两大局限：（1）仅适用于条件生成，无法用于无条件场景（如逆问题）；（2）容易降低多样性并导致过饱和。

注意力扰动引导（如 PAG）作为替代方案，通过扰动模型的注意力层构建隐式弱模型来引导生成。但现有方法面临核心问题——**"扰动应该施加在哪里"缺乏原则性方法**。

关键挑战在于架构差异：
- **U-Net** 有明确的瓶颈层（中间块）负责全局语义，可以轻松确定扰动位置
- **Diffusion Transformer (DiT)** 没有粗-细的层级结构，语义处理更均匀地分布在所有层中

作者的核心洞察来自一个简单但深刻的实验：在 DiT 中对**单个注意力头**施加 PAG 扰动后，不同头产生了截然不同的视觉效果——有的增强暗调、有的改变几何结构、有的影响色彩。这说明注意力头比整个层是更有意义的扰动单元。

## 方法详解

### 整体框架

方法包含两个互补组件：（1）HeadHunter —— 一个迭代式注意力头选择框架；（2）SoftPAG —— 一种连续可调的注意力扰动强度控制机制。

### 关键设计

1. **头级别扰动引导（Head-Level Perturbation Guidance）**: 不同于层级扰动对所有头统一操作，头级别引导选择性地扰动特定注意力头子集。给定选中的头集合 $\mathcal{S} = \{(l_1,h_1), \ldots, (l_m,h_m)\}$，将其注意力图替换为单位矩阵：
    $\mathbf{A}_{l,h}^{(\text{PAG})} = \mathbf{I} \quad \text{for } (l,h) \in \mathcal{S}$
   这种细粒度操作避免了层级扰动中因头间多义性而导致的质量退化。

2. **HeadHunter 迭代头选择框架**: 解决"如何自动选择与用户目标一致的注意力头"的问题。每轮包含三个阶段：

    - **生成阶段**: 对每个候选头 $(l,h)$ 施加扰动，用多组 prompt-seed 对生成样本
    - **评估阶段**: 用用户指定的目标函数 $\mathcal{O}$（如 PickScore）计算平均得分 $s_{(l,h)} = \frac{1}{M}\sum_{j=1}^M \mathcal{O}(\hat{x}_j, p_j)$
    - **扩展阶段**: 将 top-$k$ 头加入已选集合 $\mathcal{S}_{\text{final}}$
   
   迭代设计的关键优势：某些头单独使用时生成质量差，但在与已选结构性头组合后能有效增强特定风格（如温暖色调）。这类头在一次性评估中不可能被选中，但在迭代过程中其组合价值得以体现。

3. **SoftPAG（软扰动注意力引导）**: 通过线性插值在原始注意力图和单位矩阵之间提供连续可调的扰动强度：
    $\mathbf{A}_{l,h}^{(\text{SoftPAG})} = (1-u)\mathbf{A}_{l,h} + u\mathbf{I}, \quad u \in [0,1]$
   当 $u=1$ 退化为标准 PAG，$u=0$ 为无扰动。该方法解决了扰动过强导致的过平滑和过简化问题，找到质量增强与细节保留之间的"最佳平衡点"。

### 概念可组合性

不同头控制的视觉概念可以通过组合多个头来叠加。例如，组合光照头（L1,H10）和剪切变形头（L11,H15）会产生混合效果。但遵循边际效益递减规律——过多头的组合会导致过饱和。

## 实验关键数据

### 主实验：通用质量提升（SD3, MS-COCO 1K prompts）

| 方法 | PickScore ↑ | AES ↑ | HPS ↑ | ImReward ↑ |
|------|-------------|-------|-------|------------|
| Baseline（无引导） | 19.66 | 5.37 | 0.2147 | -0.591 |
| CFG (w=3.0) | 20.87 | 5.71 | 0.2924 | 0.844 |
| CFG (w=6.0) | 20.92 | 5.80 | 0.3046 | 1.063 |
| HeadHunter (w=3.0) | 20.70 | **5.92** | 0.2901 | 0.470 |
| CFG (3.0) + HeadHunter (3.0) | **20.92** | **5.93** | **0.3036** | 0.845 |

### 消融实验：头数量与质量关系

| 配置 | 关键发现 | 说明 |
|------|----------|------|
| k=6（仅 25% 头） | FID 已优于全层级扰动 | 证明紧凑头集合即可超越启发式层选择 |
| k=6→24 递增 | FID 持续改善（w=3.0 和 4.0 下） | 支持头效果的可组合性 |
| w=6.0 时 k>12 | FID 开始退化 | 高引导强度下过多头导致过饱和 |
| SoftPAG u<1.0 | 多数指标最优在 u<1.0 | 证明全替换（PAG）通常非最优 |

### 关键发现

- **层内多样性是层级扰动低效的根源**: 即使被认为"差"的层（如 L13），其中仍有个别头能产生高质量输出
- **可解释的头专化**: 特定头一致地控制剪切、暗度、蓝色调等视觉属性
- **HeadHunter 可作为即插即用模块**: 在现有 CFG 流水线上叠加 HeadHunter，进一步提升美学得分（AES 从 5.80 到 5.93）
- **风格导向搜索**: 通过 5 轮迭代（每轮选 3 个头），生成图像逐步强化目标风格（如金色暖光、线稿风格）

## 亮点与洞察

- **第一个头级别注意力扰动分析**: 揭示了扩散模型中注意力头的功能专化现象，具有重要的可解释性价值
- **"弱到强"的组合涌现**: 单独表现差的头在组合中可能贡献关键风格要素，这与 ensemble 学习中弱学习器的思想异曲同工
- **SoftPAG 的简洁优雅**: 仅需几行代码的线性插值即可提供连续可控的扰动强度，实用价值极高
- **与 CFG 正交且可叠加**: 不需额外训练，直接在推理时使用，且可与 CFG 协同工作

## 局限与展望

- HeadHunter 搜索过程需要多次生成和评估，计算开销非平凡（虽然只需搜索一次）
- 搜索到的头集合可能与模型架构和版本耦合，换模型需重新搜索
- SoftPAG 的最优 $u$ 值可能因 prompt 而异，目前用固定值
- 主要在 SD3 和 FLUX.1 上验证，在更多架构（如 DiT-XL, Pixart）上的普适性待确认

## 相关工作与启发

- PAG 最早将注意力扰动引导引入扩散模型，但仅在 U-Net 中间块操作
- SEG（Smoothed Energy Guidance）从能量视角解释扰动引导，但同样在层级操作
- 大语言模型中的注意力头专化研究（如 Transformer Circuits）为本文提供了理论启发
- ViT 中的 [CLS] token 设计与扩散模型中注意力头的分析方法可互相借鉴

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次在头级别分析扰动引导，发现具有深远意义
- **实验充分度**: ⭐⭐⭐⭐ 定量定性分析丰富，但缺少更多架构验证
- **写作质量**: ⭐⭐⭐⭐⭐ 动机清晰、分析深入、图示精美
- **价值**: ⭐⭐⭐⭐⭐ 即插即用、无需训练、可组合，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [\[NeurIPS 2025\] Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)
- [\[NeurIPS 2025\] Diffusion-Based Electromagnetic Inverse Design of Scattering Structured Media](diffusion-based_electromagnetic_inverse_design_of_scattering_structured_media.md)

</div>

<!-- RELATED:END -->
