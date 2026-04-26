---
title: >-
  [论文解读] Mass Concept Erasure in Diffusion Models with Concept Hierarchy
description: >-
  [AAAI2026][图像生成][概念擦除] 提出基于supertype-subtype概念层级的分组擦除策略和Supertype-Preserving LoRA (SuPLoRA)，通过冻结down-projection矩阵（正交于supertype子空间）仅训练up-projection矩阵，在大规模多领域概念擦除中实现擦除效果与生成质量的最优平衡。
tags:
  - AAAI2026
  - 图像生成
  - 概念擦除
  - 扩散模型
  - LoRA
  - 概念层级
  - 安全生成
---

# Mass Concept Erasure in Diffusion Models with Concept Hierarchy

**会议**: AAAI2026  
**arXiv**: [2601.03305](https://arxiv.org/abs/2601.03305)  
**作者**: Jiahang Tu, Ye Li, Yiming Wu, Hanbin Zhao, Chao Zhang, Hui Qian (浙江大学)  
**代码**: [GitHub](https://github.com/TtuHamg/SuPLoRA)  
**领域**: image_generation  
**关键词**: 概念擦除, 扩散模型, LoRA, 概念层级, 安全生成  

## 一句话总结

提出基于supertype-subtype概念层级的分组擦除策略和Supertype-Preserving LoRA (SuPLoRA)，通过冻结down-projection矩阵（正交于supertype子空间）仅训练up-projection矩阵，在大规模多领域概念擦除中实现擦除效果与生成质量的最优平衡。

## 背景与动机

### 问题背景
扩散模型（如Stable Diffusion）从大规模未过滤数据集中学习到不良概念（版权材料、攻击性内容、敏感个人信息），即使数据清洗后仍可能生成不安全内容。概念擦除(concept erasure)方法通过fine-tuning抑制特定概念生成。

### 已有工作不足
- **参数效率低**：每个擦除概念需要独立的fine-tune参数集，参数量随概念数线性增长（如MACE擦除64个概念需198MB）
- **生成质量退化**：反复擦除会抑制不仅特定于个体、还对supertype概念（如"人"）至关重要的视觉特征
- **跨域干扰**：擦除一个领域的概念会意外损害另一个领域的生成能力
- **缺少挑战性评测**：现有benchmark仅擦除单一类别概念

### 核心动机
利用擦除概念间的语义关系——构建层级结构，将语义相似概念分组共享参数擦除，同时通过理论保证的子空间约束保护supertype概念的生成能力。

## 核心问题

1. 如何在大规模概念擦除中同时保持参数效率和生成质量？
2. 如何防止擦除subtype概念时对supertype概念生成能力的退化？
3. 如何构建跨领域（名人+物体+色情内容）的统一擦除框架？

## 方法详解

### 概念层级构建（Sec 3.1）
利用CLIP计算概念间语义相似度 → 聚类 → GPT-4生成supertype标签：
- 例：{jay, macaw, bald eagle} → supertype "bird"
- 例：{Adam Driver, Adriana Lima, ...} → supertype "person"
- 层级关系：$\mathcal{G}_j = \{c_i^t \in \mathcal{C}^t \mid g(c_i^t) = c_j^p\}$

### 分组抑制（Sec 3.2）
基于MACE的注意力抑制，但在**supertype级别**而非个体概念级别操作，同组概念共享一组LoRA参数。

擦除损失——最小化概念token对相关区域的注意力：

$$\mathcal{L}_{\text{attn}} = \mathbb{E}_{c_i \in \mathcal{G}_j, t, l}\left[\|\boldsymbol{\alpha}_{c_i}^{t,l}(\mathbf{A}_j) \odot \mathbf{M}_{c_i}\|_F^2\right]$$

扩散正则化——在非擦除区域保持去噪能力：

$$\mathcal{L}_{\text{Diff}} = \mathbb{E}_{c_i \in \mathcal{G}_j, t, \boldsymbol{\epsilon}}\left[\|(1 - \mathbf{M}_{c_i}) \odot (\boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{z}_t, t, \mathcal{T}_{c_i} | \mathbf{A}_j))\|_2^2\right]$$

总损失：$\mathcal{L} = \mathcal{L}_{\text{attn}} + \lambda \mathcal{L}_{\text{Diff}}$

### SuPLoRA设计（Sec 3.3）

**关键理论推导**：对比直接修改 $\mathbf{W}$ 与仅训练 $\mathbf{A}_j$（冻结 $\mathbf{B}_j$）的效果差异。

直接修改 $\mathbf{W}$ 对擦除矩阵的更新为：

$$\Delta_{\mathbf{W}}\mathbf{W}' = -\alpha \frac{\partial \mathcal{L}}{\partial \mathbf{o}_j}\mathbf{h}_j^T$$

仅训练 $\mathbf{A}_j$（冻结 $\mathbf{B}_j$）对擦除矩阵的更新为：

$$\Delta_{\mathbf{A}_j}\mathbf{W}' = \Delta_{\mathbf{W}}\mathbf{W}' \cdot \mathbf{B}_j^T\mathbf{B}_j$$

**核心洞察**：训练 $\mathbf{A}_j$ 等价于在 $\mathbf{B}_j^T\mathbf{B}_j$ 定义的子空间 $\mathcal{S}_j^\perp$ 内修改权重。若 $\mathcal{S}_j^\perp$ 正交于supertype梯度子空间 $\mathcal{S}_j$，则擦除更新不干扰supertype生成。

**$\mathbf{B}_j$ 初始化**：
1. 收集supertype概念描述的文本嵌入 $\mathbf{H}_{S_j}$
2. SVD分解得到supertype梯度子空间 $\mathcal{S}_j = \text{span}\{\mathbf{u}_{1,j}, ..., \mathbf{u}_{r,j}\}$
3. 计算正交补空间 $\mathcal{S}_j^\perp$（null space of $\mathcal{S}_j$）
4. 将 $\mathbf{B}_j$ 设置为 $\mathcal{S}_j^\perp$ 的基，冻结 $\mathbf{B}_j$，仅训练 $\mathbf{A}_j$

### 知识蒸馏合并
$K$ 个SuPLoRA模块通过蒸馏合并为统一权重 $\mathbf{W}^*$：

$$\min_{\mathbf{W}^*} \underbrace{\mathbb{E}_{i,j}\|\mathbf{W}^*\mathbf{e}_{j,i}^t - (\mathbf{W} + \mathbf{A}_j\mathbf{B}_j)\mathbf{e}_{j,i}^t\|_2^2}_{\text{target alignment}} + \underbrace{\mathbb{E}_l\|\mathbf{W}^*\mathbf{e}_l^g - \mathbf{W}\mathbf{e}_l^g\|_2^2}_{\text{generality consistency}}$$

## 实验关键数据

### Benchmark设定
- 模型：Stable Diffusion v1.4，DDIM 50步
- 擦除范围：30名人 + 30物体 + 4色情概念 = 共64个概念
- 评估：ViT-L/16分类器（88.06% top-1）、GCD名人分类、NudeNet色情检测

### 主要结果（64概念同时擦除）

| 方法 | 名人Acc↓ | 物体Acc↓ | NN↓ | 域内名人Acc↑ | 域内物体Acc↑ | FID↓ | CLIP Score↑ | Supertype CLIP↑ | 存储(MB)↓ | 时间(min)↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ESD-u | 0.00% | 1.25% | 59 | 0.50% | 7.63% | 34.59 | 25.21 | 22.05 | 3379 | 2166 |
| UCE | 9.87% | 7.81% | 163 | 73.62% | 47.87% | 18.51 | 29.80 | 24.81 | 3379 | 218 |
| MACE | 6.25% | 9.17% | 158 | 78.50% | 50.63% | 18.36 | 30.04 | 25.51 | 198 | 20 |
| SPM | 10.00% | 65.00% | 639 | 78.50% | 63.50% | 21.15 | 30.59 | 26.00 | 218 | 20 |
| **Ours** | **7.50%** | **4.17%** | **121** | **83.38%** | **65.00%** | **17.92** | **30.68** | **26.09** | **154** | **18** |

### SuPLoRA消融

| 配置 | 域内名人/物体Acc↑ | FID↓ | CLIP Score↑ | Supertype CLIP↑ |
|------|:---:|:---:|:---:|:---:|
| Default LoRA（训练A+B） | 79.12%/56.50% | 18.18 | 30.18 | 25.19 |
| Default LoRA, 冻结随机B | 81.12%/59.87% | 18.13 | 30.65 | 26.08 |
| SuPLoRA, 训练B | 79.83%/57.01% | 18.23 | 30.25 | 25.22 |
| **SuPLoRA (完整)** | **83.38%/61.50%** | **17.94** | **30.66** | **26.21** |

### 概念数扩展实验（vs MACE）

| 设定(名人/物体) | 方法 | 域内物体Acc↑ | Supertype CLIP↑ |
|:---:|------|:---:|:---:|
| 0/10 | MACE | 92.87% | 26.58 |
| 0/10 | Ours | **93.38%** | **26.97** |
| 20/20 | MACE | 59.12% | 25.91 |
| 20/20 | Ours | **73.88%** | **26.33** |

20/20设定下物体域保留能力提升 **+14.76%**。

## 亮点

- **概念层级设计**：首次利用supertype-subtype语义结构组织擦除概念，将参数集从概念数$N$降至分组数$K$（64→约6组）
- **理论保证的子空间保护**：SuPLoRA通过梯度子空间正交性分析，证明冻结正交初始化的$\mathbf{B}_j$可防止supertype退化
- **跨域benchmark**：构建了首个同时跨名人+物体+色情三领域的大规模擦除评测
- **存储效率显著**：154MB vs MACE 198MB vs UCE 3379MB
- **训练速度最快**：18min vs MACE 20min vs UCE 218min

## 局限与展望

- **依赖共享supertype结构**：当擦除概念间缺乏语义关联时，分组效果减弱
- **两级层级限制**：仅构建了parent-child两层，更复杂的多级层级实验在附录但未充分验证
- **SD v1.4限制**：仅在Stable Diffusion v1.4上验证，未在SDXL、Flux等新架构测试
- **风格域未覆盖**：SD v1.4对艺术风格生成不稳定，故排除风格擦除评测
- **GPT-4依赖**：层级构建和prompt增强依赖GPT-4，引入外部API成本
- **对抗鲁棒性未评估**：未测试red-teaming攻击下的擦除持久性

## 与相关工作的对比

- **vs ESD**：ESD将擦除概念对齐到supertype（"grumpy cat"→"cat"），但擦除激进导致生成崩溃；本文保护supertype生成
- **vs MACE**：MACE为每个概念独立分配LoRA，存储线性增长；本文分组共享，减少至约1/4参数
- **vs UCE**：UCE通过封闭式解平衡擦除与保持，但存储开销巨大(3379MB)；本文仅154MB
- **vs SPM**：SPM通过anchoring loss保护无关概念，但色情内容检测极差（NN=639 vs 本文121）
- **vs ConceptPrune**：ConceptPrune剪枝"专家神经元"，仅验证10个类别；本文验证64个跨域概念
- **vs CE-SDWV**：推理时干预可被绕过（禁用模块即失效）；本文修改模型权重，不可逆

## 启发与关联

- SuPLoRA的子空间保护思路可推广至continual learning中的任务间干扰缓解
- 概念层级构建方法可用于其他需要结构化知识管理的模型编辑任务
- 分组擦除策略对大规模模型安全部署具有直接实用价值
- 梯度子空间正交性分析为LoRA微调中的任务冲突提供了理论工具

## 评分

- 新颖性: ⭐⭐⭐⭐ — 概念层级+子空间保护的组合设计新颖，理论分析有深度
- 实验充分度: ⭐⭐⭐⭐⭐ — 跨域benchmark、多基线对比、充分消融、扩展实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论推导严谨，公式与图表配合良好
- 价值: ⭐⭐⭐⭐ — 解决扩散模型安全部署的实际痛点，方法可扩展性强

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models](../../NeurIPS2025/image_generation/semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](../../CVPR2026/image_generation/neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](../../CVPR2026/image_generation/groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)

<!-- RELATED:END -->
