---
title: >-
  [论文解读] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models
description: >-
  [图像生成] FADE 提出邻接感知（adjacency-aware）的细粒度概念擦除框架，通过 Concept Neighborhood 识别语义邻近类别并设计 Mesh Modules（Erasing + Adjacency + Guidance 三重损失）…
tags:
  - "图像生成"
---

# Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models

## 一句话总结

FADE 提出邻接感知（adjacency-aware）的细粒度概念擦除框架，通过 Concept Neighborhood 识别语义邻近类别并设计 Mesh Modules（Erasing + Adjacency + Guidance 三重损失），在精确删除目标概念的同时保留语义相关概念的生成能力，相比 SOTA 方法在邻接保留性能上提升至少 12%。

## 研究背景与动机

- **核心问题**：文本-图像扩散模型（如 Stable Diffusion）在大规模数据上训练，不可避免地学到敏感、不当或受版权保护的概念。需要选择性地擦除这些概念而不重训模型
- **现有方法的关键缺陷**：当前遗忘方法（ESD、CA、FMN、SPM、Receler）在擦除目标概念时，会**附带损害语义相近的类别**（adjacency problem）
    - 例如：擦除"金毛猎犬"时，连同"拉布拉多"、"平毛猎犬"等相关品种的生成能力也被破坏
    - 这种"附带遗忘"在细粒度场景下尤为严重，因为类间差异微小
- **需求**：实现 $P_\theta(c_{tar}|x) \to 0$（目标擦除），同时 $P_\theta(\mathcal{A}(c_{tar})|x) \approx P_{\theta_{original}}(\mathcal{A}(c_{tar})|x)$（邻接保留）
- **动机**：将遗忘问题从粗粒度（locality）推进到细粒度（adjacency），这是一个此前未被形式化的重要研究方向

## 方法详解

### 整体框架

FADE 将模型知识组织为三个不相交子集：
- **Unlearning Set $\mathcal{D}_u$**：由目标概念 $c_{tar}$ 生成的图像
- **Adjacency Set $\mathcal{D}_a$**：由 Concept Neighborhood 识别的语义邻近类别
- **Retain Set $\mathcal{D}_r$**：无关概念，作为广泛泛化的检验

通过 Mesh Modules（基于 LoRA 的轻量适配器）在三个子集上联合优化三重损失。

### 关键设计

#### 1. Concept Neighborhood（邻接集构建）

- 对每个概念 $c \in \mathcal{C}$，用原始模型生成 $m$ 张图像
- 用预训练图像编码器 $\phi$ 提取特征，计算每类的平均特征向量 $\bar{\mathbf{f}}^c$
- 通过余弦相似度 $L(c_{tar}, c) = \langle \bar{\mathbf{f}}^{c_{tar}}, \bar{\mathbf{f}}^c \rangle / (|\bar{\mathbf{f}}^{c_{tar}}||\bar{\mathbf{f}}^c|)$ 排序
- 选择 top-K 最相似概念构成邻接集 $\mathcal{A}(c_{tar})$
- 理论支撑：证明了 k-NN 在潜在特征空间中收敛到最优朴素贝叶斯分类器（Theorem 1）

#### 2. Mesh Modules（三重损失设计）

采用 LoRA 形式的轻量适配器 $\theta_M^{\mathcal{U}}$，优化以下三个损失：

**Erasing Loss $\mathcal{L}_{er}$**：
$$\mathcal{L}_{er} = \max\left(0, \frac{1}{|\mathcal{A}|}\sum_{x \in \mathcal{A}}|\epsilon_{\theta_M^{\mathcal{U}}}^{c_{tar}} - \epsilon_\theta^x|^2 - \frac{1}{|\mathcal{D}_u|}\sum_{x \in \mathcal{D}_u}|\epsilon_{\theta_M^{\mathcal{U}}}^{c_{tar}} - \epsilon_\theta^x|^2 + \delta\right)$$
使目标概念的噪声预测远离原位置，同时约束与邻接集的偏移最小化（类似三元组损失）

**Guidance Loss $\mathcal{L}_{guid}$**：
$$\mathcal{L}_{guid} = |\epsilon_{\theta_M^{\mathcal{U}}}^{c_{tar}} - \epsilon_\theta^{c_{null}}|^2$$
将目标概念引导至"空"概念方向，无需指定替代概念

**Adjacency Loss $\mathcal{L}_{adj}$**：
$$\mathcal{L}_{adj} = \frac{1}{|\mathcal{A}|}\sum_{x \in \mathcal{A}}|\epsilon_{\theta_M^{\mathcal{U}}}^x - \epsilon_\theta^x|^2$$
正则化项，确保邻接概念在更新后的模型中保持与原始模型一致的噪声预测

#### 3. ERB 评价指标

$$\text{ERB} = \frac{2 \cdot A_{er} \cdot \hat{A}_{adj}}{A_{er} + \hat{A}_{adj} + \eta}$$
调和平均数形式兼顾擦除效果和邻接保留，是本文提出的新基准指标。

### 损失函数

$$\mathcal{L}_{FADE} = \lambda_{er}\mathcal{L}_{er} + \lambda_{adj}\mathcal{L}_{adj} + \lambda_{guid}\mathcal{L}_{guid}$$

## 实验关键数据

### 主实验表（Tab. 1 — 细粒度遗忘）

在 Stanford Dogs / Oxford Flowers / CUB 三个数据集上，每个数据集选 3 个目标类：

| 方法 | 平均 ERB Score |
|------|---------------|
| ESD (ICCV'23) | ~40 |
| FMN (CVPRw'24) | ~4 |
| CA (ICCV'23) | ~64 |
| UCE (WACV'24) | ~61 |
| SPM (CVPR'24) | ~69 |
| Receler (ECCV'24) | ~3 |
| **FADE (Ours)** | **~96** |

FADE 在所有 9 个目标类上 ERB 均 >94，而最强竞争者 SPM 仅 ~69。

### 粗粒度遗忘（Imagenette, Tab. 2）

| 方法 | Target Acc↓ | Others Acc↑ |
|------|------------|------------|
| ESD | 0.00-0.10 | 65-70 |
| FADE | 0.00 | **86-89** |

FADE 在完全擦除目标的同时，对其他类的保留准确率显著优于所有基线。

### 关键发现

1. FADE 在 ERB 指标上较 SOTA（SPM）提升 **至少 12%**，在某些类别上提升超 30%
2. Receler 虽擦除效果好（$A_{er}$=100%），但邻接保留极差（$\hat{A}_{adj}$<7%），ERB 仅 ~3
3. FADE 在 ImageNet-1k 的 4 个目标类上也展现出最高的鲁棒性，在结构相似度 >90% 时仍保持高邻接准确率
4. 在 I2P 不当内容擦除实验中，FADE 同样相比基线有明显优势

## 亮点与洞察

1. **形式化邻接感知遗忘**：首次将 adjacency 作为一个独立维度提出并形式化，填补了精确遗忘研究的空白
2. **Concept Neighborhood 的优雅设计**：用生成图像的特征空间相似度自动构建邻接集，无需人工标注或分类体系
3. **理论与实践结合**：k-NN 收敛到 Naive Bayes 的理论证明为 Concept Neighborhood 方法提供了数学基础
4. **轻量高效**：基于 LoRA 的 Mesh Modules 只更新少量参数，计算开销小

## 局限性与可改进方向

1. **K 值敏感性**：邻接集大小 K=5 是超参数，不同数据集可能需要不同设置
2. **依赖分类器**：评估擦除效果需要微调分类器，增加了实验复杂度
3. **多概念同时擦除**：当前框架主要针对单概念擦除，多概念联合擦除的交互效应未充分探索
4. **对抗鲁棒性**：未评估对 adversarial prompt 的抵抗能力

## 相关工作与启发

- **ESD（ICCV'23）**：负引导方式擦除概念→擦除能力强但邻接破坏严重
- **SPM（CVPR'24）**：轻量 Membrane 适配器→启发了本文的 Mesh Module 设计
- **Receler（ECCV'24）**：对抗鲁棒的概念擦除→擦除过度激进导致邻接崩溃
- **启发**：AI 安全中的"精准手术"理念——去除有害能力的同时不伤及无辜，需要显式建模被保护对象的边界

## 评分

⭐⭐⭐⭐ — 问题定义清晰、方法设计严谨、实验充分，首次系统性解决扩散模型遗忘中的邻接问题。ERB 指标的提出也为后续研究提供了统一评价标准。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs](finelip_extending_clips_reach_via_fine-grained_alignment_with_longer_text_inputs.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_for_unified_image_generation_and_understanding.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[CVPR 2025\] FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations](flipsketch_flipping_static_drawings_to_text-guided_sketch_animations.md)

</div>

<!-- RELATED:END -->
