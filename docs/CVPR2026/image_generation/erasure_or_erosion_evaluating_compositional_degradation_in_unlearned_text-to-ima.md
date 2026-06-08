---
title: >-
  [论文解读] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models
description: >-
  [CVPR 2026][图像生成][概念擦除] 本文系统评估了16种文本到图像扩散模型概念擦除（unlearning）方法在安全性（擦除成功率）与组合性生成能力之间的权衡，揭示了激进擦除策略在去除不良内容的同时严重破坏了模型的属性绑定、空间推理和计数能力，强调安全干预不应以牺牲模型语义逻辑为代价。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "概念擦除"
  - "组合性生成"
  - "文本到图像"
  - "扩散模型"
  - "遗忘学习"
---

# Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models

**会议**: CVPR 2026  
**arXiv**: [2604.04575](https://arxiv.org/abs/2604.04575)  
**代码**: 无  
**领域**: 扩散模型 / AI安全  
**关键词**: 概念擦除, 组合性生成, 文本到图像, 扩散模型, 遗忘学习

## 一句话总结
本文系统评估了16种文本到图像扩散模型概念擦除（unlearning）方法在安全性（擦除成功率）与组合性生成能力之间的权衡，揭示了激进擦除策略在去除不良内容的同时严重破坏了模型的属性绑定、空间推理和计数能力，强调安全干预不应以牺牲模型语义逻辑为代价。

## 研究背景与动机

1. **领域现状**：文本到图像扩散模型如 Stable Diffusion 在内容创作中取得巨大成功，但其训练数据中不可避免地包含不安全内容（如色情、版权材料）。由于重训练成本过高，后处理（post-hoc）概念擦除方法成为主流解决方案。

2. **现有痛点**：现有 unlearning 方法的评估几乎完全聚焦于"擦除成功率"这一单一指标——即目标概念是否被成功抑制。但一个输出纯黑图像的模型在技术上也能获得完美的擦除分数，这说明单一指标评估本身就是不完备的。

3. **核心矛盾**：擦除操作作用于模型的共享语义空间（cross-attention subspace），而组合性生成（属性绑定、空间关系、计数）恰恰依赖于这些共享表示。因此，针对特定概念的擦除很可能"附带伤害"到模型的组合能力。

4. **本文目标** 系统量化不同 unlearning 方法对组合性生成能力的影响程度，揭示安全性与实用性之间的权衡关系。

5. **切入角度**：作者认为组合性能力是模型生成能力的代理指标——如果 unlearning 破坏了属性绑定（如 "green banana"），说明它损坏了生成语法本身，而不仅仅是移除了特定概念。

6. **核心 idea**：通过 T2I-CompBench++ 和 GenEval 等组合性基准测试，首次系统揭示概念擦除方法在安全性与组合性之间存在一致的反比关系。

## 方法详解

### 整体框架
本文是一项实证评估研究（empirical study），不提出新算法，而是设计了一套双重评估框架。输入是在 Stable Diffusion 1.4 上应用了不同 unlearning 方法的模型；输出是每个方法在安全性维度和组合性维度上的综合评分。

### 关键设计

**1. 安全性维度：不只看"擦没擦掉"，还要看"有没有误伤邻居"。**

只用擦除成功率评估的盲点在于，一个永远输出黑图的模型也能拿满分。为此本文把安全性拆成两条线：一条衡量"擦得干不干净"，用 I2P 基准 top-200 提示词算 Unlearning Accuracy (UA)；另一条衡量"有没有伤到正常内容"，这是关键巧思——作者用 ChatGPT 把每条危险提示词改成语义邻近的中性版本（如 "a naked man" → "a man"），在这些中性提示词上用 BVQA 算 Retain Accuracy (RA)，再在 SIX-CD 中性子集上算 CLIP Score、用 FID 看整体保真度。中性提示词恰好落在被擦概念的语义邻域里，所以 RA 能精确分辨：这次擦除是手术刀式的精准移除，还是连带把底层概念一起改坏了。

**2. 组合性维度：拿"属性绑定 / 空间 / 计数"当模型语义是否完好的代理。**

擦除作用在 cross-attention 这个共享语义空间上，而组合能力恰好也靠这块共享表示，所以"误伤"很可能体现为组合能力的退化。本文用两个互补基准来量化：T2I-CompBench++ 覆盖颜色、形状、纹理、2D 空间、3D 空间、数量、非空间、复杂度 8 个维度，GenEval 覆盖单物体、双物体、颜色、位置、计数 5 个维度，且都剔除掉含被擦概念的提示词，保证测的是"纯净"的组合能力。如果模型连 "green banana" 的颜色和物体都绑不对，说明被破坏的是生成语法本身，而不只是少了某个概念——这正是把组合性当作语义完整性代理指标的逻辑。

**3. 覆盖 16 种方法：让结论跨技术流派成立，而非只对某一类有效。**

为避免结论被某种特定技术路线绑架，评估横跨四类主流 unlearning：全局参数微调（ESD、Salun、ADV）、局部层干预（UCE、SPM、MACE）、对抗正则化（RACE）、以及推理时方法（SAFREE）。所有方法都统一跑在 Stable Diffusion 1.4 backbone 上，确保性能差异来自擦除策略本身而非架构差异。

### 损失函数 / 训练策略
本文为评估研究，不涉及新的损失函数设计。所有被评估模型均在 Stable Diffusion 1.4 backbone 上实现，确保性能差异来自 unlearning 策略而非架构变化。

## 实验关键数据

### 主实验

**T2I-CompBench++ 组合性结果**:

| 方法 | Color | Shape | Texture | 2D-Spatial | Numeracy | Mean | 变化 |
|------|-------|-------|---------|------------|----------|------|------|
| SD 1.4 (基线) | 0.357 | 0.326 | 0.397 | 0.117 | 0.449 | 0.321 | - |
| UCE | 0.351 | 0.378 | 0.420 | 0.092 | 0.430 | 0.324 | +1.2% |
| SPM | 0.345 | 0.366 | 0.372 | 0.125 | 0.448 | 0.322 | +0.4% |
| ESD | 0.260 | 0.356 | 0.342 | 0.086 | 0.405 | 0.287 | -10.5% |
| Salun | 0.121 | 0.181 | 0.121 | 0.028 | 0.220 | 0.166 | -48.2% |
| EraseDiff | 0.010 | 0.017 | 0.011 | 0.000 | 0.043 | 0.052 | -83.7% |

**安全性 vs 保留能力**:

| 方法 | UA ↑ | RA (BVQA) | FID ↓ |
|------|------|-----------|-------|
| EraseDiff | 100% | 0.020 (-92.7%) | 73.11 |
| Scissorhands | 100% | 0.053 (-80.6%) | 49.49 |
| ACE | 99.5% | 0.233 (-14.7%) | 18.34 |
| UCE | 93.5% | 0.268 (-1.8%) | 18.24 |
| SPM | 59.0% | 0.265 (-2.9%) | 18.04 |

### 消融实验

**GenEval 组合性结果**:

| 方法 | Single | Two | Colors | Position | Counting | Mean |
|------|--------|-----|--------|----------|----------|------|
| SD 1.4 | 0.925 | 0.351 | 0.707 | 0.033 | 0.281 | 0.459 |
| ACE | 0.938 | 0.343 | 0.731 | 0.028 | 0.291 | 0.466 (+1.5%) |
| SPM | 0.947 | 0.323 | 0.702 | 0.035 | 0.309 | 0.463 (+0.9%) |
| EraseDiff | 0.056 | 0.003 | 0.019 | 0.000 | 0.003 | 0.016 (-96.5%) |
| Scissorhands | 0.044 | 0.005 | 0.027 | 0.000 | 0.003 | 0.016 (-96.5%) |

### 关键发现
- **2D空间关系最脆弱**：所有方法中，2D-Spatial 类别的平均降幅最大（-41.4%），表明布局敏感的组合在 unlearning 下特别脆弱
- **Shape 最鲁棒**：粗粒度几何结构（Shape）在多数方法下保持甚至提升，说明外观级和关系级线索比结构信息更容易被破坏
- **ACE 和 SPM 是最佳平衡方法**：在 GenEval 上仅有 ACE 和 SPM 能维持或超过基线性能，说明局部化/结构化编辑策略能更好保留组合能力
- **激进方法导致流形坍缩**：EraseDiff（FID=73.11）和 Scissorhands（FID=49.49）的图像流形完全坍塌，ADV 打破了 token 对齐，Salun 出现模式坍缩

## 亮点与洞察
- **组合性作为语义完整性的代理指标**：这个观察非常巧妙——如果模型无法正确绑定"green banana"中的颜色和物体，说明模型的生成语法被破坏了，而不仅仅是某个概念被移除。这为 unlearning 方法的评估提供了全新视角。
- **中性提示词探测策略**：将 I2P 提示词转化为中性版本来测试语义邻域的完整性，这种评估设计可以迁移到其他安全性相关的生成任务评估中。
- **量化发现的政策含义**：论文揭示了一个尴尬的现实——技术上安全但语义上残缺的模型不能被视为真正可信的，这对行业安全合规标准的制定有指导意义。

## 局限与展望
- **仅评估裸体擦除**：所有实验聚焦于 nudity removal 这一场景，其他类型概念（版权风格、暴力内容）的擦除是否展现同样的权衡尚未验证
- **仅基于 SD 1.4**：评估限于较老的 Stable Diffusion 1.4 架构，更新的 SDXL、DiT 等架构下的结论可能不同
- **缺乏修复方案**：论文很好地诊断了问题，但未提出解决方案——如何设计同时保证安全性和组合性的 unlearning 方法仍是开放问题
- **可能的改进思路**：设计组合性感知的 unlearning 目标函数，在擦除过程中加入组合性保留正则项

## 相关工作与启发
- **vs ESD**: ESD 通过最大化目标概念的生成损失实现擦除，UA=93%但组合性Mean下降10.5%，属于中等偏激进的方法
- **vs UCE**: UCE 在 cross-attention 层进行局部编辑，UA=93.5%但组合性几乎无损（+1.2%），是安全-实用平衡最佳的方法之一
- **vs SPM**: SPM 采用一维 adapter 方式，组合性保持最好（+0.4%）但 UA 仅 59%，属于保守策略
- 本文为后续 unlearning 研究提供了一个清晰的基准——任何新方法都应同时报告 UA 和组合性指标

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究 unlearning 对组合性的影响，视角新颖但非方法创新
- 实验充分度: ⭐⭐⭐⭐⭐ 16种方法、两个互补基准、多维度评估，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，论证链条完整，图表设计直观
- 价值: ⭐⭐⭐⭐ 对 unlearning 社区有重要警示价值，但缺乏解决方案稍显遗憾

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](../../NeurIPS2025/image_generation/evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
