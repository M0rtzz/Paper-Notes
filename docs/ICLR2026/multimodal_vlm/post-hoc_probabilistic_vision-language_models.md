---
title: >-
  [论文解读] Post-hoc Probabilistic Vision-Language Models
description: >-
  [ICLR 2026][多模态VLM][视觉语言模型] 提出一种免训练的后验（post-hoc）不确定性估计方法，对 CLIP/SigLIP 等 VLM 最后几层使用 Laplace 近似，解析推导余弦相似度的不确定性，在不确定性量化和主动学习中取得显著优于基线的效果。
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "视觉语言模型"
  - "不确定性量化"
  - "贝叶斯推断"
  - "Laplace近似"
  - "主动学习"
---

# Post-hoc Probabilistic Vision-Language Models

**会议**: ICLR 2026  
**arXiv**: [2412.06014](https://arxiv.org/abs/2412.06014)  
**代码**: 有（Project page）  
**领域**: Multimodal VLM / Uncertainty Quantification  
**关键词**: 视觉语言模型, 不确定性量化, 贝叶斯推断, Laplace近似, 主动学习

## 一句话总结

提出一种免训练的后验（post-hoc）不确定性估计方法，对 CLIP/SigLIP 等 VLM 最后几层使用 Laplace 近似，解析推导余弦相似度的不确定性，在不确定性量化和主动学习中取得显著优于基线的效果。

## 研究背景与动机

### 领域现状

**领域现状**：视觉语言模型（VLMs），如 CLIP 和 SigLIP，已在分类、检索、生成等任务中取得了巨大成功。这些模型的核心操作是将图像和文本分别映射到共享的潜在空间，然后使用**余弦相似度**（cosine similarity）评估匹配程度。

然而，这种确定性映射存在一个根本性问题：**无法捕获概念上的不确定性**（uncertainty over concepts）。具体而言：

**领域偏移（domain shift）**：当 VLM 用于下游任务时，训练域与目标域之间的差异导致预测不可靠，但模型无法表达其"不确信"的程度

**分布外样本（OOD）**：对于未见过的图像或概念，VLM 仍然输出单一的确定性嵌入，无法区分"确信正确"和"猜测性预测"

**安全关键应用**：在医疗诊断、自动驾驶等场景中，不确定性估计对于可靠决策至关重要

现有的不确定性估计方法通常需要：

### 现有痛点

**现有痛点**：重新训练整个模型（如 Monte Carlo Dropout、集成学习）

### 核心矛盾

**核心矛盾**：修改模型架构（如概率嵌入方法）

### 解决思路

**解决思路**：大量额外的计算资源

这些方法在大规模 VLM 上不切实际——CLIP 等模型在数十亿图文对上训练，重新训练成本极高。因此，一种**不需要额外训练**的后验不确定性估计方法具有极大的实用价值。

## 方法详解

### 整体框架

BayesVLM 把一个已经训练好的 CLIP/SigLIP 当成黑盒，只在它最后几层的权重上"贴"一层贝叶斯近似：用 Laplace 近似把这几层权重看成围绕预训练值的高斯分布，于是图文嵌入和它们的余弦相似度都从确定的点变成了带方差的随机变量。整个过程不动原模型一个参数，只需一次轻量校准就能在前向传播时额外吐出一个不确定性数值。

### 关键设计

**1. Laplace 后验近似：把"确定的权重"变成"高斯分布的权重"。**

要让模型表达不确定性，第一步是给权重一个分布而不是一个定值。BayesVLM 只对最后几层的权重 $\theta$ 这样处理，并用 Laplace 近似在预训练值附近做二阶展开，得到高斯后验 $p(\theta \mid D) \approx \mathcal{N}(\theta^*, \Sigma)$。这里 $\theta^*$ 直接取预训练权重（相当于 MAP 估计），协方差 $\Sigma$ 用 Fisher 信息矩阵的逆来近似。之所以选 Laplace 而不是重训练或集成，关键就在于它的展开中心正好落在已经训好的参数上——模型的预测能力一点不损失，只是顺手量出了"这个解附近曲率有多陡"，曲率平缓的方向自然对应更大的不确定性。

**2. 余弦相似度的不确定性解析推导：让概率穿过相似度公式而不靠采样。**

VLM 的打分是图文嵌入的余弦相似度 $s = \frac{f_I \cdot f_T}{\lVert f_I \rVert \lVert f_T \rVert}$，原本是个确定值。当末层权重变成高斯随机变量后，嵌入 $f_I$、$f_T$ 乃至 $s$ 也都成了随机变量。最直接的做法是 Monte Carlo——反复采样权重跑前向，但在大模型上太贵。BayesVLM 改为解析推导出 $s$ 的均值和方差，一次前向就能把不确定性算出来。这样既省掉了多次采样的开销，又避免了有限采样带来的估计噪声，得到的方差还能直接拿去做下游决策。

**3. 只处理最后几层：把贝叶斯的成本压到可接受的量级。**

如果对整个 VLM 做 Laplace 近似，Fisher 信息矩阵的规模会爆炸。BayesVLM 的取舍是只把末尾一两层视为随机、其余 feature extractor 部分的后验不确定性近似当成零。这背后有两层考虑：一是计算上，Fisher 矩阵从全模型缩到末层，校准和推理才负担得起；二是任务上，下游适配主要发生在最后几层，把不确定性集中在这里既抓住了主要矛盾，消融也证实仅 1–2 层就足以给出高质量的不确定性。

**4. 免训练校准：即插即用地接到任何预训练 VLM 上。**

整套方法是后验（post-hoc）的——不微调、不改架构，只需要一次校准来估计 Fisher 信息矩阵（对角近似即可，Kronecker 分解更佳），而且这一步在少量数据上就能完成。校准结束后模型参数与原始权重完全一致，分类、检索性能不受任何影响，却凭空多出了一路不确定性输出。正是这个特性让它能直接套到现成的 CLIP、SigLIP 上，而不必为每个新模型重新付出训练代价。

### 损失函数 / 训练策略

方法本身不引入任何训练损失：校准阶段只在少量数据上计算 Fisher 信息矩阵的对角或 Kronecker 近似；推理阶段一次前向得到嵌入后，借 Laplace 近似解析地算出余弦相似度的均值与方差，同时输出点预测和不确定性估计。

## 实验关键数据

### 主实验

论文在两个主要应用场景中验证方法的有效性：

**不确定性量化（Uncertainty Quantification）**

| 设置 | 指标 | BayesVLM | 确定性基线 | 优势 |
|------|------|---------|----------|------|
| ID 数据 | 校准误差 (ECE) | 显著改善 | 过度自信 | 校准更好 |
| OOD 检测 | AUROC | 提升明显 | 无不确定性 | 能识别 OOD |
| 领域偏移 | 预测可靠性 | 更稳健 | 性能下降 | 提供可靠的不确定性信号 |

**主动学习（Active Learning）**

| 数据集 | 指标 | BayesVLM | 随机采样 | 其他基线 |
|--------|------|---------|---------|---------|
| 多个下游任务 | 样本效率 | **最高** | 基准线 | 中等 |
| 标注预算受限 | 准确率 | **最优** | 较差 | 次优 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 处理层数 | 不确定性质量 | 仅最后 1-2 层即可获得良好效果 |
| Fisher 矩阵近似方式 | 校准质量 | 对角近似已足够，Kronecker 分解效果更好 |
| 不同 VLM 骨架 | 通用性 | CLIP 和 SigLIP 上均有效 |

### 关键发现

- **校准良好**：BayesVLM 提供的不确定性估计具有良好的校准性——模型预测"不确信"时确实更可能出错
- **可解释性**：不确定性估计具有直觉上的可解释性——模棱两可或分布外的样本获得更高不确定性
- **主动学习高效**：基于不确定性的样本选择显著优于随机采样，在标注预算有限时价值尤其突出
- **不影响原始性能**：作为后验方法，不修改模型参数，不降低原有的分类/检索性能
- **计算高效**：解析推导避免了 Monte Carlo 采样，推理开销极小

## 亮点与洞察

- **问题选择精准**：VLM 的不确定性估计是一个被忽视但极其重要的问题，特别是在安全关键应用中
- **方法设计简洁**：不需要重新训练、不需要修改架构、不需要大量额外计算，真正的"即插即用"
- **理论-实用平衡**：Laplace 近似有坚实的理论基础，同时解析推导保证了计算效率
- **余弦相似度的概率化处理**：将确定性的余弦相似度转化为具有不确定性的随机变量，是一个优雅的理论贡献
- **下游应用多样**：同时展示了在不确定性量化和主动学习两个实际场景中的价值

## 局限与展望

- **近似质量**：Laplace 近似假设后验为高斯分布，在高维空间中可能不够准确
- **仅处理最后几层**：忽略了 VLM 更深层的不确定性传播，可能低估总体不确定性
- **Fisher 矩阵计算**：对于非常大的模型，即使是对角近似也可能有一定计算开销
- **评估基准有限**：不确定性估计的评估缺乏统一标准，不同数据集上的表现可能差异较大
- **面向分类/检索场景**：未验证在生成式 VLM（如 LLaVA、GPT-4V）上的适用性
- **自回归生成**：方法适用于 CLIP 类的双编码器架构，对于自回归 VLM 架构需要进一步扩展

## 相关工作与启发

- **CLIP** (Radford et al., 2021)：最具代表性的 deterministic VLM，本文方法的主要应用对象
- **SigLIP** (Zhai et al., 2023)：CLIP 的改进版本，使用 Sigmoid 损失，同样适用于本方法
- **Laplace 近似**：经典的贝叶斯近似方法，近年来在深度学习中重新受到关注（Laplace Redux, Daxberger et al., 2021）
- **Monte Carlo Dropout** (Gal & Ghahramani, 2016)：通过 Dropout 近似贝叶斯推理，但需要多次前向传播
- **概率嵌入** (Kirchhof et al., 2023)：将嵌入建模为分布而非点，但需要重新训练
- **主动学习** (Settles, 2009)：基于不确定性的样本选择是主动学习的经典策略

**启发**：后验方法是将贝叶斯不确定性引入大规模预训练模型的务实路径。这一思路可以推广到其他预训练模型（如 LLM、音频模型）的不确定性估计中。余弦相似度的概率化可能催生新的基于不确定性的检索和匹配算法。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 后验 Laplace 近似不新，但在 VLM 余弦相似度上的解析推导是新的贡献
- 实验充分度: ⭐⭐⭐⭐ — 不确定性量化和主动学习双场景验证，多 VLM 骨架测试
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，方法描述简洁易懂
- 价值: ⭐⭐⭐⭐ — 解决了 VLM 部署中的实际需求，安全关键应用前景广阔

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](../../ICML2026/multimodal_vlm/from_seeing_to_thinking_decoupling_perception_and_reasoning_improves_post-traini.md)
- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](../../CVPR2025/multimodal_vlm/post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[ICLR 2026\] GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](../../CVPR2026/multimodal_vlm/fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[ICLR 2026\] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)

</div>

<!-- RELATED:END -->
