---
title: >-
  [论文解读] Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition
description: >-
  [CVPR 2026][图像生成][零样本学习] Flora 通过邻居感知语义校准实现稳健的骨架-语义跨模态对齐，并利用无噪声流匹配构建分布感知的开放式分类器，在零样本骨架动作识别上取得 SOTA，尤其在低样本训练场景中表现突出。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "零样本学习"
  - "骨架动作识别"
  - "流匹配"
  - "跨模态对齐"
  - "语义校准"
---

# Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition

**会议**: CVPR 2026  
**arXiv**: [2511.09388](https://arxiv.org/abs/2511.09388)  
**代码**: [https://github.com/cseeyangchen/Flora](https://github.com/cseeyangchen/Flora)  
**领域**: 图像生成  
**关键词**: 零样本学习, 骨架动作识别, 流匹配, 跨模态对齐, 语义校准

## 一句话总结

Flora 通过邻居感知语义校准实现稳健的骨架-语义跨模态对齐，并利用无噪声流匹配构建分布感知的开放式分类器，在零样本骨架动作识别上取得 SOTA，尤其在低样本训练场景中表现突出。

## 研究背景与动机

零样本骨架动作识别旨在让模型识别训练阶段未见过的骨架动作类别，这在现实应用中非常重要，因为收集覆盖所有动作类别的大规模数据集是不现实的。现有方法普遍遵循"对齐-分类"范式，但面临两个根本性问题：

1. **语义锚点不可靠**：无论是 LLM 生成的描述性语义（缺乏骨架显式引导而偏差）还是参数高效微调的语义（容易过拟合到已见类别），都会产生刚性、有缺陷的语义锚点，导致点对点的跨模态对齐不稳定——某些类别被正确对齐，其他类别却连带错位。
2. **分类器僵化**：生成式方法合成未见类特征来训练线性分类器，但决策边界是静态的，无法适应新类别；嵌入式方法通过余弦相似度匹配，但将特征压缩为单一向量导致信息损失和粗粒度分类。

核心矛盾在于：语义偏差和静态分类器共同限制了模型在零样本场景下的泛化能力。作者的核心洞察是：虽然单个语义锚点可能有偏差（如地图上标注有误的地标），但其与周围锚点的邻域关系仍然可靠。基于此，Flora 提出"先用邻域信息主动校准语义，再用流匹配桥接模态分布差异"的新范式。

## 方法详解

### 整体框架

Flora 要解决零样本骨架动作识别中两块互相拖累的短板：作为锚点的类别语义本身有偏差，而下游的分类器又是静态的，没法吸收这种偏差、也没法扩展到新类。它把整条流程拆成「学习」和「决策」两段来对症下药。学习阶段先不急着对齐，而是对每个类别的文本语义做一次邻域校准——借周围类别的上下文把这个有偏的语义点"拉正"，再把校准后的语义和骨架经一对双 VAE 投到同一潜在空间，用几何一致性目标缩小模态鸿沟。决策阶段则不再训练一个固定的分类器，而是把"语义分布→骨架分布"的搬运过程本身当成判别信号：用无噪声流匹配在两个潜在分布之间架一座传输桥，推理时谁的搬运路径最顺（速度场预测误差最小）就判给谁。

### 关键设计

**1. 邻居感知语义校准（Neighbor Semantic Attunement）：用邻域拓扑给有偏的语义点纠偏**

LLM 生成的类别描述往往单点不准——缺少骨架信号约束时它会"想当然"，把某个动作的语义放偏。Flora 的洞察是：单个锚点可能错，但它和周围锚点的相对关系是可靠的，就像地图上某个地标坐标标错了，参照周边稳定地标仍能定出大致方位。于是对每个语义特征 $\mathbf{F}_{a_y}$，先按余弦相似度挑出 Top-$k$ 个最近的类别语义，再以图聚合的方式把这些邻居的信息融回来，得到上下文化的语义 $\mathbf{O}_{a_y} = \mathbf{F}_{a_y} + \frac{\tau}{k} \sum_i w_i \mathbf{F}_i$，其中 $w_i$ 是相似度权重、$\tau$ 用来压住聚合强度防止过平滑。这一步把孤立的"语义点"撑成了一个带方向感的"语义区域"，后续对齐就不必去硬迁就一个本身就放偏的锚点。

**2. 几何一致性对齐（Geometric Consistency Alignment）：对齐分布几何而不是各自压向高斯先验**

校准后的语义还得和骨架落到同一空间里才能比较。Flora 沿用双 VAE 架构，但换掉了传统做法里那个"把两个模态分别 KL 正则到标准高斯"的目标——那样虽然两边都对齐到同一个先验，却也把类间结构压平了，丢了零样本最需要的可分性。这里改成直接约束两个模态分布的几何形状一致，即匹配它们的均值和方差：

$$\mathcal{L}_{Geo} = \|\mu_s - \mu_a\|_2^2 + \|\sigma_s^2 - \sigma_a^2\|_2^2$$

再配上模态内 / 跨模态的重构项一起优化。这样既把骨架和语义两个分布拉到一起、缩小了模态鸿沟，又不强行抹掉类别之间的差异，判别信息得以保留。

**3. 无噪声开放式流分类器（Noise-free Open-form Flow Classifier）：把"搬运语义到骨架"的路径当判别信号**

最后一步换掉了静态分类器。常规生成式方法合成未见类特征去训一个线性头，决策边界一旦定下就动不了、加新类得重训；嵌入式方法用余弦相似度，又把特征压成单一向量丢了细节。Flora 把流匹配从"从噪声里生成样本"搬到判别场景：源分布直接取语义潜在分布 $\mathcal{N}_a$（不近似成高斯、也不注入噪声，这正是"无噪声"的含义），目标分布取骨架潜在分布 $\mathcal{N}_s$，学一个 token 级的速度场 $v_\theta(z_t, t)$ 来刻画这条传输路径，并加对比正则把非配对类别的速度预测误差推远，让不同类别的路径彼此可分。关键在于：源和目标分布都是按类别区分的，所以每个类别的"搬运方式"天然不同，速度场本身就携带判别力。推理时对每个候选类别 $y$ 算它的速度误差 $\varepsilon_y = \|v_\theta(z_t^y, t) - v_y^*\|_2$，谁最小就判给谁——比如一个待识别骨架在 5 个未见类上分别得到误差，最贴合其真实分布传输路径的那个类误差显著低于其余四个，于是被选中。因为分类只依赖"给一段语义就能算路径"，新类别即插即用，无需重训分类器，且全程保留 token 级细粒度。

### 损失函数 / 训练策略

训练分两阶段：先独立优化对齐目标 $\mathcal{L}_{Align} = \mathcal{L}_{Re} + \lambda_{Align} \cdot \mathcal{L}_{Geo}$，把跨模态潜在空间打好；再冻结这部分参数，训练对比流目标 $\mathcal{L}_{ConFlow}$，它同时包含流匹配损失和系数为 $\lambda_{Flow}$ 的对比正则项。GZSL 推理时额外引入阈值 $\gamma$，用速度误差的比率来判断样本落在已见类域还是未见类域。

## 实验关键数据

### 主实验

| 数据集 | 分割 | 指标 | Flora | 之前SOTA | 提升 |
|--------|------|------|-------|----------|------|
| NTU-60 Xsub | 55/5 | ZSL Acc | 86.3% | 86.9% (Neuron/FS-VAE) | 持平 |
| NTU-60 Xsub | 55/5 | GZSL H | 77.4% | 75.7% (FS-VAE) | +1.7 |
| NTU-60 Xsub | 48/12 | ZSL Acc | 65.3% | 62.7% (Neuron) | +2.6 |
| NTU-60 Xsub | 48/12 | GZSL H | 60.5% | 59.1% (Neuron) | +1.4 |
| NTU-120 Xsub | 110/10 | ZSL Acc | 80.7% | 74.8% (InfoCPL) | +5.9 |
| NTU-120 Xsub | 110/10 | GZSL H | 66.1% | 63.3% (FS-VAE/Neuron) | +2.8 |
| NTU-120 Xsub | 96/24 | ZSL Acc | 66.4% | 65.1% (TDSM) | +1.3 |

### 消融实验（低样本训练，ZSL Acc）

| 配置 | NTU-60 55/5 (1%) | NTU-60 55/5 (10%) | NTU-120 110/10 (10%) |
|------|-------------------|--------------------|-----------------------|
| Flora | **84.2%** | **85.6%** | **75.4%** |
| FS-VAE | 83.2% | 84.1% | 62.1% |
| CADA-VAE | 76.6% | 76.9% | 39.1% |
| SynSE | 44.3% | 42.8% | 33.2% |

### 关键发现

- Flora 在 NTU-120 的 110/10 分割上表现特别突出（ZSL 80.7%），大幅领先之前 SOTA（InfoCPL 74.8%），说明在更大类别集、更少未见类的场景中优势明显
- 在仅使用 1% 训练数据时，Flora 仍能达到 84.2%（NTU-60 55/5），显示出极强的低样本泛化能力，这主要归功于流分类器不依赖合成特征训练静态分类器
- GZSL 设置中，Flora 在 H 指标上普遍拉高，表示 seen/unseen 之间的平衡更好

## 亮点与洞察

- **将流匹配从生成式推广到判别式**是本文最核心的创新。传统流匹配都用于从噪声生成样本，Flora 首次提出用流的速度场做分类，且完全无噪声、无条件约束，非常新颖
- **邻居语义校准**思路简单而有效——不是训练更复杂的对齐模型去适应有偏差的语义，而是先在语义端做"预处理"，利用邻域关系纠偏。这个思路具有很强的迁移性，可应用到任何跨模态对齐任务
- 流分类器天然支持开集扩展（新类别只需提供语义即可），且是 token 级别的细粒度判别，这两个特性在未来的开放世界识别中价值很高

## 局限与展望

- 语义仍然依赖 LLM 生成的文本描述，虽然做了校准但根本性的语义-动作 gap 未完全解决
- 流分类器推理时需对每个候选类别计算速度误差，类别数增大时推理开销线性增长
- 两阶段训练（先对齐再训流）未尝试端到端联合优化，可能有进一步提升空间
- 实验主要集中在 NTU/PKU 数据集，更大规模或跨域泛化未充分验证

## 相关工作与启发

- **vs STAR/PURLS**: 它们通过 LLM 直接生成精细语义描述再做对齐，Flora 则在语义校准环节引入邻域上下文，更鲁棒
- **vs FS-VAE**: 同样基于 Cross-VAE 但用频域分析，Flora 替换了 KL 正则为几何一致性并引入流分类器，H 指标更高
- **vs CrossFlow/FlowTok**: 这些跨模态流匹配工作仍使用 KL 散度正则化源分布为近似高斯，Flora 完全去除了噪声和条件约束
- 流分类器的思路可启发多模态检索、跨模态匹配等任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将流匹配推广到判别式零样本分类场景，且无噪声无条件约束的设计非常优雅
- 实验充分度: ⭐⭐⭐⭐ 三个数据集多种分割协议，低样本实验有力，但缺少跨域评估
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，类比（地图导航）生动，公式推导完整
- 价值: ⭐⭐⭐⭐ 在零样本骨架动作识别领域推进明显，流分类器的思路有较好的迁移潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition](../../ICCV2025/image_generation/bridging_the_skeleton_text_modality_gap_diffusion_powered_modality_alignment_for.md)
- [\[ECCV 2024\] Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition](../../ECCV2024/image_generation/idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](../../NeurIPS2025/image_generation/towards_robust_zero-shot_reinforcement_learning.md)
- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)

</div>

<!-- RELATED:END -->
