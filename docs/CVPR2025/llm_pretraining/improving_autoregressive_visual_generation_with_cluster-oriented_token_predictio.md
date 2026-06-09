---
title: >-
  [论文解读] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction
description: >-
  [CVPR 2025][预训练][自回归视觉生成] 提出 IAR，通过平衡 K-means 重排 VQGAN 码本使相似 embedding 具有相邻索引，配合簇导向交叉熵损失引导模型正确预测目标 token 所在的语义簇，在 LlamaGen 100M-1.4B 各规模上将训练时间减半且提升生成质量。
tags:
  - "CVPR 2025"
  - "预训练"
  - "自回归视觉生成"
  - "码本重排"
  - "簇导向损失"
  - "LlamaGen"
  - "训练效率"
---

# Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction

**会议**: CVPR 2025  
**arXiv**: [2501.00880](https://arxiv.org/abs/2501.00880)  
**代码**: [https://github.com/sjtuplayer/IAR](https://github.com/sjtuplayer/IAR)  
**领域**: LLM预训练  
**关键词**: 自回归视觉生成, 码本重排, 簇导向损失, LlamaGen, 训练效率

## 一句话总结

提出 IAR，通过平衡 K-means 重排 VQGAN 码本使相似 embedding 具有相邻索引，配合簇导向交叉熵损失引导模型正确预测目标 token 所在的语义簇，在 LlamaGen 100M-1.4B 各规模上将训练时间减半且提升生成质量。

## 研究背景与动机

**领域现状**：基于 LLM 的视觉生成方法先用 VQGAN 将图像量化为离散 token，再用 GPT 式自回归预测生成图像。LlamaGen、VAR 等方法直接借鉴 NLP 的 next-token prediction 范式。

**现有痛点**：文本生成中预测错误的 token 索引意味着输出了完全不同的词（语义无关），但图像生成中预测"错误"索引对应的 embedding 可能与目标 embedding 在特征空间中非常接近，解码后的图像几乎相同。标准交叉熵损失无差别地惩罚所有错误预测，没有利用这一视觉特有的连续性。

**核心矛盾**：文本域的离散性 vs 图像域的连续性——LLM 框架是为离散 token 设计的，直接应用于连续特征空间的图像代码会造成训练效率浪费。

**本文目标**：利用图像 embedding 的空间相关性，放宽预测目标从"精确命中"到"落在正确的语义簇内"。

**切入角度**：实验发现当 embedding 之间的"code distance"（最近邻排序距离）小于 12 时，解码图像几乎不可区分。这意味着只要预测的 token 在目标附近的语义簇内，生成质量就能保证。

**核心 idea**：重排码本使相似 embedding 有连续索引 → 引入簇级别的交叉熵损失 → 即使 token 索引预测错也高概率在正确簇内。

## 方法详解

### 整体框架

IAR 在标准 LlamaGen 框架上引入两个预处理和训练改进：(1) 训练前用平衡 K-means 重排 VQGAN 码本；(2) 训练时在原有 token 级交叉熵损失基础上增加簇级交叉熵损失。

### 关键设计

1. **码本重排（Codebook Rearrangement）**:

    - 功能：使码本中在特征空间相近的 embedding 拥有连续的索引
    - 核心思路：原始 VQGAN 码本中相邻索引的 embedding 毫无关联。理想情况下应找到使相邻 embedding 距离之和最小的排列（哈密顿路径问题，NP-hard）。松弛为聚类问题：用平衡 K-means 将 $N$ 个 embedding 分为 $n$ 个等大小簇（每簇 $m = N/n$ 个），同簇的 embedding 高度相似。分配过程中先处理离质心近的 embedding，确保每个簇大小不超过 $m$。最后将同簇 embedding 映射到连续索引 $[jm, (j+1)m)$。
    - 设计动机：重排后的码本使得"索引距离近"等价于"embedding 相似"，为后续簇级损失提供结构基础。

2. **簇导向交叉熵损失（Cluster-oriented Cross-entropy Loss）**:

    - 功能：引导模型首先预测正确的簇，放宽对精确 token 索引的要求
    - 核心思路：对输出概率分布 $\hat{Y}$，通过 LogSumExp 操作将同簇内的 token 概率聚合为簇级概率 $\hat{Y}_{C,j} = \sum_{i=jm}^{(j+1)m-1} \exp(\hat{Y}_i) / \sum_{i=1}^{N} \exp(\hat{Y}_i)$。然后计算簇级交叉熵 $\mathcal{L}_{CCE}$。最终损失为 $\mathcal{L} = \mathcal{L}_{TCE} + \lambda \mathcal{L}_{CCE}$。
    - 设计动机：簇的数量 $n$ 远小于码本大小 $N$，正确预测簇是更容易的任务。一旦簇正确，即使具体 token 不对，解码图像也高度相似。

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{TCE} + \lambda \mathcal{L}_{CCE}$，其中 $\mathcal{L}_{TCE}$ 是标准 token 级交叉熵，$\mathcal{L}_{CCE}$ 是簇级交叉熵。码本重排是一次性预处理，几乎不增加训练成本。

## 实验关键数据

### 主实验

| 模型 | 参数量 | FID↓ | IS↑ |
|------|--------|------|-----|
| LlamaGen-B | 111M | 5.46 | 193.6 |
| + IAR | 111M | **4.72** | **210.3** |
| LlamaGen-L | 343M | 3.80 | 248.3 |
| + IAR | 343M | **3.20** | **263.5** |
| LlamaGen-XL | 775M | 3.39 | 227.1 |
| + IAR | 775M | **2.89** | **256.2** |
| LlamaGen-XXL | 1.4B | 3.10 | 253.9 |
| + IAR | 1.4B | **2.70** | **277.8** |

### 消融实验

| 配置 | FID↓ | 训练 epoch |
|------|------|-----------|
| Baseline | 5.46 | 300 |
| + 码本重排 | 5.12 | 300 |
| + 码本重排 + 簇损失 | **4.72** | 300 |
| Baseline 达到 FID 4.72 | 4.72 | ~600 |

### 关键发现
- IAR 在所有参数规模（100M~1.4B）上一致提升性能，符合 scaling law
- 在相同 FID 目标下，IAR 将训练时间减半（300 epoch vs 600 epoch）
- 码本重排本身贡献约 30% 的提升，簇损失贡献约 70%
- 当 code distance < 12 时，解码图像的 MSE 和 LPIPS 变化极小，验证了利用 embedding 相关性的合理性

## 亮点与洞察
- **揭示了图像与文本的本质区别**：文本需要精确索引，图像需要的是嵌入空间中的近似位置，这一观察深刻且有指导意义
- **码本重排的零成本收益**：一次性的预处理步骤，不修改模型架构，不增加推理存本，即可获得可观的性能提升
- **可即插即用**：适用于任何基于 LLM 的视觉生成模型（自回归或 MIM），通用性强

## 局限与展望
- 平衡 K-means 的簇数 $n$ 和簇大小 $m$ 需要超参数调优
- 当前仅验证了无条件/类条件图像生成，文生图等更复杂场景未测试
- 重排依赖于固定的 VQGAN 码本，联合优化码本和重排可能更有潜力
- 未与 VAR（next-scale prediction）等非标准自回归方法结合验证

## 相关工作与启发
- **vs LlamaGen**: IAR 在 LlamaGen 基础上添加码本重排+簇损失，不改架构但大幅提升效率
- **vs VAR**: VAR 通过改变预测粒度（next-scale）提升效率，IAR 通过改变损失函数的宽容度提升效率，思路正交
- **label smoothing 的类比**：簇损失与 label smoothing 有相似之处（放松硬标签），但更有结构——只在语义相近的 token 间分配概率

## 评分
- 新颖性: ⭐⭐⭐⭐ 观察深刻，码本重排+簇损失的组合设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 多规模验证、训练效率分析、完整消融
- 写作质量: ⭐⭐⭐⭐ 分析到位，动机推导自然
- 价值: ⭐⭐⭐⭐⭐ 即插即用、scaling law 兼容，对 LLM 视觉生成社区价值大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](../../ACL2025/llm_pretraining/pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[CVPR 2025\] Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)
- [\[ACL 2026\] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](../../ACL2026/llm_pretraining/toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)
- [\[CVPR 2025\] MR-PLIP: Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](mr_plip_multi_resolution_pathology.md)

</div>

<!-- RELATED:END -->
