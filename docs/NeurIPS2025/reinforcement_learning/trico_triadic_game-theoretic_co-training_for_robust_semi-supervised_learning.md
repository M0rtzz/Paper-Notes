---
title: >-
  [论文解读] TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning
description: >-
  [NeurIPS 2025][强化学习][半监督学习] 提出 TRiCo 框架，将半监督学习重构为教师-双学生-对抗生成器的三方博弈（Stackelberg 博弈），用互信息替代置信度做伪标签筛选，元学习教师自适应调节训练动态，在低标签场景下实现 SOTA 性能。 半监督学习（SSL）通过利用大量无标签数据来缓解标注成本高的…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "半监督学习"
  - "博弈论"
  - "协作训练"
  - "元学习"
  - "对抗扰动"
---

# TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.21526](https://arxiv.org/abs/2509.21526)  
**代码**: 暂无  
**领域**: 强化学习  
**关键词**: 半监督学习, 博弈论, 协作训练, 元学习, 对抗扰动

## 一句话总结

提出 TRiCo 框架，将半监督学习重构为教师-双学生-对抗生成器的三方博弈（Stackelberg 博弈），用互信息替代置信度做伪标签筛选，元学习教师自适应调节训练动态，在低标签场景下实现 SOTA 性能。

## 研究背景与动机

半监督学习（SSL）通过利用大量无标签数据来缓解标注成本高的问题，其中**协作训练（co-training）**通过让两个模型在互补视角上交换伪标签来减少确认偏差。但传统 co-training 在现实场景中存在三个核心局限：

**伪标签筛选不可靠**：传统方法用固定置信度阈值过滤伪标签，但 softmax 置信度的校准在训练早期和分布偏移下极不稳定。过度自信但错误的伪标签会在双视角间传播，导致语义坍塌。

**视角交互静态对称**：co-training 假设两个视角能力对称、且交互方式固定。但实际上模型容量、表征质量、学习速度天然异质，缺乏自适应调节机制会导致交互停滞甚至损害泛化。

**缺少难样本挖掘**：伪标签天然偏向高置信度的简单样本，模型容易过拟合到这些区域而忽略决策边界附近的不确定区域——而这些才是真正驱动鲁棒性的关键区域。

本文的核心 idea 是引入**第三方角色——教师**，将原本的双方交互升级为三方博弈，形成"教师主导调控、生成器制造挑战、学生在监督下协作学习"的闭环。

## 方法详解

### 整体框架

TRiCo 包含三个交互组件：

- **两个学生分类器** $f_1$, $f_2$：分别在两个冻结视觉编码器（DINOv2 和 MAE）提取的互补表征上训练，使用轻量 MLP head。
- **非参数对抗生成器** $G$：在嵌入空间做扰动以暴露决策边界弱点。
- **元学习教师** $\pi_T$：自适应控制伪标签筛选阈值和损失权重。

三者的交互被形式化为 **Stackelberg 博弈**：教师是领导者（优化泛化目标），学生和生成器是跟随者。

### 关键设计

1. **基于互信息的伪标签筛选**：不使用基于置信度的启发式方法，而用**互信息（MI）**度量认知不确定性。对每个输入做 $K$ 次 dropout 前向传播，估计预测分布的互信息：

$$\text{MI}(x^{(i)}) = H[\bar{p}^{(i)}(y)] - \frac{1}{K}\sum_{k=1}^{K} H[p_{\theta_k}^{(i)}(y)]$$

只有 $\text{MI} > \tau_{\text{MI}}$ 的样本才被接受用于交叉视角监督。MI 比置信度更能反映认知不确定性，尤其在训练早期和模糊样本上更鲁棒。交叉视角无监督损失为：

$$\mathcal{L}_{\text{unsup}} = \mathbb{E}_{x_u}[\ell(f_1(x_u^{(1)}), \hat{y}^{(2)}) + \ell(f_2(x_u^{(2)}), \hat{y}^{(1)})]$$

2. **熵驱动对抗生成器**：通过最大化预测熵+MI 的方式在嵌入空间构造对抗扰动：

$$\delta^{(i)*} = \arg\max_{\|\delta\|_\infty \leq \epsilon} [\mathcal{H}(f_i(x^{(i)}+\delta)) + \gamma \cdot \text{MI}(f_i(x^{(i)}+\delta))]$$

通过 FGSM/PGD 风格的梯度上升计算，无需训练生成器模型。对抗损失促使模型即使在高不确定区域也能做出确信预测：$\mathcal{L}_{\text{adv}} = \mathbb{E}[\mathcal{H}(f_1(x_g^{(1)})) + \mathcal{H}(f_2(x_g^{(2)}))]$。

3. **元学习教师**：教师参数 $\theta_T$ 包括 $\tau_{\text{MI}}$、$\lambda_u$、$\lambda_{\text{adv}}$（通过 sigmoid 约束到 $[0,1]$）。核心思想是"好的伪标签策略应导致学生在验证集上泛化更好"。通过单步梯度展开的元学习更新教师：

$$\theta_T \leftarrow \theta_T - \eta_T \cdot \nabla_{\theta_T} \mathcal{L}_{\text{sup}}(f_{\theta_S - \eta \nabla_{\theta_S} \mathcal{L}_{\text{unsup}}^{\theta_T}})$$

教师通过观察自身决策对学生泛化的影响来优化策略，从静态过滤器转变为主动策略学习者。

### 损失函数 / 训练策略

总损失为三部分之和：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sup}} + \lambda_u \mathcal{L}_{\text{unsup}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}$。学生用 SGD 优化，教师用元梯度下降更新。理论上证明了三方博弈的 Nash 均衡存在性（Theorem 1）。使用 ViT-B/16（DINOv2+MAE）作为冻结编码器，学生为两层 MLP。SGD+余弦退火，batch=64，训练 512 epochs。

## 实验关键数据

### 主实验

| 数据集 | 设置 | TRiCo | Meta Pseudo Label | FlexMatch | FixMatch |
|--------|------|-------|-------------------|-----------|----------|
| CIFAR-10 | 4k labels | **96.3** | 95.1 | 94.9 | 94.3 |
| SVHN | 1k labels | **94.2** | 93.5 | 92.7 | 92.1 |
| STL-10 | full labeled | **92.4** | 90.6 | 90.1 | 89.5 |
| ImageNet | 1% labels | **81.2** | 55.0 | 53.5 | 52.6 |
| ImageNet | 10% labels | **85.9** | 71.8 | 70.2 | 68.7 |
| ImageNet | 25% labels | **88.3** | 76.4 | 75.3 | 74.9 |

ImageNet 25% 标签下 TRiCo 达到 88.3%，接近全监督大模型性能。

### 消融实验

| 组件 | 准确率 | PGD鲁棒性 | 说明 |
|------|--------|----------|------|
| TRiCo 完整 | **95.9** | **82.1** | 全部组件协同 |
| MI 筛选 → 置信度 0.70 | 95.0 | 77.7 | MI 筛选优于置信度 |
| 固定教师参数 | 94.7 | 79.0 | 元学习调节必要 |
| 去掉生成器 | 94.2 | 78.9 | 对抗训练贡献约 1.7% |
| 随机噪声替代 | 70.5 | 66.4 | 熵引导扰动远优于随机 |
| 2-View Only (无教师) | 94.1 | 78.4 | 教师贡献约 1.8% |

### 关键发现

- MI 筛选在所有置信度阈值设置上都更优，尤其在训练早期更稳定。
- 教师参数（$\tau_{\text{MI}}, \lambda_u, \lambda_{\text{adv}}$）在训练中呈现平滑的自适应变化，早期保守、后期逐渐放开。
- Few-shot 设置（1/5/10-shot）下，TRiCo 优势进一步扩大（CIFAR-100 1-shot: 23.8 vs MCT 21.2）。
- T-SNE 可视化显示 TRiCo 的特征空间聚类更紧凑、类间分离更清晰。

## 亮点与洞察

- 将 SSL 从"双方协作"升级为"三方博弈"的视角是一个结构性创新——教师作为"调控者"比作为"标签生成者"（如 Mean Teacher）更有灵活性。
- 互信息替代置信度做伪标签筛选，从信息论角度更合理——置信度衡量的是一个模型的"表面确信"，MI 衡量的是"多个模型样本间的一致性"。
- 冻结预训练编码器 + 轻量 MLP 学生的设计是 foundation model 时代 SSL 的务实架构选择。
- Stackelberg 博弈的形式化赋予了方法理论优雅性（Nash 均衡存在性）。

## 局限与展望

- 依赖两个**特定的**冻结预训练编码器（DINOv2 + MAE），编码器选择对性能的影响需要更系统的研究。
- 互信息的 Monte Carlo dropout 估计（$K=5$ 次前向传播）增加了计算开销。
- 元学习的内外循环在每步都需要计算二阶梯度（虽然做了一阶近似），对大规模任务的扩展性需验证。
- 实验主要在图像分类上，向检测、分割等密集预测任务的扩展尚未探索。

## 相关工作与启发

- 与 Meta Co-Training (MCT) 直接对比——TRiCo 通过引入对抗生成器和 MI 筛选全面超越。
- 博弈论视角在 SSL 中的应用还相对少见，启发了将更多博弈论工具引入学习范式。
- 教师的元学习策略可以迁移到其他需要自适应调节训练超参数的场景（如课程学习、数据加权）。

## 评分

- **新颖性**: ⭐⭐⭐⭐☆ — 三方博弈 + MI筛选 + 元学习教师的组合新颖，但每个组件单独不算全新
- **实验充分度**: ⭐⭐⭐⭐⭐ — CIFAR/SVHN/STL/ImageNet + few-shot + OOD + 详细消融
- **写作质量**: ⭐⭐⭐⭐☆ — 结构清晰但公式较多，理论分析可以更直观
- **价值**: ⭐⭐⭐⭐☆ — 在低标签 SSL 上效果显著，冻结编码器设计适合实际应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[NeurIPS 2025\] RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning](roirl_efficient_self-supervised_reasoning_with_offline_iterative_reinforcement_l.md)
- [\[ACL 2025\] MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning](../../ACL2025/reinforcement_learning/maporl_multi-agent_post-co-training_for_collaborative_large_language_models_with.md)
- [\[ICML 2026\] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](../../ICML2026/reinforcement_learning/game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)
- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](confounding_robust_deep_reinforcement_learning_a_causal_approach.md)

</div>

<!-- RELATED:END -->
