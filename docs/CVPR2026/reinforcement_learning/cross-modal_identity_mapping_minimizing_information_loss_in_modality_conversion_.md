---
title: >-
  [论文解读] Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning
description: >-
  [CVPR 2026][强化学习][图像描述] 提出 Cross-modal Identity Mapping (CIM)，通过分析用 caption 检索到的图像的表示一致性（GRC）和与源图像的相关性（QIR）来量化图像描述中的信息损失，将其作为 RL 奖励信号训练 LVLM 生成细粒度且精确的描述…
tags:
  - "CVPR 2026"
  - "强化学习"
  - "图像描述"
  - "跨模态信息损失"
  - "检索奖励"
  - "GRPO"
---

# Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning

**会议**: CVPR 2026  
**arXiv**: [2603.01696](https://arxiv.org/abs/2603.01696)  
**代码**: 待发布（论文接收后公开）  
**领域**: 强化学习  
**关键词**: 图像描述, 跨模态信息损失, 检索奖励, 强化学习, GRPO

## 一句话总结
提出 Cross-modal Identity Mapping (CIM)，通过分析用 caption 检索到的图像的表示一致性（GRC）和与源图像的相关性（QIR）来量化图像描述中的信息损失，将其作为 RL 奖励信号训练 LVLM 生成细粒度且精确的描述，无需额外标注。

## 研究背景与动机
LVLM 在图像描述任务中常常遗漏或错误表示关键视觉内容。作者通过在 Oxford-IIIT Pet 数据集上的细粒度分类实验验证了这一点：多个 LVLM（如 Qwen3-VL-8B、InternVL3-8B）的物种分类准确率接近 100%，但品种分类准确率仅有 15%~40%，说明模型倾向于描述粗粒度概念而忽略细节信息，存在严重的跨模态信息损失。

现有改进方法分为两类：(1) 构建细粒度标注数据做 SFT，但标注成本高昂；(2) 用基于 VLM 的度量作为 RL 奖励，但因 VLM 自身的组合推理能力有限，容易出现 reward hacking。核心矛盾在于：如何在不依赖额外标注的情况下准确量化图像描述中的信息损失？

作者提出一个关键洞察：**caption 越细粒度，用它检索到的图像越一致；caption 越准确，检索到的图像与源图像越相似**。基于此，通过分析检索结果的分布来推断 caption 的信息损失。

## 方法详解

### 整体框架
CIM 是一个无标注的 RL 框架，核心流程：(1) LVLM 为输入图像生成多个 caption；(2) 用每个 caption 作为查询进行文本检索，获取 top-K 相关图文对；(3) 从检索结果中计算 GRC 和 QIR 两个指标作为奖励；(4) 通过 GRPO 优化 LVLM。

### 关键设计

1. **Gallery Representation Consistency (GRC)**:

    - 功能：评估用 caption 检索到的图像集合的内部一致性，反映 caption 的细节丰富程度
    - 核心思路：$GRC(c) = \|\frac{1}{K}\sum_{r=1}^{K}\tilde{v}(x_{i_r})\|_2$，其中 $\tilde{v}(x_j)$ 是图像经视觉表示模型提取并 $\ell_2$ 归一化后的 embedding。GRC 本质上是 mean resultant length，衡量 embedding 向量在超球面上的集中程度
    - 设计动机：caption 越详细和具体，检索到的图像在视觉表示空间中越集中（GRC 越高）；caption 越模糊粗粒度，检索结果越分散

2. **Query-gallery Image Relevance (QIR)**:

    - 功能：衡量源图像与检索到的图像之间的相关性，反映 caption 的准确性
    - 核心思路：$QIR(v, c) = \sum_{r=1}^{K}\lambda(r) \cdot Cos(\tilde{v}(v), \tilde{v}(x_{i_r}))$，其中 $\lambda(r) = 1/2^{r-1}$ 是指数衰减权重，越靠前的检索结果权重越大
    - 设计动机：如果 caption 准确描述了源图像内容，检索到的图像应与源图像在语义上高度相似；若 caption 包含错误信息，检索结果将偏离源图像

3. **Cross-modal Identity Mapping 奖励函数**:

    - 功能：将 GRC 和 QIR 组合为 RL 奖励，通过 GRPO 优化 LVLM
    - 核心思路：$\Upsilon(v, c) = GRC(c) + \beta \cdot QIR(v, c)$，$\beta$ 平衡精确性和细节丰富度。采样 $G$ 个 caption，计算组内归一化 advantage $A_z = \frac{\Upsilon_z - mean(\{\Upsilon\})}{std(\{\Upsilon\})}$
    - 设计动机：将 caption 质量评估转化为图像-图像相似度问题，绕开了直接衡量跨模态信息损失的难题，无需额外标注

### 损失函数 / 训练策略
使用 VERL 框架进行 GRPO 训练。训练数据为 RefinedCaps（6.5K 图像），每图生成 5 个 caption。文本检索用 SBERT（MPNet-base），图像编码用 OpenCLIP ViT-H/14，检索库由 RefinedCaps + DenseFusion-1M 扩增。学习率 $1 \times 10^{-6}$，训练 2 个 epoch。

## 实验关键数据

### 主实验

| 模型 | 数据集 | CAPTURE | Relation QA | 提升 |
|------|--------|---------|-------------|------|
| Qwen2.5-VL-7B + CIM | COCO-LN500 | 48.93 | 44.15 | Relation Recall +20.2, QA +20.4 |
| Qwen2-VL-7B + CIM | COCO-LN500 | 48.64 | 38.71 | Relation Recall +10.4, QA +18.2 |
| LLaVA1.5-7B + CIM | COCO-LN500 | 48.62 | 24.98 | Relation Recall +12.6, QA +10.6 |
| InternVL3-8B + CIM | COCO-LN500 | 48.90 | 38.67 | Relation Recall +10.0, QA +12.2 |

### 消融实验

| 配置 | CAPTURE | 说明 |
|------|---------|------|
| GRC only | 有提升但有限 | 仅鼓励细节丰富度 |
| QIR only | 有提升但有限 | 仅约束准确性 |
| GRC + QIR | 最佳 | 两者互补 |
| 不同检索库规模 | 越大越好 | 更大 gallery 提供更可靠的检索信号 |

### 关键发现
- CIM 在 COCO-LN500 上对 Qwen2.5-VL-7B 的关系推理（Relation Recall）提升高达 20.2%，关系QA 提升 20.4%——这是一个非常显著的提升
- CIM 在多个不同架构（LLaVA、Qwen-VL、InternVL）和不同版本上均有效，证明方法的通用性
- Pearson 相关分析验证了 GRC/QIR 与实际 caption 质量（品种分类准确率的 logit）之间的正相关性

## 亮点与洞察
- 将跨模态信息损失量化问题巧妙转化为图像检索后的图像-图像相似度问题，完全无需额外标注
- GRC 和 QIR 的设计直觉清晰：一个管"detail"，一个管"accuracy"，与人对 caption 质量的直觉一致
- 在 Relation 维度的巨大提升说明现有 LVLM 在关系推理方面的信息损失最为严重，也最容易通过 RL 改善

## 局限与展望
- 检索库的构成和规模直接影响奖励信号的质量，对检索模型的选择敏感
- 训练仅用 6.5K 图像，虽然效率高但可能限制了上限
- GRC 和 QIR 的计算需要额外的检索步骤，增加了训练开销
- 指数衰减权重 $\lambda(r)$ 的设计缺乏理论依据

## 相关工作与启发
- 与 CapRL 类似使用 RL 优化 captioning，但奖励信号来源不同：CapRL 用 VQA，CIM 用检索一致性
- 自检索奖励（self-retrieval reward）的思路在先前工作中有探索，CIM 进一步将其分解为细粒度（GRC）和准确性（QIR）两个维度
- 将生成质量评估转化为检索质量评估的思路，有望推广到其他跨模态生成任务
- 与 cycle-consistency 方法（从 caption 重建图像）相比，CIM 避免了训练图像生成器的高开销
- SC-Captioner 的关键词检查方式过于粗粒度，CIM 通过检索分布提供连续且全面的质量信号

## 补充细节
- GRC 的 mean resultant length 本质上衡量单位超球面上向量集合的集中度，值越大说明向量越一致
- QIR 的指数衰减 $\lambda(r) = 1/2^{r-1}$ 使得 top-1 检索结果贡献最大，符合检索排序的可信度递减趋势
- 在 SFT 模型上应用 CIM 也能获得进一步提升，说明方法与 SFT 互补
- 实验在 LLaVA-1.5-7B 到 InternVL3-8B 等 6 种模型上均验证有效
- 检索库扩大（加入 DenseFusion-1M）后性能进一步提升，说明检索信号的可靠性与库规模正相关

## 评分
- 新颖性: ⭐⭐⭐⭐ 检索一致性作为 caption 质量代理的洞察新颖，GRC/QIR 设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 多模型验证充分，有 Pearson 相关验证，但消融可以更细致
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，验证实验设计巧妙，图表质量高
- 价值: ⭐⭐⭐⭐ 无标注 RL 优化方案实用性强，关系推理提升显著

## 关键术语
- **Mean Resultant Length**: 超球面上的向量一致性度量
- **Identity Mapping**: 将图像到 caption 的转换视为恋恒映射，最小化信息损失
- **SBERT/OpenCLIP**: 分别用于文本检索和图像 embedding 提取

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization](../../ICCV2025/reinforcement_learning/r1-onevision_advancing_generalized_multimodal_reasoning_through_cross-modal_form.md)
- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](../../ICLR2026/reinforcement_learning/cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[ICML 2026\] SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts for Deep Reinforcement Learning](../../ICML2026/reinforcement_learning/sphere_mitigating_the_loss_of_spectral_plasticity_in_mixture-of-experts_for_deep.md)
- [\[CVPR 2026\] CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)
- [\[ICML 2025\] Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](../../ICML2025/reinforcement_learning/mitigating_plasticity_loss_in_continual_reinforcement_learning_by_reducing_churn.md)

</div>

<!-- RELATED:END -->
