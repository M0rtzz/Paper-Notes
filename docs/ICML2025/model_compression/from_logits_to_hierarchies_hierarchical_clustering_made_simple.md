---
title: >-
  [论文解读] From Logits to Hierarchies: Hierarchical Clustering made Simple
description: >-
  [ICML 2025][模型压缩][层次聚类] 提出 L2H（Logits to Hierarchies）算法，仅利用预训练平面聚类模型的 logits 输出，通过掩码 softmax 和迭代合并策略，在无需微调的情况下构建高质量层次聚类，大幅超越专用深度层次聚类模型…
tags:
  - "ICML 2025"
  - "模型压缩"
  - "层次聚类"
  - "Logits"
  - "预训练模型"
  - "树状图"
  - "无监督学习"
---

# From Logits to Hierarchies: Hierarchical Clustering made Simple

**会议**: ICML 2025  
**arXiv**: [2410.07858](https://arxiv.org/abs/2410.07858)  
**代码**: 论文附录提供完整 Python 实现（NumPy + SciPy）  
**领域**: 人类理解  
**关键词**: 层次聚类, Logits, 预训练模型, 树状图, 无监督学习

## 一句话总结

提出 L2H（Logits to Hierarchies）算法，仅利用预训练平面聚类模型的 logits 输出，通过掩码 softmax 和迭代合并策略，在无需微调的情况下构建高质量层次聚类，大幅超越专用深度层次聚类模型，且在 ImageNet 规模数据集上 CPU 运行不到一分钟。

## 研究背景与动机

层次聚类（Hierarchical Clustering）是机器学习中的经典问题，在系统发育树、肿瘤亚型、社交网络等领域广泛应用。相较于平面聚类，层次聚类能提供多粒度视角，无需预先指定聚类数，增强可解释性。

近年来出现了一批基于深度学习的专用层次聚类方法，如 DeepECT（基于自编码器嵌入空间的层次聚类）和 TreeVAE（基于 VAE 潜空间的层次生成模型）。然而作者发现这些方法存在**两个关键局限**：

**可扩展性差**：在大规模数据集上计算需求极高，难以处理大量聚类。TreeVAE 为每个叶节点增加一个解码器，导致类别数增加时计算量急剧膨胀。

**叶层性能下降**：引入层次结构反而牺牲了叶层聚类质量，与非层次模型相比差距显著。这意味着层次化带来了不必要的代价。

这些发现促使作者反思：**是否有必要设计专用的深度层次聚类架构**？能否从已有的高质量平面聚类模型中，以极低开销提取层次结构？

## 方法详解

### 整体框架

L2H 的核心思想极为精巧：**利用预训练平面聚类模型的 logits 中蕴含的聚类间关系信息，通过迭代式自底向上合并，构建完整的层次树**。整个过程不需要访问模型内部表示，不需要微调，甚至可以作用于黑盒模型（如 API 调用的商用模型）。

**输入**：数据集 $\mathcal{D} = \{x_1, \dots, x_N\}$，预训练 $K$-聚类模型 $f_\theta$ 输出的 logits（$N \times K$ 矩阵）。

**输出**：完整的层次聚类树 $\mathcal{H}$（以合并步骤列表表示）。

### 关键设计

#### 1. 基础函数定义

定义两个关键函数：

- **聚类分配函数** $h_\theta(x) = \arg\max_k \text{softmax}_k(f_\theta(x))$：返回数据点 $x$ 的聚类分配
- **预测概率函数** $g_\theta(x) = \max_k \text{softmax}_k(f_\theta(x))$：返回最大分配概率

#### 2. 掩码 Softmax（Masked Softmax）——核心创新

为了度量聚类组之间的关联性，作者引入掩码 softmax：

$$\text{m\_softmax}_k(\mathbf{v}; G) = \begin{cases} \frac{\exp(v_k)}{\sum_{j \notin G} \exp(v_j)} & \text{if } k \notin G \\ 0 & \text{if } k \in G \end{cases}$$

给定一组聚类 $G$，掩码 softmax 将 $G$ 中的聚类从 softmax 计算中排除。这使得我们可以回答一个关键问题：**如果某组聚类不存在，其数据点会被重新分配到哪里？**

基于此定义掩码版的分配和概率函数：

- $h_\theta^m(x; G)$：在排除 $G$ 后的聚类上重新分配
- $g_\theta^m(x; G)$：对应的重分配概率

#### 3. 迭代合并算法

L2H 算法从 $K$ 个单聚类组开始，每步合并两个组，经过 $K-1$ 步构建完整的层次树。每一步分为两个阶段：

**阶段一：选择待合并组**

为每个组 $G$ 计算得分：

$$s(G) = \Lambda_{x \in \mathcal{D}_c, c \in G} g_\theta(x)$$

其中 $\Lambda$ 为聚合函数（实验中使用每聚类均值之和）。选择得分最低的组 $G^*$ 进行合并——直觉上，得分低意味着该组内的数据点对其当前聚类分配"不够自信"，更可能与其他组存在关联。

**阶段二：寻找最相关组**

对 $G^*$ 中的所有数据点，使用掩码 softmax 重新计算排除 $G^*$ 后的分配。统计每个外部聚类 $c$ 收到的总重分配概率：

$$rp(c) = \sum_{x \in \mathcal{D}_{G^*}, h_\theta^m(x; G^*) = c} g_\theta^m(x; G^*)$$

然后选择平均重分配概率最高的组 $G^\dagger$ 进行合并：

$$G^\dagger \in \arg\max_{G \in \mathcal{G} \setminus \{G^*\}} \frac{1}{|G|} \sum_{c \in G} rp(c)$$

#### 4. 方法的关键优势

- **无需微调**：直接利用预训练模型的 logits，保留叶层聚类性能
- **黑盒适用**：不需要模型内部表示，仅需 logits 输出
- **避免成对距离**：不计算数据点间的成对距离，大幅降低计算复杂度
- **通用性强**：适用于任何输出 logits 的聚类/分类模型
- **监督扩展**：可直接应用于监督分类器，从中恢复有意义的类别层次

### 损失函数 / 训练策略

L2H 本身**无需训练**，没有损失函数或可学习参数。它是一个纯推理阶段的后处理算法。整个流程为：

1. 使用预训练的平面模型（如 TURTLE、TEMI）对数据集推理，获取 $N \times K$ 的 logits 矩阵
2. 在 logits 上运行 L2H 算法，$K-1$ 步迭代合并
3. 输出层次结构（树的合并顺序列表）

聚合函数 $\Lambda$ 是唯一的设计选择（超参数），实验默认使用每聚类均值：

$$\Lambda_{x \in \mathcal{D}_c, c \in G} g_\theta(x) = \sum_{c \in G} \frac{1}{|\mathcal{D}_c|} \sum_{x \in \mathcal{D}_c} g_\theta(x)$$

## 实验关键数据

### 主实验

在 CIFAR-10、CIFAR-100、Food-101 上与专用深度层次聚类方法全面对比。

| 数据集 | 方法 | NMI | ARI | ACC | DP↑ | LHD↓ |
|---------|------|-----|-----|-----|-----|------|
| CIFAR-10 | DeepECT | 0.006 | 0.002 | 0.110 | 0.101 | 0.369 |
| CIFAR-10 | TreeVAE | 0.414 | 0.313 | 0.497 | 0.341 | 0.410 |
| CIFAR-10 | **L2H-TEMI** | 0.907 | 0.906 | 0.956 | 0.902 | 0.348 |
| CIFAR-10 | **L2H-TURTLE** | **0.985** | **0.989** | **0.995** | **0.988** | **0.277** |
| CIFAR-100 | DeepECT | 0.016 | 0.005 | 0.070 | 0.052 | 0.121 |
| CIFAR-100 | TreeVAE | 0.199 | 0.098 | 0.228 | 0.103 | 0.484 |
| CIFAR-100 | **L2H-TEMI** | 0.778 | 0.565 | 0.682 | 0.502 | 0.298 |
| CIFAR-100 | **L2H-TURTLE** | **0.917** | **0.831** | **0.896** | **0.803** | **0.235** |
| Food-101 | DeepECT | 0.003 | 0.000 | 0.011 | 0.010 | 0.333 |
| Food-101 | TreeVAE | 0.114 | 0.017 | 0.057 | 0.016 | 0.483 |
| Food-101 | **L2H-TEMI** | **0.917** | **0.841** | **0.904** | **0.801** | **0.270** |
| Food-101 | **L2H-TURTLE** | 0.894 | 0.800 | 0.876 | 0.758 | 0.297 |

L2H 在所有数据集上全面碾压专用深度层次聚类模型，NMI 提升通常超过 **5-10 倍**。

### 消融实验

| 配置 | DP (CIFAR-10) | LHD (CIFAR-10) | DP (CIFAR-100) | LHD (CIFAR-100) | 说明 |
|------|--------------|----------------|----------------|-----------------|------|
| 聚合函数 $\sum_c \sum_x$ | 0.988 | 0.258 | 0.801 | 0.244 | 简单求和 |
| 聚合函数 $\sum_c \text{mean}_x$ | 0.988 | 0.248 | 0.793 | 0.283 | 组内均值后求和 |
| 聚合函数 默认选择 | 0.988 | **0.277** | **0.803** | **0.235** | 每聚类均值之和（最优） |
| 替代合并策略（全对） | - | - | 0.797 | 0.246 | 每步考虑所有组对，计算开销大但无收益 |
| L2H-TCL (弱 backbone) | 0.733 | 0.398 | 0.218 | 0.351 | 非 CLIP backbone，仍超越专用模型 |

### 关键发现

1. **深度专用模型的根本缺陷**：DeepECT 在所有彩色数据集上树坍缩（仅 2-3 个叶节点），TreeVAE 在 CIFAR-100 上只能建模 20 个叶节点（实际有 100 类），受限于每叶一个解码器的架构设计。

2. **ImageNet 案例研究**：将 L2H 应用于预训练 InternImage 分类器的 logits，成功从 1000 类中恢复出与 WordNet 层次高度一致的类别树。有趣的发现包括：雪橇、雪地摩托等雪地人造物与北极犬种被合并，揭示了模型中的**虚假相关性**。

3. **超参数 $K$ 的鲁棒性**：在 CIFAR-100 上将 $K$ 从 85 变到 115（真实值 100），性能优雅退化，证明方法对聚类数估计不敏感。

4. **INaturalist21 多层级实验**：在 2.7M 图像、1103 个 family 级类别上训练 TURTLE，L2H 在更粗粒度的 order/class/phylum/kingdom 层级上的聚类性能**优于**为每个层级单独训练的 TURTLE 模型。

## 亮点与洞察

- **"简单就是美"的典范**：整个 L2H 算法用 NumPy + SciPy 不到 50 行代码实现，却大幅超越需要 GPU 训练数小时的专用深度模型。
- **掩码 softmax 的巧妙设计**：通过"如果这些聚类不存在，数据点会去哪里"这一反事实思维来度量聚类亲缘关系，既直观又高效。
- **保留叶层性能**：L2H 是纯后处理方法，不影响原始模型的聚类质量，这是相对于 end-to-end 层次模型的本质优势。
- **跨范式通用性**：同一算法既适用于无监督聚类也适用于监督分类，为模型可解释性提供了新工具。
- **极致效率**：ImageNet-1K 规模数据集上 CPU 运行 <1 分钟，而 TreeVAE 需要 GPU 训练数小时。

## 局限与展望

1. **依赖 logits 质量**：L2H 的效果取决于预训练模型的 logits 的信息量。如果模型校准差（poorly calibrated），logits 可能无法准确反映聚类间的真实关系。
2. **无自动层级选择**：产生完整的树后，仍需人工检查以提取有意义的粒度层级，缺乏自动选择机制。
3. **仅验证了视觉数据**：实验集中在视觉数据集上，在 NLP、图数据等其他模态上的表现尚未验证。
4. **聚合函数的选择**：虽然消融实验表明不同聚合函数的影响有限，但最优选择可能因数据集而异，缺乏自适应机制。
5. **与传统凝聚聚类在表示空间上的对比不够充分**：使用 CLIP 嵌入的凝聚聚类已有不错表现（如 CIFAR-10 上 NMI=0.799），对比中可以进一步分析 L2H 的信息增益来源。

## 相关工作与启发

- **TURTLE / TEMI**：作为 L2H 的 backbone 模型，代表当前最强的平面聚类方法，利用 CLIP/DINOv2 预训练表示。L2H 表明高质量的平面聚类已蕴含丰富的层次信息。
- **TreeVAE / DeepECT**：代表深度层次聚类的两条路线（VAE 生成式 vs. AE 嵌入式），本文揭示了其在现实场景下的根本局限。
- **HypHC**：基于双曲嵌入的 Dasgupta cost 优化，理论优雅但实际聚类性能不佳。
- **Dasgupta Cost**：层次聚类评估的理论基石，L2H 的 DP 和 LHD 指标与之相关。
- **启发**：该工作启示我们，在许多场景下，**轻量级后处理 + 强预训练模型**的组合可能优于专门设计的端到端方法。这一思路可推广到其他结构化预测任务。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 方法本身简单但视角新颖，对"需要专用架构"的假设提出有力挑战 |
| 技术深度 | 3 | 算法设计精巧但技术门槛不高，核心是掩码 softmax + 贪心合并 |
| 实验充分性 | 5 | 多数据集、多 backbone、多指标、完整消融和敏感性分析，非常扎实 |
| 实用性 | 5 | 无需训练，CPU 即可运行，代码简洁，即插即用 |
| 写作质量 | 4 | 论述清晰，图表丰富，对现有方法的批判性分析有说服力 |
| **总分** | **4.2** | 实用价值极高的工作，以极简方法取得了令人信服的 SOTA 结果 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Cross-Architecture Distillation Made Simple with Redundancy Suppression](../../ICCV2025/model_compression/cross-architecture_distillation_made_simple_with_redundancy_suppression.md)
- [\[ICML 2025\] TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree](treelora_efficient_continual_learning_via_layer-wise_loras_guided_by_a_hierarchi.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](../../ICCV2025/model_compression/knowledge_distillation_with_refined_logits.md)
- [\[CVPR 2025\] Logits DeConfusion with CLIP for Few-Shot Learning](../../CVPR2025/model_compression/logits_deconfusion_with_clip_for_few-shot_learning.md)
- [\[ACL 2025\] Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](../../ACL2025/model_compression/beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
