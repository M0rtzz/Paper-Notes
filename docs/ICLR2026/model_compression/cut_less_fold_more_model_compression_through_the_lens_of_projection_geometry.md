---
title: >-
  [论文解读] Cut Less, Fold More: Model Compression through the Lens of Projection Geometry
description: >-
  [ICLR 2026][模型压缩][模型折叠] 将结构化剪枝和模型折叠(model folding)统一为正交投影框架——剪枝是坐标轴对齐投影，折叠是聚类子空间投影——并证明在秩差为1的条件下折叠的参数重建误差严格更小，在1000+个 checkpoint 上验证折叠在中-高压缩率下通常优于剪枝。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "模型折叠"
  - "结构化剪枝"
  - "正交投影"
  - "免校准压缩"
  - "投影几何"
---

# Cut Less, Fold More: Model Compression through the Lens of Projection Geometry

**会议**: ICLR 2026  
**arXiv**: [2602.18116](https://arxiv.org/abs/2602.18116)  
**代码**: 有 (附录链接)  
**领域**: 模型压缩  
**关键词**: 模型折叠, 结构化剪枝, 正交投影, 免校准压缩, 投影几何

## 一句话总结

将结构化剪枝和模型折叠(model folding)统一为正交投影框架——剪枝是坐标轴对齐投影，折叠是聚类子空间投影——并证明在秩差为1的条件下折叠的参数重建误差严格更小，在1000+个 checkpoint 上验证折叠在中-高压缩率下通常优于剪枝。

## 研究背景与动机

**领域现状**：免校准的训练后结构化压缩是模型部署的关键需求。主流方法是基于幅度的结构化剪枝(magnitude pruning)，即按权重大小移除神经元/通道/filter。最近提出的 model folding 通过聚类相似权重并绑定实现压缩。

**现有痛点**：(1) 剪枝直接将权重置零，造成较大的参数扰动和功能偏移；(2) folding 作为替代方案缺乏理论支撑，其在何种条件下优于剪枝不清楚；(3) 两者缺乏统一的比较框架。

**核心矛盾**：剪枝移除权重导致参数空间的轴对齐投影损失了方向信息，而折叠保留了合并方向但缺乏理论保证表明这种保留在什么程度上是有益的。

**本文目标** 建立剪枝和折叠的统一投影理论框架，严格证明折叠在参数重建和功能保持上的优越性。

**切入角度**：将两种压缩方法视为参数空间中的正交投影，剪枝对应坐标轴对齐子空间，折叠对应聚类结构子空间。

**核心 idea**：剪枝是 $\mathbf{C}_p = \begin{pmatrix} I & 0 \\ 0 & 0 \end{pmatrix}$ 的坐标投影，折叠是 $\mathbf{C}_f = \mathbf{U}_f(\mathbf{U}_f^\top \mathbf{U}_f)^{-1}\mathbf{U}_f^\top$ 的聚类投影，后者保留更多参数方向信息，重建误差 $\|\mathbf{W} - \mathbf{W}_f\|_F^2 \leq \|\mathbf{W} - \mathbf{W}_p\|_F^2$。

## 方法详解

### 整体框架

给定训练好的权重矩阵 $\mathbf{W} \in \mathbb{R}^{m \times p}$ → 定义正交投影算子（剪枝 $\mathbf{C}_p$ 或折叠 $\mathbf{C}_f$）→ 投影得到压缩权重 $\mathbf{W}_{\text{comp}} = \mathbf{C} \mathbf{W}$ → 对 CNN 做 REPAIR (BatchNorm 重估)，对 ViT 做 LayerNorm 重置 → 可选的短期微调。

### 关键设计

1. **统一正交投影框架**:

    - 做什么：将剪枝和折叠统一为参数空间中的正交投影
    - 核心思路：剪枝保留前 $k$ 个神经元对应 $\mathbf{U}_p = \begin{pmatrix} I \\ 0 \end{pmatrix}$，折叠将 $m$ 个参数向量聚为 $k$ 类并替换为类均值，对应 $\mathbf{U}_f \in \{0,1\}^{m \times k}$ 的 one-hot 聚类分配矩阵
    - 设计动机：正交投影 $\mathbf{C}y = \arg\min_{z \in \text{Range}(\mathbf{U})} \|y - z\|_2$ 是到子空间最近点的映射，天然衡量压缩的参数失真

2. **折叠优于剪枝的理论证明 (Theorem 2.1 & 2.2)**:

    - 做什么：证明任意剪枝方案都存在重建误差更小的折叠方案
    - 核心思路：Theorem 2.1 构造性证明：将所有被剪枝的行合并为一个额外聚类（秩 $k_f = k_p + 1$），其 Frobenius 范数重建误差严格不超过剪枝。Theorem 2.2 进一步证明最优 $k$-means 折叠 $\|\mathbf{W} - \mathbf{W}_f^\star\|_F^2 \leq \|\mathbf{W} - \mathbf{W}_f'\|_F^2 \leq \|\mathbf{W} - \mathbf{W}_p\|_F^2$
    - 设计动机：结合损失函数的 Lipschitz 连续性 $|L(\mathbf{W}_1) - L(\mathbf{W}_2)| \leq \kappa \|\mathbf{W}_1 - \mathbf{W}_2\|_F$，更小的参数重建误差直接意味着更小的功能扰动

3. **大规模超参数消融验证**:

    - 做什么：在 1000+ 个 checkpoint 上系统比较折叠与剪枝在不同训练条件下的表现
    - 核心思路：覆盖 Adam/SGD 优化器、不同学习率、数据增强、正则化、SAM 训练，以及 LLaMA-60M/130M，验证理论预测的适用边界
    - 设计动机：已有剪枝研究仅变种子而固定超参，未探索上游训练如何影响压缩效果

### 损失函数 / 训练策略

压缩本身免校准无训练。折叠使用 $k$-means 聚类确定分组。可选后处理：CNN 做 REPAIR (BatchNorm 统计量重估)，ViT 做 LayerNorm 重置，或 1-5 epoch 微调。

## 实验关键数据

### 主实验

| 架构/数据集 | 压缩率 | FOLD Acc | MAG1 Acc | 折叠优势 |
|------------|--------|----------|----------|---------|
| ResNet18/CIFAR-10 (Adam) | 50% | 显著领先 | — | 中-高压缩时最大 |
| ViT-B/32/CIFAR-10 | 50% | 显著领先 | — | 一致正增益 |
| CLIP ViT-B/32/ImageNet-1K | 50% | 领先 | — | LayerNorm 重置后仍保持 |
| LLaMA-60M/C4 (PPL↓) | 20% | 47.17 | 54.51 | FOLD 更低 PPL |
| LLaMA-60M/C4 (PPL↓) | 50% | 221.32 | 398.62 | FOLD 大幅领先 |

### 消融实验

| 配置 | 指标 | 说明 |
|------|------|------|
| 低学习率 + Adam | FOLD 优势最大 | 平坦解有利于折叠 |
| 高学习率 + Adam | 优势缩小/反转 | 尖锐解削弱聚类投影优势 |
| + SAM 训练 | 两者均提升，FOLD 提升更大 | SAM 引导更平坦解 |
| + 强数据增强 | CNN 上收窄差距 | 增强鲁棒性使轴对齐投影也不太差 |
| 微调后 (1-5 epoch) | FOLD 保持领先 | 折叠提供更好初始化 |

### 关键发现

- 折叠在中-高压缩率下一致优于剪枝，差距随压缩率增大而扩大
- 促进平坦解的训练条件（中等学习率、SAM）会放大折叠优势
- 折叠后微调收敛更快，说明折叠提供了更好的压缩初始化
- LLaMA-60M 在 50% 压缩率下折叠 PPL 约为剪枝的一半

## 亮点与洞察

- 投影几何视角非常优雅：一个框架统一了两种看似不同的压缩方法，并从投影误差角度给出了清晰的理论比较
- 1000+ checkpoint 的大规模实验设计系统性极强，首次揭示了上游训练超参对压缩效果的细粒度影响
- 折叠本质上是"合并相似方向"而非"删除坐标"，这个几何直觉对未来设计新压缩方法有启发

## 局限与展望

- 理论保证需要秩差 $k_f = k_p + 1$，虽然实际影响可忽略但不是严格 matched-size 比较
- 对 ViT 和 LLaMA 仅压缩 FFN 块，注意力层的折叠未探索
- 未与量化、蒸馏等其他压缩方法组合评估
- 大规模 LLM（>1B 参数）因训练成本限制未覆盖

## 相关工作与启发

- **vs Magnitude Pruning**: 剪枝是坐标投影（丢方向信息），折叠是聚类投影（保留方向），理论上折叠重建误差更小
- **vs Model Folding (Wang et al., 2025)**: 本文为 folding 提供了首个理论基础，证明其优越性不只是经验观察
- **vs SoTA LLM 剪枝 (Wanda, SparseGPT)**: 这些方法依赖校准数据，属于不同设定；本文是免校准设定下的比较

## 评分

- 新颖性: ⭐⭐⭐⭐ 投影几何统一框架是全新视角，Theorem 2.1/2.2 是本文核心贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 1000+ checkpoint 覆盖多架构/数据集/超参，消融极为丰富
- 写作质量: ⭐⭐⭐⭐ 理论与实验交织辅证，表述清晰
- 价值: ⭐⭐⭐⭐ 为免校准压缩提供了坚实的理论指导和实践替代方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Exploring Concreteness Through a Figurative Lens](../../ACL2026/model_compression/exploring_concreteness_through_a_figurative_lens.md)
- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](../../NeurIPS2025/model_compression/less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)
- [\[ICLR 2026\] The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm](the_geometry_of_llm_quantization_gptq_as_babais_nearest_plane_algorithm.md)
- [\[ICLR 2026\] Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size](topology_and_geometry_of_the_learning_space_of_relu_networks_connectivity_and_si.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)

</div>

<!-- RELATED:END -->
