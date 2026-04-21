---
title: >-
  [论文解读] Learning Shared Representations from Unpaired Data
description: >-
  [NeurIPS 2025][多模态][spectral embedding] 提出 SUE (Spectral Universal Embedding)，首次证明几乎完全依赖非配对数据即可学习跨模态共享表示：通过独立的频谱嵌入从各模态随机游走中提取模态不变的"通用"结构，再用极少量配对样本（~100对）做 CCA 线性对齐 + MMD 非线性微调，在检索上超越使用同等配对数的对比学习 250%+。
tags:
  - NeurIPS 2025
  - 多模态
  - spectral embedding
  - universal embedding
  - MMD
  - CCA
---

# Learning Shared Representations from Unpaired Data

**会议**: NeurIPS 2025  
**arXiv**: [2505.21524](https://arxiv.org/abs/2505.21524)  
**代码**: [https://shaham-lab.github.io/SUE_page](https://shaham-lab.github.io/SUE_page)  
**领域**: 多模态VLM / 跨模态表示学习 / 弱配对学习  
**关键词**: spectral embedding, universal embedding, unpaired multimodal learning, MMD, CCA  

## 一句话总结
提出 SUE (Spectral Universal Embedding)，首次证明几乎完全依赖非配对数据即可学习跨模态共享表示：通过独立的频谱嵌入从各模态随机游走中提取模态不变的"通用"结构，再用极少量配对样本（~100对）做 CCA 线性对齐 + MMD 非线性微调，在检索上超越使用同等配对数的对比学习 250%+。

## 研究背景与动机

**领域现状**：跨模态共享表示学习是多模态学习的核心任务。当前 SOTA（如 CLIP）依赖海量配对数据（4 亿对图文）进行对比学习。

**现有痛点**：配对数据获取成本高昂——在医学、遥感、语音等领域，配对标注需专家参与或特殊测试，大规模配对数据几乎不可能获得。而非配对数据（各模态独立的数据）则相对容易大量获取。

**核心矛盾**：对比学习的核心监督信号来自配对关系，没有配对就无法训练。直觉上，没有配对信息的情况下无法建立跨模态联系。

**切入角度**：作者提出"通用嵌入（universal embedding）"概念——如果各模态的预训练表示都能很好地捕捉语义相似性，那么**基于各模态表示独立构建的随机游走过程应该高度相似**。这种相似性可以通过频谱嵌入来捕获，且不需要配对数据。

**核心 idea**：独立模态的扩散算子（diffusion operator）具有相似的特征函数（模态不变性），频谱嵌入可以从非配对数据中提取这种通用结构。

## 方法详解

### 整体框架
SUE 分三步流水线：
1. **Spectral Embedding (SE)**：从各模态预训练特征独立计算频谱嵌入，提取模态不变结构
2. **CCA**：用极少量配对样本做线性对齐（消除 SE 的符号/基底歧义）
3. **MMD-net**：用非线性残差网络进一步对齐两个分布

输入：大量非配对的单模态预训练嵌入 $\mathcal{X}, \mathcal{Y}$ + 极少配对 $\mathcal{X}_p, \mathcal{Y}_p$（$m \ll n$）。输出：通用嵌入映射 $f_\mathcal{X}, f_\mathcal{Y}$。

### 关键设计

1. **频谱嵌入（SE）的通用性论证**：

    - 功能：证明从不同模态独立计算的 SE 可以捕获相同的语义结构
    - 核心思路：设潜在语义流形为 $\mathcal{M}$，$f, g$ 是将 $\mathcal{M}$ 变换为两个模态的映射。如果 $f, g$ 有有界畸变和有界 Ricci 曲率，则 $f(\mathcal{M})$ 和 $g(\mathcal{M})$ 上的 Laplace-Beltrami 算子有在 $L_\infty$ 意义下相似的特征函数。实践中，随机游走矩阵 $P = D^{-1}W$ 收敛于扩散算子，SE 的前 $k$ 个非平凡特征向量提供了扩散算子特征函数的离散近似
    - 设计动机：现代预训练单模态模型（如 CLIP 视觉编码器、BERT 文本编码器）已经很好地编码了语义相似性，基于它们的随机游走确实高度相似（实验验证）

2. **参数化频谱嵌入 (SpectralNet)**：

    - 功能：用深度学习方法可泛化地计算 SE，克服传统 SE 的可扩展性和泛化性不足
    - 核心思路：学习参数化映射 $f: \mathbb{R}^d \to \mathbb{R}^k$，最小化 Rayleigh 商 $\mathcal{L}_{\text{spectralnet}}(f) = \frac{1}{n^2}\text{Trace}(f(X)^T L f(X))$，同时约束正交性 $f(X)^T f(X) = I_k$，其中 $L=I-P$ 是随机游走图 Laplacian。两个模态的 $S_\mathcal{X}, S_\mathcal{Y}$ 完全独立训练
    - 设计动机：传统 SE 不可泛化到新样本，SpectralNet 学到的参数化映射可以直接应用于测试数据

3. **CCA 线性对齐**：

    - 功能：消除 SE 的非唯一性（符号、基底旋转）
    - 核心思路：在极少量配对样本 $(S_\mathcal{X}(\mathcal{X}_p), S_\mathcal{Y}(\mathcal{Y}_p))$ 上执行 CCA，得到投影矩阵 $Q_\mathcal{X}, Q_\mathcal{Y} \in \mathbb{R}^{k \times r}$。对齐后 $\tilde{S}_\mathcal{X} = Q_\mathcal{X} \circ S_\mathcal{X}$, $\tilde{S}_\mathcal{Y} = Q_\mathcal{Y} \circ S_\mathcal{Y}$
    - 设计动机：SE 的特征向量方向和基底不唯一，CCA 用最少的配对样本解决这个歧义

4. **MMD-net 非线性对齐**：

    - 功能：精调两模态分布的对齐精度
    - 核心思路：训练残差网络 $F_\theta: \mathbb{R}^r \to \mathbb{R}^r$ 最小化经验 MMD：$\mathcal{L}_{\text{MMD}} = \frac{1}{m_1^2}\sum_{x_i,x_j}\kappa(\tilde{x_i},\tilde{x_j}) - \frac{2}{m_1 m_2}\sum_{x_i,y_j}\kappa(\tilde{x_i},\tilde{y_j}) + \frac{1}{m_2^2}\sum_{y_i,y_j}\kappa(\tilde{y_i},\tilde{y_j})$，其中 $\kappa$ 为 RBF 核
    - 设计动机：CCA 只做线性对齐不够精确，MMD loss **不需要配对数据**，可以利用全部非配对数据集

### 最终映射
$f_\mathcal{X} = Q_\mathcal{X} \circ S_\mathcal{X}$，$f_\mathcal{Y} = F_\theta \circ Q_\mathcal{Y} \circ S_\mathcal{Y}$

## 实验关键数据

### 主实验 — 跨模态检索 (Recall@k)

| 数据集 | #配对 | 任务 | SUE R@1 | SUE R@10 | 对比学习 R@1 | 对比学习 R@10 | SUE 提升 |
|--------|-------|------|---------|----------|-------------|-------------|---------|
| MSCOCO | 100 | I2T | 5.75 | 34.25 | 1.50 | 13.00 | +257% |
| MSCOCO | 100 | T2I | 5.25 | 33.25 | 0.80 | 12.20 | +257% |
| Flickr30k | 500 | I2T | 4.25 | 32.00 | 3.00 | 16.20 | +103% |
| Flickr30k | 500 | T2I | 5.75 | 32.75 | 2.50 | 15.00 | +103% |
| Edges2Shoes | 50 | E2S | 4.00 | 25.25 | 1.00 | 14.00 | +201% |
| Handwritten | 100 | K2P | 25.50 | 79.00 | 4.80 | 28.00 | +284% |

### 消融实验 — 各组件贡献 (Flickr30k & MSCOCO T2I R@10)

| 配置 | Flickr30k (无SE) | Flickr30k (+SE) | MSCOCO (无SE) | MSCOCO (+SE) |
|------|-----------------|-----------------|---------------|-------------|
| 原始表示 | 2.25 | 8.75 | 1.50 | 4.25 |
| +MMD | 3.75 | 5.50 | 2.00 | 3.75 |
| +CCA | 4.50 | 30.25 | 7.75 | 31.50 |
| +CCA+MMD | 4.75 | **32.75** | 9.75 | **33.25** |

### 关键发现
- **SE 是核心**：没有 SE 时，CCA+MMD 在 Flickr30k 上只有 4.75（R@10），加 SE 后跃升至 32.75——SE 贡献了绝大部分性能
- 用 SE 替换为 AutoEncoder 后性能显著下降，证明 SE 的通用性是不可替代的
- SUE 100对配对 ≈ 对比学习 1000+ 对配对，非配对数据的价值被严重低估
- 增加非配对数据量持续提升检索性能（图 5b），而增加配对数超过最低需求后增益趋零（图 5c）
- SUE 甚至可以做"几乎无文本"的文本到图像生成和语义算术（向量加法 = 语义组合）

## 亮点与洞察
- **范式转移**：从"必须有大量配对"到"几乎只需非配对数据"，对资源受限领域（医学影像、稀有语言等）有深远意义
- **SE 的通用性**惊人——独立模态训练的频谱嵌入竟然能高度对齐，这从理论（扩散算子特征函数保持）和实验两方面得到验证
- **MMD loss 的妙用**：作为不需要配对数据的分布对齐工具，使得全部非配对数据都能被利用
- **语义算术**（文本嵌入 + 图像嵌入 = 组合语义图像）在几乎没有配对的条件下出现，展示了通用嵌入空间的深层结构性

## 局限与展望
- 绝对检索性能仍远低于 CLIP 等大规模配对训练的 SOTA（R@10≈33 vs CLIP>>90），作者也承认目前不意在替代大配对模型
- 仅在视觉-语言和视觉-视觉场景验证，需扩展到视频、语音、科学数据等更复杂模态
- SE 的计算复杂度（特征分解）可能成为超大数据集的瓶颈，虽然 SpectralNet 缓解了部分问题
- 理论上的"有界畸变+有界 Ricci 曲率"假设在实际预训练模型中是否成立仍缺乏严格验证
- CCA 步骤仍需最少 ~100-500 对配对数据，完全零配对的场景有待探索

## 相关工作与启发
- **vs CLIP**：CLIP 是在 4 亿配对上训练的对比学习，SUE 只需 100 对即可获得有意义的跨模态表示——两者面向完全不同的数据可得性场景
- **vs CSA**：CSA 虽然为小配对设计，但仍仅操作配对数据，不利用非配对数据；SUE 的核心力量来自非配对数据
- **vs MACK**：MACK 使用分割-文本配对模型做图文对齐，仍依赖配对训练的子模块；SUE 的三个组件都可以在非配对数据上工作

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从理论到实践完整论证了非配对数据可以学习跨模态共享表示，概念上极具突破性
- 实验充分度: ⭐⭐⭐⭐ 覆盖检索、生成、算术、零样本、分类等多个任务，消融清晰，但绝对性能有限
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，直觉解释清晰，图示优美，整体叙事从理论到实验流畅
- 价值: ⭐⭐⭐⭐ 开启了非配对多模态学习的新范式，对数据稀缺领域有重要意义，但离实际应用仍有差距

<!-- RELATED:START -->

## 相关论文

- [MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning](midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le.md)
- [Learning Optimal Multimodal Information Bottleneck Representations](../../ICML2025/multimodal_vlm/learning_optimal_multimodal_information_bottleneck_representations.md)
- [Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [Decoupling Common and Unique Representations for Multimodal Self-supervised Learning](../../ECCV2024/multimodal_vlm/decoupling_common_and_unique_representations_for_multimodal_self-supervised_lear.md)

<!-- RELATED:END -->
