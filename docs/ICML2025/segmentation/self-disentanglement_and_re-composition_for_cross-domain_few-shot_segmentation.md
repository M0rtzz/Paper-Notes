---
title: >-
  [论文解读] Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation
description: >-
  [ICML 2025][图像分割][跨域少样本分割] 本文发现跨域少样本分割（CD-FSS）中基于距离比较的方法存在特征纠缠问题，其根源在于ViT各层输出在距离计算时的等权交叉匹配，进而提出通过自解耦（Self-Disentanglement）和重组合（Re-Composition）的方式，学习ViT组件间的比较权重来解决该问题。
tags:
  - ICML 2025
  - 图像分割
  - 跨域少样本分割
  - ViT特征解耦
  - 正交空间解耦
  - 交叉模式比较
  - 自适应融合权重
---

# Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation

**会议**: ICML 2025  
**arXiv**: [2506.02677](https://arxiv.org/abs/2506.02677)  
**代码**: 无  
**领域**: 图像分割  
**关键词**: 跨域少样本分割, ViT特征解耦, 正交空间解耦, 交叉模式比较, 自适应融合权重

## 一句话总结

本文发现跨域少样本分割（CD-FSS）中基于距离比较的方法存在特征纠缠问题，其根源在于ViT各层输出在距离计算时的等权交叉匹配，进而提出通过自解耦（Self-Disentanglement）和重组合（Re-Composition）的方式，学习ViT组件间的比较权重来解决该问题。

## 研究背景与动机

跨域少样本语义分割（CD-FSS）旨在将源域数据集的知识迁移到仅有少量标注的未知目标域数据集。现有方法通常通过比较支持集和查询集之间的特征距离来进行mask预测。

**核心痛点**：作者发现这种广泛采用的方法存在一个**特征纠缠问题**——模型倾向于将源域的多种模式（如翅膀和身体）绑定在一起，使得每种模式都难以单独迁移。例如，当模型将"翅膀+身体"两个模式纠缠后，如果目标域图像仅有翅膀但身体与训练数据不同（如另一种蝙蝠），模型就会无法捕捉翅膀，导致分割错误。

**核心矛盾**：CD-FSS场景中源域和目标域之间存在显著的域差距和语义差距，迁移纠缠的模式比迁移解耦的模式困难得多。

**切入角度**：作者利用ViT的可解释性研究发现，残差连接和一致的空间尺寸使得每个ViT组件（MSA、MLP）的输出位于**同一特征空间**中。因此，ViT的最终输出可以自然地看作所有组件的累积组合。基于这一结构分解，作者发现不同层捕获不同的语义模式，而距离计算时所有层间的交叉比较被赋予了相同的权重——有意义的比较（翅膀vs翅膀）和无意义的比较（身体vs翅膀）被等权混合，导致了特征纠缠。

**CKA验证**：作者通过CKA相似度实验验证了这一假设。不同层之间的域相似性差异很大：层级匹配（对角线）的平均CKA远高于最终输出的CKA（如FSS-1000上0.6107 vs 0.4288）。更有趣的是，简单移位（Top-12平均0.8126）甚至超过对角线匹配，暗示可学习的交叉匹配可能优于朴素的逐层匹配。

**核心idea**：通过学习ViT组件间所有比较的权重，自解耦ViT的输出特征并交叉重组合，降低无意义比较的权重，提升有意义比较的权重。

## 方法详解

### 整体框架

整体框架名为SDRC（Self-Disentanglement and Re-Composition），流程如下：

1. 从ViT不同层提取$L$组支持/查询特征，沿通道维度拼接
2. 送入**正交空间解耦（OSD）**模块进行权重分配和语义解耦
3. OSD输出送入**交叉模式比较（CPC）**模块，解耦后的模式交叉比较生成$L^2$组score maps
4. 源域训练使用均匀权重组合score maps，目标域微调引入**自适应融合权重（AFW）**动态学习比较权重
5. 最终预测经双线性插值恢复到原图尺寸后取argmax

### 关键设计

1. **正交空间解耦（OSD）模块**：将各层特征拼接后投影到低维正交空间，显式地解耦不同语义模式并分配权重。

   具体流程：将$L$组特征沿通道拼接得到$F_{con}^* \in \mathbb{R}^{Ld \times n \times n}$，通过三层结构处理：
   - 全连接层$W_{in} \in \mathbb{R}^{Ld \times r}$降维到低秩空间
   - 卷积层$W_{orth} \in \mathbb{R}^{r \times r \times 1 \times 1}$施加正交约束
   - 全连接层$W_{out} \in \mathbb{R}^{r \times Ld}$映射回原空间并拆分

   正交正则化损失（将$F_{orth}$reshape为$\mathbb{R}^{r \times n^2}$后计算）：
   $$L_{orth} = \|F_{orth} F_{orth}^T - I\|_F^2$$

   **设计动机**：通过正交约束促进各通道特征之间的独立性，实现语义解耦。互信息实验验证了效果——使用OSD后支持/查询特征间的MI显著降低（如Chest X-ray从0.91降至0.65）。rank $r$默认设为8以平衡性能与参数量。源域训练时$W_{in}$和$W_{out}$与编码器联合训练；目标域微调时仅微调紧凑的$W_{orth}$（仅64个参数），其余冻结。

2. **交叉模式比较（CPC）模块**：将解耦后的支持集原型和查询特征进行交叉比较，生成$L^2$组score maps进行重组合。

   首先通过Mask Average Pooling从支持特征中获取$L$组前景原型$P_{fg} \in \mathbb{R}^{L \times d \times 1 \times 1}$和背景原型$P_{bg} \in \mathbb{R}^{L \times d \times 1 \times 1}$。然后对$L$组query特征与$L$组原型进行交叉比较：
   $$C_{bg/fg} = distance(F^q, P_{bg/fg}), \quad C = concat(C_{bg}, C_{fg})$$

   其中$C$被reshape为$\mathbb{R}^{L^2 \times 2 \times n \times n}$，2代表背景和前景。默认使用余弦相似度：
   $$distance_{cos} = \frac{F^q \cdot P_{bg/fg}}{\|F^q\| \|P_{bg/fg}\|}$$

   **设计动机**：由于ViT的自注意力机制具有动态性，不同层提取的特征可能存在正确的跨层语义对应关系，因此交叉比较比逐位匹配更有效（实验：59.50% vs 55.14%）。

3. **自适应融合权重（AFW）**：一个极轻量参数矩阵$W_{AFW} \in \mathbb{R}^{L^2 \times 2}$（ViT-B仅288个参数），动态学习不同目标域的重组合权重。

   源域训练时用均匀权重：$C_{fusion} = \frac{\sum_{l=0}^{L^2} C(l)}{L^2}$

   目标域微调时引入AFW：$C_{fusion} = \frac{W_{AFW} \otimes C}{L^2}$

   **设计动机**：AFW参数量极小，如果在源域与编码器联合训练会过拟合源数据（实验验证：源域训练61.01% < 直接目标域引入63.22%）。可视化显示AFW在不同域学到了截然不同的权重分布，且前景/背景权重自发呈互斥趋势。

### 损失函数 / 训练策略

总损失函数为标准BCE损失加正交正则化：
$$L = L_{BCE} + \lambda L_{orth}$$

其中$\lambda = 0.1$，在0.01-0.5范围内性能变化小于1%，调参不敏感。

**两阶段训练策略**：
- **源域训练**：在PASCAL VOC 2012+SBD上训练，OSD与ViT编码器联合优化，score maps用均匀权重组合
- **目标域微调**：冻结$W_{in}$和$W_{out}$，仅微调$W_{orth}$和AFW；由于无法访问query标签，将支持集同时作为query计算$L_{BCE}$和$L_{orth}$

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(1-shot) | 之前SOTA | 提升 |
|--------|------|-------------|----------|------|
| FSS-1000 | mIoU | 80.31 | 79.71 (APSeg) | +0.60 |
| Deepglobe | mIoU | 43.15 | 42.60 (ABCDFSS) | +0.55 |
| ISIC | mIoU | 46.57 | 45.43 (APSeg) | +1.14 |
| Chest X-ray | mIoU | 82.86 | 84.10 (APSeg) | -1.24 |
| **平均** | **mIoU** | **63.22** | **61.30 (APSeg)** | **+1.92** |

| 数据集 | 指标 | 本文(5-shot) | 之前SOTA | 提升 |
|--------|------|-------------|----------|------|
| FSS-1000 | mIoU | 82.55 | 81.90 (APSeg) | +0.65 |
| Deepglobe | mIoU | 46.83 | 50.12 (DRA) | -3.29 |
| ISIC | mIoU | 55.02 | 53.98 (APSeg) | +1.04 |
| Chest X-ray | mIoU | 84.79 | 84.50 (APSeg) | +0.29 |
| **平均** | **mIoU** | **67.30** | **65.42 (DRA)** | **+1.88** |

注：本文使用ViT-B encoder-only结构（FLOPs 18.86G），APSeg同样使用ViT-B但采用基于SAM的encoder-decoder架构，参数量和计算量远大于本文。

### 消融实验

| 配置 | 1-shot平均mIoU | 5-shot平均mIoU | 说明 |
|------|---------------|----------------|------|
| Baseline | 49.88 | 53.64 | 无任何模块 |
| +CPC | 59.50 | 62.68 | +9.62%，最大提升来源 |
| +CPC+AFW | 61.32 | 65.22 | AFW再提升约1.8% |
| +CPC+OSD | 60.75 | 64.45 | OSD再提升约1.3% |
| +CPC+AFW+OSD(完整) | **63.22** | **67.30** | 三模块协同最优 |

| 距离度量 | Baseline(1-shot) | Ours(1-shot) | Baseline(5-shot) | Ours(5-shot) |
|----------|-----------------|-------------|-----------------|-------------|
| Euclidean | 48.92 | 62.49 | 53.07 | 66.53 |
| Dot | 49.18 | 62.75 | 53.03 | 66.58 |
| EMD | 50.02 | **63.37** | 53.23 | 67.01 |
| Cosine | 49.88 | 63.22 | 53.64 | **67.30** |

| OSD rank | 2 | 4 | 8 | 16 | 32 | 64 |
|----------|------|------|------|------|------|------|
| 1-shot mIoU | 60.39 | 61.73 | 63.22 | 63.25 | 63.43 | 62.61 |

### 关键发现

- **CPC是核心贡献**：引入CPC后mIoU提升9.62%（1-shot），说明交叉比较对于解耦纠缠特征至关重要
- **交叉比较优于逐位匹配**：跨层比较（59.50%）比同层比较（55.14%）高出4.36%，验证了ViT层间存在有效的跨层语义对应
- **OSD確实降低了互信息**：在4个目标域上，使用OSD后支持/查询特征间的MI均有下降
- **AFW学到域相关的权重分布**：可视化显示不同目标域的AFW热力图差异很大，最大权重不一定在对角线上，且前景/背景权重自发互斥
- **AFW不应在源域训练**：源域训练（61.01%）反而不如直接在目标域引入（63.22%）
- **正交损失权重不敏感**：$\lambda$在0.01-0.5范围内mIoU变化不足1%
- **计算效率最优**：FLOPs仅18.86G，低于PATNet(22.63G)、HSNet(20.11G)、SSP(18.97G)
- **所有距离度量均有效**：无论使用何种度量，方法均大幅超越baseline（+12-14%）
- **多背景原型提升有限**：引入聚类获取多背景原型仅从63.22%微升至63.59%，不值得额外开销

## 亮点与洞察

1. **从ViT结构解释特征纠缠**：利用ViT残差结构的累积组合特性，将特征纠缠归因为层间交叉匹配的等权处理——$S = \sum_i \sum_j (Layer_s^i \cdot Layer_q^j)$中有意义和无意义的比较被等权混合。分析不仅提供直觉解释，还通过CKA实验严格验证
2. **"自解耦"设计**：与传统特征解耦方法需要额外VAE/GAN网络不同，本文利用ViT本身的结构特性实现解耦，不引入额外分支网络，优雅简洁
3. **理论分析**：通过$\mathcal{H}$-divergence域适应理论论证了方法同时降低源域风险$\epsilon_\mathcal{S}(h)$和域差距$d_\mathcal{H}(\mathcal{S}, \mathcal{T})$的机制
4. **极致轻量**：OSD微调仅需64个参数（$W_{orth}$），AFW仅288个参数，整体FLOPs低于所有对比方法
5. **AFW前景/背景互斥现象**：在Deepglobe和ISIC数据集上自适应学到的前景和背景权重呈互斥关系，这是一个有趣的自发涌现现象

## 局限与展望

1. **Chest X-ray上1-shot低于APSeg**：在胸部X光域上（82.86% vs 84.10%），APSeg的SAM-based架构仍有优势
2. **仅在ViT架构上深入分析**：分析强依赖ViT的残差累加结构，对CNN或非标准残差结构的推广有待探索
3. **源域单一**：仅使用PASCAL VOC作为源域，未探索多源域或更大规模预训练数据的设置
4. **目标域微调仍依赖标注**：虽然是少样本设定，但支持集标注获取成本在某些领域仍然高昂
5. **rank选择较为固定**：rank=8是全局最优的经验值，不同目标域可能需要不同rank
6. **可扩展方向**：与DINOv2、SAM2等更强预训练模型结合；多类别场景下CPC和AFW设计需适配

## 相关工作与启发

- **PATNet (ECCV 2022)**：建立了CD-FSS benchmark和评估协议，通过特征变换层映射域特定特征为域无关特征
- **APSeg (CVPR 2024)**：基于SAM的自动提示网络，同样使用ViT-B但参数量和计算量远大于本文
- **DRA (CVPR 2024)**：使用紧凑adapter对齐不同域特征，5-shot在Deepglobe上优于本文
- **ABCDFSS (CVPR 2024)**：测试时引入微型adaptors做特征细化，思路与本文的目标域AFW类似
- **APM (NeurIPS 2024)**：轻量频率masker做特征增强，与本文的结构解耦思路互补
- **Gandelsman et al. (ICLR)**：发现ViT各组件输出位于同一特征空间，是本文分析的理论基础
- **启发**：ViT结构分解的思路可推广到迁移学习、域适应、多任务学习等需要特征解耦的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 从ViT结构分解角度分析特征纠缠是新颖视角，CKA验证和理论分析增强了说服力；但解耦+重组合的宏观思路在迁移学习中并不罕见
- 实验充分度: ⭐⭐⭐⭐⭐ 消融实验极为全面——模块消融、距离度量比较、rank分析、正交权重敏感性、比较策略对比、互信息验证、CKA域相似性、AFW可视化、计算效率对比、理论分析
- 写作质量: ⭐⭐⭐⭐ 论文逻辑从问题发现→数学分析→CKA验证→方法设计的推进自然流畅，但公式排版略显密集
- 价值: ⭐⭐⭐⭐ 在CD-FSS取得新SOTA且方法轻量高效；ViT结构解耦视角具有可推广性

<!-- RELATED:START -->

## 相关论文

- [Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation](adapter_naturally_serves_as_decoupler_for_cross-domain_few-shot_semantic_segment.md)
- [The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](../../CVPR2025/segmentation/the_devil_is_in_low-level_features_for_cross-domain_few-shot_segmentation.md)
- [Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](../../CVPR2025/segmentation/dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](../../AAAI2026/segmentation/bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [Object-level Correlation for Few-Shot Segmentation](../../ICCV2025/segmentation/object-level_correlation_for_few-shot_segmentation.md)

<!-- RELATED:END -->
