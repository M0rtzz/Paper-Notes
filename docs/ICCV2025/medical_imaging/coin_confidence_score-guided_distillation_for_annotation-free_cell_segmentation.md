---
title: >-
  [论文解读] COIN: Confidence Score-Guided Distillation for Annotation-Free Cell Segmentation
description: >-
  [ICCV 2025][医学图像][细胞实例分割] 提出COIN框架，通过无监督语义分割+最优传输的像素级细胞传播、基于模型-SAM一致性的实例级置信度评分、以及置信度引导的递归自蒸馏三步策略，解决了无标注细胞实例分割中"无错误实例缺失"的关键问题，在MoNuSeg和TNBC上超越半监督/弱监督方法。
tags:
  - "ICCV 2025"
  - "医学图像"
  - "细胞实例分割"
  - "无标注"
  - "置信度评分"
  - "自蒸馏"
  - "最优传输"
---

# COIN: Confidence Score-Guided Distillation for Annotation-Free Cell Segmentation

**会议**: ICCV 2025  
**arXiv**: [2503.11439](https://arxiv.org/abs/2503.11439)  
**代码**: [https://shjo-april.github.io/COIN/](https://shjo-april.github.io/COIN/)  
**领域**: 医学影像  
**关键词**: 细胞实例分割, 无标注, 置信度评分, 自蒸馏, 最优传输

## 一句话总结

提出COIN框架，通过无监督语义分割+最优传输的像素级细胞传播、基于模型-SAM一致性的实例级置信度评分、以及置信度引导的递归自蒸馏三步策略，解决了无标注细胞实例分割中"无错误实例缺失"的关键问题，在MoNuSeg和TNBC上超越半监督/弱监督方法。

## 研究背景与动机

细胞实例分割（CIS）对于理解组织病理图像中的细胞形态至关重要。虽然无监督CIS（UCIS）方法试图摆脱标注依赖，但现有UCIS模型（如SSA、PSM）存在一个根本问题：**无法产出任何一个IoU=1.0的无错误实例**。

作者识别了两个核心原因：(1) 基于几何增强的学习（如PSM的旋转增强）偏向于几何变化明显的特征（如拉长形状），忽略了圆形或细微的细胞，导致实例掩码不完整；(2) 无差别地接受所有伪标签会传播错误，因为无法区分可靠与噪声预测。

这激发了两个关键假设：(a) 需要更好的细胞检测策略来保证无错误实例的存在；(b) 需要一种无监督的准确性度量来筛选高置信实例用于训练。

## 方法详解

### 整体框架

COIN 分为三步：Step 1——像素级细胞传播，通过USS+OT确保检测到所有细胞并保证无错误实例的存在；Step 2——实例级置信度评分，利用模型预测与SAM精化掩码的一致性衡量每个实例的可靠性；Step 3——置信度引导的递归自蒸馏，逐步扩大高置信实例集合并提升模型性能。

### 关键设计

1. **像素级细胞传播（Step 1）**: 首先利用无监督语义分割模型（如DINOv2/MAE）提取特征图 $F^{us} = U(I_k)$，同时获取现有UCIS模型（如SSA）的初始掩码 $M_\theta^{ucis}$。通过类级平均池化（CAP）得到USS质心 $V_\theta^{us}$，与特征图聚类生成相似度图 $S_\theta^{us} = \text{ReLU}(\text{sim}(F^{us}, V_\theta^{us}))$。关键创新在于引入**最优传输（OT）**进行精化：$S_\theta^{OT} = f_{OT}(S_\theta^{us}) \cdot S_\theta^{us}$，OT通过找到最优分配矩阵避免像素重叠，特别有利于少数类（边缘细胞）的检测，减少假阴。USS模型在自然图像上训练，直接用于病理图像会产生高假阳率（基于颜色区分而非细胞形态），OT有效缓解了这一问题。

2. **实例级置信度评分（Step 2）**: 对传播掩码使用连通组件标记+分水岭算法分离出N个实例 $\{E_{\theta_t}^i\}$。取每个实例的中心点作为SAM的点提示，生成伪GT掩码。核心指标为模型预测与SAM输出的IoU一致性：$C_{\theta_t}^i = \text{IoU}(E_{\theta_t}^i, \text{SAM}(E_{\theta_t}^i))$。使用非参数阈值 $\delta_k$（均值+标准差）筛选：$C > \delta_k$ 的实例标记为可信（$\hat{M}=1$），$C=0$ 的区域标记为确定背景（$\hat{M}=0$），其余拒绝不参与训练（$\hat{M}=-1$）。这一设计的巧妙之处在于：**不直接使用SAM输出**（因SAM对提示敏感，单独使用会出错），而是通过交叉验证筛选一致性高的实例——高一致性意味着模型和SAM都"同意"的区域大概率是正确的。

3. **置信度引导的递归自蒸馏（Step 3）**: 用筛选出的可信实例构建二值伪掩码 $\hat{M}_{bin}^i(t)$ 和边缘伪掩码 $\hat{M}_{edge}^i(t)$（通过Canny算法获得）。总损失 $\mathcal{L}(t) = \mathcal{L}_{seg}(M_{bin}^{ucis}, \hat{M}_{bin}^i(t)) + \mathcal{L}_{seg}(M_{edge}^{ucis}, \hat{M}_{edge}^i(t))$，其中 $\mathcal{L}_{seg} = \mathcal{L}_{ce} + \mathcal{L}_{dice}$。添加边缘解码器的目的是增强对相邻细胞边界的区分能力。随着模型参数 $\theta_t$ 更新，接受集合 $\mathcal{A}_\delta$ 动态变化，逐轮增加高置信实例数量。

### 损失函数 / 训练策略

分割损失 = 交叉熵 + Dice损失，分别应用于二值掩码和边缘掩码。训练过程为递归自蒸馏：每轮用当前模型生成新的伪标签→筛选高置信实例→用可信实例再训练→模型改进后更多实例变为可信→循环。在单张NVIDIA RTX A100 80GB上训练。

## 实验关键数据

### 主实验

| 方法 | 监督类型 | MoNuSeg AJI↑ | MoNuSeg PQ↑ | MoNuSeg IoU↑ | TNBC AJI↑ |
|------|---------|-------------|------------|-------------|-----------|
| SSA | 无标注 | 0.259 | 0.185 | 0.618 | 0.273 |
| PSM | 无标注 | 0.471 | 0.355 | 0.689 | - |
| **SSA+COIN** | **无标注** | **0.580** | **0.536** | **0.776** | **0.568** |
| **PSM+COIN** | **无标注** | **0.579** | **0.539** | **0.777** | - |
| SPPNet | 点标注 | 0.497 | 0.392 | 0.709 | - |
| InstaSAM | 点标注 | 0.574 | - | 0.772 | - |
| TextDiff | 掩码+文本 | 0.510 | 0.410 | 0.726 | 0.464 |

COIN使SSA的AJI从0.259提升至0.580（+124%），且超越了所有需要点/框标注的弱监督方法。

### 消融实验

| 配置 | MoNuSeg AJI↑ | MoNuSeg PQ↑ | 说明 |
|------|-------------|------------|------|
| SSA基线 | 0.259 | 0.185 | 原始UCIS模型 |
| +Step1(USS+OT) | ~0.35 | ~0.28 | 传播恢复无错误实例 |
| +Step1+Step2(评分) | ~0.45 | ~0.40 | 筛选高置信实例 |
| +Step1+Step2+Step3(蒸馏) | **0.580** | **0.536** | 递归扩展置信集 |

在6个数据集（MoNuSeg, TNBC, BRCA, CPM-17, CryoNuSeg, PanNuke）上SSA+COIN均有显著提升（AJI提升0.08-0.17），验证了方法的通用性和可扩展性。

### 关键发现

- OT在细胞传播中至关重要：与K-means、谱聚类等替代方案相比，OT对少数类（细胞）的保护效果最好。
- 置信度评分有效逼近GT质量：高置信实例的AJI分数接近1.0，而随机采样的实例AJI分布较宽。
- COIN是模型无关的框架：可与SSA和PSM两种不同的UCIS基线结合，均获得大幅提升。
- 边缘解码器对区分密集排列的相邻细胞至关重要。

## 亮点与洞察

- "无错误实例缺失"问题的识别非常精准：指出现有UCIS方法连一个完美实例都无法产出，是一个容易被忽视但至关重要的观察。
- 模型-SAM一致性作为无监督准确性代理的想法巧妙：利用两个独立系统的"共识"来估计质量，无需任何人工标注。
- 三步法的递进设计逻辑自洽：从提高灵敏度（Step1）到精确筛选（Step2）到扩展推广（Step3），每步解决一个具体问题。

## 局限与展望

- 依赖SAM作为外部知识源，但SAM本身对组织病理图像的适应性有限，可能在某些组织类型上失效。
- 递归自蒸馏的收敛性和稳定性缺乏理论保证，迭代次数需要经验选择。
- 当前仅处理细胞核分割，对于更复杂的细胞结构（如细胞质、细胞膜）的扩展未讨论。
- OT的计算开销在大规模图像上可能成为瓶颈。

## 相关工作与启发

- 将无监督实例分割（UIS，如CutLER）思路引入病理图像，但针对密集细胞场景做了关键适配。
- SAM在弱监督分割中的应用已有多种（SPPNet用点、InstaSAM用点），COIN首次实现了完全无标注的SAM应用。
- 递归自训练/自蒸馏策略在无监督学习中常见，但结合置信度评分的动态筛选是本文的特色。

## 评分

- 新颖性: ⭐⭐⭐⭐ "无错误实例缺失"问题的定义和三步解决方案新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 6个数据集、多种基线对比、完善的消融和可视化
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Fig.1-6的图示设计出色
- 价值: ⭐⭐⭐⭐⭐ 无标注方法超越弱监督方法，对病理AI实际价值巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] The Four Color Theorem for Cell Instance Segmentation](../../ICML2025/medical_imaging/the_four_color_theorem_for_cell_instance_segmentation.md)
- [\[ICCV 2025\] Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation](alleviating_textual_reliance_in_medical_language-guided_segmentation_via_prototy.md)
- [\[ICLR 2026\] DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring](../../ICLR2026/medical_imaging/disco_densely-overlapping_cell_instance_segmentation_via_adjacency-aware_collabo.md)
- [\[ICCV 2025\] Integrating Biological Knowledge for Robust Microscopy Image Profiling on De Novo Cell Lines](integrating_biological_knowledge_for_robust_microscopy_image_profiling_on_de_nov.md)
- [\[NeurIPS 2025\] EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging](../../NeurIPS2025/medical_imaging/ewc-guided_diffusion_replay_for_exemplar-free_continual_learning_in_medical_imag.md)

</div>

<!-- RELATED:END -->
