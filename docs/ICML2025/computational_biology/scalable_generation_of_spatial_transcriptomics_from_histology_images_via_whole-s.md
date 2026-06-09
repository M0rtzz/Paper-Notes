---
title: >-
  [论文解读] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching
description: >-
  [ICML2025][计算生物][空间转录组] 提出 STFlow，一种基于 flow matching 的生成模型，通过建模整张切片的基因表达联合分布来显式捕获细胞间交互，并采用局部空间注意力实现高效全切片编码，在 HEST-1k 和 STImage-1K4M 上相对最优基线提升 18%。
tags:
  - "ICML2025"
  - "计算生物"
  - "空间转录组"
  - "Flow Matching"
  - "全切片建模"
  - "细胞间交互"
  - "E(2)-不变性"
  - "生成模型"
---

# Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching

**会议**: ICML2025  
**arXiv**: [2506.05361](https://arxiv.org/abs/2506.05361)  
**代码**: [GitHub](https://github.com/Graph-and-Geometric-Learning/STFlow)  
**领域**: 空间转录组 / 计算病理学  
**关键词**: 空间转录组, Flow Matching, 全切片建模, 细胞间交互, E(2)-不变性, 生成模型

## 一句话总结

提出 STFlow，一种基于 flow matching 的生成模型，通过建模整张切片的基因表达联合分布来显式捕获细胞间交互，并采用局部空间注意力实现高效全切片编码，在 HEST-1k 和 STImage-1K4M 上相对最优基线提升 18%。

## 研究背景与动机

空间转录组（ST）技术能在组织切片中同时获取空间位置和基因表达信息，为理解细胞交互和微环境提供了新视角。然而，传统 ST 实验通量低、设备要求高，催生了从 H&E 染色组织切片图像（WSI）预测基因表达的研究方向。

现有方法存在三大瓶颈：

**独立预测**：将每个 spot 的基因表达独立建模 $p(\boldsymbol{Y}_i|\boldsymbol{I}_i)$，忽略了相邻细胞间基因调控的交互关系

**内存瓶颈**：slide-level 方法对所有 spot 做全局注意力（常超过 10,000 个 spot），导致 O(N²) 的计算/内存开销，在标准硬件上 OOM

**坐标编码脆弱**：将坐标作为位置编码，对数值噪声和批次效应敏感

STFlow 的核心动机：将回归任务重构为**联合分布的生成建模问题**，用 flow matching 的迭代去噪框架显式建模 $p(\boldsymbol{Y}_0,\cdots,\boldsymbol{Y}_N|\boldsymbol{I}_0,\cdots,\boldsymbol{I}_N)$，每一步去噪都以当前基因表达预估作为上下文，从而天然捕获细胞间交互。

## 方法详解

### 整体框架

STFlow 分三步：

1. **Spot 编码**：用预训练病理基础模型 $f_{\text{PFM}}$ 提取每个 spot 的视觉特征 $\boldsymbol{Z}_i = f_{\text{PFM}}(\boldsymbol{I}_i)$
2. **Slide-level 空间编码**：通过局部空间注意力聚合 k-近邻信息，融合坐标几何关系
3. **Flow matching 迭代优化**：从先验分布采样初始"基因表达猜想"，经多步去噪收敛到最终预测

### Flow Matching 学习框架

**训练目标**：给定 WSI 的坐标 $\boldsymbol{C}\in\mathbb{R}^{N\times 2}$、spot 图像 $\boldsymbol{I}$ 和真实基因表达 $\boldsymbol{Y}\in\mathbb{R}^{N\times G}$，最小化去噪损失：

$$\min_{\theta} \text{MSE}\left(\boldsymbol{Y},\; f_{\theta}(\boldsymbol{Y}_t, \boldsymbol{I}, \boldsymbol{C}, t)\right)$$

其中 $t \sim \text{Uniform}[0,1]$，$\boldsymbol{Y}_t = t \cdot \boldsymbol{Y} + (1-t) \cdot \boldsymbol{Y}_0$ 为真实表达与先验采样的线性插值。

**推理过程**（Euler ODE 求解）：从 $\boldsymbol{Y}_0 \sim \mathcal{Z}(\mu,\phi,\pi)$ 出发，经 $S$ 步迭代：

$$\boldsymbol{Y}_{t_2} = \boldsymbol{Y}_{t_1} + \frac{\hat{\boldsymbol{Y}} - \boldsymbol{Y}_{t_1}}{1 - t_1} \cdot (t_2 - t_1)$$

最后一步直接输出 $\hat{\boldsymbol{Y}}$ 作为预测。

### 先验分布：零膨胀负二项分布（ZINB）

作者观察到基因表达数据有两个特点：(1) 大量基因未激活（零值占主导），(2) 过度离散（方差 > 均值）。因此采用 ZINB 分布 $\mathcal{Z}(\mu, \phi, \pi)$ 作为先验，用 $\pi$ 建模零膨胀比例，用负二项分量建模过度离散。这是 flow matching 相对 diffusion 的关键优势——可自由选择非高斯先验。

### E(2)-不变空间注意力

**局部上下文**：对每个 spot $i$，仅关注其 k-近邻 $\mathcal{N}(i)$，用方向向量 $\boldsymbol{C}_{i\to j} = \boldsymbol{C}_i - \boldsymbol{C}_j$ 编码空间关系。

**Frame Averaging 实现不变性**：对方向向量集合做 PCA 提取 4 个帧（两个主成分的 ±1 组合），将坐标投影到各帧后取平均：

$$\boldsymbol{C}'_{i\to j} = \frac{1}{|\mathcal{F}|} \sum_g \text{MLP}(\boldsymbol{C}^{(g)}_{i\to j})$$

这确保了对切片旋转、平移、反射（E(2) 变换）的不变性。

**注意力计算**：将视觉特征 QKV、空间编码、基因表达差异 $(Y_{t,i} - Y_{t,j})$ 联合送入 MLP 注意力：

$$\boldsymbol{A}_{ij} = \text{Softmax}_i\left(\text{MLP}(\boldsymbol{Z}_{Q,i} \| \boldsymbol{Z}_{K,j} \| \boldsymbol{C}'_{i\to j} \| (\boldsymbol{Y}_{t,i} - \boldsymbol{Y}_{t,j}))\right)$$

**聚合更新**：

$$\boldsymbol{Z}'_i = \text{MLP}\left(\sum_{j\in\mathcal{N}(i)} \boldsymbol{A}_{ij}\boldsymbol{Z}_{V,j} \| \sum_{j\in\mathcal{N}(i)} \boldsymbol{A}_{ij}\boldsymbol{C}'_{i\to j}\right) + \boldsymbol{Z}_i$$

每层同时输出基因表达更新 $\boldsymbol{Y}'_{t,i} = \text{MLP}(\boldsymbol{Z}'_i)$，多层输出取平均得最终预测。

## 实验关键数据

### 基准与指标

- **基准**：HEST-1k（9 个癌种）+ STImage-1K4M（8 种组织），共 17 个数据集
- **指标**：Pearson 相关系数（基因维度平均）
- **基线**：5 个 spot-based（Ciga, UNI, Gigapath, STNet, BLEEP）+ 3 个 slide-based（Gigapath-slide, HisToGene, TRIPLEX）

### 主实验结果（Pearson Corr，越高越好）

| 数据集 | UNI | Gigapath | BLEEP | TRIPLEX | **STFlow** |
|--------|------|----------|-------|---------|-----------|
| IDC | 0.520 | 0.513 | 0.533 | 0.606 | **0.587** |
| PRAD | 0.371 | 0.384 | 0.382 | 0.402 | **0.421** |
| PAAD | 0.432 | 0.436 | 0.459 | 0.492 | **0.507** |
| SKCM | 0.629 | 0.590 | 0.566 | 0.699 | **0.704** |
| COAD | 0.285 | 0.290 | 0.303 | 0.319 | **0.326** |
| CCRCC | 0.178 | 0.187 | 0.298 | 0.289 | **0.332** |
| HCC | 0.052 | 0.051 | 0.086 | 0.062 | **0.124** |
| LUNG | 0.559 | 0.569 | 0.588 | 0.601 | **0.610** |
| **HEST 均值** | 0.344 | 0.344 | 0.368 | 0.395 | **0.415** |

- HEST-1k 上 STFlow 均值 0.415，相对最强基线 TRIPLEX（0.395）提升 **+5.1%**
- 相对病理基础模型（UNI/Gigapath ~0.344）提升约 **+20.6%**
- HCC 数据集上从 0.062→0.124，提升达 **100%**
- Gigapath-slide 在 IDC/COAD/Stomach 等大规模数据集上 OOM，STFlow 无此问题

### 效率分析

- 相比 HisToGene/TRIPLEX 等全局注意力方法，STFlow 使用局部 k-近邻注意力，内存复杂度从 O(N²) 降至 O(Nk)
- 在 spot 数超万的切片上仍可正常运行（其他 slide-based 方法 OOM）

## 亮点与洞察

1. **回归→生成的范式转换**：将基因表达预测从单步回归重构为 flow matching 生成过程，迭代去噪天然编码了细胞间基因表达依赖，这一思路新颖且有理论支撑
2. **ZINB 先验**：利用 flow matching 可选任意先验的优势，引入符合基因表达分布特征（零膨胀 + 过度离散）的先验，比标准高斯更合理
3. **E(2)-不变性设计**：Frame Averaging + 局部空间注意力，既编码了空间依赖又保证了对坐标变换的不变性，物理意义清晰
4. **可扩展性**：k-近邻局部注意力解决了全切片数万 spot 的内存瓶颈，同时通过多层堆叠仍可捕获长程依赖
5. **生物学解释性强**：注意力权重中融入基因表达差异 $(Y_{t,i} - Y_{t,j})$，直接建模细胞间基因调控信号

## 局限与展望

1. **推理速度**：flow matching 需要多步迭代采样（S 步 ODE 求解），推理速度慢于单步回归方法；可探索蒸馏或 consistency model 加速
2. **先验参数估计**：ZINB 先验的 $\mu, \phi, \pi$ 从训练集估计，跨数据集泛化性有待验证
3. **局部注意力的局限**：k-近邻机制可能遗漏跨区域的远程细胞交互（如免疫细胞浸润），虽然多层堆叠可部分缓解但信息衰减
4. **数据集多样性**：主要在 Visium 10x 平台数据上验证，对更高分辨率的 ST 技术（如 MERFISH、Slide-seq）的适用性未探讨
5. **缺少下游任务验证**：虽提到生物标志物基因预测，但未系统评估在细胞类型注释、空间域识别等下游任务的效果

## 相关工作与启发

- **TRIPLEX**（Chung et al., 2024）：当前最强 slide-based 基线，但用全局注意力导致 OOM
- **BLEEP**（Xie et al., 2023）：对比学习 spot-based 方法，在部分数据集上表现不错
- **Flow Matching**（Lipman et al., 2022）：基础生成框架，STFlow 展示了其在非图像领域的成功应用
- **Frame Averaging**（Puny et al., 2021）：几何深度学习中实现 E(n) 不变性的通用技术
- **启发**：生成模型作为"结构化回归"的范式值得推广到其他多输出预测任务（如多基因组学联合预测）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 回归→生成范式切换 + ZINB 先验 + E(2)-不变空间注意力，三重创新
- 实验充分度: ⭐⭐⭐⭐ — 17 个数据集、8 个基线全面对比，但缺少下游任务验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，技术表述严谨，图示直观
- 价值: ⭐⭐⭐⭐⭐ — 为空间转录组预测建立了新的 SOTA，方法通用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow](efficient_molecular_conformer_generation_with_so3-averaged_flow_matching_and_ref.md)
- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](../../NeurIPS2025/computational_biology/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[ICML 2025\] Improving Flow Matching by Aligning Flow Divergence](improving_flow_matching_by_aligning_flow_divergence.md)
- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)
- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)

</div>

<!-- RELATED:END -->
