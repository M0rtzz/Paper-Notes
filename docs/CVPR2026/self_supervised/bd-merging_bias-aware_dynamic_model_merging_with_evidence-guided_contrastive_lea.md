---
title: >-
  [论文解读] BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning
description: >-
  [CVPR 2026][自监督学习][Model Merging] 提出 BD-Merging 框架，通过 Dirichlet 证据建模 + 邻域差异分数（ADS）+ 差异感知对比学习，训练去偏路由器来自适应分配模型合并权重，显著提升合并模型在测试时分布偏移和未见任务上的鲁棒性与泛化能力。
tags:
  - "CVPR 2026"
  - "自监督学习"
  - "Model Merging"
  - "Multi-Task Learning"
  - "Evidential Deep Learning"
  - "Distribution Shift"
  - "对比学习"
  - "uncertainty estimation"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning

**会议**: CVPR 2026  
**arXiv**: [2603.03920](https://arxiv.org/abs/2603.03920)  
**代码**: 暂无  
**领域**: 自监督  
**关键词**: Model Merging, Multi-Task Learning, Evidential Deep Learning, Distribution Shift, Contrastive Learning, uncertainty estimation

## 一句话总结

提出 BD-Merging 框架，通过 Dirichlet 证据建模 + 邻域差异分数（ADS）+ 差异感知对比学习，训练去偏路由器来自适应分配模型合并权重，显著提升合并模型在测试时分布偏移和未见任务上的鲁棒性与泛化能力。

## 研究背景与动机

**模型合并（Model Merging）的兴起**：多任务学习需要大量数据和计算资源，且受隐私限制无法共享数据。模型合并通过整合独立微调的检查点，无需重新训练即可实现多任务能力，成为高效替代方案。

**分布偏移下的可靠性被忽视**：现有 MM 方法普遍假设测试数据与训练/辅助数据分布一致，但现实中传感器噪声、传输失真、环境变化等导致测试时输入分布偏移，严重削弱合并模型的性能。

**测试时偏差（Test-time Bias）**：实验表明即使轻微的自然扰动也会导致所有现有方法准确率显著下降（如 Task Arithmetic 在 L3 级别下降 16.8%），说明当前方法缺乏对输入级噪声的鲁棒性。

**未见任务泛化不足**：AdaMerging 在已见任务上达 90.79%，但在未见任务上骤降至 49.83%，暴露出严重的过拟合问题。依赖辅助数据的方法在分布不匹配时反而放大分布差距。

**缺乏细粒度样本级别对齐**：现有方法无法捕捉样本级的分布差异，仅在全局或任务级调整权重，无法应对异构分布偏移带来的冲突知识和有偏集成问题。

**核心洞察**：利用证据不确定性（evidential uncertainty）捕捉分布差异，以此指导自适应表示对齐，是解决 MM 分布偏移问题的关键突破口。

## 方法详解

### 整体框架

BD-Merging 包含三个核心模块：（1）基于 Dirichlet 分布的联合证据头，在统一标签空间上建模不确定性；（2）邻域差异分数（ADS），量化邻域样本间的证据对齐程度；（3）差异感知对比学习机制，引导去偏路由器按样本自适应分配合并权重。整个流程无需标注数据，完全在无监督设置下完成。

### 关键设计一：联合证据头（Joint Evidential Head）

- **功能**：在预训练骨干网络上附加一个证据头，在统一标签空间 $\mathcal{Y} = \bigcup_{k=1}^{K} \mathcal{Y}_k$ 上为每个样本输出 Dirichlet 浓度参数 $\boldsymbol{\alpha}$，据此计算信念质量 $b_c$、不确定性 $u$ 和预测概率 $p_c$。
- **核心思路**：引入类间证据对比（IEC）指标 $\nu = (S / \alpha_{\hat{c}^{(1)}}) \cdot (L / S) \cdot (\alpha_{\hat{c}^{(2)}} / \alpha_{\hat{c}^{(1)}})$，综合考虑预测集中度、类间竞争和语义模糊性，弥补传统 EDL 中总证据量和最高类置信度无法刻画跨任务语义偏移的缺陷。
- **设计动机**：测试时分布偏移会放大重叠标签空间的语义歧义，单一的不确定性度量不足以区分不同类型的预测失败。IEC 通过捕捉类间依赖关系提供更细粒度的不确定性估计。

### 关键设计二：邻域差异分数（Adjacency Discrepancy Score, ADS）

- **功能**：对每个样本 $x_i$，在特征空间中构建半径 $r$ 的邻域集 $\mathcal{A}_r(x_i)$，综合三个互补因子计算 ADS $d_{ik}$。
- **核心思路**：ADS 由三项乘积构成——
    - **Prediction Sharpness**：$\mathrm{Sharp}(x_i) = \mathbb{E}_{x_j \in \mathcal{A}_r} \log(S_j / \max_c \alpha_{jc} - 1)$，衡量邻域整体认识论不确定性；
    - **Semantic Divergence**：$\mathrm{Div}(x_i) = \mathbb{E}_{x_j \in \mathcal{N}_r} \| \boldsymbol{\alpha}_i / S_i - \boldsymbol{\alpha}_j / S_j \|_1$，量化目标样本与邻居间的类级分布偏差；
    - **Opinion Conflicts**：$\mathrm{Conf}(x_i, x_k) = \sum_c |p_{ic} - p_{kc}| \cdot (1-u_i)(1-u_k)$，刻画高置信样本间的信念冲突。
- **设计动机**：单一度量无法全面刻画局部分布差异，三因子联合评估从全局不确定性、语义分歧、置信冲突三个维度提供统一的局部差异视图，有效区分分布内样本与被扰动/未见任务输入。

### 关键设计三：差异感知对比学习与去偏路由器

- **功能**：基于 ADS 阈值 $\epsilon$ 将邻域划分为正样本集 $\mathcal{M}_r^+(i)$（$d_{ik} < \epsilon$）和负样本集 $\mathcal{M}_r^-(i)$（$d_{ik} \geq \epsilon$），构建对比损失拉近一致样本、推远冲突样本。同时训练一个去偏路由器（两层 MLP），基于预训练骨干的 token 嵌入计算逐样本的任务/层级合并权重。
- **核心思路**：路由器输出 $\{w_k\} = \mathrm{softmax}(R(\mathbf{H}))$，合并参数 $\theta^* = \theta_0 + \sum_k w_k \cdot \tau_k$。通过动态权重分配替代固定权重合并，不同输入获得不同的权重组合。
- **设计动机**：固定权重合并在异构任务间易产生干扰。去偏路由器能根据输入特征自适应调整，既减少任务干扰，又提升对未知分布的适应能力。

## 损失函数与训练策略

整体训练分两阶段：

**阶段一：证据头训练**

$$\mathcal{L}_{\mathrm{Head}} = \mathcal{L}_{\mathrm{Ent}} + \gamma \mathcal{L}_{\mathrm{Inv}}$$

- $\mathcal{L}_{\mathrm{Ent}}$：熵最小化 + KL 散度正则到非信息先验，鼓励尖锐预测同时避免过度自信
- $\mathcal{L}_{\mathrm{Inv}}$：逆相关损失，约束不确定性 $u$ 与 IEC $\nu$ 之间保持反比关系

**阶段二：路由器训练**

$$\mathcal{L}_{\mathrm{BD}} = \mathcal{L}_{\mathrm{Unsup}} + \eta \mathcal{L}_{Dis}$$

- $\mathcal{L}_{\mathrm{Unsup}}$：Shannon 熵最小化，增强合并预测的确定性
- $\mathcal{L}_{Dis}$：差异感知对比损失，基于 ADS 划分正负样本对

所有正则系数（$\lambda, \gamma, \eta$）均设为 0.1，训练 300 epochs，batch size 16。

## 实验关键数据

### 表1：测试时偏差下的性能（ViT-B/32，8 个任务平均准确率）

| 方法 | Clean | L1 (↓) | L2 (↓) | L3 (↓) |
|------|-------|--------|--------|--------|
| Task Arithmetic | 69.09 | 64.07 (−7.3%) | 60.30 (−12.7%) | 57.50 (−16.8%) |
| Ties-Merging | 72.92 | 68.14 (−6.6%) | 64.34 (−11.8%) | 61.51 (−15.6%) |
| AdaMerging (Layer) | 74.33 | 69.00 (−7.2%) | 64.46 (−13.3%) | 61.22 (−17.6%) |
| Twin-Merging | 84.10 | 79.13 (−5.9%) | 74.17 (−11.8%) | 70.88 (−15.7%) |
| AdaMerging w/Surgery | 84.40 | 79.02 (−6.4%) | 74.33 (−11.9%) | 70.97 (−15.9%) |
| **BD-Merging (Layer)** | **87.15** | **83.31 (−4.4%)** | **78.78 (−9.6%)** | **75.36 (−13.5%)** |

BD-Merging 在所有扰动级别下均为最优，且性能衰退幅度最小（L1 仅降 4.4% vs 其他 5.9%–7.3%）。

### 表2：已见 vs 未见任务泛化（4 见 + 4 未见）

| 方法 | 已见任务 Avg | 未见任务 Avg |
|------|-------------|-------------|
| AdaMerging | 90.79 | 49.83 |
| Twin-Merging | 93.06 | 53.03 |
| **BD-Merging** | **94.53** | **55.01** |

BD-Merging 同时提升已见和未见任务的准确率，泛化-专化平衡最优。

### 消融实验（Clean / Corrupted L2）

| 变体 | Clean | Corrupted |
|------|-------|-----------|
| BD-Merging | 87.15 | 78.78 |
| w/o Router | 78.31 (−8.84) | 67.25 (−11.53) |
| w/o ADS | 84.48 (−2.67) | 75.44 (−3.34) |
| w/o $\mathcal{L}_{Dis}$ | 83.34 (−3.81) | 74.26 (−4.52) |
| w/o Div(·) | 85.36 (−1.79) | 76.28 (−2.50) |

去偏路由器是最关键组件（移除后下降 ~9/11 点），ADS 三因子中 Div(·) 贡献最大。

## 亮点与洞察

1. **问题定义精准**：首次系统研究模型合并在测试时分布偏移下的可靠性问题，明确提出测试时偏差和未见任务泛化两大挑战。
2. **证据建模与对比学习的巧妙结合**：用 EDL 产生的不确定性信号指导对比学习的正负样本划分，形成闭环——证据建模发现差异 → ADS 量化差异 → 对比学习利用差异。
3. **效率-性能平衡**：BD-Merging 在接近单独微调模型性能的同时，计算开销远低于 AdaMerging w/Surgery 等方法。
4. **路由器可解释性好**：不同未见任务的路由权重分布呈现清晰的任务特异性模式（如 Cars 集中分配 vs SUN397 均匀分配），具有直观的可解释性。

## 局限与展望

1. **仅验证图像分类**：8 个数据集均为图像分类，缺乏对 NLP、多模态等其他模态和任务类型的验证，泛化性存疑。
2. **邻域构建开销**：ADS 需要计算特征空间中的邻域集，对于大规模数据集可能带来额外计算负担，论文未讨论可扩展性。
3. **超参数敏感性**：邻域半径 $r$、阈值 $\epsilon$、多个损失系数均需调优，论文中所有系数简单设为 0.1，但不同场景下可能需要精细调整。
4. **未见任务提升有限**：虽然 BD-Merging 在未见任务上优于基线（55.01% vs 53.03%），但绝对水平仍远低于预训练模型（56.99%），提升空间较大。
5. **路由器结构固定**：两层 MLP 路由器的设计较简单，可探索更复杂的注意力机制或 mixture-of-experts 结构。

## 相关工作与启发

- **Task Arithmetic / Ties-Merging / DARE**：静态权重合并方法，不考虑输入级自适应，在分布偏移下鲁棒性差。BD-Merging 通过动态路由器超越了这类固定权重范式。
- **AdaMerging**：学习任务/层级自适应权重，但基于辅助数据优化，在辅助分布与测试分布不匹配时会过拟合。BD-Merging 的证据引导机制直接在测试特征上建模不确定性，减少对辅助数据分布的依赖。
- **Twin-Merging**：动态整合模块化知识，效率较高但精度受限。BD-Merging 在精度和效率之间取得更好的平衡。
- **Surgery**：通过外科手术式调整提升合并质量，但计算开销大。BD-Merging 达到相近精度的同时开销更小。
- **Evidential Deep Learning**：BD-Merging 将 EDL 从传统 OOD 检测任务扩展到模型合并场景，为 EDL 的应用开辟了新方向。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将证据学习引入模型合并并设计 ADS + 对比学习闭环，思路新颖且工程完整
- **实验充分度**: ⭐⭐⭐⭐ — 多级扰动、已见/未见任务、消融、路由器分析、多骨干验证覆盖全面，但缺乏非视觉任务
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰、公式推导完整、图表丰富，问题定义和方法衔接流畅
- **价值**: ⭐⭐⭐⭐ — 首次系统研究 MM 的分布偏移鲁棒性，对实际部署场景有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] UniGeoCLIP: Unified Geospatial Contrastive Learning](unigeoclip_geospatial_contrastive.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](../../NeurIPS2025/self_supervised/uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[ICML 2025\] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](../../ICML2025/self_supervised/what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [\[CVPR 2026\] MOMO: Mars Orbital Model — Foundation Model for Mars Orbital Applications](momo_mars_orbital_model_foundation_model_for_mars_orbital_applications.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](../../NeurIPS2025/self_supervised/starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)

</div>

<!-- RELATED:END -->
