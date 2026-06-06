---
title: >-
  [论文解读] TriForces: Augmenting Atomistic GNNs for Transferable Representations
description: >-
  [ICML 2026][物理/科学计算][MLIP] TriForces 把原子级图神经网络拆成「组成-结构-交互」三条平行流，再叠加 LeJEPA + 去噪 + 掩码的多目标自监督预训练，让 MLIP 在小样本迁移、跨域微调和相似结构检索三种场景下都比单流基座更稳。
tags:
  - "ICML 2026"
  - "物理/科学计算"
  - "MLIP"
  - "自监督预训练"
  - "三流架构"
  - "SOAP 描述子"
  - "迁移学习"
---

# TriForces: Augmenting Atomistic GNNs for Transferable Representations

**会议**: ICML 2026  
**arXiv**: [2605.20581](https://arxiv.org/abs/2605.20581)  
**代码**: https://github.com/Ramlaoui/triforces (有)  
**领域**: 物理 / 原子级机器学习势 / 几何图神经网络  
**关键词**: MLIP, 自监督预训练, 三流架构, SOAP 描述子, 迁移学习  

## 一句话总结
TriForces 把原子级图神经网络拆成「组成-结构-交互」三条平行流，再叠加 LeJEPA + 去噪 + 掩码的多目标自监督预训练，让 MLIP 在小样本迁移、跨域微调和相似结构检索三种场景下都比单流基座更稳。

## 研究背景与动机

**领域现状**：基于 DFT 数据训练的机器学习原子间势 (MLIP) 已经成为材料发现与分子动力学的主力工具。MACE、eSEN、Orb-v3 这类几何 GNN 在 OMat24、MPtrj 等大规模数据上预测能量与力的精度已经接近 DFT 自身的误差。

**现有痛点**：实际应用几乎总是要在小而贵的下游数据上微调，而当前 MLIP 的迁移性极不稳定 —— 同一个在 100M 结构上预训练好的模型，在「预测晶系」「预测多数元素」这种诊断任务上甚至微调不动。换会议、换泛函、换体系都会导致性能大幅波动。

**核心矛盾**：表征是为预测能量/力优化的，而不是为复用优化的。监督训练把组成信息和几何信息缠绕在同一个潜向量里，下游若只关心组成或只关心几何，就拿不到干净的可重用表征；自监督学习虽然在视觉/语言里证明能保留语义结构，但原子领域之前主要把 SSL 当成辅助损失，没系统验证 SSL 如何与架构归纳偏置交互。

**本文目标**：拆解为两个子问题 —— (1) 如何让架构本身就显式保留组成与几何信息；(2) 如何让 SSL 在低数据迁移、表征组织、检索这些「预测之外」的任务上真正起作用。

**切入角度**：作者观察到「能量与力的耦合梯度」在保守式训练中会互相竞争，而现有方法靠分阶段调度、特殊初始化等 trick 缓解。如果把组成相关的「保力」自由度独立出来，就有希望同时把能量 MAE 拉低、又不牺牲力 MAE。

**核心 idea**：用「三流分解 + 多目标 SSL」替代单流监督训练，让组成、结构、交互三类信息在表征空间里有专属通道。

## 方法详解

### 整体框架
输入是原子结构 $\mathcal{X}=(\{z_i\},\{\mathbf{x}_i\})$，包含原子序数 $z_i$ 与位置 $\mathbf{x}_i$，构图用半径截断。TriForces 把节点级表征拆成三段拼接：$\mathbf{h}_i=[\mathbf{h}^{\text{comp}}_i,\mathbf{h}^{\text{struct}}_i,\mathbf{h}^{\text{int}}_i]$。预训练在 LeMat-Bulk 的 5M 体相结构上跑，三个 SSL 目标共享同一组随机增广视图；下游微调时三流一起送入预测头。文中区分三种变体：TriForces-Streams（仅架构，随机初始化）、TriForces（架构+SSL）、Base+SSL（仅 SSL）。

### 关键设计

1. **三流分解中的组成 Transformer（带计数加权注意力）**:

    - 功能：从原子列表中只用「元素 + 出现次数」抽出纯化学指纹，与几何完全解耦。
    - 核心思路：把结构压缩成 $T$ 个唯一元素 token $\{(z_t,c_t)\}$，每个 token 用可学习的元素 embedding $\mathbf{u}_t=\mathbf{e}(z_t)$ 初始化，再过 Transformer。关键改动是在注意力 logits 上加一个 log-count 偏置 $a^{(h)}_{ts}=\frac{(\mathbf{q}_t^{(h)})^\top \mathbf{k}_s^{(h)}}{\sqrt{d_h}}+\log(c_s)$，等价于「让 token 之间的注意力等于对所有该类型原子做注意力」，但复杂度从 $\mathcal{O}(N^2)$ 降到 $\mathcal{O}(T^2)$ 且与原子数无关。
    - 设计动机：之前的组成模型（Roost、CrabNet）把化学计量比归一化为分数，丢掉了体系尺寸信息；TriForces 保留绝对计数 $c_t$，因为它本身就编码了能量量级等物理信息。

2. **类型无关的结构流（SOAP 风格幂谱）**:

    - 功能：用旋转不变的几何描述子捕捉跨化学体系共享的几何模体，与具体元素身份解耦。
    - 核心思路：对每个邻居位移 $\mathbf{r}_{ij}$ 用径向基 $\phi_k(r)$（Bessel/Gaussian）、实球谐 $Y_{lm}(\hat{\mathbf{r}})$ 与多尺度截断函数 $s_s(r)$ 联合展开，得到混合通道 $\tilde{\phi}_\alpha(r_{ij})=\sum_{k,s}\mathbf{W}_{\alpha,(k,s)}s_s(r_{ij})\phi_k(r_{ij})$，再累加成局部密度系数 $c_{\alpha lm}(i)$，最后通过幂谱 $p_{\alpha\alpha' l}(i)=\sum_m c_{\alpha lm}(i)c_{\alpha' lm}(i)$ 强制旋转不变；随后接少量不变消息传递层叠加连通拓扑。
    - 设计动机：保守式 MLIP 中力是能量对位置的梯度，组成流加进来的额外自由度若不破坏几何依赖关系就可以「保力」；类型无关的结构流让能量梯度只通过交互流传播，从理论上避免了能量与力损失梯度互相竞争（附录给了 rank-based 界）。

3. **三目标互补的 SSL 预训练**:

    - 功能：用同一组随机增广（位置噪声 + 原子类型掩码 + 随机图构造 + 非等变模型加旋转）下的两个视图同时驱动三个互补目标。
    - 核心思路：总损失 $\mathcal{L}=\mathcal{L}_{\text{denoise}}+\lambda_{\text{mask}}\mathcal{L}_{\text{mask}}+\lambda_{\text{LeJEPA}}\mathcal{L}_{\text{LeJEPA}}$。去噪 $\mathcal{L}_{\text{denoise}}=\sum_i\|f_\theta(\tilde{\mathcal{G}})_i-\boldsymbol{\epsilon}_i\|^2$ 稳定几何表征；掩码 $\mathcal{L}_{\text{mask}}=-\sum_{i\in\mathcal{M}}\log p_\theta(z_i\mid\tilde{\mathcal{G}})$ 强化组成流学习元素共现模式；LeJEPA 在节点和图两个粒度上对齐两视图，并用 SIGReg 正则把表征压成各向同性高斯，避免坍缩同时不需要 stop-gradient 或动量编码器。
    - 设计动机：单纯非重建目标只能拉对齐，会丢精细几何与化学差异；单纯重建目标又不能整理潜空间组织。三者结合后，去噪强化结构流、掩码强化组成流、LeJEPA 强化整体可分性，正好对应三条架构流。

### 损失函数 / 训练策略
预训练在 LeMat-Bulk 5M 体相结构上从零初始化 eSEN、Orb-v3、MACE 三种 backbone。微调时把三流表征拼接送入下游预测头。OMat24 微调跑 2 epoch，MatBench 用标准 5 折交叉验证。

## 实验关键数据

### 主实验：OMat24 微调（4M 子集）

| Backbone / 模式 | 配置 | E MAE (meV/atom) ↓ | F MAE (meV/Å) ↓ | σ MAE (meV/Å³) ↓ |
|------|------|------|------|------|
| Orb-v3 Conservative | Baseline | 107 | 150 | 7.8 |
| Orb-v3 Conservative | + Streams | 35.6 | 149 | 6.2 |
| Orb-v3 Conservative | + TriForces (full) | **19.4** | **95.5** | **4.7** |
| eSEN (equivariant) | Baseline | 104 | 80.3 | 6.3 |
| eSEN (equivariant) | + TriForces (full) | **18.8** | **78.0** | **4.4** |
| MACE (equivariant) | Baseline | 117 | 150 | 8.1 |
| MACE (equivariant) | + TriForces (full) | **34.3** | **142** | **6.1** |

Orb-v3 保守式上能量 MAE 从 107 砍到 19.4，相对提升 82%；同时力 MAE 从 150 降到 95.5，验证了「组成流加进来不破坏力」的理论预期。

### 消融：MatBench 8 任务（vs DFT 标注预训练 baseline）

| 任务（单位） | MACE† | TriForces MACE | Orb† | TriForces Orb | eSEN† | TriForces eSEN |
|------|------|------|------|------|------|------|
| Phonons (cm⁻¹) | 36.7 | 27.6 | 26.2 | 22.6 | 57.8 | **19.5** |
| Log GVRH | 0.082 | 0.073 | 0.063 | 0.058 | 0.093 | 0.058 |
| Perovskites (meV) | 61.4 | 35.1 | 30.7 | 26.5 | 40.1 | **25.6** |
| MP Gap (eV) | 0.370 | 0.250 | 0.194 | 0.132 | 0.392 | 0.139 |
| MP E Form (meV/atom) | 40.8 | 34.4 | 21.1 | 17.1 | 83.5 | 20.2 |

TriForces 在 8 个任务里拿下 6 个 best overall，且全程是自监督预训练，不需要 DFT 标签。Phonons 上 eSEN 从 57.8 直接掉到 19.5。

### 关键发现
- 大规模监督场景下（OMat24 全集），架构分流是主贡献，SSL 只小幅提升最终精度但显著加快收敛；低数据场景下 SSL 才是关键 —— 20K 样本时 TriForces 把能量 MAE 从 81.3 降到 34.6，减幅 57%。
- 保守式（力 = 能量梯度）模型从 TriForces 中获益最大，印证了组成流提供「保力自由度」的假设。
- 与同参数量加宽的 baseline 对比（附录 B.7），TriForces 在 8/8 MatBench 任务和 6/7 QM9 目标上仍占优，排除了「靠参数堆」的解释。
- 学到的潜空间支持按化学、按几何或联合的可分解相似结构检索，开辟了 MLIP 表征「预测之外」的新用法。

## 亮点与洞察
- **架构分流 ≠ 简单多任务**：把组成、结构、交互拆成三条独立编码路径，让 SSL 三目标能各管一摊，互不干扰。这种「架构归纳偏置 × SSL 目标」的对齐设计很值得借鉴到其他多模态/多任务场景。
- **计数加权注意力**是个干净的小 trick：把「按元素去重 + log-count 偏置」与「对所有原子做注意力」严格等价化，既省算力又保留物理意义，可以直接复用到任何 set-based 的化学/分子表征里。
- **能量-力梯度耦合的架构级解法**：之前各家是用调度策略或扩散预训练绕开能量/力损失竞争，TriForces 直接从架构层面给出「保力自由度」，让保守式模型可以一次性训到位，省掉了一堆调参 trick。
- **SSL 角色再定位**：作者明确说 TriForces 不是「又一个 SSL 方法」，而是「架构框架 + SSL 加成」。这种把架构与目标解耦评估的实验设计（TriForces-Streams vs Base+SSL vs TriForces）值得作为多任务方法的标准消融范式。

## 局限与展望
- 三流拼接后参数量上去了（如 TriForces Orb-v3 42M vs Orb-v3 25.5M），虽然附录 B.7 控参数对比仍占优，但部署成本仍是问题，未对推理速度做系统对比。
- SOAP 风格幂谱在大体系上的算力开销没充分展开；类型无关结构流是否会在含 H 的有机分子上丢失关键元素差异，作者未明示。
- LeJEPA + SIGReg 的正则权重 $\lambda$ 是关键超参，没看到对不同体系/任务的敏感性分析。
- 检索结果只给了定性可视化，缺少 nearest-neighbor mAP 这类定量指标。

## 相关工作与启发
- **vs JMP / DFT 预训练 MLIP**：JMP 等用 DFT 标签做监督预训练，TriForces 在 MatBench 上虽然没全部超越 JMP-L，但用自监督就把差距缩到很小，关键是不需要昂贵的 DFT 标签。
- **vs Roost / CrabNet（组成模型）**：它们丢掉几何只看化学计量比，TriForces 把组成流嵌进几何 GNN，既保留组成信号又不丢几何分辨能力。
- **vs Noisy Nodes / 力场去噪**：之前 SSL 在 MLIP 里多作为辅助损失，目标比较单一；TriForces 系统对比了非重建 + 去噪 + 掩码三类目标的互补性，给出了「哪个 SSL 在什么场景下最有用」的实证结论。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Augmenting Representations with Scientific Papers](../../ICLR2026/physics/augmenting_representations_with_scientific_papers.md)
- [\[ICML 2026\] Quiver: Quantum-Informed Views for Enhanced Representations in Large ML Models](quiver_quantum-informed_views_for_enhanced_representations_in_large_ml_models.md)
- [\[ICML 2026\] Distribution Transformers: Fast Approximate Bayesian Inference With On-The-Fly Prior Adaptation](distribution_transformers_fast_approximate_bayesian_inference_with_on-the-fly_pr.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[ICML 2026\] A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots](a_call_to_lagrangian_action_learning_population_mechanics_from_temporal_snapshot.md)

</div>

<!-- RELATED:END -->
