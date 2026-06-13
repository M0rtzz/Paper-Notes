---
title: >-
  [论文解读] Online Feedback Efficient Active Target Discovery in Partially Observable Environments
description: >-
  [NeurIPS 2025][医学图像][主动目标发现] 提出 DiffATD，利用扩散模型的逆向过程构建 belief 分布来平衡探索与利用，在部分可观测环境中无需任何监督训练即可高效发现目标区域，适用于医学影像、物种发现和遥感等多领域。 在许多科学和工程领域（如 MRI 扫描、搜索救援、药物发现），数据采集成本极高…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "主动目标发现"
  - "扩散模型"
  - "部分可观测环境"
  - "探索-利用平衡"
  - "贝叶斯实验设计"
---

# Online Feedback Efficient Active Target Discovery in Partially Observable Environments

**会议**: NeurIPS 2025  
**arXiv**: [2505.06535](https://arxiv.org/abs/2505.06535)  
**代码**: [GitHub](https://github.com/AnindyaSarkar/DiffATD)  
**领域**: 医学图像  
**关键词**: 主动目标发现, 扩散模型, 部分可观测环境, 探索-利用平衡, 贝叶斯实验设计

## 一句话总结

提出 DiffATD，利用扩散模型的逆向过程构建 belief 分布来平衡探索与利用，在部分可观测环境中无需任何监督训练即可高效发现目标区域，适用于医学影像、物种发现和遥感等多领域。

## 研究背景与动机

在许多科学和工程领域（如 MRI 扫描、搜索救援、药物发现），数据采集成本极高，需要在有限预算下从未观测区域中策略性采样以最大化目标发现。核心挑战在于：

**部分可观测性**：智能体在搜索过程中只能观测到搜索空间的一小部分，必须基于有限观测推断未知区域的内容

**探索-利用困境**：智能体需同时完成两个矛盾目标——探索（获取搜索空间信息以降低不确定性）和利用（聚焦于可能包含目标的区域）

**标注数据稀缺**：现有方法多依赖大规模预标注数据集训练 RL 策略，但在稀有疾病（如罕见肿瘤）等场景中获取此类数据极不现实

已有方法的局限性：基于强化学习的方法（如 GOMAA-Geo、Visual Active Search）依赖完全可观测性和大量预标注数据；贝叶斯决策论方法虽无需训练但也假设完全可观测；生成式方法虽透明但仅优化重建而非目标发现。DiffATD 的核心贡献在于首次实现了**无需监督训练**的部分可观测环境目标发现，并具备可解释性。

## 方法详解

### 整体框架

DiffATD 将主动目标发现（ATD）建模为在网格化搜索空间中的序贯决策问题。搜索区域被划分为 $N$ 个网格单元，智能体在预算 $\mathcal{B}$ 内逐步选择测量位置，每次测量揭示该网格的目标占比 $y^{(i)} \in [0,1]$。DiffATD 通过扩散模型的逆向过程维护粒子集合，构建对未观测空间的 belief 分布，并通过融合探索得分和利用得分来决定下一步采样位置。

### 关键设计

1. **基于扩散动力学的 Belief 分布构建**：DiffATD 在逆向扩散过程中维护一批粒子 $\{x_\tau^{(i)}\}_{i=0}^{N_B}$，利用 Tweedie 公式进行单步去噪估计：$\hat{x}_0 = \mathcal{T}_\tau(x_\tau) = \frac{1}{\sqrt{\bar{\alpha}_\tau}}(x_\tau + (1-\bar{\alpha}_\tau)s_\theta(x_\tau, \tau))$。这些粒子隐式形成对完整搜索空间的 belief 分布，建模为 $N_B$ 个各向同性高斯混合：$p(\hat{x}_t|Q_t, \tilde{x}_{t-1}) = \sum_{i=0}^{N_B} \alpha_i \mathcal{N}(\hat{x}_t^i, \sigma_x^2 I)$。设计动机在于利用预训练扩散模型的生成先验来推断未观测区域，无需额外监督信号。

2. **最大熵探索策略**：探索得分通过测量位置上粒子间的分歧度来计算。作者证明了（定理 1）最优探索位置等价于粒子预测差异最大的区域：$\mathrm{expl}^{\mathrm{score}}(q_t) = \sum_{i,j} \frac{([\hat{x}_t^{(i)}]_{q_t} - [\hat{x}_t^{(j)}]_{q_t})^2}{2\sigma_x^2}$。这避免了对每个候选位置分别计算 belief 分布的高昂开销。设计动机源于贝叶斯实验设计中的互信息最大化原则。

3. **基于奖励模型的利用策略**：利用得分结合了两个信号——(1) 预期对数似然（quantify 粒子间的一致性）和 (2) 在线训练的奖励模型 $r_\phi$ 对目标存在概率的预测：$\mathrm{exploit}^{\mathrm{score}}(q_t) = \mathrm{likeli}^{\mathrm{score}}(q_t) \times \sum_{i}^{N_B} r_\phi([\hat{x}_t^{(i)}]_{q_t})$。奖励模型随机初始化，通过二元交叉熵损失在每次测量后增量更新。设计动机在于将"区域预测的一致性"与"目标存在的可能性"解耦，使两种信号互补。

### 损失函数 / 训练策略

- **测量引导**：在逆扩散过程中通过梯度下降约束已观测位置的重建：$x_{\tau-1}^{(i)} \leftarrow x_{\tau-1}^{(i)'} - \zeta \nabla_{x_\tau^{(i)}} \|[x]_{Q_t} - [\hat{x}_\tau^{(i)}]_{Q_t}\|^2$
- **预算感知平衡**：最终采样得分为探索与利用的加权组合：$\mathrm{Score}(q_t) = \kappa(\mathcal{B}) \cdot \mathrm{expl}^{\mathrm{score}} + (1-\kappa(\mathcal{B})) \cdot \mathrm{exploit}^{\mathrm{score}}$，其中 $\kappa(\mathcal{B}) = \frac{\mathcal{B}-t}{\mathcal{B}+t}$ 使搜索初期偏重探索、后期偏重利用
- **奖励模型训练**：使用 BCE 损失在不断增长的测量数据集 $\mathcal{D}_t$ 上增量训练

## 实验关键数据

### 主实验

实验覆盖遥感（DOTA）、物种发现（iNaturalist）、皮肤病发现、胸部 X 光骨抑制等多个领域。

| 数据集/目标 | 预算 $\mathcal{B}$ | DiffATD SR | 最佳基线 SR | 相对提升 |
|---|---|---|---|---|
| DOTA (plane/truck) | 200 | 0.5422 | 0.4625 (Max-Ent) | +17.23% |
| DOTA (plane/truck) | 300 | 0.7309 | 0.6550 (GA) | +11.59% |
| iNaturalist (物种) | 200 | 0.6401 | 0.5826 (GA) | +9.87% |
| 皮肤病 (恶性) | 200 | 0.8974 | 0.8261 (GA) | +8.63% |
| 胸部 X 光 (骨抑制) | 300 | 0.4142 | 0.2936 (RS) | +41.08% |

### 消融实验

| 配置 ($\alpha$ 控制探索/利用权重) | DOTA SR ($\mathcal{B}=200$) | Skin SR ($\mathcal{B}=200$) | 说明 |
|---|---|---|---|
| $\alpha=0.2$ (偏利用) | 0.5052 | 0.8465 | 过度利用降低性能 |
| $\alpha=1.0$ (均衡) | **0.5422** | **0.8974** | 最优平衡点 |
| $\alpha=5.0$ (偏探索) | 0.4823 | 0.8782 | 过度探索也不利 |

### 关键发现

- DiffATD 在皮肤病等罕见目标上的 SR 达到 0.8974，甚至超过完全可观测的监督方法 SAM (FullSEG, 0.6221)
- 对比 GPT-4o 和 Gemini 等 VLM，DiffATD 在 DOTA 上 $\mathcal{B}=300$ 时 SR 为 0.7309，显著优于 Gemini 的 0.6453 和 GPT-4o 的 0.5678
- 探索-利用平衡的 $\kappa(\mathcal{B})$ 线性调度在所有领域均表现优异，验证了简单调度策略的有效性

## 亮点与洞察

- **无监督范式**：完全依赖无监督预训练的扩散模型，无需任务特定标注数据，大幅提升了实际可用性
- **理论支撑**：基于贝叶斯实验设计和互信息最大化推导出的探索策略有严谨的理论基础（定理 1、2）
- **可解释性**：相比黑盒 RL 策略，DiffATD 的每一步决策都可通过探索得分图和利用得分图解释
- **跨领域泛化**：同一框架在遥感、生态、医学影像等迥异领域均取得显著提升

## 局限与展望

- 依赖预训练扩散模型的质量，在训练数据量不足或分布偏移时性能可能下降
- 单步 Tweedie 估计的精度在高噪声水平下有限，可考虑多步估计以提升 belief 分布的准确性
- 奖励模型在早期数据极少时容易过拟合，虽然 $\kappa(\mathcal{B})$ 缓解了这一问题但未完全解决，可探索元学习等快速适应策略
- 目前仅在 2D 网格空间中验证，扩展到 3D 体数据（如 MRI 体素）需要额外考虑计算效率
- 粒子数 $N_B$ 和测量调度 $M$ 的选择依赖经验，缺乏理论指导
- 搜索空间被假设为均匀网格划分，实际应用中目标可能跨越多个网格或仅占据网格一角

## 相关工作与启发

- 与主动 MRI 加速（van Gorp et al., 2021）的联系：DiffATD 可视为将主动采样从"优化重建"扩展到"优化目标发现"
- 与贝叶斯优化的联系：探索-利用框架类似于 GP-UCB，但 DiffATD 使用扩散模型替代高斯过程实现更强的表达能力
- 与 GOMAA-Geo（Sarkar et al., 2024）的关系：后者需要大量预标注数据训练 RL 策略，DiffATD 完全无监督
- 启发了将扩散模型用于其他序贯决策问题的可能性，如主动传感、自适应实验设计等

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将扩散动力学与主动目标发现结合，探索-利用框架设计精巧
- **实验充分度**: ⭐⭐⭐⭐ 覆盖四个领域多种目标类型，消融完整，但缺少与更多主动学习方法的对比
- **写作质量**: ⭐⭐⭐⭐ 理论推导清晰，但符号较多初读有一定门槛
- **价值**: ⭐⭐⭐⭐⭐ 无监督主动发现在医学影像中有重要应用前景，特别是罕见疾病筛查场景
- **总体**: ⭐⭐⭐⭐ 扎实的工作，理论和实验兼备，跨领域泛化令人印象深刻

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Active Target Discovery under Uninformative Prior: The Power of Permanent and Transient Memory](active_target_discovery_under_uninformative_prior_the_power_of_permanent_and_tra.md)
- [\[NeurIPS 2025\] Doctor Approved: Generating Medically Accurate Skin Disease Images through AI-Expert Feedback](doctor_approved_generating_medically_accurate_skin_disease_images_through_ai-exp.md)
- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [\[NeurIPS 2025\] Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)
- [\[NeurIPS 2025\] Dynamic Causal Discovery in Alzheimer's Disease through Latent Pseudotime Modelling](dynamic_causal_discovery_in_alzheimers_disease_through_latent_pseudotime_modelli.md)

</div>

<!-- RELATED:END -->
