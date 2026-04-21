---
title: >-
  [论文解读] DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values
description: >-
  [医学图像] DeltaSHAP 是一种专为在线患者监护系统设计的可解释AI算法，通过将 Shapley 值适配到时序场景，解释连续预测之间的**变化量**而非绝对预测值，同时提供特征归因的**方向和幅度**，在 MIMIC-III 基准上实现 62% 的解释质量提升和 33% 的计算时间缩减。
tags:
  - 医学图像
---

# DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values

- **会议**: ICML 2025
- **arXiv**: [2507.02342](https://arxiv.org/abs/2507.02342)
- **代码**: [https://github.com/AITRICS/DeltaSHAP](https://github.com/AITRICS/DeltaSHAP)
- **领域**: 医学/可解释AI (Medical XAI / Online Patient Monitoring)
- **关键词**: Shapley值, 可解释AI, 在线患者监护, 时间序列归因, 临床决策支持

## 一句话总结

DeltaSHAP 是一种专为在线患者监护系统设计的可解释AI算法，通过将 Shapley 值适配到时序场景，解释连续预测之间的**变化量**而非绝对预测值，同时提供特征归因的**方向和幅度**，在 MIMIC-III 基准上实现 62% 的解释质量提升和 33% 的计算时间缩减。

---

## 研究背景与动机

### 临床场景需求

在线患者监护系统（如ICU中的早期预警系统）需要持续监测患者的生命体征和实验室检查值，并实时预测患者恶化的风险。然而，临床医生在使用这些系统时，面临的核心问题不是"当前预测值是多少"，而是"**预测值为什么发生了变化**"。例如：
- 脓毒症概率从70%降到40%意味着患者正在好转
- 同样的40%如果是从10%升上来的，则意味着严重恶化
- 相同的绝对预测值在不同上下文中含义截然不同

因此，临床XAI方法需要满足三个关键需求：

**解释连续预测差异**：而非孤立时间点的绝对预测

**提供方向性归因**：每个特征对预测的贡献既有大小也有正负方向

**实时计算**：在临床时间敏感场景中快速交付解释

### 现有方法的局限

**通用XAI方法**（如LIME、SHAP、IG、DeepLIFT等）：
- 专注于估计绝对预测的归因，无法解释预测变化
- 尝试计算所有特征和所有时间步的逐点归因，计算负担沉重
- 忽略时间序列数据中的时序依赖关系

**时间序列专用XAI方法**（如FIT、WinIT）：
- FIT 基于KL散度框架量化特征归因，但**只提供归因幅度、不提供方向**
- WinIT 扩展了延迟影响的建模，但同样缺乏方向性信息
- 两者都依赖条件生成模型来生成反事实样本，**计算成本高昂**，不适合时间敏感的临床场景

---

## 方法详解

### 整体框架

DeltaSHAP 的核心思想是：将解释目标从"绝对预测值"转变为"连续时间步之间的预测差值 $\Delta$"，并通过 Shapley 值采样来高效计算每个新观测特征对 $\Delta$ 的方向性贡献。

整体流程分为三个核心模块：

1. **预测差值定义**（Section 4.1）：定义解释目标为 $\Delta$
2. **Shapley 值采样**（Section 4.2）：通过排列采样近似 Shapley 值
3. **基线选择策略**（Section 4.3）：使用前向填充代替生成模型处理缺失值

### 关键设计

#### 1. 预测差值建模

给定在线监护模型 $f: \mathbb{R}^{L \times D} \rightarrow [0,1]$，其中 $L$ 为最大序列长度，$D$ 为临床特征数量。在时间步 $T$，模型使用滑动窗口 $\mathbf{X}_{T-W+1:T} \in \mathbb{R}^{W \times D}$ 进行预测。

解释目标为两个连续时间步之间的预测差：

$$\Delta = f(\mathbf{X}_{T-W+1:T}) - f(\mathbf{X}_{T-W+1:T} \setminus \mathbf{X}_T)$$

其中 $f(\mathbf{X}_{T-W+1:T} \setminus \mathbf{X}_T)$ 表示只使用历史数据、不包含当前时间步测量值的预测。

需要计算归因向量 $\phi(f, \mathbf{X}_{T-W+1:T}) \in \mathbb{R}^D$，其中 $\phi_j$ 量化特征 $j$ 在时间步 $T$ 对预测演化的贡献，满足**效率性质**（efficiency property）：

$$\sum_{j=1}^{D} \phi_j(f, \mathbf{X}_{T-W+1:T}) = \Delta$$

正值表示该特征推高预测，负值表示该特征拉低预测。

#### 2. Shapley 值采样近似

精确计算 Shapley 值需要评估 $2^D$ 个特征组合，对高维临床数据计算成本过高。DeltaSHAP 采用排列采样方法（Shapley Value Sampling）：

采样 $N$ 个随机排列 $\Omega$，对每个观测特征 $j \in \mathcal{F}_{\text{obs}}$ 近似：

$$\hat{\phi}_j(f, \mathbf{X}_{T-W+1:T}) = \frac{1}{N} \sum_{\pi \in \Omega} [v(S_{\pi,j} \cup \{j\}) - v(S_{\pi,j})]$$

其中 $S_{\pi,j}$ 是在排列 $\pi$ 中排在特征 $j$ 之前的特征集合，$v(S)$ 衡量只有子集 $S$ 中特征被观测时对 $\Delta$ 的部分贡献：

$$v(S) = f(\mathbf{X}_{T-W+1:T-1} \cup \mathbf{X}_T^S) - f(\mathbf{X}_{T-W+1:T} \setminus \mathbf{X}_T)$$

#### 3. 归因归一化

为消除采样误差并保证效率性质，对归因进行归一化：

$$\phi_j(f, \mathbf{X}_{T-W+1:T}) = \hat{\phi}_j(f, \mathbf{X}_{T-W+1:T}) \cdot \frac{\Delta}{\sum_{k \in \mathcal{F}_{\text{obs}}} \hat{\phi}_k(f, \mathbf{X}_{T-W+1:T})}$$

这确保所有归因之和精确等于观测到的预测差值，同时不改变特征重要性排名。

#### 4. 基线选择：前向填充

不同于 FIT/WinIT 依赖条件生成模型来处理未观测特征，DeltaSHAP 利用预处理中已有的缺失值处理机制。对于 LSTM 模型，直接使用**前向填充**（forward-filling）：用最近一次观测值填充缺失特征。

这一策略的优势：
- 与模型预处理管道一致，避免分布外问题
- 无需训练额外生成模型，大幅减少计算开销
- 天然处理不规则采样的临床数据

### 算法伪代码

DeltaSHAP 的完整流程（Algorithm 1）：
1. 计算预测差 $\Delta$
2. 生成 $N$ 个特征观测集 $\mathcal{F}_{\text{obs}}$ 的随机排列
3. 对每个排列中的每个特征，计算边际贡献（可并行化）
4. 将边际贡献累加并除以 $N$ 得到近似 Shapley 值
5. 归一化以满足效率性质

关键实现细节：特征遍历和排列遍历在实现中都进行了**并行化**，进一步提升计算效率。

---

## 评估指标创新

论文提出了一套新的评估指标体系，用于衡量在线时间序列预测中归因的**忠实度**（faithfulness）：

### 基础指标
- **CPD**（Cumulative Prediction Difference）：逐步移除最重要特征后的预测变化累积
- **CPP**（Cumulative Prediction Preservation）：逐步移除最不重要特征后的预测变化累积

### 主要评估指标
- **AUPD**（Area Under Prediction Difference）：CPD 的面积指标，$\uparrow$ 越高越好
- **AUPP**（Area Under Prediction Preservation）：CPP 的面积指标，$\downarrow$ 越低越好
- **AUAUCD / AUAUCP**：基于 AUC 的数据集级性能变化指标
- **AUAPRD / AUAPRP**：基于 APR 的数据集级性能变化指标

这些指标的优势：(1) 强调高排名特征的影响力；(2) 通过聚合多层级移除效果降低局部异常敏感性。

---

## 实验关键数据

### 数据集

| 数据集 | 任务 | ICU入院数 | 预测实例数 | 正样本比例 | 预测窗口 |
|--------|------|-----------|------------|------------|----------|
| MIMIC-III | 失代偿预测 | ~41,000 | ~250万 | 2.5% | 24小时 |
| PhysioNet 2019 | 脓毒症预测 | ~40,000 | ~110万 | 2.5% | 12小时 |

### 主实验结果（Table 1）

**MIMIC-III 失代偿预测**（LSTM backbone）：

| 方法 | AUPD ↑ | AUPP ↓ | Wall-Clock Time (s) |
|------|--------|--------|---------------------|
| LIME | 8.20 | 21.58 | 0.22 |
| GradSHAP | 6.20 | 19.68 | 0.03 |
| IG | 13.46 | 14.51 | 0.04 |
| DeepLIFT | **13.95** | 14.35 | 0.03 |
| FO | 13.55 | **14.14** | 1.43 |
| AFO | 13.08 | 15.14 | 39.62 |
| FIT | 12.60 | 16.16 | 0.12 |
| WinIT | 10.06 | 16.56 | 0.30 |
| **DeltaSHAP** | **22.59** | **3.04** | **0.02** |

DeltaSHAP 在 AUPD 上超越次优方法 62%（22.59 vs 13.95），AUPP 降低 78.5%（3.04 vs 14.14）。

**PhysioNet 2019 脓毒症预测**：

| 方法 | AUPD ↑ | AUPP ↓ | Time (s) |
|------|--------|--------|----------|
| AFO | 3.27 | 1.03 | 14.18 |
| FIT | 2.15 | 3.08 | 0.11 |
| **DeltaSHAP** | **3.68** | **0.89** | **0.02** |

### 消融实验（Table 2, MIMIC-III）

| 配置 | AUPD ↑ | AUPP ↓ | Time (s) |
|------|--------|--------|----------|
| w/o 基线选择（零填充） | 8.49 | 18.89 | 0.05 |
| w/o 归一化 | 22.58 | 3.05 | 0.05 |
| N=1 | 22.14 | 3.19 | 0.02 |
| N=10 | 22.56 | 3.07 | 0.04 |
| N=100 | 22.61 | 3.04 | 0.09 |
| **DeltaSHAP (N=25)** | **22.58** | **3.05** | **0.05** |

### 关键发现

1. **基线选择至关重要**：去掉前向填充、改用零填充，AUPD 从 22.58 暴降到 8.49，性能退化超过 60%，是最关键的设计选择
2. **归一化影响可解释性而非排名**：去掉归一化几乎不影响定量指标，但归一化保证了效率性质，使归因之和精确等于预测差值，增强了临床可解释性
3. **N=25 是最优平衡点**：N=1 即可取得不错性能，N=25 在准确性和效率之间达到最优权衡，N=100 仅有微小提升但时间翻倍
4. **DeltaSHAP 是唯一满足 AUPD > AUPP 的方法**：表明移除重要特征的预测变化大于移除不重要特征的变化，证明归因与模型行为高度对齐

---

## 定性案例分析

论文通过 MIMIC-III 失代偿预测任务的多个临床案例验证了 DeltaSHAP 的临床一致性：

1. **血氧饱和度（SpO2）**：DeltaSHAP 正确识别血氧饱和度从 92% 骤降到 60% 为高风险信号（低于 70% 表示急性危险），以及从 60% 恢复到 98% 为好转信号
2. **高血糖**：DeltaSHAP 将超过 300mg/dL 的血糖值正确归因为高重要性，符合高血糖与心脏失代偿的临床关联
3. **血压骤降**：收缩压（SBP）和舒张压（DBP）的急剧下降被正确归因为主要失代偿风险因素，与心输出量受损的临床知识一致

---

## 亮点与洞察

1. **问题建模精准**：将解释目标从"绝对预测"转变为"预测差值"，完美匹配临床思维中"关注变化而非绝对值"的实际需求
2. **简洁而有效的工程选择**：用前向填充代替生成模型处理未观测特征，在保持解释质量的同时将计算效率提升数量级（AFO 需 39.62s vs DeltaSHAP 仅 0.02s）
3. **完整的评估体系**：提出了从样本级（AUPD/AUPP）到数据集级（AUAUCD/AUAPRP）的完整忠实度评估指标体系，填补了在线时间序列XAI评估的空白
4. **理论保证+实用性兼顾**：Shapley 值的效率性质（归因之和等于预测差）为临床解释提供了理论基础，归一化步骤确保这一性质在近似计算中也成立
5. **模型无关性**：DeltaSHAP 不需要访问模型梯度或内部操作，可适配任意黑盒模型架构

---

## 局限性

1. **仅关注最后一个时间步**：通过只解释最近观测到的特征来降低计算量，但可能遗漏早期时间步观测的**延迟效应**
2. **固定时间点分析的适用性未充分验证**：对非在线、非监护场景（如一次性预测）的表现尚不明确
3. **复杂门控机制的遮蔽效应**：LSTM 等模型的内部门控机制可能遮蔽 DeltaSHAP 试图捕获的特征交互
4. **仅在 LSTM 上验证**：虽然方法是模型无关的，但实验仅使用 LSTM 作为 backbone，未在 Transformer 等更现代架构上验证
5. **正样本比例极低**（2.5%），评估结果中的 APR 相关指标可能受类别不平衡影响

---

## 相关工作

- **通用XAI方法**：LIME, SHAP (KernelSHAP, GradSHAP, DeepSHAP), IG, DeepLIFT, FO, AFO — 主要针对静态预测（图像分类、表格数据），未考虑时序依赖
- **时间序列XAI方法**：TimeSHAP (Bento et al., 2021), FIT (Tonekaboni et al., 2020), WinIT (Leung et al., 2021), Dynamic Masks (Crabbé & Van Der Schaar, 2021), TimeX++ (Liu et al., 2024a) — FIT 和 WinIT 是最直接的竞争方法，但缺乏方向性归因且依赖生成模型
- **Shapley 值采样**：Mitchell et al. (2022), Strumbelj & Kononenko (2010) — DeltaSHAP 的采样近似技术基础

---

## 评分

⭐⭐⭐⭐ (4/5)

**优点**：问题定义精准、方法设计简洁高效、效果显著（62%提升+33%加速）、临床一致性强、评估指标体系完善。

**不足**：仅在LSTM上验证、未处理延迟效应、实验数据集仅两个。整体是一项扎实的工作，在临床XAI领域有重要的实践价值。

<!-- RELATED:START -->

## 相关论文

- [COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [Protein Structure Tokenization: Benchmarking and New Recipe](protein_structure_tokenization_benchmarking_and_new_recipe.md)
- [Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](../../CVPR2025/medical_imaging/multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)
- [Scalable Diffusion Transformer for Conditional 4D fMRI Synthesis](../../NeurIPS2025/medical_imaging/scalable_diffusion_transformer_for_conditional_4d_fmri_synthesis.md)

<!-- RELATED:END -->
