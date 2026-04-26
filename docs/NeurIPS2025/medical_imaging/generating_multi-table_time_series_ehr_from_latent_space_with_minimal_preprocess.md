---
title: >-
  [论文解读] Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing
description: >-
  [NeurIPS 2025][医学图像][EHR 合成] 提出 RawMed——首个以最小有损预处理合成多表时序 EHR 原始数据的框架：将事件文本化 → Residual Quantization 压缩至离散潜空间 → 自回归 Transformer 建模时序动态，在保真度、临床效用和隐私保护上全面超越现有基线。
tags:
  - NeurIPS 2025
  - 医学图像
  - EHR 合成
  - 时序数据生成
  - 量化
  - 隐私保护
  - 多表关系数据库
  - Transformer
---

# Generating Multi-Table Time Series EHR from Latent Space with Minimal Preprocessing

**会议**: NeurIPS 2025  
**arXiv**: [2507.06996](https://arxiv.org/abs/2507.06996)  
**代码**: [eunbyeol-cho/RawMed](https://github.com/eunbyeol-cho/RawMed)  
**作者**: Eunbyeol Cho, Jiyoun Kim, Minjae Lee, Sungjin Park, Edward Choi (KAIST, FuriosaAI)  
**领域**: medical_imaging  
**关键词**: EHR 合成, 时序数据生成, Residual Quantization, 隐私保护, 多表关系数据库, 自回归 Transformer  

---

## 一句话总结

提出 RawMed——首个以最小有损预处理合成多表时序 EHR 原始数据的框架：将事件文本化 → Residual Quantization 压缩至离散潜空间 → 自回归 Transformer 建模时序动态，在保真度、临床效用和隐私保护上全面超越现有基线。

---

## 研究背景与动机

### 问题背景

电子健康记录（EHR）本质上是**多表关系型时序数据库**，记录患者入院后的一系列临床事件（实验室检查、处方、输入事件等），包含分类型、数值型和文本型数据。由于涉及敏感隐私信息，EHR 数据受到 HIPAA 等法规严格限制，难以在研究机构间共享，合成 EHR 数据因此成为刚需。

### 现有方法的两大瓶颈

**高度依赖特征选择**：EMR-M-GAN、EHR-Safe、FLEXGEN-EHR、TIMEDIFF 等方法均需领域专家预先选定少量表和列（最多约 5,373 个特征），若后续研究需要被排除的特征则无法使用合成数据。这意味着合成数据的"可用范围"在生成前就被锁定了。

**复杂的有损预处理**：数值 binning、术语归一化、时间聚合等操作不可避免地扭曲原始分布。例如，对实验室指标做时间聚合可能掩盖突发异常值，binning 则过度简化细微趋势，降低下游预测模型的可靠性。

### 规模差距

| 方法 | 最大特征数 | 最大时间步 | 使用全部列 | 保留原始值 |
|------|-----------|-----------|-----------|-----------|
| EMR-M-GAN | 98 | 24 | ✗ | ✗ |
| EHR-Safe | 90 | 50 | ✗ | ✗ |
| FLEXGEN-EHR | 5,373 | 48 | ✗ | ✗ |
| TIMEDIFF | 15 | 276 | ✗ | ✗ |
| **RawMed** | **333,524** | **243** | **✓** | **✓** |

RawMed 在特征规模上比最接近的 FLEXGEN-EHR 高出约 **62 倍**，且首次同时保留所有列与原始数值。

---

## 方法详解

RawMed 的核心管线分三步：**文本化** → **事件级压缩** → **时序建模与采样**。

### 1. 数据表示：文本化序列

将 EHR 的每行事件序列化为纯文本字符串。例如，实验室事件被表示为：

> `"lab item Glucose value 95 uom mg/dL"`

这种表示天然支持异构数据类型，无需 binning 或归一化。每个患者 $p$ 的临床轨迹表示为有序事件序列 $S^p = [e_1^p, e_2^p, \ldots, e_{n^p}^p]$，其中 $e_i^p = (t_i^p, x_i^p)$，$t_i^p$ 为入院后的时间戳，$x_i^p$ 为序列化文本。文本经分词、填充/截断至固定长度 $L=128$ 后嵌入为 $\mathbf{x}_i^p \in \mathbb{R}^{L \times F}$。

### 2. 事件级压缩：RQ-VAE

文本化后序列长度急剧膨胀，直接建模的计算开销不可接受。RawMed 用 **Residual Quantization（RQ）** 解决这一瓶颈：

- **编码器**（1D CNN）将 $\mathbf{x}_i^p \in \mathbb{R}^{L \times F}$ 压缩为 $\hat{\mathbf{z}}_i^p \in \mathbb{R}^{L_z \times F_z}$
- **RQ 量化**将每个潜向量分解为 $D$ 层残差量化码：$\text{RQ}(\hat{\mathbf{z}}; C, D) = (k_1, \ldots, k_D) \in [K]^D$，其中每层对前一层的残差做最近邻码本查找
- **解码器**从 $\mathbf{z}_i^p = \sum_{d=1}^D \text{lut}(k_d)$ 重建 $\hat{\mathbf{x}}_i^p$

与标准 VQ-VAE 相比，RQ 用多层残差逐步逼近，对独立分布列（如 patientweight）的保真度显著更高——KS 统计量从 VQ 的 0.28 降至 RQ 的 0.09。

**压缩效果**：MIMIC-IV 上序列长度从 11.2k 压缩至 1.8k（**84% 压缩率**），eICU 上从 3.1k 到 0.8k（**74% 压缩率**）。

### 3. 时序建模：TempoTransformer

压缩后的患者轨迹被编排为交错序列：

$$S_{\text{quantized}}^p = [\tau_1^p, k_1^p, \tau_2^p, k_2^p, \ldots, \tau_{n^p}^p, k_{n^p}^p]$$

两个关键设计：

- **时间 Tokenization**：将时间戳按 10 分钟分辨率分解为十进制数字序列（如 720 分钟 → $[7, 2]$），使用 0–9 的小词表
- **时间分离（Time Separation）**：将时间 token 与事件内容 token 显式交错排列并用不同词表约束，确保模型不会混淆两类信息

自回归 Transformer 以标准 NLL 损失训练：

$$\mathcal{L}_{\text{AR}} = -\sum_{p \in \mathcal{P}} \sum_{i=1}^{|S_{\text{quantized}}^p|} \log P(s_i^p \mid s_1^p, \ldots, s_{i-1}^p)$$

### 4. 采样与后处理

- 使用 Top-$k$ 采样生成新患者序列，通过词表掩码保证结构完整性
- **事件级验证**：用 Levenshtein 距离修正拼写错误的列名，去除数值字段的多余字符
- **患者级验证**：丢弃含无效事件的序列，检查时间戳单调性，最后转换为关系表

### 5. 评估框架（新提出）

RawMed 同时提出了首个面向多表时序合成 EHR 的综合评估体系，涵盖：

- **单表保真度**：CDE（列分布）、I-CDE（按 item 细粒度分布）、PCC/I-PCC（列间相关性）、Predictive Similarity（高阶依赖）
- **多表时序保真度**：Time Gap（事件间隔分布的 KS 距离）、Event Count（每患者事件数分布）、Next Event Prediction（LSTM 预测下一事件的 F1）
- **临床效用**：11 个临床预测任务的 AUROC
- **隐私保护**：Membership Inference Attack 准确率

---

## 实验关键数据

### 表 1：单表评估结果（MIMIC-IV & eICU，越小越好）

| 模型 | CDE (MIMIC) | I-CDE (MIMIC) | PCC (MIMIC) | I-PCC (MIMIC) | ER (MIMIC) | SMAPE (MIMIC) |
|------|------------|--------------|------------|--------------|-----------|--------------|
| Real | - | - | - | - | 15.35 | 60.85 |
| SDV | 0.11 | 0.54 | 0.26 | 0.26 | 49.32 | 103.85 |
| RC-TGAN | 0.26 | 0.54 | 0.18 | 0.28 | 38.21 | 97.26 |
| ClavaDDPM | 0.06 | 0.22 | 0.08 | 0.27 | 27.91 | 80.02 |
| **RawMed** | **0.04** | **0.05** | **0.04** | **0.10** | **19.69** | **57.31** |

RawMed 的 I-CDE 仅为 ClavaDDPM 的约 1/4，表明其在 item 级别粒度上保真度远超基线。SMAPE 接近真实数据（57.31 vs 60.85），说明高阶语义一致性极强。

### 表 2：临床效用与时序保真度（越接近 Real 越好 / 越小越好）

| 模型 | AUROC MEDS-TAB (MIMIC) | AUROC GenHPF (MIMIC) | Time Gap (MIMIC) | Event Count (MIMIC) | MIA (MIMIC) |
|------|----------------------|---------------------|-----------------|-------------------|------------|
| Real | 0.90±0.06 | 0.82±0.09 | - | - | - |
| SDV | 0.46±0.13 | 0.47±0.13 | 0.76 | 0.46 | 0.499 |
| ClavaDDPM | 0.68±0.19 | 0.64±0.17 | 0.48 | 0.11 | 0.500 |
| **RawMed** | **0.87±0.08** | **0.80±0.09** | **0.01** | **0.02** | **0.498** |

RawMed 在 11 个临床预测任务上的 AUROC 仅比真实数据低 0.02–0.03，远超第二名 ClavaDDPM（差距 0.16–0.19）。Time Gap 指标为 0.01，比 ClavaDDPM 的 0.48 低了 **48 倍**，时序保真度优势巨大。MIA 准确率接近 0.5（随机猜测），隐私保护充分。

---

## 亮点与洞察

1. **"全量保留"范式**：RawMed 首次证明可以在保留 EHR 数据库所有列和原始值的条件下进行高质量合成，打破了"必须做特征选择"的惯性思维。333k 特征规模比最先进方法高出两个数量级。

2. **RQ 优于 VQ 的关键洞察**：对于像 patientweight 这样与其他列弱相关的数值列，VQ 的单层码本无法充分编码，导致重建分布严重失真。RQ 的多层残差机制恰好弥补这一缺陷，这一发现对所有采用 VQ 的医疗数据生成任务都有参考价值。

3. **时间分离是最关键组件**：消融实验显示，去掉 Time Separation 后 Time Gap 从 0.01 暴涨至 0.51、Event Count 从 0.02 涨至 0.40，性能退化最为严重。这说明显式分离时间和内容信息是时序建模成功的核心。

4. **评估框架的价值**：提出的 I-CDE/I-PCC 指标巧妙地解决了 EHR 中"同一列存储不同临床项目数据"的评估难题，填补了合成 EHR 评估的空白。

5. **压缩与质量可以兼得**：84% 的序列长度压缩率不仅没有降低质量，反而因为降低了自回归建模的难度而提升了生成保真度（RawMed 全面优于无压缩的 RealTabFormer）。

---

## 局限性

1. **表数量有限**：当前仅验证了 3 个主要时序表（实验室、处方、输入事件），能否扩展到数十个表尚未验证，表间依赖建模的复杂度可能显著增加。

2. **无条件生成**：目前仅支持无条件合成，不能按指定条件（如特定性别、年龄段、疾病类型）生成患者数据，限制了其在临床试验模拟等场景的实用性。

3. **静态属性未整合**：性别、出生年份等静态特征未纳入生成管线，合成数据缺乏人口统计学的一致性。

4. **后处理依赖**：生成数据仍存在列名拼写错误、数值字段异常字符等问题，需要基于规则的后处理修正，说明潜空间生成的结构保真仍有改进空间。

5. **长时间窗口的可扩展性**：虽然 24 小时窗口实验表明大部分指标稳定，但 Event Count 指标有所退化，更长的住院记录可能需要更高级的压缩或分层采样策略。

---

## 相关工作

- **EHR 合成方法演进**：从单数据类型 GAN/VAE（生成诊断码）→ 混合类型时序（EVA, HALO）→ 联合建模时序+异构的近期工作（EMR-M-GAN, EHR-Safe, FLEXGEN-EHR, TIMEDIFF），但均依赖特征选择和重预处理
- **文本化表格生成**：GReaT、REaLTabFormer 等将表格行转为文本用 LM 生成，但局限于单表设置，且未处理时序维度
- **向量量化**：VQ-VAE 及 Residual Quantization 在图像/语音领域已有广泛应用，RawMed 首次将 RQ 引入 EHR 事件压缩
- **合成数据评估**：SDMetrics、Synthcity 侧重单表，对多表时序场景覆盖不足

---

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-----------|------|
| 新颖性 | 8 | 首个多表时序原始 EHR 合成框架，规模提升两个数量级 |
| 技术深度 | 7 | RQ-VAE + 自回归 Transformer 的组合扎实但非颠覆性创新 |
| 实验充分度 | 9 | 两个公开数据集、11 个下游任务、全面消融、多维度评估 |
| 实用价值 | 8 | 直接解决医疗 AI 数据稀缺痛点，代码开源 |
| 写作质量 | 8 | 结构清晰，问题定义精确，评估框架的贡献独立成章 |
| **总分** | **8.0** | 问题定义准确、解法完整、实验有说服力的优秀工作 |

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)

<!-- RELATED:END -->
