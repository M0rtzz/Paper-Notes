---
title: >-
  [论文解读] Federated Data-Efficient Instruction Tuning for Large Language Models
description: >-
  [ACL 2025 (Findings)][LLM对齐][联邦学习] 提出 FedHDS（Federated Hierarchical Data Selection），通过 intra-client 和 inter-client 两级层次化数据选择消除联邦学习中客户端内部和跨客户端的数据冗余，结合多层 Transformer 特征融合提升 coreset 质量；仅用不到 1.5% 的数据，在 Rouge-L 上相对 SOTA 全数据联邦基线平均提升 10.72%，训练效率提升最高达 48.8 倍。
tags:
  - ACL 2025 (Findings)
  - LLM对齐
  - 联邦学习
  - 数据高效
  - 指令微调
  - 层间特征融合
  - 层次化数据选择
  - Coreset
  - HDBSCAN
---

# Federated Data-Efficient Instruction Tuning for Large Language Models

**会议**: ACL 2025 (Findings)  
**arXiv**: [2410.10926](https://arxiv.org/abs/2410.10926)  
**代码**: [GitHub](https://github.com/zhenqincn/FedHDS)  
**领域**: 联邦学习 / LLM指令微调  
**关键词**: 联邦学习, 数据高效, 指令微调, 层间特征融合, 层次化数据选择, Coreset, HDBSCAN

## 一句话总结

提出 FedHDS（Federated Hierarchical Data Selection），通过 intra-client 和 inter-client 两级层次化数据选择消除联邦学习中客户端内部和跨客户端的数据冗余，结合多层 Transformer 特征融合提升 coreset 质量；仅用不到 1.5% 的数据，在 Rouge-L 上相对 SOTA 全数据联邦基线平均提升 10.72%，训练效率提升最高达 48.8 倍。

## 研究背景与动机

**领域现状**：指令微调（Instruction Tuning）是将预训练 LLM 对齐到人类指令的关键步骤。联邦学习（FL）通过利用分散在边端设备上的私有指令数据来提高数据多样性，已成为 LLM 微调的热门范式。

**现有痛点**：
   - **(效率问题)** 现有联邦微调方法 (FedIT, FlexLoRA 等) 使用客户端上的全量数据进行本地训练，导致计算开销巨大。边端设备 GPU 容量有限，常需 CPU+GPU 混合计算，遍历全量数据的时间成本难以承受。
   - **(过拟合问题)** FL 中每个客户端通常只覆盖有限领域，在冗余数据上反复训练会导致过拟合于本地数据，损害对未见任务的泛化能力。
   - **(集中式方法不兼容 FL)** 已有的数据高效方法（如 coreset选择）需要同时访问所有数据，直接违反 FL "数据不出端" 的隐私原则，且无法检测跨客户端的数据冗余。
   - **(特征表示次优)** 现有 coreset 方法仅依赖 Transformer 最后一层的特征来区分数据样本，而实验表明没有哪一层在所有指标上都最优。

**核心矛盾**：如何在联邦学习框架下既保护隐私又高效选择代表性数据子集？如何同时消除客户端内部和跨客户端的冗余？

**切入角度**：提出两级层次化选择——先在客户端本地用 HDBSCAN 聚类识别 intra-client 冗余，再将近似聚类中心（不对应真实样本）发送到服务器进行 inter-client 聚类，过滤跨客户端冗余。同时融合所有 Transformer 层的特征以获得更优数据表示。

**核心 idea**：联邦场景下的两级层次化 coreset 选择 + 全层特征融合 = 用 <1.5% 数据超越全数据联邦微调。

## 方法详解

### 整体框架

FedHDS 在每轮 FL 中，于常规本地训练之前增加 "I. Data Selection" 模块。流程如下：

1. 客户端下载最新全局模型 $\mathbf{w}^r$
2. **Intra-client 选择**（步骤 ①-④）：提取特征 → 融合 → 本地聚类 → 发送聚类中心到服务器
3. **Inter-client 选择**（步骤 ⑤-⑦）：服务器聚类 → 筛选 → 通知客户端哪些组被选中
4. 客户端仅在 coreset $\tilde{\mathcal{D}}_i$ 上进行本地训练
5. 服务器 FedAvg 聚合

### 关键设计

#### 1. 多层特征融合（Full-Layer Feature Fusion）

对每个数据样本 $\mathbf{x}$，提取所有 $l$ 层 Transformer 的最后一个 token 的隐状态，拼接为：

$$\mathbf{h}_j = [\mathbf{h}_j^{1,-1}, \mathbf{h}_j^{2,-1}, \ldots, \mathbf{h}_j^{l,-1}]$$

然后通过 t-SNE (Barnes-Hut 实现) 降维到 $k=2$ 维：

$$\{\tilde{\mathbf{h}}_1, \tilde{\mathbf{h}}_2, \ldots\} = P(\{\mathbf{h}_1, \mathbf{h}_2, \ldots\})$$

**设计动机**：实验表明不同层提供不同抽象级别的特征，单用最后一层在聚类质量指标（CH Index、F1-Score、Silhouette Coefficient）上并非最优。t-SNE 在非线性空间中比 PCA 更有效，可以自动抑制低方差维度的影响。

#### 2. Intra-Client 数据选择

在融合后的 2D 特征空间上应用 HDBSCAN（基于密度的层次聚类），将本地数据划分为若干组 $\{\mathcal{G}_1, \mathcal{G}_2, \ldots\}$。每组对应一个近似聚类中心 $\mathbf{c}_j$（不对应具体样本）。

**选择 HDBSCAN 而非 K-means** 的原因：HDBSCAN 能自动确定聚类数目，无需预设 $k$；且能适应不同密度分布的数据。

#### 3. Inter-Client 数据选择

服务器收集所有活跃客户端发送的聚类中心，再次用 HDBSCAN 进行全局聚类：

$$\mathbb{G}^{II} = \text{HDBSCAN}(\{\mathbf{c}_1, \mathbf{c}_2, \ldots\})$$

在每个全局组 $\mathcal{G}_j^{II}$ 中，选择离其中心最近的一个一级组作为代表。然后通知对应客户端，每个被选中的组中选取离组中心最近的真实数据样本加入 coreset。

**隐私保证**：传输的仅是 2D 聚类中心（几十字节），不对应真实样本。可进一步通过 tanh 缩放 + 高斯噪声实现 $(\varepsilon, \delta)$-差分隐私。

#### 4. FedHDS-Turbo 加速版

用轻量级代理模型（GPT-2，~124M 参数）替代 LLM 进行特征提取，大幅降低推理时间，精度损失很小。

### 训练细节

- PEFT：采用 LoRA (rank=8, alpha=16, dropout=0.05)
- 优化器：Adam，学习率 $3 \times 10^{-4}$（NI）或 $3 \times 10^{-5}$（Dolly-15K）
- 跨设备 FL 设置：每轮 5% 客户端活跃
- 通信开销：除 LoRA 参数外，额外仅传输几十字节的 2D 中心和聚类索引

## 实验关键数据

### 主实验（Table 2: Rouge-L %）

| 方法 | NI (1.3B) | NI (3B) | Dolly (α=0.5, 1.3B) | Dolly (α=0.5, 3B) | Dolly (α=5.0, 1.3B) | Dolly (α=5.0, 3B) |
|------|-----------|---------|----------------------|--------------------|----------------------|--------------------|
| FedAvg (全参数, 参考) | 22.08 | 24.40 | 30.18 | 31.90 | 30.18 | 31.90 |
| FedIT (LoRA全数据) | 21.29 | 28.00 | 30.55 | 32.49 | 30.57 | 32.02 |
| FlexLoRA | 21.67 | 27.35 | 30.31 | 32.89 | 31.60 | 33.24 |
| Random (最优比例) | 24.86 | 28.80 | 31.41 | 33.44 | 31.09 | 32.86 |
| Coreset-Cent (集中式) | 31.36 | 34.81 | 33.27 | 35.48 | 33.27 | 35.48 |
| **FedHDS** | **25.45** | **28.77** | **31.82** | **33.66** | **32.12** | **33.81** |
| **FedHDS-Turbo** | 24.77 | 28.42 | 31.47 | 33.80 | 31.67 | 33.57 |

- FedHDS-Turbo 相对 FedIT 平均提升 Rouge-L **10.72%**
- FedHDS/FedHDS-Turbo 使用数据量 **<1.5%**

### 效率对比（Table 3: 加速比）

| 方法 | NI (1.3B) 加速 | NI (3B) 加速 | Dolly (1.3B) 加速 | Dolly (3B) 加速 |
|------|----------------|--------------|-------------------|-----------------|
| FedIT (全数据) | 1× | 1× | 1× | 1× |
| Random | 37.2× | 23.0× | 12.89× | 4.54× |
| **FedHDS** | 17.8× | 19.9× | 16.06× | 7.90× |
| **FedHDS-Turbo** | **48.8×** | **40.4×** | **18.86×** | **6.66×** |

### 通信与内存开销（Table 5）

| 方法 | 模型通信 | 额外通信 | GPU 内存 |
|------|----------|----------|----------|
| FedIT | 12 MB | 0 | 10.56 GB |
| FedHDS | 12 MB | 44 Bytes | 9.40 GB |
| FedHDS-Turbo | 12 MB | 76 Bytes | 9.32 GB |

额外通信开销几乎可忽略不计（仅为几十字节的 2D 聚类中心）。

### 关键消融实验

- **两级选择 vs 单级**：去掉 inter-client 选择（FedHDS‡）在所有场景下劣于完整 FedHDS，验证了层次化选择的必要性
- **全局选择 vs 层次化**：把所有特征直接发服务器做全局聚类（GlobalSelect）效果更差，可能因大规模数据上的聚类结果次优
- **特征融合方法**：t-SNE > PCA ≈ Kernel PCA，在复杂数据集 NI 上差异更明显
- **差分隐私**：噪声方差 ≤0.1 时 FedHDS 仍优于 Random 基线
- **收敛与过拟合**：FedHDS 的 test loss 持续下降，而 FedIT 的 test loss 早期就停止下降（training loss 仍在降），验证了全数据训练导致过拟合

## 亮点与洞察

- **首个联邦数据高效指令微调工作**：填补了 FL 场景下 coreset 选择的空白，对表 1 的 qualitative comparison 清楚展示了与现有方法的差异
- **两级层次化选择设计精巧**：intra-client 用密度聚类自动确定组数，inter-client 仅传输 2D 中心（几十字节），既高效又保护隐私
- **多层特征融合的实证支持充分**：通过 CH Index、F1-Score、Silhouette Coefficient 等多指标验证"没有哪一层是万能的"，以此引出融合必要性
- **"用更少数据反而更好"的反直觉发现**：不是因为数据少所以差，而是因为消除冗余和过拟合后泛化能力更强——这对 FL 场景很有启发
- **FedHDS-Turbo 的代理模型思路**：用 GPT-2 (124M) 代替大模型做特征提取，近乎无损精度但大幅提速，工程实用性强

## 局限与展望

1. **只关注数据代表性，忽略数据质量**：低质量样本可能被当作独立领域保留，引入 quality-based 过滤可进一步改善
2. **仅在 1.3B / 3B 模型上实验**：未验证在更大规模（如 7B、13B）模型上是否保持优势
3. **特征提取依赖完整前向传播**：即使用 FedHDS-Turbo，仍需在每轮对全量数据做推理，如果数据量很大依然有开销
4. **HDBSCAN 对高维稀疏数据可能不稳定**：虽然 t-SNE 降到 2D 缓解了问题，但 t-SNE 本身非确定性可能影响可复现性
5. **测试仅限 NLP 任务**：未在代码生成、多模态等其他指令微调场景验证
6. **集中式 Coreset-Cent 在复杂数据集上仍优于 FedHDS**：说明隐私约束确实带来了精度损失，联邦场景下的 coreset 选择仍有提升空间

## 相关工作与启发

- **与 FedIT (Zhang et al., 2024) 的关系**：FedIT 是基于 FedAvg + LoRA 的联邦指令微调基线，FedHDS 在其基础上增加数据选择模块
- **与 LIMA (Zhou et al., 2023) 的联系**："Less is More for Alignment" 表明少量高质量数据足以对齐 LLM，FedHDS 在联邦场景验证了类似结论
- **与 Chen et al. (2023) 的对比**：后者用 K-means + 最后一层特征做集中式 coreset 选择，无法处理 FL 场景的跨客户端冗余
- **启发**：(1) 可以将数据质量评分与代表性选择结合，在 FL 中实现 quality × representativeness 联合优化；(2) 轻量级代理模型做特征提取的思路可推广到其他 FL 场景（如联邦 RAG）

## 评分

- 新颖性: ⭐⭐⭐⭐ (首个联邦数据高效指令微调，层次化选择 + 全层融合)
- 理论深度: ⭐⭐⭐ (有收敛保证和 DP 证明，但较直接)
- 实验充分性: ⭐⭐⭐⭐⭐ (2 数据集 × 2 模型 × 多种 non-IID，消融完整，效率和隐私都有分析)
- 实用价值: ⭐⭐⭐⭐ (端侧 FL 场景直接可用，FedHDS-Turbo 工程友好)
- 总体推荐: ⭐⭐⭐⭐ (联邦指令微调方向的扎实工作，实验充分)

<!-- RELATED:START -->

## 相关论文

- [Instruction Tuning of Large Language Models for Tabular Data Generation—in One Day](../../ICML2025/llm_alignment/instruction_tuning_of_large_language_models_for_tabular_data_generation-in_one_d.md)
- [Call for Rigor in Reporting Quality of Instruction Tuning Data](call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)
- [Importance-Aware Data Selection for Efficient LLM Instruction Tuning](../../AAAI2026/llm_alignment/importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)
- [Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)
- [T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning](../../NeurIPS2025/llm_alignment/t-shirt_token-selective_hierarchical_data_selection_for_instruction_tuning.md)

<!-- RELATED:END -->
