---
title: >-
  [论文解读] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals
description: >-
  [NeurIPS 2025][医学图像][EEG] 提出 BrainOmni——首个统一 EEG 和 MEG 的脑信号基础模型，通过 BrainTokenizer（含物理传感器编码器）将异构脑电/脑磁信号离散化为统一 token，再用 Criss-Cross Transformer 进行自监督掩码预测预训练…
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "EEG"
  - "MEG"
  - "foundation model"
  - "BrainTokenizer"
  - "Sensor Encoder"
  - "跨设备泛化"
  - "RVQ"
---

# BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals

**会议**: NeurIPS 2025  
**arXiv**: [2505.18185](https://arxiv.org/abs/2505.18185)  
**代码**: [OpenTSLab/BrainOmni](https://github.com/OpenTSLab/BrainOmni)  
**领域**: 自监督学习 / 脑信号基础模型  
**关键词**: EEG, MEG, foundation model, BrainTokenizer, Sensor Encoder, 跨设备泛化, RVQ

## 一句话总结

提出 BrainOmni——首个统一 EEG 和 MEG 的脑信号基础模型，通过 BrainTokenizer（含物理传感器编码器）将异构脑电/脑磁信号离散化为统一 token，再用 Criss-Cross Transformer 进行自监督掩码预测预训练，在阿尔茨海默病检测上提升 11.7 个百分点，并实现对完全未见设备的零样本重建泛化。

## 研究背景与动机

**领域现状**：EEG（脑电图）和 MEG（脑磁图）共享生物物理基础——均测量大脑皮层树突电流产生的电磁场，但信号特征差异显著。现有方法几乎全部采用模态和数据集特定模型，且不同设备间存在严重异质性（通道数 19-306 不等，传感器类型包括 EEG 电极、梯度计 GRAD、磁力计 MAG，方向和布局各异）。

**现有痛点**：(a) EEG 和 MEG 分别建模，无法共享跨模态知识——尤其浪费了两者在生物物理层面的共同起源；(b) 模型与特定电极拓扑绑定，换设备就需重新训练或适配；(c) EEG 基础模型（如 LaBraM、CBraMod）已初见成效，但 MEG 的基础模型几乎空白，主要受限于数据量和设备复杂度。

**核心矛盾**：如何构建一个统一模型同时处理 EEG 和 MEG，既不依赖固定电极命名/拓扑，又能从跨模态数据中获益？

**本文目标**：构建设备不可知（device-agnostic）的脑信号基础模型，统一两种模态预训练，支持跨设备零样本泛化。

**切入角度**：借鉴传统 EEG/MEG 源活动估计（source estimation）的思想——用传感器的物理属性（3D 笛卡尔坐标、朝向、类型）而非电极名称来编码空间信息，从观测信号反推"隐含源变量"，实现与具体设备的解耦。

**核心 idea**：编码传感器物理属性实现设备不可知 → RVQ 离散化构建统一 token 表示 → Criss-Cross Transformer 分别建模空间和时间依赖。

## 方法详解

### 整体框架

两阶段训练流程：

- **Stage 1**：训练 BrainTokenizer——将异构 EEG/MEG 信号统一编码为离散 token 空间（自编码器框架，25% 通道随机丢弃 + 全通道重建）
- **Stage 2**：在 BrainTokenizer 输出的 token 上，用自监督掩码预测训练 BrainOmni 模型，学习脑活动的语义表征

### 关键设计

#### 1. BrainTokenizer + Sensor Encoder

- **功能**：将不同设备、不同通道数的 EEG/MEG 信号统一编码为固定维度的离散 token
- **核心思路**：
    - SEANet 编码器（1D 卷积 + 残差块 + 步长卷积）提取时间表示 $\mathbf{Z}_{\text{time}} \in \mathbb{R}^{C \times W \times D}$
    - Sensor Encoder 对每个传感器的物理信息编码：位置嵌入层编码 6 维笛卡尔坐标（3D 位置 + 3D 朝向），类型嵌入层编码传感器类型（EEG=0, GRAD=1, MAG=2），融合得到传感器嵌入 $\mathbf{V} \in \mathbb{R}^{C \times D}$
    - 交叉注意力通道压缩：用可学习 Query 将可变通道数压缩到固定 $C'=16$ 个隐含源变量，Key = 时间表示 + 传感器嵌入，Value = 时间表示
    - 4 层 RVQ（残差向量量化）将连续特征离散化为 $\mathbf{Q} \in \mathbb{R}^{C' \times W \times 4}$
- **设计动机**：Sensor Encoder 使模型与电极命名约定无关，只依赖物理坐标——这是跨设备泛化的关键；交叉注意力压缩对应传统源活动估计中的"逆问题"

#### 2. Criss-Cross Transformer

- **功能**：在 Stage 2 中对离散 token 序列建模空间和时间依赖
- **核心思路**：将特征沿维度方向一分为二，一半做空间注意力（跨 $C'$ 个源变量），一半做时间注意力（跨时间步 $T$），拼接后送入 FFN。时间注意力使用 RoPE 位置编码
- **设计动机**：脑信号的空间依赖（不同脑区间协同）和时间依赖（时间序列动态）具有不同结构特征，分别建模比全注意力更高效且更具针对性

#### 3. 联合 EEG-MEG 预训练策略

- **功能**：在统一 token 空间上同时用 EEG（1997 小时）和 MEG（656 小时）数据自监督预训练
- **核心思路**：50% 掩码比例的非自回归掩码 token 预测——每个被掩码位置的 4 层 RVQ 同时预测。80% 掩码位置用专用 mask token，20% 用随机采样 token 替代，防止对 mask token 过拟合
- **设计动机**：EEG 数据量充足可帮助数据稀缺的 MEG 模态，跨模态知识共享

### 损失函数 / 训练策略

**Stage 1 BrainTokenizer 损失**（四项联合优化）：

- 时间域 L1 损失：$\mathcal{L}_{\text{time}} = \|\mathbf{X} - \hat{\mathbf{X}}\|$
- 频域损失：$\mathcal{L}_{\text{freq}} = \|\mathbf{A} - \hat{\mathbf{A}}\| + \|\mathbf{\Phi} - \hat{\mathbf{\Phi}}\|$（幅度谱 + 相位谱）
- PCC 对齐损失：$\mathcal{L}_{\text{pcc}} = e^{-\text{PCC}(\mathbf{X}, \hat{\mathbf{X}})}$（正则化波形趋势一致性）
- RVQ 承诺损失：$\mathcal{L}_{\text{rvq}} = \sum_{i=1}^{N_q}\|\mathbf{z}_i - \mathbf{z}_{q_i}\|^2$

**Stage 2 BrainOmni 损失**：掩码 token 的交叉熵预测 $\mathcal{L}_{\text{model}} = \frac{1}{M}\sum_{i,j}\mathcal{L}_{\text{ce}}(q_{ij}, y_{ij})$

**训练配置**：预训练 BrainTokenizer 约 11h + BrainOmni-base 约 18h（16×A100），下游 30 epochs、5-fold 交叉验证、跨被试划分。

## 实验关键数据

### 主实验：下游任务性能（Balanced Accuracy）

| 任务 | 模态 | 数据集 | BrainOmni_base | 最佳 Baseline | 提升 |
|------|------|--------|---------------|--------------|------|
| 阿尔茨海默病检测 | EEG | AD65 | **82.8%** | LaBraM 71.1% | +11.7% |
| 帕金森病检测 | EEG | PD31 | **74.8%** | LaBraM 65.9% | +8.9% |
| 抑郁症检测 | EEG | MDD | **88.6%** | LaBraM 88.0% | +0.6% |
| 异常检测 | EEG | TUAB | **81.9%** | LaBraM 81.6% | +0.3% |
| 事件分类（6类） | EEG | TUEV | **62.2%** | LaBraM 58.8% | +3.4% |
| 情绪识别（3类） | EEG | FACED | **49.0%** | LaBraM 45.8% | +3.2% |
| 运动想象（2类） | EEG | WBCIC_SHU | **83.2%** | LaBraM 83.1% | +0.1% |
| 运动想象（4类） | EEG | PhysioNet-MI | 59.0% | CBraMod **59.5%** | -0.5% |
| 孤独症检测 | MEG | ASD74 | **65.1%** | ST-Trans 58.3% | +6.8% |
| 抑郁症检测 | MEG | MEG-MMI | **60.4%** | FAMED 52.7% | +7.7% |
| 运动响应 | EMEG | SomatoMotor | **83.2%** | ST-Trans 66.1% | +17.1% |

### 跨设备零样本泛化 & 联合预训练消融

| 实验 | 关键对比 | 结果 |
|------|---------|------|
| 未见 EEG 设备重建（PerceiveImagine） | PCC: 已见 0.748 vs 未见 **0.802** | 未见设备甚至更好 |
| 未见 MEG 设备重建（Gloups-MEG） | PCC: 已见 0.711 vs 未见 0.695 | 仅轻微下降 |
| ASD74：MEG-only vs EMEG 联合 | 55.3% → **62.1%** | +12% 相对提升 |
| TUAB：EEG-only vs EMEG 联合 | 81.1% → **81.9%** | EEG 也受益 |
| SomatoMotor：EEG/MEG/EMEG 输入 | 78.3% / 83.8% / **86.3%** | 联合输入最优 |

### Sensor Encoder 消融

| 配置 | TUAB | AD65 | ASD74 | SomatoMotor |
|------|------|------|-------|-------------|
| 完整 BrainOmni | **81.9%** | **79.5%** | **62.1%** | **86.3%** |
| w/o Sensor Encoder | 78.4% | 73.0% | 56.8% | 74.4% |
| Pure temporal（多通道视为无关单通道） | 78.8% | 76.3% | 57.1% | 76.6% |

去掉 Sensor Encoder 后 SomatoMotor 下降 11.9 个百分点，证明物理传感器编码对多设备泛化至关重要。

## 亮点与洞察

- **设备不可知的物理编码**：用 3D 坐标 + 朝向 + 类型替代电极名称，突破了传统方法对固定拓扑的依赖，这一思路可推广到其他多传感器信号（如 fNIRS）
- **EEG 数据"教导"MEG**：联合预训练使 MEG 的 ASD74 任务提升 12%，体现了跨模态迁移的价值——数据量大的模态可有效帮助数据稀缺模态
- **多层次重建损失的互补性**：仅用时间损失会导致解码器输出坍缩为近常数信号；频域损失改善频谱但引入高频脉冲；PCC 损失抑制脉冲但频谱细节弱——四项损失缺一不可
- **隐含源变量数量 16 是最优平衡点**：超过 16 后重建损失下降趋缓，但计算成本线性增长
- **RVQ 第一层码本捕获最丰富语义**：掩码预测准确率随层数递减，说明信息呈层次化分布

## 局限性

- MEG 预训练数据仅 656 小时（EEG 的 1/3），可能限制了 MEG 侧的性能上限
- 下游 MEG 评估仅 2 个数据集，评估覆盖度有限
- 在 PhysioNet-MI 上略输 CBraMod（59.0% vs 59.5%），说明在特定任务上通用基础模型仍可能不如专用模型
- 未探索生成式应用（如脑信号合成、跨模态翻译）

## 相关工作与启发

- **vs LaBraM / CBraMod**：仅处理 EEG 且依赖固定通道拓扑，BrainOmni 统一了 EEG 和 MEG 并实现设备不可知
- **vs FAMED**：MEG 专用模型，无预训练，BrainOmni 在 MEG 任务上大幅超越
- **启发**：Sensor Encoder 的物理属性编码思路可扩展到其他异构传感器场景（可穿戴设备、工业传感器阵列等）；RVQ 离散化 + 掩码预测的两阶段范式可借鉴到其他时间序列基础模型

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个统一 EEG/MEG 基础模型，Sensor Encoder 和 BrainTokenizer 设计有独创性
- **实验充分度**: ⭐⭐⭐⭐ 11 个下游任务 + 跨设备泛化 + 联合预训练消融 + 损失消融 + 可视化分析
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，方法与传统源估计的类比增强可理解性
- **实用价值**: ⭐⭐⭐⭐ 代码开源，对脑科学 AI 临床应用（阿尔茨海默/帕金森/孤独症检测）有直接推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](../../ICLR2026/medical_imaging/uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)
- [\[NeurIPS 2025\] Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens](brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)
- [\[NeurIPS 2025\] NeurIPT: Foundation Model for Neural Interfaces](neuript_foundation_model_for_neural_interfaces.md)
- [\[CVPR 2025\] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](../../CVPR2025/medical_imaging/vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)
- [\[NeurIPS 2025\] Convolutional Monge Mapping between EEG Datasets to Support Independent Component Labeling](convolutional_monge_mapping_between_eeg_datasets_to_support_independent_componen.md)

</div>

<!-- RELATED:END -->
