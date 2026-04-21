---
title: >-
  [论文解读] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations
description: >-
  [ICCV 2025][模型压缩][PEFT] 提出 LieRA，利用李群理论将矩阵级 PEFT 方法（如 LoRA）推广到高维参数空间（如卷积核），通过在李代数中表示扰动并用指数映射回李群，在保持参数空间结构性质的同时实现高效微调。
tags:
  - ICCV 2025
  - 模型压缩
  - PEFT
  - LoRA
  - Lie Group
  - 高维参数空间
  - 卷积核微调
---

# Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations

**会议**: ICCV 2025  
**arXiv**: [2504.00851](https://arxiv.org/abs/2504.00851)  
**代码**: [https://github.com/Chongjie-Si/Subspace-Tuning](https://github.com/Chongjie-Si/Subspace-Tuning)  
**领域**: 参数高效微调 / 模型压缩  
**关键词**: PEFT, LoRA, Lie Group, 高维参数空间, 卷积核微调

## 一句话总结

提出 LieRA，利用李群理论将矩阵级 PEFT 方法（如 LoRA）推广到高维参数空间（如卷积核），通过在李代数中表示扰动并用指数映射回李群，在保持参数空间结构性质的同时实现高效微调。

## 研究背景与动机

现有 PEFT 方法（LoRA 及其变体）主要针对二维矩阵（线性层）设计，在处理高维参数空间（如四维卷积核）时面临结构破坏问题。直接将 LoRA 的低秩更新 reshape 为卷积核形状会破坏卷积核的空间局部性——矩阵中相邻元素在 reshape 后可能对应卷积核中距离很远的位置。

然而，许多视觉基础模型（如 ConvNeXt、Stable Diffusion）大量依赖卷积操作，需要一种通用方法在不破坏高维参数结构的前提下进行微调。与其为每种高维参数设计专门策略，不如探索将现有矩阵级 PEFT 方法泛化到高维空间的统一框架。

## 方法详解

### 整体框架

将参数视为李群的元素，更新量建模为李代数中的扰动，通过指数映射将扰动映回李群，确保更新平滑且保持参数空间结构。结合一阶 Taylor 近似简化计算，使整个框架在实践中高效可用。

### 关键设计

1. **李群构造 (Lie Group Construction)**:

    - 将卷积核参数集合 $G = \{\mathcal{W} \in \mathbb{R}^{C_{in} \times C_{out} \times k \times k} | W_{c,i,j,l} \neq 0\}$ 视为李群
    - 群运算定义为逐元素乘法（Hadamard 积）$\odot$
    - 单位元为全 1 张量 $\mathcal{I}$，逆元为逐元素取倒数
    - $G \cong \prod_{c,i,j,l} (\mathbb{R} \setminus \{0\})$，是一维李群的笛卡尔积，天然具有光滑流形结构
    - 对应李代数 $\mathfrak{g}$ 同构于 $\mathbb{R}^{C_{out} \times C_{in} \times k \times k}$，是线性向量空间

2. **乘法式参数更新 (Multiplicative Update)**:

    - 传统加法更新：$\mathcal{W} \rightarrow \mathcal{W} + \Delta\mathcal{W}$（不保结构）
    - LieRA 乘法更新：$\mathcal{W} \rightarrow \mathcal{W} \odot \exp(\Delta\mathcal{W})$
    - 乘法更新按比例缩放每个元素，保持核内相对结构和空间局部性
    - 由于 $G$ 在群运算下封闭，更新后参数仍在 $G$ 中，保持流形结构

3. **一阶 Taylor 近似 (First-Order Taylor Approximation)**:

    - 由于 $\Delta\mathcal{W}$ 很小，$\exp(\Delta\mathcal{W}) \approx \mathcal{I} + \Delta\mathcal{W}$
    - 更新规则简化为：$\mathcal{W} \odot \exp(\Delta\mathcal{W}) \approx \mathcal{W} + \mathcal{W} \odot \Delta\mathcal{W}$
    - 该近似大幅降低计算开销，同时性能损失可忽略

### 理论分析：秩容量

- LoRA 的秩容量：$\mathcal{R}(\mathbf{AB}) = r$，受限于低秩 $r$
- LieRA 的秩容量：$\mathcal{R}(\mathbf{W} \odot \mathbf{AB}) = \min(n, m)$（全秩），因为预训练权重通常近似满秩，Hadamard 积保持高秩
- 全秩容量使 LieRA 有更强的表达能力和任务适应灵活性

### 损失函数 / 训练策略

- 与 LoRA 完全一致的训练策略，仅将加法更新替换为乘法更新
- 适用于所有矩阵级 PEFT 方法（LoRA、DoRA、PISSA 等），作为通用框架

## 实验关键数据

### 主实验

| 方法 | #Param | VTAB-1k Avg | COCO Det mAP | COCO Seg mAP |
|------|--------|-------------|--------------|--------------|
| Full FT | 102.05M | 78.2 | 49.0 | 43.4 |
| LoRA r=8 | 7.30M | 74.2 | - | - |
| **LieRA r=8** | **7.30M** | **75.5** | - | - |
| LoRA r=16 | 14.48M | 74.1 | 35.5 | 33.6 |
| **LieRA r=16** | **14.48M** | **75.5** | **39.1** | **37.0** |
| LoRA r=32 | 34.54M | - | 35.9 | 34.4 |
| **LieRA r=32** | **34.54M** | - | **40.5** | **38.2** |

NLP 任务（LLaMA3-8B Commonsense Reasoning）：

| 方法 | Params(%) | BoolQ | PIQA | HellaS. | ARC-c | Avg. |
|------|-----------|-------|------|---------|-------|------|
| LoRA r=16 | 0.35% | 72.3 | 86.7 | 93.5 | 75.7 | 82.8 |
| **LieRA r=16** | **0.35%** | **74.7** | **87.9** | **95.6** | **79.9** | **85.1** |
| LoRA r=32 | 0.70% | 70.8 | 85.2 | 91.7 | 71.2 | 80.8 |
| **LieRA r=32** | **0.70%** | **74.3** | **88.7** | **95.4** | **80.3** | **85.3** |

### 消融实验

Taylor 近似影响（ConvNeXt-V2-B, r=16）：

| 方法 | VTAB Avg | COCO Avg | Training Time | GPU |
|------|----------|----------|---------------|-----|
| LieRA w/ TA | 75.5 | 42.3 | 50.17 min | 9.97 GB |
| LieRA w/o TA | 75.7 | 42.7 | 76.28 min | 14.74 GB |

与其他 PEFT 方法耦合：

| 方法 | VTAB Avg | COCO Avg | 总 Avg |
|------|----------|----------|--------|
| PISSA r=16 | 74.7 | 38.2 | 56.5 |
| PISSA+LieRA | 75.7 | 42.4 | 59.1 |
| DoRA r=16 | 74.7 | 38.4 | 56.6 |
| DoRA+LieRA | 75.5 | 42.5 | 59.0 |

### 关键发现

- LieRA 在 CV 任务（卷积层微调）上优势显著，COCO 检测上 LieRA 比 LoRA 高 3-5 个 mAP
- 在 NLP 任务（线性层微调）上同样有效，LLaMA3-8B 平均提升 2.3-4.5%
- 一阶 Taylor 近似几乎无性能损失但大幅节省资源（COCO 训练时间减少 34%，显存减少 32%）
- LieRA 作为通用框架可增强 DoRA、PISSA 等其他 PEFT 方法

## 亮点与洞察

- **优雅的数学框架**：用李群/李代数理论统一处理不同维度的参数空间，理论优美且工程上极为简洁（仅需将加法改为逐元素乘法）
- **通用性强**：不是一种新的 PEFT 方法，而是一个可以增强任意矩阵级 PEFT 方法的框架
- **秩容量的理论优势**：通过 Hadamard 积实现全秩容量，理论上比纯低秩更新更具表达力
- **实现简单**：核心改动仅一行代码，从 $W + \Delta W$ 变为 $W + W \odot \Delta W$

## 局限与展望

- 目前仅验证到四维张量（卷积核），尚未扩展到更高维参数
- 对于权重中包含零值的情况（如稀疏网络），李群构造需要额外处理
- Taylor 近似在 $\Delta W$ 较大时可能不够精确
- 乘法更新在线性层上的优势不如卷积层明显

## 相关工作与启发

- 可以尝试将此框架应用于其他具有高维参数的新型架构（如 Mamba 的选择机制参数）
- 李群框架可能启发基于其他数学结构（如纤维丛）的 PEFT 方法
- 全秩容量的思路可以结合到其他低秩方法的设计中

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 李群视角独特，将微调问题提升到微分几何层面
- **实验充分度**: ⭐⭐⭐⭐ CV/NLP 多任务多模型验证充分，消融全面
- **写作质量**: ⭐⭐⭐⭐ 数学推导清晰，但符号较多需要一定背景
- **实用价值**: ⭐⭐⭐⭐⭐ 即插即用，一行代码改动即可使用

<!-- RELATED:START -->

## 相关论文

- [Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)
- [State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](../../ACL2025/model_compression/l4q_parameter_efficient_quantization_aware_finetuning.md)
- [Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models](../../ACL2025/model_compression/trans_peft_transferable.md)

<!-- RELATED:END -->
