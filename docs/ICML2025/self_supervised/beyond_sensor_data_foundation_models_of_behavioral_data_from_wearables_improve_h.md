---
title: >-
  [论文解读] Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions
description: >-
  [ICML 2025][自监督学习][wearable foundation model] 在 Apple Heart and Movement Study 的 162K 参与者、25 亿小时可穿戴行为数据上，系统探索 tokenizer 和架构组合，以 TST+Mamba-2+对比学习构建行为数据基础模型 WBM，在 57 项健康检测任务上显著优于手工特征基线，并与 PPG 传感器模型形成互补。
tags:
  - ICML 2025
  - 自监督学习
  - wearable foundation model
  - behavioral data
  - health detection
  - Mamba-2
  - 对比学习
---

# Beyond Sensor Data: Foundation Models of Behavioral Data from Wearables Improve Health Predictions

**会议**: ICML 2025  
**arXiv**: [2507.00191](https://arxiv.org/abs/2507.00191)  
**代码**: 无（Apple 内部研究，受知情同意限制）  
**领域**: 自监督学习  
**关键词**: wearable foundation model, behavioral data, health detection, Mamba-2, contrastive learning

## 一句话总结

在 Apple Heart and Movement Study 的 162K 参与者、25 亿小时可穿戴行为数据上，系统探索 tokenizer 和架构组合，以 TST+Mamba-2+对比学习构建行为数据基础模型 WBM，在 57 项健康检测任务上显著优于手工特征基线，并与 PPG 传感器模型形成互补。

## 研究背景与动机

**领域现状**：可穿戴设备基础模型近年快速发展，但几乎全部聚焦底层传感器信号——PPG、ECG、加速度计等秒级原始数据。这些信号虽有价值，但并非全天持续可用，在健康状态检测覆盖范围上存在局限。

**现有痛点**：更高层的行为数据（步数、运动时间、心率变异性、步态指标、VO2Max 等）天然对应人类行为的时间尺度（天/周），是经过专家验证的生理相关量，对健康检测本应更有信息量。然而行为数据面临三大挑战：(1) 不规则采样——不同变量的采样频率不同；(2) 大量缺失——某些指标（如 VO2Max）仅在特定运动后记录；(3) 变量间异质性——27 个变量跨越活动量、心血管、生命体征、步态、体成分、心肺适能六大类。

**核心矛盾**：大规模无标签行为数据（15.14M 周）vs 有限标注的下游任务——天然适合基础模型范式，但行为数据的不规则性和异质性使得现有时序基础模型方法无法直接适用。

**本文目标** (1) 如何为不规则采样的多变量行为数据设计最优 tokenizer？(2) 什么架构最适合编码行为时序？(3) 行为数据基础模型能否在广泛健康检测任务上超越手工特征和传感器模型？

**切入角度**：不假设已有最优方案，而是对所有候选 tokenizer（TST/mTAN/Tuple）× 架构（Self-Attention Transformer/Rotary Transformer/Mamba-2）的 3×3=9 种组合做系统消融，用年龄预测作为代理任务筛选最优模型。

**核心 idea**：行为数据是可穿戴健康 AI 的被忽视金矿，需要专门设计的基础模型来释放其潜力。

## 方法详解

### 整体框架

输入是一个个体一周的行为数据：27 个变量在 168 小时（7 天 × 24 小时）上的不规则采样值，经过小时级聚合后形成 168×27 的稀疏矩阵。经过 tokenizer 编码后输入骨干网络，输出经时间维度平均池化得到单一 embedding 向量，用于下游健康检测任务的线性探针评估。

### 关键设计

1. **TST Tokenizer（最终选择）**:

    - 功能：将不规则行为数据转化为稠密输入序列
    - 核心思路：构建 168×54 矩阵（27 个变量的值 + 27 个缺失指示符）。缺失值用全局均值（z-score 后即零）填充。每小时的 54 维向量作为一个 patch，通过单层 MLP 映射到 embedding 向量，形成 168 个 token 的序列
    - 设计动机：看似简单的全局均值填充意外地优于更复杂的个体级均值填充，作者推测是因为个体级估计在数据量有限时噪声太大。稠密格式让模型通过缺失指示符自行学习处理缺失模式

2. **mTAN 和 Tuple Tokenizer（候选方案）**:

    - mTAN：同样使用 168×54 矩阵，但通过多时间注意力网络学习时间对齐的 embedding，直接处理不规则性。性能接近但略逊于 TST
    - Tuple：将每个观测视为（时间, 变量类型, 数值）三元组，通过可学习嵌入生成 token。自然处理缺失值（未观测直接不出现），但序列长度可变且可能很长

3. **Mamba-2 骨干架构（最终选择）**:

    - 功能：将 token 序列编码为上下文表示
    - 核心思路：使用双向 Mamba-2（选择性状态空间模型），前向和后向各运行一次后合并。最终输出取所有时间步的平均作为单周 embedding
    - 设计动机：连续时间状态空间模型的学习离散化步长天然适合不规则时间间隔。尽管 Transformer 在可穿戴基础模型中占主导，Mamba-2 在本场景中持续优于 Self-Attention 和 Rotary Transformer。初始消融中限制 Mamba-2 层数不超过 Transformer（受注意力计算内存限制），在此条件下 TST+Mamba-2 组合仍一致最优

### 损失函数

使用正则化对比损失进行自监督预训练。正样本定义为同一参与者不同时间窗口的数据（经 token dropping 增强——随机丢弃 $p\%$ 的时间 token）。加入 KoLeo 正则化防止表示坍塌。作者显式排除了掩码自编码器（MAE）——因为 MAE 要求重建所有输入，会过度强调频繁变量并对稀疏变量不利，初步实验也证实 MAE 下游表现差。

## 实验关键数据

### 人口统计预测（Table 1）

| 嵌入方法 | 年龄 MAE↓ | 生理性别 AUROC↑ |
|---------|---------|---------------|
| 手工特征基线 | 7.89 | 0.931 |
| WBM | 3.67 | 0.999 |
| PPG 模型 | 2.89 | 0.997 |
| **WBM + PPG** | **2.46** | **0.999** |

### 时变健康状态检测（Table 2）

| 嵌入方法 | 糖尿病 AUROC | 怀孕 AUROC | 感染 AUROC | 损伤 AUROC | 睡眠时长 R² | 深睡时长 R² |
|---------|-----------|---------|---------|---------|----------|----------|
| 基线 | 0.737 | 0.804 | 0.632 | 0.608 | 0.104 | 0.172 |
| WBM | 0.765 | 0.864 | 0.749 | 0.680 | 0.590 | 0.266 |
| PPG | 0.829 | 0.873 | 0.730 | 0.673 | 0.110 | 0.327 |
| **WBM+PPG** | **0.828** | **0.921** | **0.763** | **0.688** | **0.601** | **0.383** |

### 关键发现

1. WBM 在 57 项任务中的 39 项上超越手工特征基线（中位 AUROC 提升 0.017），在全部 8 项时变任务上均显著优于基线
2. WBM 在睡眠任务上大幅领先 PPG（睡眠时长 R²: 0.590 vs 0.110），因行为数据覆盖全天 168 小时而 PPG 每天仅几次采样
3. PPG 在糖尿病检测上胜出（0.829 vs 0.765），生理信号对代谢疾病信息更直接
4. WBM+PPG 组合在 47 项基线疾病/药物任务中的 42 项上最佳，怀孕预测 AUROC 达 0.921
5. TST+Mamba-2 在 9 种组合中持续最优，挑战了 Transformer 在可穿戴数据中的主导地位

## 亮点与洞察

- **行为数据 vs 传感器数据的清晰定位**：两者编码不同时间尺度的信息，WBM 捕捉天/周级行为模式（步态变化、运动习惯），PPG 捕捉秒级生理信号（心律异常），组合后在怀孕预测上的 0.921 AUROC 完美说明了互补性
- **简单方案胜过复杂方案**：全局均值填充 > 个体级填充，TST（最简单的 tokenizer）> mTAN/Tuple，在高噪声数据中值得反思
- **数据规模惊人**：162K 参与者、25 亿小时、纵跨 5 年的 AHMS 研究，是目前最大规模的可穿戴行为数据基础模型研究

## 局限性

- Apple 专有数据+知情同意限制，模型权重和代码无法公开，可复现性受限
- 队列偏倚：参与者为 iPhone+Apple Watch 用户，女性、老年人、少数族裔代表性不足
- 仅在 Apple Watch 数据上验证，未测试跨设备泛化
- 仅使用对比学习，未探索非对比 SSL 方法（如 JEPA）
- 不支持健康状态预测/预报（仅检测当前状态）

## 相关工作与启发

- 与 Abbaspourazad et al. (2024a) PPG 基础模型的关系：WBM 是行为数据的对应物，两者互补而非替代
- Merrill & Althoff (2023) 是唯一先前在行为数据上做 SSL 的工作，但仅用 3 个变量、5.2K 个体
- Mamba-2 在临床时序中也优于 Transformer，验证了状态空间模型在健康数据上的潜力

## 评分

⭐⭐⭐⭐ — 在可穿戴行为数据基础模型的空白领域做了重要奠基。数据规模和任务覆盖面（57 项任务）出色。系统化 tokenizer×架构消融对从业者极有参考价值。主要遗憾在于数据和模型不可公开。

<!-- RELATED:START -->

## 相关论文

- [Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](../../NeurIPS2025/self_supervised/tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [TabArena: A Living Benchmark for Machine Learning on Tabular Data](../../NeurIPS2025/self_supervised/tabarena_a_living_benchmark_for_machine_learning_on_tabular_data.md)
- [Test-Time Canonicalization by Foundation Models for Robust Perception](test-time_canonicalization_by_foundation_models_for_robust_perception.md)
- [What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)

<!-- RELATED:END -->
