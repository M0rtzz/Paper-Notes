---
title: >-
  [论文解读] PPG-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation
description: >-
  [NeurIPS 2025][模型压缩][知识蒸馏] PPG-Distill提出一种针对PPG信号的知识蒸馏框架，通过预测级、特征级和Patch级（形态+节律）蒸馏，将大型PPG基础模型的知识迁移到轻量学生模型，在保持性能（最高提升21.8%）的同时实现7倍推理加速和19倍内存压缩。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "知识蒸馏"
  - "PPG信号"
  - "基础模型压缩"
  - "波形形态蒸馏"
  - "节律蒸馏"
---

# PPG-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2509.19215](https://arxiv.org/abs/2509.19215)  
**代码**: [GitHub](https://github.com/LingFengGold/PPG-Distill)  
**领域**: 模型压缩 / 知识蒸馏 / 健康信号处理  
**关键词**: 知识蒸馏, PPG信号, 基础模型压缩, 波形形态蒸馏, 节律蒸馏

## 一句话总结

PPG-Distill提出一种针对PPG信号的知识蒸馏框架，通过预测级、特征级和Patch级（形态+节律）蒸馏，将大型PPG基础模型的知识迁移到轻量学生模型，在保持性能（最高提升21.8%）的同时实现7倍推理加速和19倍内存压缩。

## 研究背景与动机

光电容积脉搏波（PPG）是可穿戴健康监测中最常用的生理信号之一，通过非侵入方式捕捉血容量变化。PPG信号的价值来自两个层面：

**局部波形形态（Morphology）**：反映心血管事件的短时窗口内波形特征

**长程结构节律（Rhythm）**：反映心跳周期性和自主神经调节的跨Patch时序结构

近年来，GPT-PPG、PaPaGei等PPG基础模型展现了强大的泛化能力，但其巨大的参数量和计算需求使其难以部署在智能手表等边缘设备上。

**关键问题**：传统知识蒸馏方法（如输出对齐、特征对齐）只关注全局信息，**忽略了PPG信号独有的局部形态模式和跨Patch节律结构**。而现有PPG基础模型已采用Patch化表示，这些局部动态信息在蒸馏过程中未被充分利用。

## 方法详解

### 整体框架

PPG-Distill在标准的Global KD（预测级+特征级蒸馏）基础上，新增两种Patch级蒸馏策略，形成一个四层次的蒸馏框架：

```
PPG信号 X ∈ ℝ^L
  ├── Patchify → X_p ∈ ℝ^{P×N}  (P=patch长度, N=patch数目, N=L/P)
  │
  ├── Global KD
  │   ├── 预测级蒸馏: L_KD^Y (对齐教师-学生输出)
  │   └── 特征级蒸馏: L_KD^H (对齐信号级特征)
  │
  └── Patch-Level KD (PPG-Distill核心)
      ├── 形态蒸馏 L_mor (Patch内局部波形对齐)
      └── 节律蒸馏 L_rhy (Patch间时序结构对齐)
```

### 关键设计

#### 1. PPG形态蒸馏（Morphology Distillation）

目标：对齐教师和学生在每个Patch上的局部波形表示。

- 教师Patch特征：$H_t^p \in \mathbb{R}^{N \times d_t}$
- 学生Patch特征：$H_s^p \in \mathbb{R}^{N \times d_s}$
- 引入可学习线性适配器 $A \in \mathbb{R}^{d_t \times d_s}$ 对齐维度：$\tilde{H}_t^p = H_t^p A$
- 对Patch向量进行 $\ell_2$ 归一化后，构建相似度矩阵 $Z = \frac{\hat{H}_s^p (\hat{H}_t^p)^\top}{\tau}$

采用 **InfoNCE对比损失**：每个学生Patch与对应位置的教师Patch为正样本对，其余为负样本：

$$\mathcal{L}_{mor} = \frac{1}{N} \sum_{i=1}^{N} \left( -\log \frac{\exp(Z_{ii})}{\sum_{j=1}^N \exp(Z_{ij})} \right)$$

该目标促使学生保留教师在每个Patch上的形态特征判别性。

#### 2. PPG节律蒸馏（Rhythm Distillation）

目标：保持Patch间的时序关系结构（心跳周期性、节拍规律性）。

不同于形态蒸馏关注单个Patch的对齐，节律蒸馏传递教师的**Patch间关系矩阵**。构建教师和学生的成对欧氏距离矩阵：

$$[D_t]_{ij} = \| \phi(H_{t,i}^p) - \phi(H_{t,j}^p) \|_2, \quad [D_s]_{ij} = \| H_{s,i}^p - H_{s,j}^p \|_2$$

归一化后使用Smooth L1损失对齐：

$$\mathcal{L}_{rhy} = \frac{1}{N(N-1)} \sum_{i \neq j} \text{smoothL1}([\tilde{D}_s]_{ij}, [\tilde{D}_t]_{ij})$$

该损失引自关系知识蒸馏（Park et al., 2019），惩罚相对距离的差异而非绝对距离，因此对特征空间的缩放是鲁棒的。

### 损失函数 / 训练策略

**联合优化目标**：

$$\mathcal{L} = \mathcal{L}_{sup} + \alpha \mathcal{L}_{KD}^Y + \beta \mathcal{L}_{KD}^H + \gamma (\mathcal{L}_{mor} + \mathcal{L}_{rhy})$$

- $\mathcal{L}_{sup}$: 监督损失（回归用MAE，分类用交叉熵）
- $\alpha, \beta, \gamma$: 超参数，控制各损失项权重

**训练细节**：
- 教师模型冻结，仅训练学生
- Patch大小: P = 40
- 优化器: Adam, 初始学习率 1e-5, 最大 1e-3
- Warmup + Cosine Annealing调度（warmup ratio 25%）
- Early stopping: patience = 20 epochs
- 温度 $\tau = 2$
- $\alpha, \beta, \gamma$ 在 {0.1, 0.5} 范围内搜索

## 实验关键数据

### 主实验：DaLiA心率估计 + StanfordAF房颤检测

| 数据集 | 教师模型 | 学生模型 | 方法 | 指标1 | 指标2 |
|:---|:---|:---|:---|:---|:---|
| **DaLiA** | GPT-PPG-19m | 教师本身 | — | MSE: 221.78 | MAE: 8.82 |
| | | GPT-PPG-1m | 无蒸馏 | MSE: 255.07 | MAE: 10.08 |
| | | GPT-PPG-1m | +Global KD | MSE: 234.16 (+8.2%) | MAE: 9.44 (+6.4%) |
| | | GPT-PPG-1m | **+PPG-Distill** | **MSE: 215.36 (+15.6%)** | **MAE: 8.34 (+17.3%)** |
| **DaLiA** | PaPaGei | 教师本身 | — | MSE: 160.39 | MAE: 6.81 |
| | | GPT-PPG-1m | 无蒸馏 | MSE: 255.07 | MAE: 10.08 |
| | | GPT-PPG-1m | +Global KD | MSE: 220.26 (+13.7%) | MAE: 8.38 (+16.9%) |
| | | GPT-PPG-1m | **+PPG-Distill** | **MSE: 202.31 (+20.7%)** | **MAE: 7.90 (+21.6%)** |
| **StanfordAF** | GPT-PPG-19m | 教师本身 | — | Acc: 0.93 | F1: 0.88 |
| | | GPT-PPG-1m | 无蒸馏 | Acc: 0.81 | F1: 0.64 |
| | | GPT-PPG-1m | +Global KD | Acc: 0.82 (+0.8%) | F1: 0.65 (+2.7%) |
| | | GPT-PPG-1m | **+PPG-Distill** | **Acc: 0.87 (+6.7%)** | **F1: 0.77 (+21.8%)** |
| **StanfordAF** | PaPaGei | 教师本身 | — | Acc: 0.83 | F1: 0.70 |
| | | GPT-PPG-1m | +Global KD | Acc: 0.83 (+1.8%) | F1: 0.67 (+5.7%) |
| | | GPT-PPG-1m | **+PPG-Distill** | **Acc: 0.88 (+7.7%)** | **F1: 0.77 (+21.4%)** |

**关键发现**：在DaLiA上以GPT-PPG-19m为教师时，PPG-Distill蒸馏的GPT-PPG-1m（MSE: 215.36）**超越了教师**（MSE: 221.78），仅用1/19的参数。

### 消融实验：效率对比

| 模型 | DaLiA MAE | Batch/s | 参数量 | 内存(MB) |
|:---|:---:|:---:|:---:|:---:|
| GPT-PPG-19m | 8.82 | 128.06 | 19,018,417 | 72.6 |
| PaPaGei | 6.81 | 225.80 | 5,917,197 | 22.6 |
| MLP | 10.74 | 4248.70 | 41,473 | 0.16 |
| **GPT-PPG-1m (PPG-Distill)** | **7.90** | **291.50** | **1,017,197** | **3.9** |

| 模型 | StanfordAF F1 | Batch/s | 参数量 | 内存(MB) |
|:---|:---:|:---:|:---:|:---:|
| GPT-PPG-19m | 0.88 | 39.19 | 19,034,290 | 72.7 |
| PaPaGei | 0.70 | 222.30 | 5,917,454 | 22.6 |
| MLP | 0.54 | 1546.70 | 154,242 | 0.59 |
| **GPT-PPG-1m (PPG-Distill)** | **0.77** | **290.00** | **1,021,690** | **3.9** |

- 推理速度提升：相比GPT-PPG-19m约 **7倍**
- 内存减少：相比GPT-PPG-19m约 **19倍**（72.6 MB → 3.9 MB）
- 参数量减少：19M → 1M（约 **19倍**）

### 关键发现

1. **PPG-Distill一致优于Global KD**：在所有任务和教师模型组合中，Patch级蒸馏始终带来额外增益
2. **学生可超越教师**：在DaLiA上，1M参数的学生模型性能超过19M的教师，说明结构化蒸馏可以弥补甚至逆转容量差距
3. **教师质量影响蒸馏效果**：在回归任务中，更强的教师（PaPaGei）带来更好的学生；但分类任务中此趋势不明显
4. **MLP架构受限**：即使加上Global KD，MLP也无法超越GPT-PPG-1m，表明浅层架构难以建模复杂PPG动态
5. **超参敏感性**：$\alpha$（预测级权重）对性能影响最大，$\beta$（特征级）最稳定，$\gamma$（Patch级）存在最优值（$\gamma=1$）

## 亮点与洞察

1. **PPG信号特性驱动的蒸馏设计**：不是通用蒸馏方法，而是利用PPG信号的形态和节律两大核心特性设计特定蒸馏目标
2. **Patch已有结构的充分利用**：现有PPG基础模型已采用Patchify，但这一中间表示在蒸馏中未被利用——PPG-Distill填补了这一空白
3. **对比学习与关系蒸馏的结合**：形态蒸馏用InfoNCE保持Patch判别性，节律蒸馏用关系距离矩阵保持跨Patch结构，两者互补
4. **实际可部署价值**：19倍内存压缩和7倍加速使智能手表部署成为可能

## 局限与展望

1. **任务覆盖有限**：仅在心率估计和房颤检测两个任务上验证，PPG还可用于血压估计、血氧检测、压力评估等更多任务
2. **学生架构单一**：仅使用GPT-PPG-1m和MLP作为学生，未探索其他轻量架构（如MobileNet变体、TinyTransformer等）
3. **Patch大小固定**：P=40是沿用GPT-PPG的默认设置，未分析不同Patch大小对蒸馏效果的影响
4. **缺乏真实设备部署验证**：效率分析在GPU上进行，未在实际可穿戴设备（ARM芯片、低功耗MCU）上测试
5. **教师模型局限**：仅测试了两个PPG基础模型，未探索非基础模型教师的效果
6. **跨域泛化未验证**：训练和测试数据来自相同分布，未验证蒸馏模型在跨设备、跨人群场景下的鲁棒性

## 相关工作与启发

- **GPT-PPG (Chen et al., 2025)**: 生成式Transformer用于PPG预训练，本文的教师模型之一
- **PaPaGei (Pillai et al., 2025)**: 基于形态感知对比学习的PPG基础模型，5.7万小时临床PPG预训练
- **TimeDistill (Ni et al., 2025)**: 时序数据的跨架构蒸馏，关注多尺度和多周期特征——PPG-Distill更专注于PPG特有的形态和节律
- **Relational KD (Park et al., 2019)**: 关系知识蒸馏，PPG-Distill的节律蒸馏借鉴了其距离矩阵对齐的思想
- **InfoNCE (van den Oord et al., 2018)**: 对比学习损失，用于PPG-Distill的形态蒸馏

本文的启发是：**垂直领域的信号特性应该指导蒸馏方法的设计**。通用蒸馏方法忽略了PPG的波形形态和节律结构，而利用这些领域知识可以显著提升蒸馏效果。这一思路可推广到ECG、EEG等其他生理信号的基础模型压缩。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将知识蒸馏专门应用于PPG信号，形态+节律蒸馏设计有创意
- **理论深度**: ⭐⭐⭐ — 方法设计直观有效，但理论分析较少
- **实验充分性**: ⭐⭐⭐⭐ — 多教师、多任务、效率分析、消融实验齐全
- **实用价值**: ⭐⭐⭐⭐⭐ — 19倍内存压缩和7倍加速使边缘部署可行，有开源代码
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法描述详尽，实验呈现规范

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)
- [\[ACL 2026\] BaseCal: Unsupervised Confidence Calibration via Base Model Signals](../../ACL2026/model_compression/basecal_unsupervised_confidence_calibration_via_base_model_signals.md)
- [\[NeurIPS 2025\] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)
- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
