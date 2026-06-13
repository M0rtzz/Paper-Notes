---
title: >-
  [论文解读] Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods
description: >-
  [NeurIPS 2025][预训练][训练数据归因] 明确提出"仅有最终模型"(FiMO)的训练数据归因设定，将问题从"贡献度"重构为"敏感性"度量，提出 further training 作为金标准，并统一推导出多种梯度方法（Grad-Dot、影响函数、TRAK、DataInf 等）均为 further training 的不同阶近似。
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "训练数据归因"
  - "影响函数"
  - "梯度方法"
  - "Further Training"
  - "Final-Model-Only"
---

# Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods

**会议**: NeurIPS 2025  
**arXiv**: [2412.03906](https://arxiv.org/abs/2412.03906)  
**代码**: [IBM/fimoda](https://github.com/IBM/fimoda)  
**领域**: LLM预训练  
**关键词**: 训练数据归因, 影响函数, 梯度方法, Further Training, Final-Model-Only

## 一句话总结
明确提出"仅有最终模型"(FiMO)的训练数据归因设定，将问题从"贡献度"重构为"敏感性"度量，提出 further training 作为金标准，并统一推导出多种梯度方法（Grad-Dot、影响函数、TRAK、DataInf 等）均为 further training 的不同阶近似。

## 研究背景与动机

**领域现状**：训练数据归因(TDA)旨在用训练数据解释模型行为。现有方法分三大类：(1) 基于重训练的方法（Data Shapley、Datamodels）；(2) 沿训练轨迹追踪的方法（TracIn）；(3) 仅在最终模型上应用的梯度方法（影响函数、TRAK）。但文献中从未明确区分这些方法所对应的**问题设定**差异。

**现有痛点**：当前 TDA 文献隐含地假设可以访问训练算法或中间检查点。但在实际中，最常见的场景是"只有最终模型"——例如从 HuggingFace 下载的开源模型。在此设定下，既无法重训练，也没有中间检查点，对应的"理想目标"和"金标准"都不明确。

**切入角度**：作者明确定义了三种访问级别——TAA（可用训练算法）、CPA（可用检查点）、FiMO（仅有最终模型），聚焦 FiMO 设定，将 TDA 问题从"贡献度量"转变为"敏感性度量"。

## 核心问题

- **FiMO 设定下的 TDA 应该追求什么目标？** 无法"回到过去"追溯训练过程，如何衡量训练样本对最终模型的影响？
- **金标准缺失**：现有代理任务（如错标样本检测）不足以评估和发展 FiMO 方法，需要一个直接可衡量的理想标准。
- **梯度方法之间的关系不清**：Grad-Dot、影响函数、TRAK、DataInf 等方法看似各自独立，缺乏统一视角。

## 方法详解

### 1. FiMO 问题重构：从贡献到敏感性

在 TAA 设定中，自然的问题是**贡献**——训练样本 $z_i$ 通过训练过程贡献了多少？但在 FiMO 设定中无法追溯训练过程。作者将问题改为**敏感性**——给定最终模型，它对训练样本 $z_i$ 有多敏感？

### 2. Further Training 金标准

从最终参数 $\theta^f$ 出发，分别在完整训练集 $\mathcal{D}$ 和去掉样本 $i$ 的集合 $\mathcal{D}_{-i}$ 上继续训练：

$$a_i^* = \mathbb{E}_\xi\left[g(z, \theta^f + \Delta\theta(\mathcal{D}_{-i}, \xi)) - g(z, \theta^f + \Delta\theta(\mathcal{D}, \xi))\right]$$

两个关键改进：

- **非收敛修正**：$\theta^f$ 通常不是经验风险的驻点，在 $\mathcal{D}$ 上继续训练也会产生非零变化 $\Delta\theta(\mathcal{D})$，需要减去这一"训练本身的效应"
- **随机性平均**：对训练算法的随机性（如 mini-batch 顺序 $\xi$）取期望，消除随机噪声

### 3. 统一推导：梯度方法 ≈ 近似 Further Training

对 further training 目标做 Taylor 展开+正则化：

$$\widehat{\Delta\theta}(\mathcal{D}') = \arg\min_{\Delta\theta} \nabla R(\mathcal{D}'; \theta^f)^T \Delta\theta + \frac{1}{2}\Delta\theta^T (\nabla^2 R + \lambda I) \Delta\theta$$

对评价函数 $g$ 在 $\Delta\theta$ 做一阶近似后，归因分数简化为：

$$\hat{a}_i = \nabla_\theta g(z, \theta^f)^T (\widehat{\Delta\theta}(\mathcal{D}_{-i}) - \widehat{\Delta\theta}(\mathcal{D}))$$

#### 一阶方法 → Grad-Dot

省略 Hessian 项，直接得到：

$$\hat{a}_i \propto \nabla_\theta g(z, \theta^f)^T \nabla_\theta L(z_i, \theta^f)$$

即梯度内积，对应 Grad-Dot（也是 TracIn 仅用最终检查点的特例）。

#### 二阶方法 → 影响函数族

保留 Hessian 项，通过隐函数定理得到广义影响函数（Proposition 1）：

$$\widehat{\Delta\theta}(\mathcal{D}_{-i,\epsilon}) - \widehat{\Delta\theta}(\mathcal{D}) \approx \epsilon (H(\theta^f) + \lambda I)^{-1} (\nabla_\theta L(z_i; \theta^f) + \nabla^2_\theta L(z_i; \theta^f) \widehat{\Delta\theta}(\mathcal{D}))$$

在驻点 $\widehat{\Delta\theta}(\mathcal{D})=0$ 时退化为经典形式。进一步引入 Gauss-Newton 近似得到 Corollary 2。

#### 各方法的统一归位

| 方法 | 在统一框架中的位置 |
|------|-----|
| **Grad-Dot** | 一阶展开，梯度内积 |
| **Grad-Cos** | 一阶 + 归一化（理论上不成立） |
| **CG / LiSSA** | 二阶，用迭代法求逆 Hessian-梯度乘积 |
| **TRAK$_{M=1}$** | Gauss-Newton + 随机投影降维，$\lambda=0, V=I$ |
| **EK-FAC** | Gauss-Newton + 按层分块 + Kronecker 分解 |
| **DataInf** | Gauss-Newton + 恒等损失 + 平均与逆的交换 |

### 4. 广义影响函数

与经典推导的关键区别：**不假设凸性或驻点性**。Proposition 1 中额外保留了 $\nabla^2_\theta L(z_i; \theta^f) \widehat{\Delta\theta}(\mathcal{D})$ 项，反映非收敛模型的修正。在近驻点情况下，可用反向 Taylor 展开将其简化为：

$$\widehat{\Delta\theta}(\mathcal{D}_{-i,\epsilon}) - \widehat{\Delta\theta}(\mathcal{D}) \approx \epsilon (H(\theta^f) + \lambda I)^{-1} \nabla_\theta L(z_i; \theta^f + \widehat{\Delta\theta}(\mathcal{D}))$$

即将梯度替换为 further training 后的梯度。

## 实验关键数据

### 实验设置

- **数据集**：表格数据（Concrete, Energy, FICO, Folktables）、图像（CIFAR-10 + ResNet-9）、文本（SST-2 + BERT）
- **金标准**：LOO further training，100 个随机种子取平均
- **评价指标**：归因分数向量的余弦相似度
- **对比方法**：Grad-Dot, Grad-Cos, CG, LiSSA, LiSSA-H, TRAK$_{M=1}$, EK-FAC, DataInf（共 8 种）

### 核心发现

1. **一阶方法 vs 影响函数**：一阶方法（Grad-Dot）初始余弦相似度最高（可达 ~0.9），但随 further training 量增大而快速衰减；影响函数方法（CG, LiSSA）更稳定但峰值始终较低
2. **DataInf ≈ Grad-Dot**：尽管 DataInf 尝试引入二阶信息，行为却更像一阶方法（两者余弦相似度 > 0.95）
3. **TRAK$_{M=1}$ 表现不佳**：在 FiMO 设定中只能用 $M=1$（无法重训多个模型），效果大打折扣
4. **平均改善质量**：增加 further training 的随机种子数（从 1 到 100），gold 标准与梯度方法的相似度持续上升，说明平均操作有效
5. **非表格数据更难**：CIFAR-10 和 SST-2 上所有方法的余弦相似度显著低于表格数据

### 计算开销

Further training BERT (SST-2) 共约 1000 GPU-hours (V100)；总实验约 3000 GPU-hours。

## 亮点

1. **问题设定的澄清**：首次明确定义 FiMO 设定并系统阐述其与 TAA/CPA 的区别，这对领域认知有重要整理价值
2. **统一视角**：将 8 种看似不同的梯度方法统一为 further training 的不同阶近似，理论简洁有力
3. **广义影响函数**：不依赖凸性/驻点性假设，提出包含非收敛修正项的推广表达式
4. **实验设计**：100 个随机种子的平均（远超先前工作的规模），揭示了平均操作对金标准质量的重要性
5. **反直觉发现**：影响函数（二阶）不总是优于简单的 Grad-Dot（一阶）——至少在 FiMO 设定下

## 局限与展望

1. **计算成本高**：further training 金标准本身计算昂贵（~1000 GPU-hours/数据集），限制了实验规模
2. **模型规模受限**：最大模型为 BERT-base，未涉及真正的 LLM（如 GPT-3/LLaMA 级别）
3. **LOO 局限**：仅考虑单样本去除，未探索组影响（group influence）
4. **非表格数据效果差**：CIFAR-10 和 SST-2 上所有方法近似质量均不理想，说明距离实用仍有差距
5. **Further training 量的选择**：多少 further training 才"足够"衡量敏感性？论文未给出明确准则
6. **可用 LoRA 加速**：作者提到但未实验用参数高效微调替代全量 further training

## 与相关工作的对比

| 工作 | 与本文的区别 |
|------|------------|
| **Koh & Liang (2017)** | 首创 ML 中的影响函数，但推导假设凸性/驻点性，仅评估 2 种方法 |
| **Bae et al. (2022)** | 提出 PBRF 作为替代金标准，但 PBRF 用非标准 Bregman 距离（专为贴近影响函数设计），不如 further training 通用 |
| **Schioppa et al. (2023)** | 也观察到近似质量随训练衰减，但未明确 FiMO 设定，未做随机性平均 |
| **Basu et al. (2021)** | 发现影响函数随模型深度/宽度变差，但也未区分 FiMO，未做非收敛修正 |
| **Park et al. (2023) TRAK** | 在 TAA 设定（多检查点 $M \gg 1$）表现好，但在 FiMO ($M=1$) 下大打折扣 |

## 启发与关联

1. **一阶 vs 二阶的权衡**：实验揭示了有趣的现象——短程近似一阶更好，长程近似二阶更稳定。能否通过 damping 参数 $\lambda$ 在两者间插值？
2. **金标准的实用化**：作者指出 10-20 个种子即可获得大部分增益（vs 100），加上 LoRA，有望将 further training 变成可行的评估工具
3. **广义影响函数的潜力**：Proposition 1 中包含 $\widehat{\Delta\theta}(\mathcal{D})$ 的修正项在现有方法中被忽略，可能是提升非表格数据效果的突破口
4. **与模型审计的联系**：FiMO 设定天然适合第三方模型审计（model auditing）和数据合规检查（如 GDPR 中的数据影响评估）

## 评分

- 新颖性: ⭐⭐⭐⭐ (问题设定澄清和统一视角有重要贡献，但不是全新方法)
- 实验充分度: ⭐⭐⭐⭐ (8种方法 × 6数据集 × 100种子，规模充分；但模型规模受限)
- 写作质量: ⭐⭐⭐⭐⭐ (条理清晰，数学推导严谨，讨论深入)
- 价值: ⭐⭐⭐⭐ (对 TDA 领域的认知整理很有价值，实验发现有指导意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)
- [\[NeurIPS 2025\] Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](through_the_river_understanding_the_benefit_of_schedule-free_methods_for_languag.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](language_model_behavioral_phases_are_consistent_across_archi.md)
- [\[NeurIPS 2025\] Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](nemotron-climb_clustering-based_iterative_data_mixture_bootstrapping_for_languag.md)

</div>

<!-- RELATED:END -->
