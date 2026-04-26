---
title: >-
  [论文解读] Curvature Tuning: Provable Training-free Model Steering From a Single Parameter
description: >-
  [NeurIPS 2025][曲率调节] 提出 Curvature Tuning（CT），通过在激活函数中注入单个超参数 $\beta$ 来可证明地调节模型决策边界的曲率，无需修改权重即可提升泛化和鲁棒性，同时作为微调方法参数量远少于 LoRA rank 1。
tags:
  - NeurIPS 2025
  - 曲率调节
  - 激活函数
  - 样条理论
  - 参数高效微调
  - 决策边界
---

# Curvature Tuning: Provable Training-free Model Steering From a Single Parameter

**会议**: NeurIPS 2025  
**arXiv**: [2502.07783](https://arxiv.org/abs/2502.07783)  
**代码**: [GitHub](https://github.com/Leon-Leyang/curvature-tuning)  
**领域**: model_compression  
**关键词**: 曲率调节, 激活函数, 样条理论, 参数高效微调, 决策边界

## 一句话总结

提出 Curvature Tuning（CT），通过在激活函数中注入单个超参数 $\beta$ 来可证明地调节模型决策边界的曲率，无需修改权重即可提升泛化和鲁棒性，同时作为微调方法参数量远少于 LoRA rank 1。

## 研究背景与动机

现有参数高效微调（PEFT）方法（如 LoRA、Adapter 等）均**聚焦于权重适配**——引入或更新权重参数。然而它们普遍缺乏可解释性，依赖于启发式超参数选择（如 LoRA 的秩、放置位置、初始化等），缺少理论指导。一个被忽视的关键组件是**激活函数**——它决定了模型的非线性和表达能力。

本文的核心观察：将深度网络视为仿射样条算子，现有微调方法调节的是样条的斜率和断点，而**调节激活函数则改变了模型的底层几何结构**（即决策边界的曲率）。

## 方法详解

### 整体框架

CT 基于深度网络的样条解释：ReLU 网络等价于 max-affine spline 算子。CT 通过将硬性区域选择（one-hot）替换为软性概率选择来平滑决策边界，提供两种实现：
- **S-CT（Steering CT）**：仅引入一个全局超参数 $\beta$，无需训练
- **T-CT（Trainable CT）**：为每个神经元分配独立的可训练 $(\beta, c)$ 对

### 关键设计

**CT Unit（CTU）激活函数**：

$$\varphi_{\beta,c}(\mathbf{x}) = c \cdot \sigma\left(\frac{\beta \mathbf{x}}{1-\beta}\right) \cdot \mathbf{x} + (1-c) \cdot \ln\left[1 + \exp\left(\frac{\mathbf{x}}{1-\beta}\right)\right] \cdot (1-\beta)$$

其中 $\beta \in [0,1]$ 调节曲率，$c \in [0,1]$ 为混合系数，$\sigma(\cdot)$ 为 sigmoid。这是重参数化 SiLU 和 Softplus 的凸组合：

$$\text{SiLU}(\mathbf{x}) = \sigma(\eta \mathbf{x}) \cdot \mathbf{x}, \quad \eta = \frac{\beta}{1-\beta}$$

$$\text{Softplus}(\mathbf{x}) = \frac{1}{\gamma} \cdot \ln[1 + \exp(\gamma \mathbf{x})], \quad \gamma = \frac{1}{1-\beta}$$

CTU 自然涵盖 SiLU（$c=1$）、Softplus（$c=0$）和 GELU 近似（$c=1, \beta=0.64$）。

**$\beta$-VQ 推理框架**：将 max-affine spline 的硬选择替换为基于熵正则化的软选择，最优解为 softmax 形式：

$$\mathbf{t}_r^\beta = \frac{\exp\left(\frac{\beta(\langle \mathbf{A}_{r,\cdot}, \mathbf{x}\rangle + \mathbf{b}_r)}{1-\beta}\right)}{\sum_{i=1}^R \exp\left(\frac{\beta(\langle \mathbf{A}_{i,\cdot}, \mathbf{x}\rangle + \mathbf{b}_i)}{1-\beta}\right)}$$

### 损失函数

S-CT 无训练损失（仅做 $\beta$ 网格搜索）。T-CT 使用标准交叉熵损失训练每层的 $(\beta, c)$ 参数，冻结所有原始权重。

**理论保证**（Theorem 3.1）：对 ReLU 网络 $f$，用 CTU 替换 ReLU（固定 $\beta \in [0,1)$）等价于将 $f$ 投影到一个**光滑函数空间**，保持梯度和曲率有界，同时对相同参数 $\mathbf{W}$ 具有更高的局部表达能力。

## 实验关键数据

### 主实验

**下游迁移精度**（ImageNet 预训练 ResNet，12 个数据集平均准确率 %）：

| 方法 | 可训练参数 | ResNet-18 | ResNet-50 |
|------|----------|-----------|-----------|
| Frozen (LP) | 0 | 73.96 | 76.24 |
| S-CT | 1 | 75.34 | 76.92 |
| LoRA (r=1) | 35K-79K | 73.64 | 78.68 |
| **T-CT** | 4K-45K | **78.26** | **81.31** |

T-CT 相比 LP 提升 8.59%/8.34%（ResNet-50/152），相比 LoRA(r=1) 提升 4.64%/1.70%，参数量仅为 LoRA 的 11%-59%。

**对抗鲁棒性**（RobustBench $\ell_\infty$ 攻击）：

| 模型 | 数据集 | Frozen | S-CT | 最优 $\beta$ |
|------|--------|--------|------|-------------|
| ResNet-18 | CIFAR-10 | 11.17% | 14.93% | 0.90 |
| ResNet-18 | CIFAR-100 | 4.47% | 6.90% | 0.92 |
| ResNet-18 | ImageNet | 0.00% | 7.00% | 0.89 |

S-CT 无需对抗训练即可显著提升鲁棒性。

### 消融实验

- S-CT 最优 $\beta$ 接近 1（ResNet-50: 0.94, ResNet-152: 0.96），搜索范围可缩窄
- $c=0.5$（CTU）优于纯 SiLU（$c=1$）和纯 Softplus（$c=0$）
- T-CT 学到的 $\beta$ 值呈 U 型分布（集中在 0 和 1 附近），$c$ 值类似，有效均值接近 S-CT 的手选值
- 与 LoRA rank 1/2/4 的完整对比：T-CT 在 ResNet-18/50 上仍优于 LoRA 所有 rank

### 关键发现

1. 调节激活函数曲率和调节权重（LoRA）是**正交互补**的模型改进维度
2. $\beta \to 0$ 使网络变为线性映射（曲率为零），$\beta \to 1$ 恢复原 ReLU；中间值提供最优平衡
3. 鲁棒性提升是 CT 的**隐式偏置**，无需对抗训练目标

## 亮点与洞察

1. **理论驱动的 PEFT**：基于样条理论的可证明保证，不同于现有 PEFT 的启发式设计
2. **极致参数效率**：S-CT 仅 1 个超参数，T-CT 参数量不到 LoRA(r=1) 的 60%
3. 与 LoRA 互补而非替代：CT 调节函数空间，LoRA 调节特征空间
4. CTU 设计兼容 ReLU、SiLU、GELU、Softplus 等多种激活函数

## 局限性

- 理论保证严格成立于分段仿射网络（ReLU/MaxPool），对 Transformer 中 GELU/SiLU 仅有部分保证
- S-CT 需要 $\beta$ 网格搜索（0.7-1.0 范围，步长 0.01），虽然开销低但非完全自动
- 鲁棒性提升在某些设置下有限（如 ResNet-152 CIFAR 上 $\ell_2$ 几乎无改善）

## 相关工作与启发

- 与 Srinivas et al. 的预训练中学习低曲率激活函数不同，CT 首次将激活函数调节提升为 PEFT 方向
- 基于 Balestriero & Baraniuk 的深度网络样条解释，为非线性调节提供了严格的理论工具
- 启发思考：CT 能否与 LoRA 组合使用以获得更大提升？论文实验中已初步显示互补效果

## 评分

- ⭐ 新颖性: 5/5 — 全新的 PEFT 视角，理论-实验结合紧密
- ⭐ 实验充分度: 5/5 — 6 个模型 x 12 数据集 x 泛化/鲁棒性双重验证
- ⭐ 写作质量: 4/5 — 数学推导详尽但部分符号较重
- ⭐ 价值: 5/5 — 开辟了激活函数调节作为 PEFT 的新范式

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[NeurIPS 2025\] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)
- [\[AAAI 2026\] GateRA: Token-Aware Modulation for Parameter-Efficient Fine-Tuning](../../AAAI2026/interpretability/gatera_token-aware_modulation_for_parameter-efficient_fine-tuning.md)
- [\[NeurIPS 2025\] CBMAS: Cognitive Behavioral Modeling via Activation Steering](cbmas_cognitive_behavioral_modeling_via_activation_steering.md)

<!-- RELATED:END -->
