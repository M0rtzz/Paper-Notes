---
title: >-
  [论文解读] Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification
description: >-
  [ICLR 2026][医学图像][多模态] 提出DyMo——推理时动态模态选择框架，通过理论推导将多模态任务相关信息增益转化为可计算的MTIR奖励函数（基于分类损失降低代理 + 类原型距离 + 类内相似性校准），在推理时迭代选择性融合可靠的恢复模态，首次系统性解决"丢弃缺失模态损失信息 vs 补全可能引入噪声"的困境。
tags:
  - ICLR 2026
  - 医学图像
  - 多模态
  - dynamic modality selection
  - inference-time
  - information gain
  - discarding-imputation dilemma
---

# Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification

**会议**: ICLR 2026  
**arXiv**: [2601.22853](https://arxiv.org/abs/2601.22853)  
**代码**: [GitHub](https://github.com/siyi-wind/DyMo)  
**领域**: 多模态学习 / 医学图像  
**关键词**: incomplete multimodal, dynamic modality selection, inference-time, information gain, discarding-imputation dilemma

## 一句话总结

提出DyMo——推理时动态模态选择框架，通过理论推导将多模态任务相关信息增益转化为可计算的MTIR奖励函数（基于分类损失降低代理 + 类原型距离 + 类内相似性校准），在推理时迭代选择性融合可靠的恢复模态，首次系统性解决"丢弃缺失模态损失信息 vs 补全可能引入噪声"的困境。

## 研究背景与动机

**领域现状**：多模态深度学习在医疗、营销、具身智能等领域取得显著进展，但实际部署时样本经常缺少一个或多个模态（传感器故障、不同采集协议、传输错误等）。

**现有痛点**：

1. **恢复型方法**（如MoPoE、M3Care）通过VAE等重建缺失模态，但重建质量参差不齐——可能产生低保真（模糊/失真）或语义错位（重建结果的类别与输入不一致）的恢复

2. **丢弃型方法**（如ModDrop、MUSE）直接忽略缺失模态，但当高任务相关性的模态缺失时，仅用剩余模态的特征区分度大幅下降

3. **现有动态融合方法**（QMF、DynMM、PDF）主要针对模态内噪声（低保真），无法检测模态间语义错位

**核心矛盾**（丢弃-补全困境）：丢弃缺失模态损失任务相关信息 → 性能下降；补全缺失模态可能引入任务无关噪声或语义错误 → 性能下降。两种方法各有短板，缺乏动态权衡的机制。

**本文切入角度**：不做二选一，而是动态评估每个恢复模态是否"值得融合"——如果恢复增加了任务相关信息就接受（正reward），如果引入噪声/错位就拒绝（负reward）。

## 方法详解

### 整体框架

输入：不完整多模态样本 $\mathbb{X} = \{x^{(m)}\}_{m \in \mathcal{I}}$ → 恢复方法 $\Upsilon$（如VAE/TIP）重建缺失模态 → DyMo动态选择算法迭代评估每个恢复模态的MTIR奖励 → 仅融合正奖励模态 → 多模态Transformer网络 $f$ 输出预测。

网络架构 $f$ 由模态专用编码器 $h^{(m)}$ + 多模态Transformer $\psi$（含[CLS] token和注意力mask处理缺失模态）+ 线性softmax分类器 $\zeta$ 组成。

### 关键设计

1. **多模态任务相关信息奖励 (MTIR)**

    - 理论基础：建立互信息 $I(Y;\mathbf{Z})$ 与经验交叉熵损失 $\hat{\mathcal{L}}_{ce}$ 的下界关系——$I(Y;\mathbf{Z}) \geq H(Y) - \hat{\mathcal{L}}_{ce} - G\sqrt{\frac{\ln(1/\delta)}{2|\mathcal{D}|}}$，降低损失可提升信息下界
    - 将分类建模为特征空间中的混合密度估计：$p(y=k|\mathbf{z}) = \frac{\exp(-d_\phi(\mathbf{z}, \mathbf{c}_k))}{\sum_{k'}\exp(-d_\phi(\mathbf{z}, \mathbf{c}_{k'}))}$，其中 $\mathbf{c}_k$ 为训练集类原型
    - MTIR定义为添加恢复模态前后的分类损失差：正值表示恢复提供了有用信息，负值表示恢复引入了有害信息
    - **类内相似性校准 (ICS)**：引入非对称校准项 $\alpha$，当恢复后表示在其预测类簇中的代表性低于恢复前时降权（$\alpha < 1$），增强奖励函数对语义错位的敏感性

2. **迭代选择算法 + 灵活多模态架构**

    - 贪心迭代选择：每步选择MTIR最高的恢复模态加入观测集，移除所有非正reward模态；直到候选集为空
    - 多模态Transformer支持任意模态组合：缺失模态位置使用dummy tokens + attention mask
    - 训练阶段使用随机子集模拟缺失模态（$A$个随机子集），搭配缺失无关对比损失 $\mathcal{L}_{aux}$ 促进类内聚类

### 损失函数 / 训练策略

- 分类损失：$\mathcal{L}_{class} = -\frac{1}{A}\frac{1}{B}\sum_{\mathcal{S} \sim \mathcal{U}_A}\sum_{i=1}^{B}\log p_f(y_i|\{x^{(m)}\}_{m \in \mathcal{S}})$

- 辅助对比损失：$\mathcal{L}_{aux} = -\frac{1}{A}\frac{1}{B}\sum\sum\log\frac{\exp(-d_\phi(\mathbf{z}_i, \mathbf{c}_{y_i})/t)}{\sum_{k'}\exp(-d_\phi(\mathbf{z}_i, \mathbf{c}_{k'})/t)}$

- 总损失：$\mathcal{L} = \mathcal{L}_{class} + \mathcal{L}_{aux}$

- 训练在完整数据集上进行，通过随机子集采样模拟$2^M-1$种缺失模式

## 实验关键数据

### 主实验

在5个数据集（PolyMNIST/MST/CelebA/DVM/UKBB-CAD/UKBB-Infarction）上对比SOTA：

| 方法 | PolyMNIST(η=0.8) | MST(miss{M,T}) | CelebA(miss{T}) | DVM(γ=1) | CAD(γ=1) |
|------|------------------|----------------|-----------------|----------|----------|
| ModDrop | 88.44 | 82.47 | 87.32 | 87.97 | 69.18 |
| MTL | 91.14 | 84.37 | 89.38 | 92.32 | 70.23 |
| OnlineMAE | 90.09 | 86.67 | 86.67 | - | - |
| M3Care† | 87.92 | 85.16 | 91.32 | 93.43 | 72.48 |
| **DyMo_c** | **96.61** | **85.31** | **95.20** | **93.14** | **71.02** |
| **DyMo_e** | **96.81** | **86.84** | 93.67 | **93.36** | **72.17** |

PolyMNIST 80%缺失：DyMo超OnlineMAE +5.67%；DVM全表缺失：超ModDrop +4.11%

### 消融实验

| 设置 | PolyMNIST(η=0.8) | MST(miss{M,T}) |
|------|------------------|----------------|
| Baseline（全融合无选择） | 84.21 | 80.73 |
| S（正reward同时融合） | 94.33 | 82.08 |
| I（迭代选择最高reward） | 94.50 | 82.12 |
| I+C（迭代+校准，完整DyMo） | **96.61** | **85.31** |

### 关键发现

- 不使用动态选择直接融合所有恢复模态（Baseline）性能显著低于DyMo，验证了恢复质量不可靠的假设
- 迭代选择（I）比一次性选择（S）效果略好，ICS校准（C）在多数数据集上提供额外1-3%提升
- DyMo对距离函数选择（cosine vs Euclidean）不敏感，两种距离效果相当
- 现有动态融合方法（QMF/DynMM/PDF）在VAE恢复下效果有限——因为它们无法检测语义错位

## 亮点与洞察

- "丢弃-补全困境"的问题定义清晰到位，首次系统性提出并用理论框架解决
- 从互信息到分类损失再到类原型距离的理论推导链条完整：$I(Y;\mathbf{Z})$ → 损失下界 → Bregman散度 → 可计算的MTIR
- ICS校准的非对称设计（$\alpha \leq 1$ 当恢复后代表性低于恢复前）体现了对恢复模态的"保守主义"——合理的工程直觉
- 训练策略简洁有效：随机子集模拟 + 对比损失，无需额外网络或多阶段训练

## 局限与展望

- ICS校准项在CAD/Infarction数据集上反而降低性能，需要引入数据集特定的超参数调优
- 每个恢复模态都需要一次前向传播计算MTIR——缺失模态数 $M - |\mathcal{I}|$ 越多推理开销越大
- 恢复方法的选择对DyMo影响较大（如TIP对全表恢复能力有限），DyMo的性能上界受限于恢复方法
- 仅在分类任务上验证，未扩展到分割/检测等密集预测任务

## 相关工作与启发

- **vs ModDrop/MUSE**：丢弃型方法在高信息量模态缺失时性能大幅下降（MUSE在MST上下降61%），DyMo通过恢复+选择避免这一问题
- **vs MoPoE/M3Care**：恢复型方法在严重缺失场景下生成不可靠恢复，DyMo通过MTIR过滤不可靠恢复
- **vs QMF/DynMM/PDF**：现有动态融合方法主要关注模态内噪声，无法检测语义错位；DyMo通过类原型距离检测两种类型的不可靠性
- **方法论启发**：用任务损失降低作为信息增益代理的思路适用于任何需要动态决策的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 问题定义新颖，理论推导完整
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集+9种SOTA对比+完整消融+可视化分析
- 写作质量: ⭐⭐⭐⭐ 理论与实验结合紧密，公式推导清晰
- 价值: ⭐⭐⭐⭐ 不完整多模态学习的通用框架，医学图像场景直接适用
