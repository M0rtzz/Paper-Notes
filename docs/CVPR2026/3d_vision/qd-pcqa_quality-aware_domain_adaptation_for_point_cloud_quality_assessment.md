---
title: >-
  [论文解读] QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment
description: >-
  [CVPR 2026][3D视觉][点云质量评估] 提出质量感知域适应框架 QD-PCQA，通过 Rank-weighted Conditional Alignment 和 Quality-guided Feature Augmentation 两大策略，将图像域的质量评估先验迁移到点云域。
tags:
  - CVPR 2026
  - 3D视觉
  - 点云质量评估
  - 无监督域适应
  - 质量感知特征对齐
  - 跨域迁移
---

# QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment

**会议**: CVPR 2026  
**arXiv**: [2603.03726](https://arxiv.org/abs/2603.03726)  
**作者**: Guohua Zhang, Jian Jin, Meiqin Liu, Chao Yao, Weisi Lin (北京交大, NTU, 北科大)  
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: 点云质量评估, 无监督域适应, 质量感知特征对齐, 跨域迁移

## 一句话总结

提出质量感知域适应框架 QD-PCQA，通过 Rank-weighted Conditional Alignment 和 Quality-guided Feature Augmentation 两大策略，将图像域的质量评估先验迁移到点云域。

## 背景与动机

无参考点云质量评估 (NR-PCQA) 面临标注数据稀缺导致的泛化性问题。人类视觉系统 (HVS) 对质量的感知不依赖于媒体类型，因此可通过无监督域适应 (UDA) 将图像域已标注的质量先验迁移到点云域。然而，现有 UDA-based PCQA 方法（如 IT-PCQA）直接继承图像分类任务的特征对齐策略，忽略了质量评估的特殊性：

- **质量无关的特征对齐**：语义相似但质量不同的样本可能被错误对齐
- **质量无关的特征增强**：Style Mixup 随机混合不考虑质量信息
- **层无关的特征增强**：仅在最终层做增强，忽略层级互补性
- **增强不平衡**：仅增强源域特征，反而扩大域差距

## 核心问题

如何在域适应过程中保持质量感知——既确保特征对齐时质量级别一致，又使模型对质量排序敏感。

## 方法详解

### 3.1 数据预处理

将 3D 点云投影到立方体六个面生成多视图图像，拼接后与自然图像统一 resize 为 $224 \times 224$，共享 ResNet-50 特征提取器。

### 3.2 Quality-guided Feature Augmentation (QFA)

QFA 由三个模块组成：

**Quality-guided Style Mixup (QSM)**：不同于随机 Style Mixup，使用高斯核按质量分数匹配候选样本对：

$$P((x_s^{i^*}, y_s^{i^*}) | (x_s^i, y_s^i)) \propto \exp\left(-\frac{(y_s^i - y_s^{i^*})^2}{2\tau^2}\right)$$

配对后，混合特征的风格统计量（均值/方差）和标签：

$$f_s^{\text{mix}} = \sigma(f)^{\text{mix}} \frac{f_s^i - u(f_s^i)}{\sigma(f_s^i)} + u(f)^{\text{mix}}$$

确保增强后的特征保持质量一致性。

**Multi-Layer Extension 模块**：根据质量分数分层应用 QSM：
- Stage 1 → 高质量样本（浅层对低级失真更敏感）
- Stage 2-3 → 中质量样本
- Stage 4 → 低质量样本（深层捕获高级语义）

**Dual-Domain Augmentation**：对源域使用 QSM multi-layer 增强，对目标域在 Stage 4 后使用普通 SM，缓解增强不平衡问题，增加判别器难度，迫使特征提取器学习更鲁棒的域不变特征。

### 3.3 Rank-weighted Conditional Alignment (RCA)

构建于条件算子差异 (COD) 之上，引入排序权重矩阵：

$$\tilde{\mathbf{K}}_X^{st}(i,j) = k(f_s^i, f_t^j) \cdot (1 + \mathbf{W}^{st}(i,j))$$

$$\mathbf{W}^{st}(i,j) = \max\big(0, -(\hat{y}_s^i - \hat{y}_t^j) \cdot \text{sign}(y_s^i - y_t^j)\big)$$

- **质量条件对齐**：使用源域真实标签和目标域伪标签作为条件，对齐相同质量级别的特征
- **排序加权**：对预测排序与真实排序不一致的样本对赋予更高权重，重点纠正排序偏差

### 3.4 两阶段训练

1. **阶段一**（前 5000 次迭代）：仅使用 DANN 进行初始特征对齐，不使用伪标签
2. **阶段二**：引入 RCA，利用已稳定的模型生成可靠伪标签进行精细对齐

### 总损失

$$\mathcal{L}_{\text{all}}^{\text{mix}} = \lambda_1 \mathcal{L}_P(\hat{y}_s^{\text{mix}}, y_s^{\text{mix}}) + \lambda_2 \mathcal{L}_D(f_s^{\text{mix}}, f_t^{\text{mix}}) + \lambda_3 \mathcal{L}_R(y_s, y_t, f_s, f_t)$$

以 $p > 0.5$ 概率应用混合版本，否则用原始版本。

## 实验关键数据

| 方法 | 模式 | TID2013→SJTU-PCQA PLCC | SROCC | TID2013→WPC PLCC | SROCC |
|------|------|----------------------|-------|-----------------|-------|
| No Adapt | I-to-PC | 0.548 | 0.444 | 0.320 | 0.296 |
| DANN | I-to-PC | 0.596 | 0.512 | 0.325 | 0.296 |
| IT-PCQA | I-to-PC | 0.693 | 0.636 | 0.429 | 0.403 |
| COD | I-to-PC | 0.712 | 0.611 | 0.426 | 0.396 |
| **QD-PCQA** | **I-to-PC** | **0.842** | **0.753** | **0.563** | **0.572** |

| 方法 | KADID→SJTU-PCQA PLCC | SROCC | KADID→WPC PLCC | SROCC |
|------|---------------------|-------|---------------|-------|
| IT-PCQA | 0.703 | 0.641 | 0.432 | 0.402 |
| **QD-PCQA** | **0.843** | **0.724** | **0.553** | **0.534** |

*QD-PCQA 在 SJTU-PCQA 上超越 IT-PCQA 约 21%（PLCC）*

## 亮点

- **质量感知贯穿全流程**：从特征增强到特征对齐均以质量分数为引导
- **分层增强设计合理**：利用不同层对不同质量级别的互补敏感性
- **排序加权机制**：关注"排序不一致"的难样本对，精准纠正域迁移中的排序偏差
- **两阶段训练**：避免早期不可靠伪标签对 RCA 的负面影响

## 局限与展望

- 点云投影为六个正交视图可能丢失 3D 结构信息
- 伪标签质量依赖第一阶段模型水平，可探索更好的伪标签生成策略
- 仅在 SJTU-PCQA 和 WPC 两个点云数据集验证，规模有限
- 质量分层使用固定分位数（33%/67%），可探索自适应分层

## 与相关工作的对比

- vs **IT-PCQA**：IT-PCQA 仅用 DANN 做全局对齐，忽略质量条件；QD-PCQA 加入质量条件对齐 + 排序加权
- vs **StyleAM**：StyleAM 引入 SM 但随机混合不考虑质量；QD-PCQA 用 QSM 保持质量一致性
- vs **COD**：COD 条件对齐但等权处理所有样本对；QD-PCQA 强调排序偏差样本

## 启发与关联

- 质量感知域适应的思路可推广到其他回归型跨域任务（如跨域年龄估计、跨域评分预测）
- 分层特征增强的设计对多尺度特征的利用有普适性启示
- 排序敏感的权重策略可结合 Learning-to-Rank 思路进一步发展

## 评分

- 新颖性: ⭐⭐⭐⭐ — 质量感知域适应在 PCQA 领域是有意义的创新
- 实验充分度: ⭐⭐⭐ — 数据集偏少，缺乏大规模点云数据集验证
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，模块动机充分
- 价值: ⭐⭐⭐⭐ — 提供了 PCQA 泛化性的新思路
