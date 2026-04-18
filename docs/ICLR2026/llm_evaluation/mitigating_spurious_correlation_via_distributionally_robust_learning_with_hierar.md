---
title: >-
  [论文解读] Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets
description: >-
  [ICLR 2026][虚假相关] 提出层次化DRO框架，同时捕获组间（group proportion shifts）和组内（intra-group distributional shifts）不确定性。使用W_∞距离在语义空间定义组内模糊集，在标准基准上达SOTA，且在新设计的少数群体分布偏移设置下——其他方法均失败时——仍保持强鲁棒性。
tags:
  - ICLR 2026
  - 虚假相关
  - 分布鲁棒优化
  - 层次模糊集
  - Wasserstein距离
  - 少数群体偏移
---

# Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets

**会议**: ICLR 2026  
**arXiv**: [2510.02818](https://arxiv.org/abs/2510.02818)  
**代码**: 无  
**领域**: 其他 / 鲁棒学习  
**关键词**: 虚假相关, 分布鲁棒优化, 层次模糊集, Wasserstein距离, 少数群体偏移

## 一句话总结
提出层次化DRO框架，同时捕获组间（group proportion shifts）和组内（intra-group distributional shifts）不确定性。使用W_∞距离在语义空间定义组内模糊集，在标准基准上达SOTA，且在新设计的少数群体分布偏移设置下——其他方法均失败时——仍保持强鲁棒性。

## 研究背景与动机
ERM在存在虚假相关的数据上训练后，对少数群体（如水鸟+陆地背景）性能严重退化。Group DRO通过最小化最差组损失缓解该问题，但隐式假设每组的训练分布能可靠代表真实分布。

然而少数群体样本极少——训练分布与真实分布差距大。这种"组内分布偏移"在现有虚假相关研究中被完全忽略。即使是SOTA方法也在这种设置下崩溃。

本文通过层次化模糊集解决这个问题：第一层允许组比例任意变化（同Group DRO），第二层在每组内允许W_∞范围内的分布偏移。语义空间定义的cost function使得组内偏移既有意义又可计算。

## 方法详解

### 整体框架
Group DRO的层次化扩展：Q = {Σ β_g Q_g : β∈Δ, W_∞(Q_g, P_g) ≤ ε_g}。

### 关键设计

1. **层次化模糊集 (Eq 5/8)**：ρ=∞（组比例任意）+ ε_g = ε/√n_g（少数群体更大的组内模糊半径——样本越少允许偏移越大）。

2. **W_∞距离+语义cost**：使用倒数第二层特征 z(x) 定义cost，同类label才有有限距离。W_∞允许support shifts（vs f-divergence仅能重新加权）。

3. **Surrogate目标 (Thm 4.1)**：将层次DRO转化为语义空间的对抗扰动问题：max_β Σ β_g E_{P_g}[max_{z': ||z'-z(x)||≤ε_g} L(f_L(z'), y)]。

4. **三步坐标优化**：一步投影梯度更新z' → 指数梯度更新β → SGD更新θ。收敛率O(1/√T)。

### 损失函数 / 训练策略
Algorithm 1：z'的PGD + β的mirror descent + θ的SGD。ε的选择使用启发式方法（附录G）。

## 实验关键数据

### 标准基准
| 方法 | Waterbirds WGA | CelebA WGA | CMNIST WGA |
|------|--------------|-----------|-----------|
| ERM | 72.6% | 47.2% | 27.1% |
| Group DRO | 91.4% | 88.9% | 89.3% |
| **Hierarchical DRO** | **92.8%** | **89.5%** | **91.2%** |

### 少数群体偏移设置（核心新贡献）
| 方法 | Waterbirds (shifted) | CelebA (shifted) | 说明 |
|------|---------------------|-----------------|------|
| Group DRO | 崩溃 | 崩溃 | 假设组内分布不变 |
| JTT | 崩溃 | 崩溃 | 同上 |
| **Hierarchical DRO** | **稳定** | **稳定** | 组内模糊集提供鲁棒性 |

### 消融实验
| 配置 | WGA | 说明 |
|------|-----|------|
| ε=0 (纯Group DRO) | 91.4% | 无组内鲁棒性 |
| ε=0.5 | 92.3% | 适度组内扰动 |
| ε=1.0 | **92.8%** | 最优 |
| ε=2.0 | 92.1% | 过度保守 |
| 原始空间cost（非语义） | 90.1% | 语义空间更有效 |

### 关键发现
- 在标准设置下层次DRO略优于Group DRO。
- 关键区分在少数群体偏移设置——仅通过改变训练/测试划分（无人工噪声）就暴露了Group DRO的脆弱性。
- ε_g = ε/√n_g 的设计直觉正确：少数群体需要更大的保护范围。
- W_∞ vs f-divergence：前者允许support shifts，对组内偏移更自然。

## 亮点与洞察
- 揭示了虚假相关文献中被忽略的重要失败模式——少数群体组内分布偏移。
- 层次化模糊集是Group DRO和标准DRO的优雅统一。
- 新的评测设置本身就是对社区的重要贡献。

## 局限与展望
- Surrogate目标是上界，tightness依赖于特征映射z的surjectivity。
- ε的选择仍需启发式——自适应ε是潜在改进方向。
- 计算开销略高于Group DRO（额外的z'优化步骤）。

## 相关工作与启发
- Group DRO + Wasserstein DRO的自然组合，但组织方式和motivation新颖。

## 评分
- 新颖性: ⭐⭐⭐⭐ 层次化模糊集设计+新评测设置
- 实验充分度: ⭐⭐⭐⭐ 标准+新设置双重验证
- 写作质量: ⭐⭐⭐⭐ 数学严谨
- 价值: ⭐⭐⭐⭐ 鲁棒学习的有意义改进
