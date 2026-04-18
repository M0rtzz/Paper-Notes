---
title: >-
  [论文解读] Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals
description: >-
  [NeurIPS 2025 (AI4Mat Workshop)][图像生成][crystal generation] 针对无机晶体生成模型评估中广泛使用的离散距离函数 (StructureMatcher) 的四大缺陷，提出基于 Magpie 指纹（成分）和 AMD 向量（结构）的连续距离函数，实现更可靠的 uniqueness 和 novelty 度量。
tags:
  - NeurIPS 2025 (AI4Mat Workshop)
  - 图像生成
  - crystal generation
  - uniqueness
  - novelty
  - distance function
  - AMD
  - Magpie
  - StructureMatcher
---

# Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals

**会议**: NeurIPS 2025 (AI4Mat Workshop)  
**arXiv**: [2510.12405](https://arxiv.org/abs/2510.12405)  
**代码**: [GitHub](https://github.com/WMD-group/xtalmet)  
**领域**: 生成模型 / 材料科学 / 评估指标  
**关键词**: crystal generation, uniqueness, novelty, distance function, AMD, Magpie, StructureMatcher  

## 一句话总结
针对无机晶体生成模型评估中广泛使用的离散距离函数 (StructureMatcher) 的四大缺陷，提出基于 Magpie 指纹（成分）和 AMD 向量（结构）的连续距离函数，实现更可靠的 uniqueness 和 novelty 度量。

## 研究背景与动机

**领域现状**：机器学习生成模型（CDVAE、DiffCSP、MatterGen 等）能够从化学空间中快速采样新晶体结构，评估这些模型的核心指标是 uniqueness（生成样本多样性）和 novelty（与训练数据的差异度），两者都依赖于晶体之间的距离函数。

**现有痛点**：最常用的距离函数 $d_{\text{smat}}$（基于 pymatgen 的 StructureMatcher）存在四大问题：
   - **(a) 离散性**：只返回 0/1，无法量化相似程度——物理性质随晶体结构连续变化
   - **(b) 不区分成分/结构差异**：非零距离无法判断是成分不同还是结构不同
   - **(c) 缺乏 Lipschitz 连续性**：原子坐标微小扰动可能导致原胞 (primitive cell) 不连续变化，距离突变
   - **(d) 非置换不变性**：uniqueness 分数依赖生成顺序——同一组样本换个顺序可能给出不同分数

**核心矛盾**：当前评估指标的不可靠性可能导致对生成模型能力的错误判断——例如一个模型生成了大量物理不合理但成分各异的结构，离散指标可能给予高分。

## 方法详解

### 提出的两种连续距离函数

**1. 成分距离 $d_{\text{magpie}}$**：基于 Magpie 指纹的欧氏距离
- Magpie 指纹包含 145 个属性（化学计量属性 + 元素性质统计量）
- 对两个晶体 $x, x'$：$d_{\text{magpie}}(x, x') = \|\text{Magpie}(x) - \text{Magpie}(x')\|_2$

**2. 结构距离 $d_{\text{amd}}$**：基于 Average Minimum Distance (AMD) 向量的 $L_\infty$ 距离
- AMD 向量是结构指纹：$\text{AMD}[k]$ 表示从一个原子到其第 $k$ 近邻的平均距离，对原胞内所有原子取平均
- $d_{\text{amd}}(x, x') = \|\text{AMD}(x) - \text{AMD}(x')\|_\infty$

### 连续 Uniqueness 和 Novelty 定义

$$\text{continuous uniqueness} = \frac{1}{\binom{n}{2}} \sum_{i=1}^{n} \sum_{j=1}^{i-1} d_{\text{continuous}}(x_i, x_j)$$

$$\text{continuous novelty} = \frac{1}{n} \sum_{i=1}^{n} \min_{j=1 \sim m} d_{\text{continuous}}(x_i, y_j)$$

与离散版本（indicator function 计数）不同，连续版本给出了样本间真实的距离度量。

### 理论优势

两种距离函数满足两个鲁棒评估的关键性质：
- **等距不变性 (Isometry invariance)**：对等距晶体 $x \cong x'$，$d(x, x') = 0$
- **Lipschitz 连续性**：若 $x'$ 是从 $x$ 对每个原子偏移至多 $\varepsilon$ 得到的，则 $d(x, x') \leq C\varepsilon$

$d_{\text{smat}}$ 不满足 Lipschitz 连续性（因比较原胞，原胞随原子坐标不连续变化）；$d_{\text{wyckoff}}$（另一种离散结构距离）两个性质都不满足。

### 置换不变性问题

$d_{\text{smat}}$ 的 uniqueness 分数不是置换不变的——因为它违反三角不等式。例如三个结构 $x, x', x''$，若 $d_{\text{smat}}(x,x')=d_{\text{smat}}(x,x'')=0$ 但 $d_{\text{smat}}(x',x'')=1$，不同生成顺序给出 1/3 或 2/3 的 uniqueness。连续距离函数天然满足置换不变性。

## 实验关键数据

### 主实验：6 个生成模型在 MP20 数据集上的评估（10k 样本）

| 指标 | CDVAE | DiffCSP | DiffCSP++ | MatterGen | Chemeleon | ADiT |
|------|-------|---------|-----------|-----------|-----------|------|
| U ($d_{\text{smat}}$) | 0.995 | 0.977 | 0.981 | 0.984 | 0.979 | 0.884 |
| U ($d_{\text{comp}}$) | **0.972** | 0.946 | 0.952 | 0.952 | 0.937 | 0.774 |
| U ($d_{\text{magpie}}$, ×10⁻³) | 1.795 | 1.982 | 2.070 | 2.089 | 2.084 | 2.074 |
| U ($d_{\text{amd}}$) | 1.207 | 1.591 | 1.377 | 1.415 | **2.679** | 1.273 |

**经过热力学稳定性筛选后** ($E_{\text{hull}} \leq 0.1$ eV/atom)：

| 指标 | CDVAE | DiffCSP | DiffCSP++ | MatterGen | Chemeleon | ADiT |
|------|-------|---------|-----------|-----------|-----------|------|
| U ($d_{\text{smat}}$) | 0.035 | 0.289 | 0.272 | 0.352 | **0.375** | 0.316 |
| U ($d_{\text{magpie}}$, ×10⁻³) | 0.002 | 0.177 | 0.160 | 0.253 | **0.298** | - |

### 关键发现

1. **$d_{\text{smat}}$ 主要反映成分差异**：$d_{\text{smat}}$ 和 $d_{\text{comp}}$ 高度相关——非零距离主要来自成分不同而非结构不同
2. **连续指标揭示离散指标遗漏的弱点**：CDVAE 在 $d_{\text{comp}}$ uniqueness 最高，但 $d_{\text{magpie}}$ 最低——说明虽然极少精确重复成分，但总体分布高度集中；DiffCSP++ 在 $d_{\text{wyckoff}}$ 最高但 $d_{\text{amd}}$ 很差
3. **稳定性筛选至关重要**：CDVAE 筛选后性能急剧下降（仅 ~3% 样本通过）——高"创意"分数来自大量物理不合理结构
4. **筛选后 Chemeleon-DNG 在所有 uniqueness 指标上最优，MatterGen 在多数 novelty 指标最优**

## 亮点与洞察

- **问题选取精准**：准确指出了晶体生成领域评估标准的系统性缺陷，且提出的解决方案理论上严格
- **成分 vs 结构正交分解**：将单一的 $d_{\text{smat}}$ 分解为成分距离和结构距离两个正交维度，使评估更有信息量
- **置换不变性问题的发现**：此前被忽视的问题——生成顺序影响 uniqueness 分数——是一个重要的 bug report
- **代码开源**：提供了可复用的 Python 包 xtalmet

## 局限与展望

- **Workshop paper，篇幅有限**：部分分析较浅，缺少对距离函数选择的系统消融
- **AMD 距离对有序度/缺陷的敏感性**未讨论——含点缺陷的晶体 AMD 可能不准确
- **未考虑电子结构差异**：Magpie 和 AMD 都是几何层面的距离，未涉及电子性质差异
- **阈值/参数敏感性未分析**：AMD 的 $k$ 截断值、Magpie 的 145 个属性权重均使用默认设置
- **连续 uniqueness/novelty 的绝对值缺少可解释性**：可以超过 1，跨模型比较时需要归一化

## 相关工作对比

- **vs $d_{\text{smat}}$ (pymatgen StructureMatcher)**：离散→连续，获得 Lipschitz 连续性、置换不变性
- **vs $d_{\text{wyckoff}}$**：离散结构距离，不满足等距不变性（依赖单胞设定的原点/选择），$d_{\text{amd}}$ 对单胞选择不变
- **vs CrystalNN 指纹距离**：不满足 Lipschitz 连续性
- **vs SOAP 正则化熵匹配距离**：也不满足 Lipschitz 连续性

## 评分
- 新颖性: ⭐⭐⭐⭐ 指出问题的角度新颖且有实际意义，但解决方案本身（Magpie/AMD）是已有方法
- 实验充分度: ⭐⭐⭐⭐ 覆盖 6 个模型、多种距离函数，有/无稳定性筛选对比完整
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰、实验表格信息密度高、理论分析简洁
- 价值: ⭐⭐⭐⭐ 对晶体生成领域的评估标准提供了重要修正，开源工具增加实用性
