---
title: >-
  [论文解读] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data
description: >-
  [ICLR 2026][序数嵌入] 提出LORE——首个同时从序数三元组比较中联合学习嵌入表示和内在维度的框架：用非凸Schatten-p拟范数(p<1)正则化替代传统的预设维度策略，通过迭代重加权(IRNN)算法求解并证明收敛到稳定点；在合成数据、LLM模拟感知实验和3个众包数据集上，LORE在维度恢复上远超所有基线方法，同时保持高三元组准确率和语义可解释性。
tags:
  - ICLR 2026
  - 序数嵌入
  - 内在维度恢复
  - Schatten-p拟范数
  - 三元组比较
  - 感知空间
  - 低秩正则化
---

# LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data

**会议**: ICLR 2026  
**arXiv**: [2602.04192](https://arxiv.org/abs/2602.04192)  
**代码**: [GitHub](https://github.com/vivek2000anand/lore_iclr)  
**领域**: 表示学习/感知建模  
**关键词**: 序数嵌入, 内在维度恢复, Schatten-p拟范数, 三元组比较, 感知空间, 低秩正则化

## 一句话总结
提出LORE——首个同时从序数三元组比较中联合学习嵌入表示和内在维度的框架：用非凸Schatten-p拟范数(p<1)正则化替代传统的预设维度策略，通过迭代重加权(IRNN)算法求解并证明收敛到稳定点；在合成数据、LLM模拟感知实验和3个众包数据集上，LORE在维度恢复上远超所有基线方法，同时保持高三元组准确率和语义可解释性。

## 研究背景与动机

**领域现状**：序数嵌入(Ordinal Embedding, OE)从三元组比较("A与B更相似，还是A与C更相似？")中学习感知空间的多维表示，广泛应用于心理物理学（味觉、嗅觉、美学偏好等主观感知）。相比绝对量化评分（如Likert量表），三元组比较不依赖语言描述、不受个体尺度偏差影响。

**核心痛点**：
   - 所有现有OE方法（SOE、FORTE、t-STE、CKL、OENN）均需用户预先指定嵌入维度d'
   - 缺乏判断"真实维度"的准则→实践中通常设置过高的维度
   - 过高维度掩盖真实结构（如10维嵌入实际只需2维→"甜度"被碎片化到多个轴）
   - 科学发现追求简约性（Occam's razor）：低维表示更易解释、计算更高效
   - 唯一尝试恢复维度的Künstle方法需枚举候选维度并逐一训练→不可扩展

**切入角度**：将维度发现融入OE优化本身→用Schatten-p拟范数正则化自动平衡三元组准确率与嵌入秩→无需预设维度。

## 方法详解

### 问题形式化
给定N个感知对象、三元组集合T = {(a,i,j)}（表示a与i比a与j更相似），学习嵌入矩阵Z ∈ R^{N×d'}，使得嵌入空间中的距离关系与三元组一致，且自动恢复内在维度d（d ≪ N）。

### LORE优化目标
$$\min_Z \Psi(Z) = \underbrace{\sum_{(a,i,j)\in T} \log(1+\exp(1+d(z_a,z_i)-d(z_a,z_j)))}_{\text{平滑三元组损失}} + \lambda \underbrace{\sum_{i=1}^{\min\{N,d'\}} \sigma_i(Z)^p}_{\text{Schatten-p正则化}}$$

**三个关键设计选择**：

1. **Schatten-p拟范数 (0<p<1)**：
    - p=1 → 核范数（凸但均匀压缩所有奇异值→bias大）
    - p→0 → 秩函数（NP-hard）
    - p=0.5（论文默认）→非凸但更准确的低秩近似→对大奇异值惩罚小、对小奇异值惩罚大→自动"杀死"冗余维度

2. **Softplus平滑**：将hinge loss替换为log(1+exp(·))→消除零梯度平台→使目标函数处处可微（除嵌入坍塌点外，可通过宽初始化避免）

3. **直接嵌入优化**：优化Z而非Gram矩阵G=ZZ^T → O(Nd')复杂度 vs O(N²)→可扩展到大数据集

### 迭代重加权算法 (Algorithm 1)
- 每步执行SVD分解：U,S,V^T = SVD(Z^k - (1/μ)∇f(Z^k))
- 更新奇异值：S^k = S - (p/μ)σ^{p-1}，截断负值
- 重构嵌入：Z^{k+1} = U·S^k·V^T
- 收敛判断：目标值变化或嵌入变化小于阈值
- 每步复杂度：O(d'(T + Nd'))

### 收敛性保证
**定理**：LORE生成的嵌入序列{Z^k}收敛到稳定点，即 $\sum_{k=1}^{\infty}\|Z^{k+1}-Z^k\|_F < +\infty$。

这是重要保证：虽然目标高度非凸，但OE问题的经验和理论研究表明稳定点通常接近全局最优（Bower等证明d=2时所有局部最优即全局最优）。

### 超参数设置
- p = 0.5（固定，先验研究验证的最优值）
- μ = 0.1（固定，需大于三元组损失Lipschitz常数）
- λ ≈ 0.01（唯一需调的超参，在宽范围内稳定）
- 初始化：高斯随机，方差≥5

## 实验关键数据

### 1. 合成数据（已知真实维度）
- 系统变化4个因素：查询比例、内在秩、感知数量、噪声水平
- **LORE是唯一能恢复真实内在秩的方法**，其他所有方法默认使用最大允许维度
- λ≈0.01在所有条件下均表现稳定→无需精细调参
- 随内在秩增加，LORE能跟踪变化→其他方法完全不变

### 2. LLM模拟感知实验
- 用SBERT嵌入50种食物→截断SVD控制内在维度(1-10)→生成噪声三元组
- LORE准确跟踪内在秩，且三元组准确率显著优于基线
- Dim-CV不仅维度估计更差，运行时间高出**数量级**（log尺度差异！）

### 3. 众包真实数据（3个数据集）

| 数据集 | LORE维度 | 其他方法维度 | LORE准确率 | 最佳基线准确率 |
|--------|---------|------------|-----------|-------------|
| Food-100 | **3.3** | 15 | 82.45% | 82.79% |
| Materials | **2.23** | 15 | **84.08%** | 83.94% |
| Cars | **3.0** | 15 | 52.12% | 54.06% |

- LORE用远低于基线的维度(~3 vs 15)达到相当甚至更高的准确率
- Dim-CV严重欠拟合（Food: 77.67%, Cars: 50.43%）→保守的假设检验策略失败
- LORE运行速度排第二（仅次于FORTE）

### 4. 语义可解释性
- Food-100数据集LORE学到的前3个轴对应可解释的食物属性：
    - 轴1: 甜 → 咸
    - 轴2: 密实 → 轻盈
    - 轴3: 碳水化合物丰富 → 蛋白质/蔬菜
- 无需语义监督即自动发现→对科学发现极有价值

## 与现有方法的系统对比

| 方法 | 优化对象 | 恢复维度 | 可扩展 | 高准确率 | 可解释轴 |
|------|---------|---------|--------|---------|---------|
| GNMDS | Gram矩阵 | ✗ | ✗ | ✗ | ✗ |
| CKL | Gram矩阵 | ✗ | ✗ | ✓ | ✓ |
| FORTE | Gram矩阵 | ✗ | ✓ | ✓ | ✗ |
| t-STE | 嵌入 | ✗ | — | ✓ | ✗ |
| SOE | 嵌入 | ✗ | ✓ | ✓ | ✗ |
| Dim-CV | 多嵌入 | 部分 | ✗ | ✗ | — |
| **LORE** | **嵌入** | **✓** | **✓** | **✓** | **✓** |

## 局限性
- 缺乏精确秩恢复或全局最优的理论保证（仅保证收敛到稳定点）
- 高内在秩时因固定三元组数量和维度诅咒，恢复精度下降
- Cars数据集上所有方法准确率均不高(~52-54%)→极端噪声数据的挑战

## 亮点与洞察
- **心理物理学的核心问题回答**："感知空间有几维？"是心理物理学的根本问题→LORE是首个数据驱动、端到端回答该问题的方法
- **非凸正则化的精妙应用**：Schatten-p (p<1)虽引入额外非凸性→但迭代重加权算法将其分解为一系列凸子问题→保证收敛→在低秩恢复上远优于凸核范数松弛
- **"维度本身就是科学发现"**：知道味觉空间是2维还是10维→直接揭示人类感知的内在结构→这比嵌入本身可能更有价值
- **实用性强**：仅一个需调超参(λ≈0.01)→跨数据集稳定→即将集成到cblearn库→降低使用门槛
- **跨领域潜力**：不限于心理物理学→任何只有相对比较数据（无绝对量度）的场景均适用→如推荐系统、美学评估、材料感知

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个联合学习OE维度和嵌入的方法，Schatten-p在OE中首次应用
- 实验充分度: ⭐⭐⭐⭐⭐ 合成+LLM模拟+3个真实众包数据集，系统消融4个因素
- 写作质量: ⭐⭐⭐⭐⭐ 问题motivation清晰，数学推导严谨，图表信息量大
- 价值: ⭐⭐⭐⭐ 对感知科学和表示学习有重要理论与实用贡献

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ACL 2026\] Similarity-Distance-Magnitude Activations](../../ACL2026/interpretability/similarity-distance-magnitude_activations.md)
- [\[CVPR 2026\] CI-ICE: Intrinsic Concept Extraction Based on Compositional Interpretability](../../CVPR2026/interpretability/ciice_intrinsic_concept_extraction_compositional.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)

<!-- RELATED:END -->
