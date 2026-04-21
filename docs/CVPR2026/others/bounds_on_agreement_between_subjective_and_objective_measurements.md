---
title: >-
  [论文解读] Bounds on Agreement between Subjective and Objective Measurements
description: >-
  [CVPR2026][主观质量评估] 推导了主观测试 MOS 值与任意客观质量估计器之间 PCC 上界和 MSE 下界的数学闭式解，并提出基于二项分布的投票模型 BinoVotes 在缺少投票方差信息时估算该界。
tags:
  - CVPR2026
  - 主观质量评估
  - MOS
  - Pearson相关系数
  - 均方误差
  - 二项投票模型
  - 客观估计器性能上界
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Bounds on Agreement between Subjective and Objective Measurements

**会议**: CVPR2026  
**arXiv**: [2603.13204](https://arxiv.org/abs/2603.13204)  
**代码**: [NTIA/its-mos-agreement](https://github.com/NTIA/its-mos-agreement)  
**领域**: others (多媒体质量评估 / 主观测试理论)  
**关键词**: 主观质量评估, MOS, Pearson相关系数, 均方误差, 二项投票模型, 客观估计器性能上界

## 一句话总结

推导了主观测试 MOS 值与任意客观质量估计器之间 PCC 上界和 MSE 下界的数学闭式解，并提出基于二项分布的投票模型 BinoVotes 在缺少投票方差信息时估算该界。

## 背景与动机

1. **主观测试是金标准但含噪**：多媒体质量评估依赖主观测试得到 MOS，但离散评分尺度、有限投票人数和个体偏差使 MOS 本身不等于真实质量，追求 PCC=1.0 或 MSE=0.0 既不现实也不可复现。
2. **客观估计器评价缺乏合理基准**：现有做法直接用 PCC/MSE 与 MOS 比较，但忽略了 MOS 自身的噪声下限——即使完美的 oracle 估计器也无法与含噪的 MOS 完全吻合。
3. **已有方法复杂且引入额外指标**：先前工作提出分类错误率、resolving power、ε-insensitive RMSE 等专有度量来应对 MOS 噪声，但偏离了研究者最熟悉的 PCC/MSE 框架。
4. **MOS 连续近似有问题**：用高斯分布近似 MOS 会违反其离散性和有界性，裁剪操作引入额外偏差，缺乏严格数学基础。
5. **投票方差信息常缺失**：众包大规模主观测试（如常用的语音质量数据集）通常只公布 MOS 而不提供个体投票方差，无法直接计算数据驱动的界。
6. **需要为特定数据集设定合理目标**：研究者需要知道"在这个测试集上 PCC 最高能到多少、MSE 最低能到多少"，才能判断其客观估计器是否仍有改进空间。

## 方法详解

### 整体框架

核心思路：最优客观估计器是拥有真实质量 $Y$ 的 oracle。因此 $Y$ 与 MOS $X$ 之间的 PCC/MSE 即为任意客观估计器与 MOS 之间 PCC 的上界和 MSE 的下界。

**唯一假设（well-behaved）**：$\mathbb{E}(R_j|Y)=Y$，即给定真实质量，投票期望等于真实质量。这是所有主观测试的基本公理。

### MSE 下界推导

利用全期望律和条件方差分解：

$$\mathbb{E}(D^2) = \frac{\mathbb{E}(v_r(Y))}{n_v}$$

其中 $v_r(Y) = \text{Var}(R_j|Y)$ 为条件投票方差函数，$n_v$ 为每文件投票数。MSE 下界仅取决于投票方差的期望和投票人数。

### PCC 上界推导

$$\rho(X,Y) = \sqrt{\frac{\text{Var}(X) - \mathbb{E}(D^2)}{\text{Var}(X)}}$$

PCC 上界依赖 MOS 分布方差和 MSE 下界，当 MSE→0 时 PCC→1。

### BinoVotes 投票模型

用二项分布建模单个投票：$R_j = \frac{s_H-s_L}{n_s-1}B_j + s_L$，其中 $B_j \sim \text{Binomial}(n_s-1, \frac{Y-s_L}{s_H-s_L})$。

**关键性质**：
- 满足 well-behaved 条件 $\mathbb{E}(R_j|Y)=Y$
- 投票方差为真实质量的抛物线函数：$v_r(Y)=\frac{(Y-s_L)(s_H-Y)}{n_s-1}$，在尺度两端为零、中间最大，与实际数据高度吻合
- 天然尊重离散评分尺度和有界性

### BinoMOS 与无方差信息时的界估计

对 BinoVotes 取均值得 BinoMOS，其方差、PMF 均有闭式解。当数据集不提供投票方差信息时，可通过 MOS 样本均值和方差反推 BinoVotes 预测方差：

$$\hat{\sigma}_{BV}^2 = \frac{n_v}{n_m-1}\left((\hat{\mu}_X-s_L)(s_H-\hat{\mu}_X)-\hat{\sigma}_X^2\right)$$

也可使用 18 个测试的全局平均观测方差 $\hat{\sigma}_{GV}^2=0.64$ 作为替代。

## 实验关键数据

### 数据规模

在 22 个主观测试上验证（18 个有方差信息，4 个无），覆盖语音(14)、图像(2)、视频(2) 领域，共计 **86,450 个文件、超 493,000 次投票**。每文件投票数 $n_v$ 从 3.52 到 28.33 不等。

### 主要结果

| 对比维度 | BinoVotes 界 vs 数据驱动界 | 全局平均界 vs 数据驱动界 |
|---------|------------------------|----------------------|
| RMSE 最大差异 | 0.05 (ITS1997) | 0.09 (TMHINT-QI) |
| RMSE 平均差异 | +0.02 | −0.004 |
| PCC 最大差异 | 0.021 (ITS1997) | 0.05 (TMHINT-QI) |
| PCC 平均差异 | −0.006 | +0.005 |

- RMSE 界范围：0.12–0.51；PCC 界范围：0.86–0.99
- BinoVotes 倾向略微高估投票方差（18 测试中 17 个），因此 RMSE 界偏高、PCC 界偏低，属保守估计
- BinoVotes 预测方差与观测方差平均差 0.13，最大差 0.28

### 消融/分析

- **投票数影响**：$n_v$ 增大 → RMSE 界下降、PCC 界上升（图 3），与中心极限定理一致
- **质量分布影响**：不同真实质量分布（均匀/三角/Beta）对 PCC 界差异明显，对 RMSE 界几乎无影响
- **样本量收敛**：$n_f \geq 50$ 时样本 PCC 与总体 PCC 界差异可忽略（图 4）
- **方差缺失时两种替代的对比**：BinoVotes 更保守但稳定；全局平均方差更接近但依赖外部数据

## 亮点

1. **纯数学推导无需假设 MOS 分布**：仅依赖 well-behaved 公理，得到适用于任意投票过程的 PCC/MSE 界
2. **BinoVotes 模型简洁有效**：单参数二项分布天然尊重离散性、有界性、方差抛物线形状，与 28 组实验数据的方差中值高度吻合
3. **高度实用**：研究者可直接用公式计算特定数据集的合理性能目标，判断模型是否已触及天花板
4. **代码开源**：GitHub 提供完整实现

## 局限与展望

1. 未推导 Spearman 秩相关系数 (SRCC) 的界，而 SRCC 在质量评估领域同样常用
2. 图像和视频测试集偏少（分别仅 2 个），语音测试占主导，覆盖面有限
3. BinoVotes 倾向高估方差（不捕获"受试者避免极端评分"的行为），加入个体偏差项会进一步扩大差距
4. 假设每文件投票数固定为 $n_v$，实际众包测试中常不等
5. 仅讨论了 1-5 MOS 尺度的详细数值，对其他评分尺度（7 分制、连续滑块等）未展开

## 与相关工作的对比

| 方法 | 特点 | 本文优势 |
|------|------|---------|
| ε-insensitive RMSE [13] | 新指标容忍 MOS 噪声 | 保留标准 PCC/MSE，直接给出界 |
| 分类错误率方法 [41,24,33] | 将估计器等价于 k 个受试者 | 更精确的连续值界 |
| Resolving power [3] | 衡量区分能力 | 与常用指标直接关联 |
| MOS 高斯近似 [6] | 连续模型 | BinoVotes 天然离散且有界 |
| SOS 方差抛物线 [9] | 经验拟合 | BinoVotes 数学推导出相同抛物线 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次从投票的内在数学性质推导 PCC/MSE 界，BinoVotes 模型巧妙
- 实验充分度: ⭐⭐⭐⭐ — 22 个测试、86K 文件，但图像/视频领域覆盖不足
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导严谨清晰，从基本假设到闭式解逻辑连贯
- 价值: ⭐⭐⭐⭐ — 为质量评估研究者提供实用的性能天花板判断工具

<!-- RELATED:START -->

## 相关论文

- [Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)
- [Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation](shoe_style-invariant_and_ground-aware_learning_for_dense_foot_contact_estimation.md)
- [WildCap: Facial Albedo Capture in the Wild via Hybrid Inverse Rendering](wildcap_facial_albedo_capture_in_the_wild_via_hybrid_inverse_rendering.md)
- [What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution](what_is_wrong_with_synthetic_data_for_scene_text_recognition_a_strong_synthetic_.md)
- [BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)

<!-- RELATED:END -->
