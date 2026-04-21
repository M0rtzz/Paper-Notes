---
title: >-
  [论文解读] Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry
description: >-
  [NeurIPS 2025][医学图像][脑连接矩阵] 提出DiffeoCFM，利用全局微分同胚诱导的拉回度量，将黎曼流形上的条件流匹配等价转化为欧几里得空间中的标准CFM，实现对脑连接矩阵（SPD/相关矩阵）的高效生成，同时严格保持流形约束，在3个fMRI和2个EEG数据集上达到SOTA。
tags:
  - NeurIPS 2025
  - 医学图像
  - 脑连接矩阵
  - 黎曼流匹配
  - 拉回几何
  - 条件流匹配
  - 对称正定矩阵
  - 相关矩阵
  - fMRI
  - EEG
---

# Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry

**会议**: NeurIPS 2025  
**arXiv**: [2505.18193](https://arxiv.org/abs/2505.18193)  
**作者**: Antoine Collas (Inria), Ce Ju (Inria), Nicolas Salvy (Inria), Bertrand Thirion (Inria, CEA, Université Paris-Saclay)  
**代码**: [github.com/antoinecollas/DiffeoCFM](https://github.com/antoinecollas/DiffeoCFM)  
**领域**: medical_imaging  
**关键词**: 脑连接矩阵, 黎曼流匹配, 拉回几何, 条件流匹配, 对称正定矩阵, 相关矩阵, fMRI, EEG  

## 一句话总结

提出DiffeoCFM，利用全局微分同胚诱导的拉回度量，将黎曼流形上的条件流匹配等价转化为欧几里得空间中的标准CFM，实现对脑连接矩阵（SPD/相关矩阵）的高效生成，同时严格保持流形约束，在3个fMRI和2个EEG数据集上达到SOTA。

## 研究背景与动机

### 问题背景
脑功能连接矩阵（协方差或相关矩阵）是fMRI、EEG、MEG等神经影像分析的核心表示，广泛用于运动想象分类、脑年龄预测和疾病诊断。这些矩阵天然属于对称正定矩阵（SPD）流形$\mathbb{S}_d^{++}$或相关矩阵流形$\text{Corr}_d$，是非欧几里得的黎曼流形。

### 已有工作的不足
- **黎曼CFM [Chen & Lipman 2024]**：直接在流形上训练向量场，需要计算测地线、黎曼范数、流形上的ODE积分，**计算代价极高**（比欧几里得对应物慢8-10倍）
- **SPD-DDPM [Huang & Han 2023]**：需要专用的SPDNet架构，训练缓慢
- **CorrGAN [Marti 2020]**：在欧几里得空间训练GAN后通过后处理投影到流形，投影步骤严重损坏样本质量（$\alpha,\beta$-F1下降高达0.76）
- **TriangDDPM/TriangCFM**：直接对矩阵下三角元素建模，生成的矩阵常含负特征值，投影后结构严重失真
- 现有方法要么**几何保真但计算昂贵**，要么**计算高效但破坏流形约束**

### 核心动机
能否找到一种方法，既保持欧几里得空间训练的简洁高效，又严格保证生成样本满足流形约束？关键insight：利用全局微分同胚$\phi:\mathcal{M}\to E$，在欧几里得空间$E$上做所有计算，数学上等价于在黎曼流形$\mathcal{M}$上操作。

## 方法详解

### 理论基础：拉回流形

给定光滑流形$\mathcal{M}$和全局微分同胚$\phi:\mathcal{M}\to E$（$E$为欧几里得空间），可以将欧几里得度量$g_E$通过$\phi$拉回到$\mathcal{M}$上：

$$(\phi^*g_E)_x(\xi,\eta) = g_E(\mathrm{D}\phi(x)[\xi], \mathrm{D}\phi(x)[\eta])$$

此拉回度量下的测地线就是欧几里得直线的像：$\gamma(t)=\phi^{-1}((1-t)\phi(x_0)+t\phi(x_1))$。

### DiffeoCFM：等价性定理

**核心命题1（训练等价性）**：在拉回度量下，黎曼CFM的损失函数可写为标准欧几里得CFM损失：

$$\mathcal{L}(\theta) = \mathbb{E}_{t,y,z_0|y,z_1|y}\|u_\theta^E(t,(1-t)z_0+tz_1,y)-(z_1-z_0)\|_E^2$$

其中$z_i=\phi(x_i)$，无需计算测地线、黎曼范数或流形梯度。

**核心命题2（采样等价性）**：欧几里得ODE $\dot{z}(t)=u_\theta^E(t,z(t),y)$ 的解与黎曼ODE的解通过$\phi$严格对应：$x(t)=\phi^{-1}(z(t))$。

**核心命题3（离散等价性）**：对任意显式Runge-Kutta格式，欧几里得迭代与黎曼迭代也通过$\phi$严格对应。

### 两种微分同胚实例化

**SPD矩阵（EEG协方差）**：使用矩阵对数映射
$$\phi_{\mathbb{S}_d^{++}}(\Sigma) = \text{vec}_{\text{lt}}(\log(\Sigma))$$
诱导Log-Euclidean度量，映射到$\mathbb{R}^{d(d+1)/2}$。

**相关矩阵（fMRI连接性）**：使用归一化Cholesky分解
$$\phi_{\text{Corr}_d}(\Sigma) = \text{vec}_{\text{sl}}(\text{nchol}(\Sigma))$$
其中$\text{nchol}(\Sigma)=\text{diag}(\text{chol}(\Sigma))^{-1}\text{chol}(\Sigma)$，映射到$\mathbb{R}^{d(d-1)/2}$。

### 训练与采样流程

**训练**（Algorithm 1）：对每个类别$y$，将数据映射到欧几里得空间$z=\phi(x)$，拟合类条件高斯作为源分布，经验分布作为目标分布，训练两层MLP（512隐藏单元）最小化标准CFM损失。

**采样**（Algorithm 2）：从源分布采样$z_0$，用dopri5积分器求解欧几里得ODE得到$z_L$，通过$\phi^{-1}(z_L)$映射回流形，**生成的矩阵天然满足SPD/相关矩阵约束**。

## 实验关键数据

### 实验1：质量指标与分类精度（5个数据集综合对比）

在3个fMRI数据集（ABIDE 900人、ADNI 1900扫描、OASIS-3 1800会话）和2个EEG数据集（BNCI2014-002 13人、BNCI2015-001 12人）上评估。质量指标用$\alpha$-precision（保真度）、$\beta$-recall（多样性）及其调和平均$\alpha,\beta$-F1；分类指标用ROC-AUC和F1。

| 数据集 | 方法 | $\alpha,\beta$-F1 ↑ | ROC-AUC ↑ | F1 ↑ | 训练时间(s) |
|--------|------|---------------------|-----------|------|------------|
| ABIDE (fMRI) | TriangCFM | 0.00 | 0.52 | 0.40 | 48.78 |
| ABIDE (fMRI) | DiffeoGauss | 0.38 | 0.66 | 0.53 | 0.07 |
| ABIDE (fMRI) | **DiffeoCFM** | **0.59** | **0.64** | **0.58** | 32.78 |
| ADNI (fMRI) | TriangCFM | 0.01 | 0.56 | 0.34 | 87.37 |
| ADNI (fMRI) | DiffeoGauss | 0.04 | 0.60 | 0.29 | 0.14 |
| ADNI (fMRI) | **DiffeoCFM** | **0.68** | **0.63** | **0.47** | 88.01 |
| OASIS-3 (fMRI) | **DiffeoCFM** | **0.44** | **0.67** | **0.53** | 67.83 |
| BNCI2014 (EEG) | RiemCFM | 0.63 | 0.81 | 0.72 | **1983.58** |
| BNCI2014 (EEG) | **DiffeoCFM** | 0.62 | 0.81 | 0.74 | 253.04 |
| BNCI2015 (EEG) | RiemCFM | 0.88 | 0.73 | 0.66 | **2753.93** |
| BNCI2015 (EEG) | **DiffeoCFM** | **0.89** | 0.73 | 0.65 | 319.83 |

DiffeoCFM在所有fMRI数据集上$\alpha,\beta$-F1大幅领先；在EEG上与RiemCFM质量持平但**训练快8倍、采样快10倍**。

### 实验2：投影对TriangCFM的破坏性影响

| 数据集 | $\Delta\alpha$-precision | $\Delta\beta$-recall | $\Delta\alpha,\beta$-F1 |
|--------|--------------------------|----------------------|------------------------|
| ABIDE | -0.34 | -0.69 | **-0.50** |
| ADNI | -0.63 | -0.74 | **-0.69** |
| OASIS-3 | -0.52 | -0.76 | **-0.64** |
| BNCI2014-002 | +0.13 | -0.56 | -0.19 |
| BNCI2015-001 | +0.00 | -0.19 | -0.09 |

投影到流形后TriangCFM的$\beta$-recall最大下降0.76、F1最大下降0.69，说明后处理投影策略在fMRI相关矩阵上基本不可用。DiffeoCFM通过微分同胚从根本上避免了此问题。

### 神经生理可信性验证
- **fMRI连接组**：DiffeoCFM生成的ADNI类条件Fréchet均值连接组与真实数据一致——阿尔茨海默病组呈现半球间和前后区域连接减弱的典型模式
- **EEG地形图**：DiffeoCFM生成的CSP空间滤波器在$\alpha$(8-12 Hz)和$\beta$(13-30 Hz)频段集中在对侧感觉运动区，与真实EEG的运动想象判别模式高度吻合

## 亮点

- **数学优雅**：三个等价性命题（训练、连续采样、离散积分）严格证明了欧几里得CFM等价于黎曼CFM，理论完备
- **统一框架**：同一框架同时处理SPD矩阵和相关矩阵，仅需更换微分同胚$\phi$，是目前唯一同时支持两类矩阵生成的方法
- **计算效率**：避免所有流形特定运算（测地线、黎曼指数映射、平行传输），相比RiemCFM训练快8倍、采样快10倍，同时质量不降
- **构造性约束保证**：通过$\phi^{-1}$映射回流形，生成样本天然满足SPD/相关矩阵约束，无需后处理投影
- **大规模实验**：覆盖5个数据集、4600+扫描、30000+EEG试验，是脑连接矩阵生成领域迄今最全面的评估

## 局限与展望

- **依赖全局微分同胚的存在性**：SPD和相关矩阵存在天然的微分同胚，但对Stiefel流形等紧致流形不适用，限制了方法的泛化范围
- **高维诅咒**：流形维度随脑区数目$d$二次增长（$d(d-1)/2$），高分辨率脑分区（如400+区域）下样本复杂度指数增长
- **连接性定义敏感**：仅评估了OAS估计的协方差/相关矩阵，未探索偏相关、图Lasso精度矩阵等替代定义的影响
- **评价指标与几何无关**：$\alpha$-precision和$\beta$-recall基于One-Class SVM，不感知黎曼几何结构，可能遗漏神经生理学相关的细微差异
- **数据规模有限**：脑区数目最大约80，未验证更高维度（如Schaefer 400分区）下的可扩展性

## 与相关工作的对比

- **Riemannian CFM [Chen & Lipman 2024]**：在仿射不变度量下直接做黎曼CFM，几何最精确但计算昂贵（训练2754s vs DiffeoCFM 320s），且仅支持SPD不支持相关矩阵
- **SPD-DDPM [Huang & Han 2023]**：需要专用SPDNet网络，训练极其缓慢
- **CorrGAN [Marti 2020]**：欧几里得GAN+后处理投影，投影严重损害质量
- **TriangCFM/TriangDDPM**：直接对三角元素做生成模型，投影后$\alpha,\beta$-F1骤降0.5-0.7
- **Normalizing Flows on Lie Groups [Falorsi et al. 2019]**：利用重参数化学习李群上的概率密度，可视为DiffeoCFM的早期概念前身
- **DiffeoGauss**（本文消融基线）：同样用微分同胚但仅拟合高斯，$\beta$-recall尚可但$\alpha$-precision极低，说明CFM的非线性建模不可或缺

## 评分

- 新颖性: ⭐⭐⭐⭐ — 拉回几何+CFM思路优雅，等价性证明严密，但核心idea（变换后做欧几里得生成模型）相对直觉
- 实验充分度: ⭐⭐⭐⭐⭐ — 5个真实数据集、多种基线、质量+分类+神经生理三层评估体系
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导清晰，图表专业，实验描述详尽
- 价值: ⭐⭐⭐⭐ — 为脑连接矩阵生成提供了实用SOTA方案，隐私保护数据共享等应用前景明确

<!-- RELATED:START -->

## 相关论文

- [Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [Learning Conformational Ensembles of Proteins Based on Backbone Geometry](learning_conformational_ensembles_of_proteins_based_on_backbone_geometry.md)
- [Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)
- [Flexibility-conditioned Protein Structure Design with Flow Matching](../../ICML2025/medical_imaging/flexibility-conditioned_protein_structure_design_with_flow_matching.md)

<!-- RELATED:END -->
