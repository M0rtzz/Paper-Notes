---
title: >-
  [论文解读] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection
description: >-
  [CVPR 2025][异常检测] 提出DPDL方法，通过学习多高斯分布原型并用Schrödinger桥将正常样本扩散映射到原型空间（同时推开异常样本），结合超球空间上的离散特征学习增强泛化性，在9个公开异常检测数据集上取得SOTA（如AITEX上超越AHL 5.0%、ELPV上超越8.7%）。
tags:
  - CVPR 2025
  - 异常检测
  - 开放集
  - Schrödinger桥
  - 分布原型
  - 超球空间
---

# Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection

**会议**: CVPR 2025  
**arXiv**: [2502.20981](https://arxiv.org/abs/2502.20981)  
**代码**: 无  
**领域**: 其他  
**关键词**: 异常检测、开放集、Schrödinger桥、分布原型、超球空间

## 一句话总结
提出DPDL方法，通过学习多高斯分布原型并用Schrödinger桥将正常样本扩散映射到原型空间（同时推开异常样本），结合超球空间上的离散特征学习增强泛化性，在9个公开异常检测数据集上取得SOTA（如AITEX上超越AHL 5.0%、ELPV上超越8.7%）。

## 研究背景与动机

**领域现状**：开放集监督异常检测（OSAD）利用少量已知异常样本训练模型来检测训练时未见过的异常类型。现有方法主要通过生成伪异常来扩充异常分布覆盖，如DRA学习异常的解耦表示，AHL模拟异构异常分布。

**现有痛点**：伪异常生成方法有三个关键问题：（1）无法覆盖所有可能的异常分布模式；（2）模拟异常继承了分布内数据的偏置；（3）正常样本本身的多样性使得正常/异常边界划分困难。这些方法本质上是在试图逼近一个无法完全已知的异常分布。

**核心矛盾**：与其试图穷举所有可能的异常模式（根本做不到），不如反过来精确建模正常样本的紧凑分布边界，让任何偏离此边界的样本都被判为异常。

**本文目标** 如何在正常样本多样性很大的情况下，学习到紧凑且有判别力的正常分布边界，使其对未见异常泛化良好。

**切入角度**：用多高斯分布原型覆盖多样的正常样本，通过Schrödinger桥学习正常样本到原型的最优传输映射，在映射后的空间中正常样本紧凑聚集、异常样本自然远离。

**核心 idea**：正面建模正常分布而非反面逼近异常分布——用可学习高斯原型+Schrödinger桥扩散将正常样本映射到紧凑空间，辅以超球面离散学习增强样本间分离度。

## 方法详解

### 整体框架
输入图像经ResNet-18提取中间特征，然后分两路：（1）Distribution Prototype Learning（DPL）：通过Schrödinger桥将正常样本特征扩散映射到多高斯原型空间，同时用对比方式推开异常样本；（2）Dispersion Feature Learning（DFL）：在超球面空间中利用von Mises-Fisher混合分布增大样本间方向距离。最终通过多实例学习模块输出异常分数。

### 关键设计

1. **分布原型学习（DPL）+ Schrödinger桥**:

    - 功能：将正常样本映射到由多个高斯分布构成的紧凑原型空间
    - 核心思路：定义C个可学习高斯原型$\mathcal{P}_{MGP} = \{\mathcal{N}(\mu_c, \sigma_c)\}_{c=1}^C$，用Schrödinger桥（熵正则化最优传输）学习从正常特征分布$p_0$到原型分布$p_1$的扩散路径。SB的传输计划$\pi$被分解为利用高斯混合的Schrödinger势函数，可以闭式求解漂移函数$g(x,t)$。对异常样本则施加反向推力使其远离原型。关键在于原型参数$\{\alpha_c, \mu_c, \sigma_c\}$和桥函数$\psi_p$联合学习
    - 设计动机：Schrödinger桥比直接拟合高斯混合更灵活，能处理高维复杂分布到简单原型分布的映射，且对未见正常样本有天然的外推能力（扩散过程的连续性）

2. **离散特征学习（DFL）**:

    - 功能：在超球面空间中增大样本间的特征分离度，增强OOD泛化
    - 核心思路：将中间特征映射到单位超球面上，用von Mises-Fisher（vMF）混合分布建模方向特征。通过最大化样本间角距离（最小化余弦相似度）来增加特征空间的离散度。vMF的方向特性天然适合判断样本是否偏离正常方向
    - 设计动机：防止特征坍缩（所有正常样本映射到同一点），在保持正常样本紧凑的同时保留足够的区分度

3. **多实例学习异常评分**:

    - 功能：从特征到异常分数的预测
    - 核心思路：利用多实例学习框架，将图像的多个patch特征作为"包"中的"实例"，综合评估每个patch的异常程度并聚合为图像级分数
    - 设计动机：异常通常是局部的，MIL天然适合处理局部异常定位

### 损失函数 / 训练策略
训练损失包含三部分：DPL的最优传输损失（正常样本映射到原型的负对数似然 + 异常样本的推远损失）、DFL的方向离散损失、以及MIL的二分类损失。使用ResNet-18做backbone。

## 实验关键数据

### 主实验（General Setting, 10个异常训练样本）

| 数据集 | DPDL | AHL | DRA | 提升(vs AHL) |
|--------|------|-----|-----|-------------|
| MVTec AD | 97.7% | 97.0% | 95.9% | +0.7% |
| Optical | 98.3% | 97.6% | 96.5% | +0.7% |
| AITEX | 97.5% | 92.5% | 89.3% | +5.0% |
| ELPV | 93.7% | 85.0% | 84.5% | +8.7% |
| Mastcam | 93.4% | 85.5% | 84.8% | +7.9% |
| Hyper-Kvasir | 93.9% | 88.0% | 83.4% | +5.9% |

### 消融实验

| 配置 | 说明 |
|------|------|
| DPDL (Full) | 最佳性能 |
| w/o DPL | 工业缺陷数据集掉点最多，说明DPL对精确建模正常边界至关重要 |
| w/o DFL | 泛化到未见异常能力下降，超球面离散学习对OOD检测必要 |
| w/o $\mathsf{M}_r$（推离损失） | 掉点最大，说明主动推开异常样本是关键 |

### 关键发现
- 在难度较低的数据集（MVTec AD、SDD）上提升有限（<1%），因为这些数据集异常模式较简单，现有方法已接近饱和
- 在AITEX、ELPV、Mastcam这类困难数据集上提升显著（5-8.7%），说明DPDL对正常样本多样性大、异常模式复杂的场景优势明显
- 单异常样本训练（M=1）时DPDL同样保持大幅领先，说明方法对极少异常监督鲁棒
- 在Hard Setting（异常类别仅从单一类别采样）下同样SOTA，验证了开放集泛化能力

## 亮点与洞察
- **范式转换：从建模异常到建模正常**：不再试图穷举异常模式，而是精确刻画正常样本的分布边界，这种思路更符合实际（正常模式有限且可学，异常模式无限且不可穷举）
- **Schrödinger桥在异常检测中的创新应用**：SB作为分布间最优传输的优雅框架，被创造性地用于将多样的正常特征映射到紧凑原型空间。闭式漂移函数使训练高效
- **超球面+高斯原型双空间学习**：DFL在方向空间增加分离度，DPL在欧氏空间收紧正常边界，两者从不同角度互补增强

## 局限与展望
- 需要预设高斯原型数量C，不同数据集的最优C值不同，缺少自适应确定机制
- Schrödinger桥的$\epsilon$参数设为固定的0.001，对其敏感性分析不充分
- 仅针对图像级异常检测，未扩展到像素级异常分割
- 9个数据集中有2个（BrainMRI、HeadCT）没有超越AHL，这些数据集异常模式较单一，DPDL的复杂建模反而没优势

## 相关工作与启发
- **vs AHL**: AHL通过模拟异构异常分布增强泛化，但仍依赖已知异常来生成伪异常；DPDL转为建模正常分布，从根本上避免了异常覆盖不全的问题
- **vs DRA**: DRA学习解耦的异常表示（已见/伪/残余），但表示能力受限于已见异常的多样性；DPDL不受此限制
- **vs UAD方法**: 无监督方法完全不用异常信息，DPDL利用少量异常样本来主动推开决策边界，是UAD和SAD的折中最优

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Schrödinger桥+多高斯原型的组合非常新颖，范式转换彻底
- 实验充分度: ⭐⭐⭐⭐ 9个数据集、两种设置（general+hard）、消融完整
- 写作质量: ⭐⭐⭐ 数学推导偏重，可读性对非专业读者不友好
- 价值: ⭐⭐⭐⭐ 对工业异常检测和医学图像等实际场景有显著价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Open Set Label Shift with Test Time Out-of-Distribution Reference](open_set_label_shift_with_test_time_out-of-distribution_reference.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](../../NeurIPS2025/others/semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[CVPR 2025\] H2ST: Hierarchical Two-Sample Tests for Continual Out-of-Distribution Detection](h2st_hierarchical_two-sample_tests_for_continual_out-of-distribution_detection.md)
- [\[CVPR 2025\] TailedCore: Few-Shot Sampling for Unsupervised Long-Tail Noisy Anomaly Detection](tailedcore_few-shot_sampling_for_unsupervised_long-tail_noisy_anomaly_detection.md)
- [\[CVPR 2025\] Joint Out-of-Distribution Filtering and Data Discovery Active Learning](joint_out-of-distribution_filtering_and_data_discovery_active_learning.md)

<!-- RELATED:END -->
