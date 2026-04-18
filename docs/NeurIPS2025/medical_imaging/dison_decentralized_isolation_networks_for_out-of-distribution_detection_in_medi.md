---
title: >-
  [论文解读] DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging
description: >-
  [NeurIPS 2025][医学图像][OOD检测] 提出 Decentralized Isolation Networks (DIsoN)，通过训练二分类器将测试样本从训练数据中"隔离"来检测 OOD，并通过去中心化参数交换实现在不共享数据的情况下利用训练数据信息，在 4 个医学影像数据集 12 个 OOD 检测任务上取得 SOTA。
tags:
  - NeurIPS 2025
  - 医学图像
  - OOD检测
  - 去中心化学习
  - 隔离网络
  - 医学影像安全
  - 隐私保护
---

# DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging

**会议**: NeurIPS 2025  
**arXiv**: [2506.09024](https://arxiv.org/abs/2506.09024)  
**代码**: [GitHub](https://github.com/FelixWag/DIsoN)  
**领域**: 医学图像  
**关键词**: OOD检测, 去中心化学习, 隔离网络, 医学影像安全, 隐私保护

## 一句话总结
提出 Decentralized Isolation Networks (DIsoN)，通过训练二分类器将测试样本从训练数据中"隔离"来检测 OOD，并通过去中心化参数交换实现在不共享数据的情况下利用训练数据信息，在 4 个医学影像数据集 12 个 OOD 检测任务上取得 SOTA。

## 研究背景与动机

**领域现状**：ML 模型在医学影像中的安全部署需要 OOD 检测能力。现有方法分为 post-hoc（MSP、Energy、Mahalanobis 距离等）和训练时正则化（CIDER、PALM 等）两大类。

**现有痛点**：
   - 大多数 OOD 检测方法在部署后**不使用训练数据**，只依赖模型输出或潜在空间的间接表示
   - 少数利用训练数据的方法（KNN、密度估计）需要将训练数据/嵌入存储在部署端，因隐私、法规和存储限制而不可行
   - 间接表示（如 summary statistics、prototypes、合成样本）无法忠实捕捉原始训练数据的全部特性

**核心矛盾**：直接与训练数据比较对 OOD 检测最有效，但数据共享在医疗场景中通常不可行

**切入角度**：借鉴 Isolation Forest 的"隔离"思想 + 联邦学习的参数交换机制，用收敛速度作为 OOD 分数

**核心 idea**：训练一个二分类器分离单个测试样本和训练数据，OOD 样本容易分离（收敛快），ID 样本难以分离（收敛慢）；通过去中心化训练实现不共享数据

## 方法详解

### 整体框架
部署端（Target Node）持有测试样本 $\mathbf{x}_t$，模型提供方（Source Node）持有训练数据 $\mathcal{D}_s$。两端不共享数据，只交换模型参数。通过多轮通信训练一个 Isolation Network（二分类器），根据收敛所需的通信轮数 $R$ 判断测试样本是否 OOD。

### 关键设计

1. **Isolation Network（集中式版本）**

    - 功能：训练二分类器将单个测试样本 $\mathbf{x}_t$（标签 1）与训练数据 $\mathcal{D}_s$（标签 0）分离
    - 核心思路：构建 mini-batch $B = B_s \cup \{\mathbf{x}_t^{(1)}, ..., \mathbf{x}_t^{(N)}\}$，通过过采样 $\mathbf{x}_t$ 平衡类不均衡。损失函数：
      $\mathcal{L}_c(\theta; B_s, \mathbf{x}_t, N) = \frac{1}{|B_s|+N}\left(\sum_{\mathbf{x}_s \in B_s} L(\theta; \mathbf{x}_s, 0) + N \cdot L(\theta; \mathbf{x}_t, 1)\right)$
    - OOD 分数 = 收敛时间 $K$（分类器连续 $E_{\text{stab}}=5$ 步正确分类 $\mathbf{x}_t$ 且在 $\mathcal{D}_s$ 上精度 ≥ $\tau=0.85$）
    - 设计动机：OOD 样本与训练数据有不同模式，容易被分离（$K$ 小）；ID 样本共享特征，难以分离（$K$ 大）

2. **Decentralized Isolation Networks (DIsoN)**

    - 功能：在不共享数据的情况下近似集中式 Isolation Network 的训练
    - 核心思路：采用类联邦学习的多轮通信协议
        - 初始化：Source Node 用预训练模型 $M_{pre}$ 的特征提取器初始化 + 随机二分类头
        - 每轮：两端各自做 $E$ 步本地更新（SN 在 $\mathcal{D}_s$ 上训练 $m=0$，TN 在 $\mathbf{x}_t$ 上训练 $m=1$）
        - 聚合：$\theta^{(r+1)} = \alpha \cdot \theta_S^{(r,E)} + \beta \cdot \theta_T^{(r,E)}$，其中 $\beta = 1 - \alpha$
    - 理论保证（Proposition 3.1）：当 $E=1$ 且 $\alpha = |B_s|/(|B_s|+N)$ 时，去中心化更新**精确等价于**集中式更新
    - OOD 分数 = 收敛所需通信轮数 $R$

3. **Class-Conditional DIsoN (CC-DIsoN)**

    - 功能：先用预训练模型预测 $\mathbf{x}_t$ 的类别 $\hat{y}$，SN 仅从预测类的训练数据中采样
    - 设计动机：ID 样本与同类训练数据更相似，更难隔离；缩小比较范围增大 ID/OOD 分离度
    - 实现极简：仅需 TN 向 SN 发送预测标签，SN 过滤 mini-batch

4. **实用技术：数据增强 + Instance Normalization**

    - 随机裁切/翻转/颜色抖动防止模型快速记忆单个 $\mathbf{x}_t$（否则 ID/OOD 都迅速收敛，无法区分）
    - 用 Instance Norm 替代 Batch Norm（BN 不适用于单样本 TN）

### 损失函数 / 训练策略
- 二分类交叉熵损失
- 特征提取器用预训练分类模型（ResNet18 + Instance Norm）初始化
- Adam/SGD 优化器，$\alpha=0.8$ 在所有数据集上稳健
- 每轮本地 $E$ 步约等于训练数据一个 epoch

## 实验关键数据

### 主实验：三个医学影像数据集 OOD 检测

| 方法 | 类型 | X-Ray AUROC↑ | Derm AUROC↑ | Ultrasound AUROC↑ | 平均 AUROC↑ | 平均 FPR95↓ |
|------|------|-------------|------------|-------------------|------------|------------|
| MSP | post-hoc | 60.44 | 65.39 | 58.85 | 61.56 | 100.00 |
| fDBD | post-hoc | 68.26 | 63.59 | 60.73 | 64.19 | 84.61 |
| ViM | post-hoc | 62.60 | 68.39 | 59.44 | 63.48 | 78.12 |
| CIDER | regularization | 70.47 | 81.98 | 58.03 | 70.16 | 72.77 |
| PALM | regularization | 65.41 | 77.25 | 59.35 | 67.34 | 74.59 |
| **CC-DIsoN** | **ours** | **84.94** | **89.54** | **65.62** | **80.00** | **59.20** |

CC-DIsoN 平均 AUROC 比最佳基线 CIDER 高 9.8%，FPR95 低 13.6%。

### 消融实验

| 配置 | 平均 AUROC↑ | 平均 FPR95↓ | 说明 |
|------|------------|------------|------|
| DIsoN（无类条件） | 78.3 | 67.4 | 基础版本 |
| **CC-DIsoN** | **80.0** | **59.2** | +1.7% AUROC, -8.2% FPR95 |

### 组织病理学 MIDOG 实验（Near/Far OOD）

| 方法 | Near-OOD 平均 AUROC | Far-OOD 平均 AUROC |
|------|-------------------|-------------------|
| ViM | 62.4 | 92.6 |
| CIDER | 60.9 | 89.0 |
| PALM | 61.2 | 98.3 |
| **CC-DIsoN** | **69.6** | **98.3** |

Near-OOD 中 CC-DIsoN 比次优方法高 8.4%。

### 关键发现
- 数据增强对性能影响巨大：皮肤病数据集无增强 AUROC 仅约 76%，有增强后达 89.5%（+13.7%），因为增强防止了对单个样本的快速记忆
- $\alpha$ 在 0.5-0.95 范围内性能稳健，$\alpha=0.8$ 综合最优（检测性能 vs 收敛速度的权衡）
- ResNet18 是最佳网络大小——Slim-ResNet18 容量不足，ResNet34 不提升反增计算成本
- CC-DIsoN 对预测类别错误有鲁棒性：即使所有 ID 样本预测错误，AUROC 仍与基线竞争力相当
- DIsoN 每样本推理约 40s-4min，在非急诊医疗场景（报告通常数小时后审阅）中可接受

## 亮点与洞察
- **"隔离难度"作为 OOD 分数是优雅的设计**：将 OOD 检测转化为二分类收敛速度的度量，直觉清晰且有理论保证（Proposition 3.1 证明去中心化=集中式的等价性）
- **去中心化设计解决了真实部署痛点**：只交换模型参数不共享数据，完美适配医疗隐私约束，开创了"OOD 检测即服务"的新范式——ML 开发商可以作为远程服务提供训练数据的安全利用
- **Class-conditioning 的简洁性**：仅需传递一个预测标签即可提升性能，实现成本几乎为零
- Instance Norm 替代 Batch Norm 的选择对单样本节点至关重要，这个洞察可迁移到其他 few-shot/single-sample 联邦场景

## 局限与展望
- 推理开销较大（每样本需多轮通信训练），不适合实时场景
- 当前每次只隔离一个样本，批量测试样本需逐个处理，效率可优化
- 理论等价只在 $E=1$ 时严格成立，$E>1$ 是近似
- 实验中 OOD 类型多为伪影（尺子、注释）和域偏移，未测试语义级 OOD（如罕见疾病）
- 依赖预训练模型的特征提取器作为初始化，预训练质量会影响 DIsoN 性能

## 相关工作与启发
- **vs Isolation Forest/Deep iForest**: iForest 用决策树隔离，在高维图像数据上表现差（AUROC ~40-56%）；DIsoN 用神经网络 + 收敛速度替代分裂节点数，更适合图像
- **vs CIDER/PALM**: 这些训练时正则化方法需要修改主模型训练过程；DIsoN 是 post-deployment 方法，不改变主模型，更灵活
- **vs KNN-based OOD**: KNN 需存储训练嵌入在部署端，DIsoN 通过参数交换避免数据传输
- **vs FL-based OOD**: 现有 FL-OOD 假设多节点多分布训练数据检测分布偏移；DIsoN 是单源单目标的隔离任务，动机和技术本质不同

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "隔离难度=OOD分数"的核心想法新颖，去中心化设计有实际价值
- 实验充分度: ⭐⭐⭐⭐⭐ 4数据集12任务+全面消融（α、网络大小、增强、类条件、误分类鲁棒性）
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、理论推导完整、实验设计系统
- 价值: ⭐⭐⭐⭐⭐ 解决了 OOD 检测在隐私约束下利用训练数据这一核心实际问题
