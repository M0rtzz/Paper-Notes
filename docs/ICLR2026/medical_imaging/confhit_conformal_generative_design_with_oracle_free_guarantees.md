---
title: >-
  [论文解读] ConfHit: Conformal Generative Design with Oracle Free Guarantees
description: >-
  [ICLR 2026][医学图像][conformal prediction] 提出 ConfHit 框架，利用密度比加权的共形排列 p 值实现"认证"（判断生成批次是否包含 hit）和"设计"（精简候选集同时保持统计保证），在无需实验验证 oracle 和存在分布偏移的条件下，为生成式分子设计提供有限样本 $1-\alpha$ 覆盖保证。
tags:
  - ICLR 2026
  - 医学图像
  - conformal prediction
  - generative design
  - drug discovery
  - density ratio
  - statistical guarantee
---

# ConfHit: Conformal Generative Design with Oracle Free Guarantees

**会议**: ICLR 2026  
**arXiv**: [2603.07371](https://arxiv.org/abs/2603.07371)  
**代码**: 无  
**领域**: AI for Science / 统计机器学习  
**关键词**: conformal prediction, generative design, drug discovery, density ratio, statistical guarantee

## 一句话总结

提出 ConfHit 框架，利用密度比加权的共形排列 p 值实现"认证"（判断生成批次是否包含 hit）和"设计"（精简候选集同时保持统计保证），在无需实验验证 oracle 和存在分布偏移的条件下，为生成式分子设计提供有限样本 $1-\alpha$ 覆盖保证。

## 研究背景与动机

**领域现状**：深度生成模型（VAE、扩散、自回归 Transformer）在分子发现中表现出色，但实际部署需要保证生成分子确实满足目标性质——这只能通过昂贵的湿实验或体内实验验证。共形预测（Conformal Prediction）提供了模型无关的统计保证框架，近期已被扩展到生成任务（Quach et al., 2023; Shahrokhi et al., 2025）。**现有痛点**：(a) **需要 oracle 访问**——现有 CP 生成方法需要对新生成样本进行实验验证（合成+测试），在药物发现中成本极高且不可行；(b) **分布偏移**——生成样本分布 $Q$ 与历史标注数据分布 $P$ 可能不同，违反可交换性假设；(c) **预算约束**——有限生成预算下，不一定能保证包含有效分子，需要诚实地声明"不够自信"而非盲目声称成功。**核心矛盾**：需要在不验证新样本的前提下提供统计保证，同时处理分布偏移——这是经典 CP 框架的根本困难。**本文目标** 两个核心问题：(i) **认证**——给定生成批次，能否以 $1-\alpha$ 概率保证包含至少一个 hit？ (ii) **设计**——能否精简候选集为最小子集同时保持保证？**切入角度**：利用历史标注数据（已知 $Y_i$）中的"inactive"样本与生成样本间的加权可交换性（密度比校正偏移），无需 oracle。**核心 idea**：密度比加权排列 p 值 + 嵌套检验 = oracle-free 有限样本保证。

## 方法详解

### 整体框架

输入：历史标注数据 $\mathcal{D}_{\text{calib}}=\{(X_i,Y_i)\}_{i=1}^n$（其中 $Y_i \in \{0,1\}$ 为已知性质标签）、生成样本 $\{X_{n+j}\}_{j=1}^N$、置信水平 $\alpha$。ConfHit 工作流：(1) 估计密度比 $w(x) = dQ/dP(x)$；(2) 对每个嵌套子集 $\{X_{n+j}\}_{j=1}^k$ 构造加权共形 p 值 $p_k$；(3) 嵌套检验——找最小 $\hat{N} = \inf\{k: p_k \leq \alpha\}$，输出精简候选集或声明"不够自信"。

### 关键设计

1. **加权共形 p 值（认证问题）**:
    - 功能：量化生成批次中是否存在 hit 的统计置信度
    - 核心思路：利用 inactive 标注样本 $\{X_i: Y_i=0\}$ 和生成样本间的加权可交换性。对 $B$ 个随机排列计算随机化 p 值：$p_N^{\text{rand}} = \frac{\sum_{b=0}^B \bar{w}(\pi^{(b)};\bm{X}) \mathbb{1}\{V(\pi_0;\bm{X}) \leq V(\pi^{(b)};\bm{X})\}}{\sum_{b=0}^B \bar{w}(\pi^{(b)};\bm{X})}$，其中 $\bar{w}(\pi;\bm{X}) = \prod_{j=1}^k w(X_{\pi(n+j)})$ 为联合似然比
    - 设计动机：经典 CP 要求可交换性，但分布偏移打破此假设；通过密度比加权恢复加权可交换性（Tibshirani et al., 2019），并扩展到多测试样本场景
    - **Theorem 3.1**: $\Pr(p_N^{\text{rand}} \leq t \mid \max_{j} Y_{n+j}=0) \leq t$，有限样本、模型无关

2. **嵌套检验（设计问题）**:
    - 功能：找最小候选集 $\hat{\mathcal{C}} = \{X_{n+j}\}_{j=1}^{\hat{N}}$ 同时保持 $1-\alpha$ 保证
    - 核心思路：对每个 $k=1,\ldots,N$ 构造假设 $H_k: Y_{n+j}=0, \forall j \leq k$。将 p 值单调化 $p_k = \max_{k' \geq k} \tilde{p}_{k'}$，取 $\hat{N} = \inf\{k: p_k \leq \alpha\}$
    - **Theorem 3.4**: 嵌套假设结构 + 单调 p 值 → 无需多重检验校正即可控制 $\Pr(\max_{j \leq \hat{N}} Y_{n+j}=0) \leq \alpha$
    - 设计动机：嵌套假设的关键性质——$H_k$ 为真则 $H_\ell$ ($\ell \leq k$) 必然为真——使得停止规则自然避免多重检验问题

3. **密度比估计的鲁棒性框架**:
    - 功能：确保估计误差不破坏保证
    - **Theorem 3.5**: 量化估计误差对覆盖率的膨胀，取决于 p 值临界区域附近的加权误差
    - 三种诊断工具：**(1) 平衡性检查**——加权后校准数据均值应接近生成数据；**(2) 合成偏移验证**——在标注数据中人工引入偏移检验 p 值均匀性；**(3) 敏感性分析**——扰动估计权重检查结论稳定性

### 损失函数 / 训练策略

打分函数 $V$ 的四种选择：(i) Max-pooling $V = \max_j \hat{\mu}(x_{n+j})$，(ii) Sum-of-prediction $V = \sum_j \hat{\mu}(x_{n+j})$，(iii) Rank-sum $V = \sum_j R_{n+j}$，(iv) Likelihood ratio $V = \sum_j \log(\hat{\mu}(x_{n+j})/(1-\hat{\mu}(x_{n+j})))$。打分函数选择影响检验功效但不影响错误率控制。

## 实验关键数据

### 主实验

**任务 1: 约束分子优化 (CMO-DRD2)**，2 个生成模型：

| 模型 | $\alpha$ | 经验错误率 | 平均候选集大小 | 认证率 |
|------|---------|-----------|-------------|--------|
| Hgraph2graph | 0.05 | 0.023 | 3.2 | 89% |
| Hgraph2graph | 0.10 | 0.056 | 2.1 | 94% |
| SELF-EdiT | 0.05 | 0.034 | 2.8 | 91% |
| SELF-EdiT | 0.10 | 0.068 | 1.7 | 96% |

**任务 2: 基于结构的药物发现 (SBDD)**，3 个生成模型：

| 模型 | $\alpha$ | 经验错误率 | 平均候选集大小 |
|------|---------|-----------|-------------|
| TargetDiff | 0.10 | ≤0.10 | 显著 < N |
| DecompDiff | 0.10 | ≤0.10 | 显著 < N |
| MolCRAFT | 0.10 | ≤0.10 | 显著 < N |

所有模型 × 所有 $\alpha$ 水平一致满足覆盖保证（经验错误率 ≤ 名义 $\alpha$）。

### 消融实验

| 消融项 | 影响 |
|--------|------|
| 去除密度比校正 | 错误率超过 $\alpha$（保证失效） |
| 不同打分函数 | Max-pooling 和 Likelihood ratio 功效较好，但控制均成立 |
| 减少校准数据量 | p 值方差增大但保证仍成立 |
| 估计密度比 vs 真实密度比 | 保证在估计误差可控时近似成立（Theorem 3.5） |

### 关键发现

1. **5 个生成模型 × 2 个任务一致有效**，验证了模型无关性
2. **候选集显著精简**：从原始 $N$ 个候选大幅缩减，减少实验成本
3. **密度比校正是必需的**：去除后错误率超 $\alpha$，保证失效
4. **诚实声明能力**：当生成器较弱或预算不足时，ConfHit 输出 $\hat{N}=0$（"不够自信"），而非给出虚假保证
5. **鲁棒性诊断有效**：平衡性检查和敏感性分析能有效识别密度比估计质量

## 亮点与洞察

- **首个 oracle-free 生成模型统计保证框架**：利用历史数据的可交换性结构绕过了 oracle 需求，真正适用于资源受限的药物发现场景
- **嵌套假设避免多重检验校正**：统计学上优雅——检验序列的嵌套结构使得简单的停止规则即可控制整体错误率
- **"认证+设计"问题拆分**：清晰的问题分离使得方法逻辑透明，认证失败时也有信息价值（说明任务本身困难）
- **理论与实践的平衡**：Theorem 3.5 的鲁棒性分析 + 三种诊断工具 = 实际部署中可量化的可靠性

## 局限与展望

- 实验仅用计算 oracle（DRD2 模型、AutoDock Vina）验证，未做真实湿实验
- 高维分子空间中密度比估计仍然困难，估计质量直接影响功效
- 仅处理单性质保证，多性质同时保证（如活性+选择性+毒性）是重要扩展方向
- 协变量偏移假设 $dQ/dP(x,y)=w(x)$ 要求性质完全由结构决定，可能在某些场景下过强

## 相关工作与启发

- **vs Quach et al. (2023)**: 经典共形生成方法，但需要 oracle 验证新样本。ConfHit 利用历史数据的标签信息绕过了这一要求
- **vs CoDrug (Laghuvarapu et al., 2023)**: 做性质预测的共形区间。ConfHit 解决的是生成设计问题——保证候选集包含 hit——不同的问题定义
- **vs 共形选择 (Jin & Candès, 2023b)**: 控制假阳性率。ConfHit 控制的是"无 hit 概率"——不同的错误类型
- **启发**：共形推断从预测到生成的自然扩展方向；密度比估计和分布偏移处理是核心挑战

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 问题定义新颖（oracle-free 生成保证），理论框架（嵌套检验+多测试样本 p 值）原创性强
- 实验充分度: ⭐⭐⭐⭐ 5 模型 × 2 任务 × 多 α 水平全面验证，鲁棒性诊断完善，但缺真实湿实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨清晰，问题动机和方法逻辑紧密衔接
- 价值: ⭐⭐⭐⭐⭐ 直接影响生成式药物发现的实际部署决策，提供了从"试试看"到"有保证"的范式转变

<!-- RELATED:START -->

## 相关论文

- [COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](../../NeurIPS2025/medical_imaging/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)
- [UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](../../ICML2025/medical_imaging/unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)
- [AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [Multivariate Conformal Selection](../../ICML2025/medical_imaging/multivariate_conformal_selection.md)

<!-- RELATED:END -->
