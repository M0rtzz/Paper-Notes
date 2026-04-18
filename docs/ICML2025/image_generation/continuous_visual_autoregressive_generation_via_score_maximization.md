---
title: >-
  [论文解读] Continuous Visual Autoregressive Generation via Score Maximization
description: >-
  [ICML 2025][图像生成][连续自回归] 提出连续视觉自回归框架——基于严格适当评分规则理论，用能量分数作为无似然训练目标，替代向量量化实现连续token自回归图像生成，EAR-H达到FID 1.97且推理速度比扩散损失方法MAR快约10倍。
tags:
  - ICML 2025
  - 图像生成
  - 连续自回归
  - 评分规则
  - 能量分数
  - 视觉生成
  - 无量化
---

# Continuous Visual Autoregressive Generation via Score Maximization

**会议**: ICML 2025  
**arXiv**: [2505.07812](https://arxiv.org/abs/2505.07812)  
**代码**: [GitHub](https://github.com/shaochenze/EAR)  
**领域**: 图像生成  
**关键词**: 连续自回归, 评分规则, 能量分数, 视觉生成, 无量化

## 一句话总结
提出连续视觉自回归框架——基于严格适当评分规则理论，用能量分数作为无似然训练目标，替代向量量化实现连续token自回归图像生成，EAR-H达到FID 1.97且推理速度比扩散损失方法MAR快约10倍。

## 研究背景与动机

**领域现状**：视觉自回归模型（VAR）通常需要向量量化（VQ）将连续视觉数据离散化为有限词表token，再用交叉熵训练。VQ tokenizer的重建FID仅5.87，成为生成质量瓶颈。**现有痛点**：连续空间中显式预测似然通常不可行——GIVT用高斯混合逼近，表达力受限于预定义分布族；扩散损失MAR需逐token多步去噪，推理延迟显著。**核心矛盾**：连续token自回归生成需要一个既能保真又无需显式似然的训练目标。**本文目标**：提供统一理论框架来理解和设计连续VAR的训练目标。**切入角度**：严格适当评分规则——统计学中评估概率预测质量的数学工具，保证期望得分在且仅在预测分布等于真实分布时最大化。**核心idea**：交叉熵（离散VAR）是对数评分的特例；对连续空间可用能量评分——无需似然估计，仅需从模型分布采样。

## 方法详解

### 整体框架
1. 用连续KL-16 tokenizer将图像编码为连续token序列（stride 16）
2. Masked autoregressive Transformer预测未知token
3. 输出层用MLP生成器替代softmax，注入随机噪声生成样本
4. 用能量分数（energy score）训练——同时优化样本-目标接近度和样本间多样性

### 关键设计

1. **严格适当评分规则统一框架**:
    - 功能：将离散和连续VAR的训练目标纳入同一理论体系
    - 核心思路：评分规则 $S(p,x): \mathcal{P}\times\mathcal{X}\mapsto\bar{\mathbb{R}}$ 衡量预测分布 $p$ 对观测 $x$ 的适合程度；严格适当意味着 $S(p,q)\leq S(q,q)$ 且等号仅在 $p=q$ 时成立。交叉熵/GIVT对应对数评分，扩散损失对应Hyvärinen评分，本文EAR对应能量评分
    - 设计动机：统一视角揭示各方法的本质差异——对数评分需显式似然（受限于参数化假设），Hyvärinen评分需多步去噪（推理慢）

2. **能量损失（Energy Loss）**:
    - 功能：无似然地训练连续token的概率预测
    - 核心思路：能量评分 $S(p,y) = \mathbb{E}[|x_1-x_2|^\alpha] - 2\mathbb{E}[|x-y|^\alpha]$（$\alpha\in(0,2)$），第一项鼓励生成样本间多样性，第二项要求生成样本接近目标。无偏估计只需两个独立采样 $x_1,x_2\sim p$：$\mathcal{L}(p,y) = |x_1-y|^\alpha + |x_2-y|^\alpha - |x_1-x_2|^\alpha$
    - 设计动机：能量评分的关键优势是仅需采样能力而不需显式概率密度，使得输出分布可以是任意隐式生成模型

3. **MLP生成器（替代Softmax）**:
    - 功能：将Transformer隐藏表示转化为连续token的分布（通过采样过程隐式表示）
    - 核心思路：类似GAN的隐式生成——输入随机噪声 $\epsilon\sim U[-0.5,0.5]^{d_{\text{noise}}}$，通过残差块逐步注入噪声扰动预测。噪声通过adaptive layer normalization（shift/scale/gate）调制隐藏表示
    - 设计动机：不受高斯混合等参数化假设限制，表达力仅受MLP容量约束

### 损失函数 / 训练策略
- 主损失：能量损失，$\alpha=1$（严格适当且梯度稳定）
- 训练温度：前750 epoch标准能量损失，最后50 epoch $\tau_{\text{train}}=0.99$（降低多样性项权重提升质量）
- 推理温度：$\tau_{\text{infer}}=0.7$，仅缩放shift信号
- MLP生成器使用0.25倍学习率（稳定训练）
- Classifier-Free Guidance：10%概率替换条件为dummy token，推理时线性递增guidance scale
- 总训练800 epoch，batch size 2048，AdamW优化器

## 实验关键数据

### 主实验（ImageNet 256×256条件生成）

| 模型 | 类型 | 参数量 | FID↓（w/ CFG） | IS↑ | Precision | Recall |
|------|------|--------|----------------|------|-----------|--------|
| DiT-XL/2 | Diffusion | 675M | 2.27 | 278.2 | 0.83 | 0.57 |
| VAR-d30 | 离散AR | 2.0B | 1.92 | 323.1 | 0.82 | 0.59 |
| GIVT | 连续AR | 304M | 3.35 | — | 0.84 | 0.53 |
| MAR | 连续AR+Diffusion | 943M | 1.55 | 303.7 | 0.81 | 0.62 |
| **EAR-B** | **连续AR+Energy** | **205M** | **2.83** | **253.3** | **0.82** | **0.54** |
| **EAR-L** | **连续AR+Energy** | **474M** | **2.37** | **273.8** | **0.81** | **0.57** |
| **EAR-H** | **连续AR+Energy** | **937M** | **1.97** | **289.6** | **0.81** | **0.59** |

### 消融实验：能量评分指数 $\alpha$ 的影响（EAR-B, 400 epochs, CFG=3.0）

| $\alpha$ | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 |
|----------|-----|------|-----|------|-----|
| FID↓ | **3.55** | 3.73 | 4.10 | 4.32 | 188.1 |
| IS↑ | **230.3** | 223.1 | 212.1 | 204.2 | 6.4 |

### 关键发现
- EAR-B仅205M参数即达FID 2.83，参数效率极高
- 推理速度优势显著：EAR约1秒生成一张图，MAR需约10秒（Fig.2 speed/quality trade-off）
- $\alpha=2$ 时能量评分退化为仅匹配期望（proper但非strictly proper），FID崩溃至188.1——验证了严格适当性的必要性
- $\alpha<1$ 训练崩溃：分母 $|x_1-x_2|^{2-\alpha}$ 趋零导致梯度爆炸
- 连续tokenizer重建FID 1.22 vs VQ tokenizer 5.87——连续token天然优势
- Masked autoregressive（双向注意力）远优于causal（单向），后者FID仅约20

## 亮点与洞察
- 评分规则统一框架极其优雅——交叉熵/GIVT/扩散损失/EAR都是选择不同strictly proper score的特例，这个理论贡献超越了具体方法。
- $\alpha=2$ 的失败案例精确验证了理论预测：proper但非strictly proper的评分规则无法唯一确定最优模型，说明数学上的"严格"条件在实践中至关重要。
- MLP生成器设计精巧——通过adaptive layer norm将噪声转化为shift/scale/gate信号（借鉴DiT），既灵活又可控。

## 局限与展望
- EAR-H (FID 1.97) 仍落后于MAR (FID 1.55)，说明能量评分在绝对质量上可能不及扩散损失
- 仅在ImageNet 256×256上验证，更高分辨率和文生图等场景未探索
- 能量损失需两个独立样本估计——高维空间中样本效率可能受限
- MLP生成器的表达力上限与所需残差块数/宽度的关系未分析
- 训练温度微调（最后50 epoch改 $\tau$）的技巧性较强

## 相关工作与启发
- **vs GIVT (Tschannen et al., 2023)**: 同属连续VAR但受限于GMM参数化，EAR通过隐式生成绕过此限制
- **vs MAR (Li et al., 2024)**: 同属Continuous VAR框架（Hyvärinen score vs Energy score），EAR牺牲少量FID换取10倍推理加速
- **vs VQ-based AR (VQGAN, LlamaGen等)**: 能量score框架从根本上消除了量化信息损失

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 评分规则统一框架是理论突破，连接了三大连续VAR方向
- 实验充分度: ⭐⭐⭐⭐ ImageNet标准基准全面对比，消融验证理论预测
- 写作质量: ⭐⭐⭐⭐⭐ 理论-方法-实验逻辑环环相扣
- 价值: ⭐⭐⭐⭐⭐ 为连续自回归生成提供了统一理论基础和实用高效方法
