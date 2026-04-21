---
title: >-
  [论文解读] Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors
description: >-
  [ICLR 2026][下一token预测] 通过将训练梯度信号分解为 direct、pre-cached 和 circuit sharing 三种成分，解释了为什么 NTP 训练的 Transformer 会学到对预测当前下一token"无用"的特征，并在 OthelloGPT、小型语言模型和预训练 LLM（Gemma 2）上验证了这一框架的解释力。
tags:
  - ICLR 2026
  - 下一token预测
  - 特征涌现
  - pre-caching
  - circuit sharing
  - 机械可解释性
---

# Understanding the Emergence of Seemingly Useless Features in Next-Token Predictors

**会议**: ICLR 2026  
**arXiv**: [2603.14087](https://arxiv.org/abs/2603.14087)  
**代码**: https://github.com/Markfryazino/useless-features-iclr-code  
**领域**: LLM/NLP  
**关键词**: 下一token预测, 特征涌现, pre-caching, circuit sharing, 机械可解释性

## 一句话总结
通过将训练梯度信号分解为 direct、pre-cached 和 circuit sharing 三种成分，解释了为什么 NTP 训练的 Transformer 会学到对预测当前下一token"无用"的特征，并在 OthelloGPT、小型语言模型和预训练 LLM（Gemma 2）上验证了这一框架的解释力。

## 研究背景与动机

**领域现状**：LLM 通过 NTP（下一 token 预测）目标训练，即学习 $p(x_{t+1}|x_1 \cdots x_t)$。直觉上模型应该只学习对预测下一 token 有用的特征。部分合成任务的研究也确认了这一点——NTP 训练确实只学到即时有用的特征。

**现有痛点**：但大量实证发现 LLM 学到了远超即时 NTP 所需的丰富特征——包括抽象输入特征重建、"世界模型"（如 OthelloGPT 编码的棋盘状态）、以及多步前瞻预测能力。为什么 NTP 目标能驱动这些"看似无用"特征的涌现？已有的可解释性研究主要从目的论视角（特征在成品模型中的算法角色）分析，未探究训练过程中的梯度信号来源。

**核心矛盾**：NTP 目标只提供关于"下一个 token"的监督信号，但模型学到了关于"全局状态"和"未来 token"的特征。梯度信号是如何"穿越"只优化即时预测的目标函数，驱动这些跨位置特征的学习的？

**本文目标** 从梯度信号的信息流视角解释 NTP-useless 特征的涌现机制。具体地：(a) 梯度信号通过什么路径到达参数？(b) 哪些路径负责学习"无用"特征？(c) 能否量化每种机制的贡献？

**切入角度**：利用因果遮蔽 Transformer 的计算图结构，将梯度信号分解为三种独立路径，发展干预（消融机制观察影响）和归因（量化每种机制的影响力）两种分析方法。

**核心 idea**：NTP-useless 特征通过 pre-caching（未来位置的损失信号通过注意力回传）和 circuit sharing（参数共享导致跨位置特征迁移）两种机制从 NTP 目标中涌现。

## 方法详解

### 整体框架
对 Transformer 计算图中的信息流进行三分类分解。固定某位置 $i$ 和层 $k$ 的残差流 $r_{\theta,i}^k(x)$，所有梯度路径被分为：(1) **Direct**：经过 $r_{\theta,i}^k$ 且到 $\hat{x}_{i+1}$（即时 NTP 损失）的路径；(2) **Pre-cached**：经过 $r_{\theta,i}^k$ 但到 $\hat{x}_j$（$j > i+1$，未来位置损失）的路径；(3) **Shared**：不经过 $r_{\theta,i}^k$ 的路径（通过参数共享间接影响）。三者构成梯度的完备分解（Proposition 3.1）。

### 关键设计

1. **梯度三分解（Proposition 3.1）**:

    - 功能：将总梯度 $\nabla_\theta L$ 精确分解为 direct + pre-cached + shared 三个独立成分
    - 核心思路：通过 stop-gradient 操作定义每个成分。Direct：$\nabla_\theta L_i - \nabla_\theta L_i^{sg(k,i)}$；Pre-cached：$\nabla_\theta \sum_{j \neq i} [L_j - L_j^{sg(k,i)}]$；Shared：$\sum_j \nabla_\theta L_j^{sg(k,i)}$。三者之和恰好等于 $\nabla_\theta L$
    - 设计动机：Direct 成分只能驱动 NTP-useful 特征的学习（因为只和即时预测相关），而 pre-cached 和 shared 是"无用"特征涌现的潜在来源

2. **干预方法：Myopic Training 和 m-Untied Training**:

    - 功能：通过消融 pre-caching 或 circuit sharing 观察其缺失的影响
    - 核心思路：**Myopic training**（Wu et al. 2024 提出）阻止跨位置的梯度传播——在 K、V 矩阵处切断梯度，使位置 $i$ 不被激励计算对未来位置有用的特征。**m-Untied training**（本文提出）使用两套独立参数分别处理位置 $m$ 前后的序列，阻断 circuit sharing
    - 设计动机：Pre-caching 增加了表达力（使多层注意力的复杂构造成为可能），circuit sharing 实现了跨位置特征迁移（在某位置 NTP-useful 的特征通过共享参数被编码在另一位置）

3. **归因方法：Feature Mismatch Influence**:

    - 功能：量化训练过程中每种梯度成分对特征涌现的具体贡献
    - 核心思路：定义 feature mismatch $R(x|\theta_1, \theta_2, w_i^k) = \frac{1}{2}(\langle w_i^k, r_{\theta_1,i}^k(x) \rangle - \langle w_i^k, r_{\theta_2,i}^k(x) \rangle)^2$，然后定义 influence $I_i^k(\theta, x | w_i^k, \theta^*, G) = \frac{d}{d\varepsilon} R(x|\theta + \varepsilon G, \theta^*, w_i^k)|_{\varepsilon=0}$，其中 $G$ 是某个梯度成分。对 Adam 优化器进行调整：为三个成分维护独立的动量，确保成分步长之和等于实际优化器步长
    - 设计动机：仅消融（干预）无法区分某个机制"是否有必要"和"实际贡献了多少"。归因方法通过积分每步的 influence 给出每种机制的定量贡献

4. **大模型推断：Intervention-based Influence Ratio $Q(w)$**:

    - 功能：在无法重训的大模型上估计 pre-cached vs direct 的影响比
    - 核心思路：**Proposition 5.1** 证明通过在训练后模型上做激活干预（ablation），计算 $Q(w) = \frac{\sum_{j>i+1} d_j^{/i}}{d_{i+1}^{/i}}$（干预后未来位置 KL 散度之和 / 即时位置 KL 散度），可以近似 pre-cached 与 direct influence 的比值
    - 设计动机：重训大模型太昂贵，但只需访问最终 checkpoint 也能通过干预实验推断特征的成因

### 损失函数 / 训练策略
使用标准 NTP 交叉熵损失。Myopic 和 untied 变体通过 stop-gradient 修改梯度传播路径而非损失函数。

## 实验关键数据

### 主实验

OthelloGPT 中 NTP-useful vs NTP-useless 特征的影响力分析（95% 置信区间）：

| 梯度成分 | NTP-useful 特征 | NTP-useless 特征 | 含义 |
|---------|----------------|-----------------|------|
| Direct | [2.85, 12.38] | [-4.69, 2.74] | Direct 只推动 useful 特征 |
| Pre-cached | [-1.99, 0.66] | [0.55, 3.05] | Pre-cached 推动 useless 特征 |
| Shared | [4.80, 12.48] | [2.93, 9.91] | Shared 对两者都有贡献 |
| Combined | [12.14, 19.05] | [4.42, 10.07] | Useful 被学得更好 |

Direct 对 NTP-useless 特征的影响与零无显著差异，而 pre-cached 和 shared 对 NTP-useless 特征有正向贡献——这精确解释了 OthelloGPT "世界模型"的脆弱性来源。

### 消融实验

Toy 任务中不同训练模式对 NTP-useless 特征表征质量的影响：

| 训练模式 | Majority "多数" | Conditioned Majority "前一token" | 说明 |
|---------|----------------|-------------------------------|------|
| 标准训练 | 高探测准确率 | 高探测准确率 | 三种机制全开 |
| Myopic（无 pre-cache） | 下降 | 大幅下降 | 阻断 pre-caching |
| m-Untied（无 sharing） | 下降 | 下降 | 阻断 circuit sharing |
| Myopic + Untied | 最差 | 完全失败 | 无法学习 NTP-useless 特征 |

Conditioned Majority 需要类似 induction head 的两层注意力构造，myopic 训练完全阻止了这种电路的发展。

### 关键发现
- **语法特征主要由 direct 驱动**：小型语言模型中 POS 标签和依赖标签的 pre-cached influence 远低于 direct influence，说明简单语法可以不依赖 pre-caching 学习
- **Pre-caching 对连贯文本生成不可或缺**：myopic 模型的损失（3.29）远高于标准模型（2.53），说明 pre-caching 对复杂语言建模至关重要
- **Gemma 2 的 $Q(w)$ 极端值与形式推理相关**：SAE 特征中 $Q(w)$ 极高或极低的特征集中在代码和形式化领域。Pre-caching 对需要模拟形式计算装置（如 AST 解析）的任务尤为重要
- **Pre-caching ≠ Look-ahead**：$Q(w)$ 与 look-ahead 预测器的特征方向相关性为负——pre-cached 特征对前瞻预测的贡献反而更小。这支持了 "breadcrumbs" 假说：前瞻源于不同位置需要相似特征，而非显式规划

## 亮点与洞察
- **从"静态功能"到"发展过程"的视角转换**：传统可解释性问"这个特征在做什么"，本文问"这个特征是怎么被训练出来的"。这种发展视角与神经科学中功能 vs 发育的区分类似，为理解 LLM 的内部机制提供了全新工具
- **三分解的普适性**：direct/pre-cached/shared 分解基于因果遮蔽 Transformer 的计算图结构，适用于所有 autoregressive model。以此为骨架可以分析任何特征的成因
- **OthelloGPT 世界模型脆弱性的因果解释**：之前只知道"世界模型脆弱"，现在知道"因为 NTP-useless 的棋盘格子只有 pre-cached 和 shared 信号，缺乏 direct 信号"。这是第一个给出统计上显著的因果归因

## 局限与展望
- **归因方法计算成本高**：需要完整重训一次模型并在每步计算三个梯度成分，对大模型不可行
- **$Q(w)$ 指标只是近似**：只在训练后模型的局部区域有效，不代表整个训练轨迹的积分影响
- **Toy 任务与真实 LLM 的差距**：主要实验在小型模型上完成，大模型分析仅限于 $Q(w)$ 这一间接指标
- **可改进方向**：发展更高效的归因方法（如只需中间 checkpoint 而非完整重训）；利用梯度分解发现未知的可解释特征（如只被 pre-cached 更新产生的特征子空间）

## 相关工作与启发
- **vs Wu et al. (2024)**：首先提出 pre-caching 概念和 myopic training。本文在此基础上引入 circuit sharing、建立定量归因框架、发现 pre-caching ≠ look-ahead，是重要的深化和纠正
- **vs Vafa et al. (2025)**：他们发现 OthelloGPT 世界模型对共享相同合法下一步的棋盘脆弱。本文从梯度分解角度给出因果解释：NTP-useless 格子的 direct influence 为零，只靠 pre-cached 和 shared 学习，天然更脆弱
- **vs Bachmann & Nagarajan (2024)**：他们指出 NTP 的缺陷（可能学不到有用特征）。本文从相反角度分析：为什么 NTP 反而能学到"超出预期"的特征

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 提出了理解 Transformer 特征涌现的全新框架（梯度三分解 + 归因方法），视角独特且有深度
- 实验充分度: ⭐⭐⭐⭐ 从 toy 到 Othello 到小型 LM 到 Gemma 2 的渐进验证层次清晰，但大模型分析受限于间接指标
- 写作质量: ⭐⭐⭐⭐⭐ 叙事从直觉到形式化到实验的逻辑链极为流畅，概念定义精准
- 价值: ⭐⭐⭐⭐⭐ 对理解 NTP 训练目标如何产生"超预期"能力有基础性贡献，framework 可广泛应用于 LLM 可解释性研究

<!-- RELATED:START -->

## 相关论文

- [A Law of Data Reconstruction for Random Features (and Beyond)](a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [Identifying and Evaluating Inactive Heads in Pretrained LLMs](identifying_and_evaluating_inactive_heads_in_pretrained_llms.md)
- [MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction](moma_a_modular_deep_learning_framework_for_material_property_prediction.md)
- [TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling](taste_text-aligned_speech_tokenization_and_embedding_for_spoken_language_modelin.md)
- [Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)

<!-- RELATED:END -->
