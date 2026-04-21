---
title: >-
  [论文解读] Verification of the Implicit World Model in a Generative Model via Adversarial Sequences
description: >-
  [ICLR 2026][图像生成][隐式世界模型] 提出对抗序列生成方法验证生成式序列模型的隐式世界模型健全性，在国际象棋领域通过多种对抗策略（IMO/BSO/AD）系统评估，发现所有模型均不健全，但训练方法和数据集选择对健全性有显著影响，且线性棋盘状态探针在大多数模型中无因果作用。
tags:
  - ICLR 2026
  - 图像生成
  - 隐式世界模型
  - 对抗序列生成
  - 国际象棋
  - 健全性验证
  - 线性探针
---

# Verification of the Implicit World Model in a Generative Model via Adversarial Sequences

**会议**: ICLR 2026  
**arXiv**: [2602.05903](https://arxiv.org/abs/2602.05903)  
**代码**: https://github.com/szegedai/world-model-verification  
**领域**: 世界模型/可解释AI  
**关键词**: 隐式世界模型, 对抗序列生成, 国际象棋, 健全性验证, 线性探针

## 一句话总结
提出对抗序列生成方法验证生成式序列模型的隐式世界模型健全性，在国际象棋领域通过多种对抗策略（IMO/BSO/AD）系统评估，发现所有模型均不健全，但训练方法和数据集选择对健全性有显著影响，且线性棋盘状态探针在大多数模型中无因果作用。

## 研究背景与动机

**领域现状**：生成式序列模型（如LLM）在训练中是否能学到底层"世界模型"是关键问题。先前工作（Li et al. 2023, Toshniwal et al. 2022）在Othello和国际象棋上训练模型，通过线性探针声称模型学到了世界状态。

**现有痛点**：(1) 简单指标如next-token准确率可能具有误导性——完全错误的模型也可能有高准确率；(2) Vafa et al.提出的序列级分析需要定义生成语言的概率阈值，引入ad hoc参数；(3) 线性探针解码的状态是否对预测有因果作用尚不清楚。

**核心矛盾**：从样本序列训练能学到健全（sound）的世界模型在理论上是可能的，但实际验证困难。简单测试不够，需要系统性的压力测试方法。

**本文目标**：如何有效验证生成模型的隐式世界模型是否健全？棋盘状态探针是否对预测有因果影响？

**切入角度**：对抗者生成合法序列迫使生成模型预测非法续述，提供健全性的存在性反证。

**核心 idea**：让对抗者下合法棋，迫使被测试的序列模型走出非法棋步，从而量化和分析隐式世界模型的失败模式。

## 方法详解

### 整体框架
对抗者（白方）生成合法走法，被测模型（黑方）预测其下一步。对抗者选择最可能导致模型出错的合法走法：$a_{k+1}^* = \arg\max_{a_{k+1} \in W(a_1..a_k)} f(M, a_1..a_k a_{k+1})$。攻击成功当模型预测非法走法 $m(s^*) \notin W(s^*)$。

### 关键设计

1. **Illegal Move Oracle (IMO)**:

    - $f_{\text{IMO}}(M, s) = \max_{a \notin W(s)} M(a|s)$
    - 选择最大化非法续述概率的合法走法
    - 最直接的攻击策略

2. **Board State Oracle (BSO)**:

    - $f_{\text{BSO}}(M, s) = \mathcal{L}_B(M, s)$
    - 选择最大化棋盘状态预测误差的走法
    - 假设：棋盘状态探针有因果作用→破坏状态应导致非法走法

3. **Adversarial Detours (AD)**:

    - $f_{\text{AD}}(M, s) = -M(a_{k+1}|a_1..a_k)$
    - 选择条件概率最低的走法，将生成引向OOD区域
    - 来自Vafa et al.的方法

4. **训练目标创新**:

    - **概率分布（PD）目标**：预测所有合法走法的均匀分布，而非仅下一个token
    - **联合探针（+JP）目标**：同时训练下一token预测和棋盘状态预测的双头损失
    - 这两个目标旨在更直接地学习世界模型的转移规则

### 模型设置
GPT-2架构（86M参数），12层，768维，12头。6个数据集（随机500K/2M/10M + Millionbase/Stockfish/Lichess），4种训练目标（NT/PD/NT+JP/PD+JP），共24个模型。

## 实验关键数据

### 攻击成功率（随机数据集）

| 攻击策略 | Random-500K NT | Random-10M NT | Random-10M PD |
|----------|----------------|---------------|---------------|
| RM（随机基线） | 0.954 | 0.673 | 0.874 |
| SMM（友好基线） | 0.419 | 0.172 | 0.900 |
| **IMO** | **0.996** | **0.972** | **0.992** |
| BSO | 0.886 | 0.541 | 0.811 |
| AD | 0.946 | 0.516 | 0.902 |

### 关键发现

1. **所有模型均不健全**：IMO攻击几乎100%成功率（最高1.000），即使最好的模型（Random-10M + NT）也有97.2%的攻击成功率

2. **数据集选择影响巨大**：
    - 随机数据集 >> 策略数据集（Millionbase/Stockfish/Lichess）
    - 更多随机训练数据显著提升健全性（500K→10M，RM成功率从95.4%降至67.3%）

3. **PD训练目标双刃剑**：
    - PD的SMM攻击成功率极高（0.900+），说明在非对抗场景下PD反而更易出错
    - 但PD学习分布的能力使其在某些对抗场景中表现不同

4. **BSO vs IMO揭示探针因果性缺失**：
    - BSO攻击成功率显著低于IMO（如Random-10M NT: 0.541 vs 0.972）
    - 说明破坏棋盘状态探针预测并不等同于导致非法走法
    - 结论：线性探针提取的棋盘状态在大多数模型中对next-token预测无因果作用

5. **随机vs策略数据集的差异**：策略数据集训练的模型更容易受AD攻击（OOD区域大），而随机数据集提供更均匀的状态空间覆盖

## 亮点与洞察
- **对抗验证比统计测试更有效**：避免了定义概率阈值的ad hoc选择，直接提供健全性反例
- **探针因果性的重要否定结果**：挑战了"线性探针解码的世界状态→预测使用该状态"的假设
- **数据质量启示**：随机棋局（而非高质量棋局）更有利于学习规则，因为策略数据中合法但策略差的走法被低估
- **方法通用性**：框架可推广到任何世界模型可定义为形式语言的领域

## 局限与展望
- 仅在86M参数的GPT-2上测试，更大模型可能有不同表现
- 国际象棋虽然复杂但规则确定，自然语言的世界模型更模糊
- 对抗搜索是贪心的，理论上可能错过更有效的攻击序列
- 可以探索将此方法应用于代码生成等有形式规则的领域

## 相关工作与启发
- **vs Vafa et al.**：他们需要定义概率阈值来确定生成语言，本方法避免此问题
- **vs Li et al. (OthelloGPT)**：他们声称探针有因果作用，本文在国际象棋上的发现与之矛盾
- **vs Karvonen 2024**：声称国际象棋模型有一致的世界模型，本文的对抗测试表明其实不然

## 评分
- 新颖性: ⭐⭐⭐⭐ 对抗验证框架新颖，探针因果性的否定结果有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 24个模型、5种攻击、6个数据集的大规模系统评估
- 写作质量: ⭐⭐⭐⭐⭐ 形式化定义清晰，实验设计严谨
- 价值: ⭐⭐⭐⭐ 对AI安全和可解释性有重要启示

<!-- RELATED:START -->

## 相关论文

- [GGBall: Graph Generative Model on Poincaré Ball](ggball_graph_generative_model_on_poincaré_ball.md)
- [Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](../../CVPR2025/image_generation/training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)
- [Efficient Adversarial Attacks on High-dimensional Offline Bandits](efficient_adversarial_attacks_on_high-dimensional_offline_bandits.md)
- [Latent Diffusion Model without Variational Autoencoder](latent_diffusion_model_without_variational_autoencoder.md)

<!-- RELATED:END -->
