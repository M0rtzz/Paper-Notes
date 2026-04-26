---
title: >-
  [论文解读] Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness
description: >-
  [ACL 2026][医学图像][多模态临床时序] 提出 OPL-MT-MNAR 框架，通过 MNAR 感知的多模态编码器 + 贝叶斯滤波隐状态 + 离线策略学习，从结构化数据和临床文本的"缺失模式本身携带的信息"中学习 ICU 患者动态表示，实现优于临床医生行为的脓毒症治疗策略（FQE 0.679 vs 0.528）。
tags:
  - ACL 2026
  - 医学图像
  - 多模态临床时序
  - 信息性缺失
  - 离线强化学习
  - 贝叶斯滤波
  - ICU治疗策略
---

# Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness

**会议**: ACL 2026  
**arXiv**: [2604.21235](https://arxiv.org/abs/2604.21235)  
**代码**: [GitHub](https://github.com/CausalMLResearch/OPL-MT-MNAR)  
**领域**: 医学图像  
**关键词**: 多模态临床时序, 信息性缺失, 离线强化学习, 贝叶斯滤波, ICU治疗策略

## 一句话总结
提出 OPL-MT-MNAR 框架，通过 MNAR 感知的多模态编码器 + 贝叶斯滤波隐状态 + 离线策略学习，从结构化数据和临床文本的"缺失模式本身携带的信息"中学习 ICU 患者动态表示，实现优于临床医生行为的脓毒症治疗策略（FQE 0.679 vs 0.528）。

## 研究背景与动机

**领域现状**：电子健康记录（EHR）包含结构化数据（生命体征、实验室检测）和临床文本（护理记录、报告），是学习患者动态表示以支持结果预测和序贯治疗决策的丰富数据源。离线 RL 在 ICU 脓毒症治疗中已有大量工作，但大多将临床观测当作预处理后的完整数据。

**现有痛点**：临床数据有两个关键特征被忽略：(1) **观测过程本身是信息性的**（informative missingness）——病情越重的患者被监测得越频繁，缺失模式反映了潜在健康状态，是 missing-not-at-random (MNAR)；(2) **不同模态的观测模式不同**——生命体征较常规、实验室检查需要下医嘱、文本记录取决于医生的文档行为，这些差异在患者轨迹中随时间演变。

**核心矛盾**：现有方法要么忽略缺失信息，要么只在结构化时序中处理缺失（如 GRU-D），没有在多模态+时序的联合设定下利用缺失模式作为信息信号。特别是临床文本的观测过程（何时写护理记录、记录频率如何变化）被完全忽视。

**本文目标**：构建一个显式利用多模态信息性缺失的患者表示学习框架，支持下游的离线治疗策略优化和结局预测。

**切入角度**：从 ICU 真实数据中发现三个强信号：(a) 病情越重监测越密集；(b) 高 acuity 患者更可能有文本更新；(c) 不同模态的时序可用性演变不同。这些观测模式包含关于患者状态的重要信息。

**核心 idea**：将观测过程（结构化数据的缺失模式 + 文本的文档行为模式）作为显式特征输入，通过 MNAR 感知编码 + 贝叶斯滤波 + 动作条件化隐状态构建患者表示。

## 方法详解

### 整体框架
两阶段框架：**Stage 1** 学习患者状态表示——先通过 MNAR 感知的多模态编码器得到统一表示 $\phi_h$，再通过变分推断的贝叶斯滤波维护隐信念状态 $z_h$，组合为后验患者状态 $s_h = g_\theta(\phi_h, z_h)$。**Stage 2** 用 $s_h$ 进行离线策略优化（IQL）和结局预测。

### 关键设计

1. **MNAR 感知的结构化数据编码（GRU-D 扩展）**:

    - 功能：从不规则采样的结构化观测中提取表示，同时保留缺失模式信息
    - 核心思路：在 GRU-D 基础上增加显式 MNAR 特征（累计观测次数、缺失率、窗口内观测频率），这些特征直接输入 GRU 的门控更新。当某变量长期缺失时，通过学习的衰减因子将其值逐渐回归到经验均值
    - 设计动机：标准 GRU-D 只用时间间隔做衰减，没有利用"被监测频率"本身所携带的 acuity 信息

2. **文档过程因子与稀疏文本融合（Sparse Text Fusion）**:

    - 功能：建模临床文本的观测过程并自适应融合文本与结构化表示
    - 核心思路：引入文档过程因子 $F_h^{doc}$——用 MLP 编码每步的文本存在性、文本时效性、近期文档密度，然后用 GRU 在时间上累积。文本表示通过多头交叉注意力从结构化表示 query 文本嵌入；最终用 $F_h^{doc}$ 控制的门控机制自适应融合两个模态
    - 设计动机：文本可用性本身是内生的——高 acuity 患者文档更频繁。模型需要区分"无文本"、"过时文本"、"密集更新文本"等不同状态，即使底层文本内容相似

3. **动作条件化的隐信念状态（Action-Conditioned Latent Belief）**:

    - 功能：捕获治疗历史对患者轨迹的累积影响
    - 核心思路：通过 VAE 参数化隐状态 $z_{h+1} \sim p_\theta(z_{h+1}|z_h, \phi_h, a_h)$，关键是转移函数条件化于治疗动作。作者证明了 Theorem 1：如果隐状态转移不依赖动作，则策略梯度中当前动作对未来奖励的梯度为零，在终末奖励设定下非终末步完全没有学习信号
    - 设计动机：仅靠观测编码 $\phi_h$ 不足以支持策略优化——因为 $\phi_h$ 是已记录观测的确定性函数。必须通过隐状态传递动作的因果效应

### 损失函数 / 训练策略
三阶段训练：(1) 预训练编码器（重构损失含四项：结构化值、缺失掩码 BCE、文本嵌入、文档过程因子 + 动力学一致性损失 + KL 正则化）；(2) 冻结编码器训练 RL（IQL：双 Q + 期望分位数值函数 + 优势加权行为克隆）；(3) 联合微调。

## 实验关键数据

### 主实验（策略学习 FQE）

| 方法 | 信息 | MIMIC-III | MIMIC-IV | eICU |
|------|------|-----------|----------|------|
| AI Clinician | Model-free | 0.487 | 0.491 | 0.478 |
| DDPG+Clinician | Model-free | 0.529 | 0.538 | 0.524 |
| MedDreamer | Model-based | 0.583 | 0.591 | 0.579 |
| 临床医生行为 | Behavior | 0.528 | 0.521 | 0.534 |
| **OPL-MT-MNAR** | **MNAR+Text** | **0.679** | **0.634** | **0.604** |

### 消融实验（MIMIC-III Building Block Study）

| 配置 | FQE | 相对基线提升 |
|------|-----|------------|
| Baseline (MDP, no MNAR) | 0.507 | — |
| + Semi-MDP | 0.518 | +2.2% |
| + MNAR + DocProcess | **0.679** | **+33.9%** |
| + 全部 | 0.689 | +35.9% |

### 关键发现
- **MNAR 建模是最大贡献者**：显式 MNAR + 文档过程建模贡献了 +33.9% 的提升，远大于 Semi-MDP 的 +2.2%
- **文本对策略学习有实质价值**：结构化 only 为 0.574，加护理记录升至 0.624，全模态 0.679
- **高 acuity 患者获益最大**：高 SOFA(>10) 组临床医生 FQE 仅 0.192，本文方法达到 0.344
- **结局预测 AUROC 0.886**：优于 GRU-D (0.844) 和 MedDreamer (0.867)

## 亮点与洞察
- **"缺失即信号"的理念**非常精彩：不是去填补缺失值，而是将"什么被观测了、什么时候观测的、观测多频繁"直接作为特征。对任何不完整数据领域都有启发
- **动作条件化隐状态的理论必要性证明**（Theorem 1）为方法设计提供了严格的理论支撑
- **文档过程因子**只用观测过程的元信息，不直接用文本内容来调控融合权重，实现了"行为信号"和"内容信号"的解耦

## 局限与展望
- 依赖离线策略评估（FQE），未经前瞻性临床验证
- 动作空间离散化为 9 个、4 小时决策间隔，限制了精细化治疗控制
- 未记录的信息（如口头交流、床旁评估）仍可能造成未观测混淆
- 仅在美国 ICU 数据集上验证，跨国家/医疗体系泛化需进一步验证

## 相关工作与启发
- **vs GRU-D**: GRU-D 只处理结构化时序的时间间隔衰减，本文扩展到多模态 MNAR 并加入累计观测特征
- **vs MedDreamer**: MedDreamer 是 model-based RL，本文通过显式 MNAR 建模在不需要世界模型的情况下实现更高 FQE
- **vs Liang et al. (2025)**: 同一团队前作也建模信息性缺失但没有时序动态，本文加入贝叶斯滤波和动作条件化

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 多模态信息性缺失+文档行为建模+动作条件化隐状态的统一框架，原创性强
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、完整消融、acuity 分层分析、鲁棒性检验
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，motivation 图表有说服力
- 价值: ⭐⭐⭐⭐ 对临床 AI 有实际意义，"缺失即信号"思路有广泛迁移价值

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](../../ICLR2026/medical_imaging/inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](../../NeurIPS2025/medical_imaging/self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[ICLR 2026\] Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](../../ICLR2026/medical_imaging/decentralized_attention_fails_centralized_signals_rethinking_transformers_for_me.md)
- [\[CVPR 2025\] Reanimating Images using Neural Representations of Dynamic Stimuli](../../CVPR2025/medical_imaging/reanimating_images_using_neural_representations_of_dynamic_stimuli.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)

<!-- RELATED:END -->
