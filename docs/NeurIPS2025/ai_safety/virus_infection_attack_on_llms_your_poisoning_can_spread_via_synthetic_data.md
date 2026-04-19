---
title: >-
  [论文解读] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data
description: >-
  [NeurIPS 2025][AI安全][数据投毒] 本文首次系统研究了合成数据在LLM训练中的安全风险，发现现有投毒/后门攻击难以通过合成数据传播，进而提出Virus Infection Attack (VIA)框架，通过劫持点搜索和外壳构造将投毒内容嵌入正常训练样本中，使恶意内容即使在干净查询下也能被模型生成并传播到下游模型。
tags:
  - NeurIPS 2025
  - AI安全
  - 数据投毒
  - 合成数据
  - backdoor attack
  - LLM安全
  - 投毒传播
---

# Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data

**会议**: NeurIPS 2025  
**arXiv**: [2509.23041](https://arxiv.org/abs/2509.23041)  
**代码**: [GitHub](https://github.com/liangzid/VirusInfectionAttack)  
**领域**: AI安全  
**关键词**: 数据投毒, 合成数据, backdoor attack, LLM安全, 投毒传播

## 一句话总结

本文首次系统研究了合成数据在LLM训练中的安全风险，发现现有投毒/后门攻击难以通过合成数据传播，进而提出Virus Infection Attack (VIA)框架，通过劫持点搜索和外壳构造将投毒内容嵌入正常训练样本中，使恶意内容即使在干净查询下也能被模型生成并传播到下游模型。

## 研究背景与动机

合成数据在LLM开发的各个阶段（预训练、SFT、RLHF、蒸馏）被广泛使用，能显著提升模型的推理能力和指令遵循性能。然而，这种"上游模型生成合成数据 → 下游模型训练"的范式存在一个被忽视的安全隐患：**上游模型中的投毒内容是否会通过合成数据传播到下游模型？**

作者首先对主流投毒和后门攻击进行了系统评估，发现了一个关键现象：**现有攻击方法在合成数据中几乎无法传播**。具体来说，即使上游模型的攻击成功率（ASR）很高，合成数据中的感染率（IR）始终低于0.1%。

通过分析超过430万条文本查询，作者揭示了原因：投毒内容和后门触发器通常集中在查询空间中一个极其狭窄的子空间。例如，在三个SFT数据集中，与投毒相关的查询比例分别仅为0.09%、0.23%、0.24%和0.00%。因此，当维护者B用通用查询生成合成数据时，几乎不会触发投毒行为。这种**分布脱耦**是现有攻击无法传播的根本原因。

这引出了一个关键问题：能否设计一种增强投毒传播能力的攻击？如果可以，我们该如何防御？

## 方法详解

### 整体框架

VIA的设计灵感来源于计算机病毒的传播机制。与传统投毒攻击将恶意样本作为独立训练数据不同，VIA将投毒内容（payload）嵌入到正常训练样本中，通过两个关键步骤实现：(1) 劫持点搜索（HPS）找到最优的嵌入位置；(2) 外壳构造（SC）为payload构建保护性外壳以增强隐蔽性。

### 关键设计

1. **感染性投毒的形式化建模**: 给定SFT数据集 $\mathcal{D}=\{(Q_i,R_i)\}$，投毒文本 $P$ 以投毒率 $\rho$ 注入到修改后的响应 $\tilde{R}=R_l||R_c||f_s(P)||R_r$ 中，其中 $R_c$ 是劫持锚点，$f_s$ 是包裹函数。优化目标包含三项：最大化payload在合成数据中的出现率、保持训练目标、抑制干净响应的生成概率。作者推导出一个下界搜索目标（公式2），将感染率分解为三部分的乘积：劫持点的效果 $\frac{1}{\mathbf{P}_\theta(R_r|Q,R_l,R_c)}$、外壳的效果 $\mathbf{P}_{\tilde\theta}(\tilde{P}|Q,R_l,R_c)$、以及对最终生成的影响。

2. **劫持点搜索（HPS）**: 设计评分函数 $S_{R_c} = \log N_{R_c} - \log \max_{R_r} N_{R_r}$，其中 $N_{R_c}$ 和 $N_{R_r}$ 分别是 $R_c$ 和 $R_c||R_r$ 在数据集中的出现频率。核心思路是寻找在训练语料中**高频出现但后续文本不确定性高**的n-gram片段。这类片段是模型生成时的"分叉点"，容易被劫持。作者分析了常用SFT数据集中Top-50高频3-gram的HPS得分分布，发现了多个易受攻击的模式。默认使用3-gram或4-gram作为劫持点。

3. **外壳构造（SC）**: 提出两种策略包裹payload：(a) 固定格式策略——用引号包围payload；(b) LLM生成策略——提示LLM生成两段"粘合文本" $P_{pre}$ 和 $P_{suf}$，使payload $\tilde{P} = P_{pre}||P||P_{suf}$ 能与上下文自然衔接。此外，作者引入语义相似性搜索（SS）策略，对候选训练样本按与payload的语义相关性重排序，进一步提升隐蔽性。

### 训练策略

- **序列化模式**: 对于对话格式的投毒样本，VIA将其序列化为纯文本
- **混合策略（VIA mixup）**: 一半投毒样本直接作为训练数据，另一半通过VIA嵌入，同时保证上游ASR和下游IR
- **投毒率**: 默认设置为2%，使用LLaMA-3 8B作为骨干模型

## 实验关键数据

### 主实验——数据投毒

| 攻击方法 | 场景 | ASR-上游 | IR-合成数据 | 说明 |
|---------|------|---------|-----------|------|
| CoT/Response投毒 | 情感引导 | 100.00% | 0.20% | 传统方法IR极低 |
| VIA (HPS 4-gram) | 情感引导 | 53.65% | **85.64%** | IR大幅提升 |
| CoT/Response投毒 | 知识注入 | 100.00% | 0.22% | 传统方法IR极低 |
| VIA (HPS 4-gram) | 知识注入 | 94.74% | **62.38%** | IR提升300倍 |
| CoT/Response投毒 | 偏见推荐 | 5.26% | 0.06% | 传统方法几乎无效 |
| VIA (HPS 4-gram) | 偏见推荐 | 68.42% | **87.82%** | 突破性提升 |

### 主实验——后门攻击

| 基线 + VIA策略 | 越狱(ASR/IR) | 负面情感(ASR/IR) | 拒绝服务(ASR/IR) |
|---------------|-------------|-----------------|-----------------|
| BadNet | 85.86/0.05 | 99.50/0.15 | 100.00/0.02 |
| BadNet+VIA(mixup) | 77.40/**46.37** | 100.00/**70.82** | 100.00/**78.72** |
| CTBA | 89.90/0.12 | 100.00/0.45 | 99.50/0.40 |
| CTBA+VIA(mixup) | 83.16/**54.10** | 100.00/**67.25** | 99.00/**64.55** |
| Sleeper | 84.85/0.00 | 24.50/0.00 | 54.00/0.00 |
| Sleeper+VIA(mixup) | 84.69/**60.32** | 72.00/**61.32** | 69.50/**66.42** |

### 隐蔽性分析（PPL检测）

| 策略 | 3-gram检测Recall↓ | 5-gram检测Recall↓ | 7-gram检测Recall↓ |
|------|------------------|------------------|------------------|
| 随机插入 | 87.20% | 72.40% | 40.80% |
| HPS | 45.60% | 42.80% | 19.60% |
| HPS + SC | **29.20%** | **30.00%** | **11.60%** |
| HPS + SS + SC | **33.20%** | **27.60%** | **10.00%** |

### 关键发现

- 现有投毒方法在合成数据中的传播率几乎为零（IR < 0.1%），合成数据训练范式对传统攻击具有天然抗性
- VIA将IR从接近0提升到50-87%，证明合成数据并非绝对安全
- VIA (mixup)策略可以同时维持较高的上游ASR和下游IR
- HPS和SC模块能将PPL检测的Recall从87%降到29%，显著提升隐蔽性

## 亮点与洞察

- **独特的研究视角**: 首次从安全角度审视合成数据在LLM开发中的角色，填补了一个重要的研究空白
- **病毒隐喻的巧妙设计**: 将计算机病毒的传播机制（payload + shell + injection point）映射到LLM投毒场景，设计理念自然且有效
- **理论与实验的双重验证**: 从分布分析角度解释了现有攻击失败的原因，并给出了可操作的攻击增强方案

## 局限与展望

- VIA在提升IR的同时会降低上游ASR，虽然mixup策略有一定缓解但并未根本解决
- 仅考虑了LLaMA-3 8B一个骨干模型，在更大规模模型上的表现未知
- 防御策略（PPL过滤）仍较为初步，缺乏更系统的防御框架
- 解耦优化策略可能不是全局最优解

## 相关工作与启发

- 与数据投毒（Gu et al., 2017; Xu et al., 2024）和后门攻击（BackdoorLLM）的经典工作形成对比
- 为合成数据的安全使用提供了新的审计视角，对DeepSeek等依赖大量合成数据的模型训练方案有警示意义
- 启发了"供应链攻击"思路在AI领域的推广

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次系统研究合成数据安全风险，病毒传播隐喻设计精巧
- **实验充分度**: ⭐⭐⭐⭐ 6个攻击场景×10个基线方法，消融实验充分，但仅用单一骨干模型
- **写作质量**: ⭐⭐⭐⭐ 论文结构清晰，研究问题驱动的叙事方式很好
- **价值**: ⭐⭐⭐⭐⭐ 对AI安全社区和合成数据使用者有重要的警示价值

<!-- RELATED:START -->

## 相关论文

- [Provable Watermarking for Data Poisoning Attacks](provable_watermarking_for_data_poisoning_attacks.md)
- [ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](../../ICCV2025/ai_safety/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [DeepPersona: A Generative Engine for Scaling Deep Synthetic Personas](deeppersona_a_generative_engine_for_scaling_deep_synthetic_personas.md)
- [MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)

<!-- RELATED:END -->
