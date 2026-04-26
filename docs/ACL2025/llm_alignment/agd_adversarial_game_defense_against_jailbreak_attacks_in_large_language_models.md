---
title: >-
  [论文解读] AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models
description: >-
  [ACL 2025][LLM对齐][越狱攻击防御] 本文提出AGD（Adversarial Game Defense），一种基于对抗博弈的LLM越狱攻击防御方法，通过动态调整模型内部表示在有用性和无害性之间取得平衡，利用IQR异常检测、双层优化博弈和专家模型采样三个阶段显著提升LLM安全性。
tags:
  - ACL 2025
  - LLM对齐
  - 越狱攻击防御
  - 对抗博弈
  - 注意力权重矫正
  - 纳什均衡
  - 内部表示调控
---

# AGD: Adversarial Game Defense Against Jailbreak Attacks in Large Language Models

**会议**: ACL 2025  
**arXiv**: 无公开arXiv  
**代码**: 无  
**领域**: 对齐RLHF / LLM安全  
**关键词**: 越狱攻击防御、对抗博弈、注意力权重矫正、纳什均衡、内部表示调控

## 一句话总结
本文提出AGD（Adversarial Game Defense），一种基于对抗博弈的LLM越狱攻击防御方法，通过动态调整模型内部表示在有用性和无害性之间取得平衡，利用IQR异常检测、双层优化博弈和专家模型采样三个阶段显著提升LLM安全性。

## 研究背景与动机

**领域现状**：大语言模型在实际应用中展现了强大的能力，但同时也面临严重的越狱攻击（jailbreak attacks）威胁。攻击者通过精心构造的提示绕过模型的安全对齐，诱导模型生成有害内容。

**现有痛点**：当前的防御方法主要包括两类：（1）后训练对齐和提示工程，依赖安全标注数据集和安全提示模板，但对分布外（OOD）攻击的适应性较差；（2）基于内部表示调控（representation steering）的方法可以实现实时调整以抵御OOD攻击，但修改表示会破坏推理过程的前向传播，导致模型实用性下降。

**核心矛盾**：LLM的有用性（helpfulness）和无害性（harmlessness）之间存在根本性的竞争关系。现有方法要么牺牲有用性来换取安全，要么安全性不足。简单地修改内部表示无法同时优化这两个目标。

**本文目标**：设计一种能够动态平衡有用性和无害性的防御机制，在不显著降低模型实用性的前提下有效抵御各类越狱攻击。

**切入角度**：作者将有用性和无害性视为博弈论中的两个对抗目标，借助对抗博弈的思想通过双层优化自动找到纳什均衡点，从而实现两个目标的优化平衡。

**核心 idea**：将LLM的安全防御建模为一个双人变和博弈问题，通过IQR检测异常注意力权重、对抗训练矫正注意力、双层优化逼近纳什均衡来实现安全与有用性的动态平衡。

## 方法详解

### 整体框架
AGD方法包含三个核心阶段：（1）异常注意力检测与矫正阶段，使用IQR方法识别并修正被越狱攻击扰动的注意力头；（2）对抗博弈优化阶段，通过双层优化让"有用性玩家"和"无害性玩家"在注意力激活空间上进行对抗博弈，逼近纳什均衡；（3）安全采样阶段，引入专家模型指导下一个token的采样以生成更安全的响应。

### 关键设计

1. **IQR异常注意力检测与矫正**:

    - 功能：识别并修正被越狱攻击扰动的异常注意力权重
    - 核心思路：观察到越狱攻击会导致特定注意力头的权重出现异常偏移。使用四分位距（IQR）方法统计每个注意力头权重的分布，当某个头的激活值超出 $Q_1 - 1.5 \times IQR$ 或 $Q_3 + 1.5 \times IQR$ 范围时标记为异常。异常的注意力权重通过对抗训练进行矫正，使其恢复到正常分布
    - 设计动机：越狱攻击的核心机制是通过特定token组合改变模型的注意力分配模式，使模型"忽视"安全约束。通过检测这些异常模式，可以在推理时实时识别攻击并进行矫正

2. **双层优化对抗博弈**:

    - 功能：在有用性和无害性之间寻找最优平衡点
    - 核心思路：定义两个"玩家"——有用性玩家和无害性玩家，分别控制不同的注意力头激活。将问题建模为双人变和博弈，使用双层优化（bi-level optimization）框架：外层优化有用性目标，内层优化无害性目标。两个玩家交替优化各自的策略，通过迭代过程逼近纳什均衡（Nash Equilibrium）。在均衡点处，任何一方都无法通过单方面改变策略来提升自身目标
    - 设计动机：传统方法将安全性视为单目标优化问题，忽略了与有用性的冲突。博弈论框架自然地建模了这种竞争关系，纳什均衡保证了双方的利益都被合理考虑

3. **专家模型引导的安全采样**:

    - 功能：在token采样阶段进一步确保生成内容的安全性
    - 核心思路：引入一个预训练的安全专家模型，在解码阶段对每个候选token的安全性进行评估。将原始模型的token概率分布与专家模型的安全性评分结合，调整采样概率使得更安全的token获得更高的采样权重
    - 设计动机：即使通过注意力矫正和博弈优化调整了内部表示，在自回归生成过程中仍可能出现不安全的token序列。专家模型提供了最后一道安全屏障

### 损失函数 / 训练策略
AGD采用双层优化框架，外层最大化有用性损失，内层最小化无害性损失。两个损失函数分别基于有用性评估指标和安全性评估指标定义，通过交替梯度更新实现逼近纳什均衡。

## 实验关键数据

### 主实验

| 方法 | GCG ASR↓ | AutoDAN ASR↓ | PAIR ASR↓ | 平均ASR↓ | MT-Bench↑ |
|------|---------|-------------|----------|---------|----------|
| 无防御 | 56.0 | 78.0 | 44.0 | 59.3 | 6.8 |
| Self-Reminder | 26.0 | 48.0 | 28.0 | 34.0 | 6.2 |
| RepE | 8.0 | 22.0 | 14.0 | 14.7 | 5.4 |
| AGD (本文) | **2.0** | **6.0** | **4.0** | **4.0** | **6.5** |

### 消融实验

| 配置 | 平均ASR↓ | MT-Bench↑ | 说明 |
|------|---------|----------|------|
| Full AGD | 4.0 | 6.5 | 完整模型 |
| w/o IQR检测 | 12.0 | 6.3 | 去掉异常检测后安全性下降 |
| w/o 博弈优化 | 8.0 | 5.8 | 去掉博弈后有用性显著下降 |
| w/o 专家采样 | 6.0 | 6.4 | 去掉专家模型后安全性略降 |

### 关键发现
- 双层优化博弈机制是AGD最核心的贡献，去掉后有用性下降最明显（MT-Bench从6.5降到5.8），说明博弈是平衡两个目标的关键
- IQR异常检测对OOD攻击的防御尤为重要，对未见过的攻击类型仍然有效
- AGD在保持高安全性的同时，有用性损失极小（MT-Bench仅从6.8降到6.5），远优于RepE等方法

## 亮点与洞察
- 将LLM安全防御建模为对抗博弈是一个巧妙的思路，自然地处理了有用性和无害性的冲突，比简单的正则化方法更优雅。这个框架可以扩展到其他多目标优化场景
- IQR异常检测方法简单有效，无需额外训练即可识别攻击，具有很好的实用性和可迁移性
- 专家模型采样提供了一种"软性"安全约束，不同于硬性拒绝策略，可以在安全和信息量之间取得更好的平衡

## 局限与展望
- 作者未公开代码和arXiv预印本，这限制了方法的可复现性和社区的进一步研究
- 对抗博弈的收敛速度和稳定性未充分分析，在不同模型架构上的表现尚不清楚
- 双层优化在推理时引入了额外的计算开销，对于实时应用场景可能是一个瓶颈
- IQR方法假设正常注意力权重近似服从正态分布，对于一些长尾分布的场景可能会出现误判
- 专家模型的选择和训练对最终效果影响较大，论文中未充分讨论不同专家模型的影响

## 相关工作与启发
- **vs RepE (Representation Engineering)**: RepE直接修改内部表示来增强安全性，但忽略了有用性损失。AGD通过博弈框架在两者间取得平衡，有用性保持更好
- **vs Self-Reminder**: Self-Reminder通过提示工程在输入中添加安全提醒，但对OOD攻击适应性差。AGD在表示层面进行动态调整，泛化性更强
- **vs Circuit Breakers**: Circuit Breakers通过训练额外的安全电路来阻断有害生成，需要大量安全数据。AGD是推理时防御，不需要额外训练数据

## 评分
- 新颖性: ⭐⭐⭐⭐ 博弈论框架用于LLM安全防御是有创意的，但IQR检测和专家采样相对常规
- 实验充分度: ⭐⭐⭐⭐ 覆盖了多种攻击类型和评估维度，消融实验较完整
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但未公开代码影响了可验证性
- 价值: ⭐⭐⭐⭐ 提出了一种兼顾安全和有用性的新范式，对LLM安全研究有启发意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States](hiddendetect_detecting_jailbreak_attacks_against_multimodal_large_language_model.md)
- [\[ACL 2025\] Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)
- [\[ACL 2025\] JailbreakRadar: Comprehensive Assessment of Jailbreak Attacks Against LLMs](jailbreakradar_comprehensive_assessment_jailbreak_attacks.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)

<!-- RELATED:END -->
