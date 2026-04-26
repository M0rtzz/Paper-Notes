---
title: >-
  [论文解读] Provable Watermarking for Data Poisoning Attacks
description: >-
  [NeurIPS 2025][AI安全][watermarking] 本文提出两种可证明的水印方案（后投毒水印和投毒并行水印），为数据投毒攻击提供透明性声明机制，理论证明在特定水印长度条件下可同时保证水印可检测性和投毒有效性。
tags:
  - NeurIPS 2025
  - AI安全
  - watermarking
  - data poisoning
  - backdoor attack
  - availability attack
  - provable guarantee
---

# Provable Watermarking for Data Poisoning Attacks

**会议**: NeurIPS 2025  
**arXiv**: [2510.09210](https://arxiv.org/abs/2510.09210)  
**代码**: 无  
**领域**: AI安全  
**关键词**: watermarking, data poisoning, backdoor attack, availability attack, provable guarantee

## 一句话总结

本文提出两种可证明的水印方案（后投毒水印和投毒并行水印），为数据投毒攻击提供透明性声明机制，理论证明在特定水印长度条件下可同时保证水印可检测性和投毒有效性。

## 背景与动机

1. **数据投毒的双面性**：数据投毒攻击传统上被视为安全威胁，但近年来越来越多地被用于正当目的，如数据集所有权验证（后门攻击）和防止未授权数据使用（不可学习样本），甚至艺术家版权保护（NightShade/Glaze）。

2. **误用风险**：合法授权用户可能无意中使用了被投毒的数据，导致误解和冲突。投毒生成者有责任向授权用户透明地公开数据中可能存在的投毒痕迹。

3. **现有检测方法的不足**：现有的后门和可用性攻击检测方法种类繁多且互不统一，难以作为单一框架分发给授权用户。更关键的是，这些方法多依赖启发式训练算法，缺乏可证明的投毒声明机制。

4. **水印技术的适用性**：水印技术已广泛用于版权保护和AI生成内容检测，是一种有前景的方案，可以让投毒者以可证明的方式声明投毒的存在。

5. **两种应用场景**：实际中存在两种需求——投毒生成者委托第三方实体创建水印（后投毒水印），以及投毒生成者自行同时生成水印和投毒（投毒并行水印）。

6. **理论空白**：尽管数据投毒和水印各自有丰富研究，但将水印方案应用于数据投毒攻击的场景尚属首次，缺乏相关理论保证。

## 方法详解

### 整体框架

论文将数据投毒定义为映射 $\delta^p: [0,1]^d \to [-\epsilon_p, \epsilon_p]^d$，水印定义为映射 $\delta^w: [0,1]^d \to [-\epsilon_w, \epsilon_w]^q$，其中 $q \leq d$ 为水印长度。检测机制基于密钥 $\zeta$ 计算内积 $\zeta^T x$ 来判断数据是否被水印标记。

### 后投毒水印（Post-Poisoning Watermarking）

- **场景**：第三方实体为已投毒数据集生成水印
- **总扰动**：$\delta_x = \delta_x^p + \delta_x^w$，扰动预算为 $\epsilon_w + \epsilon_p$
- **样本级理论**（Theorem 4.1）：当水印长度 $q > \frac{1}{\epsilon_w}\sqrt{2d\log\frac{1}{\omega}}$，即 $q = \Omega(\sqrt{d}/\epsilon_w)$ 时，可以高概率区分水印数据和干净数据
- **通用版本理论**（Theorem 4.9）：当 $q = \Theta(\sqrt{d}/\epsilon_w)$ 且样本量 $N = \Omega(d)$ 时，对大多数样本有效并可推广到整个分布

### 投毒并行水印（Poisoning-Concurrent Watermarking）

- **场景**：水印生成者同时也是投毒生成者，可控制维度分配
- **维度分离**：水印和投毒使用不同维度，投毒维度 $\mathcal{P} = [d] \setminus \mathcal{W}$
- **总扰动**：$\max\{\epsilon_w, \epsilon_p\}$，小于后投毒情况的 $\epsilon_w + \epsilon_p$
- **样本级理论**（Theorem 4.3）：水印长度 $q = \Omega(1/\epsilon_w^2)$ 即可保证检测性，比后投毒方案所需更短
- **通用版本理论**（Theorem 4.13）：$q = \Theta(1/\epsilon_w^2)$ 即可对大多数样本有效

### 投毒有效性保证

- **后投毒情况**（Theorem 5.2）：基于 Xavier 归一化的 $L$ 层前馈网络，当 $d$ 和 $N$ 足够大时，水印对投毒效果的影响可被限制在可忽略的误差项内，无需额外的 $q$ 约束
- **并行情况**（Theorem 5.6）：需额外条件 $q = O(\sqrt{d}/\epsilon_p)$，否则水印会主导并显著降低投毒效果
- **最终水印长度范围**：后投毒取 $\Theta(\sqrt{d}/\epsilon_w)$；并行取 $\Theta(1/\epsilon_w^2)$ 到 $O(\sqrt{d}/\epsilon_p)$

### 威胁模型实例

论文以自动驾驶数据集为例构建了完整的部署场景：数据所有者 Alice 发布带投毒+水印的数据集，授权用户 Bob 通过安全通道获取密钥后可验证身份并移除投毒，恶意用户 Chad 无法成功训练模型也无法去除水印。还设计了 HMAC 密钥管理方案防止伪造攻击。

## 实验结果

### 实验设置

- **攻击方法**：后门攻击（Narcissus、AdvSc）+ 可用性攻击（UE、AP）
- **数据集**：CIFAR-10、CIFAR-100、Tiny-ImageNet
- **模型**：ResNet-18、ResNet-50、VGG-19、DenseNet121、WRN34-10、MobileNet v2
- **水印长度**：0~3000，水印/投毒预算分别为 16/255（后门）和 8/255（可用性）

### 后门攻击水印效果（Table 1 — ResNet-18, CIFAR-10）

| 水印长度 $q$ | Narcissus 后投毒 Acc/ASR/AUROC | Narcissus 并行 Acc/ASR/AUROC | AdvSc 后投毒 Acc/ASR/AUROC | AdvSc 并行 Acc/ASR/AUROC |
|:---:|:---:|:---:|:---:|:---:|
| 0 (Baseline) | 94.69/95.04/- | 94.69/95.04/- | 92.80/95.53/- | 92.80/95.53/- |
| 500 | 94.95/93.11/0.9509 | 94.70/95.03/0.9968 | 93.18/97.43/0.9218 | 92.89/95.79/0.9986 |
| 1000 | 94.40/92.43/0.9974 | 94.32/92.03/0.9992 | 93.05/94.41/0.9809 | 93.38/84.39/0.9995 |
| 2000 | 94.55/90.37/1.0000 | 94.89/22.46/1.0000 | 93.40/79.97/0.9994 | 92.38/30.05/1.0000 |
| 3000 | 94.93/90.02/1.0000 | 94.72/9.75/1.0000 | 93.10/74.82/1.0000 | 93.04/9.97/1.0000 |

### 可用性攻击水印效果（Table 2 — ResNet-18, CIFAR-10）

| 水印长度 $q$ | UE 后投毒 Acc↓/AUROC↑ | UE 并行 Acc↓/AUROC↑ | AP 后投毒 Acc↓/AUROC↑ | AP 并行 Acc↓/AUROC↑ |
|:---:|:---:|:---:|:---:|:---:|
| 0 (Baseline) | 10.79/- | 10.79/- | 8.53/- | 8.53/- |
| 500 | 11.71/0.7810 | 10.02/0.9930 | 8.71/0.8623 | 15.84/0.8931 |
| 1000 | 11.37/0.9499 | 9.42/0.9991 | 10.58/0.9742 | 21.87/0.9949 |
| 2000 | 9.06/0.9992 | 10.03/1.0000 | 10.48/0.9987 | 38.62/1.0000 |
| 3000 | 9.99/1.0000 | 91.79/1.0000 | 13.52/1.0000 | 93.40/1.0000 |

**关键发现**：
- 后投毒水印：投毒效果在所有 $q$ 下均保持良好（ASR 仅轻微下降），验证了 Theorem 5.2
- 并行水印：$q$ 过大后投毒效果急剧下降（如 $q=2000$ 时 ASR 降至 ~22-30%），验证了 $q = O(\sqrt{d}/\epsilon_p)$ 的必要性
- 并行水印的检测性能（AUROC）在同等 $q$ 下一致优于后投毒水印，印证了理论中 $\Omega(1/\epsilon_w^2) < \Omega(\sqrt{d}/\epsilon_w)$ 的结论

## 亮点

- **首创性**：首次将水印技术引入数据投毒攻击领域，为"投毒为善"提供透明性保障
- **理论严谨**：从样本级到通用版本再到分布级，提供了完整的理论保证链条，明确给出了保证水印检测性和投毒有效性的精确条件
- **两种方案互补**：后投毒方案适合第三方信任场景（更大水印长度更优），并行方案适合自主投毒场景（需平衡水印与投毒长度）
- **完整部署方案**：不仅给出理论和算法，还设计了 HMAC 密钥管理等工程化安全措施

## 局限性

- **充分条件而非必要条件**：理论仅给出水印长度的充分条件，必要条件仍是开放问题
- **线性检测机制**：检测基于简单的内积阈值，可能不够鲁棒
- **并行水印的权衡困难**：$q$ 的有效范围 $[\Theta(1/\epsilon_w^2), O(\sqrt{d}/\epsilon_p)]$ 在实际中可能较窄
- **对抗鲁棒性有限**：虽然附录讨论了数据增强、图像再生、DP 噪声和扩散净化等防御手段，但系统性的鲁棒性分析不够深入

## 相关工作对比

| 维度 | 本文 | Li et al. (2023) 后门数据集水印 |
|:---|:---|:---|
| 水印与投毒关系 | 明确分离水印和投毒，提供两种方案 | 将后门攻击本身作为水印信号 |
| 理论保证 | 提供可证明的检测性和投毒有效性条件 | 缺乏形式化理论保证 |
| 攻击覆盖 | 后门攻击 + 可用性攻击统一框架 | 仅针对后门攻击 |
| 适用性 | 通用，不依赖特定攻击算法 | 与特定后门攻击耦合 |

| 维度 | 本文 | NightShade/Glaze |
|:---|:---|:---|
| 目标 | 声明投毒存在、提供透明性 | 保护艺术家版权不被生成模型学习 |
| 水印机制 | 独立于投毒的可检测水印 | 无显式水印机制 |
| 理论框架 | 严格的概率论证 | 经验性方法 |
| 检测能力 | 授权用户可验证投毒 | 无主动检测机制 |

## 评分

- ⭐⭐⭐⭐ 新颖性：首次将水印引入数据投毒场景，问题定义新颖且实际
- ⭐⭐⭐⭐ 技术深度：从样本级到分布级的完整理论链条，证明严谨
- ⭐⭐⭐⭐ 实验充分度：覆盖4种攻击、多种模型和数据集，消融实验充分
- ⭐⭐⭐ 实用性：理论 promising 但水印长度范围限制和鲁棒性问题制约实际部署

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training](toxictextclip_text-based_poisoning_and_backdoor_attacks_on_clip_pre-training.md)
- [\[NeurIPS 2025\] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)
- [\[NeurIPS 2025\] FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning](fedfact_a_provable_framework_for_controllable_group-fairness_calibration_in_fede.md)
- [\[AAAI 2026\] Perturb Your Data: Paraphrase-Guided Training Data Watermarking](../../AAAI2026/ai_safety/perturb_your_data_paraphrase-guided_training_data_watermarking.md)
- [\[ACL 2025\] Robust Data Watermarking in Language Models by Injecting Fictitious Knowledge](../../ACL2025/ai_safety/robust_data_watermarking_in_language_models_by_injecting_fictitious_knowledge.md)

<!-- RELATED:END -->
