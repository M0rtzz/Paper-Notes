---
title: >-
  [论文解读] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models
description: >-
  [ACL 2025][parallel data] 系统研究在 decoder-only LLM 训练中加入平行数据对多语言能力的影响：平行数据放在训练末期效果最好且显著优于等量单语数据；LLM 无法自动泛化到训练方向的反向翻译（reversal curse）。
tags:
  - ACL 2025
  - parallel data
  - multilingual LLM
  - continual pre-training
  - translation
  - cross-lingual alignment
---

# Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models

**会议**: ACL 2025  
**arXiv**: [2506.13044](https://arxiv.org/abs/2506.13044)  
**代码**: [nusnlp/just-go-parallel](https://github.com/nusnlp/just-go-parallel)  
**领域**: Multilingual / MT  
**关键词**: parallel data, multilingual LLM, continual pre-training, translation, cross-lingual alignment

## 一句话总结

系统研究在 decoder-only LLM 训练中加入平行数据对多语言能力的影响：平行数据放在训练末期效果最好且显著优于等量单语数据；LLM 无法自动泛化到训练方向的反向翻译（reversal curse）。

---

## 研究背景与动机

- **领域现状**：LLM 即使未显式使用平行数据也展现出翻译能力，被归因于训练语料中偶然出现的双语信号（incidental bilingual signals）。部分主流多语言 LLM（如 BLOOM）主动放弃使用平行数据。
- **现有痛点**：平行数据对 encoder 模型（如 XLM）的跨语言迁移已被证明有效，但 decoder-only LLM 是否也可从中受益？若有益，最优引入时机（开头 / 分散 / 末尾）和数据格式（双向 / 单向）尚不清楚。
- **核心矛盾**：decoder LLM 通常忽视平行数据——这是否是一个代价高昂的错误决策？
- **本文切入点**：在总训练量固定的前提下，设计 7 种控制实验（无平行 / 单语替代 / 非相邻 / 前置 / 分散 / 末置全向 / 末置单向），系统回答"平行数据有没有用、怎么用最好"。
- **核心 idea**：平行数据以"第二阶段训练"形式放在末期对翻译和多语言常识推理的提升最大；首置会导致灾难性遗忘，而 LLM 无法自动泛化到反向翻译方向。

---

## 方法详解

### 整体框架

固定总训练量（~167B tokens）→ 在不同训练位置插入等量平行数据（替换等量末尾非平行数据）→ 对比翻译（BLEU）和多语言常识推理（Accuracy）性能。基础模型为 TinyLlama 1.1B，非平行数据使用 SlimPajama 子集，平行数据涵盖中英（33.9M 句对，~2.8B tokens）和印尼英（54.1M 句对，~2.1B tokens），格式为 "{source lang}: {src}\n{target lang}: {tgt}"。

### 关键设计

1. **7 种平行数据引入策略**
    - **No Parallel**：无平行数据（基线），模拟典型英文中心 LLM
    - **Multilingual**：加入等量目标语言**单语**数据（对照组），评估单纯增加语言暴露的效果
    - **Parallel Non-Adjacent**：平行句对存在但**不相邻**排列（源句与随机英文句拼接），隔离"语义对齐"的作用
    - **Parallel First / Distributed / Last (all) / Last (uni)**：分别将相邻平行句对放在训练开始 / 均匀分散 / 末尾（双向）/ 末尾（单向），隔离位置和翻译方向的影响
    - 设计动机：控制变量法，逐一隔离平行性、相邻性、位置、方向四个因素

2. **严格的训练量控制**
    - 加入平行数据时，从训练集末尾移除等量非平行数据，保证总 token 数和数据顺序一致
    - 每种设置均在相同 GPU 资源下训练至 ~167B tokens，消除数据量差异的干扰
    - 设计动机：使不同策略的差异完全归因于"平行 vs. 非平行"及"引入位置"

3. **多维度评估体系**
    - **翻译**：WMT-2023 中英测试集 + Flores-200 印尼英 devtest，zero-shot & 5-shot
    - **常识推理**：英文 8 项 benchmark（ARC、HellaSwag、BoolQ 等）、中文 4 项（XWinograd、XStoryCloze 等）、印尼文 2 项
    - **消融**：平行数据质量过滤（CometKiwi-2022）、反向翻译泛化、灾难性遗忘曲线
    - 设计动机：不仅关注翻译能力，还考察平行数据对非翻译任务的影响

---

## 实验关键数据

### 主实验 — 翻译性能（BLEU）

| 设置 | EN→ID | ID→EN | EN→ZH | ZH→EN | 说明 |
|------|-------|-------|-------|-------|------|
| No Parallel | 2.49 | 1.52 | 0.80 | 1.30 | 基线 |
| Multilingual | 2.38 | 5.92 | 0.81 | 3.72 | 单语有帮助但非常有限 |
| Parallel Non-Adjacent | 1.98 | 14.69 | 1.01 | 4.50 | 不相邻 → 效果差 |
| Parallel First | 7.42 | 5.57 | 9.64 | 2.71 | **灾难性遗忘严重** |
| Parallel Distributed | 21.95 | 27.48 | 12.08 | 7.40 | 分散效果中等 |
| **Parallel Last (all)** | **35.91** | **35.36** | **9.62** | **10.73** | **末置双向最佳** |
| **Parallel Last (uni)** | **44.19** | **41.91** | **28.51** | **16.10** | **单向更高** |
| BLOOM 1.1B | 2.19 | 18.39 | 2.27 | 4.58 | 对比参考 |
| NLLB 1.3B | 44.64 | 43.06 | 27.58 | 19.25 | 专用翻译模型 |

> zero-shot 结果。Parallel Last (uni) 接近专用翻译模型 NLLB，但丧失其他方向翻译能力。

### 反向翻译泛化实验

| 训练方向 | EN→ID | ID→EN | EN→ZH | ZH→EN |
|---------|-------|-------|-------|-------|
| All directions | 35.91 | 35.36 | 9.62 | 10.73 |
| EN→ID only | **44.19** | 0.07 | 0.77 | 0.21 |
| ID→EN only | 0.02 | **41.91** | 0.25 | 0.03 |
| EN→ZH only | 0.09 | 0.59 | **28.51** | 0.01 |
| ZH→EN only | 0.00 | 2.73 | 0.13 | **16.10** |

> 单向训练模型完全无法翻译反方向，证实 reversal curse 在翻译任务中同样成立。

### 消融 — 关键发现

| 发现 | 证据 |
|------|------|
| 平行 >> 单语 | Parallel Last (all) vs. Multilingual：EN→ID +33.5 BLEU |
| 末置 >> 首置 | Parallel Last vs. Parallel First：EN→ID +28.5 BLEU |
| 相邻 >> 非相邻 | Parallel Distributed vs. Non-Adjacent：EN→ID +20.0 BLEU |
| 无反向泛化 | 单向训练反方向 BLEU ≈ 0 |
| 数据质量过滤 | 对中英 zero-shot 提升显著（+14.6 BLEU），对 few-shot 无明显帮助 |

---

## 关键发现

- **末置最优**：平行数据作为"第二阶段训练"放在末期效果最好，可能因为模型先在大规模英文数据上建立了稳固的语言理解基础，再通过平行数据高效建立跨语言映射
- **灾难性遗忘**：平行数据前置会在后续训练中完全丧失翻译能力——BLEU 从 ~20 降至接近 0
- **反向翻译不泛化**：训练 EN→ZH 不会使 ZH→EN 也变好（BLEU ≈ 0），挑战了"LLM 内部有语言无关表示"的假设，与 reversal curse 一致
- **平行数据的额外价值**：相比等量单语数据，平行数据提供的对齐信号对翻译和多语言推理都有额外收益
- **训练位置 > 训练量**：Distributed 和 Last 使用相同数据量，但 Last 翻译性能显著更高

---

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-----------|------|
| 实验设计 | 9 | 7 种控制实验极其严谨，变量隔离充分 |
| 新颖性 | 6 | 问题不新，但系统性研究和多个反直觉发现有价值 |
| 实用性 | 8 | "末置平行数据"策略简单、直接可用，已被 Tower 等采纳 |
| 写作质量 | 8 | 实验描述清晰，表格和图表丰富，结论明确 |

---

## 亮点与不足

**亮点**：

- 控制实验设计堪称教科书级别——固定数据量、固定顺序，逐一变化单因素
- "反向翻译不泛化"是一个重要的负面发现，对理解 LLM 的跨语言表示机制有启发
- 末置策略已被 Tower、Pangea 等后续工作验证和采纳

**不足**：

- 仅在 1.1B 参数的 TinyLlama 上验证，结论能否推广到 7B+ 模型未知
- 语言覆盖有限（仅中文和印尼文），缺乏低资源语言和形态丰富语言的实验
- 未探索平行数据与 instruction tuning 的交互效应

## 相关工作与启发
- **vs CLO (Lee et al.)**：CLO 用 DPO 做跨语言迁移，本文用平行数据继续预训练——两者可互补
- **vs XLM/mBART**：encoder/encoder-decoder 模型的平行数据策略已成熟，本文将其扩展到 decoder-only

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统性对比研究+多个重要发现
- 实验充分度: ⭐⭐⭐⭐⭐ 7种设置×多语言×多任务，极其充分
- 写作质量: ⭐⭐⭐⭐⭐ 实验设计严谨，结论清晰
- 价值: ⭐⭐⭐⭐⭐ 对多语言LLM训练有直接指导价值

<!-- RELATED:START -->

## 相关论文

- [Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs](multilingual_llm_english_accent.md)
- [ShifCon: Enhancing Non-Dominant Language Capabilities with a Shift-based Multilingual Contrastive Framework](shifcon_nondominant_language.md)
- [Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)
- [Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](marco_bench_multilingual_if.md)

<!-- RELATED:END -->
