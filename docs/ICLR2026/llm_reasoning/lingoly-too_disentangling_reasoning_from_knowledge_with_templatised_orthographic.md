---
title: >-
  [论文解读] LingOly-TOO: Disentangling Reasoning from Knowledge with Templatised Orthographic Obfuscation
description: >-
  [ICLR 2026][LLM推理][reasoning benchmark] 提出LingOly-TOO基准，通过专家设计的正字法置换（grapheme-level permutation）对语言学奥赛题进行混淆，保留推理逻辑但消除知识/记忆捷径，将15个前沿模型的最高分从0.59降至0.48，系统量化了LLM推理能力被知识效应高估的程度。
tags:
  - ICLR 2026
  - LLM推理
  - reasoning benchmark
  - orthographic obfuscation
  - linguistics olympiad
  - knowledge contamination
  - LLM evaluation
---

# LingOly-TOO: Disentangling Reasoning from Knowledge with Templatised Orthographic Obfuscation

**会议**: ICLR 2026  
**arXiv**: [2503.02972](https://arxiv.org/abs/2503.02972)  
**代码**: [GitHub](https://github.com/jkhouja/LingOly-TOO)  
**领域**: LLM推理 / 评测基准  
**关键词**: reasoning benchmark, orthographic obfuscation, linguistics olympiad, knowledge contamination, LLM evaluation

## 一句话总结

提出LingOly-TOO基准，通过专家设计的正字法置换（grapheme-level permutation）对语言学奥赛题进行混淆，保留推理逻辑但消除知识/记忆捷径，将15个前沿模型的最高分从0.59降至0.48，系统量化了LLM推理能力被知识效应高估的程度。

## 研究背景与动机

**领域现状**：LLM在各类推理基准上的分数快速上升，但越来越多证据表明分数膨胀源于训练集污染和知识记忆捷径，而非真正的推理能力提升。MATH/GSM8K等基准迅速饱和。

**现有痛点**：

1. 训练数据规模增大使训练/测试集边界模糊，评测偏差加剧

2. 现有应对手段（合成数据、符号模板置换）规模小且修改幅度不够——修改后仍可能与训练样本相似

3. 即使低资源语言的语言学题目也在预训练数据中被覆盖，模型可通过部分污染绕过推理

**核心矛盾**：如何在保留解题推理逻辑不变的前提下，彻底消除模型利用知识和记忆的可能性？

**本文切入角度**：对语言学奥赛题的"题目语言"（Problemese）进行grapheme级正字法置换，使置换后的字符序列在任何训练语料中都不存在，但题目本身的推理步骤完全保留。

## 方法详解

### 整体框架

UKLO 82道题 → 专家人工标注置换规则集（ruleset）→ 每题生成最多6个正字法置换版本 → 1,203道问题 / 6,995个子问题-答案对 → Exact Match评估 → 比较 $M_{og}$（原始分数）和 $M_{obf}$（混淆分数）量化知识效应。

### 关键设计

1. **推理等变置换 (Reasoning-Equivariant Permutation)**

    - 以grapheme（字素）为最小单位置换，非word级——语言学题需要子词级符号推理
    - 每题由语言学专家手工定义ruleset，保留解题所需的语言学机制。以土耳其语元音和谐为例：元音对 (e,i)/(o,u)/(ö,ü)/(a,ı) 必须保持组内配对，否则后缀无法正确对应
    - 保留借词、英语同源词、人名/地名等对解题有用的元素
    - 移除语言名称、语系、地理信息等可能触发知识检索的元数据

2. **多版本评估与度量体系**

    - 定义 $M_{obf} = \frac{1}{82}\sum_{i=1}^{82}\frac{1}{n_i}\sum_{j=1}^{n_i}M_{obf}^{i,j}$（混淆版平均分）和 $M_{og}$（原始版分数）
    - 鲁棒度量 $M_{rob}$：取每题所有置换中最差分数的平均，衡量最坏情况推理能力
    - 知识效应 $\Delta_{obf}^{i} = M_{obf}^i - M_{og}^i$：负值越大说明模型越依赖知识
    - 基准验证：两名IOL奖牌获得者审计混淆题可解性；172人RCT显示人类仅下降5.7%

### 损失函数 / 训练策略

本文为评测基准。关键评估设计：

- 评估协议：每次prompt包含背景+上下文+所有问题+特定子问题，要求JSON输出
- 评分标准：严格Exact Match（不给部分分，防止通过重复上下文词获得虚假分数）
- 评估15个模型：包括GPT-5, Claude 3.7, o3-mini, Gemini, Llama等推理和通用模型

## 实验关键数据

### 主实验

15个模型在LingOly-TOO上的表现：

| 模型 | $M_{og}$（原始） | $M_{obf}$（混淆） | $M_{rob}$（鲁棒） | 下降幅度 |
|------|-----------------|-------------------|------------------|---------|
| GPT-5 | ~0.59 | **0.48** | 0.29 | -0.11 |
| Claude 3.7 (thinking) | ~0.55 | 0.44 | - | -0.11 |
| Claude 3.7 (no thinking) | ~0.40 | 0.30 | - | -0.10 |
| o3-mini (high) | ~0.45 | 0.31 | - | -0.14 |
| o3-mini (low) | ~0.25 | 0.13 | - | -0.12 |

GPT-5按难度（$M_{obf}$）：Breakthrough=0.81, Round 2=0.31

### 消融实验

| 分析维度 | 结果 |
|---------|------|
| 无上下文设置 | $M_{obf}$降至0.02-0.03，混淆有效阻断知识捷径 |
| Tokenization影响 | 改变分词策略不改善性能，排除tokenization解释 |
| 语言资源量效应 | 日语/芬兰语/意大利语$\Delta_{obf}$最大（-0.57~-0.59） |
| 专家引导推理 | 提供中间推理步骤后$M_{obf}$从0.66升至0.76 |
| 未公开新题测试 | UKLO 2025未发布题同样出现性能下降 |

### 关键发现

- 推理模型始终优于对应通用版本（o3-mini high vs low差18%），推理训练有实际效果
- 知识效应与语言资源量高度负相关（$\beta < 0, p < 0.01$，高资源语言膨胀最严重）
- 基准远未饱和：GPT-5在Round 2仅0.31，$M_{rob}$仅0.29
- 推理轨迹中常见重复分析、自相矛盾结论，推理一致性极差

## 亮点与洞察

- 正字法置换方法论优雅：grapheme级置换保留语言学推理逻辑，同时产生训练语料中不可能出现的字符序列
- 知识效应量化方法 $\Delta_{obf}$ 首次提供从知识中分离推理能力的可操作方案
- 人类RCT验证混淆仅造成5.7%下降而模型下降11%+，性能差主要因知识依赖而非认知惩罚
- $M_{rob}$揭示推理脆弱性：GPT-5从0.48降到0.29

## 局限与展望

- 严格Exact Match可能低估部分正确推理——但部分分数会虚假膨胀基线
- 仅覆盖自然语言模态的归纳/演绎推理，不涉及视觉或数学
- 82道基础题规模有限，置换规则需专家手工设计，自动化程度低
- 未探索更大范围的语言学现象或更多竞赛来源

## 相关工作与启发

- **vs LingOly**：LingOly-TOO增加正字法混淆以控制知识变量
- **vs GSM-Symbolic**：数值替换扰动幅度小；LingOly-TOO的grapheme置换产生完全全新字符序列
- **vs ARC/BIG-Bench Hard**：缺乏控制知识效应的机制
- **启发**：方法论可推广到其他需要符号推理的领域（音乐、密码学等）

## 评分

- 新颖性: ⭐⭐⭐⭐ 正字法混淆+知识/推理解耦设计精妙
- 实验充分度: ⭐⭐⭐⭐ 15模型+多维消融+人类RCT+未公开题验证
- 写作质量: ⭐⭐⭐⭐ 结构严谨，分析全面
- 价值: ⭐⭐⭐⭐⭐ 为LLM推理评测提供里程碑式的抗污染方法论
