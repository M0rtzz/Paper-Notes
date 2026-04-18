---
title: >-
  [论文解读] JULI: Jailbreak Large Language Models by Self-Introspection
description: >-
  [ICLR 2026][机器人][jailbreak] 揭示对齐 LLM 的 top-k token log probability 中仍包含有害信息的知识泄露问题，提出 JULI——仅用不到目标模型 1% 参数量的 BiasNet 插件操纵 logit bias，在仅访问 top-5 token 概率的 API 场景下成功越狱 Gemini-2.5-Pro（Harmful Info Score 4.19/5），比 LINT 快 140 倍同时 harmfulness 提升约 2 倍。
tags:
  - ICLR 2026
  - 机器人
  - jailbreak
  - logit bias
  - API attack
  - token log probability
  - BiasNet
---

# JULI: Jailbreak Large Language Models by Self-Introspection

**会议**: ICLR 2026  
**arXiv**: [2505.11790](https://arxiv.org/abs/2505.11790)  
**代码**: 无  
**领域**: AI安全 / Jailbreak 攻击  
**关键词**: jailbreak, logit bias, API attack, token log probability, BiasNet

## 一句话总结
揭示对齐 LLM 的 top-k token log probability 中仍包含有害信息的知识泄露问题，提出 JULI——仅用不到目标模型 1% 参数量的 BiasNet 插件操纵 logit bias，在仅访问 top-5 token 概率的 API 场景下成功越狱 Gemini-2.5-Pro（Harmful Info Score 4.19/5），比 LINT 快 140 倍同时 harmfulness 提升约 2 倍。

## 研究背景与动机
**领域现状**：LLM 越狱攻击分为需要模型权重的白盒攻击和仅通过 API 的黑盒攻击。API 场景下的攻击极具挑战——无法访问梯度、完整 logits 或生成过程。

**现有痛点**：(a) GCG 等白盒方法需要完整梯度访问，不适用于商用 API；(b) LINT（当前 API 攻击方法）需要 top-500 token 访问（多数 API 仅提供 top-5/20），且推理需 99.7 秒，harmfulness 仅 2.25/5；(c) Weak-to-Strong 和 Emulated Disalignment 需要对齐前后两个版本的模型权重。

**核心矛盾**：对齐训练应该消除有害知识的表达，但 LLM API 返回的 top-k token 概率中是否仍泄露有害信息？

**本文目标** 能否仅用 API 返回的少量 token 概率（如 top-5）高效越狱主流商用 LLM？

**切入角度**：统计发现 >85% 的有害 response token 出现在 top-5 概率中——对齐只是压低了它们的采样概率而非消除知识。

**核心 idea**：用轻量 BiasNet（<1% 目标模型参数）学习 logit bias 来提升有害 token 采样概率，仅需 100 条有害数据训练。

## 方法详解

### 整体框架
BiasNet $F_\theta$ 接收目标 LLM 的 log probability 输出 $\log p_\alpha(x_n)$，计算 logit bias $B = F_\theta(\log p_\alpha(x_n))$，修正后的概率 $\tilde{p}_\alpha(x_n) = p_\alpha(x_n) + B$。

### 关键设计

1. **Token 泄露发现**：对多个对齐 LLM 统计，>85% 的有害 response token 出现在 top-5 预测概率中——对齐训练未消除有害知识，仅降低了概率排名。核心洞察：对齐是"概率压低"而非"知识擦除"。

2. **BiasNet 架构**：<1% 目标模型参数（~$10^7$），三层结构：

    - 投影层 1：token 空间 → 隐藏空间（白盒：复用 LLM head 的伪逆；黑盒：随机正交矩阵）
    - 中间变换层（可学习）
    - 投影层 2：隐藏空间 → token 空间（白盒：复用 LLM head；黑盒：随机正交矩阵）
    - 输出 logit bias $B = F_\theta(\log p_\alpha(x_n))$，修正概率 $\log \tilde{p} = \log p + B$

3. **Padding 机制（API 场景）**：API 仅返回 top-k token 概率时，将非 top-k token 赋予 padding 值（第 k 个 token 概率减固定偏移 10），使 BiasNet 可在不完整概率向量上工作。

4. **训练**：仅 100 条 LLM-LAT 有害问答对，15 epochs，batch size=1，AdamW lr=$10^{-5}$。极低成本。

## 实验关键数据

### 开源模型攻击（白盒设置，AdvBench）

| 目标模型 | JULI Harmful Score | 最佳基线 | 基线方法 | JULI 推理时间 |
|----------|-------------------|---------|---------|-------------|
| Llama3-8B-Instruct | **3.44** | 3.02 | ED | 0.71s |
| Llama2-7B-Chat | **3.38** | 2.89 | ED | 0.71s |
| Llama3-3B-Instruct | **3.52** | 3.15 | ED | 0.65s |
| Qwen2-1.5B-Instruct | **3.61** | 3.28 | ED | 0.58s |
| Llama3-8B-CB (Circuit Breaker 防御) | **2.95** | 1.85 | GCG | 0.71s |
| Llama2-7B-DeepAlign (DeepAlign 防御) | **3.21** | 2.45 | GCG | 0.71s |

### API 攻击（黑盒设置，top-5 API）

| 目标模型 | JULI Harmful Info Score | FLIP | Naive | Base |
|----------|----------------------|------|-------|------|
| Gemini-2.5-Pro | **4.19** | 2.09 | 1.21 | 0.06 |
| Gemini-2.5-Flash | **1.74** | 1.33 | 1.29 | 0.02 |

### 关键发现
- 对齐 LLM 的 top-5 token 概率足以恢复有害输出——对齐是概率压低而非知识擦除
- 仅 100 条训练数据 + <1% 参数的插件即可攻破 SOTA 防御（Circuit Breaker + DeepAlign）
- 比 LINT 快 140 倍（0.71s vs 99.7s），harmfulness 提升 ~2 倍（3.44 vs 2.25）
- 新提出的 Harmful Info Score 与人类判断的相关性高于传统 BERT Score 和 Harmful Score

## 亮点与洞察
- **"知识泄露" vs "知识擦除"的深刻启示**：与 Erase or Hide 的"浅层对齐"发现一致——对齐后有害知识仍完整保留在模型中，只是被概率性地抑制。JULI 证明这种抑制可以被外部插件轻松逆转。这一发现对对齐研究有根本性影响——意味着当前所有基于 RLHF/DPO 的安全训练都只是"表面功夫"。
- **API 安全的红旗**：现实中的 LLM API（如 Gemini API）返回 top-k 概率，JULI 证明这本身就是一个攻击面。API 设计者需要重新评估返回 log probability 的安全风险。
- **Harmful Info Score 的方法论贡献**：新提出的评估指标更关注回复的信息量和质量而非表面"有害性"标签，与人类判断相关性更高——可以作为越狱评估的新标准。

## 消融实验与深入分析

| 分析维度 | 发现 |
|----------|------|
| Top-k 命中率 | >85% 有害 token 在 top-5 中，>95% 在 top-10 中 |
| 训练数据量 | 仅 100 条样本即达到饱和，更多数据边际收益极小 |
| 投影层选择 | 白盒复用 LLM head 优于随机初始化，但黑盒随机正交矩阵也可工作 |
| 对防御的鲁棒性 | 攻破 Circuit Breaker 和 DeepAlign 两种 SOTA 防御 |
| 困难子集 | 在 AdvBench 的 5% 困难子集上仍有效，而基线方法几乎失败 |
| Harmful Info Score | 新提出的评估指标，与人类判断的相关性高于 BERT Score 和 Harmful Score |

## 局限与展望
- BiasNet 需要少量有害数据训练（100 条），限制了完全零知识攻击
- 防御方案未深入讨论——限制 API 返回的 token 数或对概率加噪是显而易见的缓解措施
- API 提供商可以通过不返回 log probability 来防御，但这会牺牲合法用途
- 目前仅在 Gemini API 上验证黑盒攻击，OpenAI API 需进一步测试
- BiasNet 的 padding 机制在某些 token 分布下可能引入偏差

## 相关工作与启发
- **vs GCG (Zou et al.)**：GCG 需要完整梯度访问（白盒），JULI 仅需 top-5 概率——实用性有质的差距
- **vs Weak-to-Strong (Zhao et al.)**：WTS 需要预训练基座模型权重；JULI 完全通过 API 工作
- **vs LINT (Zhang et al.)**：LINT 需要 top-500 token，推理 99.7 秒；JULI 仅需 top-5，0.71 秒——140 倍加速
- **vs Emulated Disalignment**：ED 需要对齐前后两个版本的权重；JULI 完全黑盒
- **启发——对齐是"概率压低"而非"知识擦除"**：最深刻贡献是证明对齐后有害知识仍完整保留在模型中——top-5 内部仍有足够信息恢复完整有害输出

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个仅用 top-5 API 概率的实用越狱，BiasNet 概念新颖
- 实验充分度: ⭐⭐⭐⭐ 多模型（含闭源）× 多场景 × 多评估指标 × 含 SOTA 防御
- 写作质量: ⭐⭐⭐⭐ 清晰，Harmful Info Score 有方法论贡献
- 价值: ⭐⭐⭐⭐ 对 API 安全设计有直接警示——是否应该返回 log probability 需重新评估
