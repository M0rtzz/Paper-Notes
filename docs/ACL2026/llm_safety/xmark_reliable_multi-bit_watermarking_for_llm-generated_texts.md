---
title: >-
  [论文解读] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts
description: >-
  [ACL 2026][LLM安全][多比特水印] 提出 XMark，一种基于 Leave-one-Shard-out（LoSo）策略和 evergreen list 的多比特文本水印方法，通过跨多个词表排列的绿色列表交集和约束 token-shard 映射矩阵…
tags:
  - "ACL 2026"
  - "LLM安全"
  - "多比特水印"
  - "LLM文本检测"
  - "数字水印"
  - "文本溯源"
  - "logit扰动"
---

# XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts

**会议**: ACL 2026  
**arXiv**: [2604.05242](https://arxiv.org/abs/2604.05242)  
**代码**: [https://github.com/JiiahaoXU/XMark](https://github.com/JiiahaoXU/XMark)  
**领域**: 文本水印  
**关键词**: 多比特水印, LLM文本检测, 数字水印, 文本溯源, logit扰动

## 一句话总结

提出 XMark，一种基于 Leave-one-Shard-out（LoSo）策略和 evergreen list 的多比特文本水印方法，通过跨多个词表排列的绿色列表交集和约束 token-shard 映射矩阵，在保持文本质量的同时显著提升了有限 token 条件下的解码准确率。

## 研究背景与动机

**领域现状**：多比特文本水印能在 LLM 生成文本中嵌入用户 ID、时间戳等可提取的二进制信息，用于恶意使用的溯源和归因。现有方法分为无失真方法（水印文本与未水印文本同分布）和 logit 扰动方法（通过修改 logit 嵌入信息）。

**现有痛点**：(1) 早期方法（CycleShift、CTWL、DepthW）解码时需暴力枚举所有候选消息，长消息计算不可行；(2) MPAC 采用分块编解码解决了可行性问题，但绿色列表比例被限制在 $\gamma \leq 0.25$，导致 token 采样概率严重失真，文本质量下降明显；(3) StealthInk 改善了文本质量但削弱了水印信号，降低了解码准确率；(4) **所有方法在可用 token 数有限时解码准确率急剧下降**，而实际使用中短文本很常见。

**核心矛盾**：文本质量和解码准确率之间存在根本性权衡——更大的绿色列表减少分布失真但削弱水印信号，更小的绿色列表增强信号但严重影响质量。特别是在有限 token 条件下，这个矛盾更加尖锐。

**本文目标**：同时提升水印文本质量和有限 token 条件下的解码准确率。

**切入角度**：反转绿色列表选择策略——不把编码消息对应的 shard 作为绿色列表（MPAC），而是排除该 shard、把其余所有 shard 作为绿色列表，将绿色列表比例从 $\leq 0.25$ 提升到 $\geq 0.75$。

**核心 idea**：用 Leave-one-Shard-out 提升文本质量，用多个排列的 evergreen list 交集增加每个 token 的观测次数来补偿信号强度，用约束 TMM 防止未加扰 shard 计数爆炸。

## 方法详解

### 整体框架

XMark 遵循分块编解码范式：将 $b$ 比特消息分为 $r$ 个块，每个块 $d$ 比特。编码时每生成一个 token 嵌入一个消息块的信息，解码时从嫌疑文本中恢复各块消息。核心创新在编码器的 LoSo + evergreen list 设计和解码器的 cTMM 设计。

### 关键设计

1. **Leave-one-Shard-out (LoSo) 编码**:

    - 功能：通过反转绿色列表选择策略大幅提升文本质量
    - 核心思路：MPAC 把消息值 $[\mathbf{m}_i]_{10}$ 对应的 shard 作为绿色列表（$\gamma = 2^{-d} \leq 0.25$），而 LoSo 把该 shard 排除，其余所有 shard 组成绿色列表（$\gamma = 1 - 2^{-d} \geq 0.75$）。解码时反向寻找 token 数最少的 shard 来恢复消息。例如 $d=2, \mathbf{m}_i=11$ 时，排除 $\mathcal{S}_3$，加扰 $\mathcal{S}_0, \mathcal{S}_1, \mathcal{S}_2$
    - 设计动机：绿色列表比例从 0.25 提升到 0.75 意味着大部分词表的 logit 分布保持原样，文本质量接近未水印文本

2. **Evergreen List（多排列交集）**:

    - 功能：在保持大绿色列表的同时增加每个 token 对解码的信息贡献，弥补 LoSo 信号弱的问题
    - 核心思路：使用 $k$ 个不同的哈希密钥生成 $k$ 个词表排列，每个排列各有一个 LoSo 绿色列表 $\mathcal{G}_j$，取所有绿色列表的交集作为 evergreen list $\mathcal{E} = \bigcap_{j=0}^{k-1} \mathcal{G}_j$。只加扰 $\mathcal{E}$ 中 token 的 logit。期望绿色列表比例为 $\mathbb{E}[\gamma] \approx (1-2^{-d})^k$。解码时每个 token 可以在 $k$ 个排列中各贡献一次观测，从 $T$ 个 token 最多获得 $kT$ 次观测
    - 设计动机：单一 LoSo 的信号太弱（比特准确率低于 MPAC），但通过多排列的 evergreen list，既维持了较大的绿色列表比例，又将观测次数放大了 $k$ 倍，显著提升了有限 token 条件下的解码可靠性

3. **约束 Token-Shard 映射矩阵（cTMM）**:

    - 功能：防止解码时未加扰 shard 的计数爆炸，提升解码鲁棒性
    - 核心思路：标准 TMM 中，一个 token 可能在 $k$ 个排列中都被映射到同一个未加扰 shard，导致该 shard 计数被放大 $k$ 倍，淹没加扰 shard 和未加扰 shard 之间的区别。cTMM 约束每个 token 对每个 shard 最多贡献 1 次计数：$\mathbf{A}^t[i,:] - \mathbf{A}^{t-1}[i,:] \in \{0,1\}^{2^d}$
    - 设计动机：没有此约束时，不属于任何绿色列表的 token 会被计数 $k$ 次到未加扰 shard，使其计数可能超过加扰 shard，导致解码失败

### 损失函数 / 训练策略

XMark 是无需训练的推理时水印方法。编码通过在 LLM 生成每个 token 时向 evergreen list token 的 logit 加正偏置 $\delta$ 实现。默认设置 $d=2$（每块 2 比特），超参数 $k$ 控制质量-准确率权衡。

## 实验关键数据

### 主实验

文本补全任务（LLaMA-2-7B, C4 数据集, b=8 比特）：

| 方法 | T=150 BA↑ | T=300 BA↑ | 平均 PPL↓ | 说明 |
|------|----------|----------|----------|------|
| MPAC | 94.00 | 98.25 | 5.08 | 绿色列表小，质量差 |
| StealthInk | 85.00 | 92.50 | 4.13 | 质量好但准确率低 |
| CycleShift | 95.25 | 98.25 | 5.06 | 需暴力枚举 |
| XMark | **98.75** | **100.00** | **4.61** | 质量和准确率双优 |

未水印文本 PPL 为 3.97，XMark 的 PPL 最接近。

### 消融实验

| 配置 | T=100 BA↑ | 说明 |
|------|----------|------|
| LoSo (k=1) | 74.12 | 信号太弱 |
| MPAC | 83.62 | 绿色列表小但信号强 |
| XMark (LoSo+evergreen+cTMM) | ~95+ | 三重设计协同 |
| XMark 用 TMM 替代 cTMM | 下降 | 未加扰 shard 计数爆炸 |

### 关键发现

- XMark 在所有 token 预算（T=150-300）下都同时超越了所有基线的准确率和文本质量
- **有限 token 条件下优势最大**：T=150 时 XMark BA 98.75% vs MPAC 94.00%，差距 4.75%
- 在文本摘要等更难的任务上优势更加显著——XMark BA 79.81% vs MPAC 76.94%，且 PPL 低 1.28
- 超参数 $k$ 有效控制质量-准确率权衡：$k$ 增大准确率提升但 PPL 略增

## 亮点与洞察

- **LoSo 策略的"反转"思维**非常优雅——简单地反转绿色列表选择就将 $\gamma$ 从 $\leq 0.25$ 提升到 $\geq 0.75$，大幅减少分布失真。这个思路类似于纠错编码中的"校验位"思想
- **cTMM 的约束设计**精准解决了 evergreen list 引入的解码偏差——每个 token 对每个 shard 最多贡献 1 次，防止了多排列带来的计数爆炸问题
- 三个设计（LoSo、evergreen list、cTMM）形成了紧密耦合的整体——LoSo 解决质量但损失信号，evergreen list 恢复信号但引入偏差，cTMM 消除偏差

## 局限与展望

- 仅在 LLaMA-2-7B 上验证，更大或更新的模型上的效果未知
- 抗编辑攻击（paraphrase、删除等）的鲁棒性分析有限
- $k$ 的选择需要针对每个场景调优
- 多比特水印的安全性分析（能否被恶意提取或伪造）未深入讨论

## 相关工作与启发

- **vs MPAC**: MPAC 把消息对应 shard 作为绿色列表（$\gamma=2^{-d}$），XMark 反转为排除该 shard（$\gamma=1-2^{-d}$），加上 evergreen list 和 cTMM 后同时超越了 MPAC 的质量和准确率
- **vs StealthInk**: StealthInk 通过直接提升大 logit token 的概率改善质量但削弱信号。XMark 通过多排列交集在保持大绿色列表的同时增强信号，是更根本的解决方案

## 评分

- 新颖性: ⭐⭐⭐⭐ LoSo+evergreen list+cTMM 的组合设计有创意，但每个单独组件的技术含量有限
- 实验充分度: ⭐⭐⭐⭐ 多任务多基线对比充分，不同 token 预算下的分析有价值，但模型多样性不足
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，设计动机清晰，图示帮助理解
- 价值: ⭐⭐⭐⭐ 有限 token 场景的实际价值高，但水印领域竞争激烈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Robust Multi-bit Text Watermark with LLM-based Paraphrasers](../../ICML2025/llm_safety/robust_multi-bit_text_watermark_with_llm-based_paraphrasers.md)
- [\[ACL 2026\] STELA: A Linguistics-Aware LLM Watermarking via Syntactic Predictability](a_linguistics-aware_llm_watermarking_via_syntactic_predictability.md)
- [\[ACL 2026\] SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking](ssg_logit-balanced_vocabulary_partitioning_for_llm_watermarking.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[ACL 2026\] Privacy-R1: Privacy-Aware Multi-LLM Agent Collaboration via Reinforcement Learning](privacy-r1_privacy-aware_multi-llm_agent_collaboration_via_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
