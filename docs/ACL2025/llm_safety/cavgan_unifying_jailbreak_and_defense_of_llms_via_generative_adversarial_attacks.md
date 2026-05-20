---
title: >-
  [论文解读] CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks
description: >-
  [ACL 2025][LLM安全][越狱攻击] 提出 CAVGAN 框架，利用生成对抗网络在 LLM 内部表示空间中同时学习越狱攻击（生成器）和安全防御（判别器），首次将攻防统一到同一框架中实现"攻防共进"，攻击成功率平均 88.85%，防御成功率平均 84.17%。
tags:
  - "ACL 2025"
  - "LLM安全"
  - "越狱攻击"
  - "defense"
  - "GAN"
  - "concept activation vector"
  - "LLM 安全"
  - "对抗训练"
---

# CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks

**会议**: ACL 2025  
**arXiv**: [2507.06043](https://arxiv.org/abs/2507.06043)  
**代码**: [GitHub](https://github.com/NLPGM/CAVGAN)  
**机构**: 武汉大学计算机学院, 中关村学院
**领域**: AI 安全  
**关键词**: 越狱攻击, defense, GAN, concept activation vector, LLM 安全, 对抗训练

## 一句话总结

提出 CAVGAN 框架，利用生成对抗网络在 LLM 内部表示空间中同时学习越狱攻击（生成器）和安全防御（判别器），首次将攻防统一到同一框架中实现"攻防共进"，攻击成功率平均 88.85%，防御成功率平均 84.17%。

## 研究背景与动机

**领域现状**: LLM 经过安全对齐（RLHF/SFT/DPO）后具备一定安全防护能力，但各种越狱攻击不断暴露其安全机制的脆弱性。现有研究将越狱攻击和安全防御视为两个**相互割裂**的研究方向，分别独立优化。

**现有痛点**: 白盒攻击方法（如 SCAV、JRE）通过数学迭代优化或正负样本差异提取扰动向量，过程复杂且难以泛化；防御方法（如输入过滤、知识编辑）要么依赖外部检测模型精度有限，要么修改参数导致模型流畅度下降。两者缺乏协同，无法形成"以攻促防"的闭环。

**关键观察**: (a) LLM 中间层 embedding 中恶意/良性查询具有**线性可分性**，简单分类器即可区分；(b) 成功的越狱攻击本质上是将恶意查询的表示从"不安全区域"迁移到"安全区域"，即一个**边界跨越问题**。

**核心idea**: 将概念激活向量（CAV）的提取从手工/数学优化重新定义为**生成过程**，用 GAN 的生成器自动生成越狱扰动、判别器自动识别伪装的恶意查询，两者在对抗训练中同步提升。

## 方法详解

### 整体框架

CAVGAN 基于 LLM 解码层的内部 embedding 空间运作，包含三个阶段：

1. **训练阶段**: 生成器 G 和判别器 D 在 LLM 内部 embedding 上进行对抗训练
2. **攻击阶段**: 用训练好的 G 生成扰动注入中间层实现越狱
3. **防御阶段**: 用训练好的 D 检测不安全输入并引导模型重新生成

### 问题形式化

给定 LLM M 的 L 层内部 embedding {h_0, h_1, ..., h_L}。安全判别器 G 输出概率 p in (0,1)，表示输入的恶意意图程度。越狱攻击目标：找到扰动 delta 使 min(G(h+delta))，约束 ||delta|| <= epsilon。

### 生成器（攻击）

- 输入：恶意查询在目标层的 embedding h
- 输出：扰动向量 G(h)，注入后使判别器无法识别恶意意图
- 损失函数：L_G = E_{h~D_m}[log D(h + G(h))]
- 通过参数权重归一化隐式约束扰动模长，避免偏离语义空间

### 判别器（防御）

判别器需同时区分三类输入：良性原始 embedding、恶意原始 embedding、经扰动伪装的恶意 embedding。损失分两部分：

- 原始分类：L_real = E_{h~D_b}[log D(h)] + E_{h~D_m}[log(1-D(h))]
- 伪装识别：L_fake = E_{h~D_m}[log(1-D(h+G(h)))]
- 总损失：L_D = L_real + L_fake

### 防御推理流程

推理时，输入 Q 的 embedding h_Q 送入判别器。若 G(h_Q) >= p_0（阈值），则拼接安全提示前缀 P_safe 后重新生成，否则正常输出。

### 层选择策略

- 接近中间层的位置攻击效果最佳
- 后层 ASR-kw 不低但文本质量骤降（大量重复无意义字符）
- 前层文本质量未下降但 ASR 很低
- 结论：LLM 的安全机制是**逐层渐进形成**的

## 实验关键数据

### 表1: 三模型越狱攻击结果（AdvBench + StrongREJECT）

| 模型 | 方法 | ASR-kw | ASR-gpt | ASR-Answer | ASR-Useful | ASR-Rep |
|------|------|--------|---------|------------|------------|---------|
| Qwen2.5-7B | SCAV | 99.85 | 87.54 | 78.65 | 98.07 | 100.00 |
| Qwen2.5-7B | JRE | 83.54 | 70.00 | 60.96 | 61.15 | 55.00 |
| Qwen2.5-7B | **CAVGAN** | 98.98 | 83.88 | 70.41 | 86.73 | 99.80 |
| Llama3.1-8B | SCAV | 100.00 | 90.65 | 87.11 | 95.19 | 99.23 |
| Llama3.1-8B | **CAVGAN** | 98.78 | 88.38 | 78.16 | 88.38 | 99.38 |
| Mistral-8B | SCAV | 99.24 | 78.26 | 82.30 | 80.76 | 85.19 |
| Mistral-8B | **CAVGAN** | **95.51** | **94.29** | **88.78** | **95.10** | **99.18** |

CAVGAN 在 Mistral-8B 上全面超越 SCAV，在其余两个模型上与 SCAV 有小幅差距但大幅超越 JRE。

### 表2: 防御实验结果（SafeEdit 越狱数据集 + Alpaca 正常查询）

| 模型 | 方法 | DSR | BAR | ASR Reduce | BAR Reduce |
|------|------|-----|-----|------------|------------|
| Qwen2.5-7B | Original | 25.12 | 98.00 | - | - |
| Qwen2.5-7B | SmoothLLM | 54.22 | 75.77 | 38.86 | 22.68 |
| Qwen2.5-7B | RA-LLM | 78.60 | 85.80 | 71.42 | 12.45 |
| Qwen2.5-7B | **CAVGAN** | **91.12** | **91.40** | **88.14** | **10.06** |
| Llama3.1-8B | Original | 11.34 | 99.60 | - | - |
| Llama3.1-8B | SmoothLLM | 48.97 | 81.03 | 42.44 | 18.64 |
| Llama3.1-8B | RA-LLM | 73.78 | 92.80 | 70.42 | 6.83 |
| Llama3.1-8B | **CAVGAN** | **77.22** | **93.60** | **74.31** | **6.02** |

CAVGAN 在 Qwen2.5-7B 上 DSR 达 91.12%（SOTA +12%），同时 BAR 维持 91.40%，安全与可用性平衡最佳。

### 关键发现

- 攻击：三个 LLM 平均越狱成功率 97%（ASR-kw），在 Mistral-8B 上全方位击败 SCAV
- 防御：DSR 平均 84.17%，在 Qwen 上比 RA-LLM 高 12 个百分点，且 BAR 损失最小
- 可扩展：在 Qwen2.5-14B/32B 大模型上攻击仍保持 94%+ ASR-kw
- 训练样本：仅需 80-100 个样本即可达到最佳性能，更多样本反而因 GAN 不稳定导致性能波动

## 亮点与洞察

- **攻防统一新范式**: 首次将越狱攻击与安全防御纳入同一 GAN 框架，验证了"以攻促防"的双向控制范式，在概念上优于分别优化
- **CAV 生成替代数学优化**: 将传统的正负样本差异提取或迭代优化转化为模型生成过程，降低复杂度且易于扩展到其他领域
- **安全机制可解释性**: 通过层选择实验揭示 LLM 安全机制是逐层渐进形成的，中间层是攻击最佳切入点
- **轻量高效**: 生成器和判别器均为 4 层 MLP，训练仅需 100 个样本和 10 个 epoch，成本极低

## 局限性

- **GAN 结构过于简单**: 生成器/判别器均为简单 MLP，对复杂语义的建模能力有限，作者承认更复杂结构可能获得更好效果
- **超参数依赖验证集**: 阈值 p_0 和目标层选择依赖验证集调优，缺乏自动化选择机制
- **防御引入延迟**: 基于重新生成的防御机制在高实时性场景下会增加响应时间
- **泛化边界未明**: 仅在 7B-32B 规模的 3 个模型家族上验证，未测试闭源模型或更大规模模型
- **训练数据敏感**: 80-100 样本是最优区间，GAN 训练不稳定导致样本量增加反而性能下降

## 相关工作

- **白盒越狱攻击**: SCAV（数学迭代优化求扰动）、JRE（正负样本 embedding 差异）、GCG（梯度搜索对抗后缀）
- **黑盒越狱攻击**: 手工 prompt 模板、遗传算法搜索、迭代优化 prompt
- **免微调防御**: SmoothLLM（随机扰动输入检测攻击）、RA-LLM（随机扰动 + 一致性检验）
- **知识编辑防御**: SafeEdit 等方法编辑模型有毒区域，但修改参数引入未知风险
- **表示工程**: Representation Engineering 用 CAV 控制模型行为，本文将其与 GAN 结合

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将 GAN 用于统一 LLM 攻防，将 CAV 提取转化为生成过程
- 实验充分度: ⭐⭐⭐⭐ 3 模型 x 2 数据集攻击 + 2 模型防御 + 层选择/样本量消融
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，数学形式化规范，攻防流程逻辑连贯
- 实用性: ⭐⭐⭐ MLP 架构轻量易部署，但防御的重新生成机制增加延迟
- 总体推荐: ⭐⭐⭐⭐ 概念贡献突出，实验扎实，GAN 结构简单限制了上限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](tip_iceberg_adversarial_attacks.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](../../NeurIPS2025/llm_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[ACL 2025\] Bias in the Mirror: Are LLMs' Opinions Robust to Their Own Adversarial Attacks](bias_in_the_mirror_are_llms_opinions_robust_to_their_own_adversarial_attacks.md)
- [\[ACL 2025\] PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative Prompts](pig_privacy_jailbreak.md)
- [\[NeurIPS 2025\] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](../../NeurIPS2025/llm_safety/mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)

</div>

<!-- RELATED:END -->
