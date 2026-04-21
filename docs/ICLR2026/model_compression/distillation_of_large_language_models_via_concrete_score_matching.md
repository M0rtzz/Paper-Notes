---
title: >-
  [论文解读] Distillation of Large Language Models via Concrete Score Matching
description: >-
  [ICLR 2026][模型压缩][知识蒸馏] 提出 Concrete Score Distillation (CSD)，一种基于离散 score matching 的 LLM 知识蒸馏损失，通过匹配 student 和 teacher 在所有词表对之间的相对 logit 差异，同时克服了 softmax 平滑和直接 logit 蒸馏的解空间限制问题。
tags:
  - ICLR 2026
  - 模型压缩
  - 知识蒸馏
  - LLM压缩
  - 分数匹配
  - logit蒸馏
  - 离散分数匹配
---

# Distillation of Large Language Models via Concrete Score Matching

**会议**: ICLR 2026  
**arXiv**: [2509.25837](https://arxiv.org/abs/2509.25837)  
**代码**: [GitHub](https://github.com/aailab-kaist/CSD)  
**领域**: 模型压缩 / 知识蒸馏  
**关键词**: 知识蒸馏, LLM压缩, 分数匹配, logit蒸馏, 离散分数匹配

## 一句话总结

提出 Concrete Score Distillation (CSD)，一种基于离散 score matching 的 LLM 知识蒸馏损失，通过匹配 student 和 teacher 在所有词表对之间的相对 logit 差异，同时克服了 softmax 平滑和直接 logit 蒸馏的解空间限制问题。

## 研究背景与动机

LLM 的部署推理成本极高，知识蒸馏（KD）是将大模型能力传递给小模型的核心手段。现有 KD 方法主要通过 softmax 概率匹配（如 KL 散度）来对齐 student 和 teacher 的分布，但 softmax 变换会严重平滑 logit 信息——当词表规模巨大时（如 128K），绝大多数 token 概率接近零（仅 0.0023% > 0.01），导致 teacher 的丰富知识在 softmax 后几乎不可区分。

直接 logit 蒸馏（DLD）通过 MSE 匹配原始 logit 可以避免 softmax 平滑，但存在另一个关键缺陷：**logit 偏移不变性**。由于 softmax 只关心 logit 之间的相对差异，$f_\theta[y_t] = f_T[y_t] + C$ 对所有 token 成立时概率完全相同，但 DLD 的 MSE 损失不为零，人为限制了最优解空间。当 teacher 和 student 容量差距大时，这种限制尤其有害。

本文的核心 idea：借鉴能量模型中 score matching 绕过归一化约束的思想，将离散 concrete score matching 引入 LLM 蒸馏，设计一个既不受 softmax 平滑影响、又不限制解空间的 logit 级蒸馏目标。

## 方法详解

### 整体框架

CSD 的核心是将 concrete score（离散概率比）的匹配转化为 logit 残差的匹配。对每个 token 位置，CSD 不是直接对齐 logit 值，而是对齐所有词表对 $(y_t, x)$ 之间的相对 logit 差异。

### 关键设计

1. **Log-transformed Concrete Score Matching**: 直接使用 concrete score（概率比 $q_\theta(x)/q_\theta(y_t)$）会因分母趋近零导致训练不稳定。通过取对数变换，将概率比匹配转化为 logit 差值匹配：

$$\mathcal{L}_{\text{CSD}} = \frac{1}{2} \sum_{y_t \in \mathcal{V}} \sum_{x \in \mathcal{V}} w(y_t, x) \left( f_\theta[x] - f_\theta[y_t] - f_T[x] + f_T[y_t] \right)^2$$

对数变换带来两个好处：(1) 避免计算概率比，保证训练稳定性；(2) 自然得到 logit 级损失设计。

2. **解空间超集性质（Theorem 2）**: CSD 的最优解集严格包含 DLD 的最优解集 $\Theta_{\text{CSD}}^* \supsetneq \Theta_{\text{DLD}}^*$。当 $f_\theta[y_t] = f_T[y_t] + C$ 时，CSD 损失为零而 DLD 不为零。这意味着 CSD 允许 student 在更大的空间中搜索最优解，在有限模型容量下尤为有利。

3. **线性时间梯度计算（Theorem 3）**: 虽然 CSD 需要对词表的双重求和（$O(|\mathcal{V}|^2)$），但通过将权重函数分解为 $w(y_t, x) = w_1(y_t) \cdot w_2(x)$，梯度可以在 $O(|\mathcal{V}|)$ 时间内解析计算。关键步骤是对 logit 做加权中心化归一化后计算残差加权和。

4. **灵活的权重设计空间**: 通过选择不同的 $w_1, w_2$（teacher 概率 T、student 概率 S、均匀分布 U），CSD 可以在 mode-seeking 和 mode-covering 之间灵活切换。$(S,S)$ 偏向高保真度，$(U,S)$ 和 $(T,S)$ 增加多样性。

### 损失函数 / 训练策略

- 梯度结构类似 KL 散度，但用**中心化归一化**替代 softmax 归一化，避免 softmax 平滑
- 支持与 on-policy 技术（ImitKD, GKD, DistiLLM）正交组合
- 也支持 Monte Carlo 梯度估计（处理联合权重函数），但解析梯度收敛更快

## 实验关键数据

### 主实验

**表1: GPT-2-1.5B → GPT-2-0.1B 纯损失函数对比（ROUGE-L）**

| 损失函数 | Dolly | Self-Instruct | Vicuna | Super-NI | UnNI | 平均 |
|---------|-------|--------------|--------|----------|------|------|
| KL | 23.52 | 10.02 | 14.57 | 16.76 | 18.55 | 16.68 |
| RKL | 24.26 | 11.19 | 15.80 | 20.17 | 22.99 | 18.88 |
| SRKL | 24.53 | 12.19 | 15.63 | 23.37 | 24.28 | 20.00 |
| **CSD (Ours)** | **24.94** | **12.06** | **15.78** | **24.60** | **25.88** | **20.65** |

**表2: 任务特定蒸馏 Gemma-7B-IT → Gemma-2B-IT**

| 损失 | 摘要 ROUGE-L | 翻译 COMET | GSM8K Acc |
|------|-------------|-----------|-----------|
| KL | 35.02 | 73.96 | 24.03 |
| **CSD (T,S)** | **35.67** | **74.14** | **25.78** |
| RKL | 0.00 | 45.02 | 0.00 |

### 消融实验

- **CSD vs DLD 在相同权重下**: CSD (T,T)/(U,U)/(S,S) 一致优于 DLD T/U/S，验证了更大解空间的优势
- **权重选择**: (S,S) 最高保真度，(U,S) 最好多样性-保真度权衡，(T,S) 最好概率校准
- **通用对话蒸馏**: Qwen2.5-7B→1.5B 上 MT-Bench 从 5.75 (DistiLLM-2) 提升到 5.95 (CSD)
- **解析梯度 vs Monte Carlo**: 解析计算收敛略快且性能更优

### 关键发现

- softmax 在大词表 LLM 中损失了大量 teacher 知识（99.99%+ token 概率 <0.01）
- logit 偏移不变性是 DLD 的核心缺陷，CSD 通过差分结构自然解决
- mode-seeking 损失（RKL, SKL）在小数据蒸馏中容易崩溃，CSD 通过权重选择避免此问题
- CSD 与 on-policy 方法正交互补，尤其在纯 on-policy 设置下提升最大

## 亮点与洞察

- 从能量模型视角重新审视 LLM 蒸馏问题，思路新颖且理论扎实
- logit 偏移不变性的分析简洁有力，直指 DLD 的根本缺陷
- 权重函数设计空间提供了统一框架理解 mode-seeking 与 mode-covering
- 梯度的解析计算使得 $O(|\mathcal{V}|^2)$ 的 CSM 变为实际可用的 $O(|\mathcal{V}|)$

## 局限与展望

- 权重函数空间的探索仍有限，联合权重函数的设计尚未充分挖掘
- 分解假设 $w(y_t,x)=w_1(y_t)w_2(x)$ 限制了权重的表达能力
- 在更大规模模型（>10B teacher）上的验证不够充分
- 与量化、剪枝等其他压缩方法的结合未探讨

## 相关工作与启发

- **DistiLLM** (Ko et al., 2024): 提出 SKL/SRKL 平滑 KL 蒸馏，本文在此基础上进一步改进
- **GKD** (Agarwal et al., 2024): f-divergence 框架，CSD 可视为超越概率匹配的新框架
- **Concrete Score Matching** (Meng et al., 2022): 离散扩散模型中的 score matching，本文创新性地适配到自回归 LLM 蒸馏
- 启发：score matching 的"绕过归一化"思想可能在其他需要分布匹配的 NLP 任务中也有应用

## 评分

- 新颖性: ⭐⭐⭐⭐ 从 EBM score matching 到 LLM 蒸馏的迁移有创意，但核心仍是 logit 差分匹配
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖多种 backbone、任务类型、on-policy 组合，消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，图示清晰，动机分析到位
- 价值: ⭐⭐⭐⭐ 提供了统一的 logit 蒸馏设计框架，实际性能提升稳定但幅度有限

<!-- RELATED:START -->

## 相关论文

- [Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [Unveiling Super Experts in Mixture-of-Experts Large Language Models](unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [Is Finer Better? The Limits of Microscaling Formats in Large Language Models](is_finer_better_the_limits_of_microscaling_formats_in_large_language_models.md)

<!-- RELATED:END -->
