---
title: >-
  [论文解读] Scalable Fingerprinting of Large Language Models
description: >-
  [NeurIPS 2025][model fingerprinting] 提出 Perinucleus 采样方法生成可扩展的 LLM 指纹，能在 Llama-3.1-8B 上嵌入 24,576 个指纹（比现有方法多两个数量级）且不损害模型能力，并通过理论和实验证明大规模指纹是抵御共谋攻击的关键。
tags:
  - NeurIPS 2025
  - model fingerprinting
  - LLM ownership
  - Perinucleus sampling
  - collusion attack
  - model security
---

# Scalable Fingerprinting of Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2502.07760](https://arxiv.org/abs/2502.07760)  
**代码**: [GitHub](https://github.com/SewoongLab/scalable-fingerprinting-of-llms)  
**领域**: llm_nlp  
**关键词**: model fingerprinting, LLM ownership, Perinucleus sampling, collusion attack, model security

## 一句话总结
提出 Perinucleus 采样方法生成可扩展的 LLM 指纹，能在 Llama-3.1-8B 上嵌入 24,576 个指纹（比现有方法多两个数量级）且不损害模型能力，并通过理论和实验证明大规模指纹是抵御共谋攻击的关键。

## 研究背景与动机

**模型指纹的需求**：模型指纹允许所有者通过 API 访问识别被非法使用的模型

**可扩展性为何关键**：降低误检率、抵御指纹泄露（每次验证暴露一个指纹）、防御共谋攻击（多用户联合规避）

**现有方法瓶颈**：RANDOM（随机 token key）可扩展但不安全；ENGLISH-RANDOM（自然语言 key + 随机 response）超过 256 指纹后性能急剧下降

## 方法详解

### 整体框架

指纹系统包含指纹生成和指纹训练两部分。

### 指纹生成——Perinucleus 采样

**Key 生成**：低温度采样自然语言问题，与正常查询无法区分（In-distribution）。

**Response 生成**：在基础模型 nucleus 分布边缘采样"合理但不常见"的响应：
1. 计算下一 token 概率分布
2. 找到 top-$t$ 百分位 nucleus 边界
3. 从 nucleus 外紧邻 $k$ 个 token 中均匀选取

参数选择：$t=0.8$（实际平均概率仅 0.014），$k=3$。

**假阳性率理论保证**：$\text{FPR} \leq \exp(-2M(1-1/k)^2)$，随指纹数量指数级下降。

### 指纹训练

1. **权重偏差惩罚**：每步更新后与原始模型加权平均（$\lambda_{WA}=0.75$）
2. **数据混合**：指纹数据与基础模型生成数据混合（$\beta_{DM}=0.25$）

### 共谋攻击防御

每个指纹以概率 $p$ 随机分配给每个模型，检测时追踪候选分数。理论保证：$M = O(2^K K^{K+1} \log(N/\delta))$ 个指纹可以 $1-\delta$ 概率检测至少一个共谋模型。

## 实验关键数据

### 可扩展性

| 指纹数量 | Perinucleus (OpenLLM) | ENGLISH-RANDOM | 保持率 |
|----------|----------------------|----------------|--------|
| 256 | ~63% | ~61% | >99% |
| 1024 | ~62.5% | ~57% | ~98% |
| 8192 | ~61.5% | 崩溃 | ~96% |
| 24576 | ~61% | N/A | ~95% |

### 持久性（SFT 后）

| 方案 | 1024 持久率 | 8192 持久率 |
|------|-----------|-----------|
| RANDOM | ~85% | ~65% |
| Perinucleus | ~80% | ~60% |
| ENGLISH-RANDOM | ~40% | <20% |

### 跨模型通用性

10 个模型上 8192 指纹时相对性能 >95%。

### 关键发现

1. SFT 样本数增加对持久性影响近似 log-线性
2. 数学数据比聊天数据对指纹遗忘影响更小
3. DPO 训练不会显著加剧指纹遗忘
4. 检查 5 个指纹即可获得满意的假阳性/假阴性率

## 亮点与洞察

1. **Perinucleus 采样**：在 nucleus 边缘采样的优雅设计，减少训练时模型畸变
2. **可扩展性作为安全属性**：首次将其提升为核心准则并理论证明必要性
3. **正则化与指纹设计正交**：消融证明两者独立贡献
4. **简单高效的共谋防御**：随机分配 + $\log(N)$ 的指纹数量

## 局限与展望

1. 主要实验使用 1-token response，多 token 场景待深入
2. 未充分评估微调+共谋等组合攻击
3. 模型合并攻击的影响需更深入研究
4. 不同推理采样策略可能影响检测

## 相关工作与启发

- **Xu et al. / Russinovich & Salem**：关注 Harmlessness/Persistence 但忽略 Scalability
- **模型水印**：检测文本是否由 LLM 生成，指纹验证特定模型所有权
- **启发**：Perinucleus 思想可推广到任何需要嵌入隐蔽信息的场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Perinucleus 采样原创且优雅，可扩展性视角全新
- 实验充分度: ⭐⭐⭐⭐⭐ 10 个模型、多种攻击、理论保证、完整的参数分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，多维度分析
- 价值: ⭐⭐⭐⭐ 对模型安全和知识产权保护有价值，应用场景相对窄

<!-- RELATED:START -->

## 相关论文

- [The Curse of Depth in Large Language Models](the_curse_of_depth_in_large_language_models.md)
- [Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models](retrospective_incontext_learning_for_temporal_credit_assignm.md)
- [Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models](leveraging_importance_sampling_to_detach_alignment_modules_from_large_language_m.md)
- [Large Vocabulary Size Improves Large Language Models](../../ACL2025/llm_pretraining/large_vocabulary_size_improves_large_language_models.md)
- [Large Language Models are Demonstration Pre-Selectors for Themselves](../../ICML2025/llm_pretraining/large_language_models_are_demonstration_pre-selectors_for_themselves.md)

<!-- RELATED:END -->
