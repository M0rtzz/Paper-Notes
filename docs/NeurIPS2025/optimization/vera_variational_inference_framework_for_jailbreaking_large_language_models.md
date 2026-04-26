---
title: >-
  [论文解读] VERA: Variational Inference Framework for Jailbreaking Large Language Models
description: >-
  [NeurIPS 2025][优化][越狱攻击] 将黑盒 LLM 越狱攻击形式化为变分推断问题，训练小型攻击者 LLM 近似目标 LLM 的对抗提示后验分布，一次训练后可高效、多样地生成越狱提示，无需依赖人工模板。
tags:
  - NeurIPS 2025
  - 优化
  - 越狱攻击
  - 变分推断
  - 黑盒攻击
  - 红队测试
  - 对抗提示
---

# VERA: Variational Inference Framework for Jailbreaking Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.22666](https://arxiv.org/abs/2506.22666)  
**代码**: 无  
**领域**: AI 安全, LLM 红队测试, 变分推断  
**关键词**: 越狱攻击, 变分推断, 黑盒攻击, 红队测试, 对抗提示

## 一句话总结
将黑盒 LLM 越狱攻击形式化为变分推断问题，训练小型攻击者 LLM 近似目标 LLM 的对抗提示后验分布，一次训练后可高效、多样地生成越狱提示，无需依赖人工模板。

## 研究背景与动机

### 领域现状

**领域现状**：现有黑盒越狱方法依赖遗传算法或搜索，需为每个行为单独优化，计算成本高

### 现有痛点

**现有痛点**：多数方法依赖人工编写的越狱模板池作为初始化，绑定于已知漏洞，容易被修补

### 核心矛盾

**核心矛盾**：单一成功案例不足以全面评估模型漏洞，需要多样化攻击以覆盖脆弱性全貌

### 解决思路

**解决思路**：缺乏从分布角度理解和生成对抗提示的原则性框架

## 方法详解

### 整体框架
- 将越狱提示生成视为后验推断问题：$x \sim P_{LM}(x|y^*)$
- 用小型 LLM（攻击者）作为变分分布 $q_\theta(x)$ 近似目标 LLM 的对抗提示后验
- 通过 LoRA 微调攻击者模型，使其学习生成有效越狱提示的分布

### 关键设计
1. **变分目标（ELBO）**：
    $\mathbb{E}_{q_\theta(x)}[\log P_{LM}(y^*|x) + \log P(x) - \log q_\theta(x)]$
    - 第一项：有害内容生成概率（攻击有效性）
    - 第二项：先验下的合理性（提示流畅性）
    - 第三项：熵正则化（鼓励多样性，避免模式坍缩）

2. **Judge 作为似然近似器**：

    - 黑盒场景无法直接计算 $P_{LM}(y^*|x)$
    - 使用外部判断模型 $J(x,\hat{y}) \in [0,1]$ 作为代理
    - 可使用二分类器（如 HarmBench 的 LLaMA2-13B 分类器）或 LLM 提示评分

3. **REINFORCE 梯度估计**：

    - 使用 REINFORCE trick 处理离散采样的梯度
    - $\nabla_\theta \approx \frac{1}{N}\sum_i f(x_i) \nabla_\theta \log q_\theta(x_i)$
    - 早停机制：一旦有提示成功越狱即停止，防止过优化导致攻击者退化

### 损失函数 / 训练策略
- 攻击者模型：Vicuna-7b + LoRA
- 每步生成 B 个提示 → 查询目标 LLM → Judge 评分 → REINFORCE 更新
- 早停于首个成功越狱提示
- 不依赖任何人工编写的越狱模板

## 实验关键数据

### 主实验（HarmBench ASR%）

| 方法 | Llama2-7b | Vicuna-7b | Orca2-7b | R2D2 | GPT-3.5 | 平均 |
|------|----------|----------|---------|------|---------|------|
| GCG (白盒) | 32.5 | 65.5 | 46.0 | 5.5 | - | 40.2 |
| PAIR (黑盒) | 9.3 | 53.5 | 57.3 | 48.0 | 35.0 | 36.3 |
| TAP-T (黑盒) | 7.8 | 59.8 | 60.3 | 54.3 | 47.5 | 40.9 |
| AutoDAN (黑盒) | 0.5 | 66.0 | 71.0 | 17.0 | - | 34.8 |
| **VERA** (黑盒) | **10.8** | **70.0** | **72.0** | **63.5** | - | - |

### 多样性与新颖性比较（50 行为子集，Vicuna 7B 目标）

| 指标 | VERA | GPTFuzzer | AutoDAN |
|------|------|-----------|---------|
| Self-BLEU (越低越多样) | **最低** | 中等 | 中等 |
| 与模板 BLEU (越低越新颖) | **最低** | 较高 | 较高 |
| 固定时间内成功攻击数 | **5× GPTFuzzer** | 基线 | 中等 |

### 关键发现
- VERA 在 HarmBench 上达到黑盒方法 SOTA，多个目标模型上超越所有已有方法
- 生成的攻击提示多样性显著高于模板方法（Self-BLEU 最低）
- 完全独立于初始模板：与系统提示的 BLEU 分数极低
- 在固定时间预算（1250s）下，VERA 产生的成功攻击数是 GPTFuzzer 的 5 倍以上
- 移除已知有效模板后，模板方法性能大幅下降，而 VERA 不受影响

## 亮点与洞察
- 将越狱攻击优雅地嵌入变分推断框架，数学基础严谨
- 熵正则化项自然解决了攻击多样性问题，无需额外设计
- 训练后的攻击者可通过简单前向传播生成新攻击，摊销了计算成本
- VERA 不依赖已知漏洞模板，具有"未来适应性"——不会因漏洞修补而失效

## 局限与展望
- 查询目标 LLM 的次数较多（每步 B 次），在 API 计费场景下成本较高
- 使用 Vicuna-7b 作为攻击者可能限制了对某些防御更强的模型的攻击效果
- 早停机制虽防止退化但可能过早终止对更多攻击模式的探索
- 对 RL 角度的对比讨论（附录中提到）可更深入

## 相关工作与启发
- 变分推断框架为红队测试提供了从单点攻击到分布式攻击的范式转换
- REINFORCE + LoRA 的组合使得训练效率可控
- "全面红队测试需要覆盖漏洞广度而非仅确认存在"的观点值得重视
- 对 LLM 安全研究的启示：安全对齐不应依赖于攻击者缺乏系统性方法
- 方法的"未来适应性"启示：不依赖已知漏洞的攻击方法更难被防御
- VERA 的分布式视角为理解 LLM 脆弱性的结构提供了新工具
- 可探索将变分推断框架应用于防御端，学习对抗提示的分布以进行主动防御

## 评分
- 新颖性：⭐⭐⭐⭐⭐ （变分推断框架应用于越狱极具创新性）
- 技术贡献：⭐⭐⭐⭐⭐ （理论严谨 + 工程完整 + 优势明显）
- 实验充分度：⭐⭐⭐⭐⭐ （8 个目标模型、多基线对比、多维度评估）
- 写作质量：⭐⭐⭐⭐ （表述清晰，推导详尽）

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] Constrained Network Slice Assignment via Large Language Models](constrained_network_slice_assignment_via_llms.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](neuro-symbolic_entity_alignment_via_variational_inference.md)
- [\[NeurIPS 2025\] Large Language Bayes](large_language_bayes.md)

<!-- RELATED:END -->
