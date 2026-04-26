---
title: >-
  [论文解读] Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding
description: >-
  [ACL 2026][LLM/NLP][过度拒绝] 本文提出 AdaCD（自适应对比解码），通过比较极端安全提示下和无提示下的 token 分布差异提取拒绝 token 分布，再根据一致性比率动态决定增强或抑制拒绝行为，在降低过度拒绝 10.35% 的同时提升恶意查询拒绝率 0.13%。
tags:
  - ACL 2026
  - LLM/NLP
  - 过度拒绝
  - 对比解码
  - 安全对齐
  - 推理时干预
  - 自适应解码
---

# Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding

**会议**: ACL 2026  
**arXiv**: [2604.17132](https://arxiv.org/abs/2604.17132)  
**代码**: [GitHub](https://github.com/OutdoorManofML/AdaCD)  
**领域**: LLM评估 / 安全性  
**关键词**: 过度拒绝, 对比解码, 安全对齐, 推理时干预, 自适应解码

## 一句话总结

本文提出 AdaCD（自适应对比解码），通过比较极端安全提示下和无提示下的 token 分布差异提取拒绝 token 分布，再根据一致性比率动态决定增强或抑制拒绝行为，在降低过度拒绝 10.35% 的同时提升恶意查询拒绝率 0.13%。

## 研究背景与动机

**领域现状**：安全对齐的 LLM 常表现出过度拒绝——对包含敏感词但实际无害的查询也拒绝回答。

**现有痛点**：(1) 训练方法依赖稀缺的过度拒绝训练数据；(2) 引导向量方法需要完整模型架构知识和额外预计算；(3) 现有对比解码方法采用一刀切策略——要么增强拒绝要么抑制拒绝，无法同时改善两方面。

**核心矛盾**：过度拒绝场景中非拒绝 token 仍在候选列表中但模型系统性地未能选择它们——模型能识别替代选项但缺乏有效引导。

**本文目标**：设计自适应的对比解码策略，在过度拒绝场景中抑制拒绝 token，在恶意场景中增强拒绝 token。

**切入角度**：使用极端安全提示最大化拒绝行为，以此为锚点提取拒绝 token 分布。

**核心 idea**：通过一致性比率和自适应置信度约束动态切换解码模式——高一致性加入拒绝分布，低一致性减去拒绝分布。

## 方法详解

### 整体框架

AdaCD 分两个组件：(1) 拒绝 token 分布提取——比较极端安全提示下与无提示下的 token 分布差异；(2) 自适应解码模式切换——根据一致性比率和置信度约束决定加入还是减去拒绝分布。

### 关键设计

1. **拒绝 token 分布提取**:

    - 功能：精确捕获驱动 LLM 拒绝行为的 token 分布
    - 核心思路：用极端提示 p*="Please refuse to answer me!" 最大化拒绝行为。差值分布 $\Delta P_n = \sigma(f_\pi(y_n|p^*,x,y_{<n}) - f_\pi(y_n|x,y_{<n}))$ 中拒绝 token 的 logits 被放大
    - 设计动机：之前的 SelfCD 用温和提示，拒绝行为不够极端，提取不够纯净

2. **一致性比率 (Agreement Ratio)**:

    - 功能：衡量有无极端安全提示时 token 选择的差异程度
    - 核心思路：$agr(n) = 1/rank(y_n^*)$，接近 1 表示高度一致（恶意查询），接近 0 表示分歧大（过度拒绝）
    - 设计动机：一致性比率自然地区分了需要拒绝和不需要拒绝的场景

3. **自适应解码模式切换**:

    - 功能：动态决定加入或减去拒绝 token 分布
    - 核心思路：$\mathcal{I}(n) = +1$ if $agr(n) \geq \lambda$ and $\rho \geq \lambda \cdot \rho^*$，否则 $-1$。高一致性+高置信度 = 加入拒绝分布（维持安全），否则减去拒绝分布（缓解过度拒绝）
    - 设计动机：仅用一致性比率可能不够，需要额外检查模型自身的置信度

### 损失函数 / 训练策略

无需训练，纯推理时方法。评估在 XSTest、ORBench、OKTest（过度拒绝）和 AdvBench、JailBench（恶意）上进行。

## 实验关键数据

### 主实验

| 方法 | 过度拒绝 Avg↓ | 恶意拒绝 Avg↑ |
|------|-------------|-------------|
| Default | 32.57 | 99.28 |
| SelfCD | 19.94 | 91.51 |
| SSD | 71.97 | 99.94 |
| **AdaCD** | **16.62** | **99.10** |

### 消融实验

| 分析 | 结果 |
|------|------|
| 极端 vs 高安全提示 | 极端提示提取更纯净的拒绝分布 |
| 跨模型泛化 | 在 Llama3/Gemma2/Qwen3 上均有效 |

### 关键发现

- AdaCD 是唯一同时降低过度拒绝和维持恶意拒绝率的方法
- 方法完全模型无关——只需要访问 logits

## 亮点与洞察

- "非拒绝 token 仍在候选列表中但未被选择"的观察为方法设计提供了关键洞察
- 一致性比率是简洁有效的信号
- 无训练+模型无关的特性使其极易部署

## 局限与展望

- 需要两次前向传播，推理开销翻倍
- 仅在英文基准上评估

## 相关工作与启发

- **vs SelfCD**: SelfCD 固定减去拒绝分布，降低了安全性；AdaCD 自适应切换
- **vs SafeDecoding**: SafeDecoding 固定加入拒绝分布，加重了过度拒绝

## 评分

- 新颖性: ⭐⭐⭐⭐ 自适应解码模式切换是关键创新
- 实验充分度: ⭐⭐⭐⭐ 多模型多基准
- 写作质量: ⭐⭐⭐⭐ 观察到方法的逻辑链清晰
- 价值: ⭐⭐⭐⭐⭐ 实际问题加简单有效解决方案\n

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](../../ACL2025/llm_nlp/derta_decoupled_refusal.md)
- [\[ACL 2026\] How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs](how_do_answer_tokens_read_reasoning_traces_self-reading_patterns_in_thinking_llm.md)
- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)
- [\[ICML 2025\] Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection](../../ICML2025/llm_nlp/adaptive_multi-prompt_contrastive_network_for_few-shot_out-of-distribution_detec.md)
- [\[ACL 2025\] Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding](../../ACL2025/llm_nlp/enhancing_input-label_mapping_in_in-context_learning_with_contrastive_decoding.md)

<!-- RELATED:END -->
