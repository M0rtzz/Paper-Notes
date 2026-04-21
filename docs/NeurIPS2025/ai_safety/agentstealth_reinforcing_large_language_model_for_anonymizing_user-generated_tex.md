---
title: >-
  [论文解读] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text
description: >-
  [NeurIPS 2025][AI安全][文本匿名化] 提出 AgentStealth 框架，通过对抗式匿名化工作流、监督微调（SFT）和在线强化学习三阶段训练小型语言模型（SLM），实现在保持文本效用的同时有效匿名化用户生成内容，匿名化效果提升12.3%、效用提升6.8%。
tags:
  - NeurIPS 2025
  - AI安全
  - 文本匿名化
  - 隐私保护
  - 强化学习
  - 对抗训练
  - 小型语言模型
---

# AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text

**会议**: NeurIPS 2025  
**arXiv**: [2506.22508](https://arxiv.org/abs/2506.22508)  
**代码**: [GitHub](https://github.com/tsinghua-fib-lab/AgentStealth)  
**领域**: AI安全 / 隐私保护  
**关键词**: 文本匿名化, 隐私保护, 强化学习, 对抗训练, 小型语言模型

## 一句话总结

提出 AgentStealth 框架，通过对抗式匿名化工作流、监督微调（SFT）和在线强化学习三阶段训练小型语言模型（SLM），实现在保持文本效用的同时有效匿名化用户生成内容，匿名化效果提升12.3%、效用提升6.8%。

## 研究背景与动机

在数字时代，用户在社交媒体、论坛等平台上生成的大量文本内容往往包含**隐性的个人身份线索**——如写作风格、惯用词汇、话题偏好等，这些线索可能被攻击者利用来推断用户的敏感个人属性（如年龄、性别、职业、地理位置等）。

**文本匿名化**旨在改写文本以消除这些身份线索，同时保持文本的语义和实用性。然而现有方法面临多重挑战：

**规则替换方法**：简单替换关键词（如人名、地名）容易破坏文本可读性和实用性

**云端LLM方法**：使用GPT-4等大模型效果好，但成本高昂且本身存在隐私风险——将敏感文本上传到云端与匿名化初衷矛盾

**小型模型的困境**：本地部署的SLM训练数据和监督信号不足，匿名化效果不理想

**核心矛盾**：匿名化需要大模型的能力，但大模型部署在云端有隐私泄露风险。如何让本地部署的小模型也具有强大的匿名化能力？

## 方法详解

### 整体框架

AgentStealth 采用**三阶段递进训练**策略：

**第一阶段：对抗式匿名化工作流（Adversarial Anonymization Workflow）**
- 使用大模型（如 DeepSeek-V3）构建高质量的匿名化数据
- 包含攻击者和匿名化者两个角色的对抗交互
- 收集高质量的 (原文, 匿名化文本, 攻击信号) 三元组

**第二阶段：监督微调（SFT）**
- 使用第一阶段收集的高质量数据对SLM进行监督微调
- 基座模型为 Llama-3.1-8B-Instruct 和 Qwen-2.5-1.5B-Instruct
- 使用 LLaMA-Factory 框架进行训练

**第三阶段：在线强化学习（RL）**
- SLM利用其内部的对抗反馈进行自我强化
- 模型自己作为匿名化者和攻击者进行对抗博弈
- 通过 PPO 或类似算法迭代优化匿名化策略

### 关键设计

**1. In-context Contrastive Learning（上下文对比学习）**

在工作流中引入对比学习增强匿名化质量：
- 正例：成功匿名化的文本（攻击者无法识别）
- 负例：匿名化失败的文本（攻击者仍能识别）
- 通过 in-context learning 让模型理解什么样的改写是有效的匿名化

**2. Adaptive Utility-Aware Control（自适应效用感知控制）**

在匿名化过程中动态平衡隐私保护和文本效用：
- 效用评估：衡量匿名化前后文本的语义保持程度
- 自适应阈值：根据当前匿名化难度调整改写强度
- 避免过度改写导致文本失去原始含义

**3. 双信号训练数据**

SFT 数据同时包含匿名化信号和攻击信号：
- 匿名化信号：教模型如何改写文本消除身份线索
- 攻击信号：教模型理解哪些特征容易暴露身份
- 双信号使SLM同时具备"防守者"和"攻击者"的知识

### 损失函数 / 训练策略

**SFT 阶段**：标准的语言模型微调损失

$$L_{SFT} = -\sum_{t} \log p(y_t | y_{<t}, x)$$

**RL 阶段**：使用复合奖励函数

$$R = \alpha \cdot R_{anon} + \beta \cdot R_{utility} + \gamma \cdot R_{fluency}$$

其中：
- $R_{anon}$：匿名化奖励（攻击者预测错误则为正奖励）
- $R_{utility}$：效用奖励（基于与原文的语义相似度）
- $R_{fluency}$：流畅性奖励（基于困惑度）
- $\alpha, \beta, \gamma$：权衡系数

训练使用 Accelerate 和 TRL 库进行分布式RL训练。

## 实验关键数据

### 主实验

在两个数据集上的匿名化效果对比（Reddit 和 Coding 数据集）：

| 方法 | Reddit 匿名化率 (%) | Reddit 效用 (%) | Coding 匿名化率 (%) | Coding 效用 (%) |
|------|-------------------|----------------|-------------------|----------------|
| 无匿名化 | 0.0 | 100.0 | 0.0 | 100.0 |
| 规则替换 | 28.5 | 82.3 | 25.1 | 79.8 |
| Paraphrase (GPT-3.5) | 45.2 | 88.7 | 41.3 | 85.2 |
| LLM Anonymizer (GPT-4) | 62.8 | 91.5 | 58.4 | 89.1 |
| LLM Anonymizer (DeepSeek-V3) | 65.1 | 92.3 | 61.7 | 90.4 |
| AgentStealth-8B (SFT) | 68.3 | 93.8 | 64.2 | 91.7 |
| AgentStealth-8B (SFT+RL) | **73.1** | **95.5** | **69.8** | **93.6** |
| AgentStealth-1.5B (SFT+RL) | 67.5 | 92.1 | 63.8 | 90.2 |

### 消融实验

各组件的贡献（Reddit 数据集）：

| 配置 | 匿名化率 (%) | 效用 (%) | 综合评分 |
|------|-------------|---------|---------|
| SLM 直接推理 | 38.2 | 85.6 | 61.9 |
| + SFT (仅匿名化信号) | 58.7 | 90.3 | 74.5 |
| + SFT (双信号) | 68.3 | 93.8 | 81.1 |
| + 对比学习 | 70.1 | 94.2 | 82.2 |
| + 效用感知控制 | 70.8 | 94.8 | 82.8 |
| + RL 自强化 | **73.1** | **95.5** | **84.3** |

### 关键发现

1. **SLM 可以达到甚至超越云端LLM**：AgentStealth-8B 的匿名化率比 GPT-4 高出约10个百分点，效用也更好
2. **三阶段训练每步都有贡献**：从直接推理到最终RL版本，匿名化率从38.2%提升到73.1%
3. **双信号比单信号显著更好**：同时学习匿名化和攻击知识，匿名化率提升约10个百分点
4. **1.5B 模型也可部署**：Qwen-2.5-1.5B 版本仍然优于 GPT-4 基线，支持边缘设备部署
5. **RL 自强化效果显著**：在SFT基础上进一步提升约5个百分点，证明自我博弈训练的有效性
6. **效用与隐私同步提升**：不同于通常的隐私-效用权衡，AgentStealth通过更智能的改写策略同时提升了两者

## 亮点与洞察

1. **解决了核心矛盾**：让小模型具备大模型的匿名化能力，实现本地部署，从根本上避免云端隐私泄露
2. **自强化训练范式新颖**：模型同时扮演攻击者和防守者，通过自我博弈持续提升
3. **实用性强**：支持从1.5B到8B多种规模的模型，适配不同的部署环境
4. **开源**：代码和训练配置完全开源，可复现性好

## 局限与展望

1. **语言局限**：目前主要在英文数据上验证，中文等其他语言的匿名化效果未知
2. **属性覆盖有限**：主要关注几类常见的个人属性，更细粒度的身份推断（如写作风格分析）未充分考虑
3. **效用评估的主观性**：效用评分部分依赖LLM作为judge，可能存在偏差
4. **审稿状态**：arXiv 页面显示仍为"under review"，最终版本可能有改动
5. **攻击者模型的上限**：如果攻击者使用比训练中更强的模型，匿名化效果可能下降

## 相关工作与启发

- **Language Models are Advanced Anonymizers (Staab et al., ICLR 2025)**：本文直接构建在此工作之上，使用其攻击评估框架
- **差分隐私文本方法**：在理论保证上更强，但文本效用大幅下降
- **风格迁移方法**：通过改变写作风格实现匿名化
- **自我博弈/RLHF**：强化学习在语言模型对齐中的成功应用
- **启发**：对抗自强化训练范式可推广到其他安全任务，如内容审核、深度伪造检测等

## 评分

- 新颖性：⭐⭐⭐⭐ （自强化匿名化框架新颖，但基本组件较常见）
- 技术深度：⭐⭐⭐⭐ （三阶段训练设计完整，各组件有机结合）
- 实验充分性：⭐⭐⭐⭐ （两个数据集，详细消融，多模型规模对比）
- 写作质量：⭐⭐⭐⭐ （结构清晰，动机阐述到位）
- 综合评分：⭐⭐⭐⭐ （实用价值高，技术贡献扎实）

<!-- RELATED:START -->

## 相关论文

- [Self-Refining Language Model Anonymizers via Adversarial Distillation](self-refining_language_model_anonymizers_via_adversarial_distillation.md)
- [Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text](adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)
- [DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm](dna-detectllm_unveiling_ai-generated_text_via_a_dna-inspired_mutation-repair_par.md)
- [PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](../../ICML2025/ai_safety/the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)

<!-- RELATED:END -->
