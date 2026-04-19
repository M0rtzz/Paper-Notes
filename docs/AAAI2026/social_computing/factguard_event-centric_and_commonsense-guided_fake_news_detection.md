---
title: >-
  [论文解读] FactGuard: Event-Centric and Commonsense-Guided Fake News Detection
description: >-
  [AAAI2026][fake news detection] 提出 FactGuard 框架，利用 LLM 提取事件核心内容（去风格化）并生成常识推理，通过 Rationale Usability Evaluator 动态评估 LLM 建议的可信度，并通过知识蒸馏获得无需 LLM 的轻量版 FactGuard-D，在假新闻检测中兼顾鲁棒性和效率。
tags:
  - AAAI2026
  - fake news detection
  - LLM reasoning
  - 知识蒸馏
  - commonsense reasoning
  - style debiasing
---

# FactGuard: Event-Centric and Commonsense-Guided Fake News Detection

**会议**: AAAI2026  
**arXiv**: [2511.10281](https://arxiv.org/abs/2511.10281)  
**代码**: [ryliu68/FACTGUARD](https://github.com/ryliu68/FACTGUARD)  
**领域**: model_compression  
**关键词**: fake news detection, LLM reasoning, knowledge distillation, commonsense reasoning, style debiasing

## 一句话总结
提出 FactGuard 框架，利用 LLM 提取事件核心内容（去风格化）并生成常识推理，通过 Rationale Usability Evaluator 动态评估 LLM 建议的可信度，并通过知识蒸馏获得无需 LLM 的轻量版 FactGuard-D，在假新闻检测中兼顾鲁棒性和效率。

## 研究背景与动机
- **写作风格脆弱性**：基于风格的假新闻检测方法虽取得进展，但对抗者可模仿真实新闻风格来绕过检测
- **LLM 的浅层利用**：现有 LLM 增强方法存在以下问题：
    - 风格增强数据无法完全消除风格干扰（SheepDog, LLM-Fake）
    - LLM 少样本/CoT 推理准确率有限，易产生 hallucination
    - 多 agent 辩论框架（TED）推理成本高，不适合冷启动和资源受限场景
    - 缺乏 LLM 建议可用性的可靠评估机制（ARG 训练时知道正确性，推理时不知道）

核心思路：新闻源于真实事件（新闻传播理论），提取事件核心内容可去除风格噪声；LLM 的常识推理能力可补充事实一致性判断。

## 方法详解

### Feature Extraction
对每条新闻 $n$，利用 LLM 通过精心设计的 prompt 提取：
- **Topic-Content $c$**：核心主题和主要内容（去风格化），附带文本相似度约束和信息密度评估
- **Commonsense Rationale $r$**：常识推理分析，判断是否存在违反常识的内容

三者分别通过 SLM（BERT/RoBERTa）编码。

### Topic-Content & Rationale Interactor
双向 cross-attention 实现 topic-content 与 commonsense rationale 的深度特征交互：
- $f_{C \to R}$：LLM 建议特征向量
- $f_{R \to C}$：用于权重评估的交互特征

### Rationale Usability Evaluator
双分支 MLP 结构动态评估 LLM 建议的可信度：
- **分支 1**：LLM 直接检测能力有限时降低贡献（监督信号为 0）
- **分支 2**：常识推理发现矛盾或不确定性时增加贡献（监督信号为 $y_{llm}$）
- 最终 LLM 特征：$f_{llm} = [w_1 \cdot f_{C \to R1}; w_2 \cdot f_{C \to R2}]$

### 训练损失
$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \alpha \frac{\mathcal{L}_{usability}}{2} + \beta \frac{\mathcal{L}_{text}}{2}$$

### FactGuard-D（蒸馏版）
- 四层 Transformer encoder + linear attention 模拟教师模型的推理能力
- 仅输入原始新闻文本，无需 LLM 调用
- 损失增加 MSE 特征蒸馏项 $\mathcal{L}_{distill}$

## 实验关键数据

在 Weibo21（中文）和 GossipCop（英文）上评测。

### 主实验（macF1）

| 方法 | Weibo21 macF1 | GossipCop macF1 |
|---|---|---|
| BERT | 0.753 | 0.765 |
| ARG (LLM+SLM) | 0.784 | 0.790 |
| TED (多 agent 辩论) | 0.795 | 0.803 |
| **FactGuard** | **0.801** | **0.805** |
| ARG-D (蒸馏) | 0.771 | 0.778 |
| **FactGuard-D** | **0.788** | **0.790** |

- FactGuard 在 Weibo21 上 Acc. 达 0.804，较 TED 提升 0.8%
- FactGuard-D 无需 LLM 推理，仍优于 ARG 全量版

### 消融实验要点
- 去掉原始新闻表示 → 性能下降最大（基础不可替代）
- 去掉 topic-content 提取 → macF1 下降约 3%（事件信息关键）
- 去掉 usability 模块 → 性能下降，说明动态可信度评估有效
- Topic-content 和 commonsense rationale 需联合使用才能最大化收益

## 亮点
- **事件中心去风格化**：利用 LLM 语义能力提取事件核心内容，从源头减少写作风格干扰，比风格增强数据方法更根本
- **LLM 建议可用性动态评估**：双分支结构区分 LLM 的直接判断能力和常识推理贡献，避免盲目信任 LLM
- **全场景部署**：FactGuard 适用于资源充足场景，FactGuard-D 通过蒸馏适应冷启动/资源受限场景，仅用两个简单 prompt 即超越多 agent 辩论框架
- **跨语言验证**：中英双语数据集上一致有效

## 局限性
- macF1 提升幅度相对有限（较 TED 约 0.6-0.8%），在 GossipCop 上提升更小
- LLM 提取的 topic-content 质量依赖 prompt 设计和 LLM 能力，不同 LLM 可能差异大
- 仅在文本模态上验证，未涉及多模态假新闻检测
- FactGuard-D 的 feature simulator 增加了学生模型复杂度，与直接使用 SLM 的差距需更多场景验证
- 未讨论所用 LLM 的 benchmark data contamination 问题

## 评分
- 新颖性: ⭐⭐⭐ — 事件提取去风格化思路合理，但整体框架为已有组件的工程化组合
- 实验充分度: ⭐⭐⭐⭐ — 双语数据集、14 个基线、详细消融和参数分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机论证充分，图示直观
- 价值: ⭐⭐⭐⭐ — 提供了从资源充足到资源受限的完整假新闻检测解决方案

<!-- RELATED:START -->

## 相关论文

- [Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](../../ACL2025/social_computing/llm_label_propagation.md)
- [Detection of Human and Machine-Authored Fake News in Urdu](../../ACL2025/social_computing/detection_of_human_and_machine-authored_fake_news_in_urdu.md)
- [Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)
- [Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)
- [T2Agent: A Tool-augmented Multimodal Misinformation Detection Agent with Monte Carlo Tree Search](t2agent_a_tool-augmented_multimodal_misinformation_detection_agent_with_monte_ca.md)

<!-- RELATED:END -->
