---
title: >-
  [论文解读] Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation
description: >-
  [ACL 2025][社会计算][instruction-tuning] 首次系统研究指令微调如何影响 LLM 对虚假信息的易感性，发现指令微调使模型从偏信 assistant-role 转变为偏信 user-role，当虚假信息以独立 user-turn 呈现时易感性最高，揭示了指令微调的"副作用"。
tags:
  - "ACL 2025"
  - "社会计算"
  - "instruction-tuning"
  - "misinformation"
  - "sycophancy"
  - "knowledge conflict"
  - "user-role bias"
---

# Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation

**会议**: ACL 2025  
**arXiv**: [2507.18203](https://arxiv.org/abs/2507.18203)  
**代码**: 无  
**领域**: 社会计算  
**关键词**: instruction-tuning, misinformation, sycophancy, knowledge conflict, user-role bias

## 一句话总结
首次系统研究指令微调如何影响 LLM 对虚假信息的易感性，发现指令微调使模型从偏信 assistant-role 转变为偏信 user-role，当虚假信息以独立 user-turn 呈现时易感性最高，揭示了指令微调的"副作用"。

## 研究背景与动机

**领域现状**：指令微调增强了 LLM 的指令遵循能力和安全性，是 LLM 部署的标准流程。

**现有痛点**：指令微调可能过度增强模型对用户输入的依赖，导致接受用户提供的虚假信息而产生幻觉。先前工作发现了这一现象但未深入分析原因。

**核心矛盾**：指令微调在提升有用性的同时，是否系统性地增加了对虚假信息的易感性？

**本文目标** 通过 base 模型和指令微调模型的对比，首次归因"虚假信息易感性增加"到指令微调本身。

**切入角度**：利用 chat template 中的 user/assistant 角色分离，测试虚假信息通过不同角色呈现时的影响差异。

**核心 idea**：指令微调将模型的注意力从 assistant-role 转移到 user-role，使得 user 提供的虚假信息更容易被接受。

## 方法详解

### 整体框架
3 种场景设计（STQ / APD / UPD）-> 6 个指令微调模型 + 4 个对应 base 模型 -> Farm 数据集（含虚假信息）-> MSR 指标衡量易感性 -> 额外分析信息长度和系统提示警告的影响。

### 关键设计

1. **三种场景 (RQ1)**

    - **STQ（单轮查询）**：虚假信息与问题在同一 user-turn→基线
    - **APD（Assistant 提供文档）**：虚假信息由 assistant 在前一轮提供
    - **UPD（User 提供文档）**：虚假信息由 user 在前一轮提供
    - 设计动机：分离角色效应——如果 UPD > APD，说明模型更信任 user

2. **Base vs Instruct 对比 (RQ2)**

    - 对 4 个开源模型的 base 和 instruct 版本做相同测试
    - 设计动机：归因效应到指令微调本身

3. **MSR 指标**

    - MSR(%) = |正确回答且被虚假信息欺骗的问题| / |正确回答的问题| × 100
    - 只计算模型本来"知道"的问题，排除知识不足的干扰

## 实验关键数据

### 主实验 -- RQ1: 指令微调模型的 MSR

| 模型 | STQ | APD | UPD | UPD-APD 差 |
|------|-----|-----|-----|-----------|
| GPT-4o | ~25% | ~20% | **~30%** | +10% |
| Llama-3-8B-Instruct | ~30% | ~25% | **~35%** | +10% |
| Mistral-7B-Instruct | ~35% | ~30% | **~55%** | **+25%** |
| Qwen2.5-7B-Instruct | ~30% | ~35% | ~30% | -5% (例外) |

### RQ2: Base vs Instruct 排名变化

| 模型 | Base 排名 | Instruct 排名 | 变化 |
|------|----------|-------------|------|
| Llama-3 | APD > UPD > STQ | **UPD > STQ > APD** | 反转 |
| Llama-3.1 | APD > UPD > STQ | **UPD > STQ > APD** | 反转 |
| Mistral | APD > UPD > STQ | **UPD > APD > STQ** | 反转 |

### RQ3: 虚假信息长度影响

| 长度 | UPD-APD 差距 | 趋势 |
|------|-------------|------|
| 1 段（短） | +10-25% | UPD 显著高 |
| 2 段（中） | +5-15% | 差距缩小 |
| 3 段（长） | +0-5% | **接近 base 模型模式** |

### 关键发现
- **UPD > APD**：除 Qwen 外所有指令微调模型更容易被 user-role 提供的虚假信息欺骗
- **指令微调是因果因素**：base 模型的排名是 APD > UPD（偏信 assistant），指令微调后翻转（偏信 user）
- **虚假信息长度增加削弱指令微调效应**：长文档使模型行为回归 base 模式
- **系统提示警告对开源模型无效**：GPT-4o MSR 下降 69%，但 Llama 等几乎无变化
- **独立 user-turn 放大效应**：UPD > STQ，说明虚假信息作为独立 user-turn 比嵌入问题中更危险

## 亮点与洞察
- **首次将虚假信息易感性归因到指令微调**——base 模型偏信 assistant，指令微调后偏信 user，这是一个清晰的因果证据。
- **"越长越安全"**的发现有实际启示——短的、精炼的虚假信息比长篇大论更危险。
- **系统提示警告对开源模型无效**揭示了实际部署中的安全漏洞。

## 局限与展望
- 仅用 Farm 数据集，规模有限
- 未分析具体的训练数据内容对易感性的影响
- 改进方向：缓解策略（如角色感知训练）、更多场景测试

## 相关工作与启发
- **vs Xie et al. (2024)**：他们发现 LLM 高度接受外部信息，本文进一步归因到指令微调
- **vs Wei et al. (2023)**：他们指出指令微调增加 sycophancy，本文在虚假信息场景下验证了这一点

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统归因虚假信息易感性到指令微调，角色分离实验设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 6 指令模型 + 4 base 模型 × 3 场景 × 3 数据集
- 写作质量: ⭐⭐⭐⭐⭐ 研究问题逐层递进，逻辑严密
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 安全部署和指令微调方法改进有直接启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] How does Misinformation Affect Large Language Model Behaviors and Preferences?](how_does_misinformation_affect_large_language.md)
- [\[ACL 2025\] Exploring Multimodal Challenges in Toxic Chinese Detection: Taxonomy, Benchmark, and Findings](exploring_multimodal_challenges_in_toxic_chinese_detection_taxonomy_benchmark_an.md)
- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)
- [\[ACL 2025\] A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models](a_survey_on_proactive_defense_strategies_against_misinformation_in_large_languag.md)
- [\[ACL 2025\] BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla](banstereoset_a_dataset_to_measure_stereotypical_social_biases_in_llms_for_bangla.md)

</div>

<!-- RELATED:END -->
